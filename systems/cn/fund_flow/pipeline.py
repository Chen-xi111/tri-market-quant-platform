"""AKShare 驱动的 A 股大盘、板块与个股资金流日扫描 pipeline。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import gzip
import json
import os
from pathlib import Path
import time
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
HORIZONS = {"今日": "today", "5日": "d5", "10日": "d10"}
SECTOR_TYPES = {"industry": "行业资金流", "concept": "概念资金流"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / ".fund_flow_data"
ProgressCallback = Callable[[str, int, int], None]


class FundFlowUnavailable(RuntimeError):
    """所有资金流接口均不可用。"""


class FundFlowProvider(Protocol):
    name: str
    version: str

    def market_flow(self) -> pd.DataFrame: ...

    def sector_flow(self, indicator: str, sector_type: str) -> pd.DataFrame: ...

    def stock_flow(self, indicator: str) -> pd.DataFrame: ...


class AKShareFundFlowProvider:
    """对 AKShare 资金流接口做短重试，不改变其返回字段。"""

    name = "AKShare / 东方财富（同花顺回退）"

    def __init__(self, attempts: int = 2, retry_delay: float = 0.75):
        import akshare as ak

        self._ak = ak
        self.version = getattr(ak, "__version__", "unknown")
        self.attempts = max(1, attempts)
        self.retry_delay = max(0.0, retry_delay)

    def _call(self, func, *args, **kwargs) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                frame = func(*args, **kwargs)
                if frame is None or frame.empty:
                    raise ValueError("接口返回空数据")
                return frame
            except Exception as exc:  # 外部免费接口偶发空响应，允许一次短重试
                last_error = exc
                if attempt + 1 < self.attempts:
                    time.sleep(self.retry_delay * (attempt + 1))
        assert last_error is not None
        raise last_error

    def market_flow(self) -> pd.DataFrame:
        return self._call(self._ak.stock_market_fund_flow)

    def sector_flow(self, indicator: str, sector_type: str) -> pd.DataFrame:
        try:
            return self._call(
                self._ak.stock_sector_fund_flow_rank,
                indicator=indicator,
                sector_type=sector_type,
            )
        except Exception:
            symbol = "即时" if indicator == "今日" else f"{indicator}排行"
            fallback = (
                self._ak.stock_fund_flow_industry
                if sector_type == "行业资金流"
                else self._ak.stock_fund_flow_concept
            )
            return self._call(fallback, symbol=symbol)

    def stock_flow(self, indicator: str) -> pd.DataFrame:
        try:
            return self._call(self._ak.stock_individual_fund_flow_rank, indicator=indicator)
        except Exception:
            symbol = "即时" if indicator == "今日" else f"{indicator}排行"
            return self._call(self._ak.stock_fund_flow_individual, symbol=symbol)


def _json_default(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


def _frame_records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    clean = frame.replace([np.inf, -np.inf], np.nan).astype(object)
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


@dataclass
class ScanSnapshot:
    metadata: dict
    market: pd.DataFrame = field(default_factory=pd.DataFrame)
    industries: pd.DataFrame = field(default_factory=pd.DataFrame)
    concepts: pd.DataFrame = field(default_factory=pd.DataFrame)
    stocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: tuple[str, ...] = ()

    @property
    def snapshot_date(self) -> str:
        return str(self.metadata["snapshot_date"])

    def to_payload(self) -> dict:
        return {
            "schema_version": 1,
            "metadata": self.metadata,
            "market": _frame_records(self.market),
            "industries": _frame_records(self.industries),
            "concepts": _frame_records(self.concepts),
            "stocks": _frame_records(self.stocks),
            "errors": list(self.errors),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ScanSnapshot":
        return cls(
            metadata=payload.get("metadata", {}),
            market=pd.DataFrame(payload.get("market", [])),
            industries=pd.DataFrame(payload.get("industries", [])),
            concepts=pd.DataFrame(payload.get("concepts", [])),
            stocks=pd.DataFrame(payload.get("stocks", [])),
            errors=tuple(payload.get("errors", [])),
        )


class SnapshotStore:
    """以每日一个 gzip JSON 文件原子保存资金流快照。"""

    def __init__(self, data_dir: str | Path | None = None):
        configured = data_dir or os.environ.get("CN_FUND_FLOW_DATA_DIR")
        self.data_dir = Path(configured) if configured else DEFAULT_DATA_DIR
        self.snapshot_dir = self.data_dir / "snapshots"

    def save(self, snapshot: ScanSnapshot) -> Path:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        destination = self.snapshot_dir / f"{snapshot.snapshot_date}.json.gz"
        temporary = Path(f"{destination}.tmp")
        payload = json.dumps(
            snapshot.to_payload(), ensure_ascii=False, allow_nan=False, default=_json_default
        )
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, destination)
        return destination

    def dates(self) -> list[str]:
        if not self.snapshot_dir.exists():
            return []
        return sorted(
            (path.name.removesuffix(".json.gz") for path in self.snapshot_dir.glob("*.json.gz")),
            reverse=True,
        )

    def load(self, snapshot_date: str | None = None) -> ScanSnapshot | None:
        available = self.dates()
        selected = snapshot_date or (available[0] if available else None)
        if selected is None:
            return None
        path = self.snapshot_dir / f"{selected}.json.gz"
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return ScanSnapshot.from_payload(json.load(handle))


def _pick_column(frame: pd.DataFrame, *needles: str, exact: tuple[str, ...] = ()) -> str | None:
    for candidate in exact:
        if candidate in frame.columns:
            return candidate
    for column in frame.columns:
        label = str(column)
        if all(needle in label for needle in needles):
            return column
    return None


def _parse_number(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "-", "--", "None", "nan"}:
        return np.nan
    multiplier = 1.0
    if text.endswith("亿"):
        multiplier, text = 100_000_000.0, text[:-1]
    elif text.endswith("万"):
        multiplier, text = 10_000.0, text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def _numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return frame[column].map(_parse_number).astype("float64")


def _stock_code(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, float, np.number)):
        return str(int(value)).zfill(6)
    text = str(value).strip().split(".", 1)[0]
    return text.zfill(6) if text.isdigit() else text


def normalize_market_flow(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    date_col = _pick_column(frame, exact=("日期", "date"))
    output = pd.DataFrame(index=frame.index)
    if date_col:
        parsed = pd.to_datetime(frame[date_col], errors="coerce")
        output["date"] = parsed.dt.strftime("%Y-%m-%d")
    else:
        output["date"] = pd.Series(range(len(frame)), index=frame.index).astype(str)
    mappings = {
        "sh_close": ("上证", "收盘价"),
        "sh_change_pct": ("上证", "涨跌幅"),
        "sz_close": ("深证", "收盘价"),
        "sz_change_pct": ("深证", "涨跌幅"),
        "main_net": ("主力净流入", "净额"),
        "main_ratio": ("主力净流入", "净占比"),
        "ultra_net": ("超大单净流入", "净额"),
        "big_net": ("大单净流入", "净额"),
        "middle_net": ("中单净流入", "净额"),
        "small_net": ("小单净流入", "净额"),
    }
    for target, needles in mappings.items():
        output[target] = _numeric(frame, _pick_column(frame, *needles))
    return output.dropna(subset=["date"]).reset_index(drop=True)


def normalize_rank_flow(frame: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    output = pd.DataFrame(index=frame.index)
    if entity_type == "stock":
        code_col = _pick_column(frame, exact=("代码", "股票代码"))
        name_col = _pick_column(frame, exact=("名称", "股票简称"))
        output["code"] = frame[code_col].map(_stock_code) if code_col else ""
        output["name"] = frame[name_col].astype(str).str.strip() if name_col else output["code"]
        price_col = _pick_column(frame, exact=("最新价",))
        output["price"] = _numeric(frame, price_col)
    else:
        name_col = _pick_column(frame, exact=("名称", "行业", "概念"))
        if name_col is None:
            return pd.DataFrame()
        output["name"] = frame[name_col].astype(str).str.strip()

    output["change_pct"] = _numeric(frame, _pick_column(frame, "涨跌幅"))
    main_net_column = _pick_column(frame, "主力净流入", "净额")
    if main_net_column is None:
        main_net_column = _pick_column(frame, exact=("资金流入净额", "净额"))
    output["main_net"] = _numeric(frame, main_net_column)
    # 同花顺板块接口的“净额”数值单位固定为亿元；统一换算为元。
    if entity_type == "sector" and main_net_column == "净额":
        output["main_net"] *= 100_000_000
    output["main_ratio"] = _numeric(frame, _pick_column(frame, "主力净流入", "净占比"))
    output["ultra_net"] = _numeric(frame, _pick_column(frame, "超大单净流入", "净额"))
    output["big_net"] = _numeric(frame, _pick_column(frame, "大单净流入", "净额"))
    output["middle_net"] = _numeric(frame, _pick_column(frame, "中单净流入", "净额"))
    output["small_net"] = _numeric(frame, _pick_column(frame, "小单净流入", "净额"))
    leader_col = _pick_column(frame, "主力净流入最大股", exact=("领涨股",))
    if leader_col:
        output["leader"] = frame[leader_col].astype(str).str.strip()

    identity = "code" if entity_type == "stock" else "name"
    return output[output[identity].astype(str).str.len() > 0].reset_index(drop=True)


def _capital_signal(row: pd.Series, net_columns: list[str]) -> str:
    values = [row[column] for column in net_columns if pd.notna(row.get(column))]
    today = row.get("main_net_today")
    if not values:
        return "数据不足"
    if len(values) >= 2 and all(value > 0 for value in values):
        return "连续流入"
    if len(values) >= 2 and all(value < 0 for value in values):
        return "连续流出"
    if pd.notna(today) and today > 0:
        return "当日流入"
    if pd.notna(today) and today < 0:
        return "当日流出"
    return "方向分化"


def _price_heat(value) -> str:
    if pd.isna(value):
        return "未知"
    if value >= 7:
        return "高位活跃"
    if value >= 3:
        return "偏热"
    if value <= -3:
        return "走弱"
    return "温和"


def build_cross_horizon_rank(
    frames: dict[str, pd.DataFrame], entity_type: str
) -> pd.DataFrame:
    identity = "code" if entity_type == "stock" else "name"
    merged: pd.DataFrame | None = None
    name_columns: list[str] = []
    for horizon_key, frame in frames.items():
        if frame is None or frame.empty:
            continue
        working = frame.copy()
        rename = {
            column: f"{column}_{horizon_key}"
            for column in working.columns
            if column != identity
        }
        working = working.rename(columns=rename)
        if entity_type == "stock" and f"name_{horizon_key}" in working:
            name_columns.append(f"name_{horizon_key}")
        merged = working if merged is None else merged.merge(working, on=identity, how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame()
    if entity_type == "stock":
        merged["name"] = merged[name_columns].bfill(axis=1).iloc[:, 0]
        merged = merged.drop(columns=name_columns)

    component_weights = {
        "main_net_today": 0.30,
        "main_ratio_today": 0.15,
        "main_net_d5": 0.20,
        "main_ratio_d5": 0.10,
        "main_net_d10": 0.15,
        "main_ratio_d10": 0.10,
    }
    numerator = pd.Series(0.0, index=merged.index)
    denominator = pd.Series(0.0, index=merged.index)
    for column, weight in component_weights.items():
        if column not in merged:
            continue
        valid = merged[column].notna()
        percentile = merged[column].rank(pct=True, method="average") * 100
        numerator = numerator.add(percentile.fillna(0) * weight, fill_value=0)
        denominator = denominator.add(valid.astype(float) * weight, fill_value=0)
    merged["flow_score"] = (numerator / denominator.replace(0, np.nan)).round(1)

    net_columns = [column for column in ("main_net_today", "main_net_d5", "main_net_d10") if column in merged]
    merged["positive_periods"] = merged[net_columns].gt(0).sum(axis=1) if net_columns else 0
    merged["available_periods"] = merged[net_columns].notna().sum(axis=1) if net_columns else 0
    merged["capital_signal"] = merged.apply(lambda row: _capital_signal(row, net_columns), axis=1)
    change_column = "change_pct_today"
    merged["price_heat"] = (
        merged[change_column].map(_price_heat) if change_column in merged else "未知"
    )
    merged = merged.sort_values("flow_score", ascending=False, na_position="last").reset_index(drop=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))
    return merged


def _latest_weekday(moment: datetime) -> str:
    candidate = moment.date()
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate.isoformat()


def _market_from_stock_aggregate(
    today_stocks: pd.DataFrame,
    previous: ScanSnapshot | None,
    now: datetime,
) -> pd.DataFrame:
    if today_stocks is None or today_stocks.empty:
        return pd.DataFrame()
    data_date = _latest_weekday(now)
    row = {
        "date": data_date,
        "sh_close": np.nan,
        "sh_change_pct": np.nan,
        "sz_close": np.nan,
        "sz_change_pct": np.nan,
        "main_net": today_stocks["main_net"].sum(min_count=1),
        "main_ratio": np.nan,
        "ultra_net": today_stocks["ultra_net"].sum(min_count=1),
        "big_net": today_stocks["big_net"].sum(min_count=1),
        "middle_net": today_stocks["middle_net"].sum(min_count=1),
        "small_net": today_stocks["small_net"].sum(min_count=1),
    }
    history = previous.market.copy() if previous is not None else pd.DataFrame()
    if not history.empty and "date" in history:
        history = history[history["date"].astype(str) != data_date]
    return pd.concat([history, pd.DataFrame([row])], ignore_index=True).tail(120)


class FundFlowPipeline:
    """顺序调用免费接口，允许局部失败，并将可用部分保存为日快照。"""

    def __init__(
        self,
        provider: FundFlowProvider | None = None,
        store: SnapshotStore | None = None,
    ):
        self.provider = provider or AKShareFundFlowProvider()
        self.store = store or SnapshotStore()

    def run(self, progress: ProgressCallback | None = None) -> ScanSnapshot:
        errors: list[str] = []
        market = pd.DataFrame()
        sector_frames: dict[str, dict[str, pd.DataFrame]] = {
            "industry": {},
            "concept": {},
        }
        stock_frames: dict[str, pd.DataFrame] = {}
        tasks: list[tuple[str, Callable[[], pd.DataFrame], Callable[[pd.DataFrame], None]]] = []

        tasks.append(("大盘资金", self.provider.market_flow, lambda frame: None))
        for sector_key, sector_label in SECTOR_TYPES.items():
            for indicator, horizon_key in HORIZONS.items():
                tasks.append(
                    (
                        f"{sector_label} · {indicator}",
                        lambda i=indicator, s=sector_label: self.provider.sector_flow(i, s),
                        lambda frame, s=sector_key, h=horizon_key: sector_frames[s].__setitem__(
                            h, normalize_rank_flow(frame, "sector")
                        ),
                    )
                )
        for indicator, horizon_key in HORIZONS.items():
            tasks.append(
                (
                    f"个股资金 · {indicator}",
                    lambda i=indicator: self.provider.stock_flow(i),
                    lambda frame, h=horizon_key: stock_frames.__setitem__(
                        h, normalize_rank_flow(frame, "stock")
                    ),
                )
            )

        total = len(tasks)
        for position, (label, loader, sink) in enumerate(tasks, start=1):
            if progress:
                progress(label, position - 1, total)
            try:
                raw = loader()
                if label == "大盘资金":
                    market = normalize_market_flow(raw)
                else:
                    sink(raw)
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
            if progress:
                progress(label, position, total)

        industries = build_cross_horizon_rank(sector_frames["industry"], "sector")
        concepts = build_cross_horizon_rank(sector_frames["concept"], "sector")
        stocks = build_cross_horizon_rank(stock_frames, "stock")
        market_derived = False
        if market.empty and "today" in stock_frames:
            market = _market_from_stock_aggregate(
                stock_frames["today"], self.store.load(), datetime.now(SHANGHAI_TZ)
            )
            market_derived = not market.empty
        if market.empty and industries.empty and concepts.empty and stocks.empty:
            detail = " | ".join(errors) if errors else "接口未返回数据"
            raise FundFlowUnavailable(detail)

        now = datetime.now(SHANGHAI_TZ)
        data_date = _latest_weekday(now)
        if not market.empty and market["date"].notna().any():
            data_date = str(market["date"].dropna().iloc[-1])
        coverage = {
            "market": not market.empty,
            "industry": sorted(sector_frames["industry"]),
            "concept": sorted(sector_frames["concept"]),
            "stock": sorted(stock_frames),
        }
        snapshot = ScanSnapshot(
            metadata={
                "snapshot_date": now.date().isoformat(),
                "data_date": data_date,
                "scanned_at": now.isoformat(timespec="seconds"),
                "provider": self.provider.name,
                "provider_version": self.provider.version,
                "market_derived_from_stocks": market_derived,
                "coverage": coverage,
            },
            market=market.tail(120).reset_index(drop=True),
            industries=industries,
            concepts=concepts,
            stocks=stocks,
            errors=tuple(errors),
        )
        self.store.save(snapshot)
        return snapshot
