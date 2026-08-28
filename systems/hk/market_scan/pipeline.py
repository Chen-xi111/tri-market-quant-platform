"""腾讯批量行情驱动的港股全市场扫描 pipeline。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import gzip
import json
import os
from pathlib import Path
from typing import Callable, Protocol
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config as cfg


HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / ".hk_scan_data"
ProgressCallback = Callable[[str, int, int], None]


class MarketScanUnavailable(RuntimeError):
    """港股批量行情完全不可用。"""


class HKMarketProvider(Protocol):
    name: str
    version: str

    def stock_quotes(self, progress: ProgressCallback | None = None) -> pd.DataFrame: ...


def _json_default(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    clean = frame.replace([np.inf, -np.inf], np.nan).astype(object)
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


@dataclass
class ScanSnapshot:
    metadata: dict
    market: pd.DataFrame = field(default_factory=pd.DataFrame)
    stocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: tuple[str, ...] = ()

    @property
    def snapshot_date(self) -> str:
        return str(self.metadata["snapshot_date"])

    def to_payload(self) -> dict:
        return {
            "schema_version": 1,
            "metadata": self.metadata,
            "market": _records(self.market),
            "stocks": _records(self.stocks),
            "errors": list(self.errors),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ScanSnapshot":
        return cls(
            metadata=payload.get("metadata", {}),
            market=pd.DataFrame(payload.get("market", [])),
            stocks=pd.DataFrame(payload.get("stocks", [])),
            errors=tuple(payload.get("errors", [])),
        )


class SnapshotStore:
    """每天原子保存一份 gzip JSON 扫描快照。"""

    def __init__(self, data_dir: str | Path | None = None):
        configured = data_dir or os.environ.get("HK_MARKET_SCAN_DATA_DIR")
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


class TencentHKMarketProvider:
    """批量拉取腾讯港股快照；首次发现全代码，之后复用每周更新的股票池。"""

    name = "腾讯证券批量行情"
    version = "hk-quote-v1"
    chunk_size = 70
    max_workers = 8
    request_timeout = 6.0

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.universe_path = self.data_dir / "universe.json"

    def _known_codes(self) -> tuple[list[str], bool]:
        try:
            age = datetime.now().timestamp() - self.universe_path.stat().st_mtime
            payload = json.loads(self.universe_path.read_text(encoding="utf-8"))
            codes = [str(code).zfill(5) for code in payload.get("codes", [])]
            if codes and age < 7 * 24 * 60 * 60:
                return sorted(set(codes)), False
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            pass
        return [str(code).zfill(5) for code in range(1, 10_000)], True

    @staticmethod
    def _number(fields: list[str], index: int):
        try:
            value = fields[index]
            return float(value) if value not in ("", "-") else np.nan
        except (IndexError, TypeError, ValueError):
            return np.nan

    @classmethod
    def _parse_line(cls, line: str) -> dict | None:
        try:
            fields = line.split('"', 2)[1].split("~")
        except IndexError:
            return None
        if len(fields) < 38 or fields[0] != "100":
            return None
        code = str(fields[2]).zfill(5)
        price = cls._number(fields, 3)
        previous = cls._number(fields, 4)
        if not code.isdigit() or not np.isfinite(price) or price <= 0:
            return None
        instrument_type = fields[63] if len(fields) > 63 else ""
        if instrument_type and instrument_type != "GP":
            return None
        timestamp = fields[30] if len(fields) > 30 else ""
        parsed_time = pd.to_datetime(timestamp, errors="coerce")
        return {
            "code": code,
            "name": str(fields[1]).strip() or code,
            "price": price,
            "prev_close": previous,
            "open": cls._number(fields, 5),
            "high": cls._number(fields, 33),
            "low": cls._number(fields, 34),
            "volume": cls._number(fields, 36),
            "turnover": cls._number(fields, 37),
            "change": cls._number(fields, 31),
            "change_pct": cls._number(fields, 32),
            "turnover_rate": cls._number(fields, 59),
            "lot_size": cls._number(fields, 60),
            "quote_time": parsed_time.isoformat() if pd.notna(parsed_time) else timestamp,
            "instrument_type": instrument_type or "GP",
        }

    def _fetch_chunk(self, codes: list[str]) -> list[dict]:
        symbols = ",".join(f"hk{code}" for code in codes)
        request = urllib.request.Request(
            f"https://qt.gtimg.cn/q={symbols}",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            text = response.read().decode("gb18030", errors="replace")
        return [record for line in text.splitlines() if (record := self._parse_line(line))]

    def _save_universe(self, frame: pd.DataFrame) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            temporary = self.universe_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "updated_at": datetime.now(HONG_KONG_TZ).isoformat(timespec="seconds"),
                        "codes": frame["code"].astype(str).tolist(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self.universe_path)
        except OSError:
            return

    def stock_quotes(self, progress: ProgressCallback | None = None) -> pd.DataFrame:
        codes, discovery = self._known_codes()
        preset_codes = [cfg.normalize_ticker(ticker)[2] for ticker in cfg.PRESET_STOCKS]
        codes = sorted(set([*codes, *preset_codes]))
        chunks = [codes[index:index + self.chunk_size] for index in range(0, len(codes), self.chunk_size)]
        rows: list[dict] = []
        errors: list[str] = []
        if progress:
            progress("发现港股代码" if discovery else "读取港股股票池", 0, len(chunks))
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="hk-market") as executor:
            futures = {executor.submit(self._fetch_chunk, chunk): position for position, chunk in enumerate(chunks)}
            for completed, future in enumerate(as_completed(futures), start=1):
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    errors.append(f"批次 {futures[future] + 1}: {type(exc).__name__}: {exc}")
                if progress:
                    progress("腾讯实时批量报价", completed, len(chunks))
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise MarketScanUnavailable("；".join(errors) or "腾讯批量行情返回空数据")
        frame = frame.drop_duplicates("code", keep="last").sort_values("code").reset_index(drop=True)
        frame.attrs["partial_errors"] = tuple(errors)
        frame.attrs["universe_discovery"] = discovery
        if discovery:
            self._save_universe(frame)
        return frame


def score_market(frame: pd.DataFrame) -> pd.DataFrame:
    """将成交活跃度、动量、换手和收盘位置合成为 0–100 热度分。"""
    stocks = frame.copy()
    for column in ("turnover", "change_pct", "turnover_rate", "price", "high", "low"):
        stocks[column] = pd.to_numeric(stocks.get(column), errors="coerce")
    liquidity_pct = np.log1p(stocks["turnover"].clip(lower=0)).rank(pct=True) * 100
    momentum_pct = stocks["change_pct"].rank(pct=True) * 100
    turnover_pct = stocks["turnover_rate"].rank(pct=True) * 100
    price_range = (stocks["high"] - stocks["low"]).replace(0, np.nan)
    close_location = ((stocks["price"] - stocks["low"]) / price_range * 100).clip(0, 100).fillna(50)
    heat = 0.45 * liquidity_pct + 0.25 * momentum_pct + 0.15 * turnover_pct.fillna(50) + 0.15 * close_location
    heat -= np.where(stocks["turnover"].fillna(0) < 2_000_000, 15, 0)
    heat -= np.where(stocks["price"].fillna(0) < 0.10, 20, 0)
    heat -= np.where(stocks["change_pct"].abs().fillna(0) > 20, 15, 0)
    stocks["heat_score"] = np.clip(heat, 0, 100).round(1)
    stocks["flow_score"] = stocks["heat_score"]
    stocks["liquidity_score"] = liquidity_pct.round(1)
    stocks["momentum_score"] = momentum_pct.round(1)
    stocks["close_location"] = close_location.round(1)
    stocks["change_pct_today"] = stocks["change_pct"]
    stocks["market_state"] = np.select(
        [
            (stocks["change_pct"] > 0) & (stocks["liquidity_score"] >= 70),
            stocks["change_pct"] > 0,
            (stocks["change_pct"] < 0) & (stocks["liquidity_score"] >= 70),
        ],
        ["高成交上涨", "上涨", "高成交调整"],
        default="调整",
    )
    stocks = stocks.sort_values(
        ["heat_score", "turnover"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    stocks.insert(0, "rank", range(1, len(stocks) + 1))
    return stocks


def _market_row(stocks: pd.DataFrame, data_date: str) -> dict:
    changes = stocks["change_pct"].dropna()
    return {
        "date": data_date,
        "turnover": stocks["turnover"].sum(min_count=1),
        "rising": int((changes > 0).sum()),
        "falling": int((changes < 0).sum()),
        "unchanged": int((changes == 0).sum()),
        "breadth": round(float((changes > 0).mean() * 100), 1) if len(changes) else np.nan,
        "median_change": round(float(changes.median()), 2) if len(changes) else np.nan,
        "hot_count": int((stocks["heat_score"] >= 75).sum()),
        "liquid_count": int((stocks["turnover"].fillna(0) >= 10_000_000).sum()),
    }


class MarketScanPipeline:
    def __init__(self, provider: HKMarketProvider | None = None, store: SnapshotStore | None = None):
        self.store = store or SnapshotStore()
        self.provider = provider or TencentHKMarketProvider(self.store.data_dir)

    def run(self, progress: ProgressCallback | None = None) -> ScanSnapshot:
        raw = self.provider.stock_quotes(progress)
        errors = tuple(raw.attrs.get("partial_errors", ()))
        stocks = score_market(raw)
        quote_dates = pd.to_datetime(stocks["quote_time"], errors="coerce").dropna()
        now = datetime.now(HONG_KONG_TZ)
        data_date = (
            quote_dates.max().date().isoformat()
            if not quote_dates.empty
            else (now.date() - timedelta(days=1 if now.weekday() else 3)).isoformat()
        )
        previous = self.store.load()
        market = previous.market.copy() if previous is not None else pd.DataFrame()
        if not market.empty and "date" in market:
            market = market[market["date"].astype(str) != data_date]
        market = pd.concat([market, pd.DataFrame([_market_row(stocks, data_date)])], ignore_index=True).tail(120)
        snapshot = ScanSnapshot(
            metadata={
                "snapshot_date": now.date().isoformat(),
                "data_date": data_date,
                "scanned_at": now.isoformat(timespec="seconds"),
                "latest_quote_at": str(stocks["quote_time"].max()),
                "provider": self.provider.name,
                "provider_version": self.provider.version,
                "universe_count": len(stocks),
                "universe_discovery": bool(raw.attrs.get("universe_discovery")),
                "coverage": "00001–09999 普通股（腾讯返回有效代码）",
            },
            market=market.reset_index(drop=True),
            stocks=stocks,
            errors=errors,
        )
        self.store.save(snapshot)
        return snapshot
