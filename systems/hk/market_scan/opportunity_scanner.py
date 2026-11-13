"""港股市场热度与原有 7 策略、样本外证据融合的机会扫描器。"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from market_data import fetch_history
from quant_research import QuantResearchEngine, ResearchEvidence
from strategies.composite import CompositeStrategy
from trade_decision import risk_reward_metrics
from watchlist import load_user_stocks

from .pipeline import HONG_KONG_TZ, ProgressCallback, ScanSnapshot, SnapshotStore

MIN_HEAT_SCORE = 55.0
MIN_LIQUIDITY_SCORE = 45.0
MIN_DAILY_TURNOVER = 5_000_000.0
MIN_COMPOSITE_SCORE = 5.0
MIN_CONFIDENCE = 0.43
MIN_BUY_VOTES = 3
MIN_TREND_QUALITY = 70.0
MIN_OPPORTUNITY_SCORE = 68.0
MAX_FRESHNESS_LAG_DAYS = 5
MIN_CURRENT_REWARD_RISK = 1.30
MAX_STOP_DISTANCE_PCT = 0.12


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
class OpportunitySnapshot:
    metadata: dict
    evaluated: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: tuple[str, ...] = ()

    @property
    def snapshot_date(self) -> str:
        return str(self.metadata["snapshot_date"])

    @property
    def recommended(self) -> pd.DataFrame:
        if self.evaluated.empty or "is_recommended" not in self.evaluated:
            return pd.DataFrame()
        return self.evaluated[self.evaluated["is_recommended"].fillna(False)].copy()

    def to_payload(self) -> dict:
        return {
            "schema_version": 2,
            "metadata": self.metadata,
            "evaluated": _records(self.evaluated),
            "errors": list(self.errors),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> OpportunitySnapshot:
        return cls(
            metadata=payload.get("metadata", {}),
            evaluated=pd.DataFrame(payload.get("evaluated", [])),
            errors=tuple(payload.get("errors", [])),
        )


class OpportunityStore:
    def __init__(self, data_dir: str | Path | None = None):
        configured = data_dir or os.environ.get("HK_MARKET_SCAN_DATA_DIR")
        self.data_dir = Path(configured) if configured else SnapshotStore().data_dir
        self.snapshot_dir = self.data_dir / "opportunities"

    def save(self, snapshot: OpportunitySnapshot) -> Path:
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

    def load(self, snapshot_date: str | None = None) -> OpportunitySnapshot | None:
        available = self.dates()
        selected = snapshot_date or (available[0] if available else None)
        if selected is None:
            return None
        path = self.snapshot_dir / f"{selected}.json.gz"
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return OpportunitySnapshot.from_payload(json.load(handle))


def build_candidate_pool(
    snapshot: ScanSnapshot,
    max_candidates: int = 30,
    heat_top: int = 22,
    turnover_top: int = 10,
    always_candidates: dict[str, str] | None = None,
) -> pd.DataFrame:
    stocks = snapshot.stocks.copy()
    if stocks.empty:
        return pd.DataFrame()
    normalized_always: dict[str, str] = {}
    for value, label in (always_candidates or {}).items():
        try:
            normalized_always[cfg.normalize_ticker(str(value))[2]] = str(label)
        except ValueError:
            continue
    codes = stocks["code"].astype(str).str.zfill(5)
    turnover_rank = pd.to_numeric(stocks["turnover"], errors="coerce").rank(
        method="min", ascending=False
    )
    selected = stocks[
        pd.to_numeric(stocks["rank"], errors="coerce").le(heat_top)
        | turnover_rank.le(turnover_top)
        | codes.isin(normalized_always)
    ].copy()
    if selected.empty:
        return selected

    def source(row: pd.Series) -> str:
        labels = []
        if float(row.get("rank") or 99999) <= heat_top:
            labels.append("全市热度榜")
        if float(turnover_rank.loc[row.name]) <= turnover_top:
            labels.append("成交额前列")
        raw = str(row.get("code") or "").zfill(5)
        if raw in normalized_always:
            labels.append(normalized_always[raw])
        return " · ".join(labels)

    selected["discovery_source"] = selected.apply(source, axis=1)
    selected["always_included"] = selected["code"].astype(str).str.zfill(5).isin(normalized_always)
    selected = selected.sort_values(
        ["always_included", "heat_score", "turnover"], ascending=[False, False, False]
    )
    return selected.drop_duplicates("code").head(max_candidates).reset_index(drop=True)


HistoryLoader = Callable[[str, int], tuple[pd.DataFrame | None, str | None, tuple[str, ...]]]
SignalEvaluator = Callable[[pd.DataFrame], object]
ResearchEvaluator = Callable[[pd.DataFrame], ResearchEvidence | dict]


class OpportunityScanner:
    """热度只负责发现候选；7 策略和样本外证据共同决定推荐。"""

    def __init__(
        self,
        store: OpportunityStore | None = None,
        history_loader: HistoryLoader | None = None,
        signal_evaluator: SignalEvaluator | None = None,
        research_evaluator: ResearchEvaluator | None = None,
        max_candidates: int = 30,
        max_workers: int = 4,
        always_candidates: dict[str, str] | None = None,
    ):
        self.store = store or OpportunityStore()
        self.history_loader = history_loader or (
            lambda ticker, days: fetch_history("tencent", ticker, days)
        )
        self.signal_evaluator = signal_evaluator or (
            lambda frame: CompositeStrategy().evaluate(frame.tail(365))
        )
        self.research_evaluator = research_evaluator or QuantResearchEngine().evaluate
        self.max_candidates = max(1, max_candidates)
        self.max_workers = max(1, min(max_workers, 8))
        self.always_candidates = always_candidates

    @staticmethod
    def _safe(value, default: float = 0.0) -> float:
        try:
            number = float(value)
            return number if np.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _default_always() -> dict[str, str]:
        candidates = {ticker: "平台预设股" for ticker in cfg.PRESET_STOCKS}
        for ticker in load_user_stocks(cfg.normalize_ticker):
            candidates[ticker] = "用户自选股"
        return candidates

    @classmethod
    def _trend_quality(cls, history: pd.DataFrame) -> float:
        close = history["close"].astype(float)
        ma20 = history["MA20"] if "MA20" in history else close.rolling(20).mean()
        ma60 = history["MA60"] if "MA60" in history else close.rolling(60).mean()
        latest, latest20, latest60 = float(close.iloc[-1]), cls._safe(ma20.iloc[-1]), cls._safe(ma60.iloc[-1])
        previous20 = cls._safe(ma20.iloc[-6], latest20) if len(ma20) >= 6 else latest20
        atr = cls._safe(history["ATR"].iloc[-1], latest * 0.02) if "ATR" in history else latest * 0.02
        checks = [
            (latest > latest20, 25),
            (latest > latest60, 25),
            (latest20 > latest60, 20),
            (latest20 > previous20, 15),
            (latest <= latest20 + 2.5 * max(atr, 0.001), 15),
        ]
        return float(sum(weight for passed, weight in checks if passed))

    def _evaluate_one(self, row: dict, data_date: str) -> dict:
        result = dict(row)
        code = str(row.get("code") or "").zfill(5)
        result.update(
            code=code,
            is_recommended=False,
            evaluation_status="行情不可用",
            rejection_reasons="",
            selection_path="未通过",
        )
        ticker = cfg.normalize_ticker(code)[0]
        history, source, failures = self.history_loader(ticker, 800)
        result["market_source"] = source or ""
        result["market_failures"] = " | ".join(failures or ())
        if history is not None and not history.empty:
            expected = pd.Timestamp(data_date).date()
            dates = pd.to_datetime(history.index, errors="coerce")
            history = history[dates.date <= expected].copy()
        if history is None or history.empty or len(history) < 60:
            result["rejection_reasons"] = "历史 K 线不足 60 根"
            return result

        signal = self.signal_evaluator(history)
        try:
            research = self.research_evaluator(history)
            research_record = research.to_record() if hasattr(research, "to_record") else dict(research)
        except Exception as exc:  # noqa: BLE001 - quarantine one-symbol research failures
            research_record = ResearchEvidence(
                "研究失败", False, "不可评估", reasons=(f"{type(exc).__name__}: {exc}",)
            ).to_record()
        latest_date = pd.Timestamp(history.index[-1]).date()
        expected = pd.Timestamp(data_date).date()
        freshness = max(0, (expected - latest_date).days)
        score = self._safe(getattr(signal, "score", 0))
        confidence = self._safe(getattr(signal, "confidence", 0))
        technical = np.clip((score + 10) * 5, 0, 100)
        heat = self._safe(row.get("heat_score"))
        liquidity = self._safe(row.get("liquidity_score"))
        trend = self._trend_quality(history)
        opportunity = round(technical * 0.60 + heat * 0.15 + confidence * 100 * 0.10 + trend * 0.15, 1)
        latest_close = float(history["close"].iloc[-1])
        stop_loss = self._safe(getattr(signal, "stop_loss", 0))
        take_profit = self._safe(getattr(signal, "take_profit", 0))
        price_risk = risk_reward_metrics(latest_close, stop_loss, take_profit)
        result.update(
            evaluation_status="已完成",
            latest_bar_date=latest_date.isoformat(),
            freshness_lag_days=freshness,
            latest_close=round(latest_close, 3),
            strategy_direction=str(getattr(signal, "direction", "neutral")),
            strategy_score=round(score, 1),
            strategy_confidence=round(confidence, 2),
            buy_votes=int(getattr(signal, "buy_votes", 0) or 0),
            sell_votes=int(getattr(signal, "sell_votes", 0) or 0),
            recommended_position=float(getattr(signal, "recommended_position", 0) or 0),
            stop_loss=round(stop_loss, 3),
            take_profit=round(take_profit, 3),
            reward_risk=round(price_risk["reward_risk_now"], 2),
            stop_distance_pct=round(price_risk["stop_distance_pct"], 4),
            trend_quality=trend,
            opportunity_score=opportunity,
            **research_record,
        )

        gaps = []
        change = self._safe(row.get("change_pct"))
        turnover = self._safe(row.get("turnover"))
        if change > 20:
            gaps.append(f"单日上涨 {change:.1f}% 已过热，不追涨")
        if change < -15:
            gaps.append(f"单日下跌 {change:.1f}% 波动异常")
        if turnover < MIN_DAILY_TURNOVER:
            gaps.append(f"成交额 HKD {turnover:,.0f}<5,000,000")
        if freshness > MAX_FRESHNESS_LAG_DAYS:
            gaps.append(f"日 K 滞后 {freshness} 天")
        if stop_loss <= 0 or stop_loss >= latest_close:
            gaps.append("保护止损无效或已跌破")
        if take_profit <= latest_close:
            gaps.append("止盈目标已到或无效")
        if stop_loss > 0 and stop_loss < latest_close and take_profit > latest_close:
            if price_risk["reward_risk_now"] < MIN_CURRENT_REWARD_RISK:
                gaps.append(
                    f"现价盈亏比 {price_risk['reward_risk_now']:.2f}R<"
                    f"{MIN_CURRENT_REWARD_RISK:.2f}R"
                )
            if price_risk["stop_distance_pct"] > MAX_STOP_DISTANCE_PCT:
                gaps.append(
                    f"止损距离 {price_risk['stop_distance_pct']:.1%}>"
                    f"{MAX_STOP_DISTANCE_PCT:.0%}"
                )
        if opportunity < MIN_OPPORTUNITY_SCORE:
            gaps.append(f"综合机会分 {opportunity:.1f}<{MIN_OPPORTUNITY_SCORE:.0f}")
        if result["strategy_direction"] != "strong_buy" or score < MIN_COMPOSITE_SCORE:
            gaps.append(f"7 策略分 {score:+.1f}<+{MIN_COMPOSITE_SCORE:.0f} 或非强买")
        if result["buy_votes"] < MIN_BUY_VOTES or result["sell_votes"] > 1:
            gaps.append(f"策略投票 {result['buy_votes']} 买/{result['sell_votes']} 卖未通过")
        if confidence < MIN_CONFIDENCE:
            gaps.append(f"策略信心 {confidence:.0%}<{MIN_CONFIDENCE:.0%}")
        if heat < MIN_HEAT_SCORE or liquidity < MIN_LIQUIDITY_SCORE:
            gaps.append(f"热度/流动性 {heat:.0f}/{liquidity:.0f} 未通过")
        if trend < MIN_TREND_QUALITY:
            gaps.append(f"趋势质量 {trend:.0f}<{MIN_TREND_QUALITY:.0f}")
        if not bool(result.get("research_gate_pass")):
            gaps.append("样本外证据：" + str(result.get("research_reasons") or "未通过"))
        if not gaps:
            result["is_recommended"] = True
            result["selection_path"] = "热度发现·7 策略强买·样本外验证"
        result["rejection_reasons"] = "；".join(gaps)
        return result

    def run(self, snapshot: ScanSnapshot, progress: ProgressCallback | None = None) -> OpportunitySnapshot:
        always = self._default_always() if self.always_candidates is None else self.always_candidates
        candidates = build_candidate_pool(snapshot, self.max_candidates, always_candidates=always)
        rows = candidates.to_dict(orient="records")
        evaluated = []
        if progress:
            progress("准备批量策略评分", 0, max(1, len(rows)))
        if rows:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(rows)), thread_name_prefix="hk-score") as executor:
                futures = {
                    executor.submit(self._evaluate_one, row, str(snapshot.metadata.get("data_date"))): row
                    for row in rows
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    evaluated.append(future.result())
                    if progress:
                        progress(f"策略评分 · {futures[future].get('name', '')}", completed, len(rows))
        frame = pd.DataFrame(evaluated)
        if not frame.empty:
            frame = frame.sort_values(
                ["is_recommended", "research_probability", "research_quality", "opportunity_score", "heat_score"],
                ascending=[False, False, False, False, False],
                na_position="last",
            ).reset_index(drop=True)
            frame.insert(0, "opportunity_rank", range(1, len(frame) + 1))
        errors = tuple(
            f"{row.get('code')} {row.get('name')}: {row.get('rejection_reasons')}"
            for row in evaluated
            if row.get("evaluation_status") != "已完成"
        )
        now = datetime.now(HONG_KONG_TZ)
        result = OpportunitySnapshot(
            metadata={
                "snapshot_date": snapshot.snapshot_date,
                "data_date": snapshot.metadata.get("data_date"),
                "scored_at": now.isoformat(timespec="seconds"),
                "candidate_count": len(frame),
                "evaluated_count": int(frame.get("evaluation_status", pd.Series(dtype=str)).eq("已完成").sum()),
                "recommended_count": int(frame.get("is_recommended", pd.Series(dtype=bool)).fillna(False).sum()),
                "strategy": "港股热度发现 + 7 策略 + 防泄漏样本外验证",
                "round_trip_cost_bps": round(QuantResearchEngine().round_trip_cost * 10_000, 1),
            },
            evaluated=frame,
            errors=errors,
        )
        self.store.save(result)
        return result
