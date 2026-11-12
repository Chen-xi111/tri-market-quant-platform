"""资金流热点与原有多策略的每日机会扫描器。"""

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
from quant_research import QuantResearchEngine, ResearchEvidence
from trade_decision import risk_reward_metrics
from trading_rules import price_limit_pct
from watchlist import load_user_stocks

import config as cfg
from market_data import fetch_history
from strategies.composite import CompositeStrategy

from .pipeline import SHANGHAI_TZ, ProgressCallback, ScanSnapshot, SnapshotStore

MIN_FLOW_SCORE = 75.0
MIN_COMPOSITE_SCORE = 5.0
MIN_TECHNICAL_PATH_SCORE = 5.0
MIN_RESEARCH_TRIGGER_SCORE = 3.0
MIN_CONFIDENCE = 0.28
MIN_TECHNICAL_CONFIDENCE = 0.43
MIN_BUY_VOTES = 2
MIN_TECHNICAL_BUY_VOTES = 3
MIN_OPPORTUNITY_SCORE = 70.0
MIN_POSITIVE_PERIODS = 2
MAX_FRESHNESS_LAG_DAYS = 5
MIN_FUND_TREND_QUALITY = 60.0
MIN_TECHNICAL_TREND_QUALITY = 75.0
MIN_RESEARCH_TREND_QUALITY = 75.0
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
    """按资金流快照日期原子保存批量评分结果。"""

    def __init__(self, data_dir: str | Path | None = None):
        configured = data_dir or os.environ.get("CN_FUND_FLOW_DATA_DIR")
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


def _sector_leader_tags(
    frame: pd.DataFrame,
    label: str,
    limit: int,
) -> dict[str, set[str]]:
    tags: dict[str, set[str]] = {}
    if frame is None or frame.empty or "leader_today" not in frame:
        return tags
    selected = frame.copy()
    if "main_net_today" in selected:
        selected = selected[selected["main_net_today"].fillna(0) > 0]
    if "flow_score" in selected:
        selected = selected[selected["flow_score"].fillna(0) >= 70]
        selected = selected.sort_values("flow_score", ascending=False)
    for _, row in selected.head(limit).iterrows():
        leader = str(row.get("leader_today") or "").strip()
        sector = str(row.get("name") or "").strip()
        if leader and leader.lower() != "nan" and sector:
            tags.setdefault(leader, set()).add(f"{label}·{sector}")
    return tags


def build_candidate_pool(
    snapshot: ScanSnapshot,
    max_candidates: int = 50,
    global_top: int = 18,
    sector_leaders_per_type: int = 8,
    always_candidates: dict[str, str] | None = None,
) -> pd.DataFrame:
    """合并资金榜、热门板块领涨股与必扫自选，生成去重候选池。"""
    stocks = snapshot.stocks.copy()
    if stocks.empty:
        return pd.DataFrame()
    stocks["name"] = stocks["name"].astype(str).str.strip()

    leader_tags: dict[str, set[str]] = {}
    for name, tags in _sector_leader_tags(
        snapshot.industries, "热门行业", sector_leaders_per_type
    ).items():
        leader_tags.setdefault(name, set()).update(tags)
    for name, tags in _sector_leader_tags(
        snapshot.concepts, "热门概念", sector_leaders_per_type
    ).items():
        leader_tags.setdefault(name, set()).update(tags)

    normalized_always: dict[str, str] = {}
    for value, label in (always_candidates or {}).items():
        try:
            raw = cfg.normalize_ticker(str(value))[2]
            normalized_always[raw] = str(label)
        except ValueError:
            continue

    rank = pd.to_numeric(stocks.get("rank"), errors="coerce")
    codes = stocks["code"].astype(str).str.zfill(6)
    leader_mask = stocks["name"].isin(leader_tags)
    global_mask = rank.le(global_top)
    always_mask = codes.isin(normalized_always)
    candidates = stocks[leader_mask | global_mask | always_mask].copy()
    if candidates.empty:
        return candidates

    def source_text(row: pd.Series) -> str:
        sources = set(leader_tags.get(str(row["name"]), set()))
        if pd.notna(row.get("rank")) and float(row["rank"]) <= global_top:
            sources.add("全市资金榜")
        raw_code = str(row.get("code") or "").zfill(6)
        if raw_code in normalized_always:
            sources.add(normalized_always[raw_code])
        return " · ".join(sorted(sources))

    candidates["discovery_source"] = candidates.apply(source_text, axis=1)
    candidates["sector_mentions"] = candidates["name"].map(
        lambda name: len(leader_tags.get(str(name), set()))
    )
    candidates["always_included"] = candidates["code"].astype(str).str.zfill(6).isin(
        normalized_always
    )
    candidates = candidates.sort_values(
        ["always_included", "sector_mentions", "flow_score"],
        ascending=[False, False, False],
        na_position="last",
    )
    candidates = candidates.drop_duplicates(subset=["code"]).head(max_candidates)
    return candidates.reset_index(drop=True)


HistoryLoader = Callable[[str, int], tuple[pd.DataFrame | None, str | None, tuple[str, ...]]]
SignalEvaluator = Callable[[pd.DataFrame], object]
ResearchEvaluator = Callable[[pd.DataFrame], ResearchEvidence | dict]


class OpportunityScanner:
    """7 策略只触发候选，样本外研究证据决定能否进入推荐区。"""

    def __init__(
        self,
        store: OpportunityStore | None = None,
        history_loader: HistoryLoader | None = None,
        signal_evaluator: SignalEvaluator | None = None,
        research_evaluator: ResearchEvaluator | None = None,
        max_candidates: int = 50,
        max_workers: int = 4,
        always_candidates: dict[str, str] | None = None,
    ):
        self.store = store or OpportunityStore()
        self.history_loader = history_loader or (
            lambda ticker, days: fetch_history("eastmoney", ticker, days)
        )
        self.signal_evaluator = signal_evaluator or (
            lambda frame: CompositeStrategy().evaluate(frame.tail(365))
        )
        self.research_evaluator = research_evaluator or QuantResearchEngine().evaluate
        self.max_candidates = max(1, max_candidates)
        self.max_workers = max(1, min(max_workers, 8))
        self.always_candidates = always_candidates

    @staticmethod
    def _risk_name(name: str) -> bool:
        normalized = name.upper().replace(" ", "")
        return "ST" in normalized or "退" in normalized

    @staticmethod
    def _safe_number(value, default: float = 0.0) -> float:
        try:
            number = float(value)
            return number if np.isfinite(number) else default
        except (TypeError, ValueError):
            return default

    @classmethod
    def _trend_quality(cls, history: pd.DataFrame) -> tuple[float, dict[str, bool]]:
        close = history["close"].astype(float)
        ma20 = history["MA20"].astype(float) if "MA20" in history else close.rolling(20).mean()
        ma60 = history["MA60"].astype(float) if "MA60" in history else close.rolling(60).mean()
        if "ATR" in history:
            atr = cls._safe_number(history["ATR"].iloc[-1], close.iloc[-1] * 0.02)
        else:
            true_range = pd.concat(
                [
                    history["high"] - history["low"],
                    (history["high"] - close.shift(1)).abs(),
                    (history["low"] - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = cls._safe_number(
                true_range.ewm(span=14, adjust=False).mean().iloc[-1],
                close.iloc[-1] * 0.02,
            )
        latest = float(close.iloc[-1])
        latest_ma20 = cls._safe_number(ma20.iloc[-1], latest)
        latest_ma60 = cls._safe_number(ma60.iloc[-1], latest)
        previous_ma20 = (
            cls._safe_number(ma20.iloc[-6], latest_ma20)
            if len(ma20) >= 6
            else latest_ma20
        )
        checks = {
            "price_above_ma20": latest > latest_ma20,
            "price_above_ma60": latest > latest_ma60,
            "ma20_above_ma60": latest_ma20 > latest_ma60,
            "ma20_rising": latest_ma20 > previous_ma20,
            "not_extended": latest <= latest_ma20 + 2.5 * max(atr, 0.01),
        }
        weights = {
            "price_above_ma20": 25,
            "price_above_ma60": 25,
            "ma20_above_ma60": 20,
            "ma20_rising": 15,
            "not_extended": 15,
        }
        return float(sum(weights[key] for key, passed in checks.items() if passed)), checks

    @staticmethod
    def _default_always_candidates() -> dict[str, str]:
        candidates = {ticker: "平台预设股" for ticker in cfg.PRESET_STOCKS}
        for ticker in load_user_stocks(cfg.normalize_ticker):
            candidates[ticker] = "用户自选股"
        return candidates

    def _evaluate_one(self, row: dict, data_date: str) -> dict:
        result = dict(row)
        code = str(row.get("code") or "").zfill(6)
        name = str(row.get("name") or code).strip()
        result.update(
            {
                "code": code,
                "name": name,
                "is_recommended": False,
                "evaluation_status": "行情不可用",
                "rejection_reasons": "",
            }
        )
        try:
            ticker = cfg.normalize_ticker(code)[0]
        except ValueError as exc:
            result["rejection_reasons"] = str(exc)
            return result

        # 研究层需要足够长的样本外区间；原 7 策略仍只读取最近 365 根。
        history, actual_source, failures = self.history_loader(ticker, 800)
        result["market_source"] = actual_source or ""
        result["market_failures"] = " | ".join(failures or ())
        if history is not None and not history.empty:
            # 回看旧资金快照时严禁引入快照之后的 K 线，避免未来函数。
            expected_date = pd.Timestamp(data_date).date()
            index_dates = pd.to_datetime(history.index, errors="coerce")
            history = history[index_dates.date <= expected_date].copy()
        if history is None or history.empty or len(history) < 60:
            result["rejection_reasons"] = "历史 K 线不足 60 根，无法跑完原策略"
            return result

        signal = self.signal_evaluator(history)
        try:
            research = self.research_evaluator(history)
            research_record = (
                research.to_record() if hasattr(research, "to_record") else dict(research)
            )
        except Exception as exc:  # noqa: BLE001 - quarantine one-symbol research failures
            research_record = ResearchEvidence(
                status="研究失败",
                gate_pass=False,
                current_signal="不可评估",
                reasons=(f"{type(exc).__name__}: {exc}",),
            ).to_record()
        latest_date = pd.Timestamp(history.index[-1]).date()
        expected_date = pd.Timestamp(data_date).date()
        freshness_lag = max(0, (expected_date - latest_date).days)
        composite_score = float(getattr(signal, "score", 0) or 0)
        confidence = float(getattr(signal, "confidence", 0) or 0)
        # 原策略 -10~+10 线性映射到 0~100；资金只占 15%，不再一票否决。
        technical_score = max(0.0, min(100.0, (composite_score + 10.0) * 5.0))
        flow_score = self._safe_number(row.get("flow_score"))
        trend_quality, trend_checks = self._trend_quality(history)
        opportunity_score = round(
            technical_score * 0.65
            + flow_score * 0.15
            + confidence * 100 * 0.10
            + trend_quality * 0.10,
            1,
        )
        latest_close = float(history["close"].iloc[-1])
        stop_loss = self._safe_number(getattr(signal, "stop_loss", 0))
        take_profit = self._safe_number(getattr(signal, "take_profit", 0))
        price_risk = risk_reward_metrics(latest_close, stop_loss, take_profit)
        result.update(
            {
                "evaluation_status": "已完成",
                "latest_bar_date": latest_date.isoformat(),
                "freshness_lag_days": freshness_lag,
                "latest_close": round(latest_close, 3),
                "strategy_direction": str(getattr(signal, "direction", "neutral")),
                "strategy_score": round(composite_score, 1),
                "strategy_confidence": round(confidence, 2),
                "buy_votes": int(getattr(signal, "buy_votes", 0) or 0),
                "sell_votes": int(getattr(signal, "sell_votes", 0) or 0),
                "recommended_position": float(
                    getattr(signal, "recommended_position", 0) or 0
                ),
                "stop_loss": round(stop_loss, 3),
                "take_profit": round(take_profit, 3),
                "reward_risk": round(price_risk["reward_risk_now"], 2),
                "stop_distance_pct": round(price_risk["stop_distance_pct"], 4),
                "trend_quality": round(trend_quality, 1),
                "trend_checks_passed": sum(trend_checks.values()),
                "opportunity_score": opportunity_score,
                "selection_path": "未通过",
                **research_record,
            }
        )

        common_reasons: list[str] = []
        if self._risk_name(name):
            common_reasons.append("ST/退市风险标记")
        positive_periods = int(row.get("positive_periods") or 0)
        today_net = self._safe_number(row.get("main_net_today"))
        capital_signal = str(row.get("capital_signal") or "数据不足")
        direction = result["strategy_direction"]
        change_pct = self._safe_number(row.get("change_pct_today"))
        board_limit = price_limit_pct(ticker) * 100
        max_daily_change = board_limit * 0.70
        min_daily_change = -board_limit * 0.50
        if change_pct > max_daily_change:
            common_reasons.append(
                f"当日涨幅 {change_pct:.2f}% 接近涨停（过热线 {max_daily_change:.1f}%）"
            )
        if change_pct < min_daily_change:
            common_reasons.append(
                f"当日涨幅 {change_pct:.2f}% 过弱（风险线 {min_daily_change:.1f}%）"
            )
        if freshness_lag > MAX_FRESHNESS_LAG_DAYS:
            common_reasons.append(f"日 K 滞后资金数据 {freshness_lag} 天")
        if stop_loss <= 0 or stop_loss >= latest_close:
            common_reasons.append(
                f"保护止损 {stop_loss:.3f} 无效或已被现价 {latest_close:.3f} 跌破"
            )
        if take_profit <= latest_close:
            common_reasons.append("策略止盈目标已到或无效")
        if stop_loss > 0 and stop_loss < latest_close and take_profit > latest_close:
            if price_risk["reward_risk_now"] < MIN_CURRENT_REWARD_RISK:
                common_reasons.append(
                    f"现价盈亏比 {price_risk['reward_risk_now']:.2f}R<"
                    f"{MIN_CURRENT_REWARD_RISK:.2f}R"
                )
            if price_risk["stop_distance_pct"] > MAX_STOP_DISTANCE_PCT:
                common_reasons.append(
                    f"止损距离 {price_risk['stop_distance_pct']:.1%}>"
                    f"{MAX_STOP_DISTANCE_PCT:.0%}"
                )
        if not bool(result.get("research_gate_pass")):
            research_reason = str(result.get("research_reasons") or "样本外证据未通过")
            common_reasons.append(f"量化研究：{research_reason}")

        fund_path_gaps: list[str] = []
        if opportunity_score < MIN_OPPORTUNITY_SCORE:
            fund_path_gaps.append(
                f"前置机会分 {opportunity_score:.1f}<{MIN_OPPORTUNITY_SCORE:.0f}"
            )
        if direction != "strong_buy" or composite_score < MIN_COMPOSITE_SCORE:
            fund_path_gaps.append(f"策略 {composite_score:+.1f}<+{MIN_COMPOSITE_SCORE:.0f}")
        if result["buy_votes"] < MIN_BUY_VOTES:
            fund_path_gaps.append(f"买票 {result['buy_votes']}<{MIN_BUY_VOTES}")
        if confidence < MIN_CONFIDENCE:
            fund_path_gaps.append(f"信心 {confidence:.0%}<{MIN_CONFIDENCE:.0%}")
        if flow_score < MIN_FLOW_SCORE:
            fund_path_gaps.append(f"资金 {flow_score:.1f}<{MIN_FLOW_SCORE:.0f}")
        if positive_periods < MIN_POSITIVE_PERIODS:
            fund_path_gaps.append(f"流入周期 {positive_periods}/3")
        if today_net <= 0:
            fund_path_gaps.append("当日未净流入")
        if trend_quality < MIN_FUND_TREND_QUALITY:
            fund_path_gaps.append(
                f"趋势质量 {trend_quality:.0f}<{MIN_FUND_TREND_QUALITY:.0f}"
            )

        technical_path_gaps: list[str] = []
        if opportunity_score < MIN_OPPORTUNITY_SCORE:
            technical_path_gaps.append(
                f"前置机会分 {opportunity_score:.1f}<{MIN_OPPORTUNITY_SCORE:.0f}"
            )
        if direction != "strong_buy" or composite_score < MIN_TECHNICAL_PATH_SCORE:
            technical_path_gaps.append(
                f"策略 {composite_score:+.1f}<+{MIN_TECHNICAL_PATH_SCORE:.1f}"
            )
        if result["buy_votes"] < MIN_TECHNICAL_BUY_VOTES:
            technical_path_gaps.append(
                f"买票 {result['buy_votes']}<{MIN_TECHNICAL_BUY_VOTES}"
            )
        if result["sell_votes"] > 1:
            technical_path_gaps.append(f"卖票 {result['sell_votes']}>1")
        if confidence < MIN_TECHNICAL_CONFIDENCE:
            technical_path_gaps.append(
                f"信心 {confidence:.0%}<{MIN_TECHNICAL_CONFIDENCE:.0%}"
            )
        if trend_quality < MIN_TECHNICAL_TREND_QUALITY:
            technical_path_gaps.append(
                f"趋势质量 {trend_quality:.0f}<{MIN_TECHNICAL_TREND_QUALITY:.0f}"
            )
        if capital_signal == "连续流出":
            technical_path_gaps.append("资金连续流出")

        research_path_gaps: list[str] = []
        if direction not in ("buy", "strong_buy"):
            research_path_gaps.append(f"策略方向 {direction} 不是买入")
        if composite_score < MIN_RESEARCH_TRIGGER_SCORE:
            research_path_gaps.append(
                f"策略触发分 {composite_score:+.1f}<+{MIN_RESEARCH_TRIGGER_SCORE:.0f}"
            )
        if result["buy_votes"] < MIN_BUY_VOTES:
            research_path_gaps.append(f"买票 {result['buy_votes']}<{MIN_BUY_VOTES}")
        if result["sell_votes"] > 1:
            research_path_gaps.append(f"卖票 {result['sell_votes']}>1")
        if confidence < MIN_TECHNICAL_CONFIDENCE:
            research_path_gaps.append(
                f"信心 {confidence:.0%}<{MIN_TECHNICAL_CONFIDENCE:.0%}"
            )
        if trend_quality < MIN_RESEARCH_TREND_QUALITY:
            research_path_gaps.append(
                f"趋势质量 {trend_quality:.0f}<{MIN_RESEARCH_TREND_QUALITY:.0f}"
            )
        if capital_signal == "连续流出":
            research_path_gaps.append("资金连续流出")

        fund_path = not fund_path_gaps
        technical_path = not technical_path_gaps
        research_path = not research_path_gaps
        if not common_reasons and (fund_path or technical_path or research_path):
            result["is_recommended"] = True
            if fund_path:
                selection_path = "资金·策略共振"
            elif technical_path:
                selection_path = "技术先行·资金未连续流出"
            else:
                selection_path = "量化验证主导·资金未恶化"
            result["selection_path"] = selection_path + "·样本外验证"
            result["rejection_reasons"] = ""
        else:
            path_summary = []
            if not fund_path:
                path_summary.append("资金共振：" + "、".join(fund_path_gaps))
            if not technical_path:
                path_summary.append("技术先行：" + "、".join(technical_path_gaps))
            if not research_path:
                path_summary.append("量化主导：" + "、".join(research_path_gaps))
            result["rejection_reasons"] = "；".join([*common_reasons, *path_summary])
        return result

    def run(
        self,
        snapshot: ScanSnapshot,
        progress: ProgressCallback | None = None,
    ) -> OpportunitySnapshot:
        always_candidates = (
            self._default_always_candidates()
            if self.always_candidates is None
            else self.always_candidates
        )
        candidates = build_candidate_pool(
            snapshot,
            max_candidates=self.max_candidates,
            always_candidates=always_candidates,
        )
        rows = candidates.to_dict(orient="records")
        total = len(rows)
        evaluated: list[dict] = []
        if progress:
            progress("准备批量策略评分", 0, max(total, 1))
        if rows:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, total),
                thread_name_prefix="opportunity-score",
            ) as executor:
                futures = {
                    executor.submit(
                        self._evaluate_one, row, str(snapshot.metadata.get("data_date"))
                    ): row
                    for row in rows
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    evaluated.append(future.result())
                    if progress:
                        progress(
                            f"策略评分 · {futures[future].get('name', '')}",
                            completed,
                            total,
                        )

        frame = pd.DataFrame(evaluated)
        if not frame.empty:
            frame = frame.sort_values(
                [
                    "is_recommended",
                    "research_probability",
                    "research_quality",
                    "opportunity_score",
                    "flow_score",
                ],
                ascending=[False, False, False, False, False],
                na_position="last",
            ).reset_index(drop=True)
            frame.insert(0, "opportunity_rank", range(1, len(frame) + 1))
        errors = tuple(
            f"{row.get('code')} {row.get('name')}: {row.get('rejection_reasons')}"
            for row in evaluated
            if row.get("evaluation_status") != "已完成"
        )
        now = datetime.now(SHANGHAI_TZ)
        result = OpportunitySnapshot(
            metadata={
                "snapshot_date": snapshot.snapshot_date,
                "data_date": snapshot.metadata.get("data_date"),
                "scored_at": now.isoformat(timespec="seconds"),
                "candidate_count": len(frame),
                "evaluated_count": int(
                    frame.get("evaluation_status", pd.Series(dtype=str)).eq("已完成").sum()
                ),
                "recommended_count": int(
                    frame.get("is_recommended", pd.Series(dtype=bool)).fillna(False).sum()
                ),
                "strategy": "7 策略触发 + 防泄漏样本外量化证据",
                "candidate_sources": ["全市资金榜", "热门板块领涨股", "平台预设股", "用户自选股"],
                "thresholds": {
                    "flow_score": MIN_FLOW_SCORE,
                    "composite_score": MIN_COMPOSITE_SCORE,
                    "technical_path_score": MIN_TECHNICAL_PATH_SCORE,
                    "research_trigger_score": MIN_RESEARCH_TRIGGER_SCORE,
                    "confidence": MIN_CONFIDENCE,
                    "buy_votes": MIN_BUY_VOTES,
                    "opportunity_score": MIN_OPPORTUNITY_SCORE,
                    "research_probability": 0.55,
                    "research_quality": 50.0,
                },
                "quant_research": {
                    "model": "L2 正则逻辑回归",
                    "features": "仅 OHLCV 派生的动量、趋势、波动率和量价特征",
                    "target": "未来 10 个交易日扣除双边费用与 5bp/边滑点后收益为正",
                    "validation": "扩展窗口样本外验证，训练与测试间隔离 10 个交易日",
                    "fund_flow_model": "暂缓：历史资金流快照不足",
                    "round_trip_cost_bps": round(
                        QuantResearchEngine().round_trip_cost * 10_000, 1
                    ),
                    "eligible_count": int(
                        frame.get("research_status", pd.Series(dtype=str))
                        .isin(["样本外通过", "样本外未通过"])
                        .sum()
                    ),
                    "validated_count": int(
                        frame.get("research_status", pd.Series(dtype=str))
                        .eq("样本外通过")
                        .sum()
                    ),
                    "current_gate_count": int(
                        frame.get("research_gate_pass", pd.Series(dtype=bool))
                        .fillna(False)
                        .sum()
                    ),
                },
            },
            evaluated=frame,
            errors=errors,
        )
        self.store.save(result)
        return result
