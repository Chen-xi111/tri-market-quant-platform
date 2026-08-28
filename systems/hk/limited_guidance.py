"""短历史行情的降级观察。

完整的七策略投票至少需要 60 根日 K。新股等标的达不到这个门槛时，
这个模块只给出可验证的短周期状态，不生成仓位或交易指令。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import pandas as pd


MIN_OBSERVATION_BARS = 20
FULL_STRATEGY_BARS = 60


@dataclass(frozen=True)
class LimitedGuidance:
    status: str
    label: str
    score: int
    confidence: float
    available_bars: int
    required_bars: int = FULL_STRATEGY_BARS
    price: float = 0.0
    stop_reference: float | None = None
    target_reference: float | None = None
    rsi: float | None = None
    macd_hist: float | None = None
    reasons: list[str] = field(default_factory=list)
    actionable: bool = False


def _finite(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def build_limited_guidance(frame: pd.DataFrame) -> LimitedGuidance:
    """基于现有指标构建不可下单的短历史观察。"""
    bars = 0 if frame is None else len(frame)
    if frame is None or frame.empty:
        return LimitedGuidance("insufficient", "无法观察", 50, 0.0, bars)

    price = float(frame["close"].iloc[-1])
    if bars < MIN_OBSERVATION_BARS:
        return LimitedGuidance(
            "insufficient",
            "样本不足",
            50,
            0.0,
            bars,
            price=price,
            reasons=[f"至少需要 {MIN_OBSERVATION_BARS} 根日 K 才能计算短周期状态"],
        )

    last = frame.iloc[-1]
    evidence: list[tuple[int, str]] = []
    ma5 = _finite(last.get("MA5"))
    ma20 = _finite(last.get("MA20"))
    rsi = _finite(last.get("RSI"))
    macd_hist = _finite(last.get("MACD_HIST"))

    if ma5 is not None:
        evidence.append((1 if price > ma5 else -1, f"现价{'高于' if price > ma5 else '低于'} MA5"))
    if ma5 is not None and ma20 is not None:
        evidence.append((1 if ma5 > ma20 else -1, f"MA5 {'高于' if ma5 > ma20 else '低于'} MA20"))
    if rsi is not None:
        rsi_vote = 1 if rsi >= 55 else (-1 if rsi <= 45 else 0)
        evidence.append((rsi_vote, f"RSI(14) {rsi:.1f}"))
    if macd_hist is not None:
        evidence.append((1 if macd_hist > 0 else -1, f"MACD 柱 {'为正' if macd_hist > 0 else '为负'}"))

    ten_day_return = float(frame["close"].pct_change(10).iloc[-1])
    if math.isfinite(ten_day_return):
        evidence.append(
            (1 if ten_day_return > 0 else -1, f"10 日涨跌 {ten_day_return:+.1%}")
        )

    raw_score = sum(vote for vote, _ in evidence)
    max_score = max(1, len(evidence))
    score = round(50 + 50 * raw_score / max_score)
    if score >= 65:
        label = "偏强观察"
    elif score <= 35:
        label = "偏弱观察"
    else:
        label = "中性观察"

    # 短历史结论的信心度永不超过 45%，避免和完整策略混淆。
    completeness = min(1.0, bars / FULL_STRATEGY_BARS)
    evidence_ratio = min(1.0, len(evidence) / 5)
    confidence = min(0.45, 0.45 * completeness * evidence_ratio)

    atr = _finite(last.get("ATR"))
    if atr is None or atr <= 0:
        recent_range = (frame["high"] - frame["low"]).tail(14).median()
        atr = _finite(recent_range) or price * 0.03

    return LimitedGuidance(
        "limited",
        label,
        max(0, min(100, score)),
        confidence,
        bars,
        price=price,
        stop_reference=max(0.001, price - 2 * atr),
        target_reference=price + 2 * atr,
        rsi=rsi,
        macd_hist=macd_hist,
        reasons=[reason for _, reason in evidence],
    )
