"""面向 A 股隔日与当日决策的可执行策略。

所有历史特征只使用当前及更早的 K 线。信号在收盘后形成，历史回测在
下一交易日开盘尝试成交，从而避免把当天收盘价当作可提前成交的价格。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


StrategyMode = Literal["overnight", "today"]


@dataclass(frozen=True)
class StrategyProfile:
    mode: StrategyMode
    label: str
    score_threshold: float
    max_hold_bars: int
    stop_atr: float
    first_target_r: float
    second_target_r: float
    cooldown_bars: int


PROFILES: dict[StrategyMode, StrategyProfile] = {
    "overnight": StrategyProfile(
        mode="overnight",
        label="隔日波段（2–5 个交易日）",
        score_threshold=64.0,
        max_hold_bars=5,
        stop_atr=1.35,
        first_target_r=1.6,
        second_target_r=2.4,
        cooldown_bars=3,
    ),
    "today": StrategyProfile(
        mode="today",
        label="当日决策（持仓卖出 / 新仓隔夜）",
        score_threshold=70.0,
        max_hold_bars=2,
        stop_atr=1.05,
        first_target_r=1.35,
        second_target_r=2.0,
        cooldown_bars=2,
    ),
}


@dataclass(frozen=True)
class ActionPlan:
    mode: StrategyMode
    action: str
    action_level: str
    score: float
    setup_valid: bool
    entry_low: float
    entry_high: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward_1: float
    risk_reward_2: float
    buy_lot_size: int
    suggested_quantity: int
    sell_quantity: int
    reason: str
    invalidation: str
    data_date: pd.Timestamp


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _validate_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        raise ValueError("没有可用于策略计算的行情数据")
    missing = [column for column in REQUIRED_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f"行情缺少字段：{', '.join(missing)}")
    frame = history.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if len(frame) < 80:
        raise ValueError("至少需要 80 根日 K 线才能生成短周期策略")
    return frame


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + relative_strength)
    return result.fillna(50.0)


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def prepare_features(history: pd.DataFrame, mode: StrategyMode = "overnight") -> pd.DataFrame:
    """生成完全因果的特征、评分和收盘后信号。"""
    profile = PROFILES[mode]
    frame = _validate_history(history)
    close = frame["close"]
    previous_close = close.shift(1)

    frame["ema5"] = close.ewm(span=5, adjust=False).mean()
    frame["ema10"] = close.ewm(span=10, adjust=False).mean()
    frame["ema20"] = close.ewm(span=20, adjust=False).mean()
    frame["ema60"] = close.ewm(span=60, adjust=False).mean()
    frame["atr14"] = _atr(frame)
    frame["rsi14"] = _rsi(close)
    frame["return_1d"] = close.pct_change()
    frame["return_3d"] = close.pct_change(3)
    frame["volatility20"] = close.pct_change().rolling(20).std()
    frame["volume_ratio"] = frame["volume"] / frame["volume"].rolling(20).mean()
    frame["prior_high_5"] = frame["high"].shift(1).rolling(5).max()
    frame["prior_low_5"] = frame["low"].shift(1).rolling(5).min()
    frame["prior_low_10"] = frame["low"].shift(1).rolling(10).min()
    intraday_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["close_location"] = ((close - frame["low"]) / intraday_range).fillna(0.5)

    trend_checks = pd.concat(
        [
            (close > frame["ema20"]).astype(float),
            (frame["ema20"] > frame["ema60"]).astype(float),
            (frame["ema20"] > frame["ema20"].shift(5)).astype(float),
        ],
        axis=1,
    )
    frame["trend_component"] = trend_checks.mean(axis=1)

    touched_fast_support = frame["low"] <= frame["ema10"] * 1.01
    closed_above_fast = close >= frame["ema10"] * 0.995
    not_extended = close <= frame["ema20"] + frame["atr14"] * 2.2
    pullback_rebound = touched_fast_support & closed_above_fast & not_extended
    shallow_breakout = (close > frame["prior_high_5"]) & (
        close <= frame["ema20"] + frame["atr14"] * 2.6
    )
    frame["setup_component"] = np.select(
        [pullback_rebound, shallow_breakout], [1.0, 0.85], default=0.2
    )

    momentum = (
        (frame["return_3d"] > -0.025)
        & (frame["return_3d"] < 0.09)
        & frame["rsi14"].between(45, 72)
        & (close >= previous_close * 0.99)
    )
    frame["momentum_component"] = momentum.astype(float)

    healthy_volume = frame["volume_ratio"].between(0.75, 2.8)
    volume_confirmed = healthy_volume & (
        (frame["return_1d"] >= 0) | (frame["volume_ratio"] <= 1.15)
    )
    frame["volume_component"] = volume_confirmed.astype(float)

    quality = (frame["close_location"] >= 0.52) & (
        frame["volatility20"] <= frame["volatility20"].rolling(120, min_periods=20).quantile(0.8)
    )
    frame["quality_component"] = quality.astype(float)

    frame["score"] = (
        30 * frame["trend_component"]
        + 25 * frame["setup_component"]
        + 20 * frame["momentum_component"]
        + 15 * frame["volume_component"]
        + 10 * frame["quality_component"]
    )
    overbought_penalty = (frame["rsi14"] > 76) | (close > frame["ema20"] + frame["atr14"] * 3)
    weak_close_penalty = frame["close_location"] < 0.25
    frame.loc[overbought_penalty, "score"] -= 20
    frame.loc[weak_close_penalty, "score"] -= 10
    frame["score"] = frame["score"].clip(0, 100)

    frame["entry_low"] = np.maximum(frame["ema5"], close - 0.28 * frame["atr14"])
    frame["entry_high"] = close + (0.12 if mode == "today" else 0.18) * frame["atr14"]
    volatility_stop = close - profile.stop_atr * frame["atr14"]
    technical_stop = np.minimum(frame["ema20"] * 0.99, frame["prior_low_5"] * 0.995)
    frame["stop_loss"] = np.maximum(volatility_stop, technical_stop)
    invalid_stop = frame["stop_loss"] >= close * 0.995
    frame.loc[invalid_stop, "stop_loss"] = volatility_stop[invalid_stop]
    risk = (close - frame["stop_loss"]).clip(lower=0.01)
    frame["target_1"] = close + profile.first_target_r * risk
    frame["target_2"] = close + profile.second_target_r * risk

    raw_signal = (
        (frame["score"] >= profile.score_threshold)
        & (frame["trend_component"] >= 2 / 3)
        & (frame["entry_low"] < frame["entry_high"])
        & frame["atr14"].notna()
    )
    # 只在评分首次突破阈值或冷却期结束后触发，避免每天重复发出同一买点。
    signal = pd.Series(False, index=frame.index)
    last_signal_position = -10_000
    for position, is_candidate in enumerate(raw_signal.fillna(False).to_numpy()):
        crossed = position == 0 or not bool(raw_signal.iloc[position - 1])
        cooled = position - last_signal_position > profile.cooldown_bars
        if is_candidate and (crossed or cooled):
            signal.iloc[position] = True
            last_signal_position = position
    frame["entry_signal"] = signal
    frame["signal_reason"] = ""
    frame.loc[signal, "signal_reason"] = (
        "趋势"
        + (frame.loc[signal, "trend_component"] * 100).round().astype(int).astype(str)
        + "%｜评分"
        + frame.loc[signal, "score"].round(1).astype(str)
    )
    return frame


def _round_price(value: float) -> float:
    return round(max(0.01, float(value)) + 1e-12, 2)


def _position_size(
    capital: float,
    entry_price: float,
    stop_loss: float,
    risk_pct: float,
    max_position_pct: float,
    lot_size: int,
) -> int:
    risk_per_share = max(0.01, entry_price - stop_loss)
    risk_limited = capital * risk_pct / risk_per_share
    value_limited = capital * max_position_pct / max(entry_price, 0.01)
    return max(0, int(min(risk_limited, value_limited) // lot_size) * lot_size)


def buy_lot_size(ticker: str | None) -> int:
    """返回保守的买入数量单位；科创板按 200 股，其余按 100 股。"""
    if not ticker:
        return 100
    digits = "".join(character for character in str(ticker) if character.isdigit())
    raw = digits[-6:]
    return 200 if raw.startswith(("688", "689")) else 100


def build_action_plan(
    features: pd.DataFrame,
    *,
    mode: StrategyMode,
    current_price: float,
    capital: float,
    risk_pct: float,
    max_position_pct: float,
    ticker: str | None = None,
    holding_quantity: int = 0,
    sellable_quantity: int = 0,
    average_cost: float = 0.0,
) -> ActionPlan:
    """把最新策略状态转换为可检查的买卖计划。"""
    if features is None or features.empty:
        raise ValueError("没有策略特征，无法生成计划")
    profile = PROFILES[mode]
    lot_size = buy_lot_size(ticker)
    latest = features.iloc[-1]
    current_price = float(current_price or latest["close"])
    entry_low = _round_price(latest["entry_low"])
    entry_high = _round_price(latest["entry_high"])
    stop_loss = _round_price(latest["stop_loss"])
    target_1 = _round_price(latest["target_1"])
    target_2 = _round_price(latest["target_2"])
    quantity = _position_size(
        capital,
        (entry_low + entry_high) / 2,
        stop_loss,
        risk_pct,
        max_position_pct,
        lot_size,
    )
    score = float(latest["score"])
    setup_valid = bool(
        score >= profile.score_threshold
        and latest["trend_component"] >= 2 / 3
        and stop_loss < current_price
    )

    if holding_quantity > 0:
        sellable = max(0, min(int(sellable_quantity), int(holding_quantity)))
        cost_based_stop = (
            average_cost - 1.1 * float(latest["atr14"]) if average_cost > 0 else stop_loss
        )
        stop_loss = _round_price(max(stop_loss, cost_based_stop))
        cost_risk = max(0.01, (average_cost or current_price) - stop_loss)
        target_1 = _round_price(
            max(target_1, (average_cost or current_price) + profile.first_target_r * cost_risk)
        )
        target_2 = _round_price(
            max(target_2, (average_cost or current_price) + profile.second_target_r * cost_risk)
        )
        if sellable <= 0:
            action, level = "今日不可卖", "warning"
            reason = "持仓均为今日买入，A 股 T+1 下需下一交易日才能卖出。"
            sell_quantity = 0
        elif current_price <= stop_loss:
            action, level = "执行止损", "sell"
            reason = "现价已触及止损线；优先卖出全部可卖数量。"
            sell_quantity = sellable
        elif current_price >= target_2:
            action, level = "止盈清仓", "sell"
            reason = "现价达到第二止盈位；卖出剩余可卖仓位。"
            sell_quantity = sellable
        elif current_price >= target_1:
            action, level = "分批止盈", "reduce"
            reason = "现价达到第一止盈位；建议卖出约一半可卖仓位。"
            sell_quantity = max(100, (sellable // 200) * 100) if sellable >= 100 else sellable
        elif score < 48 or current_price < float(latest["ema20"]):
            action, level = "减仓防守", "reduce"
            reason = "评分转弱或跌破 EMA20；建议降低一半可卖仓位。"
            sell_quantity = max(100, (sellable // 200) * 100) if sellable >= 100 else sellable
        else:
            action, level = "持有并设条件单", "hold"
            reason = "趋势尚未失效；持有，但必须同时监控止损和两档止盈。"
            sell_quantity = 0
    else:
        sell_quantity = 0
        if not setup_valid:
            action, level = "观望", "hold"
            reason = (
                f"当前评分 {score:.1f}，未达到 {profile.score_threshold:.0f} 分或趋势过滤未通过。"
            )
        elif quantity < 100:
            action, level = "资金/风险不足", "warning"
            reason = "按当前资金和止损距离计算不足 100 股，不建议强行放大风险。"
        elif entry_low <= current_price <= entry_high:
            action, level = "进入买入区", "buy"
            reason = "价格位于计划买入区，可分两笔执行首笔 50%，买入后今日不可卖。"
        elif current_price > entry_high:
            action, level = "不追高", "hold"
            reason = "现价高于计划买入上沿，等待回落或下一次新信号。"
        else:
            action, level = "等待确认", "hold"
            reason = "现价低于买入区，等待重新站回买入下沿，不做下跌接刀。"

    entry_mid = (entry_low + entry_high) / 2
    plan_risk = max(0.01, entry_mid - stop_loss)
    return ActionPlan(
        mode=mode,
        action=action,
        action_level=level,
        score=round(score, 1),
        setup_valid=setup_valid,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        risk_reward_1=round((target_1 - entry_mid) / plan_risk, 2),
        risk_reward_2=round((target_2 - entry_mid) / plan_risk, 2),
        buy_lot_size=lot_size,
        suggested_quantity=quantity,
        sell_quantity=sell_quantity,
        reason=reason,
        invalidation=f"收盘跌破 ¥{stop_loss:.2f}，或趋势评分低于 48 分，计划失效。",
        data_date=pd.Timestamp(features.index[-1]),
    )


def score_breakdown(features: pd.DataFrame) -> pd.DataFrame:
    latest = features.iloc[-1]
    rows = [
        ("趋势结构", 30, latest["trend_component"]),
        ("回调/突破形态", 25, latest["setup_component"]),
        ("短线动量", 20, latest["momentum_component"]),
        ("量价确认", 15, latest["volume_component"]),
        ("波动与收盘质量", 10, latest["quality_component"]),
    ]
    return pd.DataFrame(
        [
            {
                "因子": name,
                "权重": weight / 100,
                "通过度": float(value),
                "贡献分": round(weight * float(value), 1),
            }
            for name, weight, value in rows
        ]
    )
