"""美股横向优先级与执行准备度。

横截面分数用于缩小研究范围；绝对执行分、硬性风控门槛和策略入场状态
用于判断候选股现在是否值得进入模拟盘。这里不把相对排名包装成买入信号。
"""

from __future__ import annotations

from collections.abc import Mapping
from math import floor, isfinite, log10

import numpy as np
import pandas as pd

MIN_RANKING_BARS = 90
DEFAULT_EXECUTION_PROFILE = "balanced"

STRATEGY_LABELS = {
    "trend_rider": "趋势骑乘",
    "ma_crossover": "双均线趋势",
    "bollinger_reversion": "布林带回归",
    "vol_breakout": "量价突破",
    "multi_tf": "多周期共振",
    "alpha_factors": "Alpha 多因子",
    "ml_predict": "机器学习",
}

EXECUTION_PROFILES = {
    "conservative": {
        "label": "稳健", "min_bars": 160, "min_price": 10.0,
        "min_dollar_volume": 50_000_000.0, "max_atr_pct": 0.06,
        "max_gap_risk": 0.08, "max_data_age_days": 3,
        "position_multiplier": 0.75,
    },
    "balanced": {
        "label": "平衡", "min_bars": 120, "min_price": 5.0,
        "min_dollar_volume": 20_000_000.0, "max_atr_pct": 0.09,
        "max_gap_risk": 0.12, "max_data_age_days": 5,
        "position_multiplier": 1.0,
    },
    "opportunity": {
        "label": "机会", "min_bars": 90, "min_price": 2.0,
        "min_dollar_volume": 5_000_000.0, "max_atr_pct": 0.14,
        "max_gap_risk": 0.18, "max_data_age_days": 7,
        "position_multiplier": 0.65,
    },
}


def _finite(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _last(series: pd.Series, default: float = 0.0) -> float:
    return _finite(series.iloc[-1] if len(series) else default, default)


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(_finite(value), low, high))


def _round_down_lot(shares: float, lot_size: int) -> int:
    step = max(1, int(lot_size))
    return max(0, floor(max(0.0, float(shares)) / step) * step)


def _strategy_fit(strategy_key: str, feature: dict) -> float:
    """用于候选池内相对比较的轻量策略代理，不重复拟合模型。"""
    trend = feature["trend_raw"]
    momentum = feature["momentum_raw"]
    stability = feature["stability_raw"]
    distance_ma20 = feature["distance_ma20"]
    rsi = feature["rsi"]
    boll_pct = feature["boll_pct"]
    breakout = feature["breakout_20"]
    volume_ratio = feature["volume_ratio"]

    if strategy_key == "trend_rider":
        return 0.50 * trend + 0.25 * momentum - 0.25 * abs(distance_ma20 - 0.01)
    if strategy_key == "ma_crossover":
        return 0.65 * trend + 0.35 * momentum
    if strategy_key == "bollinger_reversion":
        return (
            -0.50 * abs(boll_pct - 0.15)
            - 0.30 * abs(rsi - 38.0) / 100
            + 0.20 * max(trend, -0.08)
        )
    if strategy_key == "vol_breakout":
        return 0.50 * breakout + 0.30 * np.log(max(volume_ratio, 0.1)) + 0.20 * momentum
    if strategy_key == "multi_tf":
        agreement_penalty = -abs(feature["return_20"] - feature["return_60"] / 3)
        return 0.45 * trend + 0.45 * momentum + 0.10 * agreement_penalty
    if strategy_key == "alpha_factors":
        return 0.35 * trend + 0.40 * momentum + 0.25 * stability
    return 0.35 * trend + 0.35 * momentum + 0.30 * stability


def extract_stock_features(ticker: str, frame: pd.DataFrame, strategy_key: str) -> dict:
    """从一只股票的日 K 提取可比较、可风控的特征。"""
    if frame is None or len(frame) < MIN_RANKING_BARS:
        raise ValueError(f"日 K 不足 {MIN_RANKING_BARS} 根")
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"缺少字段: {', '.join(sorted(missing))}")

    data = frame.sort_index().copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    if len(data) < MIN_RANKING_BARS:
        raise ValueError(f"有效日 K 不足 {MIN_RANKING_BARS} 根")

    close, open_, high, low = data["close"], data["open"], data["high"], data["low"]
    volume = data["volume"].fillna(0).clip(lower=0)
    returns = close.pct_change()
    price = _last(close)
    previous_close = _finite(close.iloc[-2], price)
    ma20_series, ma60_series = close.rolling(20).mean(), close.rolling(60).mean()
    ma20 = _finite(data["MA20"].iloc[-1]) if "MA20" in data else _last(ma20_series)
    ma60 = _finite(data["MA60"].iloc[-1]) if "MA60" in data else _last(ma60_series)
    ma20_then = _finite(ma20_series.iloc[-6], ma20)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr_fallback = true_range.ewm(span=14, adjust=False).mean()
    atr = _finite(data["ATR"].iloc[-1], _last(atr_fallback)) if "ATR" in data else _last(atr_fallback)
    atr = max(atr, price * 0.002)

    boll_mid = _finite(data["BOLL_MID"].iloc[-1], ma20) if "BOLL_MID" in data else ma20
    rolling_std = close.rolling(20).std()
    fallback_std = _last(rolling_std)
    boll_lower = _finite(data["BOLL_LOWER"].iloc[-1], boll_mid - 2 * fallback_std) if "BOLL_LOWER" in data else boll_mid - 2 * fallback_std
    boll_upper = _finite(data["BOLL_UPPER"].iloc[-1], boll_mid + 2 * fallback_std) if "BOLL_UPPER" in data else boll_mid + 2 * fallback_std

    return_20 = _finite(close.iloc[-1] / close.iloc[-21] - 1)
    return_60 = _finite(close.iloc[-1] / close.iloc[-61] - 1)
    volatility = _finite(returns.tail(60).std() * np.sqrt(252), 1.0)
    rolling_peak = close.tail(120).cummax()
    max_drawdown = _finite((close.tail(120) / rolling_peak - 1).min(), -1.0)
    avg_dollar_volume = _finite((close * volume).tail(20).median())
    resistance_20 = _finite(high.shift(1).tail(20).max(), price)
    resistance_60 = _finite(high.shift(1).tail(60).max(), resistance_20)
    support_20 = _finite(low.shift(1).tail(20).min(), price - 2 * atr)
    gap_series = (open_ / prev_close - 1).replace([np.inf, -np.inf], np.nan)
    gap_risk = _finite(gap_series.abs().tail(60).quantile(0.95), 1.0)
    latest_gap = _finite(open_.iloc[-1] / previous_close - 1)
    distance_ma20 = price / ma20 - 1 if ma20 else 0.0
    trend_raw = (
        0.45 * (price / ma60 - 1 if ma60 else 0.0)
        + 0.35 * (ma20 / ma60 - 1 if ma60 else 0.0)
        + 0.20 * (ma20 / ma20_then - 1 if ma20_then else 0.0)
    )
    momentum_raw = 0.60 * return_20 + 0.40 * return_60
    stability_raw = -(0.50 * volatility + 0.30 * abs(max_drawdown) + 0.20 * gap_risk)
    rsi = _finite(data["RSI"].iloc[-1], 50.0) if "RSI" in data else 50.0
    boll_pct = _finite(data["BOLL_PCT"].iloc[-1], 0.5) if "BOLL_PCT" in data else 0.5
    volume_ratio = _finite(data["VOL_RATIO"].iloc[-1], 1.0) if "VOL_RATIO" in data else 1.0
    finite_ratio = float(data[["open", "high", "low", "close", "volume"]].tail(120).notna().mean().mean())

    feature = {
        "ticker": ticker, "price": price, "previous_close": previous_close,
        "latest_open": _last(open_), "latest_high": _last(high), "latest_low": _last(low),
        "latest_close_up": bool(close.iloc[-1] > open_.iloc[-1] and close.iloc[-1] > previous_close),
        "bars": len(data), "latest_date": pd.Timestamp(data.index[-1]).tz_localize(None),
        "return_20": return_20, "return_60": return_60,
        "volatility_60": volatility, "max_drawdown_120": max_drawdown,
        "avg_dollar_volume_20": avg_dollar_volume, "gap_risk_60": gap_risk,
        "latest_gap": latest_gap, "atr": atr, "atr_pct": atr / price if price else 1.0,
        "ma20": ma20, "ma60": ma60,
        "ma20_slope_5": ma20 / ma20_then - 1 if ma20_then else 0.0,
        "support_20": support_20, "resistance_20": resistance_20,
        "resistance_60": resistance_60, "boll_mid": boll_mid,
        "boll_lower": boll_lower, "boll_upper": boll_upper,
        "trend_raw": trend_raw, "momentum_raw": momentum_raw,
        "stability_raw": stability_raw, "liquidity_raw": log10(max(avg_dollar_volume, 1.0)),
        "distance_ma20": distance_ma20, "rsi": rsi, "boll_pct": boll_pct,
        "volume_ratio": volume_ratio,
        "breakout_20": price / resistance_20 - 1 if resistance_20 else 0.0,
        "finite_ratio": finite_ratio,
    }
    feature["strategy_raw"] = _strategy_fit(strategy_key, feature)
    return feature


def build_execution_setup(feature: Mapping[str, object], strategy_key: str) -> dict:
    """把技术特征转换为策略特定的等待条件和价格计划。"""
    price = _finite(feature.get("price"))
    atr = max(_finite(feature.get("atr"), price * 0.02), price * 0.002)
    ma20 = _finite(feature.get("ma20"), price)
    ma60 = _finite(feature.get("ma60"), price)
    support = _finite(feature.get("support_20"), price - 2 * atr)
    resistance20 = _finite(feature.get("resistance_20"), price + 2 * atr)
    resistance60 = _finite(feature.get("resistance_60"), resistance20)
    boll_lower = _finite(feature.get("boll_lower"), ma20 - 2 * atr)
    boll_mid = _finite(feature.get("boll_mid"), ma20)
    boll_upper = _finite(feature.get("boll_upper"), ma20 + 2 * atr)
    rsi = _finite(feature.get("rsi"), 50.0)
    volume_ratio = _finite(feature.get("volume_ratio"), 1.0)
    distance_atr = (price - ma20) / atr
    rising_ma = _finite(feature.get("ma20_slope_5")) > 0
    strong_up = price > ma20 > ma60 and rising_ma
    uptrend = price > ma60 and ma20 >= ma60
    downtrend = price < ma20 < ma60
    reversal = bool(feature.get("latest_close_up", False))

    state, score = "watch", 40.0
    entry_low, entry_high = ma20 - 0.35 * atr, ma20 + 0.65 * atr
    trigger = min(_finite(feature.get("latest_high"), price) + 0.05 * atr, entry_high)
    max_chase = ma20 + 1.25 * atr
    condition = "等待回调到 MA20 附近并出现向上确认"
    reason = "趋势尚未同时满足排列、回调和不过热条件"
    structural_target, second_target = resistance20, resistance60

    if strategy_key == "trend_rider":
        score = (
            (50 if strong_up else 35 if uptrend else 5)
            + (30 if -0.35 <= distance_atr <= 0.65 else 18 if -1 <= distance_atr <= 1.25 else 0)
            + (15 if 40 <= rsi <= 65 else 8 if 30 <= rsi < 72 else 0)
            + (5 if volume_ratio >= 0.8 else 0)
        )
        if downtrend or distance_atr > 2.0 or rsi >= 78:
            state, reason = "avoid", "下降结构、过度偏离 MA20 或 RSI 过热，不追价"
        elif strong_up and entry_low <= price <= entry_high and rsi < 68:
            state, reason = "ready", "上升排列成立，价格回到 MA20 入场区"
        else:
            reason = "趋势可跟踪，但需等回调进入计划区"

    elif strategy_key == "ma_crossover":
        entry_low, entry_high = ma20 - 0.25 * atr, ma20 + 0.85 * atr
        trigger = min(_finite(feature.get("latest_high"), price) + 0.05 * atr, entry_high)
        max_chase = ma20 + 1.50 * atr
        condition = "多头排列保持，回踩 MA20 后重新转强"
        score = (
            (55 if strong_up else 38 if uptrend else 5)
            + (30 if -0.25 <= distance_atr <= 0.85 else 15 if distance_atr <= 1.5 else 0)
            + (15 if rsi < 70 else 0)
        )
        if downtrend or distance_atr > 2.2 or rsi >= 80:
            state, reason = "avoid", "均线结构未通过或价格已明显扩张"
        elif uptrend and entry_low <= price <= entry_high and rsi < 72:
            state, reason = "ready", "均线多头状态保持，处于回踩确认区"
        else:
            reason = "多头状态可观察，等待回踩而不是追交叉"

    elif strategy_key == "bollinger_reversion":
        entry_low, entry_high = max(0.01, boll_lower - 0.15 * atr), boll_lower + 0.55 * atr
        trigger = _finite(feature.get("latest_high"), price) + 0.05 * atr
        max_chase = min(boll_mid, entry_high + 0.50 * atr)
        condition = "触及下轨后出现阳线/高点确认，目标先看中轨"
        boll_pct = _finite(feature.get("boll_pct"), 0.5)
        score = (
            (40 if -0.10 <= boll_pct <= 0.25 else 22 if boll_pct <= 0.40 else 0)
            + (25 if 28 <= rsi <= 45 else 12 if 20 <= rsi < 52 else 0)
            + (20 if not downtrend else 0) + (15 if reversal else 5)
        )
        structural_target, second_target = boll_mid, boll_upper
        if downtrend or price > boll_mid or rsi < 18:
            state, reason = "avoid", "下降趋势/回归空间不足，避免接飞刀或追已反弹价格"
        elif entry_low <= price <= entry_high and 25 <= rsi <= 47 and reversal:
            state, reason = "ready", "下轨与 RSI 条件成立，并已有初步反转确认"
        else:
            reason = "均值回归区域接近，但尚缺价格反转确认"

    elif strategy_key == "vol_breakout":
        trigger = resistance20 * 1.002
        entry_low, entry_high = trigger, trigger + 0.50 * atr
        max_chase = trigger + 1.00 * atr
        distance_to_trigger = (price - trigger) / atr
        condition = "放量收盘站上 20 日前高，且不超过追价上限"
        score = (
            (45 if uptrend else 25 if not downtrend else 0)
            + (30 if -0.5 <= distance_to_trigger <= 0.5 else 18 if -1.5 <= distance_to_trigger < -0.5 else 0)
            + (25 if volume_ratio >= 1.5 else 12 if volume_ratio >= 1.1 else 0)
        )
        if downtrend or price > max_chase or abs(_finite(feature.get("latest_gap"))) > 0.12:
            state, reason = "avoid", "突破过度扩张、趋势不支持或最新跳空过大"
        elif entry_low <= price <= entry_high and volume_ratio >= 1.5 and not downtrend:
            state, reason = "ready", "价格与成交量均确认突破，且仍在允许追价范围内"
        else:
            reason = "接近突破位，等待收盘价与成交量同时确认"

    else:
        alignment = _finite(feature.get("return_20")) > 0 and _finite(feature.get("return_60")) > 0
        score = (
            (50 if strong_up else 32 if uptrend else 5)
            + (25 if -0.35 <= distance_atr <= 0.85 else 12 if distance_atr <= 1.5 else 0)
            + (15 if alignment else 5) + (10 if rsi < 70 else 0)
        )
        condition = "趋势与中短周期方向一致，回调后再确认"
        if downtrend or distance_atr > 2.0 or rsi >= 78:
            state, reason = "avoid", "通用筛选结构不佳或已经明显过热"
        elif strong_up and alignment and entry_low <= price <= entry_high and rsi < 70:
            state, reason = "ready", "趋势、周期方向和回调位置同时通过"
        else:
            reason = "可继续研究，等待趋势与入场位置同时满足"
        if strategy_key == "ml_predict":
            condition += "；还需切换单股运行模型确认"

    entry_center = max(0.01, (entry_low + entry_high) / 2)
    stop_candidate = max(support - 0.20 * atr, entry_center - 2.0 * atr)
    stop_loss = min(stop_candidate, entry_low - 0.25 * atr, entry_center - 0.75 * atr)
    stop_loss = max(0.01, stop_loss)
    risk_per_share = max(entry_center - stop_loss, price * 0.002)
    if strategy_key == "vol_breakout":
        target_1 = entry_center + 2.0 * risk_per_share
        target_2 = entry_center + 3.0 * risk_per_share
    else:
        target_1 = structural_target
        target_2 = max(second_target, entry_center + 2.5 * risk_per_share)
    reward_risk = max(0.0, (target_1 - entry_center) / risk_per_share)
    if state == "ready" and reward_risk < 1.30:
        state, reason = "watch", reason + "；但到首目标的收益风险比不足 1.3"
    if strategy_key == "ml_predict" and state == "ready":
        state, reason = "watch", reason + "；横向页不训练单股模型"

    return {
        "setup_state": state, "setup_score": round(_clip(score), 1),
        "setup_reason": reason, "entry_condition": condition,
        "entry_low": max(0.01, entry_low), "entry_high": max(0.01, entry_high),
        "entry_center": entry_center, "breakout_trigger": max(0.01, trigger),
        "max_chase": max(0.01, max_chase), "stop_loss": stop_loss,
        "target_1": max(0.01, target_1), "target_2": max(0.01, target_2),
        "risk_per_share": risk_per_share, "reward_risk": reward_risk,
    }


def _percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    return (numeric.rank(method="average", pct=True, ascending=True) * 100).fillna(0.0)


def _liquidity_absolute(value: float, minimum: float) -> float:
    ratio = value / max(minimum, 1.0)
    return 100.0 if ratio >= 5 else 85.0 if ratio >= 2 else 70.0 if ratio >= 1 else 40.0 if ratio >= 0.5 else 10.0


def _risk_absolute(row: pd.Series, profile: Mapping[str, float]) -> float:
    atr_pct, max_atr = _finite(row["atr_pct"], 1.0), float(profile["max_atr_pct"])
    if atr_pct <= 0.03:
        atr_score = 100.0
    elif atr_pct <= max_atr:
        atr_score = 100 - 35 * (atr_pct - 0.03) / max(max_atr - 0.03, 0.001)
    else:
        atr_score = 35.0 if atr_pct <= max_atr * 1.5 else 5.0
    gap, max_gap = _finite(row["gap_risk_60"], 1.0), float(profile["max_gap_risk"])
    gap_score = 100.0 if gap <= 0.03 else 70.0 if gap <= max_gap else 20.0
    drawdown = abs(_finite(row["max_drawdown_120"], -1.0))
    drawdown_score = 100.0 if drawdown <= 0.15 else 75.0 if drawdown <= 0.25 else 40.0 if drawdown <= 0.40 else 10.0
    return _clip(0.45 * atr_score + 0.25 * gap_score + 0.30 * drawdown_score)


def _reward_risk_score(value: float) -> float:
    return 100.0 if value >= 2.0 else 82.0 if value >= 1.5 else 68.0 if value >= 1.3 else 45.0 if value >= 1.0 else 10.0


def _focus_label(score: float, status: str) -> str:
    if status == "技术面可执行" and score >= 72:
        return "优先核验"
    if score >= 60 and status != "暂缓":
        return "候选观察"
    return "普通跟踪" if score >= 45 else "暂不优先"


def _reason(row: pd.Series) -> str:
    if not row["guardrail_pass"]:
        return f"风控未通过：{row['failed_checks']}"
    if row["setup_state"] == "avoid":
        return row["setup_reason"]
    positives = []
    if row["strategy_score"] >= 70:
        positives.append("池内策略适配居前")
    if row["absolute_risk_score"] >= 75:
        positives.append("波动/回撤较可控")
    if row["absolute_liquidity_score"] >= 85:
        positives.append("流动性充足")
    return f"{'、'.join(positives[:2]) or '基础门槛通过'}；{row['setup_reason']}"


def rank_stocks(
    histories: Mapping[str, pd.DataFrame], strategy_key: str,
    names: Mapping[str, str] | None = None, *,
    profile_key: str = DEFAULT_EXECUTION_PROFILE, capital: float = 100_000.0,
    risk_pct: float = 0.01, max_position_pct: float = 0.30,
    lot_size: int = 1, reference_date=None,
) -> pd.DataFrame:
    """横向研究分 + 绝对执行分 + 风险预算仓位。"""
    if profile_key not in EXECUTION_PROFILES:
        raise ValueError(f"未知执行模式: {profile_key}")
    profile = EXECUTION_PROFILES[profile_key]
    rows = []
    for ticker, frame in histories.items():
        try:
            feature = extract_stock_features(ticker, frame, strategy_key)
            feature.update(build_execution_setup(feature, strategy_key))
            rows.append(feature)
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    for output, source in {
        "trend_score": "trend_raw", "momentum_score": "momentum_raw",
        "risk_score": "stability_raw", "liquidity_score": "liquidity_raw",
        "strategy_score": "strategy_raw",
    }.items():
        result[output] = _percentile(result[source])
    result["research_score"] = (
        0.35 * result["strategy_score"] + 0.20 * result["trend_score"]
        + 0.15 * result["momentum_score"] + 0.15 * result["risk_score"]
        + 0.15 * result["liquidity_score"]
    ).clip(0, 100)

    reference = (
        pd.Timestamp(result["latest_date"].max()).normalize()
        if reference_date is None
        else pd.Timestamp(reference_date).tz_localize(None).normalize()
    )
    result["data_age_days"] = result["latest_date"].map(
        lambda value: max(0, int((reference - pd.Timestamp(value).normalize()).days))
    )
    result["data_score"] = np.select(
        [result["bars"] >= 252, result["bars"] >= 160, result["bars"] >= 120],
        [100.0, 88.0, 74.0], default=55.0,
    )
    freshness_score = np.select(
        [result["data_age_days"] <= 1, result["data_age_days"] <= 3,
         result["data_age_days"] <= profile["max_data_age_days"]],
        [100.0, 85.0, 70.0], default=15.0,
    )
    result["absolute_data_score"] = 0.65 * result["data_score"] + 0.35 * freshness_score
    result["absolute_liquidity_score"] = result["avg_dollar_volume_20"].map(
        lambda value: _liquidity_absolute(value, float(profile["min_dollar_volume"]))
    )
    result["absolute_risk_score"] = result.apply(lambda row: _risk_absolute(row, profile), axis=1)
    result["reward_risk_score"] = result["reward_risk"].map(_reward_risk_score)
    result["execution_score"] = (
        0.35 * result["setup_score"] + 0.20 * result["absolute_risk_score"]
        + 0.15 * result["absolute_liquidity_score"] + 0.15 * result["absolute_data_score"]
        + 0.15 * result["reward_risk_score"]
    ).clip(0, 100)

    def checks_for(row: pd.Series) -> dict:
        return {
            "历史样本": int(row["bars"]) >= int(profile["min_bars"]),
            "价格门槛": _finite(row["price"]) >= float(profile["min_price"]),
            "流动性": _finite(row["avg_dollar_volume_20"]) >= float(profile["min_dollar_volume"]),
            "ATR 波动": _finite(row["atr_pct"], 1.0) <= float(profile["max_atr_pct"]),
            "跳空风险": _finite(row["gap_risk_60"], 1.0) <= float(profile["max_gap_risk"]),
            "数据新鲜": int(row["data_age_days"]) <= int(profile["max_data_age_days"]),
            "字段完整": _finite(row["finite_ratio"]) >= 0.98,
        }

    result["guardrail_checks"] = result.apply(checks_for, axis=1)
    result["guardrail_pass"] = result["guardrail_checks"].map(lambda checks: all(checks.values()))
    result["failed_checks"] = result["guardrail_checks"].map(
        lambda checks: "、".join(label for label, passed in checks.items() if not passed) or "—"
    )
    result["execution_status"] = np.select(
        [~result["guardrail_pass"], result["setup_state"].eq("avoid"),
         result["setup_state"].eq("ready") & result["reward_risk"].ge(1.30)],
        ["暂缓", "暂缓", "技术面可执行"], default="等待触发",
    )
    if strategy_key == "ml_predict":
        result.loc[result["execution_status"] == "技术面可执行", "execution_status"] = "等待单股模型"

    result["total_score"] = 0.45 * result["research_score"] + 0.55 * result["execution_score"]
    result.loc[~result["guardrail_pass"], "total_score"] = result.loc[~result["guardrail_pass"], "total_score"].clip(upper=44.0)
    result.loc[result["setup_state"] == "avoid", "total_score"] = result.loc[result["setup_state"] == "avoid", "total_score"].clip(upper=39.0)
    waiting = result["execution_status"].isin(["等待触发", "等待单股模型"])
    result.loc[waiting, "total_score"] = result.loc[waiting, "total_score"].clip(upper=71.0)
    result["total_score"] = result["total_score"].clip(0, 100).round(1)

    capital_value = max(0.0, float(capital))
    risk_budget = capital_value * max(0.0, float(risk_pct))
    max_position_value = (
        capital_value * max(0.0, float(max_position_pct))
        * float(profile["position_multiplier"])
    )

    def size_plan(row: pd.Series) -> pd.Series:
        if not row["guardrail_pass"] or row["setup_state"] == "avoid":
            shares = 0
        else:
            risk_limited = _round_down_lot(risk_budget / max(_finite(row["risk_per_share"]), 0.001), lot_size)
            score_modifier = 1.0 if row["execution_score"] >= 75 else 0.75 if row["execution_score"] >= 60 else 0.50
            value_limited = _round_down_lot(
                max_position_value * score_modifier / max(_finite(row["entry_center"]), 0.01), lot_size
            )
            shares = min(risk_limited, value_limited)
        first = _round_down_lot(shares * 0.5, lot_size)
        if shares > 0 and first == 0:
            first = shares
        return pd.Series({
            "planned_shares": int(shares), "first_shares": int(first),
            "planned_value": shares * _finite(row["entry_center"]),
            "planned_risk": shares * _finite(row["risk_per_share"]),
        })

    result = pd.concat([result, result.apply(size_plan, axis=1)], axis=1)
    result["risk_budget"] = risk_budget
    result["profile_key"], result["profile_label"] = profile_key, profile["label"]
    result["name"] = result["ticker"].map(dict(names or {})).fillna(result["ticker"])
    result["focus"] = result.apply(lambda row: _focus_label(row["total_score"], row["execution_status"]), axis=1)
    result["strategy_label"] = STRATEGY_LABELS.get(strategy_key, strategy_key)
    result["reason"] = result.apply(_reason, axis=1)
    result = result.sort_values(
        ["total_score", "execution_score", "strategy_score", "absolute_liquidity_score"],
        ascending=False,
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result
