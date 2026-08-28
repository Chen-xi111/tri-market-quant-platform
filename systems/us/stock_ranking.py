"""美股横向重点股评分。

评分是候选池内的相对排名，不是绝对买入信号。每只股先提取趋势、
动量、风险、流动性和策略适配特征，再在同一候选池中转换为百分位分。
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


MIN_RANKING_BARS = 90

STRATEGY_LABELS = {
    "trend_rider": "趋势骑乘",
    "ma_crossover": "双均线趋势",
    "bollinger_reversion": "布林带回归",
    "vol_breakout": "量价突破",
    "multi_tf": "多周期共振",
    "alpha_factors": "Alpha 多因子",
    "ml_predict": "机器学习",
}


def _finite(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _last(series: pd.Series, default: float = 0.0) -> float:
    return _finite(series.iloc[-1] if len(series) else default, default)


def _strategy_fit(strategy_key: str, feature: dict) -> float:
    trend = feature["trend_raw"]
    momentum = feature["momentum_raw"]
    stability = feature["stability_raw"]
    distance_ma20 = feature["distance_ma20"]
    rsi = feature["rsi"]
    boll_pct = feature["boll_pct"]
    breakout = feature["breakout_20"]
    volume_ratio = feature["volume_ratio"]

    if strategy_key == "trend_rider":
        pullback_quality = -abs(distance_ma20 - 0.01)
        return 0.50 * trend + 0.25 * momentum + 0.25 * pullback_quality
    if strategy_key == "ma_crossover":
        return 0.65 * trend + 0.35 * momentum
    if strategy_key == "bollinger_reversion":
        # 靠近下轨、RSI 靠近 38，但不奖励极端破位。
        band_setup = -abs(boll_pct - 0.15)
        rsi_setup = -abs(rsi - 38.0) / 100
        return 0.50 * band_setup + 0.30 * rsi_setup + 0.20 * max(trend, -0.08)
    if strategy_key == "vol_breakout":
        return 0.50 * breakout + 0.30 * math.log(max(volume_ratio, 0.1)) + 0.20 * momentum
    if strategy_key == "multi_tf":
        agreement_penalty = -abs(feature["return_20"] - feature["return_60"] / 3)
        return 0.45 * trend + 0.45 * momentum + 0.10 * agreement_penalty
    if strategy_key == "alpha_factors":
        return 0.35 * trend + 0.40 * momentum + 0.25 * stability
    # ML 页面的横向初筛不训练单独模型，以通用多因子代理，
    # 防止一次排名对几十只股重复拟合。
    return 0.35 * trend + 0.35 * momentum + 0.30 * stability


def extract_stock_features(ticker: str, frame: pd.DataFrame, strategy_key: str) -> dict:
    """从一只股的日 K 提取横向可比特征。"""
    if frame is None or len(frame) < MIN_RANKING_BARS:
        raise ValueError(f"日 K 不足 {MIN_RANKING_BARS} 根")
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"缺少字段: {', '.join(sorted(missing))}")

    data = frame.sort_index().copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce").fillna(0)
    returns = close.pct_change()
    price = _last(close)
    ma20 = _finite(data["MA20"].iloc[-1]) if "MA20" in data else _last(close.rolling(20).mean())
    ma60 = _finite(data["MA60"].iloc[-1]) if "MA60" in data else _last(close.rolling(60).mean())
    ma20_then = _finite(close.rolling(20).mean().iloc[-21], ma20)
    return_20 = _finite(close.iloc[-1] / close.iloc[-21] - 1)
    return_60 = _finite(close.iloc[-1] / close.iloc[-61] - 1)
    volatility = _finite(returns.tail(60).std() * np.sqrt(252), 1.0)
    rolling_peak = close.tail(120).cummax()
    max_drawdown = _finite((close.tail(120) / rolling_peak - 1).min(), -1.0)
    avg_dollar_volume = _finite((close * volume).tail(20).mean())
    resistance = _finite(high.shift(1).rolling(20).max().iloc[-1], price)
    distance_ma20 = price / ma20 - 1 if ma20 else 0.0
    trend_raw = (
        0.45 * (price / ma60 - 1 if ma60 else 0.0)
        + 0.35 * (ma20 / ma60 - 1 if ma60 else 0.0)
        + 0.20 * (ma20 / ma20_then - 1 if ma20_then else 0.0)
    )
    momentum_raw = 0.60 * return_20 + 0.40 * return_60
    stability_raw = -(0.65 * volatility + 0.35 * abs(max_drawdown))
    rsi = _finite(data["RSI"].iloc[-1], 50.0) if "RSI" in data else 50.0
    boll_pct = _finite(data["BOLL_PCT"].iloc[-1], 0.5) if "BOLL_PCT" in data else 0.5
    volume_ratio = _finite(data["VOL_RATIO"].iloc[-1], 1.0) if "VOL_RATIO" in data else 1.0

    feature = {
        "ticker": ticker,
        "price": price,
        "bars": len(data),
        "latest_date": pd.Timestamp(data.index[-1]),
        "return_20": return_20,
        "return_60": return_60,
        "volatility_60": volatility,
        "max_drawdown_120": max_drawdown,
        "avg_dollar_volume_20": avg_dollar_volume,
        "trend_raw": trend_raw,
        "momentum_raw": momentum_raw,
        "stability_raw": stability_raw,
        "liquidity_raw": math.log10(max(avg_dollar_volume, 1.0)),
        "distance_ma20": distance_ma20,
        "rsi": rsi,
        "boll_pct": boll_pct,
        "volume_ratio": volume_ratio,
        "breakout_20": price / resistance - 1 if resistance else 0.0,
    }
    feature["strategy_raw"] = _strategy_fit(strategy_key, feature)
    return feature


def _percentile(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    ranked = numeric.rank(method="average", pct=True, ascending=higher_is_better) * 100
    # pandas ascending=True 会让大值获得更高的 pct；参数名保留业务语义。
    return ranked.fillna(0.0)


def _focus_label(score: float) -> str:
    if score >= 75:
        return "重点研究"
    if score >= 60:
        return "候选观察"
    if score >= 45:
        return "普通跟踪"
    return "暂不优先"


def _reason(row: pd.Series) -> str:
    positives = []
    if row["strategy_score"] >= 70:
        positives.append("策略适配度高")
    if row["trend_score"] >= 70:
        positives.append("趋势结构居前")
    if row["momentum_score"] >= 70:
        positives.append("中短期动量较强")
    if row["risk_score"] >= 70:
        positives.append("波动与回撤较可控")
    if row["liquidity_score"] >= 70:
        positives.append("成交流动性好")
    cautions = []
    if row["rsi"] >= 75:
        cautions.append("RSI 过热")
    if row["risk_score"] <= 30:
        cautions.append("高波动/深回撤")
    if row["liquidity_score"] <= 20:
        cautions.append("流动性偏弱")
    if abs(row["distance_ma20"]) >= 0.15:
        cautions.append("偏离 MA20 较远")
    summary = "、".join(positives[:2]) or "综合表现居中"
    return summary + (f"；注意{'、'.join(cautions[:2])}" if cautions else "")


def rank_stocks(
    histories: Mapping[str, pd.DataFrame],
    strategy_key: str,
    names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """对候选池做横向百分位评分，返回由高到低的排名。"""
    rows = []
    for ticker, frame in histories.items():
        try:
            rows.append(extract_stock_features(ticker, frame, strategy_key))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["trend_score"] = _percentile(result["trend_raw"])
    result["momentum_score"] = _percentile(result["momentum_raw"])
    result["risk_score"] = _percentile(result["stability_raw"])
    result["liquidity_score"] = _percentile(result["liquidity_raw"])
    result["strategy_score"] = _percentile(result["strategy_raw"])
    result["data_score"] = np.select(
        [result["bars"] >= 252, result["bars"] >= 160, result["bars"] >= 120],
        [100.0, 85.0, 72.0],
        default=60.0,
    )
    base_score = (
        0.35 * result["strategy_score"]
        + 0.20 * result["trend_score"]
        + 0.15 * result["momentum_score"]
        + 0.15 * result["risk_score"]
        + 0.15 * result["liquidity_score"]
    )
    result["total_score"] = base_score * (0.85 + 0.15 * result["data_score"] / 100)
    result["total_score"] = result["total_score"].clip(0, 100).round(1)
    result["name"] = result["ticker"].map(dict(names or {})).fillna(result["ticker"])
    result["focus"] = result["total_score"].map(_focus_label)
    result["strategy_label"] = STRATEGY_LABELS.get(strategy_key, strategy_key)
    result["reason"] = result.apply(_reason, axis=1)
    result = result.sort_values(
        ["total_score", "strategy_score", "liquidity_score"], ascending=False
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result
