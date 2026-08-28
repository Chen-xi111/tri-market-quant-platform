"""回测结果的实盘准备度与样本外稳健性评估。"""

from __future__ import annotations


def assess_readiness(metrics: dict) -> dict:
    """把关键门槛转换成可解释的 0~100 准备度分数。"""
    checks = [
        ("至少 5 笔已完成交易", metrics.get("total_trades", 0) >= 5, 25),
        ("跑赢买入持有基准", metrics.get("alpha", 0) > 0, 20),
        ("夏普比率不低于 0.5", metrics.get("sharpe_ratio", 0) >= 0.5, 20),
        ("最大回撤不超过 20%", metrics.get("max_drawdown", -100) >= -20, 20),
        ("盈亏比不低于 1.1", metrics.get("profit_factor", 0) >= 1.1, 15),
    ]
    score = sum(weight for _, passed, weight in checks if passed)
    if score >= 80:
        label = "基础检查通过（仍需样本外）"
        color = "green"
    elif score >= 60:
        label = "基础检查部分通过"
        color = "orange"
    else:
        label = "基础检查未通过"
        color = "red"

    return {
        "score": score,
        "label": label,
        "color": color,
        "checks": {name: passed for name, passed, _ in checks},
    }


def robustness_score(train_metrics: dict, validation_metrics: dict) -> float:
    """样本外优先，兼顾收益衰减和最少交易数的稳健性评分。"""
    validation_sharpe = float(validation_metrics.get("sharpe_ratio", 0) or 0)
    validation_return = float(validation_metrics.get("total_return", 0) or 0)
    train_return = float(train_metrics.get("total_return", 0) or 0)
    trades = int(validation_metrics.get("total_trades", 0) or 0)
    decay_penalty = abs(train_return - validation_return) / 50
    trade_penalty = max(0, 3 - trades) * 0.35
    loss_penalty = 0.5 if validation_return < 0 else 0.0
    return round(validation_sharpe - decay_penalty - trade_penalty - loss_penalty, 3)


SCAN_GRIDS = {
    "trend_rider": ("pullback_pct", "回调容忍度", [0.01, 0.015, 0.02, 0.03, 0.04]),
    "ma_crossover": ("fast", "快速均线周期", [3, 5, 8, 10, 12]),
    "bollinger_reversion": ("std", "布林带标准差", [1.5, 1.8, 2.0, 2.3, 2.6]),
    "vol_breakout": ("vol_ratio", "突破量比门槛", [1.1, 1.3, 1.5, 1.8, 2.2]),
    "multi_tf": ("rsi_max", "RSI 上限", [60, 65, 70, 75, 80]),
    "alpha_factors": ("entry_percentile", "Alpha 入场分位", [0.55, 0.62, 0.70, 0.78, 0.85]),
    "ml_predict": ("prob_threshold", "上涨概率门槛", [0.50, 0.52, 0.55, 0.58, 0.62]),
}


def scan_grid(strategy_key: str) -> tuple[str, str, list]:
    """返回确实被对应策略读取的可扫描参数。"""
    return SCAN_GRIDS[strategy_key]
