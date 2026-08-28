"""
回测绩效指标计算
"""

import pandas as pd
import numpy as np
from typing import Dict


def compute_metrics(
    equity_curve: pd.Series,
    trades: list,
    risk_free_rate: float = 0.02,
    benchmark: pd.Series = None,
) -> Dict:
    """
    计算全套绩效指标
    trades: [(entry_date, exit_date, entry_price, exit_price, pnl, pnl_pct, reason), ...]
    benchmark: 买入持有基准的权益曲线（可选）
    """
    metrics = {}

    if len(equity_curve) < 2:
        return {"error": "数据不足"}

    returns = equity_curve.pct_change().dropna()

    # ---- 收益指标 ----
    metrics["total_return"] = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    years = max((len(equity_curve) - 1) / 252, 1 / 252)
    growth = max(float(equity_curve.iloc[-1] / equity_curve.iloc[0]), 0.0)
    metrics["annual_return"] = ((growth ** (1 / years)) - 1) * 100 if growth > 0 else -100.0

    # ---- 风险指标 ----
    metrics["volatility"] = returns.std() * np.sqrt(252) * 100
    metrics["max_drawdown"] = _max_drawdown(equity_curve) * 100
    metrics["max_drawdown_days"] = _max_drawdown_days(equity_curve)

    # ---- 风险调整收益 ----
    excess = returns.mean() * 252 - risk_free_rate
    metrics["sharpe_ratio"] = excess / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    # Sortino
    downside = returns[returns < 0].std() * np.sqrt(252)
    metrics["sortino_ratio"] = excess / downside if downside > 0 else 0
    # Calmar
    metrics["calmar_ratio"] = metrics["annual_return"] / max(abs(metrics["max_drawdown"]), 0.01)

    # ---- 交易统计 ----
    if trades:
        pnls = [t[4] for t in trades]
        pnl_pcts = [t[5] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        metrics["total_trades"] = len(trades)
        metrics["win_rate"] = len(wins) / len(trades) * 100 if trades else 0
        metrics["avg_win"] = np.mean(wins) if wins else 0
        metrics["avg_loss"] = np.mean(losses) if losses else 0
        metrics["profit_factor"] = (
            sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
        )
        metrics["avg_trade_pnl"] = np.mean(pnls)
        # BacktestEngine 已按百分数记录单笔收益，不能再次乘 100。
        metrics["avg_trade_pnl_pct"] = np.mean(pnl_pcts)
        metrics["max_consecutive_wins"] = _max_streak(pnls, "win")
        metrics["max_consecutive_losses"] = _max_streak(pnls, "loss")
        metrics["largest_win"] = max(pnls) if wins else 0
        metrics["largest_loss"] = min(pnls) if losses else 0
    else:
        metrics["total_trades"] = 0
        metrics["win_rate"] = 0
        metrics["profit_factor"] = 0

    # ---- 基准对比（买入持有）----
    if benchmark is not None and len(benchmark) >= 2:
        bench_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
        metrics["benchmark_return"] = bench_return
        metrics["alpha"] = metrics["total_return"] - bench_return
        # 基准最大回撤
        bench_peak = benchmark.cummax()
        bench_dd = ((benchmark - bench_peak) / bench_peak).min() * 100
        metrics["benchmark_drawdown"] = bench_dd
        metrics.update(tracking_error=0.0, information_ratio=0.0, beta=0.0)
        aligned = pd.concat(
            [equity_curve.pct_change().rename("strategy"), benchmark.pct_change().rename("benchmark")],
            axis=1,
        ).dropna()
        if len(aligned) >= 2:
            active = aligned["strategy"] - aligned["benchmark"]
            tracking_error = float(active.std() * np.sqrt(252))
            metrics["tracking_error"] = tracking_error * 100
            metrics["information_ratio"] = (
                float(active.mean() * 252 / tracking_error) if tracking_error > 0 else 0.0
            )
            benchmark_variance = float(aligned["benchmark"].var())
            metrics["beta"] = (
                float(aligned["strategy"].cov(aligned["benchmark"]) / benchmark_variance)
                if benchmark_variance > 0 else 0.0
            )
    else:
        metrics["benchmark_return"] = 0.0
        metrics["alpha"] = metrics.get("total_return", 0)
        metrics["tracking_error"] = 0.0
        metrics["information_ratio"] = 0.0
        metrics["beta"] = 0.0

    # ---- 交易持续时间统计 ----
    if trades:
        durations = []
        for t in trades:
            entry_d, exit_d = t[0], t[1]
            if hasattr(entry_d, 'date') and hasattr(exit_d, 'date'):
                dur = (exit_d - entry_d).days
                durations.append(dur)
        if durations:
            metrics["avg_holding_days"] = round(np.mean(durations), 1)
            metrics["max_holding_days"] = max(durations)

    return metrics


def _max_drawdown(equity: pd.Series) -> float:
    """最大回撤（比例）"""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown.min()


def _max_drawdown_days(equity: pd.Series) -> int:
    """最大回撤持续天数"""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    underwater = drawdown < 0
    # 找最长连续水下天数
    streaks = underwater.astype(int).groupby((~underwater).cumsum()).cumsum()
    return streaks.max()


def _max_streak(pnls: list, kind: str) -> int:
    """最大连续盈利/亏损次数"""
    max_s = cur = 0
    for p in pnls:
        if (kind == "win" and p > 0) or (kind == "loss" and p <= 0):
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def metrics_summary(metrics: Dict) -> str:
    """指标可读摘要"""
    lines = [
        "=" * 50,
        "  📊 回测绩效报告",
        "=" * 50,
        f"  总收益率:     {metrics.get('total_return', 0):+.1f}%",
        f"  年化收益:     {metrics.get('annual_return', 0):+.1f}%",
        f"  年化波动:     {metrics.get('volatility', 0):.1f}%",
        f"  最大回撤:     {metrics.get('max_drawdown', 0):.1f}%",
        f"  夏普比率:     {metrics.get('sharpe_ratio', 0):.2f}",
        f"  索提诺比率:   {metrics.get('sortino_ratio', 0):.2f}",
        f"  卡尔玛比率:   {metrics.get('calmar_ratio', 0):.2f}",
        "-" * 50,
        f"  总交易次数:   {metrics.get('total_trades', 0)}",
        f"  胜率:         {metrics.get('win_rate', 0):.1f}%",
        f"  盈亏比:       {metrics.get('profit_factor', 0):.2f}",
        f"  平均盈利:     {metrics.get('avg_win', 0):.2f} CNY",
        f"  平均亏损:     {metrics.get('avg_loss', 0):.2f} CNY",
        f"  最大盈利:     {metrics.get('largest_win', 0):.2f} CNY",
        f"  最大亏损:     {metrics.get('largest_loss', 0):.2f} CNY",
        f"  最大连赢:     {metrics.get('max_consecutive_wins', 0)}次",
        f"  最大连亏:     {metrics.get('max_consecutive_losses', 0)}次",
        "-" * 50,
        f"  买入持有:     {metrics.get('benchmark_return', 0):+.1f}%",
        f"  超额收益:     {metrics.get('alpha', 0):+.1f}%",
        f"  平均持仓:     {metrics.get('avg_holding_days', 0):.0f}天",
        "=" * 50,
    ]
    return "\n".join(lines)
