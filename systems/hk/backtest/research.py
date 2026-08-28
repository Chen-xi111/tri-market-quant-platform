"""研究级回测验证：扩展窗口、成本压力与选择偏差校正。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd

from .engine import BacktestEngine


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def to_dict(self) -> dict:
        return asdict(self)


def expanding_walk_forward_splits(
    length: int,
    n_splits: int = 4,
    min_train_bars: int = 160,
    test_bars: int | None = None,
    gap_bars: int = 1,
) -> list[WalkForwardFold]:
    length = int(length)
    n_splits = max(3, int(n_splits))
    min_train_bars = max(80, int(min_train_bars))
    gap_bars = max(0, int(gap_bars))
    available = length - min_train_bars - gap_bars
    if available < n_splits * 20:
        raise ValueError(f"至少需要 {min_train_bars + gap_bars + n_splits * 20} 根日 K，当前只有 {length} 根")
    if test_bars is None:
        test_bars = min(63, available // n_splits)
    test_bars = max(20, int(test_bars))
    if test_bars * n_splits > available:
        test_bars = available // n_splits
    first_test = length - test_bars * n_splits
    folds = []
    for offset in range(n_splits):
        test_start = first_test + offset * test_bars
        test_end = min(length, test_start + test_bars)
        train_end = test_start - gap_bars
        if train_end >= min_train_bars:
            folds.append(WalkForwardFold(offset + 1, 0, train_end, test_start, test_end))
    if len(folds) < 3:
        raise ValueError("无法形成至少 3 个有效走样本外折叠")
    return folds


def probabilistic_sharpe_confidence(returns: pd.Series, target_sharpe: float = 0.0) -> float:
    values = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 20:
        return 0.0
    std = float(values.std(ddof=1))
    if std <= 0:
        return 1.0 if float(values.mean()) > 0 else 0.0
    daily_sharpe = float(values.mean()) / std
    target_daily = float(target_sharpe) / sqrt(252)
    skewness = float(values.skew()) if len(values) > 2 else 0.0
    kurtosis = float(values.kurt() + 3) if len(values) > 3 else 3.0
    denominator = sqrt(max(1e-12, 1 - skewness * daily_sharpe + (kurtosis - 1) * daily_sharpe**2 / 4))
    z_score = (daily_sharpe - target_daily) * sqrt(len(values) - 1) / denominator
    return float(np.clip(0.5 * (1 + erf(z_score / sqrt(2))), 0, 1))


def adjusted_selection_confidence(raw_confidence: float, trials: int) -> float:
    return float(np.clip(1 - max(1, int(trials)) * (1 - float(raw_confidence)), 0, 1))


def block_bootstrap_annual_return_ci(returns: pd.Series, block_size: int = 10, samples: int = 500, random_state: int = 42) -> tuple[float | None, float | None]:
    values = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(values) < 30:
        return None, None
    block_size = max(2, min(int(block_size), len(values)))
    rng = np.random.default_rng(random_state)
    estimates = np.empty(max(200, int(samples)))
    for sample_index in range(len(estimates)):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block_size)))
        draw = np.concatenate([values[(start + np.arange(block_size)) % len(values)] for start in starts])[: len(values)]
        estimates[sample_index] = draw.mean() * 252 * 100
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _compound_return(returns: pd.Series) -> float:
    values = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return float(((1 + values).prod() - 1) * 100) if len(values) else 0.0


def _research_score(summary: dict) -> float:
    score = 15 * np.clip(summary["positive_fold_ratio"], 0, 1)
    score += 15 * np.clip(summary["median_sharpe"] / 1.2, 0, 1)
    score += 15 * np.clip(summary["median_alpha"] / 8, 0, 1)
    score += 15 * np.clip((summary["worst_drawdown"] + 30) / 25, 0, 1)
    score += 20 * np.clip(summary["selection_confidence"], 0, 1)
    score += 10 * np.clip(summary["total_trades"] / 12, 0, 1)
    if summary["total_trades"] > 0:
        score += 10 * np.clip((summary["stressed_return"] + 2) / 8, 0, 1)
    return round(float(np.clip(score, 0, 100)), 1)


def evaluate_parameter_grid(df: pd.DataFrame, strategy_class, param_name: str, param_values: list, engine_kwargs: dict, n_splits: int = 4, gap_bars: int = 1, base_slippage_bps: float = 5.0, stressed_slippage_bps: float = 15.0) -> dict:
    clean = df.copy().sort_index()
    folds = expanding_walk_forward_splits(len(clean), n_splits=n_splits, gap_bars=gap_bars)
    summaries, fold_rows = [], []
    trial_count = max(1, len(param_values))
    for parameter_value in param_values:
        parameter_returns, stressed_returns, parameter_folds = [], [], []
        for fold in folds:
            evaluation_start = clean.index[fold.test_start]
            window = clean.iloc[: fold.test_end].copy()
            kwargs = dict(engine_kwargs); kwargs["slippage_bps"] = float(base_slippage_bps)
            result = BacktestEngine(strategy_class(params={param_name: parameter_value}), **kwargs).run(window, trade_start_date=evaluation_start)
            stress_kwargs = dict(engine_kwargs); stress_kwargs["slippage_bps"] = float(stressed_slippage_bps)
            stressed = BacktestEngine(strategy_class(params={param_name: parameter_value}), **stress_kwargs).run(window, trade_start_date=evaluation_start)
            equity = result["evaluation_equity_curve"]
            stressed_equity = stressed["evaluation_equity_curve"]
            parameter_returns.append(equity.pct_change().dropna()); stressed_returns.append(stressed_equity.pct_change().dropna())
            metrics = result["metrics"]
            record = {"参数值": parameter_value, "折叠": fold.fold, "训练截止": str(clean.index[fold.train_end - 1].date()), "验证开始": str(evaluation_start.date()), "验证结束": str(clean.index[fold.test_end - 1].date()), "验证收益%": round(float(metrics.get("total_return", 0) or 0), 2), "超额收益%": round(float(metrics.get("alpha", 0) or 0), 2), "夏普": round(float(metrics.get("sharpe_ratio", 0) or 0), 3), "最大回撤%": round(float(metrics.get("max_drawdown", 0) or 0), 2), "交易数": int(metrics.get("total_trades", 0) or 0)}
            fold_rows.append(record); parameter_folds.append(record)
        joined = pd.concat(parameter_returns).sort_index() if parameter_returns else pd.Series(dtype=float)
        stressed_joined = pd.concat(stressed_returns).sort_index() if stressed_returns else pd.Series(dtype=float)
        raw_confidence = probabilistic_sharpe_confidence(joined)
        summary = {"参数值": parameter_value, "fold_count": len(parameter_folds), "positive_fold_ratio": float(np.mean([row["超额收益%"] > 0 for row in parameter_folds])), "median_sharpe": float(np.median([row["夏普"] for row in parameter_folds])), "median_alpha": float(np.median([row["超额收益%"] for row in parameter_folds])), "worst_drawdown": float(min(row["最大回撤%"] for row in parameter_folds)), "total_trades": int(sum(row["交易数"] for row in parameter_folds)), "oos_return": _compound_return(joined), "stressed_return": _compound_return(stressed_joined), "raw_confidence": raw_confidence, "selection_confidence": adjusted_selection_confidence(raw_confidence, trial_count)}
        summary["annual_return_ci_low"], summary["annual_return_ci_high"] = block_bootstrap_annual_return_ci(joined)
        summary["research_score"] = _research_score(summary)
        checks = {"至少 3 个走样本外折叠": summary["fold_count"] >= 3, "至少 75% 折叠跑赢基准": summary["positive_fold_ratio"] >= 0.75, "折叠中位超额收益为正": summary["median_alpha"] > 0, "样本外完成交易不少于 12 笔": summary["total_trades"] >= 12, "最差折叠回撤不超过 25%": summary["worst_drawdown"] >= -25, "多重试验校正置信度不低于 80%": summary["selection_confidence"] >= 0.8, "块自助年化收益下界为正": summary["annual_return_ci_low"] is not None and summary["annual_return_ci_low"] > 0, "15bp 滑点压力后仍为正收益": summary["total_trades"] > 0 and summary["stressed_return"] > 0}
        summary["checks"] = checks; summary["gate_pass"] = all(checks.values()) and summary["research_score"] >= 70; summary["status"] = "研究验证通过" if summary["gate_pass"] else "证据仍不足"
        summaries.append(summary)
    summaries.sort(key=lambda item: item["research_score"], reverse=True)
    return {"rows": summaries, "fold_rows": fold_rows, "best": summaries[0] if summaries else None, "splits": [fold.to_dict() for fold in folds], "trial_count": trial_count, "base_slippage_bps": float(base_slippage_bps), "stressed_slippage_bps": float(stressed_slippage_bps)}
