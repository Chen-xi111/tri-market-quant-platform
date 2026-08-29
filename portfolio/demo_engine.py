"""Small, deterministic research engine for the public single-process demo.

The demo intentionally uses transparent long-only rules. Signals are shifted by
one bar before they earn returns, and turnover costs are deducted on every
position change. The production market services keep their richer engines.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA_DIR = ROOT / "demo_data"

STRATEGIES = {
    "trend": "Medium-term trend",
    "mean_reversion": "Short-term mean reversion",
    "breakout": "Price breakout",
    "multi_factor": "Transparent multi-factor",
}

PARAMETER_LABELS = {
    "trend": "Slow moving-average window",
    "mean_reversion": "Entry z-score",
    "breakout": "Breakout lookback",
    "multi_factor": "Composite entry threshold",
}

PARAMETER_GRIDS = {
    "trend": [80, 100, 120, 150, 180],
    "mean_reversion": [0.75, 1.0, 1.25, 1.5, 1.75],
    "breakout": [30, 40, 55, 70, 90],
    "multi_factor": [0.35, 0.45, 0.55, 0.65, 0.75],
}

DEFAULT_PARAMETERS = {
    "trend": 120,
    "mean_reversion": 1.25,
    "breakout": 55,
    "multi_factor": 0.55,
}


@dataclass(frozen=True)
class DemoResult:
    market_key: str
    symbol: str
    strategy_key: str
    cost_bps: float
    split_date: pd.Timestamp
    history: pd.DataFrame
    metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    parameter_stability: pd.DataFrame
    parameter_label: str


def evidence_diagnostics(result: DemoResult) -> dict[str, float | int | str]:
    """Summarize held-out evidence without turning it into a trading recommendation."""
    train = result.metrics.loc["Train"]
    validation = result.metrics.loc["Validation"]
    alpha = float(validation["strategy_return"] - validation["benchmark_return"])
    sharpe_decay = float(validation["sharpe"] - train["sharpe"])
    stability = (
        result.parameter_stability["validation_sharpe"]
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
    )
    positive_settings = int((stability > 0).sum())

    if validation["sharpe"] > 0 and alpha > 0 and positive_settings >= 3:
        classification = "Consistent in this snapshot"
    elif validation["sharpe"] > 0 or alpha > 0 or positive_settings >= 3:
        classification = "Mixed in this snapshot"
    else:
        classification = "Weak in this snapshot"

    return {
        "classification": classification,
        "validation_alpha": alpha,
        "sharpe_decay": sharpe_decay,
        "positive_settings": positive_settings,
        "tested_settings": len(stability),
    }


@lru_cache(maxsize=4)
def _load_demo_manifest_cached(modified_ns: int) -> dict:
    del modified_ns
    return json.loads((DEMO_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def load_demo_manifest() -> dict:
    """Load versioned fixture metadata and return a mutation-safe copy."""
    manifest_path = DEMO_DATA_DIR / "manifest.json"
    cached = _load_demo_manifest_cached(manifest_path.stat().st_mtime_ns)
    return copy.deepcopy(cached)


@lru_cache(maxsize=24)
def _load_demo_prices_cached(file_name: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    frame = pd.read_csv(DEMO_DATA_DIR / file_name, parse_dates=["date"])
    frame = frame.set_index("date").sort_index()
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=numeric)


def load_demo_prices(market_key: str, symbol: str) -> pd.DataFrame:
    """Load a price snapshot with automatic invalidation when its file changes."""
    manifest = load_demo_manifest()
    metadata = manifest["markets"][market_key]["symbols"][symbol]
    fixture_path = DEMO_DATA_DIR / metadata["file"]
    cached = _load_demo_prices_cached(metadata["file"], fixture_path.stat().st_mtime_ns)
    return cached.copy(deep=True)


def _stateful_long(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    state = 0.0
    values: list[float] = []
    for should_enter, should_exit in zip(entry.fillna(False), exit_.fillna(False)):
        if state == 0 and bool(should_enter):
            state = 1.0
        elif state == 1 and bool(should_exit):
            state = 0.0
        values.append(state)
    return pd.Series(values, index=entry.index, dtype=float)


def strategy_position(
    prices: pd.DataFrame,
    strategy_key: str,
    parameter: float | None = None,
) -> pd.Series:
    close = prices["close"].astype(float)
    volume = prices["volume"].astype(float)
    parameter = DEFAULT_PARAMETERS[strategy_key] if parameter is None else parameter

    if strategy_key == "trend":
        fast = close.rolling(40).mean()
        slow = close.rolling(int(parameter)).mean()
        raw = (fast > slow).astype(float)
    elif strategy_key == "mean_reversion":
        mean = close.rolling(20).mean()
        std = close.rolling(20).std().replace(0, np.nan)
        zscore = (close - mean) / std
        raw = _stateful_long(zscore < -float(parameter), zscore >= 0)
    elif strategy_key == "breakout":
        lookback = int(parameter)
        entry_level = close.rolling(lookback).max().shift(1)
        exit_level = close.rolling(20).min().shift(1)
        volume_ok = volume > volume.rolling(20).mean()
        raw = _stateful_long((close > entry_level) & volume_ok, close < exit_level)
    elif strategy_key == "multi_factor":
        trend = close / close.rolling(120).mean() - 1
        momentum = close.pct_change(60)
        volatility = close.pct_change().rolling(20).std() * np.sqrt(252)
        trend_score = trend.rank(pct=True)
        momentum_score = momentum.rank(pct=True)
        low_vol_score = 1 - volatility.rank(pct=True)
        composite = 0.4 * trend_score + 0.4 * momentum_score + 0.2 * low_vol_score
        raw = (composite >= float(parameter)).astype(float)
    else:
        raise KeyError(f"Unknown strategy: {strategy_key}")

    # A signal observed at the end of day t can earn returns only from t+1.
    return raw.shift(1).fillna(0.0).rename("position")


def _returns_frame(
    prices: pd.DataFrame,
    strategy_key: str,
    cost_bps: float,
    parameter: float | None = None,
) -> pd.DataFrame:
    position = strategy_position(prices, strategy_key, parameter)
    market_return = prices["close"].pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy_return = position * market_return - turnover * float(cost_bps) / 10_000
    output = prices[["close", "volume"]].copy()
    output["position"] = position
    output["turnover"] = turnover
    output["strategy_return"] = strategy_return
    output["benchmark_return"] = market_return
    output["strategy_equity"] = (1 + strategy_return).cumprod()
    output["benchmark_equity"] = (1 + market_return).cumprod()
    output["drawdown"] = output["strategy_equity"] / output["strategy_equity"].cummax() - 1
    return output


def _metric_row(frame: pd.DataFrame, period: str) -> dict:
    returns = frame["strategy_return"]
    benchmark = frame["benchmark_return"]
    equity = (1 + returns).cumprod()
    benchmark_equity = (1 + benchmark).cumprod()
    volatility = float(returns.std(ddof=0))
    sharpe = float(returns.mean() / volatility * np.sqrt(252)) if volatility > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    return {
        "period": period,
        "strategy_return": float(equity.iloc[-1] - 1),
        "benchmark_return": float(benchmark_equity.iloc[-1] - 1),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "trades": int((frame["position"].diff() > 0).sum()),
        "exposure": float(frame["position"].mean()),
    }


def _metrics(history: pd.DataFrame, split_index: int) -> pd.DataFrame:
    train = history.iloc[:split_index]
    validation = history.iloc[split_index:]
    return pd.DataFrame(
        [_metric_row(train, "Train"), _metric_row(validation, "Validation")]
    ).set_index("period")


def run_demo_research(
    market_key: str,
    symbol: str,
    strategy_key: str,
    cost_bps: float,
    train_fraction: float = 0.7,
) -> DemoResult:
    prices = load_demo_prices(market_key, symbol)
    split_index = max(80, min(len(prices) - 40, int(len(prices) * train_fraction)))
    split_date = prices.index[split_index]
    history = _returns_frame(prices, strategy_key, cost_bps)
    metrics = _metrics(history, split_index)

    cost_rows = []
    for scenario_cost in [0, 5, 10, 15, 25, 40]:
        scenario = _returns_frame(prices, strategy_key, scenario_cost).iloc[split_index:]
        row = _metric_row(scenario, "Validation")
        cost_rows.append(
            {
                "cost_bps": scenario_cost,
                "validation_return": row["strategy_return"],
                "validation_sharpe": row["sharpe"],
                "max_drawdown": row["max_drawdown"],
            }
        )

    parameter_rows = []
    for parameter in PARAMETER_GRIDS[strategy_key]:
        scenario = _returns_frame(prices, strategy_key, cost_bps, parameter)
        scenario_metrics = _metrics(scenario, split_index)
        parameter_rows.append(
            {
                "parameter": parameter,
                "train_sharpe": scenario_metrics.loc["Train", "sharpe"],
                "validation_sharpe": scenario_metrics.loc["Validation", "sharpe"],
                "validation_return": scenario_metrics.loc["Validation", "strategy_return"],
            }
        )

    return DemoResult(
        market_key=market_key,
        symbol=symbol,
        strategy_key=strategy_key,
        cost_bps=float(cost_bps),
        split_date=split_date,
        history=history,
        metrics=metrics,
        cost_sensitivity=pd.DataFrame(cost_rows),
        parameter_stability=pd.DataFrame(parameter_rows),
        parameter_label=PARAMETER_LABELS[strategy_key],
    )
