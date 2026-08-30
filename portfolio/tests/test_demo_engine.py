from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from portfolio.demo_engine import (
    DEMO_DATA_DIR,
    STRATEGIES,
    evidence_diagnostics,
    load_demo_manifest,
    load_demo_prices,
    run_demo_research,
    strategy_position,
)


@pytest.mark.parametrize("market_key", ["cn", "hk", "us"])
def test_demo_manifest_has_three_clean_instruments_per_market(market_key):
    manifest = load_demo_manifest()
    symbols = manifest["markets"][market_key]["symbols"]

    assert len(symbols) == 3
    for symbol, metadata in symbols.items():
        prices = load_demo_prices(market_key, symbol)
        assert len(prices) == metadata["rows"]
        assert len(prices) >= 300
        assert prices.index.is_monotonic_increasing
        assert not prices[["open", "high", "low", "close", "volume"]].isna().any().any()
        fixture = DEMO_DATA_DIR / metadata["file"]
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == metadata["sha256"]


@pytest.mark.parametrize("strategy_key", list(STRATEGIES))
def test_demo_strategies_are_lagged_long_only(strategy_key):
    prices = load_demo_prices("us", "SPY")
    position = strategy_position(prices, strategy_key)

    assert set(position.unique()).issubset({0.0, 1.0})
    assert position.index.equals(prices.index)
    assert position.iloc[0] == 0.0


def test_higher_costs_do_not_improve_fixed_strategy_return():
    result = run_demo_research("hk", "0700.HK", "mean_reversion", 15)
    returns = result.cost_sensitivity.sort_values("cost_bps")["validation_return"]

    assert returns.is_monotonic_decreasing


def test_cached_price_fixtures_return_independent_frames():
    first = load_demo_prices("us", "SPY")
    original = first.iloc[0]["close"]
    first.iloc[0, first.columns.get_loc("close")] = -1

    second = load_demo_prices("us", "SPY")
    assert second.iloc[0]["close"] == original


def test_research_split_and_robustness_outputs_are_complete():
    result = run_demo_research("cn", "600019.SS", "trend", 12)

    assert result.history.index.min() < result.split_date < result.history.index.max()
    assert list(result.metrics.index) == ["Train", "Validation"]
    assert len(result.parameter_stability) == 5
    assert len(result.cost_sensitivity) == 6
    assert pd.notna(result.metrics.to_numpy()).all()


def test_evidence_diagnostics_are_bounded_and_descriptive():
    result = run_demo_research("us", "SPY", "trend", 10)
    diagnostics = evidence_diagnostics(result)

    assert diagnostics["classification"] in {
        "Consistent in this snapshot",
        "Mixed in this snapshot",
        "Weak in this snapshot",
    }
    assert 0 <= diagnostics["positive_settings"] <= diagnostics["tested_settings"]
    assert diagnostics["tested_settings"] == len(result.parameter_stability)
    assert isinstance(diagnostics["validation_alpha"], float)
