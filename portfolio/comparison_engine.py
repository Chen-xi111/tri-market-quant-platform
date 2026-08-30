"""Cross-market comparison helpers for the fixed public research snapshots."""

from __future__ import annotations

import pandas as pd

from portfolio.demo_engine import (
    STRATEGIES,
    evidence_diagnostics,
    load_demo_manifest,
    run_demo_research,
)

MARKET_ORDER = ("cn", "hk", "us")


def build_cross_market_comparison(
    strategy_key: str,
    cost_basis: str = "venue",
    common_cost_bps: float = 10,
) -> pd.DataFrame:
    """Apply one rule to every fixture and return comparable held-out evidence."""
    if strategy_key not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {strategy_key}")
    if cost_basis not in {"venue", "common"}:
        raise ValueError("cost_basis must be 'venue' or 'common'")
    if common_cost_bps < 0:
        raise ValueError("common_cost_bps must be non-negative")

    manifest = load_demo_manifest()
    rows: list[dict] = []
    for market_key in MARKET_ORDER:
        market = manifest["markets"][market_key]
        cost_bps = (
            float(market["default_cost_bps"]) if cost_basis == "venue" else float(common_cost_bps)
        )
        for symbol, metadata in market["symbols"].items():
            result = run_demo_research(market_key, symbol, strategy_key, cost_bps)
            train = result.metrics.loc["Train"]
            validation = result.metrics.loc["Validation"]
            diagnostics = evidence_diagnostics(result)
            rows.append(
                {
                    "market_key": market_key,
                    "market": market["label"],
                    "currency": market["currency"],
                    "symbol": symbol,
                    "name": metadata["name"],
                    "cost_bps": cost_bps,
                    "sample_start": metadata["start"],
                    "sample_end": metadata["end"],
                    "train_return": train["strategy_return"],
                    "validation_return": validation["strategy_return"],
                    "benchmark_return": validation["benchmark_return"],
                    "excess_return": diagnostics["validation_alpha"],
                    "train_sharpe": train["sharpe"],
                    "validation_sharpe": validation["sharpe"],
                    "sharpe_decay": diagnostics["sharpe_decay"],
                    "max_drawdown": validation["max_drawdown"],
                    "exposure": validation["exposure"],
                    "entries": int(validation["trades"]),
                    "positive_settings": diagnostics["positive_settings"],
                    "tested_settings": diagnostics["tested_settings"],
                    "evidence": diagnostics["classification"],
                }
            )
    return pd.DataFrame(rows)
