"""Deterministic, non-sensitive exports for public research evidence."""

from __future__ import annotations

import json

from portfolio.demo_engine import DemoResult, evidence_diagnostics


def _records(frame, include_index: bool = False) -> list[dict]:
    export = frame.reset_index() if include_index else frame
    return json.loads(export.to_json(orient="records", date_format="iso"))


def evidence_json(result: DemoResult, metadata: dict) -> bytes:
    """Return a deterministic JSON evidence bundle for one demo result."""
    payload = {
        "schema_version": 1,
        "instrument": {
            "market": result.market_key,
            "symbol": result.symbol,
            "name": metadata["name"],
            "source": metadata["source"],
            "sample_start": metadata["start"],
            "sample_end": metadata["end"],
            "rows": metadata["rows"],
            "fixture_sha256": metadata["sha256"],
        },
        "research": {
            "strategy": result.strategy_key,
            "one_way_cost_bps": result.cost_bps,
            "split_date": result.split_date.date().isoformat(),
            "signal_lag_bars": 1,
            "position_constraint": "long_or_cash",
        },
        "diagnostics": evidence_diagnostics(result),
        "metrics": _records(result.metrics, include_index=True),
        "cost_sensitivity": _records(result.cost_sensitivity),
        "parameter_stability": _records(result.parameter_stability),
        "disclaimer": (
            "Descriptive evidence from a fixed public-provider snapshot; "
            "not investment advice or a live-trading authorization."
        ),
    }
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def history_csv(result: DemoResult) -> bytes:
    """Return the complete auditable return path as UTF-8 CSV bytes."""
    return result.history.reset_index().to_csv(index=False).encode("utf-8")


def export_stem(result: DemoResult) -> str:
    symbol = result.symbol.replace(".", "_").lower()
    cost = f"{result.cost_bps:g}".replace(".", "p")
    return f"{result.market_key}_{symbol}_{result.strategy_key}_{cost}bps"


def comparison_export_stem(
    strategy_key: str,
    cost_basis: str,
    common_cost_bps: float,
    market_keys: list[str],
) -> str:
    """Build an auditable file stem for a filtered cross-market scorecard."""
    if cost_basis not in {"venue", "common"}:
        raise ValueError("cost_basis must be 'venue' or 'common'")
    if not market_keys:
        raise ValueError("market_keys must not be empty")

    if cost_basis == "venue":
        cost = "venue-defaults"
    else:
        cost_value = f"{common_cost_bps:g}".replace(".", "p")
        cost = f"common-{cost_value}bps"
    markets = "-".join(market_keys)
    return f"cross_market_{strategy_key}_{cost}_{markets}"
