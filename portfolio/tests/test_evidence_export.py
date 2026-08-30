from __future__ import annotations

import json

from portfolio.demo_engine import load_demo_manifest, run_demo_research
from portfolio.evidence_export import evidence_json, export_stem, history_csv


def test_evidence_exports_are_deterministic_and_complete():
    result = run_demo_research("us", "SPY", "trend", 7)
    metadata = load_demo_manifest()["markets"]["us"]["symbols"]["SPY"]

    first = evidence_json(result, metadata)
    second = evidence_json(result, metadata)
    payload = json.loads(first)
    csv_lines = history_csv(result).decode("utf-8").splitlines()

    assert first == second
    assert payload["instrument"]["fixture_sha256"] == metadata["sha256"]
    assert payload["research"]["signal_lag_bars"] == 1
    assert len(payload["metrics"]) == 2
    assert len(payload["parameter_stability"]) == 5
    assert len(csv_lines) == len(result.history) + 1
    assert export_stem(result) == "us_spy_trend"
