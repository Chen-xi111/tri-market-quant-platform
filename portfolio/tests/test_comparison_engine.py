from __future__ import annotations

import numpy as np

from portfolio.comparison_engine import build_cross_market_comparison


def test_venue_comparison_covers_all_nine_snapshots():
    frame = build_cross_market_comparison("trend", "venue")

    assert len(frame) == 9
    assert set(frame["market_key"]) == {"cn", "hk", "us"}
    assert set(frame["cost_bps"]) == {7.0, 12.0, 15.0}
    assert frame.groupby("market_key")["symbol"].nunique().to_dict() == {
        "cn": 3,
        "hk": 3,
        "us": 3,
    }


def test_common_cost_comparison_is_complete_and_finite():
    frame = build_cross_market_comparison("breakout", "common", 25)
    metric_columns = [
        "validation_return",
        "excess_return",
        "validation_sharpe",
        "max_drawdown",
    ]

    assert set(frame["cost_bps"]) == {25.0}
    assert np.isfinite(frame[metric_columns].to_numpy()).all()
    assert (frame["positive_settings"] <= frame["tested_settings"]).all()
