import unittest

import numpy as np
import pandas as pd

from indicators import compute_all
from stock_ranking import _strategy_fit, rank_stocks


def sample_frame(
    start: float,
    end: float,
    volume: float,
    rows: int = 260,
    noise: float = 0.0,
) -> pd.DataFrame:
    index = pd.bdate_range("2025-01-02", periods=rows)
    close = np.linspace(start, end, rows)
    if noise:
        close = close + np.sin(np.arange(rows) * 1.7) * noise
    close = pd.Series(close, index=index).clip(lower=1)
    return compute_all(pd.DataFrame({
        "open": close * 0.998,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
    }, index=index))


class StockRankingTests(unittest.TestCase):
    def test_trend_strategy_prioritizes_strong_liquid_trend(self):
        histories = {
            "UP": sample_frame(20, 55, 8_000_000),
            "FLAT": sample_frame(30, 31, 3_000_000),
            "DOWN": sample_frame(50, 28, 5_000_000),
        }

        ranking = rank_stocks(histories, "trend_rider")

        self.assertEqual(ranking.iloc[0]["ticker"], "UP")
        self.assertEqual(list(ranking["rank"]), [1, 2, 3])
        self.assertTrue(ranking["total_score"].between(0, 100).all())

    def test_strategy_fit_changes_with_selected_strategy(self):
        trend_candidate = {
            "trend_raw": 0.20, "momentum_raw": 0.15, "stability_raw": -0.20,
            "distance_ma20": 0.01, "rsi": 62, "boll_pct": 0.75,
            "breakout_20": 0.02, "volume_ratio": 1.1,
            "return_20": 0.10, "return_60": 0.25,
        }
        reversion_candidate = {
            "trend_raw": 0.01, "momentum_raw": -0.03, "stability_raw": -0.18,
            "distance_ma20": -0.04, "rsi": 38, "boll_pct": 0.15,
            "breakout_20": -0.08, "volume_ratio": 0.9,
            "return_20": -0.03, "return_60": 0.02,
        }

        self.assertGreater(
            _strategy_fit("trend_rider", trend_candidate),
            _strategy_fit("trend_rider", reversion_candidate),
        )
        self.assertGreater(
            _strategy_fit("bollinger_reversion", reversion_candidate),
            _strategy_fit("bollinger_reversion", trend_candidate),
        )

    def test_short_history_is_excluded_without_breaking_other_scores(self):
        histories = {
            "GOOD": sample_frame(20, 40, 5_000_000),
            "SHORT": sample_frame(10, 12, 1_000_000, rows=30),
        }

        ranking = rank_stocks(histories, "alpha_factors", {"GOOD": "Good Co"})

        self.assertEqual(list(ranking["ticker"]), ["GOOD"])
        self.assertEqual(ranking.iloc[0]["name"], "Good Co")
        self.assertEqual(ranking.iloc[0]["total_score"], 50.0)

    def test_more_stable_stock_receives_higher_risk_quality_score(self):
        histories = {
            "STABLE": sample_frame(20, 28, 2_000_000, noise=0.05),
            "VOLATILE": sample_frame(20, 28, 2_000_000, noise=4.0),
        }

        ranking = rank_stocks(histories, "ml_predict").set_index("ticker")

        self.assertGreater(
            ranking.loc["STABLE", "risk_score"],
            ranking.loc["VOLATILE", "risk_score"],
        )


if __name__ == "__main__":
    unittest.main()
