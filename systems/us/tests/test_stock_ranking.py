import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from indicators import compute_all
from stock_ranking import _strategy_fit, build_execution_setup, rank_stocks
from trade_plan import build_trade_plan


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
    def test_trend_strategy_never_promotes_downtrend_above_valid_candidates(self):
        histories = {
            "UP": sample_frame(20, 55, 8_000_000),
            "FLAT": sample_frame(30, 31, 3_000_000),
            "DOWN": sample_frame(50, 28, 5_000_000),
        }

        ranking = rank_stocks(histories, "trend_rider")

        self.assertIn(ranking.iloc[0]["ticker"], {"UP", "FLAT"})
        self.assertEqual(ranking.iloc[-1]["ticker"], "DOWN")
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
        self.assertTrue(0 <= ranking.iloc[0]["total_score"] <= 100)
        self.assertTrue(ranking.iloc[0]["guardrail_pass"])

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

    def test_low_liquidity_candidate_is_capped_and_has_no_position(self):
        ranking = rank_stocks(
            {
                "LIQUID": sample_frame(20, 28, 5_000_000),
                "THIN": sample_frame(20, 32, 100),
            },
            "alpha_factors",
            capital=100_000,
            risk_pct=0.01,
            max_position_pct=0.30,
        ).set_index("ticker")

        self.assertFalse(ranking.loc["THIN", "guardrail_pass"])
        self.assertEqual(ranking.loc["THIN", "execution_status"], "暂缓")
        self.assertEqual(ranking.loc["THIN", "planned_shares"], 0)
        self.assertLessEqual(ranking.loc["THIN", "total_score"], 44)
        self.assertIn("流动性", ranking.loc["THIN", "failed_checks"])

    def test_stale_history_is_not_actionable(self):
        ranking = rank_stocks(
            {"OLD": sample_frame(20, 28, 5_000_000)},
            "trend_rider",
            reference_date="2027-01-15",
        ).iloc[0]

        self.assertFalse(ranking["guardrail_pass"])
        self.assertIn("数据新鲜", ranking["failed_checks"])
        self.assertEqual(ranking["planned_shares"], 0)

    def test_position_plan_respects_risk_and_value_limits(self):
        capital = 100_000
        ranking = rank_stocks(
            {"PLAN": sample_frame(20, 28, 5_000_000)},
            "alpha_factors",
            capital=capital,
            risk_pct=0.01,
            max_position_pct=0.20,
        ).iloc[0]

        self.assertLessEqual(ranking["planned_risk"], capital * 0.01 + 0.01)
        self.assertLessEqual(ranking["planned_value"], capital * 0.20 + 0.01)

    def test_trend_setup_marks_extended_price_as_avoid(self):
        setup = build_execution_setup({
            "price": 130, "atr": 5, "ma20": 100, "ma60": 90,
            "ma20_slope_5": 0.02, "support_20": 95,
            "resistance_20": 135, "resistance_60": 140,
            "rsi": 70, "volume_ratio": 1.2, "latest_high": 132,
            "return_20": 0.20, "return_60": 0.35,
        }, "trend_rider")

        self.assertEqual(setup["setup_state"], "avoid")
        self.assertIn("不追价", setup["setup_reason"])

    def test_trade_plan_requires_selected_strategy_to_be_ready(self):
        frame = sample_frame(20, 28, 5_000_000)
        last = frame.index[-1]
        frame.loc[last, ["open", "high", "low", "close"]] = [39, 41, 38, 40]
        composite = SimpleNamespace(
            direction="buy", confidence=0.8, recommended_position=0.30,
        )

        plan = build_trade_plan(
            frame, composite, 100_000, 0.01, 0.30, 1,
            strategy_key="trend_rider",
        )

        self.assertFalse(plan["checks"]["当前策略到达入场条件"])
        self.assertFalse(plan["actionable"])
        self.assertEqual(plan["shares"], 0)


if __name__ == "__main__":
    unittest.main()
