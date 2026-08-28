import unittest

import numpy as np
import pandas as pd

from quant_research import QuantResearchEngine


def regime_history(periods: int = 560) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    regime = np.where((np.arange(periods) // 35) % 2 == 0, 0.005, -0.0035)
    daily_return = regime + rng.normal(0, 0.004, periods)
    close = 20 * np.cumprod(1 + daily_return)
    index = pd.bdate_range("2024-01-02", periods=periods)
    close = pd.Series(close, index=index)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, periods)),
            "high": close * (1.006 + rng.uniform(0, 0.003, periods)),
            "low": close * (0.994 - rng.uniform(0, 0.003, periods)),
            "close": close,
            "volume": 1_000_000 * (1 + np.abs(daily_return) * 20),
        },
        index=index,
    )


class QuantResearchTests(unittest.TestCase):
    def test_small_sample_is_never_allowed_to_influence_recommendation(self):
        result = QuantResearchEngine().evaluate(regime_history(180))
        self.assertEqual(result.status, "样本不足")
        self.assertFalse(result.gate_pass)

    def test_features_do_not_change_when_only_future_rows_change(self):
        engine = QuantResearchEngine()
        history = regime_history()
        cutoff = 360
        original = engine.build_features(history).iloc[:cutoff].copy()
        mutated = history.copy()
        mutated.iloc[cutoff:, mutated.columns.get_loc("close")] *= 1.8
        changed = engine.build_features(mutated).iloc[:cutoff]
        pd.testing.assert_frame_equal(original, changed)

    def test_forward_labels_leave_the_last_horizon_unlabeled(self):
        engine = QuantResearchEngine(horizon=10)
        history = regime_history()
        labeled, current = engine._research_frame(history)
        self.assertEqual(current.index[-1], history.index[-1])
        self.assertLessEqual(labeled.index[-1], history.index[-11])

    def test_walk_forward_result_is_deterministic_and_has_purged_oos_data(self):
        engine = QuantResearchEngine(bootstrap_samples=300)
        first = engine.evaluate(regime_history())
        second = engine.evaluate(regime_history())
        self.assertNotEqual(first.status, "样本不足")
        self.assertGreaterEqual(first.fold_count, 4)
        self.assertGreaterEqual(first.oos_observations, 120)
        self.assertEqual(first, second)
        self.assertAlmostEqual(engine.round_trip_cost * 10_000, 21.2, places=1)


if __name__ == "__main__":
    unittest.main()
