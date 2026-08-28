import unittest

import numpy as np
import pandas as pd

from backtest.validation import assess_readiness, robustness_score, scan_grid
from backtest.research import (
    adjusted_selection_confidence,
    block_bootstrap_annual_return_ci,
    expanding_walk_forward_splits,
    probabilistic_sharpe_confidence,
)


class BacktestValidationTests(unittest.TestCase):
    def test_readiness_requires_multiple_independent_checks(self):
        result = assess_readiness({
            "total_trades": 8,
            "alpha": 4.0,
            "sharpe_ratio": 0.8,
            "max_drawdown": -12.0,
            "profit_factor": 1.4,
        })
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["color"], "green")

    def test_robustness_penalizes_in_sample_collapse(self):
        stable = robustness_score(
            {"total_return": 10},
            {"total_return": 8, "sharpe_ratio": 1.0, "total_trades": 4},
        )
        collapsed = robustness_score(
            {"total_return": 30},
            {"total_return": -5, "sharpe_ratio": -0.2, "total_trades": 1},
        )
        self.assertGreater(stable, collapsed)

    def test_every_scan_parameter_is_strategy_specific(self):
        self.assertEqual(scan_grid("alpha_factors")[0], "entry_percentile")
        self.assertEqual(scan_grid("ml_predict")[0], "prob_threshold")

    def test_walk_forward_folds_are_ordered_and_embargoed(self):
        folds = expanding_walk_forward_splits(420, n_splits=4, gap_bars=2)
        self.assertEqual(len(folds), 4)
        self.assertTrue(all(fold.train_end + 2 == fold.test_start for fold in folds))
        self.assertTrue(all(left.test_end <= right.test_start for left, right in zip(folds, folds[1:])))

    def test_multiple_trials_reduce_selection_confidence(self):
        self.assertLess(adjusted_selection_confidence(0.98, 5), 0.98)

    def test_probabilistic_sharpe_and_block_bootstrap_are_deterministic(self):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.003, 0.01, 160))
        self.assertGreater(probabilistic_sharpe_confidence(returns), 0.5)
        self.assertEqual(
            block_bootstrap_annual_return_ci(returns, samples=200, random_state=9),
            block_bootstrap_annual_return_ci(returns, samples=200, random_state=9),
        )


if __name__ == "__main__":
    unittest.main()
