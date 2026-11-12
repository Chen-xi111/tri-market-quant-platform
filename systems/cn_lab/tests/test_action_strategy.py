import unittest

import numpy as np
import pandas as pd

from action_backtest import run_backtest
from action_strategy import build_action_plan, prepare_features


def trending_history(periods: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    index = pd.bdate_range("2025-01-02", periods=periods)
    trend = 10 * np.cumprod(1 + rng.normal(0.0008, 0.009, periods))
    close = pd.Series(trend, index=index)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, periods))
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.008
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.992
    volume = pd.Series(1_000_000 * (1 + rng.uniform(-0.2, 0.3, periods)), index=index)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def prepared_execution_frame() -> pd.DataFrame:
    index = pd.bdate_range("2026-01-05", periods=6)
    frame = pd.DataFrame(
        [
            (10.0, 10.2, 9.8, 10.0, 1_000_000),
            (10.0, 10.2, 9.8, 10.0, 1_000_000),
            (10.0, 10.7, 8.7, 10.2, 1_000_000),
            (10.1, 12.0, 8.5, 10.5, 1_000_000),
            (10.5, 10.8, 10.2, 10.6, 1_000_000),
            (10.6, 10.9, 10.3, 10.7, 1_000_000),
        ],
        index=index,
        columns=["open", "high", "low", "close", "volume"],
    )
    frame["atr14"] = 1.0
    frame["entry_signal"] = False
    frame.loc[index[1], "entry_signal"] = True
    frame["entry_low"] = 9.8
    frame["entry_high"] = 10.2
    frame["stop_loss"] = 9.0
    return frame


class ActionStrategyTests(unittest.TestCase):
    def test_features_before_cutoff_do_not_use_future_rows(self):
        history = trending_history()
        cutoff = 190
        original = prepare_features(history, "overnight").iloc[:cutoff].copy()
        mutated = history.copy()
        mutated.iloc[cutoff:, mutated.columns.get_loc("close")] *= 1.8
        changed = prepare_features(mutated, "overnight").iloc[:cutoff]
        pd.testing.assert_frame_equal(original, changed)

    def test_same_bar_stop_and_target_uses_conservative_stop(self):
        frame = prepared_execution_frame()
        result = run_backtest(
            frame,
            ticker="600019.SS",
            mode="overnight",
            initial_capital=100_000,
            risk_pct=0.01,
            max_position_pct=0.30,
            slippage_bps=0,
            start_position=1,
        )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].entry_date, frame.index[2])
        self.assertEqual(result.trades[0].exit_date, frame.index[3])
        self.assertEqual(result.trades[0].exit_reason, "计划止损")
        self.assertEqual(result.trades[0].exit_price, 9.0)

    def test_t_plus_one_prevents_entry_day_exit(self):
        frame = prepared_execution_frame()
        frame.loc[frame.index[2], ["high", "low"]] = [13.0, 8.0]
        frame.loc[frame.index[3], ["high", "low"]] = [10.8, 9.8]
        result = run_backtest(
            frame,
            ticker="600019.SS",
            mode="today",
            initial_capital=100_000,
            risk_pct=0.01,
            max_position_pct=0.30,
            slippage_bps=0,
            start_position=1,
        )
        self.assertTrue(result.open_position or result.trades)
        if result.trades:
            self.assertGreater(result.trades[0].exit_date, result.trades[0].entry_date)

    def test_plan_quantity_uses_board_lots_and_t_plus_one_sellability(self):
        features = prepare_features(trending_history(), "overnight")
        latest = float(features["close"].iloc[-1])
        flat_plan = build_action_plan(
            features,
            mode="overnight",
            current_price=latest,
            capital=100_000,
            risk_pct=0.01,
            max_position_pct=0.30,
        )
        self.assertEqual(flat_plan.suggested_quantity % 100, 0)

        holding_plan = build_action_plan(
            features,
            mode="overnight",
            current_price=latest,
            capital=100_000,
            risk_pct=0.01,
            max_position_pct=0.30,
            holding_quantity=1000,
            sellable_quantity=0,
            average_cost=latest,
        )
        self.assertEqual(holding_plan.action, "今日不可卖")
        self.assertEqual(holding_plan.sell_quantity, 0)


if __name__ == "__main__":
    unittest.main()
