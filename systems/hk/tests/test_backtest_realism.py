import unittest

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics
from strategies.base import Strategy


class OneEntryStrategy(Strategy):
    name = "one_entry"

    def __init__(self, stop=90.0, target=110.0, position_pct=1.0):
        super().__init__({})
        self.stop = stop
        self.target = target
        self.position_pct = position_pct

    def generate_signals(self):
        df = self._df.copy()
        df["signal"] = 0
        df["signal_price"] = df["close"]
        df["signal_reason"] = ""
        df["position_pct"] = 0.0
        df["stop_loss"] = float("nan")
        df["take_profit"] = float("nan")
        df.iloc[1, df.columns.get_loc("signal")] = 1
        df.iloc[1, df.columns.get_loc("position_pct")] = self.position_pct
        df.iloc[1, df.columns.get_loc("stop_loss")] = self.stop
        df.iloc[1, df.columns.get_loc("take_profit")] = self.target
        self._df = df
        return df


def price_frame(rows):
    return pd.DataFrame(
        rows,
        index=pd.bdate_range("2026-01-01", periods=len(rows)),
        columns=["open", "high", "low", "close", "volume"],
    )


class BacktestExecutionTests(unittest.TestCase):
    def test_take_profit_is_persisted_after_entry_bar(self):
        frame = price_frame([
            (100, 101, 99, 100, 1000),
            (100, 101, 99, 100, 1000),
            (100, 102, 96, 100, 1000),
            (100, 111, 99, 110, 1000),
        ])
        engine = BacktestEngine(
            OneEntryStrategy(), initial_capital=10_000,
            commission=0, slippage_bps=0, max_position_pct=1,
            risk_pct=1, lot_size=1,
        )

        result = engine.run(frame)

        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0][3], 110.0)
        self.assertEqual(result["trades"][0][6], "目标止盈")

    def test_trailing_stop_does_not_activate_before_profit_threshold(self):
        frame = price_frame([
            (100, 101, 99, 100, 1000),
            (100, 101, 99, 100, 1000),
            (100, 102, 96, 100, 1000),
        ])
        engine = BacktestEngine(
            OneEntryStrategy(stop=90, target=120), initial_capital=10_000,
            commission=0, slippage_bps=0, max_position_pct=1,
            risk_pct=1, lot_size=1,
        )

        result = engine.run(frame)

        self.assertEqual(result["trades"][0][6], "回测结束平仓")

    def test_position_respects_risk_budget_and_board_lot(self):
        frame = price_frame([
            (10, 10.2, 9.8, 10, 1000),
            (10, 10.2, 9.8, 10, 1000),
            (10, 10.2, 8.0, 9, 1000),
        ])
        engine = BacktestEngine(
            OneEntryStrategy(stop=9, target=12), initial_capital=10_000,
            commission=0, minimum_commission=0, stamp_duty_rate=0,
            sfc_levy_rate=0, afrc_levy_rate=0, trading_fee_rate=0,
            slippage_bps=0, max_position_pct=0.5,
            risk_pct=0.02, lot_size=100,
        )

        result = engine.run(frame)

        entry, exit_ = result["trades"][0][2:4]
        pnl = result["trades"][0][4]
        self.assertEqual(entry, 10.0)
        self.assertEqual(exit_, 9.0)
        self.assertEqual(pnl, -200.0)
        self.assertEqual(result["metrics"]["total_trades"], 1)

    def test_signal_is_filled_at_next_open(self):
        frame = price_frame([
            (100, 101, 99, 100, 1000),
            (100, 101, 99, 100, 1000),
            (105, 106, 103, 104, 1000),
        ])
        engine = BacktestEngine(
            OneEntryStrategy(stop=90, target=120), initial_capital=10_000,
            commission=0, minimum_commission=0, stamp_duty_rate=0,
            sfc_levy_rate=0, afrc_levy_rate=0, trading_fee_rate=0,
            slippage_bps=0, max_position_pct=1, risk_pct=1, lot_size=1,
        )

        result = engine.run(frame)

        self.assertEqual(result["trades"][0][2], 105.0)
        self.assertEqual(result["execution_assumptions"]["execution_price"], "next_open")

    def test_trade_percent_is_not_scaled_twice(self):
        equity = pd.Series(
            [10_000.0, 10_100.0],
            index=pd.bdate_range("2026-01-01", periods=2),
        )
        trades = [(equity.index[0], equity.index[1], 100, 101, 100, 1.0, "test")]

        metrics = compute_metrics(equity, trades, risk_free_rate=0)

        self.assertEqual(metrics["avg_trade_pnl_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
