import unittest

import pandas as pd

from strategies.base import Signal, SignalType
from strategies.composite import CompositeStrategy


def history() -> pd.DataFrame:
    index = pd.bdate_range(end="2026-08-21", periods=70)
    close = pd.Series(10.0, index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


class FakeStrategy:
    def __init__(self, signal: Signal):
        self.signal = signal
        self.frame = None

    def set_data(self, frame):
        self.frame = frame

    def generate_signals(self):
        return self.frame

    def get_signals_list(self):
        return [self.signal]


class CompositeSignalValidityTests(unittest.TestCase):
    def test_buy_signal_below_its_stop_no_longer_contributes_buy_score(self):
        frame = history()
        signal = Signal(
            type=SignalType.BUY,
            price=11.0,
            date=frame.index[-2],
            stop_loss=10.5,
            take_profit=13.0,
            reason="test buy",
        )
        strategy = CompositeStrategy()
        strategy._strategies = {"fake": (FakeStrategy(signal), 2.0, "伪策略")}

        result = strategy.evaluate(frame)

        self.assertEqual(result.buy_votes, 0)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.strategy_details[0]["方向"], "invalidated")
        self.assertIn("止损", result.strategy_details[0]["原因"])

    def test_older_valid_signal_has_decayed_weight(self):
        frame = history()
        signal = Signal(
            type=SignalType.BUY,
            price=9.5,
            date=frame.index[-4],
            stop_loss=9.0,
            take_profit=12.0,
            reason="test buy",
        )
        strategy = CompositeStrategy()
        strategy._strategies = {"fake": (FakeStrategy(signal), 2.0, "伪策略")}

        result = strategy.evaluate(frame)

        self.assertEqual(result.buy_votes, 1)
        self.assertEqual(result.score, 5.5)
        self.assertIn("权重55%", result.strategy_details[0]["原因"])


if __name__ == "__main__":
    unittest.main()
