import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import config as cfg
import market_data
from execution.auto_trader import AutoTrader
from execution.base import OrderSide
from market_data import (
    fetch_history,
    fetch_quote,
    history_source_order,
    quote_from_history,
    source_order,
)
from strategies.base import Signal, SignalType
from trade_plan import build_trade_plan


def sample_history(rows: int = 90) -> pd.DataFrame:
    index = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=rows)
    close = pd.Series(range(rows), index=index, dtype=float) / 10 + 10
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


class EmptySource:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_history(self, period, days):
        return pd.DataFrame()

    def get_realtime(self):
        return {}


class GoodSource:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_history(self, period, days):
        return sample_history()

    def get_realtime(self):
        return {"ticker": self.ticker, "price": 10.5, "source": "good"}


class ExplodingSource:
    def __init__(self, ticker):
        self.ticker = ticker

    def get_history(self, period, days):
        raise RuntimeError("source unavailable")

    def get_realtime(self):
        raise RuntimeError("source unavailable")


SOURCES = {
    "eastmoney": ("东方财富", EmptySource, ""),
    "yfinance": ("Yahoo", GoodSource, ""),
    "akshare": ("AKShare", ExplodingSource, ""),
}


class TickerNormalizationTests(unittest.TestCase):
    def test_normalizes_supported_us_formats(self):
        self.assertEqual(
            cfg.normalize_ticker("aapl"),
            ("AAPL", "AAPL", "AAPL"),
        )
        self.assertEqual(
            cfg.normalize_ticker("US.NVDA"),
            ("NVDA", "NVDA", "NVDA"),
        )
        self.assertEqual(cfg.normalize_ticker("ORCL.US"), ("ORCL", "ORCL", "ORCL"))

    def test_rejects_invalid_codes_before_network_calls(self):
        for value in ("", "1234", "$AAPL", "AAPL!", "TOO-LONG-SYMBOL"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cfg.normalize_ticker(value)


class MarketDataSwitchingTests(unittest.TestCase):
    def test_slow_akshare_is_not_an_automatic_fallback(self):
        self.assertEqual(
            source_order("eastmoney", SOURCES),
            ["eastmoney", "yfinance"],
        )
        self.assertEqual(
            source_order("akshare", SOURCES),
            ["akshare", "eastmoney", "yfinance"],
        )
        self.assertEqual(
            history_source_order("eastmoney", SOURCES),
            ["yfinance", "eastmoney"],
        )

    def test_history_falls_back_and_reports_the_failed_source(self):
        frame, actual_source, failures = fetch_history(
            "akshare", "AAPL", 365, SOURCES
        )

        self.assertEqual(actual_source, "yfinance")
        self.assertIsNotNone(frame)
        self.assertIn("RSI", frame.columns)
        self.assertEqual(len(failures), 2)
        self.assertIn("source unavailable", failures[0])

    def test_quote_is_loaded_independently_from_history(self):
        quote, actual_source, failures = fetch_quote(
            "eastmoney", "NVDA", SOURCES
        )
        self.assertEqual(actual_source, "yfinance")
        self.assertEqual(quote["ticker"], "NVDA")
        self.assertTrue(failures)

    def test_quote_fallback_keeps_the_selected_ticker(self):
        quote = quote_from_history(sample_history(), "ORCL", "Yahoo")
        self.assertEqual(quote["ticker"], "ORCL")
        self.assertEqual(quote["price"], sample_history()["close"].iloc[-1])
        self.assertIn("历史回退", quote["source"])

    def test_last_good_market_data_cache_round_trip(self):
        with TemporaryDirectory() as temp_dir:
            original_cache_dir = market_data.CACHE_DIR
            market_data.CACHE_DIR = Path(temp_dir)
            try:
                frame = sample_history()
                market_data._save_history_cache("NVDA", frame, "yfinance")
                market_data._save_quote_cache(
                    "NVDA", {"price": 182.62, "source": "Yahoo"}, "yfinance"
                )

                cached_frame, cached_history_source = market_data._load_history_cache("NVDA")
                cached_quote, cached_quote_source = market_data._load_quote_cache("NVDA")

                self.assertEqual(cached_history_source, "yfinance")
                self.assertTrue(cached_frame.attrs["market_data_cached"])
                self.assertEqual(cached_quote_source, "yfinance")
                self.assertTrue(cached_quote["_cached"])
                self.assertEqual(cached_quote["price"], 182.62)
            finally:
                market_data.CACHE_DIR = original_cache_dir


class PaperTradingSafetyTests(unittest.TestCase):
    def test_zero_position_pct_sell_signal_closes_current_symbol(self):
        trader = AutoTrader.paper(100_000, symbol="AAPL")
        trader.broker.place_order("AAPL", 10.0, 100, OrderSide.BUY)
        signal = Signal(
            type=SignalType.SELL,
            price=11.0,
            date=pd.Timestamp("2026-08-10"),
            position_pct=0.0,
        )

        result = trader.process_signal(signal)

        self.assertTrue(result.success)
        self.assertEqual(trader.broker.get_positions(), [])

    def test_manual_orders_cannot_bypass_position_limit(self):
        trader = AutoTrader.paper(100_000, symbol="AAPL")
        result = trader.place_manual_order(10.0, 4_000, OrderSide.BUY)

        self.assertFalse(result.success)
        self.assertIn("上限", result.message)
        self.assertFalse(trader.is_paused)

    def test_paper_fill_includes_slippage_and_fees(self):
        trader = AutoTrader.paper(100_000, symbol="AAPL")
        result = trader.place_manual_order(10.0, 100, OrderSide.BUY)

        self.assertTrue(result.success)
        self.assertGreater(result.order.price, 10.0)
        self.assertGreater(trader.broker.get_account()["fees_paid"], 0)


class TradePlanTests(unittest.TestCase):
    def test_actionable_plan_uses_whole_lots_and_risk_budget(self):
        frame, _, _ = fetch_history("yfinance", "ORCL", 365, SOURCES)
        composite = SimpleNamespace(
            direction="buy", confidence=0.5, recommended_position=0.3,
        )

        plan = build_trade_plan(frame, composite, 100_000, 0.02, 0.3, 1)

        self.assertEqual(plan["shares"] % 1, 0)
        self.assertLessEqual(plan["planned_value"], 30_000)
        self.assertLessEqual(plan["planned_risk"], 2_000)

    def test_neutral_consensus_never_generates_buy_quantity(self):
        frame, _, _ = fetch_history("yfinance", "ORCL", 365, SOURCES)
        composite = SimpleNamespace(
            direction="neutral", confidence=0.8, recommended_position=0.3,
        )

        plan = build_trade_plan(frame, composite, 100_000, 0.02, 0.3, 1)

        self.assertFalse(plan["actionable"])
        self.assertEqual(plan["shares"], 0)


if __name__ == "__main__":
    unittest.main()
