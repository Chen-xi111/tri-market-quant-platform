import unittest
from unittest.mock import patch

import pandas as pd

import config as cfg
from data_sources.yfinance_source import YFinanceSource


def chart_payload():
    return {
        "chart": {
            "error": None,
            "result": [{
                "meta": {
                    "currency": "USD",
                    "exchangeTimezoneName": "America/New_York",
                    "regularMarketPrice": 33.0,
                    "chartPreviousClose": 31.0,
                    "regularMarketTime": 1_777_000_000,
                },
                "timestamp": [1_776_000_000, 1_776_086_400, 1_776_172_800],
                "indicators": {
                    "quote": [{
                        "open": [20.0, 30.0, 32.0],
                        "high": [22.0, 32.0, 34.0],
                        "low": [19.0, 29.0, 31.0],
                        "close": [21.0, 31.0, 33.0],
                        "volume": [1000, 1100, 1200],
                    }],
                    "adjclose": [{"adjclose": [10.5, 31.0, 33.0]}],
                },
            }],
        }
    }


class YahooChartFallbackTests(unittest.TestCase):
    def test_intc_is_known_nasdaq_symbol(self):
        self.assertEqual(cfg.PRESET_STOCKS["INTC"], "英特尔")
        self.assertEqual(cfg.ibkr_primary_exchange("INTC"), "NASDAQ")

    def test_empty_yfinance_history_falls_back_to_chart_api(self):
        source = YFinanceSource("INTC")
        with patch.object(
            source, "_yfinance_history", return_value=pd.DataFrame()
        ), patch.object(source, "_fetch_chart", return_value=chart_payload()):
            frame = source.get_history(days=365)

        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.attrs["market_data_transport"], "yahoo_chart_api")
        # 第一天复权因子为 0.5，OHLC 必须一起复权。
        self.assertAlmostEqual(frame["close"].iloc[0], 10.5)
        self.assertAlmostEqual(frame["open"].iloc[0], 10.0)

    def test_chart_quote_remains_available_when_fast_info_fails(self):
        source = YFinanceSource("INTC")
        with patch.object(source, "_fetch_chart", return_value=chart_payload()):
            quote = source._quote_from_chart()

        self.assertEqual(quote["ticker"], "INTC")
        self.assertEqual(quote["price"], 33.0)
        self.assertEqual(quote["source"], "Yahoo Finance Chart")


if __name__ == "__main__":
    unittest.main()
