import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from data_sources.eastmoney_source import EastMoneyUSSource
from data_sources.futu_source import FutuOpenDSource
from quote_freshness import (
    freshness_label,
    merge_live_quote_into_history,
    quote_age_seconds,
    quote_time_text,
)


class QuoteFreshnessTests(unittest.TestCase):
    def test_market_timestamp_is_preferred_over_receive_timestamp(self):
        quote = {
            "_market_time": "2026-08-10T10:00:00-04:00",
            "_received_at": "2026-08-10T14:00:30+00:00",
        }
        now = datetime(2026, 8, 10, 14, 1, tzinfo=timezone.utc)
        self.assertEqual(quote_age_seconds(quote, now=now), 60)
        self.assertIn("10:00:00", quote_time_text(quote))

    def test_delayed_and_yahoo_best_effort_are_not_badged_as_live(self):
        self.assertEqual(
            freshness_label({"price": 1, "_market_data_type": "delayed"}, True),
            ("延迟行情", "orange"),
        )
        self.assertEqual(
            freshness_label({"price": 1, "_market_data_type": "best_effort"}, True),
            ("源端延迟未知", "orange"),
        )

    def test_live_quote_updates_current_daily_bar_without_mutating_history(self):
        frame = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [102.0] * 30,
                "low": [99.0] * 30,
                "close": [101.0] * 30,
                "volume": [1000.0] * 30,
            },
            index=pd.bdate_range(
                end="2026-08-07", periods=30, tz="America/New_York",
            ),
        )
        quote = {
            "price": 110.0, "open": 105.0, "high": 112.0, "low": 104.0,
            "volume": 5000, "_market_time": "2026-08-10T15:30:00+00:00",
            "_market_data_type": "public_snapshot",
        }
        merged = merge_live_quote_into_history(frame, quote)
        self.assertEqual(len(merged), len(frame) + 1)
        self.assertEqual(merged.iloc[-1]["close"], 110.0)
        self.assertEqual(frame.iloc[-1]["close"], 101.0)
        self.assertTrue(merged.attrs["live_bar_merged"])
        self.assertIsInstance(merged.index, pd.DatetimeIndex)
        self.assertIsNotNone(merged.index.tz)

    def test_eastmoney_parses_us_exchange_and_source_timestamp(self):
        payload = {
            "data": {
                "f43": 309.35, "f44": 312.38, "f45": 307.01,
                "f46": 312.05, "f47": 46876815, "f48": 1.45e10,
                "f57": "AAPL", "f58": "苹果", "f60": 311.3,
                "f86": 1787342400, "f169": -1.95, "f170": -0.63,
            }
        }
        with patch.object(EastMoneyUSSource, "_fetch_json", return_value=payload):
            quote = EastMoneyUSSource("AAPL").get_realtime()
        self.assertEqual(quote["_secid"], "105.AAPL")
        self.assertEqual(quote["price"], 309.35)
        self.assertEqual(quote["_market_time"], "2026-08-21T20:00:00+00:00")

    def test_futu_fails_fast_when_opend_is_not_running(self):
        source = FutuOpenDSource("AAPL")
        with patch.object(source, "opend_is_listening", return_value=False):
            result = source.get_realtime()
        self.assertIn("11111", result["error"])

    def test_futu_reuses_connection_and_preserves_millisecond_timestamp(self):
        context = MagicMock()
        context.get_market_snapshot.return_value = (
            0,
            pd.DataFrame([{
                "code": "US.AAPL", "name": "Apple", "last_price": 309.35,
                "prev_close_price": 311.30, "open_price": 312.05,
                "high_price": 312.38, "low_price": 307.01,
                "volume": 46_876_815,
                "update_time": "2026-08-21 19:59:59.839",
            }]),
        )
        FutuOpenDSource.close_shared_session()
        try:
            with (
                patch("futu.OpenQuoteContext", return_value=context) as constructor,
                patch.object(FutuOpenDSource, "opend_is_listening", return_value=True),
                patch.object(FutuOpenDSource, "_min_snapshot_interval_seconds", 0),
            ):
                first = FutuOpenDSource("AAPL").get_realtime()
                second = FutuOpenDSource("AAPL").get_realtime()

            constructor.assert_called_once_with(host="127.0.0.1", port=11111)
            self.assertEqual(context.get_market_snapshot.call_count, 2)
            self.assertFalse(first["_connection_reused"])
            self.assertTrue(second["_connection_reused"])
            self.assertEqual(
                first["_market_time"], "2026-08-21T19:59:59.839000-04:00",
            )
        finally:
            FutuOpenDSource.close_shared_session()


if __name__ == "__main__":
    unittest.main()
