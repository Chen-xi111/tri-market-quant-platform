import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from market_clock import is_hk_quote_session


HK = ZoneInfo("Asia/Hong_Kong")


class MarketClockTests(unittest.TestCase):
    def test_weekday_session_is_live(self):
        self.assertTrue(is_hk_quote_session(datetime(2026, 8, 10, 10, 30, tzinfo=HK)))
        self.assertTrue(is_hk_quote_session(datetime(2026, 8, 10, 13, 0, tzinfo=HK)))

    def test_after_hours_and_weekend_use_daily_close(self):
        self.assertFalse(is_hk_quote_session(datetime(2026, 8, 10, 20, 0, tzinfo=HK)))
        self.assertFalse(is_hk_quote_session(datetime(2026, 8, 9, 10, 30, tzinfo=HK)))


if __name__ == "__main__":
    unittest.main()
