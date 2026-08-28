import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from market_clock import CNMarketPhase, cn_market_phase, is_cn_quote_session


CN = ZoneInfo("Asia/Shanghai")


class MarketClockTests(unittest.TestCase):
    def test_weekday_session_is_live(self):
        self.assertTrue(is_cn_quote_session(datetime(2026, 8, 10, 9, 20, tzinfo=CN)))
        self.assertTrue(is_cn_quote_session(datetime(2026, 8, 10, 10, 30, tzinfo=CN)))
        self.assertTrue(is_cn_quote_session(datetime(2026, 8, 10, 13, 0, tzinfo=CN)))

    def test_lunch_break_is_identified_but_quotes_remain_available(self):
        moment = datetime(2026, 8, 10, 12, 0, tzinfo=CN)
        self.assertEqual(cn_market_phase(moment), CNMarketPhase.LUNCH)
        self.assertTrue(is_cn_quote_session(moment))

    def test_after_hours_and_weekend_use_daily_close(self):
        self.assertFalse(is_cn_quote_session(datetime(2026, 8, 10, 20, 0, tzinfo=CN)))
        self.assertFalse(is_cn_quote_session(datetime(2026, 8, 9, 10, 30, tzinfo=CN)))


if __name__ == "__main__":
    unittest.main()
