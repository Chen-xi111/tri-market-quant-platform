import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from market_clock import (
    HKMarketPhase,
    hk_market_phase,
    is_hk_continuous_session,
    is_hk_quote_session,
)


HK = ZoneInfo("Asia/Hong_Kong")


class MarketClockTests(unittest.TestCase):
    def test_weekday_session_is_live(self):
        self.assertTrue(is_hk_quote_session(datetime(2026, 8, 10, 9, 10, tzinfo=HK)))
        self.assertTrue(is_hk_quote_session(datetime(2026, 8, 10, 10, 30, tzinfo=HK)))
        self.assertTrue(is_hk_quote_session(datetime(2026, 8, 10, 13, 0, tzinfo=HK)))

    def test_only_continuous_sessions_are_orderable(self):
        self.assertEqual(
            hk_market_phase(datetime(2026, 8, 10, 9, 10, tzinfo=HK)),
            HKMarketPhase.PRE_OPEN,
        )
        self.assertTrue(
            is_hk_continuous_session(datetime(2026, 8, 10, 10, 30, tzinfo=HK))
        )
        self.assertEqual(
            hk_market_phase(datetime(2026, 8, 10, 12, 30, tzinfo=HK)),
            HKMarketPhase.LUNCH,
        )
        self.assertFalse(
            is_hk_continuous_session(datetime(2026, 8, 10, 12, 30, tzinfo=HK))
        )
        self.assertEqual(
            hk_market_phase(datetime(2026, 8, 10, 16, 5, tzinfo=HK)),
            HKMarketPhase.CLOSING_AUCTION,
        )
        self.assertFalse(
            is_hk_continuous_session(datetime(2026, 8, 10, 16, 5, tzinfo=HK))
        )

    def test_after_hours_and_weekend_use_daily_close(self):
        self.assertFalse(is_hk_quote_session(datetime(2026, 8, 10, 20, 0, tzinfo=HK)))
        self.assertFalse(is_hk_quote_session(datetime(2026, 8, 9, 10, 30, tzinfo=HK)))


if __name__ == "__main__":
    unittest.main()
