import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from market_clock import (
    NEW_YORK_TZ,
    USMarketPhase,
    is_us_quote_session,
    is_us_regular_session,
    us_market_phase,
)


class MarketClockTests(unittest.TestCase):
    def test_regular_pre_and_after_hours(self):
        self.assertEqual(
            us_market_phase(datetime(2026, 8, 10, 5, 0, tzinfo=NEW_YORK_TZ)),
            USMarketPhase.PRE_MARKET,
        )
        self.assertTrue(
            is_us_regular_session(datetime(2026, 8, 10, 10, 30, tzinfo=NEW_YORK_TZ))
        )
        self.assertEqual(
            us_market_phase(datetime(2026, 8, 10, 18, 0, tzinfo=NEW_YORK_TZ)),
            USMarketPhase.AFTER_HOURS,
        )
        self.assertFalse(
            is_us_quote_session(datetime(2026, 8, 10, 21, 0, tzinfo=NEW_YORK_TZ))
        )

    def test_weekend_and_exchange_holiday_are_closed(self):
        self.assertEqual(
            us_market_phase(datetime(2026, 8, 9, 10, 30, tzinfo=NEW_YORK_TZ)),
            USMarketPhase.CLOSED,
        )
        # 2026-07-03 observes Independence Day.
        self.assertEqual(
            us_market_phase(datetime(2026, 7, 3, 10, 30, tzinfo=NEW_YORK_TZ)),
            USMarketPhase.CLOSED,
        )

    def test_early_close_moves_to_after_hours_at_1pm(self):
        self.assertEqual(
            us_market_phase(datetime(2026, 7, 2, 14, 0, tzinfo=NEW_YORK_TZ)),
            USMarketPhase.AFTER_HOURS,
        )

    def test_shanghai_times_convert_across_dst(self):
        shanghai = ZoneInfo("Asia/Shanghai")
        self.assertTrue(
            is_us_regular_session(datetime(2026, 1, 5, 22, 30, tzinfo=shanghai))
        )
        self.assertTrue(
            is_us_regular_session(datetime(2026, 8, 10, 21, 30, tzinfo=shanghai))
        )


if __name__ == "__main__":
    unittest.main()
