import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_scan.pipeline import MarketScanPipeline, SnapshotStore, score_market


class FakeProvider:
    name = "fake"
    version = "1"

    def stock_quotes(self, progress=None):
        frame = pd.DataFrame(
            [
                {
                    "code": "00700", "name": "腾讯控股", "price": 457.0,
                    "prev_close": 451.4, "open": 451.0, "high": 458.0, "low": 449.0,
                    "volume": 22_000_000, "turnover": 10_000_000_000,
                    "change": 5.6, "change_pct": 1.24, "turnover_rate": 1.2,
                    "quote_time": "2026-08-24T16:08:00", "instrument_type": "GP",
                },
                {
                    "code": "01164", "name": "中广核矿业", "price": 2.97,
                    "prev_close": 2.72, "open": 2.70, "high": 3.01, "low": 2.69,
                    "volume": 75_000_000, "turnover": 217_000_000,
                    "change": 0.25, "change_pct": 9.19, "turnover_rate": 4.95,
                    "quote_time": "2026-08-24T16:08:00", "instrument_type": "GP",
                },
                {
                    "code": "09999", "name": "低流动股", "price": 0.08,
                    "prev_close": 0.08, "open": 0.08, "high": 0.08, "low": 0.08,
                    "volume": 1000, "turnover": 80, "change": 0, "change_pct": 0,
                    "turnover_rate": 0.01, "quote_time": "2026-08-24T16:08:00",
                    "instrument_type": "GP",
                },
            ]
        )
        frame.attrs["partial_errors"] = ()
        frame.attrs["universe_discovery"] = False
        return frame


class MarketScanTests(unittest.TestCase):
    def test_heat_score_penalizes_illiquid_penny_stock(self):
        scored = score_market(FakeProvider().stock_quotes())
        liquid = scored[scored["code"] == "00700"].iloc[0]
        penny = scored[scored["code"] == "09999"].iloc[0]
        self.assertGreater(liquid["heat_score"], penny["heat_score"])
        self.assertEqual(scored.iloc[0]["rank"], 1)

    def test_pipeline_round_trips_snapshot_and_market_breadth(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(Path(directory))
            snapshot = MarketScanPipeline(FakeProvider(), store).run()
            self.assertEqual(snapshot.metadata["data_date"], "2026-08-24")
            self.assertEqual(snapshot.metadata["universe_count"], 3)
            self.assertEqual(snapshot.market.iloc[-1]["rising"], 2)
            loaded = store.load(snapshot.snapshot_date)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.stocks), 3)


if __name__ == "__main__":
    unittest.main()
