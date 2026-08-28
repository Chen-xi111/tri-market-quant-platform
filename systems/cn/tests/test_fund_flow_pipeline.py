import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from fund_flow.pipeline import FundFlowPipeline, SnapshotStore
from fund_flow.scheduler import should_scan_today


class FakeProvider:
    name = "fake"
    version = "1"

    def market_flow(self):
        return pd.DataFrame(
            {
                "日期": ["2026-08-20", "2026-08-21"],
                "上证-收盘价": [3500, 3520],
                "上证-涨跌幅": [0.2, 0.5],
                "深证-收盘价": [11000, 11100],
                "深证-涨跌幅": [0.1, 0.8],
                "主力净流入-净额": [-2_000_000_000, 3_000_000_000],
                "主力净流入-净占比": [-1.0, 1.5],
                "超大单净流入-净额": [-1_000_000_000, 2_000_000_000],
                "大单净流入-净额": [-1_000_000_000, 1_000_000_000],
                "中单净流入-净额": [500_000_000, -400_000_000],
                "小单净流入-净额": [1_500_000_000, -2_600_000_000],
            }
        )

    def sector_flow(self, indicator, sector_type):
        multiplier = {"今日": 1, "5日": 3, "10日": 5}[indicator]
        return pd.DataFrame(
            {
                "名称": ["强势板块", "弱势板块"],
                f"{indicator}涨跌幅": [2.0, -1.0],
                f"{indicator}主力净流入-净额": [2_000_000_000 * multiplier, -500_000_000 * multiplier],
                f"{indicator}主力净流入-净占比": [5.0, -2.0],
                f"{indicator}超大单净流入-净额": [1_000_000_000, -300_000_000],
                f"{indicator}大单净流入-净额": [1_000_000_000, -200_000_000],
                f"{indicator}中单净流入-净额": [-500_000_000, 100_000_000],
                f"{indicator}小单净流入-净额": [-1_500_000_000, 400_000_000],
                f"{indicator}主力净流入最大股": ["示例股份", "示例科技"],
            }
        )

    def stock_flow(self, indicator):
        multiplier = {"今日": 1, "5日": 3, "10日": 5}[indicator]
        return pd.DataFrame(
            {
                "代码": [10, 600010],
                "名称": ["领先股份", "落后股份"],
                "最新价": [12.3, 2.1],
                f"{indicator}涨跌幅": [1.5, -0.8],
                f"{indicator}主力净流入-净额": [900_000_000 * multiplier, -100_000_000 * multiplier],
                f"{indicator}主力净流入-净占比": [6.0, -1.0],
                f"{indicator}超大单净流入-净额": [500_000_000, -80_000_000],
                f"{indicator}大单净流入-净额": [400_000_000, -20_000_000],
                f"{indicator}中单净流入-净额": [-200_000_000, 40_000_000],
                f"{indicator}小单净流入-净额": [-700_000_000, 60_000_000],
            }
        )


class PartiallyFailingProvider(FakeProvider):
    def sector_flow(self, indicator, sector_type):
        if sector_type == "概念资金流":
            raise ValueError("temporary response error")
        return super().sector_flow(indicator, sector_type)


class MarketFailingProvider(FakeProvider):
    def market_flow(self):
        raise ValueError("market endpoint unavailable")


class FundFlowPipelineTests(unittest.TestCase):
    def test_pipeline_scores_and_round_trips_daily_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SnapshotStore(directory)
            snapshot = FundFlowPipeline(FakeProvider(), store).run()

            self.assertEqual(snapshot.metadata["data_date"], "2026-08-21")
            self.assertEqual(snapshot.industries.iloc[0]["name"], "强势板块")
            self.assertEqual(snapshot.industries.iloc[0]["capital_signal"], "连续流入")
            self.assertEqual(snapshot.stocks.iloc[0]["code"], "000010")
            self.assertGreater(
                snapshot.stocks.iloc[0]["flow_score"],
                snapshot.stocks.iloc[1]["flow_score"],
            )

            loaded = store.load(snapshot.snapshot_date)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.market), 2)
            self.assertEqual(loaded.stocks.iloc[0]["name"], "领先股份")

    def test_partial_failure_keeps_successful_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = FundFlowPipeline(
                PartiallyFailingProvider(), SnapshotStore(directory)
            ).run()
            self.assertFalse(snapshot.industries.empty)
            self.assertTrue(snapshot.concepts.empty)
            self.assertEqual(len(snapshot.errors), 3)

    def test_market_total_falls_back_to_stock_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = FundFlowPipeline(
                MarketFailingProvider(), SnapshotStore(directory)
            ).run()
            self.assertTrue(snapshot.metadata["market_derived_from_stocks"])
            self.assertEqual(len(snapshot.market), 1)
            self.assertAlmostEqual(snapshot.market.iloc[0]["main_net"], 800_000_000)

    def test_scheduler_runs_only_after_close_on_unsaved_weekday(self):
        timezone = ZoneInfo("Asia/Shanghai")
        before = datetime(2026, 8, 24, 15, 0, tzinfo=timezone)
        after = datetime(2026, 8, 24, 15, 21, tzinfo=timezone)
        weekend = datetime(2026, 8, 23, 16, 0, tzinfo=timezone)
        self.assertFalse(should_scan_today(before, []))
        self.assertTrue(should_scan_today(after, []))
        self.assertFalse(should_scan_today(after, ["2026-08-24"]))
        self.assertFalse(should_scan_today(weekend, []))


if __name__ == "__main__":
    unittest.main()
