import json
import unittest
from unittest.mock import patch

import pandas as pd

from data_sources.tencent_source import TencentSource
from indicators import compute_all
from limited_guidance import build_limited_guidance


def sample_history(rows: int) -> pd.DataFrame:
    index = pd.bdate_range("2026-01-02", periods=rows)
    close = pd.Series(range(rows), index=index, dtype=float) * 0.1 + 10
    return compute_all(pd.DataFrame({
        "open": close - 0.05,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": 1_000 + close * 10,
    }, index=index))


class LimitedGuidanceTests(unittest.TestCase):
    def test_short_but_usable_history_produces_non_actionable_observation(self):
        guidance = build_limited_guidance(sample_history(35))

        self.assertEqual(guidance.status, "limited")
        self.assertFalse(guidance.actionable)
        self.assertLessEqual(guidance.confidence, 0.45)
        self.assertGreater(guidance.target_reference, guidance.price)
        self.assertLess(guidance.stop_reference, guidance.price)

    def test_very_short_history_is_reported_as_insufficient(self):
        guidance = build_limited_guidance(sample_history(10))

        self.assertEqual(guidance.status, "insufficient")
        self.assertEqual(guidance.confidence, 0)
        self.assertFalse(guidance.actionable)


class TencentHistoryRetryTests(unittest.TestCase):
    def test_empty_responses_are_retried_through_alternate_endpoint(self):
        empty = json.dumps({"data": {"hk09880": {}}}).encode()
        valid = json.dumps({
            "data": {
                "hk09880": {
                    "day": [["2026-08-26", "10", "10.5", "10.8", "9.9", "1200"]]
                }
            }
        }).encode()

        with patch.object(
            TencentSource, "_request", side_effect=[empty, empty, empty, valid]
        ), patch("data_sources.tencent_source.time.sleep"):
            frame = TencentSource("09880.HK").get_history(days=365)

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame["close"].iloc[-1], 10.5)


if __name__ == "__main__":
    unittest.main()
