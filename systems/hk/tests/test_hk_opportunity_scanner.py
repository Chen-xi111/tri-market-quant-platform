import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from market_scan.opportunity_scanner import OpportunityScanner, OpportunityStore, build_candidate_pool
from market_scan.pipeline import ScanSnapshot


def history(ticker: str) -> pd.DataFrame:
    index = pd.bdate_range(end="2026-08-24", periods=260)
    close = pd.Series(np.linspace(10, 20, len(index)), index=index)
    frame = pd.DataFrame(
        {"open": close - 0.1, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1_000_000},
        index=index,
    )
    frame.attrs["ticker"] = ticker
    return frame


def snapshot() -> ScanSnapshot:
    return ScanSnapshot(
        metadata={"snapshot_date": "2026-08-24", "data_date": "2026-08-24"},
        stocks=pd.DataFrame(
            [
                {"rank": 1, "code": "01164", "name": "中广核矿业", "heat_score": 90.0, "flow_score": 90.0, "liquidity_score": 85.0, "turnover": 200_000_000, "change_pct": 3.0},
                {"rank": 50, "code": "00700", "name": "腾讯控股", "heat_score": 45.0, "flow_score": 45.0, "liquidity_score": 99.0, "turnover": 10_000_000_000, "change_pct": 1.0},
            ]
        ),
    )


class OpportunityTests(unittest.TestCase):
    def test_candidate_pool_includes_heat_turnover_and_watchlist_paths(self):
        pool = build_candidate_pool(snapshot(), always_candidates={"0700.HK": "平台预设股"})
        self.assertEqual(set(pool["code"]), {"01164", "00700"})
        self.assertIn("平台预设股", pool.loc[pool["code"] == "00700", "discovery_source"].iloc[0])

    def test_strict_gate_requires_research_and_strategy(self):
        def loader(ticker, days):
            return history(ticker), "fake", ()

        def signal(_frame):
            return SimpleNamespace(
                direction="strong_buy", score=6.0, confidence=0.7, buy_votes=4,
                sell_votes=0, recommended_position=0.2, stop_loss=18.0, take_profit=24.0,
            )

        def research(_frame):
            return {
                "research_gate_pass": True, "research_probability": 0.65,
                "research_quality": 70.0, "research_status": "样本外通过",
                "research_reasons": "通过",
            }

        with tempfile.TemporaryDirectory() as directory:
            result = OpportunityScanner(
                store=OpportunityStore(directory), history_loader=loader,
                signal_evaluator=signal, research_evaluator=research,
                always_candidates={}, max_candidates=1,
            ).run(snapshot())
            self.assertEqual(result.recommended["code"].tolist(), ["01164"])

    def test_high_score_is_rejected_when_reward_risk_is_too_low(self):
        def loader(ticker, days):
            return history(ticker), "fake", ()

        def signal(_frame):
            return SimpleNamespace(
                direction="strong_buy", score=7.0, confidence=0.75,
                buy_votes=4, sell_votes=0, recommended_position=0.2,
                stop_loss=18.0, take_profit=20.5,
            )

        def research(_frame):
            return {
                "research_gate_pass": True, "research_probability": 0.65,
                "research_quality": 70.0, "research_status": "样本外通过",
                "research_reasons": "通过",
            }

        row = snapshot().stocks.iloc[0].to_dict()
        result = OpportunityScanner(
            history_loader=loader, signal_evaluator=signal,
            research_evaluator=research, always_candidates={},
        )._evaluate_one(row, "2026-08-24")

        self.assertFalse(result["is_recommended"])
        self.assertLess(result["reward_risk"], 1.3)
        self.assertIn("现价盈亏比", result["rejection_reasons"])


if __name__ == "__main__":
    unittest.main()
