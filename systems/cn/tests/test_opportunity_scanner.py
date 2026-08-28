import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from fund_flow.opportunity_scanner import (
    OpportunityScanner,
    OpportunityStore,
    build_candidate_pool,
)
from fund_flow.pipeline import ScanSnapshot
from quant_research import ResearchEvidence


def sample_history(ticker: str) -> pd.DataFrame:
    index = pd.bdate_range(end="2026-08-21", periods=90)
    close = pd.Series(range(90), index=index, dtype=float) / 10 + 10
    frame = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )
    frame.attrs["ticker"] = ticker
    return frame


def passed_research(frame: pd.DataFrame) -> ResearchEvidence:
    return ResearchEvidence(
        status="样本外通过",
        gate_pass=True,
        current_signal="看多",
        current_probability=0.61,
        evidence_quality=68.0,
        oos_observations=180,
        fold_count=5,
        oos_auc=0.57,
        oos_brier=0.23,
        baseline_brier=0.25,
        brier_skill=0.08,
        signal_count=25,
        signal_hit_rate=0.60,
        signal_avg_net_return=0.018,
        return_ci_low=-0.004,
        return_ci_high=0.036,
        positive_fold_ratio=0.60,
        reasons=("样本外证据与当前概率均通过",),
    )


def scan_snapshot() -> ScanSnapshot:
    stocks = pd.DataFrame(
        [
            {
                "rank": 1, "code": "300750", "name": "强势股", "flow_score": 98.0,
                "positive_periods": 3, "main_net_today": 900_000_000,
                "change_pct_today": 2.0, "capital_signal": "连续流入",
            },
            {
                "rank": 2, "code": "600010", "name": "中性股", "flow_score": 96.0,
                "positive_periods": 3, "main_net_today": 500_000_000,
                "change_pct_today": 1.0, "capital_signal": "连续流入",
            },
            {
                "rank": 30, "code": "302132", "name": "板块龙头", "flow_score": 90.0,
                "positive_periods": 3, "main_net_today": 200_000_000,
                "change_pct_today": 19.0, "capital_signal": "连续流入",
            },
        ]
    )
    industries = pd.DataFrame(
        [{"name": "高端制造", "flow_score": 92, "main_net_today": 1, "leader_today": "板块龙头"}]
    )
    return ScanSnapshot(
        metadata={"snapshot_date": "2026-08-23", "data_date": "2026-08-21"},
        industries=industries,
        concepts=pd.DataFrame(),
        stocks=stocks,
    )


class OpportunityScannerTests(unittest.TestCase):
    def test_candidate_pool_includes_hot_sector_leader_outside_global_top(self):
        pool = build_candidate_pool(scan_snapshot())
        leader = pool[pool["code"] == "302132"].iloc[0]
        self.assertIn("热门行业·高端制造", leader["discovery_source"])

    def test_candidate_pool_always_includes_low_ranked_preset_or_watchlist_stock(self):
        pool = build_candidate_pool(
            scan_snapshot(), always_candidates={"302132.SZ": "平台预设股"}
        )
        forced = pool[pool["code"] == "302132"].iloc[0]
        self.assertTrue(forced["always_included"])
        self.assertIn("平台预设股", forced["discovery_source"])

    def test_high_conviction_technical_path_does_not_require_positive_daily_flow(self):
        def loader(ticker, days):
            return sample_history(ticker), "fake", ()

        def evaluator(frame):
            return SimpleNamespace(
                direction="strong_buy",
                score=6.0,
                confidence=0.71,
                buy_votes=4,
                sell_votes=1,
                recommended_position=0.18,
                stop_loss=17.0,
                take_profit=22.0,
            )

        row = {
            "rank": 4302,
            "code": "000831",
            "name": "中国稀土",
            "flow_score": 25.0,
            "positive_periods": 1,
            "main_net_today": -40_000_000,
            "change_pct_today": -2.0,
            "capital_signal": "当日流出",
            "discovery_source": "平台预设股",
        }
        scanner = OpportunityScanner(
            history_loader=loader,
            signal_evaluator=evaluator,
            research_evaluator=passed_research,
            always_candidates={},
        )

        result = scanner._evaluate_one(row, "2026-08-21")

        self.assertTrue(result["is_recommended"])
        self.assertEqual(
            result["selection_path"], "技术先行·资金未连续流出·样本外验证"
        )

    def test_strict_gate_only_recommends_strong_fresh_non_overheated_signal(self):
        def loader(ticker, days):
            return sample_history(ticker), "fake", ()

        def evaluator(frame):
            strong = frame.attrs["ticker"].startswith("300750") or frame.attrs["ticker"].startswith("302132")
            return SimpleNamespace(
                direction="strong_buy" if strong else "neutral",
                score=6.0 if strong else 1.0,
                confidence=0.43 if strong else 0.14,
                buy_votes=3 if strong else 1,
                sell_votes=0,
                recommended_position=0.18 if strong else 0,
                stop_loss=17.0,
                take_profit=22.0,
            )

        with tempfile.TemporaryDirectory() as directory:
            store = OpportunityStore(directory)
            result = OpportunityScanner(
                store=store,
                history_loader=loader,
                signal_evaluator=evaluator,
                research_evaluator=passed_research,
                max_workers=2,
                always_candidates={},
            ).run(scan_snapshot())

            self.assertEqual(result.recommended["code"].tolist(), ["300750"])
            overheated = result.evaluated[result.evaluated["code"] == "302132"].iloc[0]
            self.assertIn("过热", overheated["rejection_reasons"])
            loaded = store.load("2026-08-23")
            self.assertEqual(loaded.metadata["recommended_count"], 1)

    def test_validated_research_path_can_override_old_setup_score_cutoff(self):
        def loader(ticker, days):
            return sample_history(ticker), "fake", ()

        def evaluator(frame):
            return SimpleNamespace(
                direction="buy",
                score=4.6,
                confidence=0.57,
                buy_votes=3,
                sell_votes=1,
                recommended_position=0.18,
                stop_loss=17.0,
                take_profit=22.0,
            )

        row = {
            "rank": 4302,
            "code": "000831",
            "name": "中国稀土",
            "flow_score": 27.8,
            "positive_periods": 1,
            "main_net_today": -43_000_000,
            "change_pct_today": -2.0,
            "capital_signal": "当日流出",
            "discovery_source": "平台预设股",
        }
        result = OpportunityScanner(
            history_loader=loader,
            signal_evaluator=evaluator,
            research_evaluator=passed_research,
            always_candidates={},
        )._evaluate_one(row, "2026-08-21")

        self.assertLess(result["opportunity_score"], 70)
        self.assertTrue(result["is_recommended"])
        self.assertEqual(
            result["selection_path"], "量化验证主导·资金未恶化·样本外验证"
        )


if __name__ == "__main__":
    unittest.main()
