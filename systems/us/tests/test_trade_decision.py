import unittest

from trade_decision import assess_trade_decision


def plan():
    return {
        "actionable": True, "entry_low": 9.8, "entry_high": 10.2,
        "breakout_trigger": 10.5, "max_chase": 10.8, "stop_loss": 9.0,
        "target_1": 12.0, "atr": 0.5, "risk_budget": 1_000.0,
        "planned_value": 10_000.0, "shares": 1_000, "first_shares": 500,
        "add_shares": 500, "lot_size": 1,
        "checks": {"数据新鲜": True, "综合信号偏多": True},
    }


class TradeDecisionTests(unittest.TestCase):
    def test_trade_window_and_session_gate(self):
        opened = assess_trade_decision(plan(), 10.0, session_open=True)
        self.assertTrue(opened["can_order_now"])
        self.assertEqual(opened["order_shares_now"], 500)
        closed = assess_trade_decision(plan(), 10.0, session_open=False)
        self.assertEqual(closed["headline"], "计划就绪，开市复核")

    def test_chasing_cached_data_and_weak_reward_are_blocked(self):
        chased = assess_trade_decision(plan(), 11.0, session_open=True)
        self.assertIn("现价超过追价上限", chased["hard_blockers"])
        cached = assess_trade_decision(plan(), 10.0, session_open=True, data_is_cached=True)
        self.assertFalse(cached["can_order_now"])
        weak = plan()
        weak["target_1"] = 10.8
        self.assertEqual(
            assess_trade_decision(weak, 10.0, session_open=True)["headline"],
            "禁止新开仓",
        )

    def test_delayed_quote_reason_blocks_regular_session_entry(self):
        decision = assess_trade_decision(
            plan(),
            10.0,
            session_open=True,
            data_block_reason="常规交易时段报价状态为“延迟行情”，不用于新开仓",
        )
        self.assertEqual(decision["headline"], "禁止新开仓")
        self.assertFalse(decision["checks"]["实时报价质量达标"])
        self.assertEqual(decision["order_shares_now"], 0)


if __name__ == "__main__":
    unittest.main()
