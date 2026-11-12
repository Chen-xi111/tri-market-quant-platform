import unittest

from trade_decision import assess_trade_decision, quote_requires_execution_block


def plan():
    return {
        "actionable": True, "entry_low": 9.8, "entry_high": 10.2,
        "breakout_trigger": 10.5, "max_chase": 10.8, "stop_loss": 9.0,
        "target_1": 12.0, "atr": 0.5, "risk_budget": 1_000.0,
        "planned_value": 10_000.0, "shares": 1_000, "first_shares": 500,
        "add_shares": 500, "lot_size": 100,
        "checks": {"数据新鲜": True, "综合信号偏多": True},
    }


class TradeDecisionTests(unittest.TestCase):
    def test_historical_fallback_quote_requires_execution_block(self):
        self.assertTrue(quote_requires_execution_block({"source": "Yahoo Finance(历史回退)"}))
        self.assertTrue(quote_requires_execution_block({"price": 10.0, "_cached": True}))
        self.assertFalse(quote_requires_execution_block({"price": 10.0, "source": "腾讯证券"}))

    def test_pullback_zone_is_orderable_only_during_session(self):
        decision = assess_trade_decision(plan(), 10.0, session_open=True)
        self.assertEqual(decision["status_key"], "orderable")
        self.assertEqual(decision["order_shares_now"], 500)
        closed = assess_trade_decision(plan(), 10.0, session_open=False)
        self.assertEqual(closed["status_key"], "ready_later")
        self.assertEqual(closed["order_shares_now"], 0)

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

    def test_low_turnover_is_a_hong_kong_specific_hard_gate(self):
        decision = assess_trade_decision(
            plan(), 10.0, session_open=True, median_turnover=2_000_000,
        )
        self.assertEqual(decision["headline"], "禁止新开仓")
        self.assertFalse(decision["checks"]["20日中位成交额达标"])
        self.assertEqual(decision["order_shares_now"], 0)


if __name__ == "__main__":
    unittest.main()
