import unittest

from trade_decision import assess_trade_decision, quote_requires_execution_block


def plan():
    return {
        "actionable": True,
        "entry_low": 9.8,
        "entry_high": 10.2,
        "breakout_trigger": 10.5,
        "max_chase": 10.8,
        "stop_loss": 9.0,
        "target_1": 12.0,
        "atr": 0.5,
        "risk_budget": 1_000.0,
        "planned_value": 10_000.0,
        "shares": 1_000,
        "first_shares": 500,
        "add_shares": 500,
        "lot_size": 100,
        "checks": {"数据新鲜": True, "综合信号偏多": True},
    }


class TradeDecisionTests(unittest.TestCase):
    def test_historical_fallback_quote_requires_execution_block(self):
        self.assertTrue(quote_requires_execution_block({"source": "腾讯证券(历史回退)"}))
        self.assertTrue(quote_requires_execution_block({"price": 10.0, "_cached": True}))
        self.assertFalse(quote_requires_execution_block({"price": 10.0, "source": "腾讯证券"}))

    def test_pullback_zone_is_orderable_only_during_session(self):
        decision = assess_trade_decision(plan(), 10.0, session_open=True)
        self.assertEqual(decision["status_key"], "orderable")
        self.assertEqual(decision["order_shares_now"], 500)

        closed = assess_trade_decision(plan(), 10.0, session_open=False)
        self.assertEqual(closed["status_key"], "ready_later")
        self.assertEqual(closed["order_shares_now"], 0)

    def test_high_score_plan_cannot_override_chasing_or_cached_data(self):
        chased = assess_trade_decision(plan(), 11.0, session_open=True)
        self.assertEqual(chased["headline"], "禁止新开仓")
        self.assertIn("现价超过追价上限", chased["hard_blockers"])

        cached = assess_trade_decision(plan(), 10.0, session_open=True, data_is_cached=True)
        self.assertFalse(cached["can_order_now"])
        self.assertIn("行情来自缓存或历史 K 线回退，不能据此新开仓", cached["hard_blockers"])

    def test_current_reward_risk_is_a_hard_gate(self):
        weak = plan()
        weak["target_1"] = 10.8
        decision = assess_trade_decision(weak, 10.0, session_open=True)
        self.assertEqual(decision["headline"], "禁止新开仓")
        self.assertLess(decision["reward_risk_now"], 1.3)

    def test_limit_up_price_blocks_a_new_position(self):
        decision = assess_trade_decision(
            plan(), 10.0, session_open=True, limit_up=10.0,
        )
        self.assertEqual(decision["headline"], "禁止新开仓")
        self.assertIn("现价已到常规涨停价，不追涨挂新仓", decision["hard_blockers"])
        self.assertEqual(decision["order_shares_now"], 0)


if __name__ == "__main__":
    unittest.main()
