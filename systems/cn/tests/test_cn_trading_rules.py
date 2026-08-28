import unittest
from datetime import date, timedelta

from execution.base import OrderSide
from execution.paper_broker import PaperBroker
from trading_rules import calculate_trade_fee, price_limit_bounds, price_limit_pct


class CNTradingRuleTests(unittest.TestCase):
    def test_main_board_and_growth_board_limits(self):
        self.assertEqual(price_limit_pct("600010"), 0.10)
        self.assertEqual(price_limit_pct("300750"), 0.20)
        self.assertEqual(price_limit_pct("302132"), 0.20)
        self.assertEqual(price_limit_pct("920222"), 0.30)
        self.assertEqual(price_limit_bounds(10.0, "600010"), (9.0, 11.0))

    def test_fees_include_minimum_commission_and_sell_stamp_duty(self):
        buy = calculate_trade_fee(1_000, "BUY")
        sell = calculate_trade_fee(1_000, "SELL")
        self.assertEqual(buy["commission"], 5.0)
        self.assertEqual(buy["stamp_duty"], 0.0)
        self.assertEqual(sell["stamp_duty"], 0.5)
        self.assertGreater(sell["total"], buy["total"])

    def test_buy_requires_board_lot_and_same_day_stock_is_not_sellable(self):
        broker = PaperBroker(initial_cash=100_000, slippage_bps=0)
        rejected = broker.place_order("SZ.000831", 10.0, 50, OrderSide.BUY)
        self.assertFalse(rejected.success)

        bought = broker.place_order("SZ.000831", 10.0, 100, OrderSide.BUY)
        self.assertTrue(bought.success)
        self.assertEqual(broker.get_positions()[0]["sellable_qty"], 0)
        sold_same_day = broker.place_order("SZ.000831", 10.1, 100, OrderSide.SELL)
        self.assertFalse(sold_same_day.success)
        self.assertIn("T+1", sold_same_day.message)

        broker._positions["SZ.000831"]["lots"][0]["buy_date"] = date.today() - timedelta(days=1)
        sold_next_day = broker.place_order("SZ.000831", 10.1, 100, OrderSide.SELL)
        self.assertTrue(sold_next_day.success)


if __name__ == "__main__":
    unittest.main()
