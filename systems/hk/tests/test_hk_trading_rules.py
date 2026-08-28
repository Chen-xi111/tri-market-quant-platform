import unittest

from trading_rules import calculate_trade_fee


class HKTradingRuleTests(unittest.TestCase):
    def test_fee_model_separates_stamp_duty_and_exchange_levies(self):
        fees = calculate_trade_fee(
            10_001,
            "BUY",
            commission_rate=0,
            minimum_commission=0,
        )

        self.assertEqual(fees["stamp_duty"], 11.0)
        self.assertGreater(fees["trading_fee"], fees["afrc_levy"])
        self.assertAlmostEqual(fees["total"], sum(value for key, value in fees.items() if key != "total"))


if __name__ == "__main__":
    unittest.main()
