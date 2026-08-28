import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import config as cfg
from watchlist import load_user_stocks, remember_user_stock


class WatchlistTests(unittest.TestCase):
    def test_custom_stock_survives_reload_and_is_normalized(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stocks.json"
            code, added = remember_user_stock(
                "HK.02172", cfg.PRESET_STOCKS, cfg.normalize_ticker, path,
            )
            self.assertEqual(code, "2172.HK")
            self.assertFalse(added, "预设股票不应重复写入自选清单")

            code, added = remember_user_stock(
                "815", cfg.PRESET_STOCKS, cfg.normalize_ticker, path,
            )
            self.assertEqual(code, "0815.HK")
            self.assertTrue(added)
            self.assertEqual(load_user_stocks(cfg.normalize_ticker, path), ["0815.HK"])

            _, added_again = remember_user_stock(
                "00815.HK", cfg.PRESET_STOCKS, cfg.normalize_ticker, path,
            )
            self.assertFalse(added_again)
            self.assertEqual(load_user_stocks(cfg.normalize_ticker, path), ["0815.HK"])

    def test_corrupt_or_invalid_entries_do_not_break_loading(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stocks.json"
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(load_user_stocks(cfg.normalize_ticker, path), [])
            path.write_text(json.dumps({"tickers": ["700", "bad", "HK.00700"]}), encoding="utf-8")
            self.assertEqual(load_user_stocks(cfg.normalize_ticker, path), ["0700.HK"])


if __name__ == "__main__":
    unittest.main()
