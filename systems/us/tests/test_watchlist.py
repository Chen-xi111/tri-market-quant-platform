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
                "US.PLTR", cfg.PRESET_STOCKS, cfg.normalize_ticker, path,
            )
            self.assertEqual(code, "PLTR")
            self.assertTrue(added)
            self.assertEqual(load_user_stocks(cfg.normalize_ticker, path), ["PLTR"])

            _, added_again = remember_user_stock(
                "pltr.us", cfg.PRESET_STOCKS, cfg.normalize_ticker, path,
            )
            self.assertFalse(added_again)
            self.assertEqual(load_user_stocks(cfg.normalize_ticker, path), ["PLTR"])

            _, preset_added = remember_user_stock(
                "aapl", cfg.PRESET_STOCKS, cfg.normalize_ticker, path,
            )
            self.assertFalse(preset_added)
            self.assertEqual(load_user_stocks(cfg.normalize_ticker, path), ["PLTR"])

    def test_corrupt_or_invalid_entries_do_not_break_loading(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stocks.json"
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(load_user_stocks(cfg.normalize_ticker, path), [])
            path.write_text(json.dumps({"tickers": ["pltr", "$bad", "US.PLTR"]}), encoding="utf-8")
            self.assertEqual(load_user_stocks(cfg.normalize_ticker, path), ["PLTR"])


if __name__ == "__main__":
    unittest.main()
