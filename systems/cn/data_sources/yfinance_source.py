"""
Yahoo Finance 数据源 — 沪深 A 股历史行情备用源
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from .base import DataSource
import config as cfg


class YFinanceSource(DataSource):
    name = "yfinance"
    _REQUEST_TIMEOUT_SECONDS = 8

    def __init__(self, ticker: str = None):
        super().__init__(ticker or cfg.TICKER)
        self._stock = yf.Ticker(self.ticker)

    def get_realtime(self) -> dict:
        try:
            # ``Ticker.info`` 需要拉取大量基本面字段，在切股时经常成为
            # 数秒级阻塞点。fast_info 只读行情所需的小字段集。
            fast = self._stock.fast_info
            price = fast.get("lastPrice")
            prev = fast.get("previousClose") or fast.get("regularMarketPreviousClose")

            # 如果实时接口拿不到 price，用历史数据兜底
            if price is None:
                hist = self.get_history(period="1d", days=10)
                if not hist.empty:
                    last = hist.iloc[-1]
                    prev = hist.iloc[-2]["close"] if len(hist) > 1 else last["close"]
                    price = last["close"]
                    return {
                        "ticker": self.ticker,
                        "name": cfg.PRESET_STOCKS.get(self.ticker, self.ticker),
                        "price": price,
                        "open": last["open"],
                        "high": last["high"],
                        "low": last["low"],
                        "prev_close": prev,
                        "volume": last["volume"],
                        "change": round(price - prev, 3),
                        "change_pct": round((price / prev - 1) * 100, 2) if prev else 0.0,
                        "currency": fast.get("currency") or "CNY",
                        "source": "Yahoo Finance(历史回退)",
                    }
                return {}

            change = price - prev if prev and price else None
            change_pct = round((price / prev - 1) * 100, 2) if prev and price else None

            return {
                "ticker": self.ticker,
                "name": cfg.PRESET_STOCKS.get(self.ticker, self.ticker),
                "price": price,
                "open": fast.get("open"),
                "high": fast.get("dayHigh"),
                "low": fast.get("dayLow"),
                "prev_close": prev,
                "volume": fast.get("lastVolume"),
                "avg_volume": fast.get("threeMonthAverageVolume"),
                "market_cap": fast.get("marketCap"),
                "change": round(change, 3) if change else None,
                "change_pct": change_pct,
                "currency": fast.get("currency") or "CNY",
                "source": "Yahoo Finance",
            }
        except Exception:
            return {}

    def get_history(self, period: str = "1d", days: int = 365) -> pd.DataFrame:
        intraday_map = {
            "1m": "7d", "5m": "60d", "15m": "60d",
            "30m": "60d", "60m": "730d", "1h": "730d",
        }
        try:
            if period in intraday_map:
                df = self._stock.history(
                    period=intraday_map[period],
                    interval=period,
                    timeout=self._REQUEST_TIMEOUT_SECONDS,
                )
            else:
                start = datetime.now() - pd.Timedelta(days=days)
                df = self._stock.history(
                    start=start,
                    interval=period,
                    timeout=self._REQUEST_TIMEOUT_SECONDS,
                )
        except Exception:
            return pd.DataFrame()

        if df.empty:
            return df

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df.dropna(how="all", inplace=True)
        return df

    def get_info(self) -> dict:
        try:
            return self._stock.info or {}
        except Exception:
            return {}
