"""
数据获取模块 — 基于 yfinance 获取A 股实时与历史行情
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import time
import config as cfg


class DataFetcher:
    """A 股行情数据获取器"""

    def __init__(self, ticker: str = cfg.TICKER):
        self.ticker = ticker
        self._stock = yf.Ticker(ticker)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_time: Dict[str, float] = {}

    # ----- 基本信息 -----
    def get_info(self) -> dict:
        """获取标的基本信息"""
        try:
            return self._stock.info or {}
        except Exception:
            return {}

    # ----- 历史K线 -----
    def get_history(
        self,
        period: str = cfg.PRIMARY_PERIOD,
        days: int = cfg.HISTORY_DAYS,
        use_cache: bool = True,
        cache_ttl: float = 10.0,
    ) -> pd.DataFrame:
        """
        获取历史K线数据
        period: 1m/5m/15m/30m/60m/1h/1d/1wk/1mo
        """
        cache_key = f"hist_{period}_{days}"
        now = time.time()

        if use_cache and cache_key in self._cache:
            if now - self._cache_time.get(cache_key, 0) < cache_ttl:
                return self._cache[cache_key].copy()

        # 对于日内周期，yfinance 有最大范围限制
        intraday_map = {
            "1m": "7d", "2m": "60d", "5m": "60d",
            "15m": "60d", "30m": "60d", "60m": "730d", "1h": "730d",
        }
        if period in intraday_map:
            df = self._stock.history(period=period, interval=period)
        else:
            df = self._stock.history(period=f"{days}d", interval=period)

        if not df.empty:
            df = self._clean_dataframe(df)
            self._cache[cache_key] = df.copy()
            self._cache_time[cache_key] = now

        return df

    # ----- 实时报价 -----
    def get_realtime(self) -> dict:
        """获取实时报价快照"""
        try:
            info = self._stock.info or {}
            fast = self._stock.fast_info or {}
            return {
                "ticker":      self.ticker,
                "name":        info.get("longName", cfg.TICKER_NAME_CN),
                "price":       info.get("currentPrice") or info.get("regularMarketPrice"),
                "open":        info.get("open") or info.get("regularMarketOpen"),
                "high":        info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "low":         info.get("dayLow") or info.get("regularMarketDayLow"),
                "prev_close":  info.get("previousClose") or info.get("regularMarketPreviousClose"),
                "volume":      info.get("volume") or info.get("regularMarketVolume"),
                "avg_volume":  info.get("averageVolume"),
                "market_cap":  info.get("marketCap"),
                "pe_ratio":    info.get("trailingPE"),
                "bid":         info.get("bid"),
                "ask":         info.get("ask"),
                "currency":    info.get("currency", "CNY"),
                "change":      info.get("regularMarketChange"),
                "change_pct":  info.get("regularMarketChangePercent"),
                "52w_high":    info.get("fiftyTwoWeekHigh"),
                "52w_low":     info.get("fiftyTwoWeekLow"),
            }
        except Exception:
            return {}

    # ----- 多周期数据 -----
    def get_multi_period(self, periods: list = None) -> Dict[str, pd.DataFrame]:
        """批量获取多周期数据"""
        if periods is None:
            periods = [cfg.PRIMARY_PERIOD] + cfg.SECONDARY_PERIODS
        result = {}
        for p in periods:
            result[p] = self.get_history(period=p)
        return result

    # ----- 内部辅助 -----
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """清洗 DataFrame"""
        df = df.copy()
        # 标准化列名
        rename_map = {
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
        # 删除全NaN行
        df.dropna(how="all", inplace=True)
        return df


# ============================================================
# 便捷函数
# ============================================================

_fetcher: Optional[DataFetcher] = None


def get_fetcher() -> DataFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher


def quick_quote() -> dict:
    """快速获取实时报价"""
    return get_fetcher().get_realtime()


def quick_history(period: str = "1d", days: int = 365) -> pd.DataFrame:
    """快速获取历史数据"""
    return get_fetcher().get_history(period=period, days=days)


if __name__ == "__main__":
    fetcher = DataFetcher()
    print("=== 实时报价 ===")
    quote = fetcher.get_realtime()
    for k, v in quote.items():
        print(f"  {k}: {v}")

    print("\n=== 日线（最近5条）===")
    df = fetcher.get_history(period="1d", days=365)
    print(df.tail(5).to_string())
