"""
AKShare 数据源 — 国产开源金融数据库，支持资金流向、分钟线等深度数据
"""

import pandas as pd
import numpy as np
from datetime import datetime
from .base import DataSource
import config as cfg


class AKShareSource(DataSource):
    """
    AKShare 封装
    - 实时行情: stock_hk_spot_em() → 东方财富港股实时
    - 历史K线: stock_hk_hist() → 港股日线/分钟线
    - 资金流向: stock_hk_fund_flow_individual() → 个股资金流
    """

    name = "akshare"

    def __init__(self, ticker: str = None):
        super().__init__(ticker or cfg.TICKER)
        # yfinance "1164.HK" → AKShare "01164"
        self._symbol = self.ticker.replace(".HK", "")
        if len(self._symbol) < 5:
            self._symbol = "0" + self._symbol

    def get_realtime(self) -> dict:
        try:
            import akshare as ak
            df = ak.stock_hk_spot_em()
            row = df[df["代码"] == self._symbol]
            if row.empty:
                return {}

            r = row.iloc[0]
            return {
                "ticker": self.ticker,
                "name": r.get("名称", cfg.TICKER_NAME),
                "price": self._safe_float(r.get("最新价")),
                "open": self._safe_float(r.get("今开")),
                "high": self._safe_float(r.get("最高")),
                "low": self._safe_float(r.get("最低")),
                "prev_close": self._safe_float(r.get("昨收")),
                "volume": self._safe_float(r.get("成交量")),
                "amount": self._safe_float(r.get("成交额")),
                "change": self._safe_float(r.get("涨跌额")),
                "change_pct": self._safe_float(r.get("涨跌幅")),
                "market_cap": self._safe_float(r.get("总市值")),
                "currency": "HKD",
                "source": "AKShare(东方财富)",
            }
        except Exception as e:
            return {"error": str(e)}

    def get_history(self, period: str = "1d", days: int = 365) -> pd.DataFrame:
        try:
            import akshare as ak

            # 周期映射
            period_map = {
                "1d": "daily", "1w": "weekly", "1M": "monthly",
                "60m": "60", "30m": "30", "15m": "15", "5m": "5", "1m": "1",
            }
            ak_period = period_map.get(period, "daily")

            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - pd.Timedelta(days=days)).strftime("%Y%m%d")

            df = ak.stock_hk_hist(
                symbol=self._symbol,
                period=ak_period,
                start_date=start,
                end_date=end,
                adjust="qfq",  # 前复权
            )

            if df.empty:
                return df

            # 标准化列名
            col_map = {
                "开盘": "open", "收盘": "close", "最高": "high",
                "最低": "low", "成交量": "volume", "成交额": "amount",
                "日期": "date",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            return df

        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _safe_float(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
