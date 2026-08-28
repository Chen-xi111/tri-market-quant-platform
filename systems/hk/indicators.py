"""
技术指标计算模块 — 均线、MACD、RSI、KDJ、布林带、ATR、成交量分析
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import config as cfg


class IndicatorCalculator:
    """技术指标计算器"""

    def __init__(self, df: pd.DataFrame):
        """
        df 需包含: open, high, low, close, volume
        """
        self.df = df.copy()
        self._ensure_columns()

    def _ensure_columns(self):
        """确保必要的列存在"""
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame 缺少列: {col}")

    # ================================================================
    #  均线 (MA / EMA)
    # ================================================================
    def add_ma(self, periods: list = None) -> pd.DataFrame:
        """添加简单移动均线"""
        if periods is None:
            periods = cfg.MA_PERIODS
        for p in periods:
            self.df[f"MA{p}"] = self.df["close"].rolling(window=p).mean()
        return self.df

    def add_ema(self, periods: list = None) -> pd.DataFrame:
        """添加指数移动均线"""
        if periods is None:
            periods = [12, 26]
        for p in periods:
            self.df[f"EMA{p}"] = self.df["close"].ewm(span=p, adjust=False).mean()
        return self.df

    # ================================================================
    #  MACD
    # ================================================================
    def add_macd(
        self, fast: int = None, slow: int = None, signal: int = None
    ) -> pd.DataFrame:
        """添加 MACD 指标"""
        fast = fast or cfg.MACD_FAST
        slow = slow or cfg.MACD_SLOW
        signal = signal or cfg.MACD_SIGNAL

        ema_fast = self.df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["close"].ewm(span=slow, adjust=False).mean()
        self.df["MACD_DIF"] = ema_fast - ema_slow
        self.df["MACD_DEA"] = self.df["MACD_DIF"].ewm(span=signal, adjust=False).mean()
        self.df["MACD_HIST"] = 2 * (self.df["MACD_DIF"] - self.df["MACD_DEA"])
        return self.df

    # ================================================================
    #  RSI
    # ================================================================
    def add_rsi(self, period: int = None) -> pd.DataFrame:
        """添加 RSI 指标"""
        period = period or cfg.RSI_PERIOD
        delta = self.df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        self.df["RSI"] = 100 - (100 / (1 + rs))
        return self.df

    # ================================================================
    #  KDJ
    # ================================================================
    def add_kdj(self, n: int = None, m1: int = None, m2: int = None) -> pd.DataFrame:
        """添加 KDJ 指标"""
        n = n or cfg.KDJ_N
        m1 = m1 or cfg.KDJ_M1
        m2 = m2 or cfg.KDJ_M2

        low_n = self.df["low"].rolling(window=n).min()
        high_n = self.df["high"].rolling(window=n).max()

        rsv = ((self.df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)) * 100

        self.df["KDJ_K"] = rsv.ewm(com=m1 - 1, adjust=False).mean()
        self.df["KDJ_D"] = self.df["KDJ_K"].ewm(com=m2 - 1, adjust=False).mean()
        self.df["KDJ_J"] = 3 * self.df["KDJ_K"] - 2 * self.df["KDJ_D"]
        return self.df

    # ================================================================
    #  布林带 (Bollinger Bands)
    # ================================================================
    def add_bollinger(
        self, period: int = None, std: float = None
    ) -> pd.DataFrame:
        """添加布林带"""
        period = period or cfg.BOLL_PERIOD
        std = std or cfg.BOLL_STD

        self.df["BOLL_MID"] = self.df["close"].rolling(window=period).mean()
        rolling_std = self.df["close"].rolling(window=period).std()
        self.df["BOLL_UPPER"] = self.df["BOLL_MID"] + std * rolling_std
        self.df["BOLL_LOWER"] = self.df["BOLL_MID"] - std * rolling_std
        self.df["BOLL_WIDTH"] = self.df["BOLL_UPPER"] - self.df["BOLL_LOWER"]
        self.df["BOLL_PCT"] = (self.df["close"] - self.df["BOLL_LOWER"]) / (
            self.df["BOLL_UPPER"] - self.df["BOLL_LOWER"]
        ).replace(0, np.nan)
        return self.df

    # ================================================================
    #  ATR (Average True Range)
    # ================================================================
    def add_atr(self, period: int = None) -> pd.DataFrame:
        """添加 ATR"""
        period = period or cfg.ATR_PERIOD
        high, low, close = self.df["high"], self.df["low"], self.df["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.df["ATR"] = tr.ewm(span=period, adjust=False).mean()
        return self.df

    # ================================================================
    #  成交量分析
    # ================================================================
    def add_volume_analysis(self, ma_period: int = None) -> pd.DataFrame:
        """添加成交量分析指标"""
        ma_period = ma_period or cfg.VOL_MA_PERIOD
        self.df["VOL_MA"] = self.df["volume"].rolling(window=ma_period).mean()
        self.df["VOL_RATIO"] = self.df["volume"] / self.df["VOL_MA"].replace(0, np.nan)
        vol_surge = self.df["VOL_RATIO"] >= cfg.VOL_SURGE_RATIO
        price_up = self.df["close"] > self.df["open"]
        self.df["VOL_SURGE"] = "normal"
        self.df.loc[vol_surge & price_up, "VOL_SURGE"] = "bull"
        self.df.loc[vol_surge & ~price_up, "VOL_SURGE"] = "bear"
        return self.df

    # ================================================================
    #  价格形态 / 支撑阻力
    # ================================================================
    def add_support_resistance(self, lookback: int = 20) -> pd.DataFrame:
        """添加近期支撑/阻力位"""
        self.df["RESIST_H"] = self.df["high"].rolling(window=lookback).max()
        self.df["SUPPORT_L"] = self.df["low"].rolling(window=lookback).min()
        return self.df

    # ================================================================
    #  一键添加所有指标
    # ================================================================
    def add_all(self) -> pd.DataFrame:
        """添加全部技术指标"""
        self.add_ma()
        self.add_ema()
        self.add_macd()
        self.add_rsi()
        self.add_kdj()
        self.add_bollinger()
        self.add_atr()
        self.add_volume_analysis()
        self.add_support_resistance()
        return self.df


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """便捷函数：对DataFrame计算全部指标"""
    return IndicatorCalculator(df).add_all()


if __name__ == "__main__":
    from data_fetcher import quick_history
    df = quick_history("1d", 365)
    ic = IndicatorCalculator(df)
    result = ic.add_all()
    cols = [c for c in result.columns if c not in ("open", "high", "low", "close", "volume")]
    print("指标列:", cols)
    print(result[["close"] + cols[:10]].tail(5).to_string())
