"""
策略3: 量价突破
- 价格突破 N 日最高价 + 成交量放大 → 买入
- 价格跌破 N 日最低价 + 成交量放大 → 卖出
- ATR 确认突破有效性
"""

import pandas as pd
import numpy as np
from .base import Strategy
import config as cfg


class VolBreakoutStrategy(Strategy):
    name = "vol_breakout"
    description = "量价突破：放量突破前高/前低入场"

    def __init__(self, params: dict = None):
        p = cfg.STRAT_VOL_BREAKOUT.copy()
        p.update(params or {})
        super().__init__(p)

    def generate_signals(self) -> pd.DataFrame:
        df = self._df
        lookback = self.params["lookback"]
        vol_ratio = self.params["vol_ratio"]
        breakout_pct = self.params["breakout_pct"]
        use_atr = self.params["atr_filter"]

        # 前N日最高/最低
        df["hh"] = df["high"].rolling(lookback).max()
        df["ll"] = df["low"].rolling(lookback).min()

        # 成交量均线
        df["vol_ma"] = df["volume"].rolling(cfg.VOL_MA_PERIOD).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma"].replace(0, np.nan)

        # ATR
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=cfg.ATR_PERIOD, adjust=False).mean()

        # 突破列
        df["breakout_up"] = df["close"] > df["hh"].shift(1) * (1 + breakout_pct)
        df["breakout_down"] = df["close"] < df["ll"].shift(1) * (1 - breakout_pct)

        df["signal"] = 0
        df["signal_price"] = np.nan
        df["signal_reason"] = ""
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["position_pct"] = 0.0

        for i in range(lookback + 10, len(df)):
            price = df["close"].iloc[i]
            vol_ok = df["vol_ratio"].iloc[i] >= vol_ratio
            atr_val = df["atr"].iloc[i]

            # 放量突破前高
            if df["breakout_up"].iloc[i] and vol_ok:
                df.iloc[i, df.columns.get_loc("signal")] = 1
                df.iloc[i, df.columns.get_loc("signal_price")] = price
                hh_val = df["hh"].iloc[i-1]
                reason = f"放量突破{lookback}日高 {hh_val:.3f}，量比{df['vol_ratio'].iloc[i]:.1f}"
                df.iloc[i, df.columns.get_loc("signal_reason")] = reason
                df.iloc[i, df.columns.get_loc("position_pct")] = cfg.MAX_POSITION_PCT * 0.8
                if not pd.isna(atr_val):
                    df.iloc[i, df.columns.get_loc("stop_loss")] = price - cfg.STOP_LOSS_ATR_MULT * atr_val
                    df.iloc[i, df.columns.get_loc("take_profit")] = price + cfg.TAKE_PROFIT_ATR_MULT * atr_val

            # 放量跌破前低
            if df["breakout_down"].iloc[i] and vol_ok:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                df.iloc[i, df.columns.get_loc("signal_price")] = price
                ll_val = df["ll"].iloc[i-1]
                reason = f"放量跌破{lookback}日低 {ll_val:.3f}，量比{df['vol_ratio'].iloc[i]:.1f}"
                df.iloc[i, df.columns.get_loc("signal_reason")] = reason

        return df
