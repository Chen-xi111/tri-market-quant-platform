"""
策略2: 布林带均值回归
- 价格触及下轨 + RSI 超卖 → 买入
- 价格触及上轨 + RSI 超买 → 卖出
- 回归中轨时平仓
"""

import pandas as pd
import numpy as np
from .base import Strategy
import config as cfg


class BollingerReversionStrategy(Strategy):
    name = "bollinger_reversion"
    description = "布林带均值回归：触轨反向交易 + RSI 确认"

    def __init__(self, params: dict = None):
        p = cfg.STRAT_BOLL_REVERSION.copy()
        p.update(params or {})
        super().__init__(p)

    def generate_signals(self) -> pd.DataFrame:
        df = self._df
        period = self.params["period"]
        std = self.params["std"]
        entry_pct = self.params["entry_pct"]
        use_rsi = self.params["rsi_filter"]

        # 布林带
        df["boll_mid"] = df["close"].rolling(period).mean()
        df["boll_std"] = df["close"].rolling(period).std()
        df["boll_upper"] = df["boll_mid"] + std * df["boll_std"]
        df["boll_lower"] = df["boll_mid"] - std * df["boll_std"]

        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/cfg.RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/cfg.RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        df["signal"] = 0
        df["signal_price"] = np.nan
        df["signal_reason"] = ""
        df["position_pct"] = 0.0

        for i in range(period + 10, len(df)):
            price = df["close"].iloc[i]
            upper = df["boll_upper"].iloc[i]
            lower = df["boll_lower"].iloc[i]
            mid = df["boll_mid"].iloc[i]
            rsi_val = df["rsi"].iloc[i]

            # 买入：触下轨 + RSI 不超卖（或有反弹迹象）
            near_lower = price <= lower * (1 + entry_pct)
            rsi_ok = rsi_val < 45 if use_rsi else True

            if near_lower and rsi_ok and not pd.isna(mid):
                df.iloc[i, df.columns.get_loc("signal")] = 1
                df.iloc[i, df.columns.get_loc("signal_price")] = price
                reason = f"触布林下轨 {lower:.3f} (RSI={rsi_val:.1f})，均值回归买入"
                df.iloc[i, df.columns.get_loc("signal_reason")] = reason
                df.iloc[i, df.columns.get_loc("position_pct")] = cfg.MAX_POSITION_PCT * 0.7

            # 卖出：触上轨
            near_upper = price >= upper * (1 - entry_pct)
            rsi_over = rsi_val > 55 if use_rsi else True

            if near_upper and rsi_over:
                df.iloc[i, df.columns.get_loc("signal")] = -1
                df.iloc[i, df.columns.get_loc("signal_price")] = price
                reason = f"触布林上轨 {upper:.3f} (RSI={rsi_val:.1f})，回归卖出"
                df.iloc[i, df.columns.get_loc("signal_reason")] = reason

        return df
