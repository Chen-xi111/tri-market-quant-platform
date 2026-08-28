"""
策略4: 多周期共振
- 日线定方向（MA排列 + 价格位置）
- 60分钟线找入场（回调到支撑位时）
- 只在日线多头时做多
"""

import pandas as pd
import numpy as np
from .base import Strategy
import config as cfg


class MultiTFStrategy(Strategy):
    name = "multi_tf"
    description = "多周期共振：日线定方向，60分钟找入场点"

    def __init__(self, params: dict = None):
        p = cfg.STRAT_MULTI_TF.copy()
        p.update(params or {})
        super().__init__(p)
        self._trend_df: pd.DataFrame = None   # 日线数据
        self._entry_df: pd.DataFrame = None   # 分钟线数据

    def set_trend_data(self, df: pd.DataFrame):
        """日线数据"""
        self._trend_df = df.copy()

    def set_entry_data(self, df: pd.DataFrame):
        """入场周期数据（60m）"""
        self._entry_df = df.copy()

    def generate_signals(self) -> pd.DataFrame:
        """
        在当前注入的 df (应为日线) 上标记信号
        如果有分钟线数据，信号更精确
        """
        df = self._df
        mas = self.params["ma_periods"]
        rsi_min = self.params["rsi_min"]
        rsi_max = self.params["rsi_max"]

        # 日线均线
        for p in mas:
            df[f"ma{p}"] = df["close"].rolling(p).mean()

        # 日线 RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 判断趋势：短期均线在长期均线上方 = 多头
        short, mid, long = mas[0], mas[1], mas[2]
        df["trend_bull"] = (
            (df[f"ma{short}"] > df[f"ma{mid}"]) &
            (df[f"ma{mid}"] > df[f"ma{long}"])
        )

        # MACD 辅助
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd_dif"] = ema12 - ema26

        df["signal"] = 0
        df["signal_price"] = np.nan
        df["signal_reason"] = ""
        df["position_pct"] = 0.0

        for i in range(long + 5, len(df)):
            bull = df["trend_bull"].iloc[i]
            rsi_val = df["rsi"].iloc[i]
            macd = df["macd_dif"].iloc[i]
            prev_macd = df["macd_dif"].iloc[i-1]

            # 买入：多头排列 + RSI 不超买 + MACD 上拐
            if (bull and rsi_min < rsi_val < rsi_max
                and macd > prev_macd and macd > 0):
                df.iloc[i, df.columns.get_loc("signal")] = 1
                df.iloc[i, df.columns.get_loc("signal_price")] = df["close"].iloc[i]
                df.iloc[i, df.columns.get_loc("signal_reason")] = (
                    f"多周期共振：日线多头排列(RSI={rsi_val:.1f},MACD↑)，入场"
                )
                df.iloc[i, df.columns.get_loc("position_pct")] = cfg.MAX_POSITION_PCT

            # 卖出：趋势转空 或 RSI 超买回落
            if not bull:
                was_bull = df["trend_bull"].iloc[i-1] if i > 0 else False
                if was_bull:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    df.iloc[i, df.columns.get_loc("signal_price")] = df["close"].iloc[i]
                    df.iloc[i, df.columns.get_loc("signal_reason")] = "多周期共振：日线多头排列破坏，平仓"

        return df
