"""
策略7: 趋势骑乘
================
专为趋势行情设计，核心思想：
  - 用多级均线判断趋势强度（MA5 > MA20 > MA60 = 强趋势）
  - 回调到 MA20 附近时入场（不追高）
  - 跌破 MA60 或 ATR 止损才离场
  - 趋势加速时可以加仓

这个策略不追求频繁交易，而是抓住大趋势持有到底。
"""

import pandas as pd
import numpy as np
from .base import Strategy
import config as cfg


class TrendRiderStrategy(Strategy):
    name = "trend_rider"
    description = "趋势骑乘：回调入场+均线止损+趋势加速加仓"

    def __init__(self, params: dict = None):
        defaults = {
            "fast": 5, "mid": 20, "slow": 60,
            "pullback_pct": 0.02,   # 回调到 MA20 的 2% 范围内
            "accelerate_threshold": 0.03,  # 3天内涨超阈值=加速
        }
        defaults.update(params or {})
        super().__init__(defaults)

    def generate_signals(self) -> pd.DataFrame:
        df = self._df
        f = self.params["fast"]
        m = self.params["mid"]
        s = self.params["slow"]
        pb_pct = self.params["pullback_pct"]

        df["ma_f"] = df["close"].rolling(f).mean()
        df["ma_m"] = df["close"].rolling(m).mean()
        df["ma_s"] = df["close"].rolling(s).mean()

        # ATR
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=14, adjust=False).mean()

        # RSI（用于回调确认）
        ret = close.diff()
        gain = ret.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-ret).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - 100 / (1 + rs)

        # 趋势强度得分
        df["trend_score"] = 0.0
        for i in range(s + 5, len(df)):
            score = 0
            if df["ma_f"].iloc[i] > df["ma_m"].iloc[i]: score += 1
            if df["ma_m"].iloc[i] > df["ma_s"].iloc[i]: score += 1
            if close.iloc[i] > df["ma_s"].iloc[i]: score += 1
            # MA20 斜率
            if not pd.isna(df["ma_m"].iloc[i]) and not pd.isna(df["ma_m"].iloc[i-5]):
                if df["ma_m"].iloc[i] > df["ma_m"].iloc[i-5]: score += 1
            df.iloc[i, df.columns.get_loc("trend_score")] = score / 4.0

        df["signal"] = 0
        df["signal_price"] = np.nan
        df["signal_reason"] = ""
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["position_pct"] = 0.0

        in_position = False
        entry_price = 0.0
        highest = 0.0
        stop_loss = 0.0

        for i in range(s + 10, len(df)):
            price = close.iloc[i]
            ma_mid = df["ma_m"].iloc[i]
            ma_slow = df["ma_s"].iloc[i]
            atr_val = df["atr"].iloc[i]
            rsi_val = df["rsi"].iloc[i]
            trend = df["trend_score"].iloc[i]

            if pd.isna(ma_mid) or pd.isna(ma_slow) or pd.isna(atr_val):
                continue

            if not in_position:
                # 入场条件：
                # 1. 趋势得分高（>=0.75）
                # 2. 价格回调到 MA20 附近
                # 3. RSI 不超买
                near_ma20 = abs(price - ma_mid) / ma_mid <= pb_pct
                trend_strong = trend >= 0.75
                rsi_ok = rsi_val < 65

                if trend_strong and near_ma20 and rsi_ok:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    df.iloc[i, df.columns.get_loc("signal_price")] = price
                    df.iloc[i, df.columns.get_loc("signal_reason")] = (
                        f"趋势骑乘：趋势得分{trend:.0%}，回调至MA{m}附近，RSI={rsi_val:.0f}"
                    )
                    df.iloc[i, df.columns.get_loc("position_pct")] = cfg.MAX_POSITION_PCT
                    sl = price - cfg.STOP_LOSS_ATR_MULT * atr_val
                    df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                    df.iloc[i, df.columns.get_loc("take_profit")] = price + cfg.TAKE_PROFIT_ATR_MULT * atr_val
                    in_position = True
                    entry_price = price
                    highest = price
                    stop_loss = sl
            else:
                # 持仓管理
                highest = max(highest, df["high"].iloc[i])
                if price > entry_price:
                    new_stop = highest - cfg.TRAILING_STOP_ATR_MULT * atr_val
                    stop_loss = max(stop_loss, new_stop)

                # 平仓条件
                hit_stop = price <= stop_loss
                broke_ma60 = price < ma_slow
                trend_reversed = trend < 0.25

                if hit_stop:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    df.iloc[i, df.columns.get_loc("signal_price")] = price
                    df.iloc[i, df.columns.get_loc("signal_reason")] = (
                        f"移动止损 {stop_loss:.3f}，平仓"
                    )
                    in_position = False
                elif broke_ma60 and trend_reversed:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    df.iloc[i, df.columns.get_loc("signal_price")] = price
                    df.iloc[i, df.columns.get_loc("signal_reason")] = (
                        f"跌破MA{s} + 趋势得分{trend:.0%}，平仓"
                    )
                    in_position = False

        self._df = df
        return df
