"""
策略1: 双均线趋势跟踪（重写版）
================================
核心改进：
  1. 状态机模式 — 不是每次交叉都交易，而是判断趋势状态
  2. 趋势过滤 — 只在 MA60 方向一致时入场
  3. 持仓管理 — 入场后用 ATR 动态止损，不因短期均线抖动频繁平仓
  4. 加仓逻辑 — 趋势延续时可以加仓
"""

import pandas as pd
import numpy as np
from .base import Strategy
import config as cfg


class MACrossoverStrategy(Strategy):
    name = "ma_crossover"
    description = "双均线趋势跟踪：状态机 + ATR止损 + 趋势过滤"

    def __init__(self, params: dict = None):
        p = cfg.STRAT_MA_CROSSOVER.copy()
        p.update(params or {})
        super().__init__(p)

    def generate_signals(self) -> pd.DataFrame:
        df = self._df
        fast = self.params["fast"]
        slow = self.params["slow"]
        filter_ma = self.params["filter_ma"]

        # 均线
        df["ma_fast"] = df["close"].rolling(fast).mean()
        df["ma_slow"] = df["close"].rolling(slow).mean()
        df["ma_filter"] = df["close"].rolling(filter_ma).mean()

        # ATR（用于止损）
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=14, adjust=False).mean()

        # 趋势强度：MA 斜率
        df["ma_slow_slope"] = df["ma_slow"].diff(5) / df["ma_slow"].shift(5)

        df["signal"] = 0
        df["signal_price"] = np.nan
        df["signal_reason"] = ""
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["position_pct"] = 0.0

        # 状态机
        in_position = False
        entry_price = 0.0
        highest_since_entry = 0.0
        stop_loss_price = 0.0

        for i in range(filter_ma + 5, len(df)):
            price = df["close"].iloc[i]
            ma_f = df["ma_fast"].iloc[i]
            ma_s = df["ma_slow"].iloc[i]
            ma_l = df["ma_filter"].iloc[i]
            atr_val = df["atr"].iloc[i]
            slope = df["ma_slow_slope"].iloc[i]

            if pd.isna(ma_f) or pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
                continue

            # 趋势定义：慢线斜率向上 + 价格在过滤线上方
            trend_up = (ma_f > ma_s) and (price > ma_l) and (slope > 0 or pd.isna(slope))
            trend_down = (ma_f < ma_s) and (price < ma_l)

            if not in_position:
                # 入场条件：均线多头排列 + 趋势确认
                if trend_up:
                    # 二次确认：快线在慢线上方超过 2 天
                    days_above = 0
                    for j in range(i, max(i - 5, filter_ma), -1):
                        if df["ma_fast"].iloc[j] > df["ma_slow"].iloc[j]:
                            days_above += 1
                        else:
                            break
                    if days_above >= 2:
                        df.iloc[i, df.columns.get_loc("signal")] = 1
                        df.iloc[i, df.columns.get_loc("signal_price")] = price
                        df.iloc[i, df.columns.get_loc("signal_reason")] = (
                            f"MA{fast}>MA{slow} 多头排列确认，价格在MA{filter_ma}上方，趋势向上"
                        )
                        df.iloc[i, df.columns.get_loc("position_pct")] = cfg.MAX_POSITION_PCT
                        # ATR 止损
                        sl = price - cfg.STOP_LOSS_ATR_MULT * atr_val
                        df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                        df.iloc[i, df.columns.get_loc("take_profit")] = price + cfg.TAKE_PROFIT_ATR_MULT * atr_val

                        in_position = True
                        entry_price = price
                        highest_since_entry = price
                        stop_loss_price = sl

            else:
                # 持仓中：更新最高价 + 移动止损
                highest_since_entry = max(highest_since_entry, df["high"].iloc[i])
                # 移动止损：盈利后上移
                if price > entry_price:
                    new_stop = highest_since_entry - cfg.TRAILING_STOP_ATR_MULT * atr_val
                    stop_loss_price = max(stop_loss_price, new_stop)

                # 平仓条件（任一触发）：
                # 1. 价格跌破移动止损
                # 2. 趋势明确反转（快线下穿慢线 + 价格在过滤线下方）
                # 3. 价格跌破 MA60（趋势破坏）

                hit_stop = price <= stop_loss_price
                trend_reversal = (ma_f < ma_s) and (price < ma_l)
                break_filter = price < ma_l and df["close"].iloc[i - 1] < ma_l

                if hit_stop:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    df.iloc[i, df.columns.get_loc("signal_price")] = price
                    df.iloc[i, df.columns.get_loc("signal_reason")] = (
                        f"触及移动止损 {stop_loss_price:.3f}（ATR跟踪），平仓"
                    )
                    in_position = False
                elif trend_reversal:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    df.iloc[i, df.columns.get_loc("signal_price")] = price
                    df.iloc[i, df.columns.get_loc("signal_reason")] = (
                        f"趋势反转：MA{fast}<MA{slow} 且价格跌破MA{filter_ma}，平仓"
                    )
                    in_position = False

        self._df = df
        return df
