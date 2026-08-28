"""
综合信号聚合器
================
同时运行全部策略，投票产生统一的交易指导。

输出：
  - 综合方向（强烈买入/买入/中性/卖出/强烈卖出）
  - 各策略投票明细
  - 建议仓位
  - 止损止盈参考
  - 信心评分
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from .base import Signal, SignalType
from .ma_crossover import MACrossoverStrategy
from .bollinger_reversion import BollingerReversionStrategy
from .vol_breakout import VolBreakoutStrategy
from .multi_tf import MultiTFStrategy
from .alpha_factors import AlphaFactorStrategy
from .ml_strategy import MLStrategy
from .trend_rider import TrendRiderStrategy
import config as cfg


@dataclass
class CompositeSignal:
    """综合信号"""
    direction: str             # strong_buy / buy / neutral / sell / strong_sell
    score: float               # -10 ~ +10
    confidence: float          # 0 ~ 1
    buy_votes: int
    sell_votes: int
    neutral_votes: int
    recommended_position: float  # 建议仓位比例
    stop_loss: float
    take_profit: float
    strategy_details: List[dict] = field(default_factory=list)
    summary: str = ""


# 策略注册（权重越高贡献越大）
STRATEGY_REGISTRY = {
    "trend_rider":       (TrendRiderStrategy,     3.0, "趋势骑乘"),
    "ma_crossover":      (MACrossoverStrategy,    2.0, "双均线趋势"),
    "alpha_factors":     (AlphaFactorStrategy,    2.5, "Alpha多因子"),
    "multi_tf":          (MultiTFStrategy,        2.0, "多周期共振"),
    "ml_predict":        (MLStrategy,             1.5, "XGBoost"),
    "bollinger_reversion": (BollingerReversionStrategy, 1.0, "布林带回归"),
    "vol_breakout":      (VolBreakoutStrategy,    1.5, "量价突破"),
}


class CompositeStrategy:
    """多策略投票聚合器"""

    def __init__(self, params: dict = None):
        self.params = params or {}
        self._strategies: Dict[str, Tuple] = {}

        for key, (cls, weight, label) in STRATEGY_REGISTRY.items():
            self._strategies[key] = (cls(), weight, label)

    def evaluate(self, df: pd.DataFrame) -> CompositeSignal:
        """
        对 DataFrame 运行全部策略，聚合投票
        """
        if df is None or len(df) < 60:
            return CompositeSignal(
                direction="neutral", score=0, confidence=0,
                buy_votes=0, sell_votes=0, neutral_votes=len(self._strategies),
                recommended_position=0, stop_loss=0, take_profit=0,
                summary="数据不足"
            )

        price = df["close"].iloc[-1]
        atr_val = self._get_atr(df)

        votes = []          # (strategy_name, direction, weight, reason)
        buy_score = 0.0
        sell_score = 0.0
        buy_count = 0
        sell_count = 0
        neutral_count = 0
        stop_losses = []
        take_profits = []
        details = []

        for key, (strategy, weight, label) in self._strategies.items():
            try:
                strategy.set_data(df.copy())
                sdf = strategy.generate_signals()
                sigs = strategy.get_signals_list()

                # 只使用最近 5 根 K 线内的信号。历史上曾经出现过的信号
                # 不能继续冒充“当前信号”，否则切股后会给出过期交易建议。
                recent_sig = None
                if sigs:
                    cutoff = df.index[-5] if len(df) > 5 else df.index[0]
                    recent = [s for s in sigs if s.date >= cutoff]
                    if recent:
                        recent_sig = recent[-1]

                if recent_sig and recent_sig.type == SignalType.BUY:
                    direction = "buy"
                    buy_score += weight
                    buy_count += 1
                    if recent_sig.stop_loss:
                        stop_losses.append(recent_sig.stop_loss)
                    if recent_sig.take_profit:
                        take_profits.append(recent_sig.take_profit)
                    reason = recent_sig.reason[:40]
                elif recent_sig and recent_sig.type == SignalType.SELL:
                    direction = "sell"
                    sell_score += weight
                    sell_count += 1
                    reason = recent_sig.reason[:40]
                else:
                    direction = "neutral"
                    neutral_count += 1
                    # 检查当前趋势方向
                    if "ma_f" in sdf.columns and "ma_m" in sdf.columns:
                        ma_f = sdf["ma_f"].iloc[-1]
                        ma_m = sdf["ma_m"].iloc[-1]
                        if not pd.isna(ma_f) and not pd.isna(ma_m):
                            if ma_f > ma_m:
                                direction = "lean_buy"
                                buy_score += weight * 0.3
                            else:
                                direction = "lean_sell"
                                sell_score += weight * 0.3
                    reason = "无近期信号"

                votes.append((label, direction, weight, reason))
                details.append({
                    "策略": label,
                    "方向": direction,
                    "权重": weight,
                    "原因": reason,
                })

            except Exception as e:
                neutral_count += 1
                details.append({
                    "策略": label, "方向": "error",
                    "权重": weight, "原因": f"计算失败: {str(e)[:30]}",
                })

        # 综合评分
        total_weight = sum(w for _, w, _ in self._strategies.values())
        score = (buy_score - sell_score) / total_weight * 10  # -10 ~ +10

        # 方向判断
        if score >= 5:
            direction = "strong_buy"
        elif score >= 2:
            direction = "buy"
        elif score > -2:
            direction = "neutral"
        elif score > -5:
            direction = "sell"
        else:
            direction = "strong_sell"

        # 信心评分
        total_active = buy_count + sell_count
        confidence = total_active / len(self._strategies) if self._strategies else 0

        # 建议仓位
        if direction in ("strong_buy", "buy"):
            recommended = min(cfg.MAX_POSITION_PCT, abs(score) / 10 * cfg.MAX_POSITION_PCT)
        else:
            recommended = 0.0

        # 止损止盈（取最保守的）
        sl = max(stop_losses) if stop_losses else (price - cfg.STOP_LOSS_ATR_MULT * atr_val)
        tp = min(take_profits) if take_profits else (price + cfg.TAKE_PROFIT_ATR_MULT * atr_val)

        # 汇总文本
        dir_emoji = {
            "strong_buy": "🟢🟢 强烈买入",
            "buy": "🟢 偏多买入",
            "neutral": "⚪ 中性观望",
            "sell": "🔴 偏空卖出",
            "strong_sell": "🔴🔴 强烈卖出",
        }
        summary = (
            f"{dir_emoji.get(direction, direction)} | "
            f"评分 {score:+.1f}/10 | "
            f"买{buy_count} 卖{sell_count} 中性{neutral_count} | "
            f"信心 {confidence:.0%}"
        )

        return CompositeSignal(
            direction=direction,
            score=round(score, 1),
            confidence=round(confidence, 2),
            buy_votes=buy_count,
            sell_votes=sell_count,
            neutral_votes=neutral_count,
            recommended_position=round(recommended, 3),
            stop_loss=sl,
            take_profit=tp,
            strategy_details=details,
            summary=summary,
        )

    @staticmethod
    def _get_atr(df: pd.DataFrame) -> float:
        if "ATR" in df.columns:
            val = df["ATR"].iloc[-1]
            if not pd.isna(val):
                return val
        # 手动计算
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        val = atr.iloc[-1]
        return val if not pd.isna(val) else close.iloc[-1] * 0.02
