"""
交易信号引擎 — 多指标综合评分，生成买卖信号
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import config as cfg


@dataclass
class Signal:
    """单个交易信号"""
    name: str
    direction: str        # "bull" / "bear"
    weight: float
    description: str
    date: pd.Timestamp = None


@dataclass
class ScoreResult:
    """综合评分结果"""
    total_score: float
    direction: str        # "strong_buy" / "buy" / "neutral" / "sell" / "strong_sell"
    signals: List[Signal] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class SignalEngine:
    """多指标信号引擎"""

    def __init__(self, df: pd.DataFrame):
        """
        df 需已包含所有技术指标列（通过 IndicatorCalculator 计算）
        """
        self.df = df.copy()
        self.latest = df.iloc[-1] if not df.empty else None
        self.prev = df.iloc[-2] if len(df) > 1 else None

    # ================================================================
    #  均线信号
    # ================================================================
    def _ma_signals(self) -> List[Signal]:
        signals = []
        if self.latest is None or self.prev is None:
            return signals

        close = self.latest["close"]

        # 均线支撑 / 压力
        for p in cfg.MA_PERIODS:
            col = f"MA{p}"
            if col not in self.df.columns:
                continue
            ma_val = self.latest[col]
            if pd.isna(ma_val):
                continue
            if close > ma_val:
                signals.append(Signal(
                    name=f"MA{p}_支撑",
                    direction="bull",
                    weight=cfg.SIGNAL_WEIGHTS["ma_support"] / max(len(cfg.MA_PERIODS), 1),
                    description=f"价格在MA{p}({ma_val:.3f})上方，均线支撑",
                ))
            else:
                signals.append(Signal(
                    name=f"MA{p}_压力",
                    direction="bear",
                    weight=cfg.SIGNAL_WEIGHTS["ma_resistance"] / max(len(cfg.MA_PERIODS), 1),
                    description=f"价格在MA{p}({ma_val:.3f})下方，均线压力",
                ))

        # 金叉 / 死叉（短均上穿/下穿长均）
        short_ma = cfg.MA_PERIODS[0]
        long_ma = cfg.MA_PERIODS[2]  # MA20
        if f"MA{short_ma}" in self.df.columns and f"MA{long_ma}" in self.df.columns:
            prev_short = self.prev[f"MA{short_ma}"]
            prev_long = self.prev[f"MA{long_ma}"]
            cur_short = self.latest[f"MA{short_ma}"]
            cur_long = self.latest[f"MA{long_ma}"]
            if (not pd.isna(cur_short) and not pd.isna(cur_long) and
                not pd.isna(prev_short) and not pd.isna(prev_long)):
                if prev_short <= prev_long and cur_short > cur_long:
                    signals.append(Signal(
                        name="MA_金叉",
                        direction="bull",
                        weight=cfg.SIGNAL_WEIGHTS["ma_golden_cross"],
                        description=f"MA{short_ma}上穿MA{long_ma}，金叉信号",
                    ))
                elif prev_short >= prev_long and cur_short < cur_long:
                    signals.append(Signal(
                        name="MA_死叉",
                        direction="bear",
                        weight=cfg.SIGNAL_WEIGHTS["ma_death_cross"],
                        description=f"MA{short_ma}下穿MA{long_ma}，死叉信号",
                    ))

        return signals

    # ================================================================
    #  MACD 信号
    # ================================================================
    def _macd_signals(self) -> List[Signal]:
        signals = []
        if self.latest is None or self.prev is None:
            return signals

        if "MACD_DIF" not in self.df.columns:
            return signals

        dif = self.latest["MACD_DIF"]
        dea = self.latest["MACD_DEA"]
        hist = self.latest["MACD_HIST"]
        prev_dif = self.prev["MACD_DIF"]
        prev_dea = self.prev["MACD_DEA"]

        if pd.isna(dif) or pd.isna(dea):
            return signals

        # DIF 与 DEA 关系
        if dif > dea:
            signals.append(Signal(
                name="MACD_多头",
                direction="bull",
                weight=0.5,
                description=f"DIF({dif:.4f}) > DEA({dea:.4f})，MACD多头排列",
            ))
        else:
            signals.append(Signal(
                name="MACD_空头",
                direction="bear",
                weight=-0.5,
                description=f"DIF({dif:.4f}) < DEA({dea:.4f})，MACD空头排列",
            ))

        # 金叉 / 死叉
        if (not pd.isna(prev_dif) and not pd.isna(prev_dea) and
            not pd.isna(dif) and not pd.isna(dea)):
            if prev_dif <= prev_dea and dif > dea:
                signals.append(Signal(
                    name="MACD_金叉",
                    direction="bull",
                    weight=cfg.SIGNAL_WEIGHTS["macd_golden_cross"],
                    description="MACD DIF上穿DEA，金叉信号",
                ))
            elif prev_dif >= prev_dea and dif < dea:
                signals.append(Signal(
                    name="MACD_死叉",
                    direction="bear",
                    weight=cfg.SIGNAL_WEIGHTS["macd_death_cross"],
                    description="MACD DIF下穿DEA，死叉信号",
                ))

        # 柱状图方向
        if not pd.isna(hist):
            prev_hist = self.prev["MACD_HIST"]
            if not pd.isna(prev_hist):
                if hist > 0 and prev_hist <= 0:
                    signals.append(Signal(
                        name="MACD_转正",
                        direction="bull",
                        weight=1.0,
                        description="MACD柱由负转正，动能转多",
                    ))
                elif hist < 0 and prev_hist >= 0:
                    signals.append(Signal(
                        name="MACD_转负",
                        direction="bear",
                        weight=-1.0,
                        description="MACD柱由正转负，动能转空",
                    ))

        return signals

    # ================================================================
    #  RSI 信号
    # ================================================================
    def _rsi_signals(self) -> List[Signal]:
        signals = []
        if self.latest is None or "RSI" not in self.df.columns:
            return signals

        rsi = self.latest["RSI"]
        if pd.isna(rsi):
            return signals

        if rsi <= cfg.RSI_OVERSOLD:
            signals.append(Signal(
                name="RSI_超卖",
                direction="bull",
                weight=cfg.SIGNAL_WEIGHTS["rsi_oversold"],
                description=f"RSI({rsi:.1f}) ≤ {cfg.RSI_OVERSOLD}，进入超卖区",
            ))
        elif rsi >= cfg.RSI_OVERBOUGHT:
            signals.append(Signal(
                name="RSI_超买",
                direction="bear",
                weight=cfg.SIGNAL_WEIGHTS["rsi_overbought"],
                description=f"RSI({rsi:.1f}) ≥ {cfg.RSI_OVERBOUGHT}，进入超买区",
            ))
        else:
            # 倾向性
            if rsi > 50:
                signals.append(Signal(
                    name="RSI_偏多",
                    direction="bull",
                    weight=0.3,
                    description=f"RSI({rsi:.1f}) > 50，偏多",
                ))
            else:
                signals.append(Signal(
                    name="RSI_偏空",
                    direction="bear",
                    weight=-0.3,
                    description=f"RSI({rsi:.1f}) < 50，偏空",
                ))

        return signals

    # ================================================================
    #  KDJ 信号
    # ================================================================
    def _kdj_signals(self) -> List[Signal]:
        signals = []
        if self.latest is None or "KDJ_K" not in self.df.columns:
            return signals

        k = self.latest["KDJ_K"]
        d = self.latest["KDJ_D"]
        j = self.latest["KDJ_J"]
        if pd.isna(k) or pd.isna(d) or pd.isna(j):
            return signals

        if j < 0:
            signals.append(Signal(
                name="KDJ_超卖",
                direction="bull",
                weight=cfg.SIGNAL_WEIGHTS["kdj_oversold"],
                description=f"KDJ J值({j:.1f}) < 0，超卖钝化",
            ))
        elif j > 100:
            signals.append(Signal(
                name="KDJ_超买",
                direction="bear",
                weight=cfg.SIGNAL_WEIGHTS["kdj_overbought"],
                description=f"KDJ J值({j:.1f}) > 100，超买钝化",
            ))

        # K/D 金叉死叉
        if self.prev is not None:
            prev_k = self.prev["KDJ_K"]
            prev_d = self.prev["KDJ_D"]
            if not pd.isna(prev_k) and not pd.isna(prev_d):
                if prev_k <= prev_d and k > d:
                    signals.append(Signal(
                        name="KDJ_金叉",
                        direction="bull",
                        weight=1.0,
                        description="KDJ K线上穿D线，金叉",
                    ))
                elif prev_k >= prev_d and k < d:
                    signals.append(Signal(
                        name="KDJ_死叉",
                        direction="bear",
                        weight=-1.0,
                        description="KDJ K线下穿D线，死叉",
                    ))

        return signals

    # ================================================================
    #  布林带信号
    # ================================================================
    def _bollinger_signals(self) -> List[Signal]:
        signals = []
        if self.latest is None or "BOLL_UPPER" not in self.df.columns:
            return signals

        close = self.latest["close"]
        upper = self.latest["BOLL_UPPER"]
        lower = self.latest["BOLL_LOWER"]
        mid = self.latest["BOLL_MID"]
        pct = self.latest.get("BOLL_PCT")

        if pd.isna(upper) or pd.isna(lower):
            return signals

        # 触及上轨 / 下轨
        if close >= upper * 0.98:
            signals.append(Signal(
                name="BOLL_触上轨",
                direction="bear",
                weight=cfg.SIGNAL_WEIGHTS["boll_upper_touch"],
                description=f"价格({close:.3f})触及布林上轨({upper:.3f})，压力位",
            ))
        if close <= lower * 1.02:
            signals.append(Signal(
                name="BOLL_触下轨",
                direction="bull",
                weight=cfg.SIGNAL_WEIGHTS["boll_lower_touch"],
                description=f"价格({close:.3f})触及布林下轨({lower:.3f})，支撑位",
            ))

        # 中轨位置
        if close > mid:
            signals.append(Signal(
                name="BOLL_中轨上",
                direction="bull",
                weight=0.5,
                description=f"价格在布林中轨({mid:.3f})上方，偏强",
            ))
        else:
            signals.append(Signal(
                name="BOLL_中轨下",
                direction="bear",
                weight=-0.5,
                description=f"价格在布林中轨({mid:.3f})下方，偏弱",
            ))

        # 带宽收缩（可能变盘）
        if "BOLL_WIDTH" in self.df.columns and len(self.df) >= 20:
            widths = self.df["BOLL_WIDTH"].dropna()
            if len(widths) >= 20:
                cur_width = widths.iloc[-1]
                min_width = widths.iloc[-20:].min()
                if cur_width <= min_width * 1.05:
                    signals.append(Signal(
                        name="BOLL_收窄",
                        direction="neutral",
                        weight=0,
                        description=f"布林带收窄，可能变盘",
                    ))

        return signals

    # ================================================================
    #  成交量信号
    # ================================================================
    def _volume_signals(self) -> List[Signal]:
        signals = []
        if self.latest is None or "VOL_SURGE" not in self.df.columns:
            return signals

        surge = self.latest["VOL_SURGE"]
        ratio = self.latest.get("VOL_RATIO", 1.0)

        if surge == "bull":
            signals.append(Signal(
                name="VOL_放量上涨",
                direction="bull",
                weight=cfg.SIGNAL_WEIGHTS["vol_surge_bull"],
                description=f"放量上涨，量比{ratio:.1f}倍",
            ))
        elif surge == "bear":
            signals.append(Signal(
                name="VOL_放量下跌",
                direction="bear",
                weight=cfg.SIGNAL_WEIGHTS["vol_surge_bear"],
                description=f"放量下跌，量比{ratio:.1f}倍",
            ))

        return signals

    # ================================================================
    #  综合评分
    # ================================================================
    def evaluate(self) -> ScoreResult:
        """综合评估所有信号，返回评分结果"""
        all_signals: List[Signal] = []
        all_signals.extend(self._ma_signals())
        all_signals.extend(self._macd_signals())
        all_signals.extend(self._rsi_signals())
        all_signals.extend(self._kdj_signals())
        all_signals.extend(self._bollinger_signals())
        all_signals.extend(self._volume_signals())

        total = sum(s.weight for s in all_signals)

        # 判断方向
        if total >= cfg.SCORE_STRONG_BUY:
            direction = "strong_buy"
        elif total >= cfg.SCORE_BUY:
            direction = "buy"
        elif total > cfg.SCORE_SELL:
            direction = "neutral"
        elif total > cfg.SCORE_STRONG_SELL:
            direction = "sell"
        else:
            direction = "strong_sell"

        # 按方向分组
        bull_signals = [s for s in all_signals if s.direction == "bull"]
        bear_signals = [s for s in all_signals if s.direction == "bear"]

        return ScoreResult(
            total_score=round(total, 2),
            direction=direction,
            signals=all_signals,
            details={
                "bull_count": len(bull_signals),
                "bear_count": len(bear_signals),
                "bull_weight": round(sum(s.weight for s in bull_signals), 2),
                "bear_weight": round(sum(s.weight for s in bear_signals), 2),
                "current_price": self.latest["close"] if self.latest is not None else None,
                "date": self.latest.name if self.latest is not None else None,
            },
        )


# ================================================================
#  便捷函数
# ================================================================
def evaluate_signals(df: pd.DataFrame) -> ScoreResult:
    """快速评估信号"""
    return SignalEngine(df).evaluate()


def signal_summary(df: pd.DataFrame) -> str:
    """生成信号摘要文本"""
    result = evaluate_signals(df)
    dir_map = {
        "strong_buy": "🟢 强烈买入",
        "buy": "🟢 偏多/买入",
        "neutral": "⚪ 中性/观望",
        "sell": "🔴 偏空/卖出",
        "strong_sell": "🔴 强烈卖出",
    }
    lines = [
        f"📊 综合评分: {result.total_score:+.1f} — {dir_map.get(result.direction, result.direction)}",
        f"   多头信号: {result.details['bull_count']}个 (权重+{result.details['bull_weight']})",
        f"   空头信号: {result.details['bear_count']}个 (权重{result.details['bear_weight']})",
        f"   当前价格: {result.details['current_price']:.3f}" if result.details['current_price'] else "",
        f"   日期: {result.details['date']}" if result.details['date'] else "",
        "",
        "━━ 信号明细 ━━",
    ]
    for s in result.signals:
        emoji = "🟢" if s.direction == "bull" else ("🔴" if s.direction == "bear" else "⚪")
        lines.append(f"  {emoji} [{s.name}] W{s.weight:+.1f} — {s.description}")

    return "\n".join(lines)


if __name__ == "__main__":
    from data_fetcher import quick_history
    from indicators import compute_all

    df = quick_history("1d", 365)
    df = compute_all(df)
    print(signal_summary(df))
