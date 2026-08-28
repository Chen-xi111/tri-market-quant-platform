"""
策略5: Alpha 多因子模型（重写版）
==================================
核心改进：
  1. 状态保持 — 不是每次穿越阈值都交易，而是维持仓位状态
  2. 动态阈值 — 根据近期 alpha 分位数自适应
  3. 信号质量评分 — 综合因子一致性打分
  4. 持仓管理 — 入场后用 ATR 止损，alpha 转弱才平仓
"""

import pandas as pd
import numpy as np
from .base import Strategy
import config as cfg


class AlphaFactorStrategy(Strategy):
    name = "alpha_factors"
    description = "Alpha 多因子：动态阈值 + 状态保持 + 因子一致性评分"

    def __init__(self, params: dict = None):
        defaults = {
            "lookback": 120,
            "entry_percentile": 0.70,   # alpha 在近期 70 分位以上入场
            "exit_percentile": 0.35,    # alpha 跌到 35 分位以下平仓
            "smooth_window": 5,
            "min_factor_agree": 0.5,    # 至少 50% 因子方向一致
        }
        defaults.update(params or {})
        super().__init__(defaults)

    # ================================================================
    #  因子计算
    # ================================================================
    def _compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        close = df["close"]
        ret = close.pct_change()

        # 动量因子
        f["mom_5d"] = close.pct_change(5)
        f["mom_10d"] = close.pct_change(10)
        f["mom_20d"] = close.pct_change(20)
        f["mom_60d"] = close.pct_change(60)
        f["rev_2d"] = -close.pct_change(2)

        # 波动率因子（低波动有利）
        f["vol_5d"] = -ret.rolling(5).std()
        f["vol_20d"] = -ret.rolling(20).std()
        f["vol_ratio"] = -(ret.rolling(20).std() / ret.rolling(60).std().replace(0, np.nan))

        # ATR 比率
        high, low = df["high"], df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        f["atr_ratio"] = -atr / close

        # 趋势因子
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        f["adx"] = dx.rolling(14).mean() / 100

        # MA 排列得分
        f["ma_score"] = 0.0
        for i in range(60, len(df)):
            ma5 = close.rolling(5).mean().iloc[i]
            ma20 = close.rolling(20).mean().iloc[i]
            ma60 = close.rolling(60).mean().iloc[i]
            if not any(pd.isna(x) for x in [ma5, ma20, ma60]):
                score = 0
                if ma5 > ma20: score += 1
                if ma20 > ma60: score += 1
                if close.iloc[i] > ma60: score += 1
                f.iloc[i, f.columns.get_loc("ma_score")] = score / 3.0

        # 量能因子
        f["vol_chg_5d"] = df["volume"].pct_change(5)
        f["vol_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)

        # 技术因子
        gain = ret.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-ret).clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        f["rsi_norm"] = (rsi - 50) / 50

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        f["macd_dir"] = np.sign(dif.diff())

        boll_mid = close.rolling(20).mean()
        boll_std = close.rolling(20).std()
        f["boll_pct"] = (close - (boll_mid - 2*boll_std)) / ((4*boll_std).replace(0, np.nan)) - 0.5

        return f

    # ================================================================
    #  因子合成
    # ================================================================
    def _synthesize_alpha(self, factors: pd.DataFrame) -> tuple:
        """返回 (alpha_series, factor_directions_df)"""
        lookback = self.params["lookback"]

        factor_weights = {
            "mom_5d": 1.5, "mom_10d": 1.0, "mom_20d": 0.8,
            "rev_2d": 0.3,
            "vol_5d": 0.5, "vol_ratio": 0.4,
            "adx": 0.3, "ma_score": 0.8,
            "vol_ma_ratio": 0.3,
            "rsi_norm": 0.4, "macd_dir": 0.3, "boll_pct": 0.6,
        }

        alpha = pd.Series(0.0, index=factors.index)
        directions = pd.DataFrame(index=factors.index)

        for name, weight in factor_weights.items():
            if name not in factors.columns:
                continue
            series = factors[name].copy()
            rolling_mean = series.rolling(lookback, min_periods=30).mean()
            rolling_std = series.rolling(lookback, min_periods=30).std().replace(0, np.nan)
            z = (series - rolling_mean) / rolling_std
            z = z.clip(-3, 3)
            alpha += z.fillna(0) * weight
            directions[name] = (z > 0).astype(float)

        # 平滑
        alpha = alpha.rolling(self.params["smooth_window"], min_periods=1).mean()

        # 标准化
        a_mean = alpha.rolling(lookback, min_periods=30).mean()
        a_std = alpha.rolling(lookback, min_periods=30).std().replace(0, np.nan)
        alpha = (alpha - a_mean) / a_std

        # 因子一致性：多少比例的因子方向一致（看多）
        factor_agreement = directions.mean(axis=1)

        return alpha, factor_agreement

    # ================================================================
    #  生成信号
    # ================================================================
    def generate_signals(self) -> pd.DataFrame:
        df = self._df.copy()

        factors = self._compute_factors(df)
        alpha, agreement = self._synthesize_alpha(factors)
        df["alpha"] = alpha
        df["factor_agreement"] = agreement

        # 动态阈值：用近期分位数
        lookback = self.params["lookback"]
        entry_pct = self.params["entry_percentile"]
        exit_pct = self.params["exit_percentile"]
        min_agree = self.params["min_factor_agree"]

        entry_threshold = alpha.rolling(lookback, min_periods=30).quantile(entry_pct)
        exit_threshold = alpha.rolling(lookback, min_periods=30).quantile(exit_pct)

        # ATR
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=14, adjust=False).mean()

        df["signal"] = 0
        df["signal_price"] = np.nan
        df["signal_reason"] = ""
        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan
        df["position_pct"] = 0.0

        # 状态机
        in_position = False
        entry_price = 0.0
        highest = 0.0
        stop_loss = 0.0

        for i in range(lookback, len(df)):
            a = alpha.iloc[i]
            ag = agreement.iloc[i]
            price = close.iloc[i]
            atr_val = df["atr"].iloc[i]
            et = entry_threshold.iloc[i]
            xt = exit_threshold.iloc[i]

            if pd.isna(a) or pd.isna(et) or pd.isna(xt) or pd.isna(atr_val):
                continue

            if not in_position:
                # 入场：alpha 超过动态阈值 + 因子一致性达标
                if a > et and ag >= min_agree:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    df.iloc[i, df.columns.get_loc("signal_price")] = price
                    agree_pct = ag * 100
                    df.iloc[i, df.columns.get_loc("signal_reason")] = (
                        f"Alpha={a:.2f}σ > 动态阈值{et:.2f}，{agree_pct:.0f}%因子看多"
                    )
                    pos = min(cfg.MAX_POSITION_PCT, max(0.15, abs(a) / 3.0 * cfg.MAX_POSITION_PCT))
                    df.iloc[i, df.columns.get_loc("position_pct")] = pos
                    sl = price - cfg.STOP_LOSS_ATR_MULT * atr_val
                    df.iloc[i, df.columns.get_loc("stop_loss")] = sl
                    df.iloc[i, df.columns.get_loc("take_profit")] = price + cfg.TAKE_PROFIT_ATR_MULT * atr_val
                    in_position = True
                    entry_price = price
                    highest = price
                    stop_loss = sl
            else:
                # 持仓：更新移动止损
                highest = max(highest, df["high"].iloc[i])
                if price > entry_price:
                    new_stop = highest - cfg.TRAILING_STOP_ATR_MULT * atr_val
                    stop_loss = max(stop_loss, new_stop)

                # 平仓条件
                hit_stop = price <= stop_loss
                alpha_weak = a < xt  # alpha 跌到退出阈值
                factors_diverge = ag < (1 - min_agree)  # 因子方向分歧大

                if hit_stop:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    df.iloc[i, df.columns.get_loc("signal_price")] = price
                    df.iloc[i, df.columns.get_loc("signal_reason")] = (
                        f"移动止损 {stop_loss:.3f}，平仓"
                    )
                    in_position = False
                elif alpha_weak and factors_diverge:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    df.iloc[i, df.columns.get_loc("signal_price")] = price
                    df.iloc[i, df.columns.get_loc("signal_reason")] = (
                        f"Alpha转弱({a:.2f}<{xt:.2f}) + 因子分歧({ag*100:.0f}%)，平仓"
                    )
                    in_position = False

        # 保存因子明细
        for col in ["mom_5d", "mom_20d", "adx", "ma_score", "boll_pct", "rsi_norm"]:
            if col in factors.columns:
                df[f"factor_{col}"] = factors[col]

        self._df = df
        return df


# ============================================================
#  IC 分析
# ============================================================
def factor_ic_analysis(df: pd.DataFrame, forward_period: int = 5) -> dict:
    strat = AlphaFactorStrategy()
    factors = strat._compute_factors(df)
    forward_ret = df["close"].pct_change(forward_period).shift(-forward_period)

    ics = {}
    for col in factors.columns:
        valid = factors[col].dropna().index.intersection(forward_ret.dropna().index)
        if len(valid) < 30:
            continue
        ic = factors[col].loc[valid].rank().corr(forward_ret.loc[valid].rank())
        ics[col] = round(ic, 4)

    return ics
