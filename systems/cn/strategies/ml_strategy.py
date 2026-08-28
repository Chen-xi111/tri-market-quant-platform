"""
策略6: ML 预测策略（重写版）
============================
改进：
  1. 降低阈值（0.52）让信号更活跃
  2. 增加特征：价格位置、波动率 regime、量价配合
  3. 状态保持：持仓中只看止损/止盈，不频繁反转
  4. 概率加权仓位
"""

import pandas as pd
import numpy as np
from .base import Strategy
from .alpha_factors import AlphaFactorStrategy
import config as cfg


class MLStrategy(Strategy):
    name = "ml_predict"
    description = "XGBoost 滚动训练：降阈值+状态保持+概率加权仓位"

    def __init__(self, params: dict = None):
        defaults = {
            "train_window": 200,
            "retrain_every": 15,
            "prob_threshold": 0.52,
            "min_samples": 100,
        }
        defaults.update(params or {})
        super().__init__(defaults)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        alpha = AlphaFactorStrategy()
        factors = alpha._compute_factors(df)
        close = df["close"]

        factors["close_position"] = np.where(
            (close.rolling(60).max() - close.rolling(60).min()) > 0,
            (close - close.rolling(60).min()) / (close.rolling(60).max() - close.rolling(60).min()),
            np.nan,
        )
        factors["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan)
        factors["amplitude"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        factors["turnover_proxy"] = df["volume"] / df["volume"].rolling(60).mean().replace(0, np.nan)

        for col in ["mom_5d", "mom_20d", "boll_pct", "rsi_norm"]:
            if col in factors.columns:
                factors[f"{col}_lag1"] = factors[col].shift(1)

        return factors

    def _make_labels(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        forward = df["close"].pct_change(horizon).shift(-horizon)
        return (forward > 0).astype(int)

    def generate_signals(self) -> pd.DataFrame:
        df = self._df.copy()
        if df is None or len(df) < 60:
            df["signal"] = 0; df["signal_price"] = np.nan
            df["signal_reason"] = ""; df["position_pct"] = 0.0; df["ml_prob"] = np.nan
            return df

        train_win = self.params["train_window"]
        retrain = self.params["retrain_every"]
        prob_thresh = self.params["prob_threshold"]
        min_samples = self.params["min_samples"]

        df["signal"] = 0
        df["signal_price"] = np.nan
        df["signal_reason"] = ""
        df["position_pct"] = 0.0
        df["ml_prob"] = np.nan

        try:
            features = self._build_features(df)
            labels = self._make_labels(df)
        except Exception:
            return df

        valid = features.dropna().index.intersection(labels.dropna().index)
        if len(valid) < min_samples:
            return df

        features = features.loc[valid].fillna(0)
        labels = labels.loc[valid]

        # ATR for stop loss
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=14, adjust=False).mean()

        model = None
        scaler = None
        last_train_end = -1

        # 状态机
        in_position = False
        entry_price = 0.0
        highest = 0.0
        stop_loss = 0.0

        from sklearn.preprocessing import StandardScaler

        for i in range(min(train_win, len(features)), len(features)):
            current_date = features.index[i]

            if i - last_train_end >= retrain or model is None:
                X_train = features.iloc[:i].values.astype(np.float64)
                y_train = labels.iloc[:i].values.astype(int)

                if len(np.unique(y_train)) < 2:
                    continue

                try:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    from xgboost import XGBClassifier
                    model = XGBClassifier(
                        n_estimators=80, max_depth=3,
                        learning_rate=0.05, subsample=0.8,
                        eval_metric='logloss', verbosity=0,
                        random_state=42, n_jobs=1,
                    )
                    model.fit(X_train, y_train)
                    last_train_end = i
                except Exception:
                    continue

            if model is None or scaler is None:
                continue

            try:
                X_pred = scaler.transform(features.iloc[i:i+1].values.astype(np.float64))
                prob = float(model.predict_proba(X_pred)[0, 1])
            except Exception:
                continue

            if current_date not in df.index:
                continue
            row_idx = df.index.get_loc(current_date)
            df.iloc[row_idx, df.columns.get_loc("ml_prob")] = prob

            price = df["close"].iloc[row_idx]
            atr_val = df["atr"].iloc[row_idx] if "atr" in df.columns else price * 0.02
            if pd.isna(atr_val):
                atr_val = price * 0.02

            if not in_position:
                if prob >= prob_thresh:
                    df.iloc[row_idx, df.columns.get_loc("signal")] = 1
                    df.iloc[row_idx, df.columns.get_loc("signal_price")] = price
                    df.iloc[row_idx, df.columns.get_loc("position_pct")] = min(
                        cfg.MAX_POSITION_PCT, max(0.1, (prob - 0.5) * 2 * cfg.MAX_POSITION_PCT)
                    )
                    df.iloc[row_idx, df.columns.get_loc("signal_reason")] = (
                        f"ML预测涨 {prob:.1%} (XGBoost)"
                    )
                    in_position = True
                    entry_price = price
                    highest = price
                    stop_loss = price - cfg.STOP_LOSS_ATR_MULT * atr_val
            else:
                # 持仓中
                highest = max(highest, df["high"].iloc[row_idx])
                if price > entry_price:
                    new_stop = highest - cfg.TRAILING_STOP_ATR_MULT * atr_val
                    stop_loss = max(stop_loss, new_stop)

                # 平仓：止损 或 概率反转
                if price <= stop_loss:
                    df.iloc[row_idx, df.columns.get_loc("signal")] = -1
                    df.iloc[row_idx, df.columns.get_loc("signal_price")] = price
                    df.iloc[row_idx, df.columns.get_loc("signal_reason")] = (
                        f"ML止损 {stop_loss:.3f}，平仓"
                    )
                    in_position = False
                elif prob <= (1 - prob_thresh):
                    df.iloc[row_idx, df.columns.get_loc("signal")] = -1
                    df.iloc[row_idx, df.columns.get_loc("signal_price")] = price
                    df.iloc[row_idx, df.columns.get_loc("signal_reason")] = (
                        f"ML预测跌 {1-prob:.1%}，平仓"
                    )
                    in_position = False

        self._df = df
        return df
