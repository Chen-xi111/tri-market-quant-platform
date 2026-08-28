"""用防泄漏的扩展窗口验证，为每日候选提供样本外证据。

这个模块不再制造一张新的“策略票”。它回答三个更严格的问题：

1. 仅用当时可见的 OHLCV 特征，能否在未见样本中区分未来 10 日方向；
2. 概率预测是否优于每个训练窗口的历史胜率基线；
3. 高概率信号在计入 A 股交易费和滑点后，是否仍有正的历史净收益。

资金流历史目前尚不足以进入训练集，因此刻意不把当日资金排名回填到历史特征中。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg


FEATURE_LABELS = {
    "momentum_5d": "5日动量",
    "momentum_20d": "20日动量",
    "momentum_60d": "60日动量",
    "price_ma20_gap": "价格/MA20",
    "ma20_ma60_gap": "MA20/MA60",
    "rsi_centered": "RSI位置",
    "macd_atr": "MACD/ATR",
    "atr_pct": "波动率",
    "log_volume_ratio": "量比",
    "breakout_20d": "20日突破距离",
    "drawdown_60d": "60日回撤",
    "intraday_range": "日内振幅",
}


@dataclass(frozen=True)
class ResearchEvidence:
    """单只股票的样本外验证摘要。"""

    status: str
    gate_pass: bool
    current_signal: str
    current_probability: float | None = None
    evidence_quality: float = 0.0
    oos_observations: int = 0
    fold_count: int = 0
    oos_auc: float | None = None
    oos_brier: float | None = None
    baseline_brier: float | None = None
    brier_skill: float | None = None
    signal_count: int = 0
    signal_hit_rate: float | None = None
    signal_avg_net_return: float | None = None
    signal_median_net_return: float | None = None
    return_ci_low: float | None = None
    return_ci_high: float | None = None
    max_drawdown: float | None = None
    positive_fold_ratio: float | None = None
    bullish_regime_return: float | None = None
    bearish_regime_return: float | None = None
    drivers: str = ""
    reasons: tuple[str, ...] = ()
    calibration: list[dict] = field(default_factory=list)
    regularization_c: float | None = None
    model_version: str = "logit-ohlcv-purged-v1"

    def to_record(self) -> dict:
        record = asdict(self)
        record["research_status"] = record.pop("status")
        record["research_gate_pass"] = record.pop("gate_pass")
        record["research_signal"] = record.pop("current_signal")
        record["research_probability"] = record.pop("current_probability")
        record["research_quality"] = record.pop("evidence_quality")
        record["research_reasons"] = "；".join(record.pop("reasons"))
        record["research_calibration"] = json.dumps(
            record.pop("calibration"), ensure_ascii=False
        )
        record["research_drivers"] = record.pop("drivers")
        record["research_regularization_c"] = record.pop("regularization_c")
        record["research_model_version"] = record.pop("model_version")
        return record


class QuantResearchEngine:
    """正则逻辑回归 + 带隔离期扩展窗口的股票级研究引擎。"""

    def __init__(
        self,
        horizon: int = 10,
        min_train: int = 180,
        test_size: int = 40,
        probability_threshold: float = 0.55,
        slippage_bps: float = 5.0,
        bootstrap_samples: int = 1_000,
        random_state: int = 42,
    ):
        self.horizon = max(2, int(horizon))
        self.min_train = max(100, int(min_train))
        self.test_size = max(10, int(test_size))
        self.probability_threshold = float(probability_threshold)
        self.bootstrap_samples = max(200, int(bootstrap_samples))
        self.random_state = int(random_state)
        self.round_trip_cost = (
            2 * float(cfg.COMMISSION_RATE)
            + 2 * float(cfg.TRANSFER_FEE_RATE)
            + float(cfg.STAMP_DUTY_RATE)
            + 2 * max(0.0, float(slippage_bps)) / 10_000
        )

    @staticmethod
    def _model(regularization_c: float = 0.2) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=float(regularization_c),
                        solver="liblinear",
                        max_iter=400,
                        random_state=42,
                    ),
                ),
            ]
        )

    def _fit_tuned(
        self,
        train: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[Pipeline, float]:
        """只在外层训练集内部选正则强度，再用完整训练集重估。"""
        candidates = (0.01, 0.05, 0.2, 0.5)
        validation_size = min(self.test_size, max(20, len(train) // 5))
        validation_start = len(train) - validation_size
        inner_train_end = validation_start - self.horizon
        best_c = 0.05
        best_loss = float("inf")
        if inner_train_end >= 100:
            inner_train = train.iloc[:inner_train_end]
            inner_validation = train.iloc[validation_start:]
            inner_y = inner_train["target"].astype(int).to_numpy()
            validation_y = inner_validation["target"].astype(int).to_numpy()
            if len(np.unique(inner_y)) >= 2:
                for candidate in candidates:
                    model = self._model(candidate)
                    model.fit(inner_train[feature_columns], inner_y)
                    probability = model.predict_proba(
                        inner_validation[feature_columns]
                    )[:, 1]
                    loss = float(np.mean((probability - validation_y) ** 2))
                    if loss < best_loss:
                        best_loss = loss
                        best_c = candidate

        model = self._model(best_c)
        model.fit(train[feature_columns], train["target"].astype(int))
        return model, best_c

    @staticmethod
    def _clean_history(history: pd.DataFrame) -> pd.DataFrame:
        required = ["open", "high", "low", "close", "volume"]
        if history is None or history.empty or any(col not in history for col in required):
            return pd.DataFrame(columns=required)
        frame = history.copy()
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[~frame.index.isna()].sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=required)
        frame = frame[
            (frame["open"] > 0)
            & (frame["high"] > 0)
            & (frame["low"] > 0)
            & (frame["close"] > 0)
            & (frame["volume"] >= 0)
        ]
        return frame

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    def build_features(self, history: pd.DataFrame) -> pd.DataFrame:
        """构造只依赖当日及此前数据的固定特征，避免全样本分位数处理。"""
        frame = self._clean_history(history)
        if frame.empty:
            return pd.DataFrame()
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        volume = frame["volume"]
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(span=14, adjust=False).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        previous_high20 = high.shift(1).rolling(20).max()
        volume_ratio = volume / volume.rolling(20).mean().replace(0, np.nan)

        features = pd.DataFrame(index=frame.index)
        features["momentum_5d"] = close.pct_change(5).clip(-0.30, 0.30)
        features["momentum_20d"] = close.pct_change(20).clip(-0.60, 0.60)
        features["momentum_60d"] = close.pct_change(60).clip(-0.80, 1.50)
        features["price_ma20_gap"] = (close / ma20 - 1).clip(-0.50, 0.50)
        features["ma20_ma60_gap"] = (ma20 / ma60 - 1).clip(-0.50, 0.50)
        features["rsi_centered"] = ((self._rsi(close) - 50) / 50).clip(-1, 1)
        features["macd_atr"] = (macd / atr.replace(0, np.nan)).clip(-5, 5)
        features["atr_pct"] = (atr / close).clip(0, 0.20)
        features["log_volume_ratio"] = np.log(
            volume_ratio.clip(lower=0.05, upper=10)
        ).clip(-2, 2)
        features["breakout_20d"] = (close / previous_high20 - 1).clip(-0.50, 0.50)
        features["drawdown_60d"] = (
            close / close.rolling(60).max() - 1
        ).clip(-0.80, 0)
        features["intraday_range"] = ((high - low) / close).clip(0, 0.30)
        return features.replace([np.inf, -np.inf], np.nan)

    def _research_frame(
        self, history: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        clean = self._clean_history(history)
        features = self.build_features(clean)
        if features.empty:
            return pd.DataFrame(), pd.DataFrame()
        forward_net = clean["close"].shift(-self.horizon) / clean["close"] - 1
        forward_net = forward_net - self.round_trip_cost
        target = pd.Series(np.nan, index=forward_net.index, dtype=float)
        known = forward_net.notna()
        target.loc[known] = (forward_net.loc[known] > 0).astype(float)
        regime = clean["close"] > clean["close"].rolling(60).mean()
        joined = features.copy()
        joined["target"] = target
        joined["forward_net_return"] = forward_net
        joined["bullish_regime"] = regime
        labeled = joined.dropna(subset=[*features.columns, "target", "forward_net_return"])
        current = joined.dropna(subset=list(features.columns)).tail(1)
        return labeled, current

    @staticmethod
    def _safe_auc(y_true: np.ndarray, probability: np.ndarray) -> float | None:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, probability))

    def _non_overlapping_returns(
        self, selected: pd.DataFrame, full_index: pd.Index
    ) -> np.ndarray:
        if selected.empty:
            return np.array([], dtype=float)
        positions = pd.Series(range(len(full_index)), index=full_index)
        kept: list[float] = []
        last_position = -self.horizon
        for date, row in selected.sort_index().iterrows():
            if date not in positions:
                continue
            position = int(positions.loc[date])
            if position - last_position < self.horizon:
                continue
            kept.append(float(row["forward_net_return"]))
            last_position = position
        return np.asarray(kept, dtype=float)

    def _bootstrap_interval(self, returns: np.ndarray) -> tuple[float | None, float | None]:
        if len(returns) < 5:
            return None, None
        rng = np.random.default_rng(self.random_state)
        samples = rng.choice(
            returns,
            size=(self.bootstrap_samples, len(returns)),
            replace=True,
        )
        means = samples.mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        return float(low), float(high)

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float | None:
        if len(returns) == 0:
            return None
        equity = np.cumprod(1 + returns)
        peaks = np.maximum.accumulate(equity)
        return float(np.min(equity / peaks - 1))

    @staticmethod
    def _calibration_points(y_true: np.ndarray, probability: np.ndarray) -> list[dict]:
        if len(probability) < 20:
            return []
        frame = pd.DataFrame({"outcome": y_true, "probability": probability})
        try:
            frame["bin"] = pd.qcut(frame["probability"], q=5, duplicates="drop")
        except ValueError:
            return []
        grouped = frame.groupby("bin", observed=True).agg(
            predicted_probability=("probability", "mean"),
            realized_hit_rate=("outcome", "mean"),
            observations=("outcome", "size"),
        )
        return [
            {
                "predicted_probability": round(float(row.predicted_probability), 4),
                "realized_hit_rate": round(float(row.realized_hit_rate), 4),
                "observations": int(row.observations),
            }
            for row in grouped.itertuples()
        ]

    @staticmethod
    def _drivers(model: Pipeline, current_features: pd.DataFrame) -> str:
        scaler = model.named_steps["scale"]
        classifier = model.named_steps["logit"]
        standardized = scaler.transform(current_features)[0]
        contributions = standardized * classifier.coef_[0]
        ranked = sorted(
            zip(current_features.columns, contributions),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:4]
        return " · ".join(
            f"{FEATURE_LABELS.get(name, name)}{'+' if value >= 0 else '-'}"
            for name, value in ranked
        )

    @staticmethod
    def _quality_score(
        auc: float | None,
        brier_skill: float | None,
        average_return: float | None,
        positive_fold_ratio: float | None,
        observations: int,
    ) -> float:
        auc_component = np.clip(((auc or 0.5) - 0.5) / 0.12, 0, 1) * 30
        brier_component = np.clip((brier_skill or 0) / 0.08, 0, 1) * 25
        return_component = np.clip((average_return or 0) / 0.03, 0, 1) * 25
        stability_component = np.clip(positive_fold_ratio or 0, 0, 1) * 10
        sample_component = np.clip(observations / 250, 0, 1) * 10
        return round(float(
            auc_component
            + brier_component
            + return_component
            + stability_component
            + sample_component
        ), 1)

    def evaluate(self, history: pd.DataFrame) -> ResearchEvidence:
        """运行扩展窗口验证；所有样本外折叠均保留 10 日隔离期。"""
        clean = self._clean_history(history)
        labeled, current = self._research_frame(clean)
        minimum_rows = self.min_train + self.horizon + self.test_size
        if len(labeled) < minimum_rows or current.empty:
            return ResearchEvidence(
                status="样本不足",
                gate_pass=False,
                current_signal="不可评估",
                oos_observations=0,
                reasons=(
                    f"有效历史样本 {len(labeled)}<{minimum_rows}",
                    "不允许用小样本模型影响推荐",
                ),
            )

        feature_columns = list(FEATURE_LABELS)
        oos_parts: list[pd.DataFrame] = []
        fold_number = 0
        first_test = self.min_train + self.horizon
        for test_start in range(first_test, len(labeled), self.test_size):
            train_end = test_start - self.horizon
            test_end = min(test_start + self.test_size, len(labeled))
            train = labeled.iloc[:train_end]
            test = labeled.iloc[test_start:test_end]
            if len(train) < self.min_train or test.empty:
                continue
            y_train = train["target"].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2:
                continue
            model, chosen_c = self._fit_tuned(train, feature_columns)
            probability = model.predict_proba(test[feature_columns])[:, 1]
            fold_number += 1
            part = test[["target", "forward_net_return", "bullish_regime"]].copy()
            part["probability"] = probability
            part["baseline_probability"] = float(np.mean(y_train))
            part["fold"] = fold_number
            part["regularization_c"] = chosen_c
            oos_parts.append(part)

        if not oos_parts:
            return ResearchEvidence(
                status="样本不足",
                gate_pass=False,
                current_signal="不可评估",
                reasons=("无法形成同时包含涨跌样本的扩展窗口",),
            )

        oos = pd.concat(oos_parts).sort_index()
        y_true = oos["target"].astype(int).to_numpy()
        probability = oos["probability"].to_numpy(dtype=float)
        baseline_probability = oos["baseline_probability"].to_numpy(dtype=float)
        auc = self._safe_auc(y_true, probability)
        brier = float(np.mean((probability - y_true) ** 2))
        baseline_brier = float(np.mean((baseline_probability - y_true) ** 2))
        brier_skill = 1 - brier / baseline_brier if baseline_brier > 0 else None

        selected = oos[oos["probability"] >= self.probability_threshold]
        selected_returns = selected["forward_net_return"].to_numpy(dtype=float)
        signal_count = len(selected)
        hit_rate = float((selected_returns > 0).mean()) if signal_count else None
        average_return = float(selected_returns.mean()) if signal_count else None
        median_return = float(np.median(selected_returns)) if signal_count else None

        fold_means = selected.groupby("fold")["forward_net_return"].mean()
        positive_fold_ratio = (
            float((fold_means > 0).mean()) if len(fold_means) else None
        )
        non_overlapping = self._non_overlapping_returns(selected, clean.index)
        ci_low, ci_high = self._bootstrap_interval(non_overlapping)
        max_drawdown = self._max_drawdown(non_overlapping)

        bullish = selected[selected["bullish_regime"].astype(bool)]["forward_net_return"]
        bearish = selected[~selected["bullish_regime"].astype(bool)]["forward_net_return"]
        bullish_return = float(bullish.mean()) if len(bullish) else None
        bearish_return = float(bearish.mean()) if len(bearish) else None

        final_y = labeled["target"].astype(int).to_numpy()
        if len(np.unique(final_y)) < 2:
            return ResearchEvidence(
                status="样本不足",
                gate_pass=False,
                current_signal="不可评估",
                reasons=("完整历史标签只有单一类别",),
            )
        final_model, final_c = self._fit_tuned(labeled, feature_columns)
        current_features = current[feature_columns]
        current_probability = float(final_model.predict_proba(current_features)[0, 1])
        current_signal = (
            "看多"
            if current_probability >= self.probability_threshold
            else "偏弱"
            if current_probability < 0.45
            else "观望"
        )

        quality = self._quality_score(
            auc, brier_skill, average_return, positive_fold_ratio, len(oos)
        )
        evidence_gaps: list[str] = []
        if len(oos) < 120:
            evidence_gaps.append(f"样本外观测 {len(oos)}<120")
        if fold_number < 4:
            evidence_gaps.append(f"有效折叠 {fold_number}<4")
        if auc is None or auc < 0.52:
            evidence_gaps.append(f"样本外 AUC {(auc or 0):.3f}<0.520")
        if brier_skill is None or brier_skill <= 0:
            evidence_gaps.append("概率误差未优于历史胜率基线")
        if signal_count < 12:
            evidence_gaps.append(f"高概率信号 {signal_count}<12")
        if average_return is None or average_return <= 0:
            evidence_gaps.append("高概率信号成本后平均收益不为正")
        if positive_fold_ratio is None or positive_fold_ratio < 0.50:
            evidence_gaps.append("高概率信号正收益折叠不足一半")
        if ci_low is None or ci_low <= -0.01:
            evidence_gaps.append("非重叠收益 95% 区间下界不高于 -1%")
        if quality < 50:
            evidence_gaps.append(f"证据质量 {quality:.1f}<50")

        evidence_pass = not evidence_gaps
        gate_pass = evidence_pass and current_probability >= self.probability_threshold
        reasons = list(evidence_gaps)
        if evidence_pass and not gate_pass:
            reasons.append(
                f"当前上涨概率 {current_probability:.1%}<{self.probability_threshold:.0%}"
            )
        if gate_pass:
            reasons.append("样本外证据与当前概率均通过")

        return ResearchEvidence(
            status="样本外通过" if evidence_pass else "样本外未通过",
            gate_pass=gate_pass,
            current_signal=current_signal,
            current_probability=round(current_probability, 4),
            evidence_quality=quality,
            oos_observations=len(oos),
            fold_count=fold_number,
            oos_auc=round(auc, 4) if auc is not None else None,
            oos_brier=round(brier, 4),
            baseline_brier=round(baseline_brier, 4),
            brier_skill=round(brier_skill, 4) if brier_skill is not None else None,
            signal_count=signal_count,
            signal_hit_rate=round(hit_rate, 4) if hit_rate is not None else None,
            signal_avg_net_return=(
                round(average_return, 5) if average_return is not None else None
            ),
            signal_median_net_return=(
                round(median_return, 5) if median_return is not None else None
            ),
            return_ci_low=round(ci_low, 5) if ci_low is not None else None,
            return_ci_high=round(ci_high, 5) if ci_high is not None else None,
            max_drawdown=round(max_drawdown, 5) if max_drawdown is not None else None,
            positive_fold_ratio=(
                round(positive_fold_ratio, 4)
                if positive_fold_ratio is not None
                else None
            ),
            bullish_regime_return=(
                round(bullish_return, 5) if bullish_return is not None else None
            ),
            bearish_regime_return=(
                round(bearish_return, 5) if bearish_return is not None else None
            ),
            drivers=self._drivers(final_model, current_features),
            reasons=tuple(reasons),
            calibration=self._calibration_points(y_true, probability),
            regularization_c=final_c,
        )
