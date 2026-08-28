"""港股候选的防泄漏扩展窗口样本外证据层。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "price_ma20_gap",
    "ma20_ma60_gap",
    "rsi_centered",
    "macd_atr",
    "atr_pct",
    "log_volume_ratio",
    "breakout_20d",
    "drawdown_60d",
    "intraday_range",
)


@dataclass(frozen=True)
class ResearchEvidence:
    status: str
    gate_pass: bool
    current_signal: str
    current_probability: float | None = None
    evidence_quality: float = 0.0
    oos_observations: int = 0
    fold_count: int = 0
    oos_auc: float | None = None
    brier_skill: float | None = None
    signal_count: int = 0
    signal_hit_rate: float | None = None
    signal_avg_net_return: float | None = None
    positive_fold_ratio: float | None = None
    drivers: str = ""
    reasons: tuple[str, ...] = ()
    calibration: list[dict] = field(default_factory=list)

    def to_record(self) -> dict:
        record = asdict(self)
        record["research_status"] = record.pop("status")
        record["research_gate_pass"] = record.pop("gate_pass")
        record["research_signal"] = record.pop("current_signal")
        record["research_probability"] = record.pop("current_probability")
        record["research_quality"] = record.pop("evidence_quality")
        record["research_reasons"] = "；".join(record.pop("reasons"))
        record["research_calibration"] = json.dumps(record.pop("calibration"), ensure_ascii=False)
        return record


class QuantResearchEngine:
    """正则逻辑回归；训练与测试间隔离未来收益窗口。"""

    def __init__(
        self,
        horizon: int = 10,
        min_train: int = 180,
        test_size: int = 40,
        probability_threshold: float = 0.55,
        round_trip_cost: float = 0.00277,
    ):
        self.horizon = max(2, int(horizon))
        self.min_train = max(100, int(min_train))
        self.test_size = max(10, int(test_size))
        self.probability_threshold = float(probability_threshold)
        # 约含双边印花税、交易费、监管征费、券商佣金及 5bp/边滑点。
        self.round_trip_cost = max(0.0, float(round_trip_cost))

    @staticmethod
    def _model() -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        C=0.5,
                        solver="liblinear",
                        max_iter=400,
                        random_state=42,
                    ),
                ),
            ]
        )

    @staticmethod
    def _clean(history: pd.DataFrame) -> pd.DataFrame:
        required = ["open", "high", "low", "close", "volume"]
        if history is None or history.empty or any(column not in history for column in required):
            return pd.DataFrame(columns=required)
        frame = history.copy()
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame[~frame.index.isna()].sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=required).loc[lambda value: value["close"] > 0]

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        relative = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + relative)

    def build_features(self, history: pd.DataFrame) -> pd.DataFrame:
        frame = self._clean(history)
        if frame.empty:
            return pd.DataFrame()
        close, high, low, volume = frame["close"], frame["high"], frame["low"], frame["volume"]
        ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
        true_range = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
        ).max(axis=1)
        atr = true_range.ewm(span=14, adjust=False).mean()
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        output = pd.DataFrame(index=frame.index)
        output["momentum_5d"] = close.pct_change(5).clip(-0.30, 0.30)
        output["momentum_20d"] = close.pct_change(20).clip(-0.60, 0.60)
        output["momentum_60d"] = close.pct_change(60).clip(-0.80, 1.50)
        output["price_ma20_gap"] = (close / ma20 - 1).clip(-0.50, 0.50)
        output["ma20_ma60_gap"] = (ma20 / ma60 - 1).clip(-0.50, 0.50)
        output["rsi_centered"] = ((self._rsi(close) - 50) / 50).clip(-1, 1)
        output["macd_atr"] = (macd / atr.replace(0, np.nan)).clip(-5, 5)
        output["atr_pct"] = (atr / close).clip(0, 0.20)
        output["log_volume_ratio"] = np.log(
            (volume / volume.rolling(20).mean().replace(0, np.nan)).clip(0.05, 10)
        ).clip(-2, 2)
        output["breakout_20d"] = (close / high.shift().rolling(20).max() - 1).clip(-0.50, 0.50)
        output["drawdown_60d"] = (close / close.rolling(60).max() - 1).clip(-0.80, 0)
        output["intraday_range"] = ((high - low) / close).clip(0, 0.30)
        return output.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _quality(auc, skill, average, fold_ratio, observations) -> float:
        return round(float(
            np.clip(((auc or 0.5) - 0.5) / 0.12, 0, 1) * 30
            + np.clip((skill or 0) / 0.08, 0, 1) * 25
            + np.clip((average or 0) / 0.03, 0, 1) * 25
            + np.clip(fold_ratio or 0, 0, 1) * 10
            + np.clip(observations / 250, 0, 1) * 10
        ), 1)

    def evaluate(self, history: pd.DataFrame) -> ResearchEvidence:
        clean = self._clean(history)
        features = self.build_features(clean)
        forward = clean["close"].shift(-self.horizon) / clean["close"] - 1 - self.round_trip_cost
        labeled = features.copy()
        labeled["forward_return"] = forward
        labeled["target"] = np.where(forward.notna(), (forward > 0).astype(float), np.nan)
        labeled = labeled.dropna(subset=[*FEATURES, "forward_return", "target"])
        current = features.dropna(subset=list(FEATURES)).tail(1)
        minimum = self.min_train + self.horizon + self.test_size
        if len(labeled) < minimum or current.empty:
            return ResearchEvidence(
                status="样本不足",
                gate_pass=False,
                current_signal="不可评估",
                reasons=(f"有效历史样本 {len(labeled)}<{minimum}",),
            )

        parts = []
        fold = 0
        for test_start in range(self.min_train + self.horizon, len(labeled), self.test_size):
            train = labeled.iloc[: test_start - self.horizon]
            test = labeled.iloc[test_start : test_start + self.test_size]
            y_train = train["target"].astype(int).to_numpy()
            if test.empty or len(np.unique(y_train)) < 2:
                continue
            model = self._model().fit(train[list(FEATURES)], y_train)
            part = test[["target", "forward_return"]].copy()
            part["probability"] = model.predict_proba(test[list(FEATURES)])[:, 1]
            part["baseline"] = float(y_train.mean())
            fold += 1
            part["fold"] = fold
            parts.append(part)
        if not parts:
            return ResearchEvidence("样本不足", False, "不可评估", reasons=("无法形成扩展窗口",))

        oos = pd.concat(parts)
        y_true = oos["target"].astype(int).to_numpy()
        probabilities = oos["probability"].to_numpy()
        auc = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None
        brier = float(np.mean((probabilities - y_true) ** 2))
        baseline_brier = float(np.mean((oos["baseline"].to_numpy() - y_true) ** 2))
        skill = 1 - brier / baseline_brier if baseline_brier > 0 else None
        selected = oos[oos["probability"] >= self.probability_threshold]
        average = float(selected["forward_return"].mean()) if len(selected) else None
        hit_rate = float((selected["forward_return"] > 0).mean()) if len(selected) else None
        fold_means = selected.groupby("fold")["forward_return"].mean()
        fold_ratio = float((fold_means > 0).mean()) if len(fold_means) else None
        quality = self._quality(auc, skill, average, fold_ratio, len(oos))

        final_y = labeled["target"].astype(int).to_numpy()
        if len(np.unique(final_y)) < 2:
            return ResearchEvidence("样本不足", False, "不可评估", reasons=("历史标签单一",))
        final = self._model().fit(labeled[list(FEATURES)], final_y)
        probability = float(final.predict_proba(current[list(FEATURES)])[0, 1])
        current_signal = "看多" if probability >= 0.55 else "偏弱" if probability < 0.45 else "观望"
        gaps = []
        if len(oos) < 120:
            gaps.append(f"样本外观测 {len(oos)}<120")
        if fold < 4:
            gaps.append(f"有效折叠 {fold}<4")
        if auc is None or auc < 0.52:
            gaps.append(f"样本外 AUC {(auc or 0):.3f}<0.520")
        if skill is None or skill <= 0:
            gaps.append("概率误差未优于历史胜率")
        if len(selected) < 12:
            gaps.append(f"高概率信号 {len(selected)}<12")
        if average is None or average <= 0:
            gaps.append("成本后平均收益不为正")
        if fold_ratio is None or fold_ratio < 0.5:
            gaps.append("正收益折叠不足一半")
        if quality < 50:
            gaps.append(f"证据质量 {quality:.1f}<50")
        evidence_pass = not gaps
        gate_pass = evidence_pass and probability >= self.probability_threshold
        if evidence_pass and not gate_pass:
            gaps.append(f"当前上涨概率 {probability:.1%}<55%")
        return ResearchEvidence(
            status="样本外通过" if evidence_pass else "样本外未通过",
            gate_pass=gate_pass,
            current_signal=current_signal,
            current_probability=round(probability, 4),
            evidence_quality=quality,
            oos_observations=len(oos),
            fold_count=fold,
            oos_auc=round(auc, 4) if auc is not None else None,
            brier_skill=round(skill, 4) if skill is not None else None,
            signal_count=len(selected),
            signal_hit_rate=round(hit_rate, 4) if hit_rate is not None else None,
            signal_avg_net_return=round(average, 5) if average is not None else None,
            positive_fold_ratio=round(fold_ratio, 4) if fold_ratio is not None else None,
            reasons=tuple(gaps or ["样本外证据与当前概率均通过"]),
        )
