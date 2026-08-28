"""
策略基类 — 所有交易策略的抽象接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import pandas as pd


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """交易信号"""
    type: SignalType
    price: float
    date: pd.Timestamp
    strategy: str = ""
    confidence: float = 1.0         # 0~1 置信度
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_pct: float = 0.0       # 建议仓位比例
    metadata: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """策略抽象基类"""

    name: str = "base"
    description: str = ""

    def __init__(self, params: dict = None):
        self.params = params or {}
        self._df: Optional[pd.DataFrame] = None

    def set_data(self, df: pd.DataFrame):
        """注入行情数据"""
        self._df = df.copy()

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        在 self._df 上生成信号列
        返回的 DataFrame 需添加:
          - signal: 1(buy) / -1(sell) / 0(hold)
          - signal_price: 入场价格
          - signal_reason: 信号原因
        """
        ...

    def get_signals_list(self) -> List[Signal]:
        """将 DataFrame 信号列转换为 Signal 对象列表"""
        if self._df is None or "signal" not in self._df.columns:
            return []

        signals = []
        for idx, row in self._df.iterrows():
            if row["signal"] == 0:
                continue
            stype = SignalType.BUY if row["signal"] == 1 else SignalType.SELL
            signals.append(Signal(
                type=stype,
                price=row.get("signal_price", row["close"]),
                date=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx),
                strategy=self.name,
                reason=str(row.get("signal_reason", "")),
                stop_loss=row.get("stop_loss"),
                take_profit=row.get("take_profit"),
                position_pct=row.get("position_pct", 0.0),
            ))
        return signals

    @staticmethod
    def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        """A 上穿 B 返回 1，下穿返回 -1，否则 0"""
        above = series_a > series_b
        cross_up = above & (~above.shift(1).fillna(False))
        cross_down = (~above) & (above.shift(1).fillna(False))
        result = pd.Series(0, index=series_a.index)
        result[cross_up] = 1
        result[cross_down] = -1
        return result
