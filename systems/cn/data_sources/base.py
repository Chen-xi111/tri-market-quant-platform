"""
数据源抽象基类
"""

from abc import ABC, abstractmethod
import pandas as pd
import config as cfg


class DataSource(ABC):
    """所有行情源的统一接口"""

    name: str = "base"

    def __init__(self, ticker: str = None):
        self.ticker = ticker or cfg.TICKER

    @abstractmethod
    def get_realtime(self) -> dict:
        """实时报价快照 → {price, open, high, low, volume, ...}"""
        ...

    @abstractmethod
    def get_history(self, period: str = "1d", days: int = 365) -> pd.DataFrame:
        """历史K线 → DataFrame(open, high, low, close, volume)"""
        ...

    def get_info(self) -> dict:
        """标的基本信息"""
        return {}
