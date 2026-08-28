"""
券商抽象接口 — 下单、查仓、查账户
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"          # 等待提交
    SUBMITTED = "SUBMITTED"      # 已提交
    FILLED = "FILLED"            # 全部成交
    PARTIAL = "PARTIAL"          # 部分成交
    CANCELLED = "CANCELLED"      # 已撤单
    REJECTED = "REJECTED"        # 被拒绝
    ERROR = "ERROR"


@dataclass
class Order:
    """订单对象"""
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: int
    filled_qty: int = 0
    status: OrderStatus = OrderStatus.PENDING
    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    reason: str = ""


@dataclass
class OrderResult:
    """下单结果"""
    success: bool
    order: Optional[Order] = None
    message: str = ""


class Broker(ABC):
    """
    券商抽象接口
    所有实盘券商（如 QMT）都应实现这个接口
    """

    name: str = "base"

    # ============================================================
    #  连接
    # ============================================================
    @abstractmethod
    def connect(self) -> bool:
        """建立连接，返回是否成功"""
        ...

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...

    # ============================================================
    #  账户
    # ============================================================
    @abstractmethod
    def get_account(self) -> dict:
        """
        账户信息
        → {total_assets, cash, market_value, frozen_cash, ...}
        """
        ...

    @abstractmethod
    def get_positions(self) -> list:
        """
        持仓列表
        → [{"symbol": "SH.600010", "qty": 10000, "cost": 2.55, "price": 2.66, "pnl": ...}, ...]
        """
        ...

    # ============================================================
    #  交易
    # ============================================================
    @abstractmethod
    def place_order(
        self, symbol: str, price: float, quantity: int, side: OrderSide
    ) -> OrderResult:
        """下单"""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        ...

    @abstractmethod
    def get_orders(self, status_filter: list = None) -> List[Order]:
        """查询订单"""
        ...

    # ============================================================
    #  行情（可选，复用 data_sources 也可）
    # ============================================================
    def get_quote(self, symbol: str) -> dict:
        """实时报价（可选实现）"""
        return {}
