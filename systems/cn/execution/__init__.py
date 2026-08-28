"""
程序化交易执行层 — 券商对接 + 多层安全闸门
"""

from .base import Broker, Order, OrderSide, OrderStatus, OrderResult
from .paper_broker import PaperBroker
from .safety_gate import SafetyGate

BROKERS = {
    "paper": ("📝 模拟交易", None),  # PaperBroker 直接示例化
}
