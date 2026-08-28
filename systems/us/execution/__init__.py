"""
程序化交易执行层 — 券商对接 + 多层安全闸门
"""

from .base import Broker, Order, OrderSide, OrderStatus, OrderResult
from .paper_broker import PaperBroker
from .safety_gate import SafetyGate

# IBKR 适配器本身可以在未安装官方 ibapi 时导入，界面会给出安装指引。
from .ibkr_broker import IBKRBroker, ibapi_available

# Futu 是可选的（需要 FutuOpenD 运行中）
try:
    from .futu_broker import FutuBroker
    HAS_FUTU = True
except Exception:
    HAS_FUTU = False

BROKERS = {
    "paper": ("📝 模拟交易", None),  # PaperBroker 直接示例化
    "ibkr_paper": ("IBKR Paper", None),
}

if HAS_FUTU:
    BROKERS["futu"] = ("🔗 富途实盘", None)
