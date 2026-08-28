"""
自动交易引擎 — 策略信号 → 安全闸门 → 券商下单
=============================================
这是把策略回测结果付诸实盘的核心环节。
使用前务必先在 PaperBroker 上验证至少一周。
"""

from typing import Optional
from datetime import datetime

from .base import Broker, OrderSide, OrderResult
from .safety_gate import SafetyGate
from .paper_broker import PaperBroker
from strategies.base import Signal, SignalType
import config as cfg


class AutoTrader:
    """
    自动交易引擎

    工作流程：
      策略产生 Signal → SafetyGate 三层检查 → Broker.place_order()
      
    安全原则：
      1. 默认连 PaperBroker，手动确认后才切实盘
      2. 每笔订单必须通过 SafetyGate
      3. 异常时自动暂停，绝不自动恢复
    """

    def __init__(
        self,
        broker: Broker,
        safety_gate: SafetyGate = None,
        symbol: str = "HK.01164",
    ):
        self.symbol = symbol
        self.broker = broker
        self.gate = safety_gate or SafetyGate()
        self._paused = False
        self._pause_reason = ""
        self._log: list = []

    # ============================================================
    #  主入口：处理策略信号
    # ============================================================
    def process_signal(self, signal: Signal) -> OrderResult:
        """
        处理一个策略信号 → 风控 → 下单
        返回 OrderResult
        """
        if self._paused:
            return OrderResult(False, message=f"引擎已暂停: {self._pause_reason}")

        # 获取当前账户状态
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        current_pos = sum(
            p["qty"] for p in positions if p["symbol"] == self.symbol
        )

        side = OrderSide.BUY if signal.type == SignalType.BUY else OrderSide.SELL
        if side == OrderSide.SELL:
            # 策略的卖出信号通常 position_pct=0，含义是清仓，
            # 不能按 position_pct 计算成 0 股后误报“数量为0”。
            quantity = current_pos
        else:
            quantity = (
                int(signal.position_pct * account["total_assets"] / signal.price)
                if signal.position_pct > 0
                else 0
            )

        if quantity <= 0:
            message = "当前无持仓可卖" if side == OrderSide.SELL else "计算数量为0"
            return OrderResult(False, message=message)

        return self.place_manual_order(signal.price, quantity, side, action="STRATEGY")

    def place_manual_order(
        self,
        price: float,
        quantity: int,
        side: OrderSide,
        action: str = "MANUAL",
    ) -> OrderResult:
        """提交一笔模拟/实盘订单；手动单也必须经过同一安全闸门。"""
        if self._paused:
            return OrderResult(False, message=f"引擎已暂停: {self._pause_reason}")

        account = self.broker.get_account()
        positions = self.broker.get_positions()
        current_pos = sum(
            p["qty"] for p in positions if p["symbol"] == self.symbol
        )

        check = self.gate.check_order(
            symbol=self.symbol,
            price=float(price),
            quantity=quantity,
            side=side,
            current_position=current_pos,
            account_cash=account.get("available_cash", account.get("cash", 0)),
            account_total=account["total_assets"],
        )

        if not check.passed:
            self._log.append({
                "time": datetime.now(),
                "type": "BLOCKED",
                "action": action,
                "reason": check.reason,
            })
            # 只有账户级熔断才暂停；普通输错数量/超仓只拒单。
            if "日内亏损" in check.reason or "连续亏损" in check.reason:
                self._paused = True
                self._pause_reason = check.reason
            return OrderResult(False, message=f"风控拦截: {check.reason}")

        avg_cost = next(
            (p["cost"] for p in positions if p["symbol"] == self.symbol),
            0.0,
        )
        result = self.broker.place_order(
            symbol=self.symbol,
            price=float(price),
            quantity=quantity,
            side=side,
        )

        if result.success and side == OrderSide.SELL and result.order:
            realized_pnl = (result.order.price - avg_cost) * quantity
            self.gate.on_fill(realized_pnl)

        self._log.append({
            "time": datetime.now(),
            "type": "ORDER",
            "action": action,
            "result": result,
        })

        return result

    # ============================================================
    #  状态
    # ============================================================
    def pause(self, reason: str = "手动暂停"):
        self._paused = True
        self._pause_reason = reason

    def resume(self):
        self._paused = False
        self._pause_reason = ""

    @property
    def is_paused(self) -> bool:
        return self._paused

    def status(self) -> dict:
        return {
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "gate_status": self.gate.status(),
            "log_count": len(self._log),
            "last_action": self._log[-1] if self._log else None,
        }

    # ============================================================
    #  便捷工厂
    # ============================================================
    @classmethod
    def paper(cls, capital: float = None, symbol: str = "HK.01164") -> "AutoTrader":
        """创建模拟交易引擎"""
        broker = PaperBroker(capital or cfg.INITIAL_CAPITAL)
        gate = SafetyGate()
        return cls(broker, gate, symbol=symbol)

    @classmethod
    def futu_simulate(cls, capital: float = None) -> Optional["AutoTrader"]:
        """创建富途模拟盘引擎"""
        try:
            from .futu_broker import FutuBroker
            broker = FutuBroker(trade_env="SIMULATE")
            if broker.connect():
                gate = SafetyGate()
                return cls(broker, gate)
        except Exception as e:
            print(f"Futu 模拟盘连接失败: {e}")
        return None

    @classmethod
    def futu_real(cls, capital: float = None) -> Optional["AutoTrader"]:
        """创建富途实盘引擎 ⚠️ 真金白银"""
        try:
            from .futu_broker import FutuBroker
            broker = FutuBroker(trade_env="REAL")
            if broker.connect():
                gate = SafetyGate()
                return cls(broker, gate)
        except Exception as e:
            print(f"Futu 实盘连接失败: {e}")
        return None
