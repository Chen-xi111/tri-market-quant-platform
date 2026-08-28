"""
模拟券商 — 无风险验证策略逻辑，不涉及真实资金
"""

from .base import Broker, Order, OrderSide, OrderStatus, OrderResult
from typing import List, Optional
from datetime import date, datetime
import uuid

import config as cfg
from trading_rules import calculate_trade_fee


class PaperBroker(Broker):
    """
    纸交易券商
    - 所有订单立即成交，并计入可配置的滑点和交易费用
    - 用于策略验证和回测后验证
    """

    name = "paper"

    def __init__(
        self,
        initial_cash: float = 100_000,
        commission_rate: float = cfg.COMMISSION_RATE,
        slippage_bps: float = 5.0,
    ):
        self._cash = initial_cash
        self._positions: dict = {}          # symbol → {qty, avg_cost}
        self._orders: List[Order] = []
        self._connected = True
        self._initial_cash = initial_cash
        self._commission_rate = max(0.0, float(commission_rate))
        self._slippage_bps = max(0.0, float(slippage_bps))
        self._fees_paid = 0.0
        self._realized_pnl = 0.0

    # ---- 连接 ----
    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ---- 账户 ----
    def get_account(self) -> dict:
        mv = sum(
            p["qty"] * p.get("last_price", p["avg_cost"])
            for p in self._positions.values()
        )
        return {
            "total_assets": self._cash + mv,
            "cash": self._cash,
            "market_value": mv,
            "frozen_cash": 0.0,
            "initial_cash": self._initial_cash,
            "fees_paid": self._fees_paid,
            "realized_pnl": self._realized_pnl,
        }

    def get_positions(self) -> list:
        return [
            {
                "symbol": sym,
                "qty": pos["qty"],
                "cost": pos["avg_cost"],
                "price": pos.get("last_price", pos["avg_cost"]),
                "pnl": pos["qty"] * (pos.get("last_price", pos["avg_cost"]) - pos["avg_cost"]),
                "sellable_qty": self._sellable_qty(pos),
            }
            for sym, pos in self._positions.items()
            if pos["qty"] > 0
        ]

    # ---- 交易 ----
    def place_order(
        self, symbol: str, price: float, quantity: int, side: OrderSide
    ) -> OrderResult:
        if quantity <= 0:
            return OrderResult(False, message="数量无效")

        slip = self._slippage_bps / 10_000
        fill_price = price * (1 + slip if side == OrderSide.BUY else 1 - slip)
        gross_value = fill_price * quantity
        fee_parts = calculate_trade_fee(
            gross_value, side.value, commission_rate=self._commission_rate,
        )
        fee = fee_parts["total"]

        order = Order(
            order_id=str(uuid.uuid4())[:12],
            symbol=symbol,
            side=side,
            price=fill_price,
            quantity=quantity,
        )

        if side == OrderSide.BUY:
            if quantity % cfg.DEFAULT_LOT_SIZE != 0:
                order.status = OrderStatus.REJECTED
                self._orders.append(order)
                return OrderResult(False, order, f"A 股买入数量必须是 {cfg.DEFAULT_LOT_SIZE} 股的整数倍")
            cost = gross_value + fee
            if cost > self._cash:
                order.status = OrderStatus.REJECTED
                self._orders.append(order)
                return OrderResult(False, order, "资金不足")

            self._cash -= cost
            if symbol in self._positions:
                old_qty = self._positions[symbol]["qty"]
                old_cost = self._positions[symbol]["avg_cost"]
                new_qty = old_qty + quantity
                new_cost = (old_cost * old_qty + cost) / new_qty
                lots = self._positions[symbol].setdefault("lots", [])
                lots.append({"qty": quantity, "buy_date": date.today()})
                self._positions[symbol] = {"qty": new_qty, "avg_cost": new_cost, "lots": lots}
            else:
                self._positions[symbol] = {
                    "qty": quantity, "avg_cost": cost / quantity,
                    "lots": [{"qty": quantity, "buy_date": date.today()}],
                }

        else:  # SELL
            pos = self._positions.get(symbol, {"qty": 0})
            if quantity > pos["qty"]:
                order.status = OrderStatus.REJECTED
                self._orders.append(order)
                return OrderResult(False, order, "持仓不足")
            sellable_qty = self._sellable_qty(pos)
            if quantity > sellable_qty:
                order.status = OrderStatus.REJECTED
                self._orders.append(order)
                return OrderResult(
                    False, order,
                    f"A 股 T+1 限制：今日可卖 {sellable_qty:,} 股，今日买入部分下一交易日方可卖出",
                )

            proceeds = gross_value - fee
            self._cash += proceeds
            self._realized_pnl += proceeds - pos["avg_cost"] * quantity
            pos["qty"] -= quantity
            remaining = quantity
            new_lots = []
            for lot in pos.get("lots", []):
                if lot.get("buy_date") >= date.today() or remaining <= 0:
                    new_lots.append(lot)
                    continue
                consumed = min(int(lot["qty"]), remaining)
                remaining -= consumed
                if consumed < int(lot["qty"]):
                    new_lots.append({**lot, "qty": int(lot["qty"]) - consumed})
            pos["lots"] = new_lots
            if pos["qty"] <= 0:
                del self._positions[symbol]
            else:
                self._positions[symbol] = pos

        order.status = OrderStatus.FILLED
        order.filled_qty = quantity
        order.update_time = datetime.now()
        fee_text = (
            f"佣金 ¥{fee_parts['commission']:.2f} + 过户费 ¥{fee_parts['transfer_fee']:.2f}"
            + (f" + 印花税 ¥{fee_parts['stamp_duty']:.2f}" if fee_parts["stamp_duty"] else "")
        )
        order.reason = f"滑点 {self._slippage_bps:.1f}bp；{fee_text}"
        self._fees_paid += fee
        self._orders.append(order)

        return OrderResult(True, order, f"成交价 {fill_price:.2f}；费用 ¥{fee:.2f}（{fee_text}）")

    def cancel_order(self, order_id: str) -> bool:
        for o in self._orders:
            if o.order_id == order_id and o.status in (OrderStatus.SUBMITTED, OrderStatus.PENDING):
                o.status = OrderStatus.CANCELLED
                return True
        return False

    def get_orders(self, status_filter: list = None) -> List[Order]:
        if status_filter is None:
            return self._orders
        return [o for o in self._orders if o.status in status_filter]

    # ---- 更新市价 ----
    def update_price(self, symbol: str, price: float):
        if symbol in self._positions:
            self._positions[symbol]["last_price"] = price

    @staticmethod
    def _sellable_qty(position: dict) -> int:
        lots = position.get("lots")
        if not lots:
            return int(position.get("qty", 0))
        return sum(
            int(lot.get("qty", 0)) for lot in lots
            if lot.get("buy_date") < date.today()
        )

    def reset(self):
        self._cash = self._initial_cash
        self._positions = {}
        self._orders = []
        self._fees_paid = 0.0
        self._realized_pnl = 0.0
