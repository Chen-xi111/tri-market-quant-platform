"""Interactive Brokers TWS API 适配器。

安全边界：界面只暴露纸账户端口。真实账户端口必须在代码层显式关闭
``paper_only`` 才能使用，避免把一次配置错误变成真实订单。
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import importlib.util
import math
import socket
import threading
from time import monotonic
from typing import List, Optional

import config as cfg
from .base import Broker, Order, OrderResult, OrderSide, OrderStatus


IBAPI_AVAILABLE = importlib.util.find_spec("ibapi") is not None
PAPER_PORTS = frozenset({7497, 4002})
LIVE_PORTS = frozenset({7496, 4001})


def ibapi_available() -> bool:
    return IBAPI_AVAILABLE


if IBAPI_AVAILABLE:
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.order import Order as IBOrder
    from ibapi.wrapper import EWrapper


    class _IBApp(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self.connected_event = threading.Event()
            self.account_event = threading.Event()
            self.positions_event = threading.Event()
            self.open_orders_event = threading.Event()
            self._lock = threading.RLock()
            self._next_order_id: Optional[int] = None
            self.accounts: list[str] = []
            self.account_values: dict[str, dict[tuple[str, str], float]] = {}
            self.positions: dict[str, dict] = {}
            self.orders: dict[int, Order] = {}
            self.order_events: dict[int, threading.Event] = {}
            self.quote_values: dict[int, dict] = {}
            self.quote_events: dict[int, threading.Event] = {}
            self.request_errors: dict[int, str] = {}
            self.last_error = ""
            self._request_id = 10_000

        def nextValidId(self, orderId):  # noqa: N802 - IBKR callback name
            with self._lock:
                self._next_order_id = int(orderId)
            self.connected_event.set()

        def managedAccounts(self, accountsList):  # noqa: N802
            self.accounts = [value for value in accountsList.split(",") if value]

        def connectionClosed(self):  # noqa: N802
            self.connected_event.clear()
            # 保留先前更具体的 API 错误，避免在连接失败时被笼统提示覆盖。
            if not self.last_error:
                self.last_error = "IBKR 连接已关闭"

        def error(self, reqId, *args):  # noqa: N802
            # 兼容旧版 (code, message, reject) 和新版
            # (errorTime, code, message, reject) 回调签名。
            if len(args) >= 3 and isinstance(args[1], int):
                error_code, message = int(args[1]), str(args[2])
            elif len(args) >= 2:
                error_code, message = int(args[0]), str(args[1])
            else:
                error_code, message = -1, " ".join(map(str, args))
            informational = {2104, 2106, 2107, 2108, 2158}
            if error_code not in informational:
                detail = f"IBKR {error_code}: {message}"
                self.last_error = detail
                try:
                    request_id = int(reqId)
                except (TypeError, ValueError):
                    request_id = -1
                self.request_errors[request_id] = detail
                event = self.order_events.get(request_id) or self.quote_events.get(request_id)
                if event:
                    event.set()

        def accountSummary(self, reqId, account, tag, value, currency):  # noqa: N802
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return
            self.account_values.setdefault(account, {})[(tag, currency or "BASE")] = numeric

        def accountSummaryEnd(self, reqId):  # noqa: N802
            self.account_event.set()

        def position(self, account, contract, position, avgCost):
            if getattr(contract, "secType", "") != "STK":
                return
            qty = float(position)
            if abs(qty) < 1e-12:
                return
            symbol = str(contract.symbol).upper()
            self.positions[symbol] = {
                "symbol": symbol,
                "qty": int(qty) if qty.is_integer() else qty,
                "cost": float(avgCost),
                "price": float(avgCost),
                "pnl": 0.0,
                "account": account,
            }

        def positionEnd(self):  # noqa: N802
            self.positions_event.set()

        @staticmethod
        def _map_status(value: str) -> OrderStatus:
            status = str(value or "")
            if status == "Filled":
                return OrderStatus.FILLED
            if status in {"Cancelled", "ApiCancelled"}:
                return OrderStatus.CANCELLED
            if status == "Inactive":
                return OrderStatus.REJECTED
            if status in {"PendingSubmit", "PendingCancel", "PreSubmitted", "Submitted"}:
                return OrderStatus.SUBMITTED
            return OrderStatus.PENDING

        def openOrder(self, orderId, contract, order, orderState):  # noqa: N802
            order_id = int(orderId)
            side = OrderSide.BUY if str(order.action).upper() == "BUY" else OrderSide.SELL
            quantity = float(order.totalQuantity)
            price = float(getattr(order, "lmtPrice", 0) or 0)
            if not math.isfinite(price):
                price = 0.0
            local_order = self.orders.get(order_id) or Order(
                order_id=str(order_id),
                symbol=str(contract.symbol).upper(),
                side=side,
                price=price,
                quantity=int(quantity) if quantity.is_integer() else quantity,
            )
            local_order.status = self._map_status(getattr(orderState, "status", ""))
            self.orders[order_id] = local_order

        def openOrderEnd(self):  # noqa: N802
            self.open_orders_event.set()

        def orderStatus(  # noqa: N802
            self, orderId, status, filled, remaining, avgFillPrice, permId,
            parentId, lastFillPrice, clientId, whyHeld, mktCapPrice,
        ):
            order_id = int(orderId)
            local_order = self.orders.get(order_id)
            if local_order:
                filled_value = float(filled)
                local_order.filled_qty = (
                    int(filled_value) if filled_value.is_integer() else filled_value
                )
                local_order.status = self._map_status(status)
                if float(avgFillPrice or 0) > 0:
                    local_order.price = float(avgFillPrice)
                local_order.reason = str(whyHeld or "")
                local_order.update_time = datetime.now()
            event = self.order_events.get(order_id)
            if event:
                event.set()

        def next_request_id(self) -> int:
            with self._lock:
                self._request_id += 1
                return self._request_id

        def next_order_id(self) -> int:
            with self._lock:
                if self._next_order_id is None:
                    raise RuntimeError("IBKR 尚未返回 nextValidId")
                value = self._next_order_id
                self._next_order_id += 1
                return value

        def tickPrice(self, reqId, tickType, price, attrib):  # noqa: N802
            if price is None or float(price) <= 0:
                return
            names = {
                1: "bid", 2: "ask", 4: "last", 6: "high", 7: "low", 9: "prev_close",
                14: "open", 66: "bid", 67: "ask", 68: "last", 72: "high",
                73: "low", 75: "prev_close", 76: "open",
            }
            key = names.get(int(tickType))
            if key:
                values = self.quote_values.setdefault(int(reqId), {})
                values[key] = float(price)
                values.setdefault(
                    "_market_data_type",
                    "delayed" if int(tickType) >= 66 else "live",
                )
                if key == "last":
                    event = self.quote_events.get(int(reqId))
                    if event:
                        event.set()

        def tickSize(self, reqId, tickType, size):  # noqa: N802
            if int(tickType) in {8, 74}:
                self.quote_values.setdefault(int(reqId), {})["volume"] = float(size)

        def tickSnapshotEnd(self, reqId):  # noqa: N802
            event = self.quote_events.get(int(reqId))
            if event:
                event.set()

        def marketDataType(self, reqId, marketDataType):  # noqa: N802
            labels = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed_frozen"}
            self.quote_values.setdefault(int(reqId), {})["_market_data_type"] = (
                labels.get(int(marketDataType), f"type_{marketDataType}")
            )


class IBKRBroker(Broker):
    """IBKR 股票券商适配器；默认只允许 TWS/IB Gateway 纸账户。"""

    name = "ibkr"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 17,
        paper_only: bool = True,
        allow_outside_rth: bool = False,
        timeout: float = 5.0,
    ):
        self.host = host
        self.port = int(port)
        self.client_id = int(client_id)
        self.paper_only = bool(paper_only)
        self.allow_outside_rth = bool(allow_outside_rth)
        self.timeout = max(1.0, float(timeout))
        self._app = None
        self._thread: Optional[threading.Thread] = None
        self._last_error = ""

    @property
    def last_error(self) -> str:
        if self._app is not None and getattr(self._app, "last_error", ""):
            return self._app.last_error
        return self._last_error

    def connect(self) -> bool:
        if self.paper_only and self.port not in PAPER_PORTS:
            self._last_error = f"纸账户安全模式拒绝端口 {self.port}；请使用 7497 或 4002"
            return False
        if self.paper_only and self.port in LIVE_PORTS:
            self._last_error = "拒绝连接 IBKR 实盘端口"
            return False
        if not IBAPI_AVAILABLE:
            self._last_error = "未安装 IBKR 官方 ibapi；请查看 IBKR_SETUP.md"
            return False
        try:
            # TWS 关闭 Socket API 时，ibapi 默认会等到超时且最终只给出
            # connectionClosed。先做一次本机端口探测，可以立即给出可操作提示。
            with socket.create_connection(
                (self.host, self.port), timeout=min(self.timeout, 1.0),
            ):
                pass
        except OSError:
            app_name = "TWS Paper" if self.port == 7497 else "IB Gateway Paper"
            self._last_error = (
                f"{self.host}:{self.port} 未监听。{app_name} 已登录时，请打开“全局配置 → "
                "API → 设置”，勾选“启用 ActiveX 和 Socket 客户端”，并确认 "
                f"Socket 端口为 {self.port}。"
            )
            return False
        try:
            app = _IBApp()
            app.connect(self.host, self.port, self.client_id)
            thread = threading.Thread(target=app.run, name="ibkr-tws-api", daemon=True)
            thread.start()
            if not app.connected_event.wait(self.timeout) or not app.isConnected():
                self._last_error = app.last_error or (
                    f"无法连接 {self.host}:{self.port}；请启动并登录 TWS/IB Gateway Paper"
                )
                try:
                    app.disconnect()
                except Exception:
                    pass
                return False
            self._app = app
            self._thread = thread
            return True
        except Exception as exc:
            self._last_error = f"IBKR 连接失败: {type(exc).__name__}: {exc}"
            return False

    def disconnect(self):
        if self._app is not None:
            try:
                self._app.disconnect()
            finally:
                self._app = None

    @property
    def is_connected(self) -> bool:
        return bool(self._app is not None and self._app.isConnected())

    def _require_connection(self):
        if not self.is_connected:
            raise RuntimeError(self.last_error or "IBKR Paper 未连接")
        return self._app

    @staticmethod
    def _stock_contract(symbol: str):
        if not IBAPI_AVAILABLE:
            raise RuntimeError("未安装 ibapi")
        normalized = cfg.normalize_ticker(symbol)[1]
        contract = Contract()
        contract.symbol = normalized
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = cfg.CURRENCY
        primary_exchange = cfg.ibkr_primary_exchange(normalized)
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        return contract

    @staticmethod
    def _value(values: dict, tag: str, currency: str = "USD") -> float:
        for key in ((tag, currency), (tag, "BASE"), (tag, "")):
            if key in values:
                return float(values[key])
        return 0.0

    def get_account(self) -> dict:
        app = self._require_connection()
        app.account_values = {}
        app.account_event.clear()
        request_id = app.next_request_id()
        tags = "NetLiquidation,TotalCashValue,AvailableFunds,BuyingPower"
        app.reqAccountSummary(request_id, "All", tags)
        app.account_event.wait(self.timeout)
        try:
            app.cancelAccountSummary(request_id)
        except Exception:
            pass
        if not app.account_values:
            raise RuntimeError(app.request_errors.get(request_id) or "IBKR 未返回账户摘要")
        account_id = app.accounts[0] if app.accounts else next(iter(app.account_values))
        values = app.account_values.get(account_id) or next(iter(app.account_values.values()))
        total = self._value(values, "NetLiquidation")
        cash = self._value(values, "TotalCashValue")
        available = self._value(values, "AvailableFunds")
        return {
            "account_id": account_id,
            "currency": cfg.CURRENCY,
            "total_assets": total,
            "cash": cash,
            "available_cash": available,
            "buying_power": self._value(values, "BuyingPower"),
            "market_value": max(0.0, total - cash),
            "frozen_cash": 0.0,
        }

    def get_positions(self) -> list:
        app = self._require_connection()
        app.positions = {}
        app.positions_event.clear()
        app.reqPositions()
        app.positions_event.wait(self.timeout)
        try:
            app.cancelPositions()
        except Exception:
            pass
        return list(app.positions.values())

    def place_order(
        self, symbol: str, price: float, quantity: int, side: OrderSide
    ) -> OrderResult:
        app = self._require_connection()
        if self.paper_only and self.port not in PAPER_PORTS:
            return OrderResult(False, message="纸账户安全模式拒绝当前端口")
        if int(quantity) != quantity or quantity <= 0:
            return OrderResult(False, message="当前平台只允许正整数股订单")
        if price <= 0:
            return OrderResult(False, message="限价必须大于 0")
        if not self.allow_outside_rth:
            from market_clock import is_us_regular_session

            if not is_us_regular_session():
                return OrderResult(False, message="当前不是美股常规交易时段，订单未发送")

        order_id = app.next_order_id()
        local_order = Order(
            order_id=str(order_id),
            symbol=cfg.normalize_ticker(symbol)[1],
            side=side,
            price=round(float(price), 2),
            quantity=int(quantity),
            status=OrderStatus.PENDING,
        )
        app.orders[order_id] = local_order
        event = threading.Event()
        app.order_events[order_id] = event
        app.request_errors.pop(order_id, None)

        ib_order = IBOrder()
        ib_order.action = side.value
        ib_order.orderType = "LMT"
        ib_order.totalQuantity = Decimal(int(quantity))
        ib_order.lmtPrice = round(float(price), 2)
        ib_order.tif = "DAY"
        ib_order.outsideRth = self.allow_outside_rth
        ib_order.transmit = True
        app.placeOrder(order_id, self._stock_contract(symbol), ib_order)
        event.wait(self.timeout)

        error = app.request_errors.get(order_id)
        if error:
            local_order.status = OrderStatus.REJECTED
            local_order.reason = error
            return OrderResult(False, local_order, error)
        if local_order.status in {OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.ERROR}:
            return OrderResult(False, local_order, local_order.reason or "IBKR 拒绝订单")
        if local_order.status == OrderStatus.FILLED:
            return OrderResult(True, local_order, f"IBKR Paper 已成交 ${local_order.price:.2f}")
        if local_order.status == OrderStatus.PARTIAL:
            return OrderResult(True, local_order, "IBKR Paper 部分成交，等待后续回报")
        local_order.status = OrderStatus.SUBMITTED
        return OrderResult(True, local_order, "IBKR Paper 已接收，尚未确认成交")

    def cancel_order(self, order_id: str) -> bool:
        app = self._require_connection()
        event = app.order_events.setdefault(int(order_id), threading.Event())
        event.clear()
        try:
            app.cancelOrder(int(order_id), "")
        except TypeError:
            app.cancelOrder(int(order_id))
        event.wait(min(self.timeout, 2.0))
        order = app.orders.get(int(order_id))
        return bool(order and order.status == OrderStatus.CANCELLED)

    def get_orders(self, status_filter: list = None) -> List[Order]:
        app = self._require_connection()
        app.open_orders_event.clear()
        app.reqOpenOrders()
        app.open_orders_event.wait(min(self.timeout, 2.0))
        orders = list(app.orders.values())
        orders.sort(key=lambda value: value.update_time)
        if status_filter is None:
            return orders
        return [order for order in orders if order.status in status_filter]

    def get_quote(self, symbol: str) -> dict:
        app = self._require_connection()
        request_id = app.next_request_id()
        event = threading.Event()
        app.quote_events[request_id] = event
        app.quote_values[request_id] = {}
        app.request_errors.pop(request_id, None)
        # 有权限时 IBKR 仍优先返回实时类型 1；无权限才回退延迟类型 3。
        app.reqMarketDataType(3)
        app.reqMktData(request_id, self._stock_contract(symbol), "", True, False, [])
        started = monotonic()
        event.wait(self.timeout)
        try:
            app.cancelMktData(request_id)
        except Exception:
            pass
        values = app.quote_values.pop(request_id, {})
        app.quote_events.pop(request_id, None)
        if not values:
            return {"error": app.request_errors.get(request_id) or "IBKR 无报价权限或响应超时"}
        bid, ask = values.get("bid"), values.get("ask")
        price = values.get("last")
        if not price and bid and ask:
            price = (bid + ask) / 2
        price = price or values.get("prev_close")
        previous = values.get("prev_close")
        market_data_type = str(values.get("_market_data_type") or "unknown")
        source_suffix = {
            "live": "实时", "frozen": "冻结", "delayed": "延迟",
            "delayed_frozen": "延迟冻结",
        }.get(market_data_type, "类型待确认")
        values.update({
            "ticker": cfg.normalize_ticker(symbol)[0],
            "price": price,
            "change": (price - previous) if price and previous else 0.0,
            "change_pct": ((price / previous - 1) * 100) if price and previous else 0.0,
            "currency": cfg.CURRENCY,
            "source": f"IBKR TWS（{source_suffix}）",
            "_received_at": datetime.now(timezone.utc).isoformat(),
            "_market_data_type": market_data_type,
            "latency_seconds": round(monotonic() - started, 3),
        })
        return values
