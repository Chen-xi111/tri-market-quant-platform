"""
富途实盘券商 — 通过 FutuOpenD 网关连接真实账户
================================================
前置条件：
  1. 下载并启动 FutuOpenD（富途开放网关）
     https://www.futunn.com/download/openAPI
  2. 在 FutuOpenD 中登录你的富途账号
  3. 确认 FutuOpenD 监听在 localhost:11111
  4. 如需交易权限，在 FutuOpenD 设置中开启"交易接口"

使用方式：
  broker = FutuBroker(host="127.0.0.1", port=11111)
  if broker.connect():
      print(broker.get_account())
      result = broker.place_order("HK.01164", 2.65, 1000, OrderSide.BUY)
"""

from futu import (
    OpenSecTradeContext, OpenQuoteContext, TrdEnv, TrdMarket, TrdSide, OrderType,
    SecurityFirm, KLType, SubType, RET_OK,
)
import pandas as pd
from typing import List, Optional
from datetime import datetime

from .base import Broker, Order, OrderSide, OrderStatus, OrderResult


class FutuBroker(Broker):
    """
    富途券商 — 实盘交易接口

    港股代码格式: "HK.01164"
    交易环境: TrdEnv.REAL (实盘) / TrdEnv.SIMULATE (模拟)
    """

    name = "futu"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11111,
        trade_env: str = "SIMULATE",   # 默认模拟盘！先跑通再改 REAL
        market: str = "HK",
    ):
        self._host = host
        self._port = port

        # 安全默认：先连模拟盘
        self._trade_env = TrdEnv.SIMULATE if trade_env == "SIMULATE" else TrdEnv.REAL
        self._market = TrdMarket.HK if market == "HK" else TrdMarket.US

        self._quote_ctx: Optional[OpenQuoteContext] = None
        self._trade_ctx: Optional[OpenSecTradeContext] = None

    # ============================================================
    #  连接
    # ============================================================
    def connect(self) -> bool:
        try:
            self._quote_ctx = OpenQuoteContext(self._host, self._port)
            self._trade_ctx = OpenSecTradeContext(
                host=self._host, port=self._port,
                security_firm=SecurityFirm.FUTUSECURITIES,
            )
            # 解锁交易（如需要）
            if self._trade_env == TrdEnv.REAL:
                ret, msg = self._trade_ctx.unlock_trade("")  # 空密码=不设密码
                if ret != RET_OK:
                    print(f"[Futu] 解锁交易失败: {msg}")
            return True
        except Exception as e:
            print(f"[Futu] 连接失败: {e}")
            return False

    def disconnect(self):
        if self._quote_ctx:
            self._quote_ctx.close()
        if self._trade_ctx:
            self._trade_ctx.close()

    @property
    def is_connected(self) -> bool:
        return self._quote_ctx is not None and self._trade_ctx is not None

    # ============================================================
    #  账户
    # ============================================================
    def get_account(self) -> dict:
        if not self._trade_ctx:
            return {}

        ret, data = self._trade_ctx.accinfo_query(trd_env=self._trade_env)
        if ret != RET_OK or data.empty:
            return {}

        r = data.iloc[0]
        return {
            "total_assets": float(r.get("total_assets", 0)),
            "cash": float(r.get("cash", 0)),
            "market_value": float(r.get("market_val", 0)),
            "frozen_cash": float(r.get("frozen_cash", 0)),
            "available_cash": float(r.get("avl_withdrawal_cash", 0)),
        }

    def get_positions(self) -> list:
        if not self._trade_ctx:
            return []

        ret, data = self._trade_ctx.position_list_query(trd_env=self._trade_env)
        if ret != RET_OK or data.empty:
            return []

        positions = []
        for _, r in data.iterrows():
            positions.append({
                "symbol": r["code"],
                "qty": int(r["qty"]),
                "cost": float(r["cost_price"]),
                "price": float(r["nominal_price"]),
                "pnl": float(r["pl_val"]),
                "pnl_pct": float(r["pl_ratio"]) * 100,
            })
        return positions

    # ============================================================
    #  交易
    # ============================================================
    def place_order(
        self, symbol: str, price: float, quantity: int, side: OrderSide
    ) -> OrderResult:
        if not self._trade_ctx:
            return OrderResult(False, message="未连接交易接口")

        trd_side = TrdSide.BUY if side == OrderSide.BUY else TrdSide.SELL

        ret, data = self._trade_ctx.place_order(
            price=price,
            qty=quantity,
            code=symbol,
            trd_side=trd_side,
            order_type=OrderType.NORMAL,      # 限价单
            trd_env=self._trade_env,
            adjust_limit=0.0,                  # 不调整限价
        )

        if ret != RET_OK:
            return OrderResult(False, message=str(data))

        r = data
        order = Order(
            order_id=str(r.get("order_id", "")),
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            status=OrderStatus.SUBMITTED,
            filled_qty=int(r.get("dealt_qty", 0)),
        )

        return OrderResult(True, order, "已提交")

    def cancel_order(self, order_id: str) -> bool:
        if not self._trade_ctx:
            return False
        ret, _ = self._trade_ctx.cancel_order(order_id=order_id, trd_env=self._trade_env)
        return ret == RET_OK

    def get_orders(self, status_filter: list = None) -> List[Order]:
        if not self._trade_ctx:
            return []

        ret, data = self._trade_ctx.order_list_query(trd_env=self._trade_env)
        if ret != RET_OK or data.empty:
            return []

        status_map = {
            "SUBMITTING": OrderStatus.SUBMITTED,
            "SUBMITTED": OrderStatus.SUBMITTED,
            "FILLED_ALL": OrderStatus.FILLED,
            "FILLED_PART": OrderStatus.PARTIAL,
            "CANCELLED_ALL": OrderStatus.CANCELLED,
            "CANCELLED_PART": OrderStatus.CANCELLED,
            "DISABLED": OrderStatus.REJECTED,
            "FAILED": OrderStatus.ERROR,
        }

        orders = []
        for _, r in data.iterrows():
            s = status_map.get(r.get("order_status", ""), OrderStatus.PENDING)
            o = Order(
                order_id=str(r.get("order_id", "")),
                symbol=r["code"],
                side=OrderSide.BUY if r.get("trd_side", "") == "BUY" else OrderSide.SELL,
                price=float(r["price"]),
                quantity=int(r["qty"]),
                filled_qty=int(r.get("dealt_qty", 0)),
                status=s,
            )
            if status_filter is None or s in status_filter:
                orders.append(o)

        return orders

    # ============================================================
    #  行情（复用 Futu 行情接口，比 yfinance 更快）
    # ============================================================
    def get_quote(self, symbol: str) -> dict:
        if not self._quote_ctx:
            return {}

        ret, data = self._quote_ctx.get_market_snapshot([symbol])
        if ret != RET_OK or data.empty:
            return {}

        r = data.iloc[0]
        return {
            "price": float(r.get("last_price", 0)),
            "open": float(r.get("open_price", 0)),
            "high": float(r.get("high_price", 0)),
            "low": float(r.get("low_price", 0)),
            "prev_close": float(r.get("prev_close_price", 0)),
            "volume": float(r.get("volume", 0)),
        }
