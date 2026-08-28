"""
三层安全闸门 — 在策略信号和下单之间做最后拦截
"""

from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from .base import Order, OrderSide
import config as cfg


@dataclass
class SafetyCheck:
    """安全检查结果"""
    passed: bool
    reason: str = ""
    level: str = "PASS"     # PASS / WARN / BLOCK


class SafetyGate:
    """
    多层安全闸门

    第一层：订单限额（单笔最大金额、最大股数）
    第二层：频率限制（N秒内最多M笔）
    第三层：日内风控（当日累计亏损上限、连续亏损次数上限）
    """

    def __init__(
        self,
        max_order_value: float = None,       # 单笔最大金额
        max_position_pct: float = None,       # 最大仓位
        daily_loss_limit: float = None,       # 日内亏损上限（金额）
        max_consecutive_losses: int = 5,      # 最大连续亏损笔数
        cooldown_seconds: float = 10.0,       # 下单冷却间隔
    ):
        self.max_order_value = max_order_value or cfg.INITIAL_CAPITAL * cfg.MAX_POSITION_PCT
        self.max_position_pct = max_position_pct or cfg.MAX_POSITION_PCT
        self.daily_loss_limit = daily_loss_limit or cfg.INITIAL_CAPITAL * 0.05
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_seconds = cooldown_seconds

        # 状态跟踪
        self._order_log: List[dict] = []
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._last_order_time: Optional[datetime] = None

    # ============================================================
    #  检查入口：下单前调用
    # ============================================================
    def check_order(
        self,
        symbol: str,
        price: float,
        quantity: int,
        side: OrderSide,
        current_position: int = 0,
        account_cash: float = 0,
        account_total: float = 0,
    ) -> SafetyCheck:
        """
        下单前三层检查
        返回 SafetyCheck，passed=False 时订单被拦截
        """

        order_value = price * quantity

        if price <= 0 or quantity <= 0:
            return SafetyCheck(False, "价格和数量必须大于 0", "BLOCK")

        if side == OrderSide.SELL and quantity > current_position:
            return SafetyCheck(
                False,
                f"持仓不足：可卖 {current_position:,} 股，申请 {quantity:,} 股",
                "BLOCK",
            )

        # ---- 第一层：订单限额 ----
        # 卖出是降低风险的操作，不应因持仓升值超过买入上限而无法退出。
        if side == OrderSide.BUY and order_value > self.max_order_value:
            return SafetyCheck(
                False,
                f"订单金额 ¥{order_value:,.0f} 超过单笔上限 ¥{self.max_order_value:,.0f}",
                "BLOCK",
            )

        if account_total > 0:
            pos_pct = (current_position * price + order_value) / account_total
            if side == OrderSide.BUY and pos_pct > self.max_position_pct:
                return SafetyCheck(
                    False,
                    f"仓位将达 {pos_pct*100:.1f}% 超过上限 {self.max_position_pct*100:.0f}%",
                    "BLOCK",
                )

        if side == OrderSide.BUY and account_cash > 0 and order_value > account_cash:
            return SafetyCheck(
                False,
                f"资金不足：需 ¥{order_value:,.0f}，可用 ¥{account_cash:,.0f}",
                "BLOCK",
            )

        # ---- 第二层：频率限制 ----
        if side == OrderSide.BUY and self._last_order_time:
            elapsed = (datetime.now() - self._last_order_time).total_seconds()
            if elapsed < self.cooldown_seconds:
                return SafetyCheck(
                    False,
                    f"下单过于频繁（间隔 {elapsed:.1f}s，需 ≥ {self.cooldown_seconds}s）",
                    "BLOCK",
                )

        # ---- 第三层：日内风控 ----
        if side == OrderSide.BUY and self._daily_pnl <= -self.daily_loss_limit:
            return SafetyCheck(
                False,
                f"日内亏损 ¥{self._daily_pnl:,.0f} 已达上限 ¥{self.daily_loss_limit:,.0f}，今日禁止交易",
                "BLOCK",
            )

        if side == OrderSide.BUY and self._consecutive_losses >= self.max_consecutive_losses:
            return SafetyCheck(
                False,
                f"连续亏损 {self._consecutive_losses} 笔已达上限 {self.max_consecutive_losses}，暂停交易",
                "BLOCK",
            )

        # 全部通过
        self._last_order_time = datetime.now()
        self._order_log.append({
            "time": datetime.now(),
            "symbol": symbol,
            "side": side.value,
            "value": order_value,
        })

        return SafetyCheck(True, "安全检查通过", "PASS")

    # ============================================================
    #  成交后更新
    # ============================================================
    def on_fill(self, pnl: float):
        """成交后更新日内统计"""
        self._daily_pnl += pnl
        if pnl <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    # ============================================================
    #  重置（每日开盘调用）
    # ============================================================
    def reset_daily(self):
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._order_log = []

    # ============================================================
    #  状态查询
    # ============================================================
    def status(self) -> dict:
        return {
            "daily_pnl": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "daily_loss_limit": self.daily_loss_limit,
            "today_orders": len(self._order_log),
            "last_order": self._last_order_time.strftime("%H:%M:%S") if self._last_order_time else "—",
            "blocked": self._daily_pnl <= -self.daily_loss_limit,
        }
