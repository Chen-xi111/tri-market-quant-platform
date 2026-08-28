"""
风控模块 — 仓位计算、止损止盈、风险预算
"""

import pandas as pd
import numpy as np
from typing import Optional
import config as cfg


class RiskManager:
    """交易风险管理器"""

    def __init__(self, capital: float = None):
        self.capital = capital or cfg.INITIAL_CAPITAL

    # ============================================================
    #  仓位计算
    # ============================================================
    def position_size(
        self,
        price: float,
        atr: float = None,
        risk_pct: float = None,
        max_position_pct: float = None,
        lot_size: int = 1,
    ) -> dict:
        """
        基于 ATR 的仓位计算
        原则：单笔最大亏损 ≤ 总资金 × risk_pct
        止损距离 = STOP_LOSS_ATR_MULT × ATR
        """
        risk_pct = risk_pct or cfg.MAX_SINGLE_RISK_PCT
        max_position_pct = max_position_pct or cfg.MAX_POSITION_PCT
        lot_size = max(1, int(lot_size))
        max_risk_amount = self.capital * risk_pct

        if atr and atr > 0:
            stop_distance = cfg.STOP_LOSS_ATR_MULT * atr
            shares = int(max_risk_amount / stop_distance) if stop_distance > 0 else 0
        else:
            # 无ATR时，用固定百分比
            stop_distance = price * 0.05
            shares = int(max_risk_amount / stop_distance)

        shares = (shares // lot_size) * lot_size
        position_value = shares * price
        # 不超过最大仓位
        max_value = self.capital * max_position_pct
        if position_value > max_value:
            shares = int(max_value / price)
            shares = (shares // lot_size) * lot_size
            position_value = shares * price

        planned_risk = shares * stop_distance

        return {
            "shares": shares,
            "position_value": position_value,
            "position_pct": position_value / self.capital * 100 if self.capital > 0 else 0,
            "max_risk": planned_risk,
            "risk_budget": max_risk_amount,
            "stop_loss_price": price - (cfg.STOP_LOSS_ATR_MULT * atr) if atr else price * 0.95,
            "take_profit_price": price + (cfg.TAKE_PROFIT_ATR_MULT * atr) if atr else price * 1.10,
        }

    # ============================================================
    #  动态止损
    # ============================================================
    @staticmethod
    def trailing_stop(
        entry_price: float,
        current_high: float,
        atr: float = None,
    ) -> float:
        """计算移动止损价"""
        if atr and atr > 0:
            return current_high - cfg.TRAILING_STOP_ATR_MULT * atr
        return current_high * (1 - cfg.TRAILING_ACTIVATE_PCT)

    # ============================================================
    #  风险检查
    # ============================================================
    def check_risk_limits(self, current_positions: dict) -> list:
        """
        检查风险限制，返回违规项
        current_positions: {"ticker": {"value": float, "pnl_pct": float}, ...}
        """
        violations = []
        total_value = sum(p["value"] for p in current_positions.values())

        # 单票仓位检查
        for ticker, pos in current_positions.items():
            pct = pos["value"] / self.capital * 100
            if pct > cfg.MAX_POSITION_PCT * 100:
                violations.append(f"⚠️ {ticker} 仓位 {pct:.1f}% 超限 (max {cfg.MAX_POSITION_PCT*100:.0f}%)")

        # 单票回撤检查
        for ticker, pos in current_positions.items():
            if pos["pnl_pct"] <= cfg.ALERT_LOSS_SL:
                violations.append(f"🚨 {ticker} 亏损 {pos['pnl_pct']:.1f}% 触及止损线")

        return violations

    # ============================================================
    #  情景分析
    # ============================================================
    def scenario_analysis(
        self,
        price: float,
        atr: float,
        risk_pct: float = None,
        max_position_pct: float = None,
        lot_size: int = 1,
    ) -> dict:
        """最佳/最差/基准情景"""
        pos = self.position_size(
            price,
            atr,
            risk_pct=risk_pct,
            max_position_pct=max_position_pct,
            lot_size=lot_size,
        )

        base_return = 0.0
        best_return = cfg.TAKE_PROFIT_ATR_MULT * atr / price * pos["position_pct"]
        worst_return = -cfg.STOP_LOSS_ATR_MULT * atr / price * pos["position_pct"]

        return {
            "position": pos,
            "best_case": {
                "price": pos["take_profit_price"],
                "return_pct": round(best_return, 2),
                "pnl": round(self.capital * best_return / 100, 2),
            },
            "base_case": {
                "return_pct": 0.0,
                "pnl": 0.0,
            },
            "worst_case": {
                "price": pos["stop_loss_price"],
                "return_pct": round(worst_return, 2),
                "pnl": round(self.capital * worst_return / 100, 2),
            },
        }
