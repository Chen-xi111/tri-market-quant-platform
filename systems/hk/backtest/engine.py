"""
回测引擎 — 遍历K线模拟交易，输出权益曲线 + 交易记录
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies.base import Strategy, Signal, SignalType
from trading_rules import calculate_trade_fee
import config as cfg


class BacktestEngine:
    """事件驱动回测引擎"""

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = None,
        commission: float = None,
        minimum_commission: float = None,
        stamp_duty_rate: float = None,
        sfc_levy_rate: float = None,
        afrc_levy_rate: float = None,
        trading_fee_rate: float = None,
        slippage_bps: float = 5.0,
        max_position_pct: float = None,
        risk_pct: float = None,
        lot_size: int = cfg.DEFAULT_LOT_SIZE,
        signal_lag_bars: int = 1,
        minimum_hold_bars: int = cfg.BACKTEST_MIN_HOLD_BARS,
    ):
        self.strategy = strategy
        self.capital = cfg.INITIAL_CAPITAL if initial_capital is None else float(initial_capital)
        self.commission = cfg.COMMISSION_RATE if commission is None else max(0.0, float(commission))
        self.minimum_commission = cfg.MIN_COMMISSION if minimum_commission is None else max(0.0, float(minimum_commission))
        self.stamp_duty_rate = cfg.STAMP_DUTY_RATE if stamp_duty_rate is None else max(0.0, float(stamp_duty_rate))
        self.sfc_levy_rate = cfg.SFC_LEVY_RATE if sfc_levy_rate is None else max(0.0, float(sfc_levy_rate))
        self.afrc_levy_rate = cfg.AFRC_LEVY_RATE if afrc_levy_rate is None else max(0.0, float(afrc_levy_rate))
        self.trading_fee_rate = cfg.TRADING_FEE_RATE if trading_fee_rate is None else max(0.0, float(trading_fee_rate))
        self.slippage_bps = max(0.0, float(slippage_bps))
        self.max_position_pct = (
            cfg.MAX_POSITION_PCT if max_position_pct is None else float(max_position_pct)
        )
        self.risk_pct = cfg.MAX_SINGLE_RISK_PCT if risk_pct is None else float(risk_pct)
        self.lot_size = max(1, int(lot_size))
        self.signal_lag_bars = max(0, int(signal_lag_bars))
        self.minimum_hold_bars = max(0, int(minimum_hold_bars))

        # 结果
        self.equity_curve: pd.Series = None
        self.trades: List[Tuple] = []
        self._position = 0         # 持仓股数
        self._cash = self.capital
        self._entry_price = 0.0
        self._entry_date = None
        self._trailing_high = 0.0
        self._entry_total_cost = 0.0
        self._active_stop = 0.0
        self._take_profit = 0.0
        self._entry_bar = -1

    def run(self, df: pd.DataFrame, trade_start_date=None) -> Dict:
        """
        执行回测
        返回: {"equity_curve": Series, "trades": [...], "metrics": {...}}
        """
        self.strategy.set_data(df.copy())
        df = self.strategy.generate_signals()

        equity = pd.Series(index=df.index, dtype=float)
        equity.iloc[0] = self.capital
        self._cash = self.capital
        self._position = 0
        self.trades = []
        self._entry_price = 0.0
        self._entry_date = None
        self._trailing_high = 0.0
        self._entry_total_cost = 0.0
        self._active_stop = 0.0
        self._take_profit = 0.0
        self._entry_bar = -1

        evaluation_start = (
            pd.Timestamp(trade_start_date) if trade_start_date is not None else df.index[0]
        )

        for i in range(1, len(df)):
            date = df.index[i]
            price = df["close"].iloc[i]
            open_price = df["open"].iloc[i]
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]
            signal_index = i - self.signal_lag_bars
            signal = df["signal"].iloc[signal_index] if signal_index >= 0 else 0
            signal_price = (
                df.get("signal_price", pd.Series(np.nan, index=df.index)).iloc[signal_index]
                if signal_index >= 0 else np.nan
            )
            stop_loss = (
                df.get("stop_loss", pd.Series(np.nan, index=df.index)).iloc[signal_index]
                if signal_index >= 0 else np.nan
            )
            take_profit = (
                df.get("take_profit", pd.Series(np.nan, index=df.index)).iloc[signal_index]
                if signal_index >= 0 else np.nan
            )

            # 样本外验证时，前段数据只用于指标预热，不允许提前建立持仓。
            if pd.Timestamp(date) < evaluation_start:
                equity.iloc[i] = self.capital
                continue

            closed_this_bar = False
            can_exit = self._position > 0 and i - self._entry_bar >= self.minimum_hold_bars
            execution_price = open_price if self.signal_lag_bars > 0 else (
                signal_price if not pd.isna(signal_price) else price
            )
            if signal == -1 and can_exit:
                self._close_position(execution_price, date, "策略信号(下一交易日开盘卖出)")
                closed_this_bar = True
            elif signal == 1 and self._position == 0:
                pos_pct = df.get("position_pct", pd.Series(0, index=df.index)).iloc[signal_index]
                if pos_pct == 0:
                    pos_pct = cfg.MAX_POSITION_PCT
                self._open_long(
                    execution_price, date, pos_pct, stop_loss, take_profit, entry_bar=i
                )
                can_exit = self._position > 0 and i - self._entry_bar >= self.minimum_hold_bars

            # ---- 检查止损止盈（使用入场时保存的保护价）----
            if self._position > 0 and can_exit and not closed_this_bar:
                hit_stop = self._active_stop > 0 and low <= self._active_stop
                hit_tp = self._take_profit > 0 and high >= self._take_profit

                if hit_stop:
                    # 跳空低开时不能假设仍能以止损价成交。
                    exit_price = min(self._active_stop, open_price)
                    reason = "保护止损"
                    self._close_position(exit_price, date, reason)
                    closed_this_bar = True
                elif hit_tp:
                    exit_price = self._take_profit
                    reason = "目标止盈"
                    self._close_position(exit_price, date, reason)
                    closed_this_bar = True
                else:
                    # 仅在盈利达到激活阈值后，从下一根 K 线开始上移止损，
                    # 避免同一根 K 线内先看见最高价再假定能按未知顺序止损。
                    self._trailing_high = max(self._trailing_high, high)
                    if self._trailing_high >= self._entry_price * (1 + cfg.TRAILING_ACTIVATE_PCT):
                        atr_value = df.get("ATR", df.get("atr", pd.Series(index=df.index, dtype=float))).iloc[i]
                        if pd.isna(atr_value) or atr_value <= 0:
                            atr_value = price * 0.02
                        trailing_stop = self._trailing_high - cfg.TRAILING_STOP_ATR_MULT * atr_value
                        self._active_stop = max(self._active_stop, trailing_stop)

            equity.iloc[i] = self._equity_value(price)

        # 强制平仓
        if self._position > 0 and len(df) - 1 - self._entry_bar >= self.minimum_hold_bars:
            last_price = df["close"].iloc[-1]
            self._close_position(last_price, df.index[-1], "回测结束平仓")
            equity.iloc[-1] = self._equity_value(last_price)

        self.equity_curve = equity.dropna()

        # 计算指标（含买入持有基准）
        from .metrics import compute_metrics
        # 基准：第一天全仓买入持有
        evaluation_position = int(df.index.searchsorted(evaluation_start))
        metric_start_position = max(0, evaluation_position - 1)
        evaluation_equity = self.equity_curve.iloc[metric_start_position:]
        evaluation_trades = [t for t in self.trades if pd.Timestamp(t[0]) >= evaluation_start]
        benchmark_df = df.loc[df.index >= evaluation_start]
        if len(benchmark_df) > 0:
            benchmark = pd.Series(self.capital, index=evaluation_equity.index, dtype=float)
            first_price = benchmark_df["open"].iloc[0]
            benchmark.loc[benchmark_df.index] = self.capital * benchmark_df["close"] / first_price
        else:
            benchmark = None
        metrics = compute_metrics(evaluation_equity, evaluation_trades, benchmark=benchmark)

        return {
            "equity_curve": self.equity_curve,
            "evaluation_equity_curve": evaluation_equity,
            "trades": self.trades,
            "metrics": metrics,
            "df": df,
            "evaluation_start": evaluation_start,
            "execution_assumptions": {
                "commission_rate": self.commission,
                "minimum_commission": self.minimum_commission,
                "stamp_duty_rate": self.stamp_duty_rate,
                "sfc_levy_rate": self.sfc_levy_rate,
                "afrc_levy_rate": self.afrc_levy_rate,
                "trading_fee_rate": self.trading_fee_rate,
                "slippage_bps": self.slippage_bps,
                "max_position_pct": self.max_position_pct,
                "risk_pct": self.risk_pct,
                "lot_size": self.lot_size,
                "signal_lag_bars": self.signal_lag_bars,
                "execution_price": "next_open" if self.signal_lag_bars > 0 else "signal_price",
                "minimum_hold_bars": self.minimum_hold_bars,
            },
            "open_position": self._position,
        }

    def _open_long(
        self,
        price: float,
        date,
        position_pct: float,
        stop_loss: float = np.nan,
        take_profit: float = np.nan,
        entry_bar: int = -1,
    ):
        """开多仓"""
        fill_price = float(price) * (1 + self.slippage_bps / 10_000)
        account_equity = self._equity_value(float(price))
        max_amount = account_equity * min(max(0.0, float(position_pct)), self.max_position_pct)
        value_limited = int(max_amount / fill_price)

        protective_stop = float(stop_loss) if pd.notna(stop_loss) and 0 < stop_loss < fill_price else fill_price * 0.95
        risk_per_share = max(fill_price - protective_stop, fill_price * 0.005)
        risk_limited = int(account_equity * self.risk_pct / risk_per_share)
        shares = min(value_limited, risk_limited)
        shares = (shares // self.lot_size) * self.lot_size
        if shares == 0:
            return
        cost = self._buy_total_cost(fill_price, shares)
        while shares > 0 and (cost > self._cash or cost > max_amount):
            shares -= self.lot_size
            cost = self._buy_total_cost(fill_price, shares)
        if shares == 0:
            return
        self._cash -= cost
        self._position = shares
        self._entry_price = fill_price
        self._entry_date = date
        self._trailing_high = fill_price
        self._entry_total_cost = cost
        self._active_stop = protective_stop
        self._take_profit = (
            float(take_profit)
            if pd.notna(take_profit) and take_profit > fill_price
            else fill_price + 2.0 * risk_per_share
        )
        self._entry_bar = int(entry_bar)

    def _close_position(self, price: float, date, reason: str):
        """平仓"""
        if self._position == 0:
            return
        fill_price = float(price) * (1 - self.slippage_bps / 10_000)
        gross = self._position * fill_price
        fees = calculate_trade_fee(
            gross,
            "SELL",
            commission_rate=self.commission,
            minimum_commission=self.minimum_commission,
            stamp_duty_rate=self.stamp_duty_rate,
            sfc_levy_rate=self.sfc_levy_rate,
            afrc_levy_rate=self.afrc_levy_rate,
            trading_fee_rate=self.trading_fee_rate,
        )
        proceeds = gross - fees["total"]
        pnl = proceeds - self._entry_total_cost
        pnl_pct = pnl / self._entry_total_cost * 100 if self._entry_total_cost else 0.0
        self._cash += proceeds

        self.trades.append((
            self._entry_date,
            date,
            self._entry_price,
            fill_price,
            pnl,
            pnl_pct,
            reason,
        ))

        self._position = 0
        self._entry_price = 0.0
        self._entry_date = None
        self._trailing_high = 0.0
        self._entry_total_cost = 0.0
        self._active_stop = 0.0
        self._take_profit = 0.0
        self._entry_bar = -1

    def _equity_value(self, current_price: float) -> float:
        return self._cash + self._position * current_price

    def _buy_total_cost(self, fill_price: float, shares: int) -> float:
        if shares <= 0:
            return 0.0
        gross = shares * fill_price
        fees = calculate_trade_fee(
            gross,
            "BUY",
            commission_rate=self.commission,
            minimum_commission=self.minimum_commission,
            stamp_duty_rate=self.stamp_duty_rate,
            sfc_levy_rate=self.sfc_levy_rate,
            afrc_levy_rate=self.afrc_levy_rate,
            trading_fee_rate=self.trading_fee_rate,
        )
        return gross + fees["total"]
