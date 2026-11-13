"""短周期策略的 A 股规则回测与时间外验证。"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parents[1]
CN_DIR = PROJECT_DIR / "systems" / "cn"
for import_path in (PROJECT_DIR, CN_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from systems.cn_lab.action_strategy import (
    PROFILES,
    StrategyMode,
    buy_lot_size,
    prepare_features,
)
from trading_rules import calculate_trade_fee, price_limit_bounds


@dataclass(frozen=True)
class Trade:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    net_pnl: float
    return_pct: float
    holding_bars: int
    exit_reason: str
    fees: float


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    metrics: dict[str, float | int | str | tuple[float, float]]
    assumptions: dict[str, str]
    open_position: dict | None = None
    fold_metrics: list[dict] = field(default_factory=list)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(trade) for trade in self.trades])


def _wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    probability = wins / total
    denominator = 1 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    margin = z * sqrt(probability * (1 - probability) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _is_locked_limit(bar: pd.Series, previous_close: float, ticker: str, side: str) -> bool:
    lower, upper = price_limit_bounds(previous_close, ticker)
    limit_price = upper if side == "BUY" else lower
    prices = [float(bar[column]) for column in ("open", "high", "low", "close")]
    return max(abs(price - limit_price) for price in prices) <= 0.011


def _metrics(
    equity: pd.Series,
    trades: list[Trade],
    initial_capital: float,
    total_fees: float,
) -> dict:
    if equity.empty:
        equity = pd.Series([initial_capital], dtype=float)
    closed_returns = np.array([trade.return_pct / 100 for trade in trades], dtype=float)
    wins = int((closed_returns > 0).sum())
    losses = int((closed_returns <= 0).sum())
    win_rate = wins / len(trades) if trades else 0.0
    confidence_low, confidence_high = _wilson_interval(wins, len(trades))
    gains = sum(max(0.0, trade.net_pnl) for trade in trades)
    losses_value = abs(sum(min(0.0, trade.net_pnl) for trade in trades))
    profit_factor = (
        gains / losses_value if losses_value > 0 else (float("inf") if gains > 0 else 0.0)
    )
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1
    total_return = equity.iloc[-1] / initial_capital - 1
    daily_returns = equity.pct_change().dropna()
    sharpe = 0.0
    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252))
    expectancy = float(np.mean([trade.net_pnl for trade in trades])) if trades else 0.0
    average_holding = float(np.mean([trade.holding_bars for trade in trades])) if trades else 0.0

    if len(trades) < 12:
        status = "样本不足"
    elif profit_factor >= 1.2 and confidence_low >= 0.42 and total_return > 0:
        status = "初步可用"
    elif profit_factor >= 1.0 and total_return >= 0:
        status = "观察验证"
    else:
        status = "不建议实操"

    return {
        "total_return": float(total_return),
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": sharpe,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "win_rate_ci": (confidence_low, confidence_high),
        "profit_factor": float(profit_factor),
        "expectancy": expectancy,
        "average_holding_bars": average_holding,
        "total_fees": float(total_fees),
        "status": status,
    }


def run_backtest(
    features: pd.DataFrame,
    *,
    ticker: str,
    mode: StrategyMode,
    initial_capital: float = 100_000.0,
    risk_pct: float = 0.01,
    max_position_pct: float = 0.30,
    slippage_bps: float = 5.0,
    start_position: int = 60,
    end_position: int | None = None,
) -> BacktestResult:
    """用下一交易日开盘成交、T+1 和涨跌停约束回测。"""
    profile = PROFILES[mode]
    lot_size = buy_lot_size(ticker)
    frame = features.copy().sort_index()
    if end_position is None:
        end_position = len(frame)
    start_position = max(1, min(int(start_position), len(frame) - 1))
    end_position = max(start_position + 1, min(int(end_position), len(frame)))

    cash = float(initial_capital)
    position: dict | None = None
    trades: list[Trade] = []
    equity_values: list[float] = []
    equity_dates: list[pd.Timestamp] = []
    total_fees = 0.0

    def execute_sell(quantity: int, price: float) -> tuple[float, float]:
        nonlocal cash, total_fees
        gross = quantity * price
        fee = calculate_trade_fee(gross, "SELL")["total"]
        cash += gross - fee
        total_fees += fee
        return gross, fee

    def close_cycle(date: pd.Timestamp, reason: str) -> None:
        nonlocal position
        assert position is not None
        weighted_exit = position["exit_value"] / position["original_qty"]
        gross_pnl = position["exit_value"] - position["entry_price"] * position["original_qty"]
        net_pnl = gross_pnl - position["entry_fee"] - position["exit_fees"]
        invested = position["entry_price"] * position["original_qty"] + position["entry_fee"]
        trades.append(
            Trade(
                signal_date=position["signal_date"],
                entry_date=position["entry_date"],
                exit_date=pd.Timestamp(date),
                entry_price=round(position["entry_price"], 4),
                exit_price=round(weighted_exit, 4),
                quantity=position["original_qty"],
                gross_pnl=round(gross_pnl, 2),
                net_pnl=round(net_pnl, 2),
                return_pct=round(net_pnl / invested * 100, 3) if invested else 0.0,
                holding_bars=position["bars_held"],
                exit_reason=reason,
                fees=round(position["entry_fee"] + position["exit_fees"], 2),
            )
        )
        position = None

    for index_position in range(start_position, end_position):
        date = pd.Timestamp(frame.index[index_position])
        bar = frame.iloc[index_position]
        previous = frame.iloc[index_position - 1]
        previous_close = float(previous["close"])
        exited_today = False

        if position is not None:
            position["bars_held"] = index_position - position["entry_position"]
            can_sell = position["bars_held"] >= 1
            locked_down = _is_locked_limit(bar, previous_close, ticker, "SELL")
            if can_sell and not locked_down:
                stop = float(position["stop"])
                target_1 = float(position["target_1"])
                target_2 = float(position["target_2"])
                exit_price = None
                exit_reason = ""

                # 同一根日 K 同时触及止损和止盈时，保守地先按止损成交。
                if float(bar["open"]) <= stop:
                    exit_price = float(bar["open"]) * (1 - slippage_bps / 10_000)
                    exit_reason = "跳空止损"
                elif float(bar["low"]) <= stop:
                    exit_price = stop * (1 - slippage_bps / 10_000)
                    exit_reason = "计划止损"

                if exit_price is not None:
                    gross, fee = execute_sell(position["qty"], exit_price)
                    position["exit_value"] += gross
                    position["exit_fees"] += fee
                    position["qty"] = 0
                    close_cycle(date, exit_reason)
                    exited_today = True
                else:
                    if not position["partial_taken"] and float(bar["high"]) >= target_1:
                        quantity = position["qty"]
                        if quantity >= 2 * lot_size:
                            quantity = max(lot_size, (quantity // (2 * lot_size)) * lot_size)
                        gross, fee = execute_sell(quantity, target_1 * (1 - slippage_bps / 10_000))
                        position["exit_value"] += gross
                        position["exit_fees"] += fee
                        position["qty"] -= quantity
                        position["partial_taken"] = True
                        position["stop"] = max(position["stop"], position["entry_price"])
                        if position["qty"] == 0:
                            close_cycle(date, "第一止盈")
                            exited_today = True

                    if position is not None and float(bar["high"]) >= target_2:
                        gross, fee = execute_sell(
                            position["qty"], target_2 * (1 - slippage_bps / 10_000)
                        )
                        position["exit_value"] += gross
                        position["exit_fees"] += fee
                        position["qty"] = 0
                        close_cycle(date, "第二止盈")
                        exited_today = True
                    elif position is not None and position["bars_held"] >= profile.max_hold_bars:
                        exit_price = float(bar["close"]) * (1 - slippage_bps / 10_000)
                        gross, fee = execute_sell(position["qty"], exit_price)
                        position["exit_value"] += gross
                        position["exit_fees"] += fee
                        position["qty"] = 0
                        close_cycle(date, "到期退出")
                        exited_today = True

            if position is not None:
                position["highest"] = max(position["highest"], float(bar["high"]))
                if position["partial_taken"]:
                    trailing = position["highest"] - 1.1 * float(bar["atr14"])
                    position["stop"] = max(position["stop"], trailing)

        if position is None and not exited_today and bool(previous.get("entry_signal", False)):
            if index_position - 1 < start_position:
                pass
            elif not _is_locked_limit(bar, previous_close, ticker, "BUY"):
                open_price = float(bar["open"])
                entry_low = float(previous["entry_low"])
                entry_high = float(previous["entry_high"])
                if entry_low * 0.985 <= open_price <= entry_high * 1.01:
                    entry_price = open_price * (1 + slippage_bps / 10_000)
                    stop = float(previous["stop_loss"])
                    if stop >= entry_price * 0.995:
                        stop = entry_price - profile.stop_atr * float(previous["atr14"])
                    risk_per_share = max(0.01, entry_price - stop)
                    risk_limited = initial_capital * risk_pct / risk_per_share
                    value_limited = initial_capital * max_position_pct / entry_price
                    quantity = int(min(risk_limited, value_limited) // lot_size) * lot_size
                    while quantity >= lot_size:
                        entry_value = entry_price * quantity
                        entry_fee = calculate_trade_fee(entry_value, "BUY")["total"]
                        if entry_value + entry_fee <= cash:
                            break
                        quantity -= lot_size
                    if quantity >= lot_size:
                        entry_value = entry_price * quantity
                        entry_fee = calculate_trade_fee(entry_value, "BUY")["total"]
                        cash -= entry_value + entry_fee
                        total_fees += entry_fee
                        trade_risk = entry_price - stop
                        position = {
                            "signal_date": pd.Timestamp(frame.index[index_position - 1]),
                            "entry_date": date,
                            "entry_position": index_position,
                            "entry_price": entry_price,
                            "entry_fee": entry_fee,
                            "original_qty": quantity,
                            "qty": quantity,
                            "stop": stop,
                            "target_1": entry_price + profile.first_target_r * trade_risk,
                            "target_2": entry_price + profile.second_target_r * trade_risk,
                            "highest": entry_price,
                            "partial_taken": False,
                            "exit_value": 0.0,
                            "exit_fees": 0.0,
                            "bars_held": 0,
                        }

        market_value = 0.0 if position is None else position["qty"] * float(bar["close"])
        equity_values.append(cash + market_value)
        equity_dates.append(date)

    equity = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates), name="equity")
    open_position = None
    if position is not None:
        open_position = {
            "entry_date": position["entry_date"],
            "entry_price": position["entry_price"],
            "quantity": position["qty"],
            "stop_loss": position["stop"],
            "target_1": position["target_1"],
            "target_2": position["target_2"],
        }
    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        metrics=_metrics(equity, trades, initial_capital, total_fees),
        assumptions={
            "signal_time": "日 K 收盘后",
            "execution": "下一交易日开盘，计入 5bp 默认滑点",
            "settlement": "股票 T+1；买入当日不可卖出",
            "limits": "一字涨停不买、一字跌停不卖；不覆盖 ST/新股特殊限制",
            "costs": "佣金最低 5 元、过户费、卖出印花税",
            "same_bar": "止损与止盈同日触发时按止损优先",
        },
        open_position=open_position,
    )


def evaluate_strategy(
    history: pd.DataFrame,
    *,
    ticker: str,
    mode: StrategyMode,
    initial_capital: float = 100_000.0,
    risk_pct: float = 0.01,
    max_position_pct: float = 0.30,
) -> tuple[pd.DataFrame, BacktestResult, BacktestResult]:
    """返回特征、全样本回测与末段时间外回测。"""
    features = prepare_features(history, mode)
    warmup = 80
    split_position = max(warmup, int(len(features) * 0.65))
    full = run_backtest(
        features,
        ticker=ticker,
        mode=mode,
        initial_capital=initial_capital,
        risk_pct=risk_pct,
        max_position_pct=max_position_pct,
        start_position=warmup,
    )
    out_of_sample = run_backtest(
        features,
        ticker=ticker,
        mode=mode,
        initial_capital=initial_capital,
        risk_pct=risk_pct,
        max_position_pct=max_position_pct,
        start_position=split_position,
    )

    available = len(features) - warmup
    fold_size = max(40, available // 3)
    folds = []
    for fold_number in range(3):
        start = warmup + fold_number * fold_size
        end = len(features) if fold_number == 2 else min(len(features), start + fold_size)
        if end - start < 20:
            continue
        fold = run_backtest(
            features,
            ticker=ticker,
            mode=mode,
            initial_capital=initial_capital,
            risk_pct=risk_pct,
            max_position_pct=max_position_pct,
            start_position=start,
            end_position=end,
        )
        folds.append(
            {
                "阶段": fold_number + 1,
                "开始": features.index[start].date(),
                "结束": features.index[end - 1].date(),
                "交易数": fold.metrics["total_trades"],
                "胜率": fold.metrics["win_rate"],
                "收益率": fold.metrics["total_return"],
                "最大回撤": fold.metrics["max_drawdown"],
            }
        )
    out_of_sample.fold_metrics = folds
    return features, full, out_of_sample
