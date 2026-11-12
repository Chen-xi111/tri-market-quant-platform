"""将策略投票转换为可执行、可检查的交易计划。"""

from __future__ import annotations

from math import floor

import pandas as pd
from stock_ranking import build_execution_setup, extract_stock_features


def _round_down_lot(shares: float, lot_size: int) -> int:
    lot_size = max(1, int(lot_size))
    return max(0, floor(float(shares) / lot_size) * lot_size)


def build_trade_plan(
    df: pd.DataFrame,
    composite,
    capital: float,
    risk_pct: float,
    max_position_pct: float,
    lot_size: int = 1,
    strategy_key: str | None = None,
) -> dict:
    """
    生成入场区间、触发价、止损、两档止盈与按风险反推的股数。

    计划仅在数据新鲜、综合方向偏多且不处于下降趋势时给出买入数量。
    """
    if df is None or len(df) < 60:
        raise ValueError("至少需要 60 条历史数据才能生成交易计划")

    latest = df.iloc[-1]
    price = float(latest["close"])
    atr = float(latest.get("ATR") or price * 0.02)
    if pd.isna(atr) or atr <= 0:
        atr = price * 0.02

    ma20 = float(latest.get("MA20") or df["close"].tail(20).mean())
    ma60 = float(latest.get("MA60") or df["close"].tail(60).mean())
    ma20_previous = float(df["MA20"].iloc[-6]) if "MA20" in df and not pd.isna(df["MA20"].iloc[-6]) else ma20
    support = float(df["low"].tail(20).min())
    resistance = float(df["high"].iloc[-21:-1].max()) if len(df) > 21 else float(df["high"].max())

    if price > ma20 > ma60 and ma20 > ma20_previous:
        regime = "强上升趋势"
    elif price > ma60 and ma20 >= ma60:
        regime = "偏多趋势"
    elif price < ma20 < ma60:
        regime = "下降趋势"
    else:
        regime = "区间震荡"

    entry_center = min(price, max(ma20, price - 0.5 * atr))
    entry_low = max(0.001, entry_center - 0.25 * atr)
    entry_high = entry_center + 0.25 * atr
    trigger = max(resistance, price + 0.1 * atr)
    max_chase = entry_high + 0.5 * atr
    strategy_state = "ready"
    strategy_reason = "未指定单一策略，使用通用回调计划"
    entry_condition = "价格进入回调区后转强"

    structural_stop = support - 0.2 * atr
    atr_stop = entry_center - 2.0 * atr
    stop = max(structural_stop, atr_stop)
    stop = min(stop, entry_low - 0.2 * atr)
    stop = max(0.001, stop)

    risk_per_share = max(entry_center - stop, 0.001)
    target_1 = entry_center + 1.5 * risk_per_share
    target_2 = entry_center + 2.5 * risk_per_share

    if strategy_key:
        try:
            feature = extract_stock_features("PLAN", df, strategy_key)
            setup = build_execution_setup(feature, strategy_key)
            entry_low = float(setup["entry_low"])
            entry_high = float(setup["entry_high"])
            entry_center = float(setup["entry_center"])
            trigger = float(setup["breakout_trigger"])
            max_chase = float(setup["max_chase"])
            stop = float(setup["stop_loss"])
            target_1 = float(setup["target_1"])
            target_2 = float(setup["target_2"])
            risk_per_share = float(setup["risk_per_share"])
            strategy_state = str(setup["setup_state"])
            strategy_reason = str(setup["setup_reason"])
            entry_condition = str(setup["entry_condition"])
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            strategy_state = "watch"
            strategy_reason = f"当前策略入场检查失败：{exc}"

    index_value = df.index[-1]
    last_date = pd.Timestamp(index_value).tz_localize(None).normalize()
    today = pd.Timestamp.now().tz_localize(None).normalize()
    data_age_days = max(0, int((today - last_date).days))
    data_fresh = data_age_days <= 5
    bullish = composite.direction in ("buy", "strong_buy")
    trend_ok = regime != "下降趋势"
    confidence_ok = float(composite.confidence) >= 0.25
    strategy_ready = strategy_state == "ready"
    actionable = data_fresh and bullish and trend_ok and confidence_ok and strategy_ready

    recommended_fraction = min(
        float(max_position_pct),
        max(0.0, float(composite.recommended_position)),
    )
    risk_budget = float(capital) * float(risk_pct)
    risk_limited_shares = _round_down_lot(risk_budget / risk_per_share, lot_size)
    value_limited_shares = _round_down_lot(
        float(capital) * recommended_fraction / entry_center,
        lot_size,
    )
    shares = min(risk_limited_shares, value_limited_shares) if actionable else 0

    first_shares = _round_down_lot(shares * 0.5, lot_size)
    if shares > 0 and first_shares == 0:
        first_shares = shares
    add_shares = max(0, shares - first_shares)

    if not data_fresh:
        action = f"数据已过期 {data_age_days} 天，不下单"
    elif not bullish:
        action = "观望：综合信号未达到偏多条件"
    elif not trend_ok:
        action = "观望：下降趋势中不新开多仓"
    elif not confidence_ok:
        action = "观望：有效策略共识不足"
    elif not strategy_ready:
        action = f"等待当前策略：{strategy_reason}"
    elif shares < max(1, int(lot_size)):
        action = "观望：风险预算不足 1 股"
    else:
        action = "技术条件与多策略共识均通过，可按计划分批模拟"

    return {
        "actionable": actionable and shares > 0,
        "action": action,
        "regime": regime,
        "data_date": last_date.strftime("%Y-%m-%d"),
        "data_age_days": data_age_days,
        "price": price,
        "atr": atr,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "entry_center": entry_center,
        "breakout_trigger": trigger,
        "max_chase": max_chase,
        "support": support,
        "resistance": resistance,
        "stop_loss": stop,
        "target_1": target_1,
        "target_2": target_2,
        "risk_per_share": risk_per_share,
        "risk_budget": risk_budget,
        "recommended_fraction": recommended_fraction,
        "lot_size": max(1, int(lot_size)),
        "shares": shares,
        "first_shares": first_shares,
        "add_shares": add_shares,
        "planned_value": shares * entry_center,
        "planned_risk": shares * risk_per_share,
        "reward_risk_1": max(0.0, (target_1 - entry_center) / risk_per_share),
        "reward_risk_2": max(0.0, (target_2 - entry_center) / risk_per_share),
        "strategy_key": strategy_key,
        "strategy_setup_state": strategy_state,
        "strategy_setup_reason": strategy_reason,
        "entry_condition": entry_condition,
        "invalidation": f"收盘跌破 USD {stop:.2f} 或趋势转为下降，计划失效",
        "checks": {
            "数据新鲜": data_fresh,
            "综合信号偏多": bullish,
            "趋势过滤通过": trend_ok,
            "策略共识足够": confidence_ok,
            "当前策略到达入场条件": strategy_ready,
        },
    }


def plan_order_rows(plan: dict) -> list[dict]:
    """生成面向控制台的分批执行表。"""
    return [
        {
            "阶段": "首笔试仓",
            "条件": f"价格进入 USD {plan['entry_low']:.2f}–{plan['entry_high']:.2f}",
            "数量": plan["first_shares"],
            "价格": plan["entry_center"],
        },
        {
            "阶段": "确认加仓",
            "条件": f"{plan['entry_condition']}；确认价 USD {plan['breakout_trigger']:.2f}",
            "数量": plan["add_shares"],
            "价格": plan["breakout_trigger"],
        },
        {
            "阶段": "保护止损",
            "条件": "计划失效立即退出",
            "数量": plan["shares"],
            "价格": plan["stop_loss"],
        },
        {
            "阶段": "止盈一",
            "条件": f"首目标 {plan['reward_risk_1']:.2f}R 后减半仓",
            "数量": plan["first_shares"],
            "价格": plan["target_1"],
        },
        {
            "阶段": "止盈二",
            "条件": f"第二目标 {plan['reward_risk_2']:.2f}R 或移动止损",
            "数量": plan["add_shares"],
            "价格": plan["target_2"],
        },
    ]
