"""港股工作区内的交易决策门槛实现。"""

from __future__ import annotations

from math import floor, isfinite


def quote_requires_execution_block(quote: dict | None, source_label: str = "") -> bool:
    """缓存或由历史 K 线合成的报价只能用于查看，不能驱动新仓。"""
    if not quote:
        return True
    source = f"{source_label} {quote.get('source') or ''}"
    return bool(quote.get("_cached")) or "历史回退" in source


def _finite(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _round_down_lot(shares: float, lot_size: int) -> int:
    step = max(1, int(lot_size))
    return max(0, floor(max(0.0, float(shares)) / step) * step)


def risk_reward_metrics(current_price: float, stop_loss: float, target: float) -> dict:
    price, stop, target_value = _finite(current_price), _finite(stop_loss), _finite(target)
    risk, reward = price - stop, target_value - price
    return {
        "risk_per_share_now": max(0.0, risk),
        "reward_per_share_now": max(0.0, reward),
        "stop_distance_pct": risk / price if price > 0 and risk > 0 else 0.0,
        "reward_risk_now": reward / risk if risk > 0 and reward > 0 else 0.0,
    }


def assess_trade_decision(
    plan: dict, current_price: float, *, volume_ratio: float = 1.0,
    session_open: bool = False, data_is_cached: bool = False,
    median_turnover: float | None = None, min_median_turnover: float = 5_000_000.0,
    min_reward_risk: float = 1.30, max_stop_distance_pct: float = 0.12,
) -> dict:
    price = _finite(current_price)
    entry_low, entry_high = _finite(plan.get("entry_low")), _finite(plan.get("entry_high"))
    trigger = _finite(plan.get("breakout_trigger"), entry_high)
    atr = max(_finite(plan.get("atr"), price * 0.02), 0.001)
    max_chase = _finite(plan.get("max_chase"), max(entry_high, trigger) + 0.5 * atr)
    stop, target_1 = _finite(plan.get("stop_loss")), _finite(plan.get("target_1"))
    volume = max(0.0, _finite(volume_ratio, 1.0))
    turnover = None if median_turnover is None else max(0.0, _finite(median_turnover))
    liquidity_ok = turnover is None or turnover >= max(0.0, _finite(min_median_turnover))
    metrics = risk_reward_metrics(price, stop, target_1)
    if price <= 0:
        price_location = "价格不可用"
    elif price <= stop:
        price_location = "止损位下方"
    elif price < entry_low:
        price_location = "入场区下方，等待企稳"
    elif price <= entry_high:
        price_location = "回调入场区"
    elif price < trigger:
        price_location = "入场区上方，等待确认"
    elif price <= max_chase:
        price_location = "突破确认区"
    else:
        price_location = "超过追价上限"
    base_checks = dict(plan.get("checks") or {})
    failed_plan_checks = [label for label, passed in base_checks.items() if not passed]
    hard_blockers = []
    if price <= 0:
        hard_blockers.append("当前价格不可用")
    if data_is_cached:
        hard_blockers.append("行情来自缓存或历史 K 线回退，不能据此新开仓")
    if not liquidity_ok:
        hard_blockers.append(
            f"20 日中位成交额仅 HKD {turnover:,.0f}，低于流动性门槛"
        )
    if price > 0 and price <= stop:
        hard_blockers.append("现价已跌破计划止损")
    if price > max_chase > 0:
        hard_blockers.append("现价超过追价上限")
    if metrics["reward_risk_now"] < min_reward_risk:
        hard_blockers.append(f"按现价到第一目标仅 {metrics['reward_risk_now']:.2f}R，低于 {min_reward_risk:.2f}R")
    if metrics["stop_distance_pct"] > max_stop_distance_pct:
        hard_blockers.append(f"按现价止损距离 {metrics['stop_distance_pct']:.1%} 过宽")
    in_pullback_zone = entry_low <= price <= entry_high
    breakout_ready = trigger <= price <= max_chase and volume >= 1.20
    price_trigger_ready = in_pullback_zone or breakout_ready
    plan_ready = bool(plan.get("actionable")) and not failed_plan_checks
    setup_ready = plan_ready and not hard_blockers and price_trigger_ready
    can_order_now = setup_ready and bool(session_open)
    plan_shares, lot_size = max(0, int(plan.get("shares") or 0)), max(1, int(plan.get("lot_size") or 1))
    risk_budget, planned_value = max(0.0, _finite(plan.get("risk_budget"))), max(0.0, _finite(plan.get("planned_value")))
    risk_now = max(metrics["risk_per_share_now"], 0.001)
    position_limit = min(plan_shares, _round_down_lot(risk_budget / risk_now, lot_size), _round_down_lot(planned_value / max(price, 0.001), lot_size))
    planned_tranche = max(0, int(plan.get("first_shares") or 0))
    if breakout_ready and int(plan.get("add_shares") or 0) > 0:
        planned_tranche = int(plan.get("add_shares") or 0)
    planned_tranche = min(position_limit, planned_tranche)
    order_shares_now = planned_tranche if can_order_now else 0
    if hard_blockers:
        status_key, headline, next_action = "blocked", "禁止新开仓", "处理阻断项；不要因综合评分高而追价或放宽止损。"
    elif failed_plan_checks or not plan_ready:
        status_key, headline, next_action = "watch", "仅观察", "等待日线信号、趋势、策略共识和数据时效全部通过。"
    elif not price_trigger_ready:
        status_key, headline = "waiting", "等待入场条件"
        next_action = "等待价格重新站回入场区并出现企稳，不接下落中的价格。" if price < entry_low else "等待回踩入场区，或放量站上突破确认价。" if price < trigger else "价格已突破但量比不足 1.20，等待成交量确认。"
    elif not session_open:
        status_key, headline, next_action = "ready_later", "计划就绪，开市复核", "下一交易时段先复核跳空、公告和盘口；若越过追价上限则取消。"
    elif in_pullback_zone:
        status_key, headline, next_action = "orderable", "小仓试单条件通过", "仅按首笔数量模拟，止损单与入场计划同时准备。"
    else:
        status_key, headline, next_action = "orderable", "突破确认条件通过", "仅按确认仓数量模拟；若跌回确认价下方则停止加仓。"
    decision_checks = {
        **base_checks, "非缓存行情": not data_is_cached,
        f"现价盈亏比≥{min_reward_risk:.2f}R": metrics["reward_risk_now"] >= min_reward_risk,
        "止损距离不过宽": 0 < metrics["stop_distance_pct"] <= max_stop_distance_pct,
        "现价未跌破止损": price > stop > 0, "现价未超过追价上限": 0 < price <= max_chase,
        "20日中位成交额达标": liquidity_ok,
        "价格触发条件": price_trigger_ready, "当前为连续交易时段": bool(session_open),
    }
    return {
        "status_key": status_key, "headline": headline, "next_action": next_action,
        "price_location": price_location, "current_price": price, "volume_ratio": volume,
        "median_turnover": turnover,
        "max_chase": max_chase, "setup_ready": setup_ready, "can_order_now": can_order_now,
        "in_pullback_zone": in_pullback_zone, "breakout_ready": breakout_ready,
        "order_shares_now": order_shares_now, "planned_tranche_shares": planned_tranche,
        "planned_value_now": planned_tranche * price, "planned_risk_now": planned_tranche * risk_now,
        "failed_plan_checks": tuple(failed_plan_checks), "hard_blockers": tuple(hard_blockers),
        "checks": decision_checks, **metrics,
    }
