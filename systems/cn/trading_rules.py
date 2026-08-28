"""沪深北 A 股模拟交易规则与费用假设。"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import config as cfg


def board_name(ticker: str) -> str:
    raw = cfg.normalize_ticker(ticker)[2]
    if raw.startswith(("300", "301", "302")):
        return "创业板"
    if raw.startswith(("688", "689")):
        return "科创板"
    if raw.startswith("920"):
        return "北交所"
    return "主板"


def price_limit_pct(ticker: str) -> float:
    """返回普通存量股票的常规涨跌幅限制。

    不自动推断 ST、退市整理、新股上市初期等特殊情形，因为这些
    需要可靠的证券状态元数据。
    """
    board = board_name(ticker)
    if board == "北交所":
        return 0.30
    return 0.20 if board in ("创业板", "科创板") else 0.10


def round_price(price: float) -> float:
    return float(Decimal(str(price)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def price_limit_bounds(previous_close: float, ticker: str) -> tuple[float, float]:
    limit = price_limit_pct(ticker)
    return (
        round_price(float(previous_close) * (1 - limit)),
        round_price(float(previous_close) * (1 + limit)),
    )


def calculate_trade_fee(
    gross_value: float,
    side: str,
    commission_rate: float | None = None,
    minimum_commission: float | None = None,
    transfer_fee_rate: float | None = None,
    stamp_duty_rate: float | None = None,
) -> dict[str, float]:
    """按可配置佣金、最低佣金、过户费和卖出印花税估算费用。"""
    gross = max(0.0, float(gross_value))
    rate = cfg.COMMISSION_RATE if commission_rate is None else max(0.0, float(commission_rate))
    minimum = cfg.MIN_COMMISSION if minimum_commission is None else max(0.0, float(minimum_commission))
    transfer_rate = cfg.TRANSFER_FEE_RATE if transfer_fee_rate is None else max(0.0, float(transfer_fee_rate))
    stamp_rate = cfg.STAMP_DUTY_RATE if stamp_duty_rate is None else max(0.0, float(stamp_duty_rate))
    commission = max(gross * rate, minimum) if gross else 0.0
    transfer_fee = gross * transfer_rate
    stamp_duty = gross * stamp_rate if str(side).upper() == "SELL" else 0.0
    total = commission + transfer_fee + stamp_duty
    return {
        "commission": commission,
        "transfer_fee": transfer_fee,
        "stamp_duty": stamp_duty,
        "total": total,
    }
