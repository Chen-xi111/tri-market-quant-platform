"""港股现金市场的回测费用假设。"""

from __future__ import annotations

import math

import config as cfg


def calculate_trade_fee(
    gross_value: float,
    side: str,
    commission_rate: float | None = None,
    minimum_commission: float | None = None,
    stamp_duty_rate: float | None = None,
    sfc_levy_rate: float | None = None,
    afrc_levy_rate: float | None = None,
    trading_fee_rate: float | None = None,
) -> dict[str, float]:
    """估算佣金、双边印花税与港交所/监管征费。

    普通港股印花税按成交额 0.1% 逐笔向上取整至 1 港元；ETF、做市商、
    外国金融交易税及券商平台费等特殊情况不在这里自动推断。
    """
    gross = max(0.0, float(gross_value))
    commission_rate = cfg.COMMISSION_RATE if commission_rate is None else max(0.0, float(commission_rate))
    minimum_commission = cfg.MIN_COMMISSION if minimum_commission is None else max(0.0, float(minimum_commission))
    stamp_duty_rate = cfg.STAMP_DUTY_RATE if stamp_duty_rate is None else max(0.0, float(stamp_duty_rate))
    sfc_levy_rate = cfg.SFC_LEVY_RATE if sfc_levy_rate is None else max(0.0, float(sfc_levy_rate))
    afrc_levy_rate = cfg.AFRC_LEVY_RATE if afrc_levy_rate is None else max(0.0, float(afrc_levy_rate))
    trading_fee_rate = cfg.TRADING_FEE_RATE if trading_fee_rate is None else max(0.0, float(trading_fee_rate))
    if gross <= 0:
        return {
            "commission": 0.0,
            "stamp_duty": 0.0,
            "sfc_levy": 0.0,
            "afrc_levy": 0.0,
            "trading_fee": 0.0,
            "total": 0.0,
        }
    commission = max(gross * commission_rate, minimum_commission)
    stamp_duty = float(math.ceil(gross * stamp_duty_rate))
    sfc_levy = round(gross * sfc_levy_rate, 2)
    afrc_levy = round(gross * afrc_levy_rate, 2)
    trading_fee = round(gross * trading_fee_rate, 2)
    total = commission + stamp_duty + sfc_levy + afrc_levy + trading_fee
    return {
        "commission": commission,
        "stamp_duty": stamp_duty,
        "sfc_levy": sfc_levy,
        "afrc_levy": afrc_levy,
        "trading_fee": trading_fee,
        "total": total,
    }
