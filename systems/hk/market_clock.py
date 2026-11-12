"""港股交易阶段判断（不包含交易所节假日日历）。"""

from __future__ import annotations

from datetime import datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")


class HKMarketPhase(StrEnum):
    CLOSED = "closed"
    PRE_OPEN = "pre_open"
    MORNING = "morning"
    LUNCH = "lunch"
    AFTERNOON = "afternoon"
    CLOSING_AUCTION = "closing_auction"


def hk_market_phase(moment: datetime | None = None) -> HKMarketPhase:
    """返回港股预开市、连续交易、午休、收市竞价或休市阶段。"""
    current = moment or datetime.now(HONG_KONG_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=HONG_KONG_TZ)
    else:
        current = current.astimezone(HONG_KONG_TZ)
    if current.weekday() >= 5:
        return HKMarketPhase.CLOSED

    now = current.time()
    if time(9, 0) <= now < time(9, 30):
        return HKMarketPhase.PRE_OPEN
    if time(9, 30) <= now < time(12, 0):
        return HKMarketPhase.MORNING
    if time(12, 0) <= now < time(13, 0):
        return HKMarketPhase.LUNCH
    if time(13, 0) <= now < time(16, 0):
        return HKMarketPhase.AFTERNOON
    if time(16, 0) <= now <= time(16, 10):
        return HKMarketPhase.CLOSING_AUCTION
    return HKMarketPhase.CLOSED


def is_hk_quote_session(moment: datetime | None = None) -> bool:
    """预开市至收市竞价期间尝试实时报价，午休也保留静态报价。"""
    return hk_market_phase(moment) is not HKMarketPhase.CLOSED


def is_hk_continuous_session(moment: datetime | None = None) -> bool:
    """仅在上午和下午连续交易时段允许决策卡给出可执行数量。"""
    return hk_market_phase(moment) in (
        HKMarketPhase.MORNING,
        HKMarketPhase.AFTERNOON,
    )
