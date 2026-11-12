"""沪深 A 股交易时段判断（不包含交易所节假日日历）。"""

from __future__ import annotations

from datetime import datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


class CNMarketPhase(Enum):
    PRE_OPEN = "pre_open"
    MORNING = "morning"
    LUNCH = "lunch"
    AFTERNOON = "afternoon"
    CLOSING_AUCTION = "closing_auction"
    CLOSED = "closed"


def cn_market_phase(moment: datetime | None = None) -> CNMarketPhase:
    current = moment or datetime.now(SHANGHAI_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=SHANGHAI_TZ)
    else:
        current = current.astimezone(SHANGHAI_TZ)
    if current.weekday() >= 5:
        return CNMarketPhase.CLOSED
    now = current.time()
    if time(9, 15) <= now < time(9, 30):
        return CNMarketPhase.PRE_OPEN
    if time(9, 30) <= now <= time(11, 30):
        return CNMarketPhase.MORNING
    if time(11, 30) < now < time(13, 0):
        return CNMarketPhase.LUNCH
    if time(13, 0) <= now < time(14, 57):
        return CNMarketPhase.AFTERNOON
    if time(14, 57) <= now <= time(15, 0):
        return CNMarketPhase.CLOSING_AUCTION
    return CNMarketPhase.CLOSED


def is_cn_quote_session(moment: datetime | None = None) -> bool:
    return cn_market_phase(moment) is not CNMarketPhase.CLOSED


def is_cn_continuous_session(moment: datetime | None = None) -> bool:
    return cn_market_phase(moment) in (CNMarketPhase.MORNING, CNMarketPhase.AFTERNOON)
