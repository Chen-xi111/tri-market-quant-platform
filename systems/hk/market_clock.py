"""港股常规交易时段判断（不包含交易所节假日日历）。"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo


HONG_KONG_TZ = ZoneInfo("Asia/Hong_Kong")


def is_hk_quote_session(moment: datetime | None = None) -> bool:
    """工作日 09:30–16:10 视为需要实时接口，含午间静态报价。"""
    current = moment or datetime.now(HONG_KONG_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=HONG_KONG_TZ)
    else:
        current = current.astimezone(HONG_KONG_TZ)
    return current.weekday() < 5 and time(9, 30) <= current.time() <= time(16, 10)
