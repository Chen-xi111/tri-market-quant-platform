"""美股交易时段判断，使用纽约时区并自动处理夏令时。"""

from __future__ import annotations

from calendar import monthcalendar
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo


NEW_YORK_TZ = ZoneInfo("America/New_York")


class USMarketPhase(StrEnum):
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"


def _observed(day: date) -> date:
    if day.weekday() == 5:  # Saturday -> Friday
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday -> Monday
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    days = [week[weekday] for week in monthcalendar(year, month) if week[weekday]]
    return date(year, month, days[occurrence - 1])


def _last_weekday(year: int, month: int, weekday: int) -> date:
    days = [week[weekday] for week in monthcalendar(year, month) if week[weekday]]
    return date(year, month, days[-1])


def _easter_sunday(year: int) -> date:
    """Meeus/Jones/Butcher Gregorian Easter calculation."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def us_market_holidays(year: int) -> frozenset[date]:
    """NYSE/Nasdaq 常规整日休市规则；临时休市仍应由券商状态兜底。"""
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),       # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),       # Presidents' Day
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),         # Memorial Day
        _observed(date(year, 6, 19)),      # Juneteenth
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),       # Labor Day
        _nth_weekday(year, 11, 3, 4),      # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    # 元旦落在周六时会在上一年度 12/31 观察。
    next_new_year = _observed(date(year + 1, 1, 1))
    if next_new_year.year == year:
        holidays.add(next_new_year)
    return frozenset(holidays)


def is_us_market_early_close(day: date) -> bool:
    """常见 13:00 ET 提前收市日。"""
    thanksgiving = _nth_weekday(day.year, 11, 3, 4)
    day_after_thanksgiving = thanksgiving + timedelta(days=1)
    christmas_eve = date(day.year, 12, 24)
    independence_observed = _observed(date(day.year, 7, 4))
    before_independence = independence_observed - timedelta(days=1)
    candidates = {day_after_thanksgiving, christmas_eve, before_independence}
    return (
        day in candidates
        and day.weekday() < 5
        and day not in us_market_holidays(day.year)
    )


def us_market_phase(moment: datetime | None = None) -> USMarketPhase:
    """返回美东时间下的盘前、常规、盘后或休市状态。"""
    current = moment or datetime.now(NEW_YORK_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=NEW_YORK_TZ)
    else:
        current = current.astimezone(NEW_YORK_TZ)

    current_date = current.date()
    if current.weekday() >= 5 or current_date in us_market_holidays(current.year):
        return USMarketPhase.CLOSED

    current_time = current.time()
    regular_close = time(13, 0) if is_us_market_early_close(current_date) else time(16, 0)
    if time(4, 0) <= current_time < time(9, 30):
        return USMarketPhase.PRE_MARKET
    if time(9, 30) <= current_time < regular_close:
        return USMarketPhase.REGULAR
    if regular_close <= current_time < time(20, 0):
        return USMarketPhase.AFTER_HOURS
    return USMarketPhase.CLOSED


def is_us_quote_session(moment: datetime | None = None) -> bool:
    """盘前 04:00 至盘后 20:00 ET 期间尝试获取实时报价。"""
    return us_market_phase(moment) != USMarketPhase.CLOSED


def is_us_regular_session(moment: datetime | None = None) -> bool:
    """仅常规交易时段返回 True；自动下单应默认使用这个判断。"""
    return us_market_phase(moment) == USMarketPhase.REGULAR

