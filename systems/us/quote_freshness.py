"""行情时间戳、新鲜度判定与盘中日 K 合并。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from indicators import compute_all
from market_clock import NEW_YORK_TZ


def parse_quote_time(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=NEW_YORK_TZ)
    return parsed.astimezone(UTC)


def quote_age_seconds(quote: dict | None, now: datetime | None = None) -> float | None:
    if not quote:
        return None
    timestamp = parse_quote_time(quote.get("_market_time") or quote.get("_received_at"))
    if timestamp is None:
        return None
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return max(0.0, (current.astimezone(UTC) - timestamp).total_seconds())


def format_age(seconds: float | None) -> str:
    if seconds is None:
        return "时间未知"
    if seconds < 60:
        return f"{int(seconds)} 秒"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分 {int(seconds % 60)} 秒"
    if seconds < 86400:
        return f"{int(seconds // 3600)} 小时 {int((seconds % 3600) // 60)} 分"
    return f"{int(seconds // 86400)} 天"


def freshness_label(quote: dict | None, quote_session_open: bool) -> tuple[str, str]:
    if not quote:
        return "无报价", "red"
    if quote.get("_cached"):
        return "本地缓存", "red"
    data_type = str(quote.get("_market_data_type") or "").lower()
    if "delayed" in data_type:
        return "延迟行情", "orange"
    if data_type == "best_effort":
        return "源端延迟未知", "orange"
    if data_type == "historical_fallback":
        return "历史 K 线回退", "red" if quote_session_open else "gray"
    if not quote_session_open:
        return "休市快照", "gray"
    age = quote_age_seconds(quote)
    if age is None:
        return "更新时间待核验", "gray"
    if age <= 90:
        return "报价新鲜", "green"
    if age <= 20 * 60:
        return "报价可能延迟", "orange"
    return "报价陈旧", "red"


def execution_data_block_reason(
    quote: dict | None,
    *,
    regular_session_open: bool,
    data_is_cached: bool = False,
) -> str | None:
    """常规时段只允许明确新鲜的报价通过新开仓门槛。

    缓存数据由公共决策门槛单独报错，这里不重复生成原因。
    """
    if not regular_session_open or data_is_cached:
        return None
    label, color = freshness_label(quote, True)
    if color == "green":
        return None
    return f"常规交易时段报价状态为“{label}”，不用于新开仓"


def quote_time_text(quote: dict | None) -> str:
    if not quote:
        return "报价时点未知"
    market_time = parse_quote_time(quote.get("_market_time"))
    if market_time is not None:
        return f"行情时点 {market_time.astimezone(NEW_YORK_TZ).strftime('%Y-%m-%d %H:%M:%S')} ET"
    received = parse_quote_time(quote.get("_received_at"))
    if received is not None:
        return f"页面接收 {received.astimezone(NEW_YORK_TZ).strftime('%H:%M:%S')} ET（源端时点未知）"
    return "报价时点未知"


def merge_live_quote_into_history(frame: pd.DataFrame, quote: dict | None) -> pd.DataFrame:
    """用最新快照形成当日临时日 K，并重新计算指标。

    原始历史数据不被修改；回测仍使用已完成的历史 K 线。
    """
    if frame is None or frame.empty or not quote or not quote.get("price"):
        return frame
    if quote.get("_cached") or quote.get("_market_data_type") == "historical_fallback":
        return frame

    market_time = parse_quote_time(quote.get("_market_time") or quote.get("_received_at"))
    if market_time is None:
        return frame
    session_date = market_time.astimezone(NEW_YORK_TZ).date()
    result = frame.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)
    last_date = pd.Timestamp(result.index[-1]).date()
    values = {
        "open": float(quote.get("open") or quote["price"]),
        "high": float(quote.get("high") or quote["price"]),
        "low": float(quote.get("low") or quote["price"]),
        "close": float(quote["price"]),
        "volume": float(quote.get("volume") or 0),
    }
    if session_date < last_date:
        return frame
    if session_date == last_date:
        row_key = result.index[-1]
        for column, value in values.items():
            if column in result.columns:
                result.loc[row_key, column] = value
    else:
        row_key = pd.Timestamp(session_date)
        if result.index.tz is not None:
            row_key = row_key.tz_localize(result.index.tz)
        for column in result.columns:
            result.loc[row_key, column] = float("nan")
        for column, value in values.items():
            result.loc[row_key, column] = value
    result.sort_index(inplace=True)
    result = compute_all(result[["open", "high", "low", "close", "volume"]])
    result.attrs.update(frame.attrs)
    result.attrs["live_bar_merged"] = True
    result.attrs["live_bar_date"] = session_date.isoformat()
    return result
