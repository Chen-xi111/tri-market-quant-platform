"""富途 OpenD 美股快照数据源。

同一 Streamlit 进程内复用一条 OpenD 连接，避免每次刷新都重新建立
Socket 和 SDK 回调线程。所有快照请求串行并留有 0.55 秒间隔，
低于富途官方“30 秒 60 次”的频率上限。
"""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
import importlib.util
import socket
import sys
from threading import RLock
from time import monotonic, sleep
from typing import Any, ClassVar

import pandas as pd

import config as cfg
from market_clock import NEW_YORK_TZ

from .base import DataSource


def futu_available() -> bool:
    """Check at request time so tests and optional late installs are respected."""
    if "futu" in sys.modules:
        return True
    return importlib.util.find_spec("futu") is not None


class FutuOpenDSource(DataSource):
    name = "futu"
    host = "127.0.0.1"
    port = 11111
    _session_lock: ClassVar[RLock] = RLock()
    _quote_context: ClassVar[Any | None] = None
    _last_snapshot_at: ClassVar[float] = 0.0
    _min_snapshot_interval_seconds: ClassVar[float] = 0.55

    @classmethod
    def opend_is_listening(cls) -> bool:
        try:
            with socket.create_connection((cls.host, cls.port), timeout=0.25):
                return True
        except OSError:
            return False

    @classmethod
    def _close_context_locked(cls) -> None:
        context = cls._quote_context
        cls._quote_context = None
        if context is not None:
            try:
                context.close()
            except Exception:
                pass

    @classmethod
    def close_shared_session(cls) -> None:
        """关闭进程级富途行情连接，主要用于进程退出和测试清理。"""
        with cls._session_lock:
            cls._close_context_locked()
            cls._last_snapshot_at = 0.0

    @classmethod
    def _get_context_locked(cls):
        reused = cls._quote_context is not None
        if cls._quote_context is None:
            from futu import OpenQuoteContext

            cls._quote_context = OpenQuoteContext(host=cls.host, port=cls.port)
        return cls._quote_context, reused

    @classmethod
    def _request_snapshot(cls, codes: list[str]):
        """通过持久连接请求快照；连接异常时自动重建一次。"""
        from futu import RET_OK

        with cls._session_lock:
            remaining = (
                cls._min_snapshot_interval_seconds
                - (monotonic() - cls._last_snapshot_at)
            )
            if remaining > 0:
                sleep(remaining)

            started = monotonic()
            reused = False
            for attempt in range(2):
                context, context_reused = cls._get_context_locked()
                reused = reused or context_reused
                try:
                    ret, data = context.get_market_snapshot(codes)
                except Exception:
                    cls._close_context_locked()
                    if attempt == 0 and cls.opend_is_listening():
                        continue
                    raise

                if ret == RET_OK:
                    cls._last_snapshot_at = monotonic()
                    return ret, data, monotonic() - started, reused

                error_text = str(data).lower()
                connection_error = any(
                    marker in error_text
                    for marker in ("connect", "disconnect", "socket", "连接")
                )
                if attempt == 0 and connection_error and cls.opend_is_listening():
                    cls._close_context_locked()
                    continue

                cls._last_snapshot_at = monotonic()
                return ret, data, monotonic() - started, reused

            raise RuntimeError("富途 OpenD 行情连接重建失败")

    def get_realtime(self) -> dict:
        if not self.opend_is_listening():
            return {"error": "富途 OpenD 未启动或 11111 端口未监听"}
        if not futu_available():
            return {"error": "未安装 futu-api Python 客户端"}

        from futu import RET_OK

        symbol = cfg.normalize_ticker(self.ticker)[2]
        code = f"US.{symbol}"
        try:
            ret, data, latency_seconds, connection_reused = self._request_snapshot([code])
            if ret != RET_OK:
                return {"error": str(data)}
            if data is None or data.empty:
                return {"error": "富途 OpenD 返回空快照"}
            row = data.iloc[0]
            price = float(row.get("last_price") or 0)
            previous = float(row.get("prev_close_price") or 0)
            if price <= 0:
                return {"error": "富途 OpenD 快照没有有效最新价"}

            market_time = None
            raw_time = row.get("update_time")
            if raw_time:
                try:
                    market_time = datetime.fromisoformat(str(raw_time)).replace(
                        tzinfo=NEW_YORK_TZ
                    ).isoformat()
                except ValueError:
                    market_time = str(raw_time)
            return {
                "ticker": symbol,
                "name": str(row.get("name") or cfg.PRESET_STOCKS.get(symbol, symbol)),
                "price": price,
                "open": float(row.get("open_price") or price),
                "high": float(row.get("high_price") or price),
                "low": float(row.get("low_price") or price),
                "prev_close": previous or price,
                "volume": float(row.get("volume") or 0),
                "change": round(price - previous, 3) if previous else 0.0,
                "change_pct": round((price / previous - 1) * 100, 2) if previous else 0.0,
                "currency": cfg.CURRENCY,
                "source": "富途 OpenD",
                "_market_time": market_time,
                "_received_at": datetime.now(timezone.utc).isoformat(),
                "_market_data_type": "broker_snapshot",
                "latency_seconds": round(latency_seconds, 3),
                "_connection_reused": connection_reused,
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def get_history(self, period: str = "1d", days: int = 365) -> pd.DataFrame:
        return pd.DataFrame()


atexit.register(FutuOpenDSource.close_shared_session)
