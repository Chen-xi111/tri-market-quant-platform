"""东方财富美股轻量快照源。

复用 A/HK 平台已经验证过的单股 ``push2`` 请求方式。该源只承担报价；
历史日 K 仍交给 Yahoo，避免把快照源和策略历史数据耦合。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import urllib.parse
import urllib.request

import pandas as pd

import config as cfg
from .base import DataSource


class EastMoneyUSSource(DataSource):
    name = "eastmoney"
    _REQUEST_TIMEOUT_SECONDS = 2.5
    _MARKET_PREFIXES = {
        "NASDAQ": 105,
        "NYSE": 106,
        "ARCA": 107,
    }

    def __init__(self, ticker: str | None = None):
        super().__init__(ticker or cfg.TICKER)
        self.ticker = cfg.normalize_ticker(self.ticker)[0]

    def _candidate_secids(self) -> list[str]:
        primary = cfg.ibkr_primary_exchange(self.ticker)
        preferred = self._MARKET_PREFIXES.get(primary)
        # 已知主交易所时只发一个请求；自定义代码才依次探测三个市场。
        prefixes = [preferred] if preferred is not None else [105, 106, 107]
        return [f"{prefix}.{self.ticker}" for prefix in prefixes]

    def get_realtime(self) -> dict:
        fields = "f43,f44,f45,f46,f47,f48,f57,f58,f59,f60,f86,f169,f170,f171"
        failures = []
        for secid in self._candidate_secids():
            try:
                payload = self._fetch_json(
                    "/api/qt/stock/get",
                    {
                        "secid": secid, "fields": fields, "invt": 2, "fltt": 2,
                        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
                    },
                )
                data = payload.get("data") or {}
                if str(data.get("f57") or "").upper() != self.ticker:
                    failures.append(f"{secid}: 无匹配报价")
                    continue
                price = self._number(data.get("f43"))
                previous = self._number(data.get("f60"))
                if price is None or price <= 0:
                    failures.append(f"{secid}: 无有效最新价")
                    continue

                timestamp = self._number(data.get("f86"))
                market_time = (
                    datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
                    if timestamp and timestamp > 0 else None
                )
                change = self._number(data.get("f169"))
                change_pct = self._number(data.get("f170"))
                return {
                    "ticker": self.ticker,
                    "name": data.get("f58") or cfg.PRESET_STOCKS.get(self.ticker, self.ticker),
                    "price": price,
                    "open": self._number(data.get("f46")) or price,
                    "high": self._number(data.get("f44")) or price,
                    "low": self._number(data.get("f45")) or price,
                    "prev_close": previous or price,
                    "volume": self._number(data.get("f47")) or 0.0,
                    "amount": self._number(data.get("f48")) or 0.0,
                    "change": change if change is not None else price - (previous or price),
                    "change_pct": (
                        change_pct if change_pct is not None
                        else ((price / previous - 1) * 100 if previous else 0.0)
                    ),
                    "currency": cfg.CURRENCY,
                    "source": "东方财富美股快照",
                    "_market_time": market_time,
                    "_market_data_type": "public_snapshot",
                    "_secid": secid,
                }
            except Exception as exc:
                failures.append(f"{secid}: {type(exc).__name__}: {exc}")
        return {"error": " | ".join(failures) or "东方财富无美股快照"}

    def get_history(self, period: str = "1d", days: int = 365) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def _number(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number

    @classmethod
    def _fetch_json(cls, path: str, params: dict) -> dict:
        last_error = None
        # push2 偶发网关拒绝时，使用其公开延迟节点继续提供可核验时点的
        # 快照；界面会按 f86 的真实年龄标红/标橙，不伪装成实时价。
        for host in ("push2.eastmoney.com", "push2delay.eastmoney.com"):
            request_url = f"https://{host}{path}?{urllib.parse.urlencode(params)}"
            request = urllib.request.Request(
                request_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Referer": "https://quote.eastmoney.com/",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=cls._REQUEST_TIMEOUT_SECONDS
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(str(last_error or "东方财富快照网关不可用"))
