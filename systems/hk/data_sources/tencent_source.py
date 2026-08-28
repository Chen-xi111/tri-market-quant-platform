"""腾讯证券港股行情源——低延迟报价与前复权历史日线。"""

from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

import config as cfg
from .base import DataSource


class TencentSource(DataSource):
    """提供港股日 K 和行情快照，无需密钥。"""

    name = "tencent"
    _REQUEST_TIMEOUT_SECONDS = 5.0

    def __init__(self, ticker: str | None = None):
        super().__init__(ticker or cfg.TICKER)
        yahoo_code, _, raw_code = cfg.normalize_ticker(self.ticker)
        self.ticker = yahoo_code
        self._symbol = f"hk{raw_code}"

    @staticmethod
    def _request(url: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
        )
        with urllib.request.urlopen(
            request, timeout=TencentSource._REQUEST_TIMEOUT_SECONDS
        ) as response:
            return response.read()

    @staticmethod
    def _number(fields: list[str], index: int):
        try:
            value = fields[index]
            return float(value) if value not in ("", "-") else None
        except (IndexError, TypeError, ValueError):
            return None

    @classmethod
    def parse_quote_text(cls, text: str, ticker: str) -> dict:
        try:
            fields = text.split('"', 2)[1].split("~")
        except IndexError:
            return {}
        if len(fields) < 38:
            return {}
        price = cls._number(fields, 3)
        previous = cls._number(fields, 4)
        if price is None:
            return {}
        change = cls._number(fields, 31)
        change_pct = cls._number(fields, 32)
        if change is None and previous is not None:
            change = price - previous
        if change_pct is None and previous:
            change_pct = (price / previous - 1) * 100
        return {
            "ticker": ticker,
            "name": fields[1] or ticker,
            "price": price,
            "open": cls._number(fields, 5),
            "high": cls._number(fields, 33),
            "low": cls._number(fields, 34),
            "prev_close": previous,
            "volume": cls._number(fields, 36) or cls._number(fields, 6),
            "amount": cls._number(fields, 37),
            "change": round(change, 3) if change is not None else None,
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "quote_time": fields[30] if len(fields) > 30 else "",
            "currency": "HKD",
            "source": "腾讯证券",
        }

    @staticmethod
    def _parse_history(payload: bytes, symbol: str, cycle: str) -> pd.DataFrame:
        data = json.loads(payload.decode("utf-8")).get("data", {}).get(symbol, {}) or {}
        records = data.get(f"qfq{cycle}") or data.get(cycle) or []
        rows = []
        for record in records:
            if len(record) < 6:
                continue
            try:
                rows.append(
                    {
                        "date": pd.to_datetime(record[0]),
                        "open": float(record[1]),
                        "close": float(record[2]),
                        "high": float(record[3]),
                        "low": float(record[4]),
                        "volume": float(record[5]),
                    }
                )
            except (TypeError, ValueError):
                continue
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).set_index("date").sort_index()
        quote_fields = (data.get("qt", {}) or {}).get(symbol) or []
        if len(quote_fields) > 1 and quote_fields[1]:
            frame.attrs["stock_name"] = str(quote_fields[1]).strip()
        return frame

    def get_realtime(self) -> dict:
        try:
            text = self._request(f"https://qt.gtimg.cn/q={self._symbol}").decode(
                "gb18030", errors="replace"
            )
            quote = self.parse_quote_text(text, self.ticker)
            if quote:
                return quote
        except Exception:
            pass
        history = self.get_history(days=10)
        if history.empty:
            return {}
        last = history.iloc[-1]
        previous = float(history.iloc[-2]["close"]) if len(history) > 1 else float(last["close"])
        price = float(last["close"])
        return {
            "ticker": self.ticker,
            "name": cfg.PRESET_STOCKS.get(self.ticker, self.ticker),
            "price": price,
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "prev_close": previous,
            "volume": float(last["volume"]),
            "change": round(price - previous, 3),
            "change_pct": round((price / previous - 1) * 100, 2) if previous else 0.0,
            "currency": "HKD",
            "source": "腾讯证券（历史回退）",
        }

    def get_history(self, period: str = "1d", days: int = 365) -> pd.DataFrame:
        cycle = {"1d": "day", "1w": "week", "1M": "month"}.get(period)
        if cycle is None:
            return pd.DataFrame()
        limit = max(60, min(int(days) + 50, 1000))
        urls = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={self._symbol},{cycle},,,{limit},qfq",
            "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
            f"?param={self._symbol},{cycle},,,{limit}",
            "https://ifzq.gtimg.cn/appstock/app/kline/kline"
            f"?param={self._symbol},{cycle},,,{limit}",
        )
        # 腾讯行情偶尔返回空包或连接被提前关闭；轻量重试并
        # 切换备用入口，避免将瞬时错误误判为“该股无历史”。
        for attempt in range(2):
            for url in urls:
                try:
                    frame = self._parse_history(
                        self._request(url), self._symbol, cycle
                    )
                    if not frame.empty:
                        return frame
                except Exception:
                    continue
            if attempt == 0:
                time.sleep(0.15)
        return pd.DataFrame()
