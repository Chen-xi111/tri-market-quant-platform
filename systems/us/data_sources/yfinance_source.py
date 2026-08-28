"""Yahoo Finance 数据源 — 美股历史行情与报价快照。

``yfinance`` 偶发因 cookie/crumb 或限流返回空表。历史数据链在这种
情况下会自动转到 Yahoo 底层 Chart JSON 接口，不让单个客户端
实现故障阻断策略计算。
"""

import json
import urllib.parse
import urllib.request

import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from .base import DataSource
import config as cfg


class YFinanceSource(DataSource):
    name = "yfinance"
    _REQUEST_TIMEOUT_SECONDS = 8

    def __init__(self, ticker: str = None):
        super().__init__(ticker or cfg.TICKER)
        self.ticker = cfg.normalize_ticker(self.ticker)[0]
        # Yahoo 的类别股代码使用 BRK-B，页面同时允许用户输入 BRK.B。
        self._yahoo_symbol = self.ticker.replace(".", "-")
        self._stock = yf.Ticker(self._yahoo_symbol)

    def get_realtime(self) -> dict:
        try:
            # ``Ticker.info`` 需要拉取大量基本面字段，在切股时经常成为
            # 数秒级阻塞点。fast_info 只读行情所需的小字段集。
            fast = self._stock.fast_info
            price = fast.get("lastPrice")
            prev = fast.get("previousClose") or fast.get("regularMarketPreviousClose")

            # 如果实时接口拿不到 price，用历史数据兜底
            if price is None:
                hist = self.get_history(period="1d", days=10)
                if not hist.empty:
                    last = hist.iloc[-1]
                    prev = hist.iloc[-2]["close"] if len(hist) > 1 else last["close"]
                    price = last["close"]
                    return {
                        "ticker": self.ticker,
                        "name": cfg.PRESET_STOCKS.get(self.ticker, self.ticker),
                        "price": price,
                        "open": last["open"],
                        "high": last["high"],
                        "low": last["low"],
                        "prev_close": prev,
                        "volume": last["volume"],
                        "change": round(price - prev, 3),
                        "change_pct": round((price / prev - 1) * 100, 2) if prev else 0.0,
                        "currency": fast.get("currency") or cfg.CURRENCY,
                        "source": "Yahoo Finance(历史回退)",
                        "_market_data_type": "historical_fallback",
                    }
                return {}

            change = price - prev if prev and price else None
            change_pct = round((price / prev - 1) * 100, 2) if prev and price else None

            return {
                "ticker": self.ticker,
                "name": cfg.PRESET_STOCKS.get(self.ticker, self.ticker),
                "price": price,
                "open": fast.get("open"),
                "high": fast.get("dayHigh"),
                "low": fast.get("dayLow"),
                "prev_close": prev,
                "volume": fast.get("lastVolume"),
                "avg_volume": fast.get("threeMonthAverageVolume"),
                "market_cap": fast.get("marketCap"),
                "change": round(change, 3) if change else None,
                "change_pct": change_pct,
                "currency": fast.get("currency") or cfg.CURRENCY,
                "source": "Yahoo Finance",
                "_market_data_type": "best_effort",
            }
        except Exception:
            return self._quote_from_chart()

    def _yfinance_history(self, period: str, days: int) -> pd.DataFrame:
        """单独封装便于回归测试，保持 yfinance 为首选通道。"""
        intraday_map = {
            "1m": "7d", "5m": "60d", "15m": "60d",
            "30m": "60d", "60m": "730d", "1h": "730d",
        }
        if period in intraday_map:
            return self._stock.history(
                period=intraday_map[period],
                interval=period,
                timeout=self._REQUEST_TIMEOUT_SECONDS,
            )

        start = datetime.now() - pd.Timedelta(days=days)
        return self._stock.history(
            start=start,
            interval=period,
            timeout=self._REQUEST_TIMEOUT_SECONDS,
        )

    def _fetch_chart(self, params: dict) -> dict:
        encoded_symbol = urllib.parse.quote(self._yahoo_symbol, safe="-.")
        query = urllib.parse.urlencode(params)
        last_error = None
        for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
            request = urllib.request.Request(
                f"https://{host}/v8/finance/chart/{encoded_symbol}?{query}",
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self._REQUEST_TIMEOUT_SECONDS
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                chart = payload.get("chart") or {}
                if chart.get("error"):
                    raise RuntimeError(str(chart["error"]))
                if chart.get("result"):
                    return payload
            except Exception as exc:
                last_error = exc
        raise RuntimeError(str(last_error or "Yahoo Chart 未返回数据"))

    @staticmethod
    def _chart_to_frame(payload: dict) -> pd.DataFrame:
        result = ((payload.get("chart") or {}).get("result") or [None])[0] or {}
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quotes = indicators.get("quote") or []
        if not timestamps or not quotes:
            return pd.DataFrame()

        quote = quotes[0] or {}
        frame = pd.DataFrame(
            {
                "open": quote.get("open") or [],
                "high": quote.get("high") or [],
                "low": quote.get("low") or [],
                "close": quote.get("close") or [],
                "volume": quote.get("volume") or [],
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True),
        )

        # 与 yfinance 默认 auto_adjust 的口径保持一致：除权因子同步
        # 作用于 OHLC，避免拆股日被策略误判为暴跌。
        adjusted_sets = indicators.get("adjclose") or []
        if adjusted_sets:
            adjusted = pd.Series(
                (adjusted_sets[0] or {}).get("adjclose") or [],
                index=frame.index,
                dtype=float,
            )
            raw_close = pd.to_numeric(frame["close"], errors="coerce")
            ratio = adjusted / raw_close.replace(0, pd.NA)
            for column in ("open", "high", "low"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce") * ratio
            frame["close"] = adjusted

        timezone_name = (result.get("meta") or {}).get("exchangeTimezoneName")
        if timezone_name:
            try:
                frame.index = frame.index.tz_convert(timezone_name)
            except (TypeError, ValueError):
                pass
        frame.index = frame.index.tz_localize(None)
        frame.index.name = "date"
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        return frame[~frame.index.duplicated(keep="last")].sort_index()

    def _chart_history(self, period: str, days: int) -> pd.DataFrame:
        interval = "60m" if period == "1h" else period
        now = datetime.now(timezone.utc)
        payload = self._fetch_chart({
            "period1": int((now - pd.Timedelta(days=days + 7)).timestamp()),
            "period2": int((now + pd.Timedelta(days=1)).timestamp()),
            "interval": interval,
            "events": "div,splits",
            "includeAdjustedClose": "true",
        })
        frame = self._chart_to_frame(payload)
        frame.attrs["market_data_transport"] = "yahoo_chart_api"
        return frame

    def _quote_from_chart(self) -> dict:
        try:
            payload = self._fetch_chart({
                "range": "5d", "interval": "1d", "includePrePost": "true",
            })
            result = ((payload.get("chart") or {}).get("result") or [None])[0] or {}
            meta = result.get("meta") or {}
            frame = self._chart_to_frame(payload)
            if frame.empty:
                return {}
            last = frame.iloc[-1]
            previous = meta.get("chartPreviousClose")
            if previous is None and len(frame) > 1:
                previous = frame["close"].iloc[-2]
            price = meta.get("regularMarketPrice") or last["close"]
            previous = float(previous or price)
            price = float(price)
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
                "currency": meta.get("currency") or cfg.CURRENCY,
                "source": "Yahoo Finance Chart",
                "_market_time": datetime.fromtimestamp(
                    int(meta.get("regularMarketTime") or 0), timezone.utc
                ).isoformat() if meta.get("regularMarketTime") else None,
                "_market_data_type": "best_effort",
            }
        except Exception:
            return {}

    def get_history(self, period: str = "1d", days: int = 365) -> pd.DataFrame:
        try:
            df = self._yfinance_history(period, days)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            try:
                return self._chart_history(period, days)
            except Exception:
                return pd.DataFrame()

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df.dropna(how="all", inplace=True)
        return df

    def get_info(self) -> dict:
        try:
            return self._stock.info or {}
        except Exception:
            return {}
