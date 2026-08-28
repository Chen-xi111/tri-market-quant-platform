"""
东方财富数据源 — 免费沪深 A 股行情
API: push2.eastmoney.com / push2his.eastmoney.com
"""

import json
import urllib.request
import pandas as pd
from datetime import datetime
from .base import DataSource
import config as cfg


class EastMoneySource(DataSource):
    """
    东方财富免费 API
    - 实时行情: push2.eastmoney.com/api/qt/stock/get
    - 历史K线:  push2his.eastmoney.com/api/qt/stock/kline/get

    沪市市场代码: 1；深市市场代码: 0
    secid 示例: 1.600010 / 0.000831
    """

    name = "eastmoney"
    _REQUEST_TIMEOUT_SECONDS = 2.5

    def __init__(self, ticker: str = None):
        super().__init__(ticker or cfg.TICKER)
        yahoo_code, broker_code, raw_code = cfg.normalize_ticker(self.ticker)
        self.ticker = yahoo_code
        exchange = broker_code.split(".", 1)[0]
        market_code = 1 if exchange == "SH" else 0
        self._secid = f"{market_code}.{raw_code}"
        self._symbol = raw_code
        self._name = None

    # ================================================================
    #  实时行情
    # ================================================================
    def get_realtime(self) -> dict:
        """实时报价 — 收盘后返回最新收盘数据"""
        fields = (
            "f43,f44,f45,f46,f47,f48,f50,f51,f52,f57,f58,"
            "f60,f116,f117,f162,f167,f168,f169,f170,f171"
        )
        url = (
            f"https://push2.eastmoney.com/api/qt/stock/get"
            f"?secid={self._secid}&fields={fields}"
        )

        try:
            data = self._fetch_json(url)
            d = data.get("data", {}) or {}

            if not d:
                # API 返回空 → 尝试用历史数据兜底
                return self._fallback_from_history()

            # f43 = 最新价（盘中实时 / 盘后收盘）
            price = self._div100(d.get("f43"))
            prev = self._div100(d.get("f60"))

            # 如果实时接口拿不到 price，用 f170(收盘价) 或历史数据兜底
            if price is None:
                price = self._div100(d.get("f170")) or self._div100(d.get("f167"))
            if price is None:
                return self._fallback_from_history()

            if prev is None:
                prev = price  # 无法确定昨收就用当前价

            change = round(price - prev, 3)
            change_pct = round((price / prev - 1) * 100, 2) if prev else 0.0

            self._name = d.get("f58") or cfg.PRESET_STOCKS.get(self.ticker, self.ticker)

            return {
                "ticker": self.ticker,
                "name": self._name,
                "price": price,
                "open": self._div100(d.get("f46")),
                "high": self._div100(d.get("f44")),
                "low": self._div100(d.get("f45")),
                "prev_close": prev,
                "volume": d.get("f47"),
                "amount": d.get("f48"),
                "change": change,
                "change_pct": change_pct,
                "market_cap": d.get("f116"),
                "pe_ratio": self._div100(d.get("f162")),
                "currency": "CNY",
                "source": "东方财富",
            }
        except Exception as e:
            return self._fallback_from_history(str(e))

    # ================================================================
    #  历史K线
    # ================================================================
    def get_history(self, period: str = "1d", days: int = 365) -> pd.DataFrame:
        """获取历史K线 — 盘后也能拿到最新交易日数据"""
        klt_map = {
            "1d": 101, "1w": 102, "1M": 103,
            "60m": 60, "30m": 30, "15m": 15, "5m": 5, "1m": 1,
        }
        klt = klt_map.get(period, 101)

        end_date = datetime.now().strftime("%Y%m%d")
        # lmt 限制返回条数，多拉一些确保够用
        url = (
            f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
            f"?secid={self._secid}"
            f"&fields1=f1,f2,f3,f4,f5,f6"
            f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
            f"&klt={klt}&fqt=1&end={end_date}&lmt={days + 50}"
        )

        try:
            data = self._fetch_json(url)
            klines = (data.get("data", {}) or {}).get("klines", [])
            if not klines:
                return pd.DataFrame()

            records = []
            for line in klines:
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                records.append({
                    "date": parts[0],
                    "open": float(parts[1]),
                    "close": float(parts[2]),
                    "high": float(parts[3]),
                    "low": float(parts[4]),
                    "volume": float(parts[5]),
                    "amount": float(parts[6]),
                })

            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            return df

        except Exception:
            return pd.DataFrame()

    # ================================================================
    #  兜底：从历史数据提取最新行情
    # ================================================================
    def _fallback_from_history(self, error_msg: str = "") -> dict:
        """实时接口失败时，从历史K线提取最后一天的数据"""
        hist = self.get_history(period="1d", days=10)
        if hist.empty:
            return {"error": f"无法获取数据: {error_msg}" if error_msg else "无数据"}

        last = hist.iloc[-1]
        prev = hist.iloc[-2]["close"] if len(hist) > 1 else last["close"]
        price = last["close"]
        change = round(price - prev, 3)
        change_pct = round((price / prev - 1) * 100, 2) if prev else 0.0

        name = cfg.PRESET_STOCKS.get(self.ticker, self.ticker)

        return {
            "ticker": self.ticker,
            "name": name,
            "price": price,
            "open": last["open"],
            "high": last["high"],
            "low": last["low"],
            "prev_close": prev,
            "volume": last["volume"],
            "amount": last.get("amount", 0),
            "change": change,
            "change_pct": change_pct,
            "market_cap": None,
            "pe_ratio": None,
            "currency": "CNY",
            "source": "东方财富(历史回退)",
        }

    # ================================================================
    #  辅助
    # ================================================================
    @staticmethod
    def _div100(val) -> float:
        """东方财富 A 股价格/比率字段需除以 100。"""
        if val is None:
            return None
        try:
            v = float(val)
            if v == 0:
                return None
            return round(v / 100, 2)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _fetch_json(url: str) -> dict:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Referer": "https://quote.eastmoney.com/",
            },
        )
        with urllib.request.urlopen(
            req, timeout=EastMoneySource._REQUEST_TIMEOUT_SECONDS
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
