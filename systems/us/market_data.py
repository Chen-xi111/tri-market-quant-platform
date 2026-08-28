"""面板行情加载编排。

这里只负责数据源选择、回退和历史行情兜底；Streamlit 缓存放在
``dashboard.py`` 中，以便这些规则可以独立测试。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

import config as cfg
from data_sources import SOURCES
from indicators import compute_all


FAST_FALLBACK_SOURCES = ("eastmoney", "yfinance")
CACHE_DIR = Path(__file__).resolve().parent / ".market_cache"
HISTORY_CACHE_MAX_AGE_SECONDS = 14 * 24 * 60 * 60
QUOTE_CACHE_MAX_AGE_SECONDS = 6 * 60 * 60


def _cache_stem(ticker: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in ticker.upper())


def _save_history_cache(ticker: str, frame: pd.DataFrame, source_key: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stem = _cache_stem(ticker)
        frame.to_pickle(CACHE_DIR / f"{stem}_history.pkl")
        (CACHE_DIR / f"{stem}_history.json").write_text(
            json.dumps({
                "source": source_key,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }),
            encoding="utf-8",
        )
    except Exception:
        # 本地兜底写入失败不能影响在线行情主流程。
        return


def _load_history_cache(ticker: str):
    try:
        stem = _cache_stem(ticker)
        data_path = CACHE_DIR / f"{stem}_history.pkl"
        meta_path = CACHE_DIR / f"{stem}_history.json"
        age = datetime.now(timezone.utc).timestamp() - data_path.stat().st_mtime
        if age > HISTORY_CACHE_MAX_AGE_SECONDS:
            return None
        frame = pd.read_pickle(data_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if frame is None or frame.empty:
            return None
        frame.attrs["market_data_cached"] = True
        frame.attrs["market_data_cache_age_seconds"] = max(0, age)
        return frame, meta.get("source", "yfinance")
    except Exception:
        return None


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _save_quote_cache(ticker: str, quote: dict, source_key: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_key": source_key,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "quote": quote,
        }
        (CACHE_DIR / f"{_cache_stem(ticker)}_quote.json").write_text(
            json.dumps(payload, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
    except Exception:
        return


def _load_quote_cache(ticker: str):
    try:
        path = CACHE_DIR / f"{_cache_stem(ticker)}_quote.json"
        age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        if age > QUOTE_CACHE_MAX_AGE_SECONDS:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        quote = payload.get("quote") or {}
        if quote.get("price") is None:
            return None
        quote["_cached"] = True
        quote["_cache_age_seconds"] = max(0, age)
        quote["source"] = f"{quote.get('source', '行情')}（本地缓存）"
        return quote, payload.get("source_key", "eastmoney")
    except Exception:
        return None


def source_order(source_key: str, sources: Mapping | None = None) -> list[str]:
    """返回行情源尝试顺序。

    美股版默认使用 Yahoo；该函数仍保留可插拔注册表能力，方便测试和
    后续加入经授权的券商行情源。
    """
    registry = sources or SOURCES
    if source_key not in registry:
        raise ValueError(f"未知数据源: {source_key}")

    ordered = [source_key]
    for fallback in FAST_FALLBACK_SOURCES:
        if fallback in registry and fallback not in ordered:
            ordered.append(fallback)
    return ordered


def history_source_order(source_key: str, sources: Mapping | None = None) -> list[str]:
    """历史日 K 固定优先 Yahoo，快照源不阻塞历史加载。"""
    registry = sources or SOURCES
    ordered = source_order(source_key, registry)
    if source_key in {"eastmoney", "futu"} and "yfinance" in ordered:
        ordered.remove("yfinance")
        ordered.insert(0, "yfinance")
    return ordered


def fetch_history(
    source_key: str,
    ticker: str,
    days: int,
    sources: Mapping | None = None,
) -> tuple[pd.DataFrame | None, str | None, tuple[str, ...]]:
    """获取历史行情，返回 ``(数据, 实际数据源, 失败摘要)``。"""
    registry = sources or SOURCES
    use_local_cache = sources is None
    failures: list[str] = []

    for candidate in history_source_order(source_key, registry):
        source_label, source_class, _ = registry[candidate]
        try:
            frame = source_class(ticker).get_history(period="1d", days=days)
            if frame is None or frame.empty:
                failures.append(f"{source_label}: 无历史数据")
                continue

            frame = compute_all(frame)
            if use_local_cache:
                _save_history_cache(ticker, frame, candidate)
            return frame, candidate, tuple(failures)
        except Exception as exc:  # 单个外部数据源失败时继续回退
            failures.append(f"{source_label}: {type(exc).__name__}: {exc}")

    if use_local_cache:
        cached = _load_history_cache(ticker)
        if cached:
            frame, cached_source = cached
            failures.append("在线历史行情不可用，已使用最近成功的本地缓存")
            return frame, cached_source, tuple(failures)

    return None, None, tuple(failures)


def fetch_quote(
    source_key: str,
    ticker: str,
    sources: Mapping | None = None,
    allow_slow_fallback: bool = True,
) -> tuple[dict | None, str | None, tuple[str, ...]]:
    """独立获取实时快照，不因某个源的历史 K 线失败而丢掉实时价。"""
    registry = sources or SOURCES
    use_local_cache = sources is None
    if source_key not in registry:
        return None, None, (f"未知数据源: {source_key}",)

    failures: list[str] = []
    candidates = source_order(source_key, registry)
    if not allow_slow_fallback:
        candidates = [source_key]

    for candidate in candidates:
        source_label, source_class, _ = registry[candidate]
        try:
            quote = source_class(ticker).get_realtime()
            if (
                isinstance(quote, dict)
                and not quote.get("error")
                and quote.get("price") is not None
            ):
                quote.setdefault("_received_at", datetime.now(timezone.utc).isoformat())
                if use_local_cache:
                    _save_quote_cache(ticker, quote, candidate)
                return quote, candidate, tuple(failures)
            detail = quote.get("error", "无实时价") if isinstance(quote, dict) else "返回格式无效"
            failures.append(f"{source_label}: {detail}")
        except Exception as exc:
            failures.append(f"{source_label}: {type(exc).__name__}: {exc}")

    if use_local_cache:
        cached = _load_quote_cache(ticker)
        if cached:
            quote, cached_source = cached
            failures.append("在线实时报价不可用，已使用最近成功的本地缓存")
            return quote, cached_source, tuple(failures)

    return None, None, tuple(failures)


def quote_from_history(
    frame: pd.DataFrame,
    ticker: str,
    source_label: str,
) -> dict:
    """实时接口不可用时，从最后一根日 K 生成可用快照。"""
    last = frame.iloc[-1]
    previous = frame.iloc[-2]["close"] if len(frame) > 1 else last["close"]
    price = float(last["close"])
    previous = float(previous)

    return {
        "ticker": ticker,
        "name": cfg.PRESET_STOCKS.get(ticker, ticker),
        "price": price,
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "prev_close": previous,
        "volume": float(last.get("volume", 0) or 0),
        "change": round(price - previous, 3),
        "change_pct": round((price / previous - 1) * 100, 2) if previous else 0.0,
        "source": f"{source_label}(历史回退)",
        "_market_time": (
            last.name.isoformat() if hasattr(last.name, "isoformat") else str(last.name)
        ),
        "_received_at": datetime.now(timezone.utc).isoformat(),
        "_market_data_type": "historical_fallback",
    }
