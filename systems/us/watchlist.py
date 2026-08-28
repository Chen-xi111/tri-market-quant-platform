"""用户自选股票的轻量持久化存储。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable


WATCHLIST_PATH = Path(__file__).resolve().parent / ".user_stocks.json"


def _normalized_code(value: object, normalize_ticker: Callable[[str], tuple]) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return str(normalize_ticker(value)[0])
    except (TypeError, ValueError):
        return None


def load_user_stocks(
    normalize_ticker: Callable[[str], tuple],
    path: Path = WATCHLIST_PATH,
) -> list[str]:
    """读取自选代码；无效或损坏的记录会被安全忽略。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return []

    values = payload.get("tickers", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        return []

    saved: list[str] = []
    seen: set[str] = set()
    for value in values:
        code = _normalized_code(value, normalize_ticker)
        if code and code not in seen:
            seen.add(code)
            saved.append(code)
    return saved


def save_user_stocks(
    values: Iterable[str],
    normalize_ticker: Callable[[str], tuple],
    path: Path = WATCHLIST_PATH,
) -> list[str]:
    """标准化、去重并原子写入自选代码。"""
    saved: list[str] = []
    seen: set[str] = set()
    for value in values:
        code = _normalized_code(value, normalize_ticker)
        if code and code not in seen:
            seen.add(code)
            saved.append(code)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump({"version": 1, "tickers": saved}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
    return saved


def remember_user_stock(
    value: str,
    preset_codes: Iterable[str],
    normalize_ticker: Callable[[str], tuple],
    path: Path = WATCHLIST_PATH,
) -> tuple[str, bool]:
    """保存一个非预设代码，返回（标准代码，是否首次新增）。"""
    code = str(normalize_ticker(value)[0])
    presets = {
        normalized
        for item in preset_codes
        if (normalized := _normalized_code(item, normalize_ticker)) is not None
    }
    if code in presets:
        return code, False

    current = load_user_stocks(normalize_ticker, path)
    if code in current:
        return code, False
    save_user_stocks([*current, code], normalize_ticker, path)
    return code, True

