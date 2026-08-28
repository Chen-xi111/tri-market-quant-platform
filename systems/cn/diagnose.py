#!/usr/bin/env python3
"""A 股行情数据源诊断。

用法：``../../.venv/bin/python diagnose.py 000831``
"""

from __future__ import annotations

import sys

import config as cfg
from data_sources import SOURCES


def main() -> int:
    value = sys.argv[1] if len(sys.argv) > 1 else cfg.TICKER
    try:
        yahoo_code, broker_code, raw_code = cfg.normalize_ticker(value)
    except ValueError as exc:
        print(f"代码无效：{exc}")
        return 2

    name = cfg.PRESET_STOCKS.get(yahoo_code, "自定义标的")
    print("=" * 60)
    print(f"A 股行情诊断：{name} ({raw_code})")
    print(f"Yahoo 代码：{yahoo_code} | 市场代码：{broker_code}")
    print("=" * 60)

    successes = 0
    for key, (label, source_class, description) in SOURCES.items():
        print(f"\n{label} — {description}")
        source = source_class(yahoo_code)
        try:
            quote = source.get_realtime() or {}
            if quote.get("price") is not None and not quote.get("error"):
                print(
                    f"  ✅ 报价：¥{float(quote['price']):.2f} "
                    f"({float(quote.get('change_pct') or 0):+.2f}%)"
                )
                successes += 1
            else:
                print(f"  ⚠️ 报价不可用：{quote.get('error', '无有效价格')}")
        except Exception as exc:
            print(f"  ❌ 报价失败：{type(exc).__name__}: {exc}")

        try:
            history = source.get_history(period="1d", days=30)
            if history is not None and not history.empty:
                print(
                    f"  ✅ 日 K：{len(history)} 条，"
                    f"最新 {history.index[-1].strftime('%Y-%m-%d')} "
                    f"收盘 ¥{float(history['close'].iloc[-1]):.2f}"
                )
                successes += 1
            else:
                print("  ⚠️ 日 K 不可用")
        except Exception as exc:
            print(f"  ❌ 日 K 失败：{type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    if successes:
        print(f"诊断完成：{successes} 项请求成功。")
        return 0
    print("诊断完成：所有数据源均失败，请检查网络、DNS 或数据源限流。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
