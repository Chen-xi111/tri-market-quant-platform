#!/usr/bin/env python3
"""美股数据与 IBKR Paper 连接诊断。用法：python3 diagnose.py AAPL"""

from __future__ import annotations

import importlib.util
import socket
import sys

import config as cfg


ticker_raw = sys.argv[1] if len(sys.argv) > 1 else cfg.TICKER
yahoo_code, broker_code, display_code = cfg.normalize_ticker(ticker_raw)
name = cfg.PRESET_STOCKS.get(yahoo_code, "自定义")

print(f"标的: {name} ({display_code})")
print(f"Yahoo / IBKR / Futu: {yahoo_code} / {broker_code} / US.{display_code}")

print("\n" + "=" * 54)
print("1. Yahoo Finance")
print("=" * 54)
try:
    import yfinance as yf

    stock = yf.Ticker(yahoo_code)
    history = stock.history(period="10d", timeout=10)
    if history.empty:
        print("   ⚠️ 无历史数据")
    else:
        print(f"   ✅ 历史数据 {len(history)} 条；最新收盘 ${history['Close'].iloc[-1]:.2f}")
        try:
            last = stock.fast_info.get("lastPrice")
            print(f"   ✅ 报价快照 ${float(last):.2f}" if last else "   ⚠️ 无报价快照")
        except Exception as exc:
            print(f"   ⚠️ 报价快照不可用: {exc}")
except Exception as exc:
    print(f"   ❌ Yahoo 失败: {type(exc).__name__}: {exc}")

print("\n" + "=" * 54)
print("2. 东方财富美股快照")
print("=" * 54)
try:
    from data_sources.eastmoney_source import EastMoneyUSSource

    quote = EastMoneyUSSource(yahoo_code).get_realtime()
    if quote.get("price"):
        print(f"   ✅ 快照 ${quote['price']:.2f}")
        print(f"   ✅ 源端时点 {quote.get('_market_time', '未知')}")
        print(f"   ✅ 市场映射 {quote.get('_secid', '未知')}")
    else:
        print(f"   ❌ 快照失败: {quote.get('error', '未知原因')}")
except Exception as exc:
    print(f"   ❌ 东方财富失败: {type(exc).__name__}: {exc}")

print("\n" + "=" * 54)
print("3. 富途 OpenD 行情")
print("=" * 54)
if importlib.util.find_spec("futu") is None:
    print("   ❌ 未安装 futu-api Python 客户端")
else:
    print("   ✅ 已安装 futu-api")
try:
    with socket.create_connection(("127.0.0.1", 11111), timeout=0.5):
        print("   ✅ 富途 OpenD 端口 11111 正在监听")
except OSError:
    print("   ⚠️ 富途 OpenD 端口 11111 未监听；请先启动并登录 OpenD")

print("\n" + "=" * 54)
print("4. IBKR TWS API / Paper")
print("=" * 54)
if importlib.util.find_spec("ibapi") is None:
    print("   ❌ 未安装 IBKR 官方 ibapi Python 客户端")
    print("   请按 IBKR_SETUP.md 安装官方下载包中的 source/pythonclient")
else:
    print("   ✅ 已安装 ibapi")

for label, port in (("TWS Paper", 7497), ("IB Gateway Paper", 4002)):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            print(f"   ✅ {label} 端口 {port} 正在监听")
    except OSError:
        print(f"   ⚠️ {label} 端口 {port} 未监听")

print("\n诊断完成。券商行情未连接时，网站会回退东方财富/Yahoo。")
