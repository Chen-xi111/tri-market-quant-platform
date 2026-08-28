#!/usr/bin/env python3
"""
数据源诊断脚本 — 在终端运行，测试哪个数据源能拿到行情
用法: python3 diagnose.py 02172
"""
import sys
sys.path.insert(0, '.')
import config as cfg

ticker_raw = sys.argv[1] if len(sys.argv) > 1 else "02172"
yahoo_code, futu_code, raw_code = cfg.normalize_ticker(ticker_raw)
name = cfg.PRESET_STOCKS.get(yahoo_code, "自定义")
print(f"标的: {name} ({raw_code})")
print(f"  Yahoo: {yahoo_code}  Futu: {futu_code}")
print()

# ============================================================
# 1. 东方财富 — 直接用 urllib 测试
# ============================================================
print("=" * 50)
print("1. 东方财富 API")
print("=" * 50)
import json, urllib.request

secid = f"116.{raw_code}"
print(f"   secid: {secid}")

# 测试实时行情
url_rt = f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f43,f44,f45,f46,f47,f58,f60"
try:
    req = urllib.request.Request(url_rt, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        d = data.get("data", {}) or {}
        if d:
            price = d.get("f43")
            name_em = d.get("f58")
            print(f"   ✅ 实时行情: {name_em} 价格={price/1000 if price else 'N/A'}")
        else:
            print(f"   ⚠️ API 返回空 data: {data}")
except Exception as e:
    print(f"   ❌ 实时行情失败: {e}")

# 测试历史K线
url_hist = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57&klt=101&fqt=1&end=20260807&lmt=10"
try:
    req = urllib.request.Request(url_hist, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://quote.eastmoney.com/",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        klines = (data.get("data", {}) or {}).get("klines", [])
        if klines:
            print(f"   ✅ 历史K线: {len(klines)} 条")
            print(f"   最新: {klines[-1]}")
        else:
            print(f"   ⚠️ 历史K线为空: {data}")
except Exception as e:
    print(f"   ❌ 历史K线失败: {e}")

# ============================================================
# 2. Yahoo Finance
# ============================================================
print()
print("=" * 50)
print("2. Yahoo Finance (yfinance)")
print("=" * 50)
try:
    import yfinance as yf
    stock = yf.Ticker(yahoo_code)
    hist = stock.history(period="10d")
    if not hist.empty:
        print(f"   ✅ 历史数据: {len(hist)} 条")
        print(f"   最新收盘: {hist['Close'].iloc[-1]:.3f}")
    else:
        print(f"   ⚠️ 无历史数据")
    info = stock.info or {}
    if info:
        print(f"   名称: {info.get('longName', 'N/A')}")
        print(f"   现价: {info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# ============================================================
# 3. AKShare
# ============================================================
print()
print("=" * 50)
print("3. AKShare")
print("=" * 50)
try:
    import akshare as ak
    df = ak.stock_hk_spot_em()
    row = df[df["代码"] == raw_code]
    if not row.empty:
        r = row.iloc[0]
        print(f"   ✅ 实时行情: {r.get('名称', 'N/A')}")
        print(f"   现价: {r.get('最新价', 'N/A')}")
        print(f"   涨跌幅: {r.get('涨跌幅', 'N/A')}%")
    else:
        print(f"   ⚠️ 未找到代码 {raw_code}")

    # 历史数据
    hist = ak.stock_hk_hist(symbol=raw_code, period="daily",
                            start_date="20260101", end_date="20260807", adjust="qfq")
    if not hist.empty:
        print(f"   ✅ 历史数据: {len(hist)} 条")
        print(f"   最新: {hist.iloc[-1].to_dict()}")
    else:
        print(f"   ⚠️ 历史数据为空")
except Exception as e:
    print(f"   ❌ 失败: {e}")

print()
print("=" * 50)
print("诊断完成。如果以上都失败，可能是网络问题。")
print("如果某个源成功，在面板侧边栏选择对应的数据源即可。")

