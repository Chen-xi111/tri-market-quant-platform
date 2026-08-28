#!/usr/bin/env python3
"""命令行快速回测"""
import sys
sys.path.insert(0, '.')

from data_fetcher import quick_history
from indicators import compute_all
from strategies import MACrossoverStrategy, BollingerReversionStrategy, VolBreakoutStrategy, MultiTFStrategy, STRATEGIES
from backtest.engine import BacktestEngine
from backtest.metrics import metrics_summary
import config as cfg

print("=" * 60)
print(f"  {cfg.TICKER_NAME} {cfg.TICKER} — A 股策略回测对比")
print("=" * 60)

df = quick_history("1d", cfg.BACKTEST_YEARS * 365)
if df.empty:
    print("数据加载失败，请检查网络")
    sys.exit(1)

df = compute_all(df)
if df.empty:
    sys.exit(1)

for key, (name, StratClass) in STRATEGIES.items():
    print(f"\n{'─'*60}")
    print(f"  📊 {name}")
    print(f"{'─'*60}")

    strat = StratClass()
    engine = BacktestEngine(strat)  # 使用默认资金 CNY 100,000
    result = engine.run(df.tail(cfg.BACKTEST_YEARS * 252))

    m = result["metrics"]
    print(f"  总收益: {m.get('total_return', 0):+.1f}%  年化: {m.get('annual_return', 0):+.1f}%")
    print(f"  最大回撤: {m.get('max_drawdown', 0):.1f}%  夏普: {m.get('sharpe_ratio', 0):.2f}")
    print(f"  胜率: {m.get('win_rate', 0):.1f}%  盈亏比: {m.get('profit_factor', 0):.2f}")
    print(f"  交易次数: {m.get('total_trades', 0)}")

print("\n" + "=" * 60)
print("  回测完成。运行 ../../run_cn.sh 启动完整面板。")
