"""美股量化交易平台 — 全局配置。"""

import re

# ============================================================
#  标的（默认 + 预设股票库）
# ============================================================
MARKET = "US"
CURRENCY = "USD"
CURRENCY_SYMBOL = "$"
TICKER = "AAPL"
TICKER_NAME = "苹果"
TICKER_NAME_CN = "Apple"

# 预设股票库（Yahoo / IBKR 均使用英文股票代码）
PRESET_STOCKS = {
    "AAPL": "苹果",
    "NVDA": "英伟达",
    "ORCL": "甲骨文",
    "MSFT": "微软",
    "AMZN": "亚马逊",
    "GOOGL": "谷歌 A",
    "META": "Meta",
    "TSLA": "特斯拉",
    "AVGO": "博通",
    "AMD": "超威半导体",
    "INTC": "英特尔",
    "SPY": "标普 500 ETF",
    "QQQ": "纳斯达克 100 ETF",
}

# SMART 路由遇到同名证券时，用主上市交易所消歧。
IBKR_PRIMARY_EXCHANGES = {
    "AAPL": "NASDAQ",
    "NVDA": "NASDAQ",
    "ORCL": "NYSE",
    "MSFT": "NASDAQ",
    "AMZN": "NASDAQ",
    "GOOGL": "NASDAQ",
    "META": "NASDAQ",
    "TSLA": "NASDAQ",
    "AVGO": "NASDAQ",
    "AMD": "NASDAQ",
    "INTC": "NASDAQ",
    "SPY": "ARCA",
    "QQQ": "NASDAQ",
}

def normalize_ticker(code: str) -> tuple:
    """
    标准化美股代码。

    接受 ``AAPL``、``US.AAPL`` 和 ``AAPL.US``。返回值保持三个字段，
    以兼容原平台调用方：``(yahoo_code, broker_code, display_code)``。
    """
    if not isinstance(code, str):
        raise ValueError("美股代码必须是字符串")

    code = code.strip().upper().replace(" ", "")
    if not code:
        raise ValueError("请输入美股代码")

    raw = code.removeprefix("US.").removesuffix(".US")
    # 支持 BRK-B / BRK.B 这类类别股代码；当前交易台仍按整股处理。
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", raw):
        raise ValueError("代码格式无效，请输入 AAPL、NVDA、ORCL 等美股代码")
    return raw, raw, raw


def ibkr_primary_exchange(ticker: str) -> str:
    """返回 IBKR SMART 股票合约的主上市交易所。"""
    return IBKR_PRIMARY_EXCHANGES.get(normalize_ticker(ticker)[0], "")

# ============================================================
#  资金与风控（核心！）
# ============================================================
INITIAL_CAPITAL = 100_000.0       # 初始资金 (USD)
MAX_POSITION_PCT = 0.30           # 单票最大仓位 30%
MAX_SINGLE_RISK_PCT = 0.02        # 单笔最大风险 2%（总资金占比）
COMMISSION_RATE = 0.0005          # 回测保守费用假设 5bp（实际以账户费率为准）
DEFAULT_LOT_SIZE = 1              # 美股普通订单按 1 股递增
BACKTEST_MIN_HOLD_BARS = 0        # 美股允许日内卖出（不模拟 PDT/账户资格）
ALLOW_FRACTIONAL_SHARES = False   # 首版禁用碎股，避免券商资格/订单类型差异
ALLOW_EXTENDED_HOURS_ORDERS = False  # 默认只在常规时段下单

# ============================================================
#  止损止盈（ATR 倍数制）
# ============================================================
STOP_LOSS_ATR_MULT = 2.0          # 止损 = 入场价 - N * ATR
TAKE_PROFIT_ATR_MULT = 3.0        # 止盈 = 入场价 + N * ATR
TRAILING_STOP_ATR_MULT = 1.5      # 移动止损距离
TRAILING_ACTIVATE_PCT = 0.03      # 盈利 3% 后激活移动止损

# ============================================================
#  数据周期
# ============================================================
PRIMARY_TF = "1d"
SECONDARY_TFS = ["60m", "30m"]
BACKTEST_YEARS = 3               # 回测最长拉 3 年

# ============================================================
#  技术指标参数
# ============================================================
MA_SHORT = 5
MA_MEDIUM = 20
MA_LONG = 60

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

BOLL_PERIOD = 20
BOLL_STD = 2.0

ATR_PERIOD = 14
VOL_MA_PERIOD = 20
VOL_SURGE_RATIO = 1.8

# ============================================================
#  策略参数（各策略独立） ====================================
# ============================================================

# 策略 1: 双均线趋势跟踪
STRAT_MA_CROSSOVER = {
    "fast": 5,
    "slow": 20,
    "filter_ma": 60,       # 长期均线过滤（只做顺势）
    "min_bars": 3,         # 金叉后至少持有 3 根K线
}

# 策略 2: 布林带均值回归
STRAT_BOLL_REVERSION = {
    "period": 20,
    "std": 2.0,
    "entry_pct": 0.05,     # 触及上/下轨 5% 范围内入场
    "rsi_filter": True,    # 用 RSI 确认
}

# 策略 3: 量价突破
STRAT_VOL_BREAKOUT = {
    "lookback": 20,
    "vol_ratio": 1.5,       # 量比阈值（降低门槛）
    "breakout_pct": 0.0,    # 突破前高即触发（0%缓冲）
    "atr_filter": True,
}

# 策略 4: 多周期共振
STRAT_MULTI_TF = {
    "trend_tf": "1d",      # 日线定方向
    "entry_tf": "60m",     # 60分找入场
    "ma_periods": [5, 20, 60],
    "rsi_min": 40,
    "rsi_max": 70,
}

# ============================================================
#  面板设置
# ============================================================
PAGE_TITLE = "🔬 美股量化交易平台"
PAGE_LAYOUT = "wide"
AUTO_REFRESH_SEC = 60

# ============================================================
#  兼容 indicators.py & signal_engine.py（保留项）
# ============================================================
MA_PERIODS = [5, 10, 20, 60, 120, 250]
MA_COLORS = {5: "#FF6B6B", 10: "#FFA94D", 20: "#FFD43B", 60: "#69DB7C", 120: "#4DABF7", 250: "#DA77F2"}
KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3

SIGNAL_WEIGHTS = {
    "ma_golden_cross": 3.0, "ma_death_cross": -3.0,
    "macd_golden_cross": 2.5, "macd_death_cross": -2.5,
    "rsi_oversold": 2.0, "rsi_overbought": -2.0,
    "kdj_oversold": 1.5, "kdj_overbought": -1.5,
    "boll_lower_touch": 1.5, "boll_upper_touch": -1.5,
    "vol_surge_bull": 1.0, "vol_surge_bear": -1.0,
    "ma_support": 1.0, "ma_resistance": -1.0,
}
SCORE_STRONG_BUY = 6.0
SCORE_BUY = 3.0
SCORE_NEUTRAL = 0.0
SCORE_SELL = -3.0
SCORE_STRONG_SELL = -6.0
ALERT_PROFIT_TP1 = 5.0
ALERT_PROFIT_TP2 = 10.0
ALERT_LOSS_SL = -5.0
PRIMARY_PERIOD = "1d"
SECONDARY_PERIODS = ["60m", "30m", "15m"]
HISTORY_DAYS = 365
HISTORY_PERIOD = "1d"
