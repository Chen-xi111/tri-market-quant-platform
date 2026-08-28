"""沪深 A 股量化交易平台 — 全局配置。"""

import re

# ============================================================
#  标的（默认 + 预设股票库）
# ============================================================
MARKET = "CN"
CURRENCY = "CNY"
CURRENCY_SYMBOL = "¥"
TICKER = "600259.SS"
TICKER_NAME = "中稀有色"
TICKER_NAME_CN = "China Rare Earth Nonferrous Metals"

# 预设股票库使用 Yahoo 格式；上海证券以 .SS 结尾，深圳证券以 .SZ 结尾。
PRESET_STOCKS = {
    "600259.SS": "中稀有色",
    "600010.SS": "包钢股份",
    "000831.SZ": "中国稀土",
    "600111.SS": "北方稀土",
    "600019.SS": "宝钢股份",
    "000630.SZ": "铜陵有色",
    "601600.SS": "中国铝业",
    "000858.SZ": "五粮液",
    "600519.SS": "贵州茅台",
    "000333.SZ": "美的集团",
    "300750.SZ": "宁德时代",
    "002594.SZ": "比亚迪",
    "601318.SS": "中国平安",
}

def normalize_ticker(code: str) -> tuple:
    """
    标准化沪深北 A 股代码。

    接受 ``000831``、``SZ.000831``、``SZ000831``、``000831.SZ``、
    ``SH.600010``、``600010.SH``、Yahoo 的 ``600010.SS`` 以及
    北交所 ``BJ.920222``。返回
    ``(yahoo_code, broker_code, display_code)``。
    """
    if not isinstance(code, str):
        raise ValueError("A 股代码必须是字符串")

    code = code.strip().upper().replace(" ", "")
    if not code:
        raise ValueError("请输入 6 位 A 股代码")

    explicit_exchange = None
    for prefix in ("SH.", "SZ.", "BJ.", "SH:", "SZ:", "BJ:", "SH", "SZ", "BJ"):
        if code.startswith(prefix):
            explicit_exchange, code = prefix[:2], code[len(prefix):]
            break
    for suffix in (".SH", ".SS", ".SZ", ".BJ"):
        if code.endswith(suffix):
            suffix_exchange = "SH" if suffix in (".SH", ".SS") else suffix[1:]
            if explicit_exchange and explicit_exchange != suffix_exchange:
                raise ValueError("代码的交易所前缀与后缀不一致")
            explicit_exchange = suffix_exchange
            code = code[:-len(suffix)]
            break

    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("代码格式无效，请输入 000831、600010.SH 或 BJ.920222")

    inferred_exchange = _exchange_for_code(code)
    if explicit_exchange and explicit_exchange != inferred_exchange:
        raise ValueError(f"代码 {code} 应属于{exchange_name(inferred_exchange)}，请检查交易所后缀")

    yahoo_suffix = {"SH": "SS", "SZ": "SZ", "BJ": "BJ"}[inferred_exchange]
    return f"{code}.{yahoo_suffix}", f"{inferred_exchange}.{code}", code


def _exchange_for_code(code: str) -> str:
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    if code.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "SZ"
    if code.startswith("920"):
        return "BJ"
    raise ValueError("当前版本支持沪深北 A 股主板、创业板、科创板和北交所 920 代码")


def exchange_name(exchange: str) -> str:
    return {"SH": "沪市", "SZ": "深市", "BJ": "北交所"}.get(exchange, exchange)


def exchange_for_ticker(ticker: str) -> str:
    return normalize_ticker(ticker)[1].split(".", 1)[0]

# ============================================================
#  资金与风控（核心！）
# ============================================================
INITIAL_CAPITAL = 100_000.0       # 初始资金 (CNY)
MAX_POSITION_PCT = 0.30           # 单票最大仓位 30%
MAX_SINGLE_RISK_PCT = 0.02        # 单笔最大风险 2%（总资金占比）
COMMISSION_RATE = 0.0003           # 模拟券商佣金 0.03%（账户实际费率可能不同）
MIN_COMMISSION = 5.0               # 每笔最低佣金假设
STAMP_DUTY_RATE = 0.0005           # 卖出单边证券交易印花税 0.05%
TRANSFER_FEE_RATE = 0.00001        # 过户费近似假设，双向计入
DEFAULT_LOT_SIZE = 100             # 普通买入按 100 股为一手
BACKTEST_MIN_HOLD_BARS = 1         # A 股 T+1：买入当日不能卖出
PRICE_TICK = 0.01                  # A 股申报价最小变动单位

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
PAGE_TITLE = "🔬 A股量化交易平台"
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
