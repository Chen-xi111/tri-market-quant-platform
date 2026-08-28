"""
量化交易平台 — 全局配置
标的: 中广核矿业 01164.HK
"""

# ============================================================
#  标的（默认 + 预设股票库）
# ============================================================
TICKER = "1164.HK"             # Yahoo 格式（四位数字 + .HK）
TICKER_NAME = "中广核矿业"
TICKER_NAME_CN = "CGN Mining"

# 预设股票库（代码 → 名称）
# 港股代码格式: Yahoo用 "XXXX.HK"，富途用 "HK.XXXX"
PRESET_STOCKS = {
    "1164.HK": "中广核矿业",
    "2172.HK": "微创脑科学",
    "0700.HK": "腾讯控股",
    "9988.HK": "阿里巴巴-W",
    "3690.HK": "美团-W",
    "1810.HK": "小米集团-W",
    "9888.HK": "百度集团-SW",
    "9999.HK": "网易-S",
    "1024.HK": "快手-W",
    "2318.HK": "中国平安",
    "0388.HK": "香港交易所",
    "2013.HK": "微盟集团",
    "2196.HK": "复星医药",
    "3333.HK": "中国恒大",
    "2333.HK": "长城汽车",
    "0941.HK": "中国移动",
}

def normalize_ticker(code: str) -> tuple:
    """
    将用户输入的港股代码标准化为 5 位格式。

    仅接受 1~5 位数字，可选 ``HK.`` 前缀或 ``.HK`` 后缀。
    返回: (yahoo_format, futu_format, display_code)
    例: "02172" → ("2172.HK", "HK.02172", "02172")
        "2172.HK" → ("2172.HK", "HK.02172", "02172")
        "HK.02172" → ("2172.HK", "HK.02172", "02172")
        "700" → ("0700.HK", "HK.00700", "00700")
    """
    if not isinstance(code, str):
        raise ValueError("港股代码必须是字符串")

    code = code.strip().upper().replace(" ", "")
    if not code:
        raise ValueError("请输入港股代码")

    raw = code.replace("HK.", "").replace(".HK", "")
    if not raw.isdigit() or len(raw) > 5 or not 1 <= int(raw) <= 9999:
        raise ValueError("代码格式无效，请输入 00001~09999（如 00700 或 700.HK）")

    # 去掉前导零后重新补齐到 5 位
    raw_stripped = raw.lstrip("0") or "0"
    raw_5 = raw_stripped.zfill(5)
    yahoo = f"{raw_stripped.zfill(4)}.HK"
    futu = f"HK.{raw_5}"
    return yahoo, futu, raw_5

# ============================================================
#  资金与风控（核心！）
# ============================================================
INITIAL_CAPITAL = 100_000.0       # 初始资金 (HKD)
MAX_POSITION_PCT = 0.30           # 单票最大仓位 30%
MAX_SINGLE_RISK_PCT = 0.02        # 单笔最大风险 2%（总资金占比）
COMMISSION_RATE = 0.0003          # 券商佣金假设 3bp/边（实际按账户费率覆盖）
MIN_COMMISSION = 0.0              # 券商最低佣金差异大，默认不额外假设
STAMP_DUTY_RATE = 0.001           # 港股印花税 0.1%/边，逐笔向上取整到港元
SFC_LEVY_RATE = 0.000027          # 证监会交易征费 0.0027%/边
AFRC_LEVY_RATE = 0.0000015        # 会财局交易征费 0.00015%/边
TRADING_FEE_RATE = 0.0000565      # 港交所交易费 0.00565%/边
DEFAULT_LOT_SIZE = 1              # 页面可按证券实际每手股数覆盖
BACKTEST_MIN_HOLD_BARS = 0         # 港股允许日内卖出

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
PAGE_TITLE = "🔬 港股量化交易平台"
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
