from .base import Strategy, Signal, SignalType
from .ma_crossover import MACrossoverStrategy
from .bollinger_reversion import BollingerReversionStrategy
from .vol_breakout import VolBreakoutStrategy
from .multi_tf import MultiTFStrategy
from .alpha_factors import AlphaFactorStrategy, factor_ic_analysis
from .ml_strategy import MLStrategy
from .trend_rider import TrendRiderStrategy
from .composite import CompositeStrategy, CompositeSignal

STRATEGIES = {
    "trend_rider":          ("趋势骑乘", TrendRiderStrategy),
    "ma_crossover":         ("双均线趋势跟踪", MACrossoverStrategy),
    "bollinger_reversion":  ("布林带均值回归", BollingerReversionStrategy),
    "vol_breakout":         ("量价突破", VolBreakoutStrategy),
    "multi_tf":             ("多周期共振", MultiTFStrategy),
    "alpha_factors":        ("Alpha 多因子模型", AlphaFactorStrategy),
    "ml_predict":           ("XGBoost 机器学习", MLStrategy),
}
