"""
可插拔数据源层 — 一行切换 API
"""

from .base import DataSource
from .yfinance_source import YFinanceSource
from .eastmoney_source import EastMoneySource
from .akshare_source import AKShareSource
from .tencent_source import TencentSource


# 注册表：名称 → (标签, 类, 特点描述)
SOURCES = {
    "eastmoney": (
        "🇨🇳 东方财富",
        EastMoneySource,
        "沪深 A 股实时行情与历史 K 线，无需密钥"
    ),
    "tencent": (
        "🇨🇳 腾讯证券",
        TencentSource,
        "沪深北 A 股历史 K 线与行情快照，无需密钥"
    ),
    "yfinance": (
        "🌍 Yahoo Finance",
        YFinanceSource,
        "沪深 A 股历史行情备用源，实时性以数据授权为准"
    ),
    "akshare": (
        "📦 AKShare",
        AKShareSource,
        "A 股聚合数据源，适合辅助核对与扩展分析"
    ),
}
