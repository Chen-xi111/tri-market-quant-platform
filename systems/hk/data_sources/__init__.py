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
    "tencent": (
        "腾讯证券",
        TencentSource,
        "低延迟港股报价与日线；自动失败回退",
    ),
    "eastmoney": (
        "🇨🇳 东方财富",
        EastMoneySource,
        "免费实时港股行情，零延迟，无需额外包"
    ),
    "yfinance": (
        "🌍 Yahoo Finance",
        YFinanceSource,
        "全球覆盖，港股延迟~15分钟，数据稳定"
    ),
    "akshare": (
        "📦 AKShare",
        AKShareSource,
        "多源聚合，支持分钟线、资金流向等深度数据"
    ),
}
