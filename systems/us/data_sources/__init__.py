"""
可插拔数据源层 — 一行切换 API
"""

from .base import DataSource
from .eastmoney_source import EastMoneyUSSource
from .futu_source import FutuOpenDSource
from .yfinance_source import YFinanceSource


# 注册表：名称 → (标签, 类, 特点描述)
SOURCES = {
    "futu": (
        "富途 OpenD",
        FutuOpenDSource,
        "本机富途 OpenD 持久连接；支持秒级刷新并自动回退公开行情"
    ),
    "eastmoney": (
        "东方财富美股快照",
        EastMoneyUSSource,
        "轻量单股快照并显示实际行情时点；历史日 K 自动由 Yahoo 补充"
    ),
    "yfinance": (
        "🌍 Yahoo Finance",
        YFinanceSource,
        "美股历史行情与报价；实时性取决于 Yahoo 数据授权，交易前请以券商报价复核"
    ),
}
