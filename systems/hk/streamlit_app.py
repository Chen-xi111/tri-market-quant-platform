"""港股平台顶层导航。"""

import streamlit as st

import dashboard
from market_scan.scheduler import DailyMarketScanScheduler


@st.cache_resource
def start_market_scan_scheduler() -> DailyMarketScanScheduler:
    return DailyMarketScanScheduler().start()


start_market_scan_scheduler()

page = st.navigation(
    [
        st.Page(
            dashboard.main,
            title="个股策略工作台",
            icon=":material/candlestick_chart:",
            default=True,
        ),
        st.Page(
            "app_pages/market_scanner.py",
            title="港股全市场扫描",
            icon=":material/radar:",
        ),
        st.Page(
            "app_pages/opportunities.py",
            title="每日强信心候选",
            icon=":material/stars:",
        ),
    ],
    position="top",
)
page.run()
