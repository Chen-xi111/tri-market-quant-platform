"""A 股平台顶层导航；原工作台与资金流专区相互隔离。"""

import streamlit as st

import dashboard
from fund_flow.scheduler import DailyFundFlowScheduler


@st.cache_resource
def start_fund_flow_scheduler() -> DailyFundFlowScheduler:
    return DailyFundFlowScheduler().start()


start_fund_flow_scheduler()

page = st.navigation(
    [
        st.Page(
            dashboard.main,
            title="个股策略工作台",
            icon=":material/candlestick_chart:",
            default=True,
        ),
        st.Page(
            "app_pages/fund_flow_radar.py",
            title="全市场资金雷达",
            icon=":material/radar:",
        ),
        st.Page(
            "app_pages/opportunities.py",
            title="每日强信心候选",
            icon=":material/stars:",
        ),
        st.Page(
            "app_pages/quant_research.py",
            title="量化研究验证",
            icon=":material/science:",
        ),
    ],
    position="top",
)
page.run()
