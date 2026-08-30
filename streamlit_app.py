"""Unified entry point for the tri-market quant research platform."""

import streamlit as st

st.set_page_config(
    page_title="Tri-market quant platform",
    page_icon=":material/finance_mode:",
    layout="wide",
    initial_sidebar_state="auto",
)

st.logo(":material/candlestick_chart:", size="large")

page = st.navigation(
    {
        "": [
            st.Page(
                "app_pages/home.py",
                title="Project overview",
                icon=":material/home:",
                default=True,
            )
        ],
        "Markets": [
            st.Page(
                "app_pages/cn_market.py",
                title="China A-shares",
                icon=":material/candlestick_chart:",
                url_path="cn",
            ),
            st.Page(
                "app_pages/hk_market.py",
                title="Hong Kong equities",
                icon=":material/candlestick_chart:",
                url_path="hk",
            ),
            st.Page(
                "app_pages/us_market.py",
                title="US equities",
                icon=":material/candlestick_chart:",
                url_path="us",
            ),
        ],
        "Project": [
            st.Page(
                "app_pages/methodology.py",
                title="Research methodology",
                icon=":material/science:",
                url_path="methodology",
            )
        ],
    },
    position="sidebar",
)

page.run()
