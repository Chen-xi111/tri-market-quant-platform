"""Portfolio-facing overview page."""

import pandas as pd
import streamlit as st

from platform_hub import MARKETS, service_is_healthy

st.title("Tri-market quantitative research platform")
st.caption(
    "One research and paper-trading workspace spanning China A-shares, "
    "Hong Kong equities and US equities."
)

health = {key: service_is_healthy(market.port) for key, market in MARKETS.items()}

with st.container(horizontal=True):
    st.metric("Markets", "3", border=True)
    st.metric("Strategy families", "7", border=True)
    st.metric("Market services online", f"{sum(health.values())}/3", border=True)
    st.metric("Live-order default", "Disabled", border=True)

st.subheader("Market workspaces")
for key, market in MARKETS.items():
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader(market.name)
            st.badge(
                "Online" if health[key] else "Offline",
                color="green" if health[key] else "red",
            )
        st.write(market.focus)
        st.caption(" · ".join(market.capabilities))

st.subheader("Shared research workflow")
workflow = pd.DataFrame(
    [
        ("1. Data", "Source fallback, quote timestamps and cache boundaries"),
        ("2. Signal", "Seven explainable strategy families and composite scoring"),
        ("3. Validation", "Time-ordered splits, parameter grids and readiness gates"),
        ("4. Risk", "Position sizing, fees, slippage and market-specific constraints"),
        ("5. Execution", "Paper broker by default; explicit safety gates for adapters"),
    ],
    columns=["Stage", "Evidence in the repository"],
)
st.dataframe(workflow, hide_index=True)

st.info(
    "This project is an educational research system, not investment advice. "
    "Broker adapters are optional and live trading is not enabled by the unified launcher.",
    icon=":material/info:",
)
