"""Portfolio-facing overview page."""

import streamlit as st

from platform_hub import MARKETS, service_is_healthy
from portfolio.demo_engine import load_demo_manifest

manifest = load_demo_manifest()

with st.container(horizontal=True, vertical_alignment="center"):
    st.badge("Public portfolio project", color="blue", icon=":material/public:")
    st.badge("Interactive offline demo", color="green", icon=":material/check_circle:")
    st.badge("Research, not advice", color="gray", icon=":material/info:")

st.title("Tri-market quantitative research platform")
st.write(
    "A reproducible equity-research and paper-execution system spanning China A-shares, "
    "Hong Kong and the United States—built to expose the assumptions between raw data "
    "and a trading decision."
)

with st.container(horizontal=True):
    if st.button(
        "Try the US research demo",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    ):
        st.switch_page("app_pages/us_market.py")
    if st.button(
        "Inspect the methodology",
        icon=":material/science:",
        width="stretch",
    ):
        st.switch_page("app_pages/methodology.py")

with st.container(horizontal=True):
    st.metric(
        "Markets",
        3,
        delta="Venue-specific rules",
        delta_color="off",
        delta_arrow="off",
        border=True,
        icon=":material/language:",
    )
    st.metric(
        "Full-platform strategies",
        7,
        delta="4 transparent demo rules",
        delta_color="off",
        delta_arrow="off",
        border=True,
        icon=":material/account_tree:",
    )
    st.metric(
        "Automated tests",
        139,
        delta="Plus 19 subtests",
        delta_color="off",
        delta_arrow="off",
        border=True,
        icon=":material/verified:",
    )
    st.metric(
        "Public live-order access",
        "Disabled",
        delta="Safety by default",
        delta_color="off",
        delta_arrow="off",
        border=True,
        icon=":material/lock:",
    )

st.subheader("Choose a market")
st.caption(
    "Every workspace opens with a fixed, broker-free research demo. Local live services are optional."
)

market_pages = {
    "cn": "app_pages/cn_market.py",
    "hk": "app_pages/hk_market.py",
    "us": "app_pages/us_market.py",
}
market_icons = {
    "cn": ":material/factory:",
    "hk": ":material/domain:",
    "us": ":material/show_chart:",
}

columns = st.columns(3, vertical_alignment="top")
for column, (key, market) in zip(columns, MARKETS.items()):
    demo_market = manifest["markets"][key]
    first_snapshot = next(iter(demo_market["symbols"].values()))
    with column.container(border=True, height="stretch"):
        st.subheader(f"{market_icons[key]} {market.short_name}")
        st.write(market.focus)
        st.caption(
            f"3 instruments · fixed through {first_snapshot['end']} · "
            f"default cost {demo_market['default_cost_bps']} bps"
        )
        st.markdown("\n".join(f"- {item}" for item in market.capabilities[:3]))
        if st.button(
            f"Explore {market.short_name}",
            key=f"open_market_{key}",
            icon=":material/arrow_forward:",
            width="stretch",
        ):
            st.switch_page(market_pages[key])

st.subheader("What the repository demonstrates")
evidence_columns = st.columns(3, vertical_alignment="top")
evidence = [
    (
        ":material/database:",
        "Data engineering",
        "Provider fallbacks, timestamp freshness, versioned demo fixtures and bounded caches.",
    ),
    (
        ":material/science:",
        "Research discipline",
        "Lagged signals, chronological validation, transaction costs and parameter-neighborhood checks.",
    ),
    (
        ":material/security:",
        "Execution safety",
        "Paper-first brokers, risk gates, venue constraints and an explicit public read-only boundary.",
    ),
]
for column, (icon, title, body) in zip(evidence_columns, evidence):
    with column.container(border=True, height="stretch"):
        st.subheader(f"{icon} {title}")
        st.write(body)

st.subheader("Research pipeline")
with st.container(horizontal=True):
    for icon, title, detail in [
        (":material/database:", "1. Data", "Normalize and timestamp"),
        (":material/functions:", "2. Signal", "Generate explainable rules"),
        (":material/experiment:", "3. Validate", "Hold out future periods"),
        (":material/shield:", "4. Risk", "Apply costs and limits"),
    ]:
        with st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(detail)

with st.expander("Local service status", icon=":material/dns:", on_change="rerun") as status:
    if status.open:
        for key, market in MARKETS.items():
            online = service_is_healthy(market.port)
            st.badge(
                f"{market.short_name}: {'online' if online else 'offline'}",
                color="green" if online else "gray",
                icon=":material/check_circle:" if online else ":material/power_off:",
            )

st.caption(
    "Educational research software. Public snapshots may be delayed or revised; "
    "backtests are hypothesis evidence, not an authorization to trade."
)
