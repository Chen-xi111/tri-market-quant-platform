"""Shared presentation and health helpers for the unified Streamlit portal."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st

from portfolio.demo_ui import render_demo_dashboard


@dataclass(frozen=True)
class MarketApp:
    key: str
    name: str
    short_name: str
    port: int
    entrypoint: str
    focus: str
    capabilities: tuple[str, ...]


MARKETS: dict[str, MarketApp] = {
    "cn": MarketApp(
        key="cn",
        name="China A-share research",
        short_name="A-shares",
        port=8503,
        entrypoint="systems/cn/streamlit_app.py",
        focus="Fund-flow discovery, opportunity scoring, strategy validation and paper execution.",
        capabilities=(
            "Market-wide fund-flow radar",
            "Daily opportunity scanner",
            "A-share trading rules and fees",
            "Backtest robustness workspace",
        ),
    ),
    "hk": MarketApp(
        key="hk",
        name="Hong Kong equity research",
        short_name="Hong Kong",
        port=8502,
        entrypoint="systems/hk/streamlit_app.py",
        focus="Market scanning with Hong Kong lot, fee and limited-history handling.",
        capabilities=(
            "Market-wide scanner",
            "Strong-confidence candidates",
            "HKEX lot and fee rules",
            "Limited-history safeguards",
        ),
    ),
    "us": MarketApp(
        key="us",
        name="US equity research",
        short_name="US equities",
        port=8501,
        entrypoint="systems/us/dashboard.py",
        focus="Cross-sectional ranking, live quote health, robust backtesting and broker-safe paper trading.",
        capabilities=(
            "Cross-sectional stock ranking",
            "Quote freshness and fallback chain",
            "IBKR paper-only safety boundary",
            "Strategy tournament and stress tests",
        ),
    ),
}


def _internal_host() -> str:
    return os.getenv("QUANT_INTERNAL_HOST", "127.0.0.1")


def public_mode_enabled() -> bool:
    """Default to a read-only portfolio unless local workspaces are explicitly enabled."""
    value = os.getenv("QUANT_PUBLIC_MODE", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def market_url(market: MarketApp) -> str:
    """Return a browser URL from trusted deployment configuration only."""
    host = os.getenv("QUANT_PUBLIC_HOST", "localhost")
    scheme = os.getenv("QUANT_PUBLIC_SCHEME", "http")
    return f"{scheme}://{host}:{market.port}"


@st.cache_data(ttl=3, show_spinner=False)
def service_is_healthy(port: int) -> bool:
    host = _internal_host()
    try:
        with urlopen(f"http://{host}:{port}/_stcore/health", timeout=0.7) as response:
            return response.status == 200
    except (OSError, URLError):
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return True
        except OSError:
            return False


def render_market_page(key: str) -> None:
    market = MARKETS[key]

    st.title(market.name)
    st.caption(market.focus)

    if public_mode_enabled():
        render_demo_dashboard(key)
        return

    view = st.segmented_control(
        "Workspace mode",
        ["Interactive demo", "Local live workspace"],
        default="Interactive demo",
        required=True,
        key=f"workspace_mode_{key}",
        width="stretch",
        persist_state="session",
    )

    if view == "Interactive demo":
        render_demo_dashboard(key)
        return

    healthy = service_is_healthy(market.port)

    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(
            "Service online" if healthy else "Service offline",
            color="green" if healthy else "red",
            icon=":material/check_circle:" if healthy else ":material/error:",
        )
        st.code(market.entrypoint, language=None)
        st.link_button(
            "Open in a new tab",
            market_url(market),
            icon=":material/open_in_new:",
            disabled=not healthy,
        )

    if not healthy:
        st.error(
            "This market service is not running. Start the complete platform from "
            "the repository root with `./run_all.sh`."
        )
        st.code(f"./run_{key}.sh", language="bash")
        return

    st.iframe(market_url(market), height=1050)
