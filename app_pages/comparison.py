"""Cross-market comparison lab for the public portfolio."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from portfolio.comparison_engine import MARKET_ORDER, build_cross_market_comparison
from portfolio.demo_engine import STRATEGIES
from portfolio.evidence_export import comparison_export_stem

MARKET_LABELS = {
    "cn": "China A-shares",
    "hk": "Hong Kong equities",
    "us": "US equities",
}
MARKET_CHART_LABELS = {"cn": "China", "hk": "Hong Kong", "us": "US"}
MARKET_COLORS = ["#1D4ED8", "#6D28D9", "#15803D"]
COST_OPTIONS = [0, 5, 7, 10, 12, 15, 25, 40]


@st.cache_data(show_spinner=False, max_entries=24)
def cached_comparison(
    strategy_key: str,
    cost_basis: str,
    common_cost_bps: float,
) -> pd.DataFrame:
    return build_cross_market_comparison(strategy_key, cost_basis, common_cost_bps)


def risk_return_chart(frame: pd.DataFrame) -> alt.Chart:
    chart_data = frame.assign(drawdown_magnitude=-frame["max_drawdown"])
    points = (
        alt.Chart(chart_data)
        .mark_circle(size=150, opacity=0.85)
        .encode(
            x=alt.X(
                "drawdown_magnitude:Q",
                title="Validation drawdown magnitude",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "validation_return:Q",
                title="Validation return",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "market_short:N",
                title=None,
                legend=alt.Legend(orient="bottom"),
                scale=alt.Scale(
                    domain=[MARKET_CHART_LABELS[key] for key in MARKET_ORDER],
                    range=MARKET_COLORS,
                ),
            ),
            tooltip=[
                alt.Tooltip("symbol:N", title="Instrument"),
                alt.Tooltip("name:N", title="Name"),
                alt.Tooltip("market:N", title="Market"),
                alt.Tooltip("validation_return:Q", title="Validation return", format=".2%"),
                alt.Tooltip("excess_return:Q", title="Excess return", format=".2%"),
                alt.Tooltip("validation_sharpe:Q", title="Validation Sharpe", format=".2f"),
                alt.Tooltip("drawdown_magnitude:Q", title="Drawdown magnitude", format=".2%"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#94A3B8").encode(y="zero:Q")
    return (points + zero).properties(height=390)


def excess_return_chart(frame: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("excess_return:Q", title="Validation excess return", axis=alt.Axis(format="%")),
            y=alt.Y("symbol:N", title=None, sort="-x"),
            color=alt.Color(
                "market_short:N",
                title=None,
                legend=alt.Legend(orient="bottom"),
                scale=alt.Scale(
                    domain=[MARKET_CHART_LABELS[key] for key in MARKET_ORDER],
                    range=MARKET_COLORS,
                ),
            ),
            tooltip=[
                alt.Tooltip("symbol:N", title="Instrument"),
                alt.Tooltip("market:N", title="Market"),
                alt.Tooltip("excess_return:Q", title="Excess return", format=".2%"),
            ],
        )
        .properties(height=320)
    )


def train_validation_chart(frame: pd.DataFrame) -> alt.Chart:
    lower = float(frame[["train_sharpe", "validation_sharpe"]].min().min()) - 0.15
    upper = float(frame[["train_sharpe", "validation_sharpe"]].max().max()) + 0.15
    diagonal = pd.DataFrame({"value": [lower, upper]})
    line = (
        alt.Chart(diagonal)
        .mark_line(strokeDash=[5, 5], color="#94A3B8")
        .encode(x="value:Q", y="value:Q")
    )
    points = (
        alt.Chart(frame)
        .mark_circle(size=140, opacity=0.85)
        .encode(
            x=alt.X("train_sharpe:Q", title="Train Sharpe", scale=alt.Scale(domain=[lower, upper])),
            y=alt.Y(
                "validation_sharpe:Q",
                title="Validation Sharpe",
                scale=alt.Scale(domain=[lower, upper]),
            ),
            color=alt.Color(
                "market_short:N",
                title=None,
                legend=alt.Legend(orient="bottom"),
                scale=alt.Scale(
                    domain=[MARKET_CHART_LABELS[key] for key in MARKET_ORDER],
                    range=MARKET_COLORS,
                ),
            ),
            tooltip=[
                alt.Tooltip("symbol:N", title="Instrument"),
                alt.Tooltip("train_sharpe:Q", title="Train Sharpe", format=".2f"),
                alt.Tooltip("validation_sharpe:Q", title="Validation Sharpe", format=".2f"),
                alt.Tooltip("sharpe_decay:Q", title="Change", format="+.2f"),
            ],
        )
    )
    return (line + points).properties(height=320)


with st.container(horizontal=True, vertical_alignment="center"):
    st.badge("New portfolio view", color="blue", icon=":material/compare_arrows:")
    st.badge("9 fixed snapshots", color="gray", icon=":material/database:")
    st.badge("No pooled return claim", color="green", icon=":material/fact_check:")

st.title("Cross-market comparison lab")
st.caption(
    "Apply one transparent rule across China, Hong Kong and US snapshots. "
    "Results remain instrument-level because calendars, currencies and sample lengths differ."
)

with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        strategy_key = st.selectbox(
            "Research rule",
            list(STRATEGIES),
            format_func=STRATEGIES.get,
            key="comparison_strategy",
            width=260,
            persist_state="session",
        )
        cost_label = st.segmented_control(
            "Cost basis",
            ["Venue defaults", "Common cost"],
            default="Venue defaults",
            required=True,
            key="comparison_cost_basis",
            width=320,
            persist_state="session",
        )
        common_cost = st.selectbox(
            "Common one-way cost",
            COST_OPTIONS,
            index=COST_OPTIONS.index(10),
            format_func=lambda value: f"{value} bps",
            disabled=cost_label != "Common cost",
            key="comparison_common_cost",
            width=200,
            persist_state="session",
        )
    selected_markets = st.pills(
        "Markets shown",
        list(MARKET_ORDER),
        default=list(MARKET_ORDER),
        selection_mode="multi",
        format_func=MARKET_LABELS.get,
        key="comparison_markets",
        width="stretch",
        persist_state="session",
    )
    st.caption(
        "Venue defaults use 12 bps for China, 15 bps for Hong Kong and 7 bps for the US. "
        "Choose a common cost only for a controlled sensitivity comparison."
    )

if not selected_markets:
    st.info("Select at least one market to build the comparison.", icon=":material/info:")
    st.stop()

cost_basis = "venue" if cost_label == "Venue defaults" else "common"
result_slot = st.container()
with result_slot.skeleton(height=300):
    full_results = cached_comparison(strategy_key, cost_basis, float(common_cost))
results = full_results[full_results["market_key"].isin(selected_markets)].copy()
results["market_short"] = results["market_key"].map(MARKET_CHART_LABELS)

positive_sharpe = int((results["validation_sharpe"] > 0).sum())
median_excess = float(results["excess_return"].median())
median_drawdown = float(results["max_drawdown"].median())
median_decay = float(results["sharpe_decay"].median())

with st.container(horizontal=True):
    st.metric(
        "Instruments compared",
        len(results),
        delta=f"{len(selected_markets)} markets",
        delta_color="off",
        delta_arrow="off",
        border=True,
        icon=":material/candlestick_chart:",
    )
    st.metric(
        "Positive validation Sharpe",
        f"{positive_sharpe}/{len(results)}",
        delta="Snapshot count",
        delta_color="off",
        delta_arrow="off",
        border=True,
        icon=":material/query_stats:",
    )
    st.metric(
        "Median excess return",
        f"{median_excess:+.1%}",
        delta=f"{median_decay:+.2f} median Sharpe change",
        border=True,
        icon=":material/compare:",
    )
    st.metric(
        "Median max drawdown",
        f"{median_drawdown:.1%}",
        delta="Held-out period",
        delta_color="off",
        delta_arrow="off",
        border=True,
        icon=":material/area_chart:",
    )

best = results.sort_values("validation_sharpe", ascending=False).iloc[0]
st.info(
    f"**Descriptive readout:** {best['symbol']} has the highest validation Sharpe "
    f"({best['validation_sharpe']:.2f}) under this rule and cost basis. "
    "This fixed-snapshot comparison is not a security ranking or allocation recommendation.",
    icon=":material/insights:",
)

map_tab, scorecard_tab, method_tab = st.tabs(
    ["Risk-return map", "Scorecard", "Comparison contract"],
    key="comparison_tabs",
    on_change="rerun",
)

if map_tab.open:
    with map_tab:
        st.subheader("Held-out return versus drawdown")
        st.caption("Each point is one instrument; lower drawdown magnitude is farther left.")
        st.altair_chart(risk_return_chart(results), key="comparison_risk_return")
        chart_columns = st.columns(2, vertical_alignment="top")
        with chart_columns[0].container(border=True, height="stretch"):
            st.markdown("**Excess return by instrument**")
            st.caption("Validation strategy return minus validation buy-and-hold return.")
            st.altair_chart(excess_return_chart(results), key="comparison_excess")
        with chart_columns[1].container(border=True, height="stretch"):
            st.markdown("**Train-to-validation transfer**")
            st.caption("Points below the dashed diagonal weakened in validation.")
            st.altair_chart(train_validation_chart(results), key="comparison_transfer")

if scorecard_tab.open:
    with scorecard_tab:
        st.subheader("Instrument scorecard")
        scorecard = results.sort_values("validation_sharpe", ascending=False)[
            [
                "market",
                "symbol",
                "name",
                "cost_bps",
                "validation_return",
                "benchmark_return",
                "excess_return",
                "validation_sharpe",
                "max_drawdown",
                "positive_settings",
                "tested_settings",
                "evidence",
            ]
        ]
        st.dataframe(
            scorecard,
            hide_index=True,
            key="comparison_scorecard",
            column_config={
                "market": st.column_config.TextColumn("Market"),
                "symbol": st.column_config.TextColumn("Instrument", pinned=True),
                "name": st.column_config.TextColumn("Name"),
                "cost_bps": st.column_config.NumberColumn("Cost (bps)", format="%.0f"),
                "validation_return": st.column_config.NumberColumn(
                    "Strategy return", format="percent"
                ),
                "benchmark_return": st.column_config.NumberColumn("Buy and hold", format="percent"),
                "excess_return": st.column_config.NumberColumn("Excess return", format="percent"),
                "validation_sharpe": st.column_config.NumberColumn(
                    "Validation Sharpe", format="%.2f"
                ),
                "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
                "positive_settings": st.column_config.NumberColumn(
                    "Positive settings", format="%d"
                ),
                "tested_settings": st.column_config.NumberColumn("Tested settings", format="%d"),
                "evidence": st.column_config.TextColumn("Evidence"),
            },
        )
        st.download_button(
            "Download scorecard CSV",
            data=scorecard.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"{comparison_export_stem(strategy_key, cost_basis, float(common_cost), selected_markets)}.csv"
            ),
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
        )

if method_tab.open:
    with method_tab:
        st.subheader("Comparison contract")
        contract_columns = st.columns(3, vertical_alignment="top")
        for column, icon, title, body in [
            (
                contract_columns[0],
                ":material/functions:",
                "Same rule",
                "One deterministic signal family and parameter definition is applied to every instrument.",
            ),
            (
                contract_columns[1],
                ":material/payments:",
                "Visible costs",
                "Use venue defaults for realism or a common cost for a controlled sensitivity comparison.",
            ),
            (
                contract_columns[2],
                ":material/calendar_month:",
                "Separate histories",
                "Returns are never pooled across currencies, calendars or unequal sample windows.",
            ),
        ]:
            with column.container(border=True, height="stretch"):
                st.subheader(f"{icon} {title}")
                st.write(body)
        st.warning(
            "The nine instruments are illustrative snapshots, not a survivorship-free universe. "
            "Cross-sectional differences can reflect asset and period selection as well as the rule.",
            icon=":material/warning:",
        )
        st.dataframe(
            results[
                [
                    "market",
                    "symbol",
                    "name",
                    "currency",
                    "sample_start",
                    "sample_end",
                    "cost_bps",
                ]
            ],
            hide_index=True,
            key="comparison_universe",
            column_config={
                "market": st.column_config.TextColumn("Market"),
                "symbol": st.column_config.TextColumn("Instrument", pinned=True),
                "name": st.column_config.TextColumn("Name"),
                "currency": st.column_config.TextColumn("Currency"),
                "sample_start": st.column_config.DateColumn("Start"),
                "sample_end": st.column_config.DateColumn("End"),
                "cost_bps": st.column_config.NumberColumn("Cost (bps)", format="%.0f"),
            },
        )
