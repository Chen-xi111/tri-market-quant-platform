"""Native Streamlit UI for the public, single-process research demo."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from portfolio.demo_engine import (
    STRATEGIES,
    DemoResult,
    evidence_diagnostics,
    load_demo_manifest,
    run_demo_research,
)
from portfolio.evidence_export import evidence_json, export_stem, history_csv

STRATEGY_NOTES = {
    "trend": "Own the instrument when its 40-day average is above the selected slow average.",
    "mean_reversion": "Enter after a statistically unusual decline and exit when price returns to its mean.",
    "breakout": "Enter on a volume-confirmed high and exit after a short-horizon price breakdown.",
    "multi_factor": "Combine trend, momentum and lower volatility with fixed, visible weights.",
}


@st.cache_data(show_spinner=False, max_entries=48)
def cached_demo_research(
    market_key: str,
    symbol: str,
    strategy_key: str,
    cost_bps: float,
) -> DemoResult:
    return run_demo_research(market_key, symbol, strategy_key, cost_bps)


def _percent(value: float) -> str:
    return f"{value:+.1%}"


def _equity_chart(result: DemoResult) -> alt.Chart:
    history = result.history.reset_index()
    chart_data = history.melt(
        id_vars="date",
        value_vars=["strategy_equity", "benchmark_equity"],
        var_name="series",
        value_name="growth",
    )
    labels = {
        "strategy_equity": "Strategy after costs",
        "benchmark_equity": "Buy and hold",
    }
    chart_data["series"] = chart_data["series"].map(labels)
    lines = (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("growth:Q", title="Growth of 1.0", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("growth:Q", title="Growth", format=".3f"),
            ],
        )
    )
    split = (
        alt.Chart(pd.DataFrame({"date": [result.split_date]}))
        .mark_rule(strokeDash=[5, 5], color="#6B7280")
        .encode(x="date:T")
    )
    return (lines + split).properties(height=360)


def _drawdown_chart(result: DemoResult) -> alt.Chart:
    data = result.history.reset_index()
    return (
        alt.Chart(data)
        .mark_area(opacity=0.4)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("drawdown:Q", title="Drawdown", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%"),
            ],
        )
        .properties(height=220)
    )


def _cost_chart(frame: pd.DataFrame) -> alt.Chart:
    line = (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("cost_bps:Q", title="One-way cost (bps)"),
            y=alt.Y(
                "validation_return:Q",
                title="Validation return",
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("cost_bps:Q", title="Cost", format=".0f"),
                alt.Tooltip("validation_return:Q", title="Validation return", format=".2%"),
                alt.Tooltip("validation_sharpe:Q", title="Validation Sharpe", format=".2f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#94A3B8").encode(y="zero:Q")
    return (line + zero).properties(height=300)


def _robustness_chart(frame: pd.DataFrame, x: str, value_columns: list[str]) -> alt.Chart:
    long = frame.melt(id_vars=x, value_vars=value_columns, var_name="metric", value_name="value")
    labels = {
        "train_sharpe": "Train Sharpe",
        "validation_sharpe": "Validation Sharpe",
        "validation_return": "Validation return",
    }
    long["metric"] = long["metric"].map(labels).fillna(long["metric"])
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{x}:Q", title=x.replace("_", " ").capitalize()),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("metric:N", title=None),
            tooltip=[
                alt.Tooltip(f"{x}:Q", title=x.replace("_", " ").capitalize()),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".3f"),
            ],
        )
        .properties(height=300)
    )


def _research_summary(result: DemoResult) -> str:
    diagnostics = evidence_diagnostics(result)
    alpha = float(diagnostics["validation_alpha"])
    direction = "outperformed" if alpha > 0 else "underperformed"
    return (
        f"In the held-out period, the strategy {direction} buy and hold by "
        f"{abs(alpha):.1%}. Validation Sharpe changed by "
        f"{float(diagnostics['sharpe_decay']):+.2f} versus training, and "
        f"{diagnostics['positive_settings']} of {diagnostics['tested_settings']} "
        "neighboring parameter settings "
        "produced a positive validation Sharpe. This is descriptive evidence, not a live-trading claim."
    )


def _period_comparison(result: DemoResult) -> pd.DataFrame:
    comparison = result.metrics.reset_index().rename(columns={"period": "Period"})
    comparison["alpha"] = comparison["strategy_return"] - comparison["benchmark_return"]
    return comparison[
        [
            "Period",
            "strategy_return",
            "benchmark_return",
            "alpha",
            "sharpe",
            "max_drawdown",
            "trades",
            "exposure",
        ]
    ]


def render_demo_dashboard(market_key: str) -> None:
    manifest = load_demo_manifest()
    market = manifest["markets"][market_key]
    symbols = market["symbols"]

    first_symbol = next(iter(symbols))
    default_cost = market["default_cost_bps"]
    cost_options = [0, 5, 7, 10, 12, 15, 25, 40]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge("Interactive demo", color="blue", icon=":material/science:")
        st.badge("Fixed snapshot", color="gray", icon=":material/history:")
        st.badge("Live orders disabled", color="green", icon=":material/lock:")

    st.caption(
        "This public demo runs entirely from versioned market snapshots. "
        "It does not require a broker, API key or live quote service."
    )

    with st.container(border=True):
        st.markdown(
            "**1. Choose an instrument · 2. Choose a rule · 3. Set costs · 4. Run the analysis**"
        )
        with (
            st.form(f"demo_controls_{market_key}", border=False),
            st.container(horizontal=True, vertical_alignment="bottom"),
        ):
            symbol = st.selectbox(
                "Instrument",
                list(symbols),
                index=0,
                format_func=lambda value: f"{value} · {symbols[value]['name']}",
                key=f"demo_symbol_{market_key}",
                persist_state="session",
                width=300,
            )
            strategy_key = st.selectbox(
                "Research rule",
                list(STRATEGIES),
                format_func=STRATEGIES.get,
                key=f"demo_strategy_{market_key}",
                persist_state="session",
                width=260,
            )
            cost_bps = st.selectbox(
                "One-way turnover cost",
                cost_options,
                index=cost_options.index(default_cost),
                format_func=lambda value: f"{value} bps",
                key=f"demo_cost_{market_key}",
                persist_state="session",
                help="Deducted every time the long-only position changes.",
                width=200,
            )
            submitted = st.form_submit_button(
                "Run analysis",
                type="primary",
                icon=":material/play_arrow:",
            )

        st.caption(STRATEGY_NOTES[strategy_key])
        with st.popover(
            "Research contract",
            icon=":material/fact_check:",
        ):
            st.markdown(
                "- Chronological split: **70% train / 30% validation**\n"
                "- Signal timing: **one-bar lag**\n"
                "- Position: **long or cash**\n"
                f"- Submitted one-way cost: **{cost_bps} bps**"
            )

    if submitted:
        st.toast("Research view updated", icon=":material/check_circle:")

    metadata = symbols.get(symbol, symbols[first_symbol])
    result_slot = st.container()
    with result_slot.skeleton(height=420):
        result = cached_demo_research(market_key, symbol, strategy_key, float(cost_bps))

    validation = result.metrics.loc["Validation"]
    train = result.metrics.loc["Train"]
    validation_history = result.history.loc[result.split_date :]
    sparkline = validation_history["strategy_equity"].tail(80).tolist()

    with st.container(horizontal=True):
        st.metric(
            "Validation return",
            _percent(validation["strategy_return"]),
            delta=_percent(validation["strategy_return"] - validation["benchmark_return"]),
            delta_description="vs. buy and hold",
            border=True,
            chart_data=sparkline,
            icon=":material/trending_up:",
        )
        st.metric(
            "Validation Sharpe",
            f"{validation['sharpe']:.2f}",
            delta=f"{validation['sharpe'] - train['sharpe']:+.2f}",
            delta_description="vs. train",
            border=True,
            icon=":material/query_stats:",
        )
        st.metric(
            "Maximum drawdown",
            _percent(validation["max_drawdown"]),
            delta_color="inverse",
            border=True,
            icon=":material/area_chart:",
        )
        st.metric(
            "Completed entries",
            int(validation["trades"]),
            delta=f"{validation['exposure']:.0%} exposure",
            delta_color="off",
            delta_arrow="off",
            border=True,
            icon=":material/swap_horiz:",
        )

    overview, robustness, evidence = st.tabs(
        ["Overview", "Robustness", "Data and assumptions"],
        key=f"demo_tabs_{market_key}",
        on_change="rerun",
    )

    if overview.open:
        with overview:
            st.subheader("Time-ordered validation")
            st.caption(
                f"The dashed line marks the held-out period beginning {result.split_date.date()}. "
                f"Returns include {cost_bps} bps on each position change."
            )
            st.altair_chart(_equity_chart(result))
            diagnostics = evidence_diagnostics(result)
            message = f"**{diagnostics['classification']}.** {_research_summary(result)}"
            if diagnostics["classification"] == "Consistent in this snapshot":
                st.success(message, icon=":material/fact_check:")
            else:
                st.warning(message, icon=":material/experiment:")

            st.markdown("**Train vs. validation scorecard**")
            st.dataframe(
                _period_comparison(result),
                hide_index=True,
                column_config={
                    "strategy_return": st.column_config.NumberColumn(
                        "Strategy return", format="percent"
                    ),
                    "benchmark_return": st.column_config.NumberColumn(
                        "Benchmark return", format="percent"
                    ),
                    "alpha": st.column_config.NumberColumn("Excess return", format="percent"),
                    "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                    "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
                    "trades": st.column_config.NumberColumn("Entries", format="%d"),
                    "exposure": st.column_config.NumberColumn("Exposure", format="percent"),
                },
                key=f"period_comparison_{market_key}",
            )
            with st.container(border=True):
                st.markdown("**Drawdown path**")
                st.altair_chart(_drawdown_chart(result))

    if robustness.open:
        with robustness:
            left, right = st.columns(2)
            with left.container(border=True, height="stretch"):
                st.markdown("**Transaction-cost sensitivity**")
                st.caption("Held-out return after increasing one-way turnover costs.")
                st.altair_chart(_cost_chart(result.cost_sensitivity))
            with right.container(border=True, height="stretch"):
                st.markdown("**Parameter stability**")
                st.caption(result.parameter_label)
                st.altair_chart(
                    _robustness_chart(
                        result.parameter_stability,
                        "parameter",
                        ["train_sharpe", "validation_sharpe"],
                    )
                )
            st.dataframe(
                result.parameter_stability,
                hide_index=True,
                column_config={
                    "parameter": st.column_config.NumberColumn(result.parameter_label),
                    "train_sharpe": st.column_config.NumberColumn("Train Sharpe", format="%.2f"),
                    "validation_sharpe": st.column_config.NumberColumn(
                        "Validation Sharpe", format="%.2f"
                    ),
                    "validation_return": st.column_config.NumberColumn(
                        "Validation return", format="percent"
                    ),
                },
                key=f"parameter_table_{market_key}",
            )

    if evidence.open:
        with evidence:
            st.markdown(
                f"""
**Snapshot provenance**

- Instrument: `{symbol}` — {metadata["name"]}
- Source adapter: `{metadata["source"]}`
- Observations: {metadata["rows"]} daily bars
- Coverage: {metadata["start"]} to {metadata["end"]}
- Snapshot saved: `{metadata["snapshot_saved_at"]}`
- Fixture SHA-256: `{metadata["sha256"][:12]}…` (verified at load)

**Execution assumptions**

- The rule is long-only and evaluated on daily bars.
- Signals are shifted by one bar before earning returns.
- A cost of {cost_bps} bps is charged on each position change.
- The split is chronological; no random train/test shuffle is used.
- Public-provider snapshots are not survivorship-free institutional data.
"""
            )
            export_name = export_stem(result)
            with st.container(horizontal=True):
                st.download_button(
                    "Download return path CSV",
                    data=history_csv(result),
                    file_name=f"{export_name}_returns.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    on_click="ignore",
                )
                st.download_button(
                    "Download evidence JSON",
                    data=evidence_json(result, metadata),
                    file_name=f"{export_name}_evidence.json",
                    mime="application/json",
                    icon=":material/data_object:",
                    on_click="ignore",
                )
            st.dataframe(
                result.history.tail(120).reset_index()[
                    ["date", "close", "position", "turnover", "strategy_return", "drawdown"]
                ],
                hide_index=True,
                column_config={
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                    "close": st.column_config.NumberColumn("Close", format="%.2f"),
                    "position": st.column_config.NumberColumn("Position", format="%.0f"),
                    "turnover": st.column_config.NumberColumn("Turnover", format="%.0f"),
                    "strategy_return": st.column_config.NumberColumn(
                        "Strategy return", format="percent"
                    ),
                    "drawdown": st.column_config.NumberColumn("Drawdown", format="percent"),
                },
                key=f"evidence_table_{market_key}",
            )
