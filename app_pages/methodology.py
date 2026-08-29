"""Research standards and limitations."""

import pandas as pd
import streamlit as st

with st.container(horizontal=True, vertical_alignment="center"):
    st.badge("Inspectable assumptions", color="blue", icon=":material/visibility:")
    st.badge("Chronological holdout", color="green", icon=":material/schedule:")
    st.badge("No profitability promise", color="gray", icon=":material/gavel:")

st.title("Research methodology")
st.caption(
    "The project is designed to make assumptions inspectable, not to advertise a headline return."
)

columns = st.columns(3, vertical_alignment="top")
for column, icon, title, body in [
    (
        columns[0],
        ":material/history:",
        "Time comes first",
        "Training observations precede validation observations. Signals earn returns only on later bars.",
    ),
    (
        columns[1],
        ":material/payments:",
        "Costs stay visible",
        "Turnover, slippage, venue fees, lot sizes and position limits remain part of the result.",
    ),
    (
        columns[2],
        ":material/tune:",
        "Stability beats a peak",
        "Parameter neighborhoods and held-out degradation matter more than the single best run.",
    ),
]:
    with column.container(border=True, height="stretch"):
        st.subheader(f"{icon} {title}")
        st.write(body)

st.subheader("Validation contract")
contract = pd.DataFrame(
    [
        ("Data", "Timestamp order, source identity, missing values and quote age"),
        ("Signal", "Inputs available at decision time; one-bar lag in the public demo"),
        ("Holdout", "Chronological split with no random shuffle"),
        ("Execution", "Turnover costs, venue constraints and paper-first adapters"),
        ("Robustness", "Cost sensitivity and neighboring parameter settings"),
    ],
    columns=["Layer", "Required evidence"],
)
st.dataframe(contract, hide_index=True, key="methodology_contract")

st.subheader("Deliberate safety boundaries")
st.info(
    "Public mode exposes only fixed research snapshots on port 8500. "
    "Local workspaces and broker controls are not rendered.",
    icon=":material/lock:",
)
with st.container(border=True):
    st.markdown(
        """
- The built-in paper broker is the default execution path.
- The IBKR adapter accepts paper ports by default and rejects live ports at its safety boundary.
- Futu OpenD is optional and is never started by the platform launcher.
- Market scanners persist research snapshots; they do not submit orders.
"""
    )

limitations = st.expander(
    "Known limitations",
    icon=":material/warning:",
    on_change="rerun",
)
if limitations.open:
    with limitations:
        st.markdown(
            """
- Public data sources may be delayed, rate-limited or revised.
- Daily and snapshot research cannot reproduce exchange queue priority or tick-level market impact.
- Several strategies share technical inputs; strategy count is not evidence of diversification.
- Backtests generate hypotheses and still require independent paper observation.
"""
        )
