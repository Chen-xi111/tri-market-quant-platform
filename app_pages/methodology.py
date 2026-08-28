"""Research standards and limitations."""

import streamlit as st

st.title("Research methodology")
st.caption(
    "The project is designed to make assumptions inspectable, not to advertise a headline return."
)

st.subheader("Validation contract")
st.markdown(
    """
1. Historical observations are ordered in time; future rows are never used to fit earlier decisions.
2. Fees, slippage, lot sizes and position limits are applied inside the backtest engine.
3. Parameter scans report stability across a grid instead of keeping only the best run.
4. Readiness gates distinguish an interesting backtest from a strategy suitable for paper monitoring.
5. Data-source failures and stale quotes are shown to the user instead of silently treated as live data.
"""
)

st.subheader("Deliberate safety boundaries")
st.markdown(
    """
- The built-in paper broker is the default execution path.
- The IBKR adapter accepts paper ports by default and rejects live ports at the safety boundary.
- Futu OpenD is optional and is not started by the platform launcher.
- Market scanners persist snapshots locally; they do not submit orders.
"""
)

st.subheader("Known limitations")
st.markdown(
    """
- Public data sources may be delayed, rate-limited or revised.
- The current systems focus on daily and snapshot research, not exchange-grade event replay.
- Strategy families share common technical inputs, so diversification must not be inferred from strategy count.
- Backtest results are hypothesis-generation evidence and require independent paper observation.
"""
)
