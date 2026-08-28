# Architecture

## Runtime topology

```text
Browser
  |
  +-- :8500 unified portal
        |-- health + iframe --> :8501 US service
        |-- health + iframe --> :8502 Hong Kong service
        +-- health + iframe --> :8503 China A-share service
```

`scripts/run_all.py` owns the four child processes, waits for Streamlit health endpoints and terminates every service when the parent receives `SIGINT` or `SIGTERM`. Logs are written to the ignored `.runtime/logs/` directory.

## Why process isolation is intentional

The original market applications use short top-level module names such as `config`, `market_data`, `strategies` and `execution`. Importing three versions into one Python interpreter would create ambiguous module resolution and shared Streamlit state. Separate processes provide hard boundaries for:

- venue configuration and symbol normalization;
- market calendars, currencies, fees and lot sizes;
- broker connections and paper accounts;
- Streamlit cache and session state;
- local scan and market-data caches.

The portal is therefore an orchestration layer, not a source-code concatenation layer.

## Service layers

Each market service follows the same conceptual flow:

1. `data_sources/` adapts external providers to a standard OHLCV/quote shape.
2. `market_data.py` manages provider selection and fallback behavior.
3. `strategies/` produces explainable signals from normalized observations.
4. `backtest/` applies fills, costs, constraints, metrics and robustness checks.
5. `risk/` converts signals into bounded position sizes.
6. `execution/` keeps paper and optional broker adapters behind safety gates.
7. Streamlit pages present research, monitoring and scanner state.

Market-wide discovery is deliberately venue-specific: `fund_flow/` for A-shares and `market_scan/` for Hong Kong.
