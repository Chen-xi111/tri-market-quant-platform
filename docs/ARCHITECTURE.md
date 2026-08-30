# Architecture

## Public portfolio topology

```text
Browser
  |
  +-- :8500 Streamlit portal (one process)
        |-- project overview and methodology
        |-- cross-market comparison lab
        |-- China / Hong Kong / US demo pages
        +-- versioned fixtures -> shared demo engine -> comparison and evidence views
```

`QUANT_PUBLIC_MODE=true` makes `scripts/run_all.py` start only the portal. The portal reads repository fixtures, never renders child-service iframes and contains no path to broker controls. Docker uses this mode and exposes only port 8500.

The comparison layer sends the same strategy definition through the shared engine for each selected snapshot. It aligns the resulting metric schema for visual comparison but retains each instrument's native currency, calendar, costs and fixture provenance. The output is a research scorecard, not a pooled portfolio backtest.

## Full local topology

```text
Browser
  |
  +-- :8500 unified portal
        |-- health + iframe --> :8501 US service
        |-- health + iframe --> :8502 Hong Kong service
        +-- health + iframe --> :8503 China A-share service
```

`./run_all.sh` explicitly uses local mode. The launcher owns the services it starts, waits for Streamlit health endpoints and terminates them on `SIGINT` or `SIGTERM`. Logs are written to the ignored `.runtime/logs/` directory. A healthy child process that was already running can be reused and is not terminated by the launcher.

## Why local process isolation is intentional

The original market applications use short top-level module names such as `config`, `market_data`, `strategies` and `execution`. Importing three versions into one Python interpreter would create ambiguous module resolution and shared Streamlit state. Separate processes provide hard boundaries for:

- venue configuration and symbol normalization;
- market calendars, currencies, fees and lot sizes;
- broker connections and paper accounts;
- Streamlit cache and session state;
- local scan and market-data caches.

The portal is therefore an orchestration layer, not a source-code concatenation layer. The public demo can remain single-process because its compact engine has one shared package and no broker/provider runtime state.

## Service layers

Each full market service follows the same conceptual flow:

1. `data_sources/` adapts external providers to a standard OHLCV/quote shape.
2. `market_data.py` manages provider selection and fallback behavior.
3. `strategies/` produces explainable signals from normalized observations.
4. `backtest/` applies fills, costs, constraints, metrics and robustness checks.
5. `risk/` converts signals into bounded position sizes.
6. `execution/` keeps paper and optional broker adapters behind safety gates.
7. Streamlit pages present research, monitoring and scanner state.

Market-wide discovery remains venue-specific: `fund_flow/` for A-shares and `market_scan/` for Hong Kong.
