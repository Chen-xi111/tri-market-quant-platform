# Tri-market quantitative research platform

A unified, research-first workspace for China A-shares, Hong Kong equities and US equities. The repository combines market data fallbacks, seven explainable strategy families, time-aware backtesting, robustness checks, risk controls, market scanners and paper-execution adapters behind one local portal.

> This is an educational research project, not investment advice. Public market data can be delayed or incomplete. The unified launcher does not enable live trading.

## Why this project exists

The goal is not to present a single optimized equity curve. It is to make the complete research chain inspectable:

```text
market data -> feature/signal generation -> time-ordered validation
            -> cost and risk model -> paper execution -> monitoring
```

Each market service remains process-isolated because tickers, sessions, lot rules, fee schedules, data adapters and broker state differ across venues. The portal supplies one entry point without allowing one market's module or session state to leak into another.

## Market coverage

| Workspace | Market-specific capabilities | Default port |
|---|---|---:|
| China A-shares | Fund-flow radar, daily opportunity scoring, A-share fee/T+1 constraints, quant-research evidence page | 8503 |
| Hong Kong | Market-wide scanner, opportunity scoring, board-lot/fee rules, limited-history safeguards | 8502 |
| US equities | Cross-sectional ranking, quote-freshness chain, IBKR paper safety boundary, strategy tournament | 8501 |
| Unified portal | Health overview, embedded workspaces and research methodology | 8500 |

Shared strategy families include trend riding, moving-average crossover, Bollinger mean reversion, volume breakout, multi-timeframe confirmation, multi-factor alpha and an ML classifier. The strategy count should not be interpreted as independent diversification.

## Quick start

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
./run_all.sh
```

Open [http://localhost:8500](http://localhost:8500). One `Control-C` stops the portal and all three market services.

To run only one market:

```bash
./run_cn.sh     # A-shares, port 8503
./run_hk.sh     # Hong Kong, port 8502
./run_us.sh     # US equities, port 8501
```

Optional Futu OpenD support is installed separately:

```bash
uv sync --extra dev --extra brokers
```

The IBKR TWS Python API is distributed by Interactive Brokers rather than through supported PyPI releases; see [IBKR_SETUP.md](IBKR_SETUP.md).

## Tests

The three services intentionally retain isolated import roots. Run the complete suite through the orchestrator:

```bash
./test_all.sh
```

CI executes the same market suites independently. Tests cover backtest accounting and realism, parameter-grid validation, quote freshness, market clocks, data-source switching, watchlists, venue trading rules, scanners and broker safety.

## Reproducibility and research discipline

- Historical observations are split in time order; the UI exposes train/validation comparisons.
- Commission, slippage, lot sizes, minimum holding rules and position limits are part of the engine.
- Parameter scans report stability across grids rather than retaining only a best result.
- Quote timestamps and fallback failures remain visible instead of being silently labelled live.
- Paper trading is the default. The IBKR adapter rejects live ports unless its code-level safety boundary is explicitly changed.

See [docs/VALIDATION.md](docs/VALIDATION.md) for the evidence contract and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system boundaries.

## Docker

```bash
docker compose up --build
```

The container exposes ports 8500–8503. The browser still enters through the portal at port 8500.

## Repository layout

```text
.
├── streamlit_app.py          # unified portal
├── app_pages/                # portal pages
├── platform_hub.py           # service manifest and health checks
├── scripts/                  # process and test orchestration
├── systems/
│   ├── cn/                   # China A-share service
│   ├── hk/                   # Hong Kong service
│   └── us/                   # US service
├── docs/                     # architecture and validation notes
├── pyproject.toml            # reproducible dependencies
└── Dockerfile
```

## Current limitations

- Public data sources are not point-in-time institutional datasets and may revise history.
- Daily/snapshot research does not model exchange queue priority or tick-level market impact.
- Scheduled scanners run while their corresponding Streamlit process is alive; production scheduling would require a separate durable worker.
- Broker connectivity is environment-dependent and is not exercised in public CI.

These limitations are deliberate disclosures. Backtest output should be treated as a research result that still requires independent paper observation.
