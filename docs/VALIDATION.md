# Validation contract

This document defines what the project can and cannot claim from a backtest.

## Evidence required before a strategy is considered research-ready

1. The input data has enough history for every indicator and is ordered by timestamp.
2. Training observations precede validation observations.
3. Costs, slippage, market lot sizes and venue holding restrictions are applied.
4. The result is compared with a buy-and-hold baseline.
5. The validation period has enough completed trades to support interpretation.
6. Parameter neighborhoods are inspected; a single isolated optimum is not sufficient evidence.
7. Maximum drawdown, trade count, profit factor and out-of-sample degradation are reported with returns.

## Readiness gate

The current UI presents a transparent heuristic gate based on minimum trade count, alpha over buy-and-hold, Sharpe ratio, drawdown and profit factor. Passing this gate means only that basic checks passed. It does not authorize live trading and does not replace paper observation.

## Cost and execution assumptions

- Backtests apply per-market commission assumptions and configured slippage.
- China A-share and Hong Kong calculations use their venue-specific fee and lot helpers.
- The daily-bar engine cannot infer intrabar queue order or exchange-level market impact.
- Paper-broker results test system behavior, not real fill quality.

## Data caveats

The project uses public APIs with a fallback chain. Providers can differ in adjustment conventions, timestamp semantics, update latency and symbol coverage. The UI surfaces provider failures and quote ages, but the repository does not claim point-in-time survivorship-free institutional coverage.

## Interpretation

A useful result answers a bounded research question and remains directionally stable under time splits and reasonable cost assumptions. A negative or degraded out-of-sample result is valid evidence when it is reported honestly.
