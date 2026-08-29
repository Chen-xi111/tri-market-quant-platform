# Portfolio walkthrough

This route is designed for a short code review or quantitative-internship interview.

## One-minute product tour

1. Start `./run_public.sh` and open `http://localhost:8500`.
2. Use the overview to explain the three concerns the project joins: data engineering, research validity and execution safety.
3. Open **US equities**, keep SPY and the trend rule, then run the analysis.
4. Point out that the equity curve is split chronologically and that the KPI cards report the held-out period—not the full sample.
5. Open **Robustness** and explain why cost sensitivity and parameter neighborhoods are more credible than a single optimized result.
6. Open **Data and assumptions** to show provenance, one-bar signal lag and the exact observations behind the chart.

## Three-minute technical discussion

### Research design

- Strategies are long-only and deterministic in the public demo.
- A signal observed at the end of bar `t` earns returns only from `t+1`.
- The first 70% of observations are training context; the final 30% are validation.
- Turnover costs are deducted on every position change.
- The UI reports train/validation degradation and all neighboring parameter results.

### Engineering design

- Demo fixtures are compact, versioned and require no secrets or network access.
- File modification times invalidate in-memory fixture caches automatically.
- Cached frames are copied before use so UI mutations cannot contaminate later runs.
- Dynamic tabs and expanders avoid computing hidden robustness and service-health views.
- Public Docker mode starts one process; local mode preserves hard process boundaries across markets.

### Safety design

- Public mode is read-only and never renders broker controls.
- Local adapters default to paper execution.
- The IBKR boundary rejects live ports unless the safety code itself is deliberately changed.
- Scanners produce local research snapshots; they do not submit orders.

## Questions the project can answer

- How do you avoid look-ahead bias? See the lagged position construction and chronological split in `portfolio/demo_engine.py`.
- How do you avoid presenting an optimized peak? See the complete parameter grid and evidence classification.
- What happens when costs rise? The public robustness view recomputes held-out performance at six cost levels.
- Why not merge all three market apps into one interpreter? Their module names, caches, sessions, venue rules and broker state require isolation; only the clean public demo shares one package.
- What would you build next? Point-in-time constituent data, walk-forward retraining, portfolio-level allocation, durable scheduling and paper/live reconciliation.

## Honest limitations to volunteer

The fixtures are public-provider snapshots, not survivorship-free institutional data. Daily bars cannot model queue priority or tick-level impact. A favorable held-out result remains only a hypothesis until it survives independent paper observation.
