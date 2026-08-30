# Portfolio walkthrough

This route is designed for a short code review or quantitative-internship interview.

## One-minute product tour

1. Start `./run_public.sh` and open `http://localhost:8500`.
2. Use the overview to explain the three concerns the project joins: data engineering, research validity and execution safety.
3. Open **Cross-market lab** and show how one rule is evaluated across all nine fixed snapshots with either venue defaults or one common cost assumption.
4. Use the risk-return map and train/validation transfer view to discuss dispersion without pretending that different currencies and calendars form one portfolio return.
5. Download the complete scorecard, then open **US equities**, keep SPY and the trend rule, and run the detailed analysis.
6. Point out the chronological holdout, open **Robustness**, then use **Data and assumptions** to download the deterministic return path and JSON evidence bundle.

## Three-minute technical discussion

### Research design

- Strategies are long-only and deterministic in the public demo.
- A signal observed at the end of bar `t` earns returns only from `t+1`.
- The first 70% of observations are training context; the final 30% are validation.
- Turnover costs are deducted on every position change.
- The UI reports train/validation degradation and all neighboring parameter results.
- Cross-market results preserve instrument-level evidence and never sum or average currencies into a portfolio claim.

### Engineering design

- Demo fixtures are compact, versioned and require no secrets or network access.
- File modification times invalidate in-memory fixture caches automatically.
- Cached frames are copied before use so UI mutations cannot contaminate later runs.
- Dynamic tabs and expanders avoid computing hidden robustness and service-health views.
- Evidence downloads are deterministic and carry fixture hashes, assumptions, diagnostics and metrics for review outside the UI.
- Public Docker mode starts one process; local mode preserves hard process boundaries across markets.

### Safety design

- Public mode is read-only and never renders broker controls.
- Local adapters default to paper execution.
- The IBKR boundary rejects live ports unless the safety code itself is deliberately changed.
- Scanners produce local research snapshots; they do not submit orders.

## Questions the project can answer

- How do you avoid look-ahead bias? See the lagged position construction and chronological split in `portfolio/demo_engine.py`.
- How do you avoid presenting an optimized peak? See the complete parameter grid and evidence classification.
- How do you compare markets responsibly? The comparison lab applies the same rule, exposes both venue/default and common-cost views, and does not pool returns across currencies or calendars.
- What happens when costs rise? The public robustness view recomputes held-out performance at six cost levels.
- Why not merge all three market apps into one interpreter? Their module names, caches, sessions, venue rules and broker state require isolation; only the clean public demo shares one package.
- What would you build next? Point-in-time constituent data, walk-forward retraining, portfolio-level allocation, durable scheduling and paper/live reconciliation.

## Honest limitations to volunteer

The fixtures are public-provider snapshots, not survivorship-free institutional data. Daily bars cannot model queue priority or tick-level impact. A favorable held-out result remains only a hypothesis until it survives independent paper observation.
