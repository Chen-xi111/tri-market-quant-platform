# Changelog

All notable public-facing changes are documented here.

## 0.3.1 — 2026-08-30

### Fixed

- Single-market turnover cost is now a visible, submitted research input instead of a stateful control hidden inside a popover.
- Instrument, rule and cost selections persist when navigating between portfolio pages.
- Download file names now include the submitted cost and, for comparison scorecards, the selected market universe.
- Evidence JSON serialization rejects non-standard `NaN` and `Infinity` values.

### Verified

- Desktop and 390-pixel mobile layouts complete without horizontal overflow.
- Public and full four-process modes render successfully, including all three embedded local workspaces.
- Extreme 0/40 bps strategy grids and all public evidence bundles remain finite and machine-readable.

## 0.3.0 — 2026-08-30

### Added

- Cross-market comparison lab for applying one transparent rule to all nine versioned instruments.
- Risk-return, excess-return and train/validation transfer views with market and cost filters.
- Downloadable comparison scorecards plus deterministic per-instrument CSV and JSON evidence bundles.

### Changed

- The landing page now leads with a signature cross-market research workflow before market-level drill-down.
- Responsive controls, compact mobile legends and lazy result tabs improve small-screen use and rendering cost.
- Comparison language now explicitly keeps currencies, calendars and instrument evidence separate instead of presenting a pooled portfolio return.

## 0.2.0 — 2026-08-30

### Added

- Single-process, broker-free public portfolio mode with nine versioned market snapshots.
- Transparent trend, mean-reversion, breakout and multi-factor demo rules.
- Chronological train/validation scorecards, drawdown, cost sensitivity and parameter stability.
- Portfolio landing page, methodology page and interview walkthrough.
- Public-boundary, launcher, interaction and Streamlit page tests.
- Fixture SHA-256 integrity metadata and a public-release preflight command.

### Changed

- Docker now starts and exposes only the read-only portal on port 8500.
- The full local four-process launcher remains available through `./run_all.sh`.
- Navigation, theme, terminology, loading feedback and mobile behavior were unified.
- Demo data and research results use bounded, mutation-safe caches.

### Security

- Public mode no longer renders child workspaces or broker controls.
- Live-order access is disabled by default and local execution stays paper-first.

## 0.1.0 — 2026-08-28

- Initial unified launcher for the three isolated market research services.
- Shared portal, Docker packaging, architecture notes and market-suite CI.
