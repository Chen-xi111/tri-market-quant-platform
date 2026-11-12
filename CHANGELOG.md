# Changelog

All notable public-facing changes are documented here.

## 0.4.0 — 2026-11-12

### Added

- A current-price decision gate across the China, Hong Kong and US local workspaces.
- Explicit blockers for stale cached quotes, prices beyond the chase limit, invalid stop placement and insufficient reward/risk.
- Session-aware simulated-order sizing that respects the original risk budget, position cap and venue lot size.
- Direct local-workspace entrances on the full launcher home page while preserving the broker-free public view.
- An isolated A-share short-horizon strategy lab with next-session execution, T+1 constraints, conservative same-bar handling and out-of-sample diagnostics.

### Changed

- US cross-sectional ranking now separates research attractiveness from execution readiness and exposes conservative, balanced and opportunity profiles.
- China and Hong Kong opportunity views now surface reward/risk and decision-state fields instead of relying on an aggregate score alone.
- Venue clocks now distinguish executable continuous/regular sessions from quote-only auctions, lunch breaks and extended hours.
- Hong Kong deep links and per-instrument board-lot state no longer reset or leak across symbols.
- The US execution decision card now refreshes with the live quote while signal direction, entry zones and backtests remain anchored to completed daily bars.
- US chart fonts are registered explicitly so Chinese instrument names render without missing-glyph warnings.
- GitHub Actions now lints the cross-market decision, session and ranking modules in addition to the public portal.
- The all-suite runner and GitHub Actions now verify the A-share strategy lab as a first-class module.
- Historical-bar quote fallbacks in the China and Hong Kong workspaces are now explicit no-new-position states.
- Test-count documentation now matches the expanded 191-test suite plus 19 generated subtests.

### Verified

- All portfolio, China, Hong Kong and US test suites pass after the decision-gate integration.
- The A-share strategy lab passes dedicated look-ahead, T+1, execution-order and lot-sizing tests.
- Python 3.11 target parsing, public-release boundaries, fixture checksums and secret-path checks pass.

## 0.3.2 — 2026-08-31

### Changed

- The landing page now uses one focused signature-research card instead of two competing cross-market calls to action.
- The signature card pairs the primary workflow with the exact evidence a reviewer can verify.
- Single-market results now display the submitted instrument, strategy and cost immediately above the KPI row.
- The analysis submit action spans the research form, improving hierarchy on desktop and mobile.
- Cross-market snapshot interpretation is promoted to a clearly labelled descriptive readout.
- Theme borders, radii and semantic badge colors were refined while retaining the accessible financial-blue palette.

### Verified

- Desktop and 390-pixel mobile views render without horizontal overflow or browser-console errors.
- Streamlit page tests now protect the consolidated landing-page actions from accidental duplication.

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
