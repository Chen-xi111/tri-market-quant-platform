# Changelog

All notable public-facing changes are documented here.

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
