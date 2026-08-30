# Contributing

Contributions should preserve the project's research transparency and public safety boundaries.

## Development setup

```bash
uv sync --extra dev
./run_public.sh
```

Use `./run_all.sh` only when work requires the three isolated local market services. Optional broker dependencies must not become required by the public demo.

## Before submitting a change

```bash
python scripts/release_check.py
./test_all.sh
python -m ruff check streamlit_app.py platform_hub.py app_pages portfolio scripts
```

Please keep changes focused and include tests for research accounting, public boundaries or user interactions when they change.

## Research requirements

- Do not use future information to fit or execute earlier decisions.
- Keep transaction costs, signal timing and data provenance visible.
- Report complete parameter neighborhoods rather than only a selected optimum.
- Treat favorable results as hypotheses; do not add profitability claims.

## Safety requirements

- Public mode must remain read-only, offline-capable and broker-free.
- New broker integrations must default to paper environments and fail closed.
- Never commit credentials, local account state, provider caches or broker logs.
