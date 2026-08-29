#!/bin/bash
# Start the single-process, read-only portfolio demo on port 8500.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Project environment not found. Run: uv sync --extra dev\n' >&2
    exit 1
fi

export QUANT_PUBLIC_MODE=true
exec "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_all.py" "$@"
