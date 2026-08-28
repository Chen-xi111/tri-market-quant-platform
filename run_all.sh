#!/bin/bash
# Start the unified portal and the three isolated market services.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Project environment not found. Run: uv sync --extra dev\n' >&2
    exit 1
fi

exec "$PYTHON_BIN" "$PROJECT_DIR/scripts/run_all.py" "$@"
