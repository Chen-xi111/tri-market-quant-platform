#!/bin/bash
# Run all three market test suites in isolated working directories.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Project environment not found. Run: uv sync --extra dev\n' >&2
    exit 1
fi

exec "$PYTHON_BIN" "$PROJECT_DIR/scripts/test_all.py" "$@"
