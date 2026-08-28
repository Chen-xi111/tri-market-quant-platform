#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
exec "$PROJECT_DIR/run_cn.sh" "$@"
