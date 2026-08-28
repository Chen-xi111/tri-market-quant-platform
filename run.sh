#!/bin/bash
# 向后兼容：原来的 ./run.sh 继续启动美股系统。
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/run_us.sh" "$@"
