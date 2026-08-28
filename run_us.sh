#!/bin/bash
# 独立启动美股系统。默认端口 8501，可用第一个参数覆盖。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$PROJECT_DIR/systems/us"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
PORT="${1:-8501}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
    printf '端口无效：%s（请输入 1024–65535）\n' "$PORT" >&2
    exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '未找到项目 Python 环境：%s\n' "$PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$APP_DIR/dashboard.py" ]]; then
    printf '未找到美股程序：%s\n' "$APP_DIR/dashboard.py" >&2
    exit 1
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    printf '端口 %s 已被占用。若美股页面已经运行，请直接打开 http://localhost:%s；否则可改用 ./run_us.sh 8511\n' "$PORT" "$PORT" >&2
    exit 1
fi

mkdir -p "$APP_DIR/.market_cache" "$APP_DIR/.mpl_cache"
export MPLCONFIGDIR="$APP_DIR/.mpl_cache"
export FONTCONFIG_PATH="$APP_DIR/.mpl_cache"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

printf '启动美股系统 → http://localhost:%s\n' "$PORT"
cd "$APP_DIR"
exec "$PYTHON_BIN" -m streamlit run dashboard.py \
    --server.address=127.0.0.1 \
    --server.port="$PORT" \
    --server.runOnSave=true \
    --browser.gatherUsageStats=false
