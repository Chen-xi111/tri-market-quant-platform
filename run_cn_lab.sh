#!/bin/bash
# A 股短周期策略实验分支。默认端口 8504，可用第一个参数覆盖。
set -euo pipefail

WORKTREE_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$WORKTREE_DIR/systems/cn_lab"
PORT="${1:-8504}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
    printf '端口无效：%s（请输入 1024–65535）\n' "$PORT" >&2
    exit 2
fi

PYTHON_BIN="$WORKTREE_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    COMMON_GIT_DIR="$(git -C "$WORKTREE_DIR" rev-parse --git-common-dir)"
    if [[ "$COMMON_GIT_DIR" != /* ]]; then
        COMMON_GIT_DIR="$WORKTREE_DIR/$COMMON_GIT_DIR"
    fi
    MAIN_WORKTREE="$(cd "$(dirname "$COMMON_GIT_DIR")" && pwd)"
    PYTHON_BIN="$MAIN_WORKTREE/.venv/bin/python"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '未找到可用 Python 环境；请先在主项目建立 .venv。\n' >&2
    exit 1
fi
if [[ ! -f "$APP_DIR/streamlit_app.py" ]]; then
    printf '未找到实验平台：%s\n' "$APP_DIR/streamlit_app.py" >&2
    exit 1
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    printf '端口 %s 已被占用。可改用 ./run_cn_lab.sh 8514\n' "$PORT" >&2
    exit 1
fi

mkdir -p "$APP_DIR/.market_cache" "$APP_DIR/.mpl_cache"
export MPLCONFIGDIR="$APP_DIR/.mpl_cache"
export FONTCONFIG_PATH="$APP_DIR/.mpl_cache"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

printf '启动 A 股短周期策略实验室 → http://localhost:%s\n' "$PORT"
cd "$APP_DIR"
exec "$PYTHON_BIN" -m streamlit run streamlit_app.py \
    --server.address=127.0.0.1 \
    --server.port="$PORT" \
    --server.runOnSave=true \
    --browser.gatherUsageStats=false
