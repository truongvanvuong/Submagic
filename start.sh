#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "      SUB MAGIC SYSTEM LAUNCHER (100% WSL MODE)"
echo "========================================================"
echo ""

# 1. Python Backend Virtual Environment for Linux
cd "$DIR/backend"
if [ ! -d ".venv-linux" ]; then
    echo "[INFO] Creating Python virtual environment in WSL (.venv-linux)..."
    python3 -m venv .venv-linux
    echo "[INFO] Installing backend dependencies from requirements.txt..."
    .venv-linux/bin/pip install --upgrade pip
    .venv-linux/bin/pip install -r requirements.txt
fi

PY_BIN=".venv-linux/bin/python"
PIP_BIN=".venv-linux/bin/pip"

# Fix OpenCV GUI conflict on headless WSL (rapidocr-onnxruntime pulls opencv-python which requires libQt5Core)
if compgen -G ".venv-linux/lib/python*/site-packages/opencv_python-*.dist-info" > /dev/null; then
    echo "[INFO] Fixing OpenCV headless build for WSL (removing GUI Qt5 dependency)..."
    $PIP_BIN uninstall -y opencv-python opencv-python-headless
    $PIP_BIN install opencv-python-headless
fi

kill_port() {
    local port=$1
    # 1. Kill WSL processes listening on this port
    fuser -k -9 "${port}/tcp" 2>/dev/null || true
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:"$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    fi
    # 2. Kill any lingering Vite processes
    if [ "$port" = "5173" ]; then
        pkill -9 -f "vite" 2>/dev/null || true
    fi
    # 3. Kill Windows processes listening on this port ONLY if they are native Windows node/python (NEVER kill wslrelay.exe!)
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | ForEach-Object { \$p = Get-Process -Id \$_.OwningProcess -ErrorAction SilentlyContinue; if (\$p -and \$p.ProcessName -in @('node','python')) { Stop-Process -Id \$p.Id -Force -ErrorAction SilentlyContinue } }" 2>/dev/null || true
    fi
}

echo "[INFO] Cleaning up any old process on port 8001..."
kill_port 8001

echo "[INFO] Starting Backend on http://0.0.0.0:8001 (Auto-Reload Polling Enabled for WSL /mnt/d) ..."
if [ -z "${HF_HOME:-}" ] && [ -d "$DIR/../AI_Models/huggingface" ]; then
    export HF_HOME="$DIR/../AI_Models/huggingface"
fi
export WATCHFILES_FORCE_POLLING=true
export WATCHFILES_POLL_DELAY_MS=300
export CHOKIDAR_USEPOLLING=true
$PY_BIN -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload --reload-dir app &
BACKEND_PID=$!

cleanup() {
    echo ""
    echo "[INFO] Stopping Sub Magic..."
    kill $BACKEND_PID 2>/dev/null || true
    kill_port 8001
    kill_port 3000
    kill_port 5173
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 2. Frontend Dependencies & Launch
cd "$DIR/frontend"
echo "[INFO] Cleaning up any old process on port 3000..."
kill_port 3000
kill_port 5173
if [ ! -d "node_modules" ] || [ ! -d "node_modules/picomatch" ]; then
    echo "[INFO] Installing/repairing frontend packages (npm install)..."
    npm install --legacy-peer-deps
fi

if [ ! -d "node_modules/@rollup/rollup-linux-x64-gnu" ]; then
    echo "[INFO] Installing missing Linux Rollup binary (@rollup/rollup-linux-x64-gnu)..."
    npm install @rollup/rollup-linux-x64-gnu@4.52.3 --no-save --legacy-peer-deps
fi

WSL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
TARGET_URL="http://localhost:3000"
if [ -n "$WSL_IP" ] && command -v powershell.exe >/dev/null 2>&1; then
    RELAY_ALIVE="$(powershell.exe -NoProfile -Command "if (Get-Process wslrelay -ErrorAction SilentlyContinue) { '1' } else { '0' }" 2>/dev/null | tr -d '\r\n')"
    if [ "$RELAY_ALIVE" = "0" ]; then
        TARGET_URL="http://${WSL_IP}:3000"
        echo "[WARN] Windows wslrelay.exe is not running; using direct WSL2 IP: $TARGET_URL"
    fi
fi

# Open Windows browser after Vite has started
(sleep 2 && cmd.exe /c "start $TARGET_URL" 2>/dev/null) &

echo "[INFO] Starting Frontend on $TARGET_URL (WSL IP: http://${WSL_IP:-localhost}:3000) ..."
npm run dev -- --host 0.0.0.0
