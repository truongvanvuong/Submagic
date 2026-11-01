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

echo "[INFO] Cleaning up any old process on port 8000..."
fuser -k 8000/tcp 2>/dev/null || true

echo "[INFO] Starting Backend on http://0.0.0.0:8000 ..."
$PY_BIN -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir app &
BACKEND_PID=$!

cleanup() {
    echo ""
    echo "[INFO] Stopping Sub Magic..."
    kill $BACKEND_PID 2>/dev/null || true
    fuser -k 8000/tcp 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 2. Frontend Dependencies & Launch
cd "$DIR/frontend"
if [ ! -d "node_modules" ] || [ ! -d "node_modules/picomatch" ]; then
    echo "[INFO] Installing/repairing frontend packages (npm install)..."
    npm install --legacy-peer-deps
fi

if [ ! -d "node_modules/@rollup/rollup-linux-x64-gnu" ]; then
    echo "[INFO] Installing missing Linux Rollup binary (@rollup/rollup-linux-x64-gnu)..."
    npm install @rollup/rollup-linux-x64-gnu@4.52.3 --no-save --legacy-peer-deps
fi

# Try opening Windows browser from WSL
cmd.exe /c "start http://localhost:5173" 2>/dev/null || true

echo "[INFO] Starting Frontend on http://localhost:5173 ..."
npm run dev -- --host
