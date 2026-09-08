#!/usr/bin/env bash
set -e

# Resolve absolute directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/rx_extractor_app"

echo "========================================================"
echo " Starting Agentic Prescription FastAPI Backend (:8080)..."
echo "========================================================"

# Auto-detect Python executable (venv or system)
PYTHON_BIN=""
if [ -f "env/bin/python" ]; then
    PYTHON_BIN="env/bin/python"
elif [ -f "../env/bin/python" ]; then
    PYTHON_BIN="../env/bin/python"
elif [ -f "$SCRIPT_DIR/env/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "❌ Error: Python not found. Please install Python 3 (sudo apt install python3 python3-venv python3-pip)."
    exit 1
fi

echo "[Backend] Using Python: $($PYTHON_BIN --version 2>&1) at $PYTHON_BIN"

# Verify fastapi is installed
if ! "$PYTHON_BIN" -c "import fastapi" >/dev/null 2>&1; then
    echo "[Setup] Installing required dependencies..."
    if [ -f "../requirements.txt" ]; then
        "$PYTHON_BIN" -m pip install -r ../requirements.txt || "$PYTHON_BIN" -m pip install fastapi uvicorn
    else
        "$PYTHON_BIN" -m pip install fastapi uvicorn
    fi
fi

echo "[Backend] Launching FastAPI server on http://0.0.0.0:8080..."
exec "$PYTHON_BIN" api_server.py
