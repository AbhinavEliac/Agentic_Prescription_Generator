#!/usr/bin/env bash
set -e

# Resolve absolute directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/rx_node_app"

echo "========================================================"
echo " Starting RxAgent Node.js Studio App on http://localhost:5000..."
echo "========================================================"

# Auto-detect Node.js executable
NODE_BIN=""
if command -v node >/dev/null 2>&1; then
    NODE_BIN="node"
elif command -v nodejs >/dev/null 2>&1; then
    NODE_BIN="nodejs"
else
    echo "❌ Error: Node.js not found. Please install Node.js (sudo apt install nodejs npm)."
    exit 1
fi

echo "[Frontend] Using Node: $($NODE_BIN --version 2>&1)"

# Check if node_modules exists; if not, auto-install
if [ ! -d "node_modules" ]; then
    echo "[Frontend] First-time setup: Installing npm dependencies..."
    npm install
fi

echo "[Frontend] Launching Express web server on port 5000..."
exec "$NODE_BIN" server.js
