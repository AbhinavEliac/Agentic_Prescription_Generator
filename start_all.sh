#!/usr/bin/env bash

# Resolve project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " 🚀 Launching RxAgent Full Stack System (Linux / Ubuntu)"
echo "========================================================"

# Ensure child scripts are executable
chmod +x "$SCRIPT_DIR/start_backend.sh" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/start_frontend.sh" 2>/dev/null || true

# If running on Ubuntu Desktop with GNOME terminal, launch in separate windows
if [ -n "$DISPLAY" ] && command -v gnome-terminal >/dev/null 2>&1; then
    echo "Detected GNOME Desktop environment. Launching in separate terminal tabs..."
    gnome-terminal --title="RxAgent Backend (:8080)" -- bash -c "\"$SCRIPT_DIR/start_backend.sh\"; exec bash" &
    sleep 2
    gnome-terminal --title="RxAgent Frontend (:5000)" -- bash -c "\"$SCRIPT_DIR/start_frontend.sh\"; exec bash" &
    echo ""
    echo "Both servers launched in separate windows!"
    echo "👉 Open your browser at: http://localhost:5000"
    exit 0
fi

# Otherwise (SSH, headless server, WSL, tmux), run unified in background with trap cleanup
echo "Starting FastAPI Backend (Port 8080)..."
"$SCRIPT_DIR/start_backend.sh" &
BACKEND_PID=$!

sleep 2

echo "Starting Node.js Frontend (Port 5000)..."
"$SCRIPT_DIR/start_frontend.sh" &
FRONTEND_PID=$!

cleanup() {
    echo ""
    echo "========================================================"
    echo " 🛑 Stopping RxAgent services (PID: $BACKEND_PID, $FRONTEND_PID)..."
    echo "========================================================"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "All services stopped cleanly."
    exit 0
}

# Trap termination signals to cleanly shut down both servers on Ctrl+C
trap cleanup SIGINT SIGTERM EXIT

echo ""
echo "========================================================"
echo " ✅ Both servers are running!"
echo " 🌐 Frontend Studio : http://localhost:5000"
echo " 🔗 Backend API     : http://localhost:8080"
echo " Press [Ctrl+C] to stop all servers."
echo "========================================================"
echo ""

wait
