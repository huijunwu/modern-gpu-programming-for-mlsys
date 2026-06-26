#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8000
LOG="/tmp/modern-gpu-programming-for-mlsys.log"

# Kill any existing http.server on the port
echo ">>> Stopping old server..."
pkill -f "python3 -m http.server ${PORT}" 2>/dev/null || true
sleep 1

# Clean old build
rm -rf _build/html_en _build/html_zh

echo ">>> Building English site..."
sphinx-build -b html locale/en _build/html_en
echo "    English build done."

echo ">>> Building Chinese site..."
sphinx-build -b html locale/zh _build/html_zh
echo "    Chinese build done."

cd _build && python3 -m http.server "${PORT}" > "${LOG}" 2>&1 &

echo "    Server started (PID $!) on port ${PORT}. Log: ${LOG}"
echo "    Visit http://localhost:${PORT}"
