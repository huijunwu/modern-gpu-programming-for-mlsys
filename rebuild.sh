#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing http.server on port 8000
echo ">>> Stopping old server..."
pkill -f "python3 -m http.server 8000" 2>/dev/null || true
sleep 1

# Clean old build
rm -rf _build/html_en _build/html_zh

echo ">>> Building English site..."
sphinx-build -b html locale/en _build/html_en
echo "    English build done."

echo ">>> Building Chinese site..."
sphinx-build -b html locale/zh _build/html_zh
echo "    Chinese build done."

cd _build && python3 -m http.server 8000

echo "    Then visit http://localhost:8000"


