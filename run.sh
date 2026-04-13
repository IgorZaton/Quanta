#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_DIR="$ROOT_DIR/backend"

echo "[quanta] Rebuilding frontend..."
cd "$FRONTEND_DIR"
npm install
npm run build

echo "[quanta] Reinstalling backend package..."
cd "$BACKEND_DIR"
pip install -e .

echo "[quanta] Stopping existing quanta processes (if any)..."
pkill -f "quanta" || true
pkill -f "python -m quanta.cli" || true

echo "[quanta] Starting app..."
if [ "$#" -gt 0 ]; then
  quanta "$@"
else
  quanta --test --test-bad --cpu-only --max-samples 8 --batch-size 16
fi
