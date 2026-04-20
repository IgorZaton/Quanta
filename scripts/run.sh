#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_DIR="$ROOT_DIR/backend"

echo "[quanta] Rebuilding frontend..."
cd "$FRONTEND_DIR"
npm install
npm run build

echo "[quanta] Reinstalling backend package..."
USE_POETRY=0
if command -v poetry >/dev/null 2>&1 && [[ -f "$ROOT_DIR/pyproject.toml" ]]; then
  USE_POETRY=1
  cd "$ROOT_DIR"
  poetry install
else
  cd "$BACKEND_DIR"
  pip install -e .
fi

echo "[quanta] Stopping existing quanta processes (if any)..."
pkill -f "quanta" || true
pkill -f "python -m quanta.cli" || true

echo "[quanta] Starting app..."
if [ "$#" -gt 0 ]; then
  if [ "$USE_POETRY" -eq 1 ]; then
    poetry run quanta "$@"
  else
    quanta "$@"
  fi
else
  if [ "$USE_POETRY" -eq 1 ]; then
    poetry run quanta --test --test-bad --cpu-only --max-samples 8 --batch-size 16
  else
    quanta --test --test-bad --cpu-only --max-samples 8 --batch-size 16
  fi
fi
