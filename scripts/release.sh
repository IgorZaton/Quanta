#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_DIR="$ROOT_DIR/backend"
WEB_DIR="$BACKEND_DIR/quanta/web"
DIST_DIR="$BACKEND_DIR/dist"
RELEASE_VERSION="${RELEASE_VERSION:-}"

echo "[release] Building frontend bundle..."
cd "$FRONTEND_DIR"
npm ci
npm run build

if [[ ! -f "$FRONTEND_DIR/dist/index.html" ]]; then
  echo "[release] ERROR: frontend build output is missing dist/index.html" >&2
  exit 1
fi

echo "[release] Syncing frontend bundle into backend package data..."
mkdir -p "$WEB_DIR"
find "$WEB_DIR" -mindepth 1 ! -name "__init__.py" -exec rm -rf {} +
cp -R "$FRONTEND_DIR/dist/." "$WEB_DIR/"
if [[ ! -f "$WEB_DIR/__init__.py" ]]; then
  touch "$WEB_DIR/__init__.py"
fi

echo "[release] Building Python distribution artifacts..."
cd "$BACKEND_DIR"
python -m pip install --upgrade pip build
rm -rf "$DIST_DIR"

if [[ -n "$RELEASE_VERSION" ]]; then
  if [[ "$RELEASE_VERSION" =~ ^v ]]; then
    RELEASE_VERSION="${RELEASE_VERSION#v}"
  fi
  echo "[release] Setting backend/pyproject.toml version to $RELEASE_VERSION"
  python - "$RELEASE_VERSION" <<'PY'
from pathlib import Path
import re
import sys

version = sys.argv[1]
if not re.fullmatch(r"\d+\.\d+\.\d+([a-zA-Z0-9.\-+]*)?", version):
    raise SystemExit(f"Invalid RELEASE_VERSION: {version}")

pyproject = Path("pyproject.toml")
text = pyproject.read_text()
updated, count = re.subn(
    r'(?m)^version\s*=\s*".*"$',
    f'version = "{version}"',
    text,
    count=1,
)
if count != 1:
    raise SystemExit("Failed to update version in pyproject.toml")
pyproject.write_text(updated)
PY
fi

python -m build

echo "[release] Verifying wheel has bundled frontend..."
python - <<'PY'
from pathlib import Path
import zipfile

dist_dir = Path("dist")
wheels = sorted(dist_dir.glob("*.whl"))
if not wheels:
    raise SystemExit("No wheel was created in backend/dist")

wheel = wheels[-1]
required = {
    "quanta/web/index.html",
}
with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
missing = [name for name in required if name not in names]
if missing:
    raise SystemExit(f"Wheel {wheel.name} is missing bundled files: {missing}")

print(f"Verified wheel: {wheel}")
PY

echo "[release] Smoke-testing install from wheel..."
TEMP_DIR="$(mktemp -d)"
python -m venv "$TEMP_DIR/venv"
"$TEMP_DIR/venv/bin/pip" install --upgrade pip
"$TEMP_DIR/venv/bin/pip" install "$DIST_DIR"/*.whl
"$TEMP_DIR/venv/bin/quanta" --help >/dev/null
rm -rf "$TEMP_DIR"

echo "[release] Success. Artifacts are in $DIST_DIR"
