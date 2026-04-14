<h1 align="center">
  <img src="frontend/public/quanta-manta-icon.svg" alt="Quanta manta icon" width="42" style="vertical-align: middle; margin-right: 10px;" />
  <span style="vertical-align: middle;">Quanta</span>
</h1>

Quanta is a Keras-first quantization visualization tool.

It runs a min/max observer pipeline, builds a fake INT8 quantized path (with dequantization for fair comparison), computes error metrics, and serves an interactive graph UI with per-layer violin plots for weights and activations.

[![CI](https://github.com/SigmaConnectivityPl/Quanta/actions/workflows/ci.yml/badge.svg)](https://github.com/SigmaConnectivityPl/Quanta/actions/workflows/ci.yml)

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm (for building/running frontend)

## Install

### Backend

```bash
cd backend
pip install -e .
```

### Frontend (optional but recommended)

```bash
cd frontend
npm install
npm run build
```

If frontend is not built, the backend still starts and serves API endpoints at `/api/*`.

## Build local release artifact

Build a wheel that includes the production frontend bundle:

```bash
./scripts/release.sh
```

This produces installable artifacts in `backend/dist/` (including `.whl`).

## Run

```bash
cd backend
quanta --model /path/to/model.keras --dataset /path/to/dataset.npy
```

Useful flags:

- `--host 127.0.0.1`
- `--port 8000`
- `--batch-size 32`
- `--max-samples 512`
- `--final-artifact-profile minimal|full` (default: `minimal`)
- `--store-distributions` (shortcut for `--final-artifact-profile full`)
- `--max-cached-runs 1`
- `--test` (use bundled MNIST CNN + MNIST dataset)
- `--cpu-only`
- `--no-ui`

## Built-in Test Mode

`--test` uses assets bundled with the project/package:

- `backend/quanta/assets/mnist_cnn.keras`
- `backend/quanta/assets/mnist_test.npy`

Example:

```bash
cd backend
quanta --test --cpu-only --max-samples 64 --batch-size 8
```

## Regenerate MNIST test assets

If you want to retrain and regenerate bundled test assets:

```bash
cd backend
python scripts/train_mnist_cnn.py
```

## Artifacts

Quanta uses a two-phase cache lifecycle:

- Active run cache: `.quanta/tmp_run_<id>/` stores full artifacts while the run is active (UI/API uses this full cache).
- Finalized run cache: `.quanta/run_<id>/` keeps compact artifacts for persisted final state.

By default (`--final-artifact-profile minimal`), finalized runs include:

- `graph.json`
- `metrics.json`
- `qparams.json`
- `estimates.json`
- `run_meta.json`

If you enable `--final-artifact-profile full` (or `--store-distributions`), finalized runs also include `distributions.json`.

Temporary `tmp_run_*` directories are removed when the process exits, and finalized `run_*` directories are retained according to `--max-cached-runs` (default `1`, latest setting only).

Dataset analysis outputs (for richer EDA/statistics) are planned and tracked in `TODO.md`.

## Dev UI mode

Run backend and frontend separately:

```bash
# terminal 1
cd backend
quanta --model /path/to/model.keras --dataset /path/to/dataset.npy --no-ui

# terminal 2
cd frontend
npm run dev
```

Vite proxy is configured so frontend `/api` calls go to `http://127.0.0.1:8000`.

## Scripts

Project-level helper scripts:

- `./scripts/build.sh` rebuilds frontend and reinstalls backend editable package.
- `./scripts/run.sh` does the same setup, stops existing Quanta processes, then starts Quanta (defaults to `--test --test-bad --cpu-only --max-samples 8 --batch-size 16` when no args are passed).
- `./scripts/release.sh` builds frontend, syncs it into package data, builds Python distribution artifacts, verifies the wheel contents, and smoke-tests wheel install.

## CI/CD

GitHub Actions workflows:

- `.github/workflows/ci.yml` runs on every push and pull request:
  - Runs backend tests (`pytest -q backend/tests`).
  - Builds frontend to catch integration/build issues.
- `.github/workflows/release.yml` runs only when a GitHub Release is published:
  - Runs `./scripts/release.sh`.
  - Uploads built artifacts from `backend/dist/*`.

## Pre-commit hooks

Install and enable pre-commit locally:

```bash
python -m pip install pre-commit
pre-commit install
```

Current hooks include formatting/sanity checks and a backend test gate (`pytest -q tests`) that runs before each commit.
The `commit-msg` hook uses Commitizen, so commit messages should follow Conventional Commits (for example: `feat: add dataset analysis section`).
