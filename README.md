<h1 align="center">
  <img src="frontend/public/quanta-manta-icon.svg" alt="Quanta manta icon" width="42" style="vertical-align: middle; margin-right: 10px;" />
  <span style="vertical-align: middle;">Quanta</span>
</h1>

Quanta is a Keras-first quantization visualization tool.

It runs a min/max observer pipeline, builds a fake INT8 quantized path (with dequantization for fair comparison), computes error metrics, and serves an interactive graph UI with per-layer violin plots for weights and activations.

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm (for building/running frontend)

## Install

### Backend

```bash
cd quanta/backend
pip install -e .
```

### Frontend (optional but recommended)

```bash
cd quanta/frontend
npm install
npm run build
```

If frontend is not built, the backend still starts and serves API endpoints at `/api/*`.

## Run

```bash
cd quanta/backend
quanta --model /path/to/model.keras --dataset /path/to/dataset.npy
```

Useful flags:

- `--host 127.0.0.1`
- `--port 8000`
- `--batch-size 32`
- `--max-samples 512`
- `--test` (use bundled MNIST CNN + MNIST dataset)
- `--cpu-only`
- `--no-ui`

## Built-in Test Mode

`--test` uses assets bundled with the project/package:

- `quanta/backend/quanta/assets/mnist_cnn.keras`
- `quanta/backend/quanta/assets/mnist_test.npy`

Example:

```bash
cd quanta/backend
quanta --test --cpu-only --max-samples 64 --batch-size 8
```

## Regenerate MNIST test assets

If you want to retrain and regenerate bundled test assets:

```bash
cd quanta/backend
python scripts/train_mnist_cnn.py
```

## Artifacts

Each run writes artifacts to `.quanta/run_YYYYMMDD_HHMMSS`:

- `graph.json`
- `metrics.json`
- `distributions.json`
- `qparams.json`

## Dev UI mode

Run backend and frontend separately:

```bash
# terminal 1
cd quanta/backend
quanta --model /path/to/model.keras --dataset /path/to/dataset.npy --no-ui

# terminal 2
cd quanta/frontend
npm run dev
```

Vite proxy is configured so frontend `/api` calls go to `http://127.0.0.1:8000`.
