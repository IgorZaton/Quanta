# TODO

- [x] Design and implement lightweight cache storage:
  - Reduce per-run artifact footprint.
  - Keep only essential summaries by default.
  - Keep full distributions in active tmp run cache and store optionally in finalized runs.

- [x] Add "load from run" support:
  - CLI option to start server from an existing run directory (without recomputation).
  - UI indicator showing active run id/path.
  - Validation for missing/incompatible artifact files.

- [ ] Make project installable:
  - Package backend for pip install (`pyproject.toml` with console script entry point).
  - Document local editable install and production install flows.
  - Verify clean install on a fresh environment.

- [ ] Add dataset analysis section:
  - Summarize dataset shape, feature types, missingness, and basic stats.
  - Include distribution plots/correlation highlights in generated artifacts.
  - Expose dataset analysis in docs/README and (if available) UI.

- [ ] Add deployable model export from UI state:
  - Export current quantization config (per-layer precision + range mode + calibration settings) to ONNX.
  - Apply quantization during export using provided dataset calibration artifacts.
  - Emit export report with applied/fallback precisions, accuracy drift, and latency summary.

- [ ] Introduce ONNX as Quanta common internal type:
  - Define a stable ONNX-centered internal model contract plus source-to-ONNX name mapping.
  - Keep artifact/API schema stable while moving framework loaders (TF/PT) behind the common ONNX contract.
  - Add conversion fidelity checks (shape/op/name mapping) and failure policies for unsupported graphs.

- [ ] Remove legacy-library conversion fallbacks and support latest libs natively:
  - Replace onnx2keras/tf_keras fallback path with a latest-version compatible ONNX->TF bridge.
  - Eliminate unsafe-deserialization/legacy toggles once modern conversion path is stable.
  - Keep PT->ONNX->TF conversion reliability and graph fidelity with current library versions.

- [x] Add CI/CD:
  - CI workflow for lint/test/build on pull requests and main branch.
  - Cache dependencies and publish test artifacts/reports.
  - CD step for tagged releases (package publish and/or deployment).
