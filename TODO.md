# TODO

- [ ] Design and implement lightweight cache storage:
  - Reduce per-run artifact footprint.
  - Keep only essential summaries by default.
  - Make full distributions optional/on-demand.

- [ ] Add "load from run" support:
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

- [ ] Add CI/CD:
  - CI workflow for lint/test/build on pull requests and main branch.
  - Cache dependencies and publish test artifacts/reports.
  - CD step for tagged releases (package publish and/or deployment).
