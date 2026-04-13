# TODO

- [ ] Design and implement lightweight cache storage:
  - Reduce per-run artifact footprint.
  - Keep only essential summaries by default.
  - Make full distributions optional/on-demand.

- [ ] Add "load from run" support:
  - CLI option to start server from an existing run directory (without recomputation).
  - UI indicator showing active run id/path.
  - Validation for missing/incompatible artifact files.
