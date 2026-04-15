from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _strip_redundant_overrides(
    overrides: dict[str, str], global_mode: str
) -> dict[str, str]:
    """Remove per-layer entries that match the global mode (they're no-ops)."""
    return {k: v for k, v in overrides.items() if v != global_mode}


def normalize_strategy_key(
    range_mode: str,
    weight_mode: str,
    activation_mode: str,
    layer_weight_modes: dict[str, str],
    layer_activation_modes: dict[str, str],
) -> str:
    lw = _strip_redundant_overrides(layer_weight_modes, weight_mode)
    la = _strip_redundant_overrides(layer_activation_modes, activation_mode)
    return (
        f"{range_mode}|{weight_mode}|{activation_mode}|"
        f"{json.dumps(lw, sort_keys=True, separators=(',', ':'))}|"
        f"{json.dumps(la, sort_keys=True, separators=(',', ':'))}"
    )


def strategy_key_from_state(state: dict[str, Any]) -> str:
    return normalize_strategy_key(
        state.get("range_mode", "minmax"),
        state.get("global_weight_mode", "int8"),
        state.get("global_activation_mode", "int8"),
        state.get("layer_weight_modes", {}),
        state.get("layer_activation_modes", {}),
    )


class StrategyCache:
    def __init__(self, output_root: Path, run_id: str | None = None) -> None:
        self._output_root = output_root
        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._strategies: dict[str, Path] = {}
        self._latest_state: dict[str, Any] = {}
        self._source_config: dict[str, Any] = {}
        self._cache_path = output_root / "strategy_cache.json"
        self._started_at = datetime.utcnow().isoformat() + "Z"
        self._finalized = False

    # -- read helpers ----------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def strategy_keys(self) -> list[str]:
        return list(self._strategies.keys())

    @property
    def finalized(self) -> bool:
        return self._finalized

    def get_strategy_root(self, key: str) -> Path | None:
        root = self._strategies.get(key)
        if root is not None and root.exists():
            return root
        if root is not None:
            del self._strategies[key]
            self._save()
        return None

    def get_latest_state(self) -> dict[str, Any]:
        return dict(self._latest_state)

    def get_source_config(self) -> dict[str, Any]:
        return dict(self._source_config)

    # -- write helpers ---------------------------------------------------------

    def upsert_strategy(self, key: str, artifact_dir: Path) -> None:
        self._strategies[key] = artifact_dir
        self._save()

    def set_latest_state(self, state: dict[str, Any]) -> None:
        self._latest_state = state
        self._save()

    def set_source_config(self, config: dict[str, Any]) -> None:
        self._source_config = config
        self._save()

    # -- persistence -----------------------------------------------------------

    def _save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "strategies": {k: str(v) for k, v in self._strategies.items()},
            "latest_state": self._latest_state,
            "source_config": self._source_config,
            "started_at": self._started_at,
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, cache_path: Path) -> StrategyCache:
        data = json.loads(cache_path.read_text())
        cache = cls(cache_path.parent, run_id=data.get("run_id"))
        for key, path_str in data.get("strategies", {}).items():
            cache._strategies[key] = Path(path_str)
        cache._latest_state = data.get("latest_state", {})
        cache._source_config = data.get("source_config", {})
        cache._started_at = data.get("started_at", cache._started_at)
        return cache

    @classmethod
    def from_finalized_run(cls, run_dir: Path) -> StrategyCache:
        """Restore a cache from a finalized run directory for --load-run mode.

        The cache starts un-finalized so the session is interactive.
        """
        run_meta_path = run_dir / "run_meta.json"
        run_meta: dict[str, Any] = {}
        if run_meta_path.exists():
            run_meta = json.loads(run_meta_path.read_text())
        cache = cls(run_dir.parent, run_id=run_meta.get("run_id"))
        latest_state = run_meta.get("latest_state", {})
        cache._latest_state = latest_state
        cache._source_config = run_meta.get("source_config", {})
        key = strategy_key_from_state(latest_state)
        cache._strategies[key] = run_dir
        cache._started_at = run_meta.get("started_at", cache._started_at)
        return cache

    # -- finalize --------------------------------------------------------------

    def materialize_finalized_snapshot(self) -> Path:
        """Copy latest strategy artifacts into a finalized run directory."""
        latest_key = strategy_key_from_state(self._latest_state)
        latest_root = self._strategies.get(latest_key)
        if latest_root is None or not latest_root.exists():
            raise ValueError(f"No cached artifacts for latest strategy: {latest_key}")

        final_dir = self._output_root / f"run_{self._run_id}"
        final_dir.mkdir(parents=True, exist_ok=True)

        artifact_names = (
            "graph.json",
            "metrics.json",
            "qparams.json",
            "estimates.json",
            "distributions.json",
        )
        copied: list[str] = []
        for name in artifact_names:
            src = latest_root / name
            if src.exists():
                (final_dir / name).write_text(src.read_text())
                copied.append(name)

        run_meta: dict[str, Any] = {
            "status": "complete",
            "run_id": self._run_id,
            "profile": "full",
            "started_at": self._started_at,
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "artifacts": copied,
            "latest_state": self._latest_state,
            "source_config": self._source_config,
        }
        (final_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
        self._finalized = True
        self._cleanup_tmp_dirs()
        return final_dir

    def _cleanup_tmp_dirs(self) -> None:
        """Remove all tmp_run_* directories and the cache file under output_root."""
        if not self._output_root.exists():
            return
        for tmp_dir in self._output_root.glob("tmp_run_*"):
            if tmp_dir.is_dir():
                shutil.rmtree(tmp_dir, ignore_errors=True)
        if self._cache_path.exists():
            self._cache_path.unlink(missing_ok=True)
