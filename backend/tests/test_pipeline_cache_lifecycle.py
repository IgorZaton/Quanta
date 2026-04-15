from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from quanta.model_ir import LayerMeta, UnifiedModel
from quanta import cli
from quanta.pipeline import PipelineConfig, _prune_previous_runs, run_pipeline
from quanta.server import create_app
from quanta.strategy_cache import (
    StrategyCache,
    normalize_strategy_key,
    strategy_key_from_state,
)


class _DummyModel:
    input_shape = (None, 2)


class _DummyDatasetAdapter:
    def __init__(
        self, dataset_path: str, batch_size: int, max_samples: int | None
    ) -> None:
        del dataset_path, batch_size, max_samples
        self._data = np.zeros((4, 2), dtype=np.float32)

    def load(self) -> np.ndarray:
        return self._data

    def iter_batches(self):
        yield self._data


class PipelineCacheLifecycleTest(unittest.TestCase):
    def _fake_unified(self) -> UnifiedModel:
        return UnifiedModel(
            model=_DummyModel(),  # type: ignore[arg-type]
            metadata=[
                LayerMeta(
                    name="dummy",
                    layer_type="Dense",
                    input_shape=(None, 2),
                    output_shape=(None, 2),
                    params=0,
                    has_weights=False,
                )
            ],
            graph={"nodes": [], "edges": []},
        )

    def _fake_quant_result(self) -> dict:
        qp = mock.Mock()
        qp.scale = 1.0
        qp.zero_point = 0
        qp.min_val = -1.0
        qp.max_val = 1.0
        qp.axis = None
        return {
            "fp32_outputs": {"dummy": np.zeros((4, 2), dtype=np.float32)},
            "dequant_outputs": {"dummy": np.zeros((4, 2), dtype=np.float32)},
            "activation_distributions": {"dummy": [0.1, 0.2]},
            "weight_distributions": {"dummy": [0.3, 0.4]},
            "bias_distributions": {},
            "activation_qparams": {"dummy": qp},
            "weight_info": {},
            "layer_modes": {"weight": {}, "activation": {}},
        }

    def test_finalize_minimal_profile_keeps_only_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / ".quanta"
            config = PipelineConfig(
                model_path="dummy.keras",
                dataset_path="dummy.npy",
                output_root=str(output_root),
                final_artifact_profile="minimal",
                max_cached_runs=3,
            )
            with (
                mock.patch("quanta.pipeline.DatasetAdapter", _DummyDatasetAdapter),
                mock.patch("quanta.pipeline.ModelLoader") as loader_cls,
                mock.patch("quanta.pipeline.ModelConverter") as conv_cls,
                mock.patch(
                    "quanta.pipeline.run_fake_quant_pipeline",
                    return_value=self._fake_quant_result(),
                ),
                mock.patch(
                    "quanta.pipeline.compute_metrics",
                    return_value={"layers": {"dummy": {"mae": 0, "rmse": 0, "kl": 0}}},
                ),
                mock.patch(
                    "quanta.pipeline.estimate_tradeoffs",
                    return_value={"layers": {"dummy": {}}},
                ),
            ):
                loader_cls.return_value.load.return_value = _DummyModel()
                conv_cls.return_value.convert.return_value = self._fake_unified()
                tmp_dir = run_pipeline(config)

            self.assertTrue(tmp_dir.exists())
            self.assertTrue((tmp_dir / "distributions.json").exists())
            self.assertTrue((tmp_dir / "graph.json").exists())
            self.assertTrue((tmp_dir / "metrics.json").exists())
            self.assertTrue((tmp_dir / "qparams.json").exists())
            self.assertTrue((tmp_dir / "estimates.json").exists())

            finalized_runs = list(output_root.glob("run_*"))
            self.assertEqual(len(finalized_runs), 1)
            final_run = finalized_runs[0]
            self.assertTrue((final_run / "run_meta.json").exists())
            self.assertTrue((final_run / "graph.json").exists())
            self.assertTrue((final_run / "metrics.json").exists())
            self.assertTrue((final_run / "qparams.json").exists())
            self.assertTrue((final_run / "estimates.json").exists())
            self.assertTrue((final_run / "distributions.json").exists())

    def test_prune_only_applies_to_completed_run_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / ".quanta"
            output_root.mkdir(parents=True, exist_ok=True)

            completed = []
            for idx in range(4):
                run = output_root / f"run_20240101_00000{idx}_000000"
                run.mkdir(parents=True, exist_ok=True)
                (run / "run_meta.json").write_text(json.dumps({"status": "complete"}))
                completed.append(run)

            tmp_run = output_root / "tmp_run_20240101_999999"
            tmp_run.mkdir(parents=True, exist_ok=True)
            (tmp_run / "graph.json").write_text("{}")

            _prune_previous_runs(output_root, keep_runs=2)

            kept_completed = [
                path for path in output_root.iterdir() if path.name.startswith("run_")
            ]
            self.assertEqual(len(kept_completed), 2)
            self.assertTrue(tmp_run.exists())


class ServerDistributionAvailabilityTest(unittest.TestCase):
    def test_distributions_endpoint_returns_409_when_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_20240101_000000_000000"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "graph.json").write_text(json.dumps({"nodes": [], "edges": []}))
            (run_dir / "metrics.json").write_text(json.dumps({"layers": {}}))
            (run_dir / "qparams.json").write_text(json.dumps({"activations": {}}))
            (run_dir / "estimates.json").write_text(json.dumps({}))
            (run_dir / "run_meta.json").write_text(
                json.dumps({"status": "complete", "profile": "minimal"})
            )

            app = create_app(str(run_dir))
            client = TestClient(app)
            response = client.get("/api/layers/dummy/distributions")
            self.assertEqual(response.status_code, 409)

    def test_runtime_endpoint_exposes_active_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_20240101_000000_000000"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "graph.json").write_text(json.dumps({"nodes": [], "edges": []}))
            (run_dir / "metrics.json").write_text(json.dumps({"layers": {}}))
            (run_dir / "qparams.json").write_text(json.dumps({"activations": {}}))
            (run_dir / "estimates.json").write_text(json.dumps({}))

            cache = StrategyCache(Path(tmp), run_id="20240101_000000_000000")
            cache.upsert_strategy("minmax|int8|int8|{}|{}", run_dir)
            cache.set_latest_state(
                {
                    "range_mode": "minmax",
                    "global_weight_mode": "int8",
                    "global_activation_mode": "int8",
                    "layer_weight_modes": {},
                    "layer_activation_modes": {},
                }
            )

            app = create_app(
                str(run_dir),
                runtime_config=None,
                loaded_from_run=True,
                strategy_cache=cache,
            )
            client = TestClient(app)
            response = client.get("/api/runtime")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["run_id"], "20240101_000000_000000")
            self.assertTrue(payload["loaded_from_run"])
            self.assertFalse(payload["runtime_recompute_enabled"])
            self.assertIn("latest_state", payload)
            self.assertEqual(payload["latest_state"]["range_mode"], "minmax")


class StrategyKeyNormalizationTest(unittest.TestCase):
    def test_redundant_overrides_stripped(self) -> None:
        key_with_redundant = normalize_strategy_key(
            "minmax", "int8", "int8", {"conv1d": "int8"}, {"conv1d": "int8"}
        )
        key_clean = normalize_strategy_key("minmax", "int8", "int8", {}, {})
        self.assertEqual(key_with_redundant, key_clean)

    def test_real_overrides_kept(self) -> None:
        key = normalize_strategy_key("minmax", "int8", "int8", {"conv1d": "int4"}, {})
        self.assertIn("conv1d", key)
        self.assertIn("int4", key)

    def test_json_format_compact_and_sorted(self) -> None:
        key = normalize_strategy_key(
            "minmax",
            "int8",
            "int8",
            {"z_layer": "int4", "a_layer": "int2"},
            {},
        )
        self.assertIn('{"a_layer":"int2","z_layer":"int4"}', key)

    def test_strategy_key_from_state_matches_normalize(self) -> None:
        state = {
            "range_mode": "minmax",
            "global_weight_mode": "int8",
            "global_activation_mode": "fp16",
            "layer_weight_modes": {"conv1d": "int4"},
            "layer_activation_modes": {"conv1d": "fp16"},
        }
        from_state = strategy_key_from_state(state)
        direct = normalize_strategy_key(
            "minmax",
            "int8",
            "fp16",
            {"conv1d": "int4"},
            {"conv1d": "fp16"},
        )
        self.assertEqual(from_state, direct)
        self.assertNotIn("conv1d", from_state.split("|")[4])


class StrategyCacheTest(unittest.TestCase):
    def test_upsert_and_get_strategy_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = StrategyCache(root, run_id="test_run")
            art_dir = root / "tmp_run_1"
            art_dir.mkdir()
            (art_dir / "graph.json").write_text("{}")

            cache.upsert_strategy("minmax|int8|int8|{}|{}", art_dir)
            self.assertEqual(cache.get_strategy_root("minmax|int8|int8|{}|{}"), art_dir)
            self.assertIsNone(cache.get_strategy_root("nonexistent"))

    def test_latest_state_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = StrategyCache(root, run_id="test_run")
            state = {
                "range_mode": "clip99_99",
                "global_weight_mode": "int4",
                "global_activation_mode": "int8",
                "layer_weight_modes": {"conv1": "int2"},
                "layer_activation_modes": {},
            }
            cache.set_latest_state(state)
            self.assertEqual(cache.get_latest_state(), state)

            reloaded = StrategyCache.load(root / "strategy_cache.json")
            self.assertEqual(reloaded.get_latest_state(), state)
            self.assertEqual(reloaded.run_id, "test_run")

    def test_materialize_finalized_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            art_dir = root / "tmp_run_1"
            art_dir.mkdir()
            for name in (
                "graph.json",
                "metrics.json",
                "qparams.json",
                "estimates.json",
                "distributions.json",
            ):
                (art_dir / name).write_text(json.dumps({"artifact": name}))

            cache = StrategyCache(root, run_id="20240101_000000_000000")
            state = {
                "range_mode": "minmax",
                "global_weight_mode": "int8",
                "global_activation_mode": "int8",
                "layer_weight_modes": {},
                "layer_activation_modes": {},
            }
            cache.set_latest_state(state)
            key = strategy_key_from_state(state)
            cache.upsert_strategy(key, art_dir)

            final_dir = cache.materialize_finalized_snapshot()
            self.assertTrue(final_dir.exists())
            self.assertTrue((final_dir / "run_meta.json").exists())
            self.assertTrue((final_dir / "distributions.json").exists())
            self.assertTrue((final_dir / "graph.json").exists())
            meta = json.loads((final_dir / "run_meta.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["profile"], "full")
            self.assertIn("latest_state", meta)
            self.assertEqual(meta["latest_state"]["range_mode"], "minmax")
            self.assertTrue(cache.finalized)

    def test_from_finalized_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_20240101_000000_000000"
            run_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "graph.json",
                "metrics.json",
                "qparams.json",
                "estimates.json",
                "distributions.json",
            ):
                (run_dir / name).write_text(json.dumps({}))
            meta = {
                "status": "complete",
                "run_id": "20240101_000000_000000",
                "profile": "full",
                "artifacts": [
                    "graph.json",
                    "metrics.json",
                    "qparams.json",
                    "estimates.json",
                    "distributions.json",
                ],
                "latest_state": {
                    "range_mode": "clip99_99",
                    "global_weight_mode": "int4",
                    "global_activation_mode": "fp16",
                    "layer_weight_modes": {},
                    "layer_activation_modes": {},
                },
                "source_config": {
                    "model_path": "/fake/model.keras",
                    "dataset_path": "/fake/dataset.npy",
                    "batch_size": 16,
                },
            }
            (run_dir / "run_meta.json").write_text(json.dumps(meta))

            cache = StrategyCache.from_finalized_run(run_dir)
            self.assertEqual(cache.run_id, "20240101_000000_000000")
            self.assertFalse(cache.finalized)
            ls = cache.get_latest_state()
            self.assertEqual(ls["range_mode"], "clip99_99")
            self.assertEqual(ls["global_weight_mode"], "int4")
            key = strategy_key_from_state(ls)
            self.assertEqual(cache.get_strategy_root(key), run_dir)
            sc = cache.get_source_config()
            self.assertEqual(sc["model_path"], "/fake/model.keras")
            self.assertEqual(sc["batch_size"], 16)

    def test_materialize_includes_source_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            art_dir = root / "tmp_run_1"
            art_dir.mkdir()
            for name in (
                "graph.json",
                "metrics.json",
                "qparams.json",
                "estimates.json",
            ):
                (art_dir / name).write_text(json.dumps({}))

            cache = StrategyCache(root, run_id="20240101_000000_000000")
            state = {
                "range_mode": "minmax",
                "global_weight_mode": "int8",
                "global_activation_mode": "int8",
                "layer_weight_modes": {},
                "layer_activation_modes": {},
            }
            cache.set_latest_state(state)
            cache.set_source_config(
                {
                    "model_path": "/my/model.keras",
                    "dataset_path": "/my/data.npy",
                    "batch_size": 64,
                }
            )
            cache.upsert_strategy(strategy_key_from_state(state), art_dir)
            final_dir = cache.materialize_finalized_snapshot()

            meta = json.loads((final_dir / "run_meta.json").read_text())
            self.assertIn("source_config", meta)
            self.assertEqual(meta["source_config"]["model_path"], "/my/model.keras")
            self.assertEqual(meta["source_config"]["batch_size"], 64)


class ServerFinalizeTest(unittest.TestCase):
    def test_finalize_endpoint_creates_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "tmp_run_1"
            run_dir.mkdir()
            for name in (
                "graph.json",
                "metrics.json",
                "qparams.json",
                "estimates.json",
                "distributions.json",
            ):
                (run_dir / name).write_text(json.dumps({"artifact": name}))

            cache = StrategyCache(root, run_id="20240101_000000_000000")
            state = {
                "range_mode": "minmax",
                "global_weight_mode": "int8",
                "global_activation_mode": "int8",
                "layer_weight_modes": {},
                "layer_activation_modes": {},
            }
            cache.set_latest_state(state)
            cache.upsert_strategy(strategy_key_from_state(state), run_dir)

            app = create_app(str(run_dir), runtime_config=None, strategy_cache=cache)
            client = TestClient(app)

            resp = client.post("/api/run/finalize")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "finalized")
            self.assertIn("finalized_path", data)

            finalized_path = Path(data["finalized_path"])
            self.assertTrue((finalized_path / "distributions.json").exists())
            self.assertTrue((finalized_path / "run_meta.json").exists())

    def test_active_state_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "tmp_run_1"
            run_dir.mkdir()
            (run_dir / "graph.json").write_text("{}")

            cache = StrategyCache(root, run_id="test")
            cache.upsert_strategy("minmax|int8|int8|{}|{}", run_dir)
            app = create_app(str(run_dir), runtime_config=None, strategy_cache=cache)
            client = TestClient(app)

            resp = client.post(
                "/api/runtime/active-state",
                json={
                    "range_mode": "clip99_99",
                    "global_weight_mode": "int4",
                    "global_activation_mode": "fp16",
                    "layer_weight_modes": {"conv1": "int2"},
                    "layer_activation_modes": {},
                },
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(cache.get_latest_state()["range_mode"], "clip99_99")

    def test_cache_hit_with_redundant_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "tmp_run_1"
            run_dir.mkdir()
            (run_dir / "graph.json").write_text(json.dumps({"nodes": [], "edges": []}))
            (run_dir / "estimates.json").write_text(json.dumps({}))

            cache = StrategyCache(root, run_id="test")
            cache.upsert_strategy(
                normalize_strategy_key("minmax", "int8", "int8", {}, {}),
                run_dir,
            )

            app = create_app(str(run_dir), runtime_config=None, strategy_cache=cache)
            client = TestClient(app)

            resp_clean = client.get(
                "/api/graph?range_mode=minmax&weight_mode=int8&activation_mode=int8"
                "&layer_weight_modes=%7B%7D&layer_activation_modes=%7B%7D"
            )
            self.assertEqual(resp_clean.status_code, 200)

            resp_redundant = client.get(
                "/api/graph?range_mode=minmax&weight_mode=int8&activation_mode=int8"
                "&layer_weight_modes=%7B%22conv1d%22%3A%22int8%22%7D"
                "&layer_activation_modes=%7B%7D"
            )
            self.assertEqual(resp_redundant.status_code, 200)

    def test_close_requires_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "tmp_run_1"
            run_dir.mkdir()
            (run_dir / "graph.json").write_text("{}")

            cache = StrategyCache(root, run_id="test")
            cache.upsert_strategy("minmax|int8|int8|{}|{}", run_dir)
            app = create_app(str(run_dir), runtime_config=None, strategy_cache=cache)
            client = TestClient(app)

            resp = client.post("/api/run/close")
            self.assertEqual(resp.status_code, 409)
            self.assertIn("not been finalized", resp.json()["detail"])


class CliLoadRunModeTest(unittest.TestCase):
    def _make_run_dir(
        self, root: Path, source_config: dict[str, Any] | None = None
    ) -> Path:
        run_dir = root / "run_20240101_000000_000000"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "graph.json").write_text(json.dumps({"nodes": [], "edges": []}))
        (run_dir / "metrics.json").write_text(json.dumps({"layers": {}}))
        (run_dir / "qparams.json").write_text(json.dumps({"activations": {}}))
        (run_dir / "estimates.json").write_text(json.dumps({}))
        (run_dir / "distributions.json").write_text(
            json.dumps({"activations": {}, "weights": {}, "biases": {}})
        )
        meta: dict[str, Any] = {
            "status": "complete",
            "run_id": "20240101_000000_000000",
            "profile": "full",
            "artifacts": [
                "graph.json",
                "metrics.json",
                "qparams.json",
                "estimates.json",
                "distributions.json",
            ],
            "latest_state": {
                "range_mode": "minmax",
                "global_weight_mode": "int8",
                "global_activation_mode": "int8",
                "layer_weight_modes": {},
                "layer_activation_modes": {},
            },
        }
        if source_config:
            meta["source_config"] = source_config
        (run_dir / "run_meta.json").write_text(json.dumps(meta))
        return run_dir

    def test_validate_load_run_rejects_missing_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_20240101_000000_000000"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "graph.json").write_text("{}")
            (run_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "run_id": "20240101_000000_000000",
                        "profile": "minimal",
                        "artifacts": ["graph.json"],
                        "latest_state": {},
                    }
                )
            )
            with self.assertRaises(SystemExit):
                cli._validate_load_run_dir(str(run_dir))

    def test_main_load_run_uses_existing_artifacts_without_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._make_run_dir(Path(tmp))
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    ["quanta", "--load-run", str(run_dir), "--no-ui", "--port", "8999"],
                ),
                mock.patch("quanta.server.create_app") as create_app_mock,
                mock.patch("uvicorn.run") as uvicorn_run_mock,
                mock.patch("webbrowser.open") as webbrowser_open_mock,
            ):
                create_app_mock.return_value = object()
                cli.main()

            create_app_mock.assert_called_once()
            call_kwargs = create_app_mock.call_args
            self.assertEqual(call_kwargs[0][0], str(run_dir))
            self.assertIsNone(call_kwargs[1]["runtime_config"])
            self.assertTrue(call_kwargs[1]["loaded_from_run"])
            self.assertIsNotNone(call_kwargs[1]["strategy_cache"])
            uvicorn_run_mock.assert_called_once()
            webbrowser_open_mock.assert_not_called()

    def test_main_load_run_with_test_flag_enables_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._make_run_dir(Path(tmp))
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "quanta",
                        "--load-run",
                        str(run_dir),
                        "--test",
                        "--no-ui",
                        "--port",
                        "8999",
                    ],
                ),
                mock.patch("quanta.server.create_app") as create_app_mock,
                mock.patch("uvicorn.run"),
            ):
                create_app_mock.return_value = object()
                cli.main()

            create_app_mock.assert_called_once()
            call_kwargs = create_app_mock.call_args
            self.assertEqual(call_kwargs[0][0], str(run_dir))
            self.assertIsNotNone(call_kwargs[1]["runtime_config"])
            self.assertTrue(call_kwargs[1]["loaded_from_run"])
            self.assertIsNotNone(call_kwargs[1]["strategy_cache"])

    def test_main_load_run_auto_restores_from_source_config(self) -> None:
        """--load-run with source_config in run_meta enables recompute automatically."""
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.keras"
            dataset_path = Path(tmp) / "dataset.npy"
            model_path.write_text("{}")
            dataset_path.write_text("{}")

            run_dir = self._make_run_dir(
                Path(tmp),
                source_config={
                    "model_path": str(model_path),
                    "dataset_path": str(dataset_path),
                    "batch_size": 16,
                    "max_samples": 100,
                },
            )
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    ["quanta", "--load-run", str(run_dir), "--no-ui", "--port", "8999"],
                ),
                mock.patch("quanta.server.create_app") as create_app_mock,
                mock.patch("uvicorn.run"),
            ):
                create_app_mock.return_value = object()
                cli.main()

            create_app_mock.assert_called_once()
            call_kwargs = create_app_mock.call_args
            rc = call_kwargs[1]["runtime_config"]
            self.assertIsNotNone(rc)
            self.assertEqual(rc["model_path"], str(model_path))
            self.assertEqual(rc["dataset_path"], str(dataset_path))
            self.assertEqual(rc["batch_size"], 16)


if __name__ == "__main__":
    unittest.main()
