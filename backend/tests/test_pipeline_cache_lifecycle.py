from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from quanta.model_ir import LayerMeta, UnifiedModel
from quanta.pipeline import PipelineConfig, _prune_previous_runs, run_pipeline
from quanta.server import create_app


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
            self.assertFalse((final_run / "distributions.json").exists())

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


if __name__ == "__main__":
    unittest.main()
