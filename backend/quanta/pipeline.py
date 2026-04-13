from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .adapters import DatasetAdapter
from .artifacts import build_graph, write_artifacts
from .estimates import estimate_tradeoffs
from .metrics import compute_metrics
from .model_io import ModelConverter, ModelLoader
from .quantization import run_fake_quant_pipeline


@dataclass
class PipelineConfig:
    model_path: str
    dataset_path: str
    batch_size: int = 32
    max_samples: int | None = None
    output_root: str = ".quanta"
    intentionally_bad_layers_count: int = 0
    range_mode: str = "minmax"
    global_weight_mode: str = "int8"
    global_activation_mode: str = "int8"
    layer_weight_modes: dict[str, str] | None = None
    layer_activation_modes: dict[str, str] | None = None
    custom_objects: dict[str, Any] | None = None


def _prune_previous_runs(output_root: Path) -> None:
    if not output_root.exists():
        return
    for child in output_root.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            shutil.rmtree(child, ignore_errors=True)


def run_pipeline(config: PipelineConfig) -> Path:
    loader = ModelLoader()
    converter = ModelConverter()
    unified_model = converter.convert(loader.load(config.model_path, custom_objects=config.custom_objects))
    dataset_adapter = DatasetAdapter(config.dataset_path, batch_size=config.batch_size, max_samples=config.max_samples)
    dataset = dataset_adapter.load().astype(np.float32)
    if isinstance(unified_model.model.input_shape, list):
        raise ValueError("Multi-input models are not supported in this MVP")
    expected_shape = tuple(unified_model.model.input_shape[1:])
    sample_shape = tuple(dataset.shape[1:])
    if expected_shape != sample_shape:
        raise ValueError(f"Dataset sample shape {sample_shape} does not match model input shape {expected_shape}")

    batches = list(dataset_adapter.iter_batches())
    result = run_fake_quant_pipeline(
        unified_model.model,
        batches,
        intentionally_bad_layers_count=config.intentionally_bad_layers_count,
        range_mode=config.range_mode,
        global_weight_mode=config.global_weight_mode,
        global_activation_mode=config.global_activation_mode,
        layer_weight_modes=config.layer_weight_modes,
        layer_activation_modes=config.layer_activation_modes,
    )
    metrics = compute_metrics(result["fp32_outputs"], result["dequant_outputs"])

    metadata = unified_model.metadata
    layer_modes = result.get("layer_modes", {"weight": {}, "activation": {}})
    estimates = estimate_tradeoffs(metadata, layer_modes.get("weight", {}), layer_modes.get("activation", {}))
    graph = unified_model.graph or build_graph(unified_model.model, metadata, metrics["layers"], estimates.get("layers", {}))
    if unified_model.graph is not None:
        for node in graph.get("nodes", []):
            node_name = node.get("name", "")
            node["metrics"] = metrics["layers"].get(node_name, {"mae": 0.0, "rmse": 0.0, "kl": 0.0})
            node["estimates"] = estimates.get("layers", {}).get(node_name, {})

    output_root = Path(config.output_root)
    _prune_previous_runs(output_root)
    run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    write_artifacts(
        run_dir,
        graph=graph,
        metrics=metrics,
        activation_distributions=result["activation_distributions"],
        weight_distributions=result["weight_distributions"],
        bias_distributions=result["bias_distributions"],
        activation_qparams=result["activation_qparams"],
        weight_info=result["weight_info"],
        layer_modes=layer_modes,
        estimates=estimates,
    )
    return run_dir
