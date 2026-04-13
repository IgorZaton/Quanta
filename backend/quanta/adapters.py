from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf


@dataclass
class LayerMeta:
    name: str
    layer_type: str
    input_shape: Any
    output_shape: Any
    params: int
    has_weights: bool


class BaseModelAdapter(ABC):
    @abstractmethod
    def layer_metadata(self) -> list[LayerMeta]:
        raise NotImplementedError


class BaseObserver(ABC):
    @abstractmethod
    def observe(self, values: np.ndarray) -> None:
        raise NotImplementedError


class BaseFakeQuantEngine(ABC):
    @abstractmethod
    def run(self) -> dict[str, Any]:
        raise NotImplementedError


class DatasetAdapter:
    def __init__(self, dataset_path: str, batch_size: int = 32, max_samples: int | None = None) -> None:
        self.dataset_path = Path(dataset_path)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be > 0 when provided")
        self.batch_size = batch_size
        self.max_samples = max_samples

    def load(self) -> np.ndarray:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        if self.dataset_path.suffix.lower() != ".npy":
            raise ValueError(f"Dataset must be .npy, got: {self.dataset_path}")
        data = np.load(self.dataset_path)
        if not isinstance(data, np.ndarray):
            raise ValueError("Dataset must be a numpy ndarray")
        if data.ndim < 2:
            raise ValueError("Dataset must include batch dimension and feature dimensions")
        if self.max_samples is not None:
            data = data[: self.max_samples]
        return data.astype(np.float32)

    def iter_batches(self) -> Iterator[np.ndarray]:
        data = self.load()
        for i in range(0, len(data), self.batch_size):
            yield data[i : i + self.batch_size]


class KerasModelAdapter(BaseModelAdapter):
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file/folder not found: {self.model_path}")
        self.model = self._load_model(self.model_path)

    def _load_model(self, model_path: Path) -> tf.keras.Model:
        try:
            return tf.keras.models.load_model(model_path)
        except Exception:
            if model_path.is_dir() and (model_path / "saved_model.pb").exists():
                return self._load_savedmodel_as_keras(model_path)
            raise

    def _load_savedmodel_as_keras(self, model_path: Path) -> tf.keras.Model:
        artifact = tf.saved_model.load(str(model_path))
        signatures = artifact.signatures
        endpoint = "serving_default" if "serving_default" in signatures else next(iter(signatures.keys()), None)
        if endpoint is None:
            raise ValueError(f"SavedModel at {model_path} has no callable signatures")
        fn = signatures[endpoint]
        input_specs = list(fn.structured_input_signature[1].items())
        if len(input_specs) != 1:
            raise ValueError("SavedModel fallback currently supports single-input signatures only")
        in_name, spec = input_specs[0]
        if spec.shape.rank is None or spec.shape.rank < 2:
            raise ValueError("SavedModel input rank must be >= 2")
        layer = tf.keras.layers.TFSMLayer(str(model_path), call_endpoint=endpoint)
        x = tf.keras.Input(shape=tuple(spec.shape[1:]), dtype=spec.dtype, name=in_name)
        y = layer(x)
        if isinstance(y, dict):
            y = next(iter(y.values()))
        return tf.keras.Model(inputs=x, outputs=y, name=f"savedmodel_{model_path.name}")

    def validate_dataset_shape(self, data: np.ndarray) -> None:
        if isinstance(self.model.input_shape, list):
            raise ValueError("Multi-input Keras models are not supported in this MVP")
        expected_shape = tuple(self.model.input_shape[1:])
        sample_shape = tuple(data.shape[1:])
        if expected_shape != sample_shape:
            raise ValueError(f"Dataset sample shape {sample_shape} does not match model input shape {expected_shape}")

    def layer_metadata(self) -> list[LayerMeta]:
        meta: list[LayerMeta] = []
        for layer in self.model.layers:
            meta.append(
                LayerMeta(
                    name=layer.name,
                    layer_type=layer.__class__.__name__,
                    input_shape=getattr(layer, "input_shape", None),
                    output_shape=getattr(layer, "output_shape", None),
                    params=layer.count_params(),
                    has_weights=bool(layer.weights),
                )
            )
        return meta

    def build_probe_model(self) -> tf.keras.Model:
        outputs = [layer.output for layer in self.model.layers if hasattr(layer, "output")]
        return tf.keras.Model(inputs=self.model.input, outputs=outputs)
