from __future__ import annotations

from pathlib import Path
from typing import Any

import tensorflow as tf

from .model_inspect import iter_leaf_layers
from .model_ir import LayerMeta, UnifiedModel


class ModelLoader:
    """Loader step: read Keras model artifact."""

    def load(
        self, model_path: str, custom_objects: dict[str, Any] | None = None
    ) -> tf.keras.Model:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file/folder not found: {path}")

        if path.is_dir() and (path / "saved_model.pb").exists():
            raise ValueError(
                "SavedModel format is no longer supported in Quanta. "
                "Please provide a Keras model file (for example .keras)."
            )

        # Quanta performs inference-only analysis; skip compile-time objects
        # (optimizer/loss/metrics) to reduce custom object requirements.
        model = tf.keras.models.load_model(
            path, custom_objects=custom_objects, compile=False
        )
        return _ensure_unique_layer_names(model)


class ModelConverter:
    """Converter step: convert loaded Keras model into UnifiedModel IR."""

    def convert(self, model: tf.keras.Model) -> UnifiedModel:
        return UnifiedModel(
            model=model, metadata=self._metadata_from_keras(model), graph=None
        )

    def _metadata_from_keras(self, model: tf.keras.Model) -> list[LayerMeta]:
        meta: list[LayerMeta] = []
        for ref in iter_leaf_layers(model):
            layer = ref.layer
            meta.append(
                LayerMeta(
                    name=ref.name,
                    layer_type=layer.__class__.__name__,
                    input_shape=getattr(layer, "input_shape", None),
                    output_shape=getattr(layer, "output_shape", None),
                    params=layer.count_params(),
                    has_weights=bool(layer.weights),
                )
            )
        return meta


def _ensure_unique_layer_names(model: tf.keras.Model) -> tf.keras.Model:
    # Always normalize names via clone to avoid edge cases where duplicate
    # operation names are produced by externally serialized models.
    seen: dict[str, int] = {}

    def _clone_with_name(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        config = layer.get_config()
        base_name = config.get("name", layer.name)
        idx = seen.get(base_name, 0)
        seen[base_name] = idx + 1
        config["name"] = base_name if idx == 0 else f"{base_name}_{idx}"
        return layer.__class__.from_config(config)

    try:
        cloned = tf.keras.models.clone_model(model, clone_function=_clone_with_name)
        cloned.set_weights(model.get_weights())
        return cloned
    except Exception:
        # Keep original model if cloning is not possible for a custom object.
        return model
