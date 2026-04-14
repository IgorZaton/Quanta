from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass
class LayerRef:
    name: str
    layer: tf.keras.layers.Layer
    parent: str | None = None
    is_container: bool = False


def iter_model_layers(
    model: tf.keras.Model, prefix: str = "", parent: str | None = None
) -> list[LayerRef]:
    refs: list[LayerRef] = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        scoped = f"{prefix}/{layer.name}" if prefix else layer.name
        if isinstance(layer, tf.keras.Model):
            refs.append(
                LayerRef(name=scoped, layer=layer, parent=parent, is_container=True)
            )
            refs.extend(iter_model_layers(layer, prefix=scoped, parent=scoped))
        else:
            refs.append(
                LayerRef(name=scoped, layer=layer, parent=parent, is_container=False)
            )
    return refs


def iter_leaf_layers(model: tf.keras.Model, prefix: str = "") -> list[LayerRef]:
    return [
        ref for ref in iter_model_layers(model, prefix=prefix) if not ref.is_container
    ]
