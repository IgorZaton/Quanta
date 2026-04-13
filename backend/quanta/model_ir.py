from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf


@dataclass
class LayerMeta:
    name: str
    layer_type: str
    input_shape: Any
    output_shape: Any
    params: int
    has_weights: bool


@dataclass
class UnifiedModel:
    model: tf.keras.Model
    metadata: list[LayerMeta]
    graph: dict[str, Any] | None = None
