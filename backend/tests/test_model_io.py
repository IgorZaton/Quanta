from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from quanta.model_io import _ensure_unique_layer_names


class EnsureUniqueLayerNamesTest(unittest.TestCase):
    def _build_model(self) -> tf.keras.Sequential:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(8,), name="input"),
                tf.keras.layers.Dense(4, activation="relu", name="dense_a"),
                tf.keras.layers.Dense(2, name="dense_b"),
            ]
        )
        model(np.zeros((1, 8), dtype=np.float32), training=False)
        return model

    def test_clones_model_when_names_are_already_unique(self) -> None:
        model = self._build_model()
        normalized = _ensure_unique_layer_names(model)

        self.assertIsNot(normalized, model)
        self.assertEqual([l.name for l in normalized.layers], [l.name for l in model.layers])

    def test_clones_model_and_renames_duplicate_layers(self) -> None:
        model = self._build_model()
        sample = np.random.RandomState(7).randn(3, 8).astype(np.float32)
        expected = model(sample, training=False).numpy()

        # Simulate malformed model metadata coming from external serialization:
        # two layer configs resolve to one duplicated name.
        original_get_config_0 = model.layers[0].get_config
        original_get_config_1 = model.layers[1].get_config

        def _dup_config_0():
            cfg = original_get_config_0()
            cfg["name"] = "flatten"
            return cfg

        def _dup_config_1():
            cfg = original_get_config_1()
            cfg["name"] = "flatten"
            return cfg

        model.layers[0].get_config = _dup_config_0  # type: ignore[method-assign]
        model.layers[1].get_config = _dup_config_1  # type: ignore[method-assign]

        normalized = _ensure_unique_layer_names(model)
        new_names = [layer.name for layer in normalized.layers]

        self.assertEqual(len(new_names), len(set(new_names)))
        self.assertIn("flatten", new_names)
        self.assertIn("flatten_1", new_names)

        actual = normalized(sample, training=False).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
