from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import tensorflow as tf

from quanta.model_io import (
    ConversionStageError,
    ModelConverter,
    ModelLoader,
    TorchScriptModel,
    _ensure_unique_layer_names,
)


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
        self.assertEqual(
            [layer.name for layer in normalized.layers],
            [layer.name for layer in model.layers],
        )

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


class ModelLoaderDispatchTest(unittest.TestCase):
    def test_load_pt_dispatches_to_torchscript_loader(self) -> None:
        with (
            mock.patch("quanta.model_io.Path.exists", return_value=True),
            mock.patch("quanta.model_io.Path.is_dir", return_value=False),
        ):
            fake_scripted = mock.Mock()
            fake_torch = SimpleNamespace(
                jit=SimpleNamespace(load=mock.Mock(return_value=fake_scripted))
            )
            with mock.patch.dict("sys.modules", {"torch": fake_torch}):
                loader = ModelLoader()
                loaded = loader.load("model.pt")
        self.assertIsInstance(loaded, TorchScriptModel)
        self.assertIs(loaded.model, fake_scripted)

    def test_load_pt_without_torch_raises_stage_error(self) -> None:
        with (
            mock.patch("quanta.model_io.Path.exists", return_value=True),
            mock.patch("quanta.model_io.Path.is_dir", return_value=False),
            mock.patch.dict("sys.modules", {"torch": None}),
        ):
            loader = ModelLoader()
            with self.assertRaises(ConversionStageError) as exc:
                loader.load("model.pt")
        self.assertEqual(exc.exception.stage, "pt_load")


class ModelConverterTorchBridgeTest(unittest.TestCase):
    def test_converter_reports_stage_when_onnx_missing(self) -> None:
        converter = ModelConverter()
        fake_ts = TorchScriptModel(model=mock.Mock(), path=Path("dummy.pt"))
        with mock.patch.dict("sys.modules", {"onnx": None}):
            with self.assertRaises(ConversionStageError) as exc:
                converter.convert(fake_ts)
        self.assertEqual(exc.exception.stage, "onnx_export")
