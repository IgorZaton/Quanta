from __future__ import annotations

import unittest

import numpy as np
import tensorflow as tf

from quanta.artifacts import build_graph
from quanta.model_inspect import iter_leaf_layers, iter_model_layers
from quanta.model_ir import LayerMeta
from quanta.quantization import run_fake_quant_pipeline


class NestedModelsSupportTest(unittest.TestCase):
    def _build_nested_model(self) -> tf.keras.Model:
        block = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(6, activation="relu", name="inner_dense1"),
                tf.keras.layers.Dense(4, activation="relu", name="inner_dense2"),
            ],
            name="block",
        )
        inp = tf.keras.Input(shape=(8,), name="input")
        x = block(inp)
        out = tf.keras.layers.Dense(2, name="head")(x)
        return tf.keras.Model(inputs=inp, outputs=out, name="nested_test")

    def test_iter_model_layers_preserves_nested_container_nodes(self) -> None:
        model = self._build_nested_model()
        refs = iter_model_layers(model)
        names = [ref.name for ref in refs]
        container = next(ref for ref in refs if ref.name == "block")
        self.assertTrue(container.is_container)
        self.assertIn("block", names)
        self.assertIn("block/inner_dense1", names)
        self.assertIn("block/inner_dense2", names)

    def test_iter_leaf_layers_flattens_nested_model(self) -> None:
        model = self._build_nested_model()
        names = [ref.name for ref in iter_leaf_layers(model)]
        self.assertIn("block/inner_dense1", names)
        self.assertIn("block/inner_dense2", names)
        self.assertIn("head", names)

    def test_fake_quant_pipeline_tracks_nested_leaf_layers(self) -> None:
        model = self._build_nested_model()
        batch = np.random.RandomState(0).randn(4, 8).astype(np.float32)
        result = run_fake_quant_pipeline(
            model,
            dataset_batches=[batch],
            global_weight_mode="int8",
            global_activation_mode="int8",
        )
        keys = set(result["fp32_outputs"].keys())
        self.assertIn("block/inner_dense1", keys)
        self.assertIn("block/inner_dense2", keys)
        self.assertIn("head", keys)

    def test_graph_expands_nested_container_connections(self) -> None:
        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(6, activation="relu", name="inner_dense1"),
                tf.keras.layers.Dense(4, activation="relu", name="inner_dense2"),
            ],
            name="encoder",
        )
        inp = tf.keras.Input(shape=(8,), name="input")
        enc = encoder(inp)
        other = tf.keras.layers.Dense(4, activation="relu", name="other")(inp)
        joined = tf.keras.layers.Concatenate(name="join")([enc, other])
        out = tf.keras.layers.Dense(2, name="head")(joined)
        model = tf.keras.Model(inputs=inp, outputs=out, name="parallel_nested")

        metadata = [
            LayerMeta(
                name=ref.name,
                layer_type=ref.layer.__class__.__name__,
                input_shape=getattr(ref.layer, "input_shape", None),
                output_shape=getattr(ref.layer, "output_shape", None),
                params=ref.layer.count_params(),
                has_weights=bool(ref.layer.weights),
            )
            for ref in iter_leaf_layers(model)
        ]
        graph = build_graph(model, metadata, layer_metrics={})
        node_ids = {n["id"] for n in graph["nodes"]}
        edge_set = {(e["source"], e["target"]) for e in graph["edges"]}

        self.assertIn("input", node_ids)
        self.assertNotIn("encoder", node_ids)
        self.assertIn("encoder/inner_dense1", node_ids)
        self.assertIn("encoder/inner_dense2", node_ids)
        self.assertIn(("input", "encoder/inner_dense1"), edge_set)
        self.assertIn(("input", "other"), edge_set)
        self.assertIn(("encoder/inner_dense2", "join"), edge_set)
        self.assertIn(("other", "join"), edge_set)

    def test_graph_connects_input_to_expanded_nested_entry(self) -> None:
        inp = tf.keras.Input(shape=(8,), name="input_layer")
        enc_input = tf.keras.Input(shape=(8,), name="encoder_input")
        x = tf.keras.layers.LayerNormalization(name="normalize")(enc_input)
        x = tf.keras.layers.Dense(4, activation="relu", name="latent")(x)
        encoder = tf.keras.Model(enc_input, x, name="Encoder")
        enc = encoder(inp)
        out = tf.keras.layers.Dense(2, name="head")(enc)
        model = tf.keras.Model(inputs=inp, outputs=out, name="nested_input_map")

        metadata = [
            LayerMeta(
                name=ref.name,
                layer_type=ref.layer.__class__.__name__,
                input_shape=getattr(ref.layer, "input_shape", None),
                output_shape=getattr(ref.layer, "output_shape", None),
                params=ref.layer.count_params(),
                has_weights=bool(ref.layer.weights),
            )
            for ref in iter_leaf_layers(model)
        ]
        graph = build_graph(model, metadata, layer_metrics={})
        edge_set = {(e["source"], e["target"]) for e in graph["edges"]}
        self.assertIn(("input_layer", "Encoder/normalize"), edge_set)

    def test_graph_respects_container_multi_output_port_mapping(self) -> None:
        class SplitBlock(tf.keras.Model):
            def __init__(self) -> None:
                super().__init__(name="encoder")
                self.a = tf.keras.layers.Dense(4, activation="relu", name="a")
                self.b = tf.keras.layers.Dense(4, activation="relu", name="b")

            def call(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                xa = self.a(x)
                xb = self.b(x)
                return xa, xb

        inp = tf.keras.Input(shape=(8,), name="input")
        e0, e1 = SplitBlock()(inp)
        left = tf.keras.layers.Dense(2, name="left")(e0)
        right = tf.keras.layers.Dense(2, name="right")(e1)
        out = tf.keras.layers.Add(name="sum")([left, right])
        model = tf.keras.Model(inputs=inp, outputs=out, name="multi_out_nested")

        metadata = [
            LayerMeta(
                name=ref.name,
                layer_type=ref.layer.__class__.__name__,
                input_shape=getattr(ref.layer, "input_shape", None),
                output_shape=getattr(ref.layer, "output_shape", None),
                params=ref.layer.count_params(),
                has_weights=bool(ref.layer.weights),
            )
            for ref in iter_leaf_layers(model)
        ]
        graph = build_graph(model, metadata, layer_metrics={})
        edge_set = {(e["source"], e["target"]) for e in graph["edges"]}
        self.assertIn(("encoder/a", "left"), edge_set)
        self.assertIn(("encoder/b", "right"), edge_set)


if __name__ == "__main__":
    unittest.main()
