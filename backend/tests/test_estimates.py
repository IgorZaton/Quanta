from __future__ import annotations

from dataclasses import dataclass
import unittest

from quanta.estimates import estimate_tradeoffs


@dataclass
class LayerMeta:
    name: str
    layer_type: str
    input_shape: object
    output_shape: object
    params: int
    has_weights: bool


class EstimateTradeoffsTest(unittest.TestCase):
    def test_layer_size_and_global_totals_are_consistent(self) -> None:
        metadata = [
            LayerMeta(
                name="conv",
                layer_type="Conv2D",
                input_shape=(None, 8, 8, 3),
                output_shape=(None, 8, 8, 8),
                params=320,
                has_weights=True,
            ),
            LayerMeta(
                name="dense",
                layer_type="Dense",
                input_shape=(None, 128),
                output_shape=(None, 10),
                params=1290,
                has_weights=True,
            ),
        ]
        weight_modes = {"conv": "int4", "dense": "fp16"}
        activation_modes = {"conv": "int8", "dense": "fp32"}

        estimates = estimate_tradeoffs(metadata, weight_modes, activation_modes)
        conv = estimates["layers"]["conv"]
        dense = estimates["layers"]["dense"]
        global_est = estimates["global"]

        # Size follows bit-width-per-param rule.
        self.assertEqual(conv["size_bytes"], (320 * 4) / 8.0)
        self.assertEqual(dense["size_bytes"], (1290 * 16) / 8.0)
        self.assertEqual(conv["size_bytes_fp32"], (320 * 32) / 8.0)
        self.assertEqual(dense["size_bytes_fp32"], (1290 * 32) / 8.0)

        # Global totals are exact sums of per-layer estimates.
        self.assertAlmostEqual(
            global_est["size_bytes"], conv["size_bytes"] + dense["size_bytes"]
        )
        self.assertAlmostEqual(
            global_est["size_bytes_fp32"],
            conv["size_bytes_fp32"] + dense["size_bytes_fp32"],
        )
        self.assertAlmostEqual(
            global_est["latency_ms"], conv["latency_ms"] + dense["latency_ms"]
        )
        self.assertAlmostEqual(
            global_est["latency_ms_fp32"],
            conv["latency_ms_fp32"] + dense["latency_ms_fp32"],
        )

    def test_lower_precision_modes_reduce_estimated_size_and_latency(self) -> None:
        metadata = [
            LayerMeta(
                name="block",
                layer_type="Dense",
                input_shape=(None, 256),
                output_shape=(None, 64),
                params=100_000,
                has_weights=True,
            )
        ]

        int2_est = estimate_tradeoffs(
            metadata,
            weight_modes={"block": "int2"},
            activation_modes={"block": "int2"},
        )["layers"]["block"]
        fp32_est = estimate_tradeoffs(
            metadata,
            weight_modes={"block": "fp32"},
            activation_modes={"block": "fp32"},
        )["layers"]["block"]

        self.assertLess(int2_est["size_bytes"], fp32_est["size_bytes"])
        self.assertLess(int2_est["latency_ms"], fp32_est["latency_ms"])


if __name__ == "__main__":
    unittest.main()
