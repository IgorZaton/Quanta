from __future__ import annotations

from typing import Any

from .model_ir import LayerMeta


def _bits_for_mode(mode: str) -> int:
    if mode.startswith("int"):
        return int(mode.replace("int", ""))
    if mode == "fp16":
        return 16
    return 32


def _latency_scale(mode: str) -> float:
    return {
        "int2": 0.45,
        "int4": 0.55,
        "int8": 0.65,
        "int12": 0.78,
        "int16": 0.85,
        "fp16": 0.80,
        "fp32": 1.00,
    }.get(mode, 1.0)


def estimate_tradeoffs(
    metadata: list[LayerMeta],
    weight_modes: dict[str, str],
    activation_modes: dict[str, str],
) -> dict[str, Any]:
    per_layer: dict[str, Any] = {}
    total_size_bytes = 0.0
    total_latency_ms = 0.0
    total_size_bytes_fp32 = 0.0
    total_latency_ms_fp32 = 0.0

    for layer in metadata:
        w_mode = weight_modes.get(layer.name, "int8")
        a_mode = activation_modes.get(layer.name, "int8")
        bits = _bits_for_mode(w_mode)
        size_bytes = (layer.params * bits) / 8.0
        size_bytes_fp32 = (layer.params * 32) / 8.0
        # Lightweight relative latency proxy based on parameter count and mode scaling.
        latency_scale = (_latency_scale(w_mode) + _latency_scale(a_mode)) / 2.0
        latency_ms = (0.02 + (layer.params / 2_000_000.0)) * latency_scale
        latency_ms_fp32 = (0.02 + (layer.params / 2_000_000.0)) * _latency_scale("fp32")
        per_layer[layer.name] = {
            "weight_mode": w_mode,
            "activation_mode": a_mode,
            "size_bytes": float(size_bytes),
            "size_kb": float(size_bytes / 1024.0),
            "size_bytes_fp32": float(size_bytes_fp32),
            "size_kb_fp32": float(size_bytes_fp32 / 1024.0),
            "latency_ms": float(latency_ms),
            "latency_ms_fp32": float(latency_ms_fp32),
        }
        total_size_bytes += size_bytes
        total_latency_ms += latency_ms
        total_size_bytes_fp32 += size_bytes_fp32
        total_latency_ms_fp32 += latency_ms_fp32

    return {
        "global": {
            "size_bytes": float(total_size_bytes),
            "size_kb": float(total_size_bytes / 1024.0),
            "size_bytes_fp32": float(total_size_bytes_fp32),
            "size_kb_fp32": float(total_size_bytes_fp32 / 1024.0),
            "latency_ms": float(total_latency_ms),
            "latency_ms_fp32": float(total_latency_ms_fp32),
        },
        "layers": per_layer,
    }
