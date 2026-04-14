from __future__ import annotations

from typing import Any

import numpy as np


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def kl_divergence(a: np.ndarray, b: np.ndarray, bins: int = 128) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    lo = float(min(a_flat.min(initial=0.0), b_flat.min(initial=0.0)))
    hi = float(max(a_flat.max(initial=1.0), b_flat.max(initial=1.0)))
    if hi <= lo:
        hi = lo + 1e-6
    hist_a, edges = np.histogram(a_flat, bins=bins, range=(lo, hi), density=False)
    hist_b, _ = np.histogram(b_flat, bins=edges, density=False)
    p = hist_a.astype(np.float64) + 1e-8
    q = hist_b.astype(np.float64) + 1e-8
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _channel_metric(a: np.ndarray, b: np.ndarray, metric: str) -> list[float]:
    if a.ndim <= 1:
        if metric == "mae":
            fn = mae
        elif metric == "rmse":
            fn = rmse
        else:
            fn = kl_divergence
        return [fn(a, b)]
    moved_a = np.moveaxis(a, -1, -1)
    moved_b = np.moveaxis(b, -1, -1)
    out: list[float] = []
    for c in range(moved_a.shape[-1]):
        if metric == "mae":
            fn = mae
        elif metric == "rmse":
            fn = rmse
        else:
            fn = kl_divergence
        out.append(fn(moved_a[..., c], moved_b[..., c]))
    return out


def compute_metrics(
    fp32_outputs: dict[str, np.ndarray], dequant_outputs: dict[str, np.ndarray]
) -> dict[str, Any]:
    layer_metrics: dict[str, Any] = {}
    maes: list[float] = []
    rmses: list[float] = []
    kls: list[float] = []

    for layer_name, fp in fp32_outputs.items():
        deq = dequant_outputs[layer_name]
        layer_mae = mae(fp, deq)
        layer_rmse = rmse(fp, deq)
        layer_kl = kl_divergence(fp, deq)
        maes.append(layer_mae)
        rmses.append(layer_rmse)
        kls.append(layer_kl)
        layer_metrics[layer_name] = {
            "mae": layer_mae,
            "rmse": layer_rmse,
            "kl": layer_kl,
            "channel_mae": _channel_metric(fp, deq, "mae"),
            "channel_rmse": _channel_metric(fp, deq, "rmse"),
            "channel_kl": _channel_metric(fp, deq, "kl"),
        }

    return {
        "model": {
            "mae": float(np.mean(maes)) if maes else 0.0,
            "rmse": float(np.mean(rmses)) if rmses else 0.0,
            "kl": float(np.mean(kls)) if kls else 0.0,
        },
        "layers": layer_metrics,
    }
