from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Any

import numpy as np
import tensorflow as tf

from .model_inspect import LayerRef, iter_leaf_layers


MAX_DIST_SAMPLES = 2048
RANGE_MODES = {"minmax", "clip99_99", "clip99_999"}
LAYER_MODES = {"int2", "int4", "int8", "int12", "int16", "fp16", "fp32"}
INT8_MIN, INT8_MAX = -(2 ** 7), (2 ** 7) - 1


@dataclass
class QParams:
    scale: np.ndarray
    zero_point: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray
    axis: int | None


def _safe_scale(min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    scale = (max_val - min_val) / float((2**8 - 1))
    return np.where(scale == 0, 1e-8, scale)


def _int_bounds(bit_width: int) -> tuple[int, int]:
    qmin = -(2 ** (bit_width - 1))
    qmax = (2 ** (bit_width - 1)) - 1
    return qmin, qmax


def calc_qparams(values: np.ndarray, axis: int | None = None, range_mode: str = "minmax", bit_width: int = 8) -> QParams:
    if range_mode not in RANGE_MODES:
        raise ValueError(f"Unsupported range mode: {range_mode}")
    if axis is None or values.ndim == 0:
        if range_mode == "minmax":
            min_val = np.array(values.min(), dtype=np.float32)
            max_val = np.array(values.max(), dtype=np.float32)
        else:
            p = 99.99 if range_mode == "clip99_99" else 99.999
            lo = float((100.0 - p) / 2.0)
            hi = float(100.0 - lo)
            min_val = np.array(np.percentile(values, lo), dtype=np.float32)
            max_val = np.array(np.percentile(values, hi), dtype=np.float32)
    else:
        reduce_axes = tuple(i for i in range(values.ndim) if i != axis)
        if range_mode == "minmax":
            min_val = values.min(axis=reduce_axes).astype(np.float32)
            max_val = values.max(axis=reduce_axes).astype(np.float32)
        else:
            p = 99.99 if range_mode == "clip99_99" else 99.999
            lo = float((100.0 - p) / 2.0)
            hi = float(100.0 - lo)
            min_val = np.percentile(values, lo, axis=reduce_axes).astype(np.float32)
            max_val = np.percentile(values, hi, axis=reduce_axes).astype(np.float32)
    qmin, qmax = _int_bounds(bit_width)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scale = ((max_val - min_val) / float(qmax - qmin)).astype(np.float32)
    scale = np.where(np.isfinite(scale), scale, 1e-8)
    scale = np.where(scale == 0, 1e-8, scale)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        zp_float = np.round(qmin - (min_val / scale))
    zp_float = np.nan_to_num(zp_float, nan=0.0, posinf=float(qmax), neginf=float(qmin))
    zero_point = np.clip(zp_float, qmin, qmax).astype(np.int32)
    return QParams(scale=scale, zero_point=zero_point, min_val=min_val, max_val=max_val, axis=axis)


def quantize_dequantize(values: np.ndarray, qparams: QParams, bit_width: int = 8) -> np.ndarray:
    qmin, qmax = _int_bounds(bit_width)
    if qparams.axis is None or np.isscalar(qparams.scale):
        q = np.round(values / qparams.scale + qparams.zero_point)
        q = np.clip(q, qmin, qmax)
        return (q - qparams.zero_point) * qparams.scale
    shape = [1] * values.ndim
    shape[qparams.axis] = -1
    scale = np.reshape(qparams.scale, shape)
    zp = np.reshape(qparams.zero_point, shape)
    q = np.round(values / scale + zp)
    q = np.clip(q, qmin, qmax)
    return (q - zp) * scale


def apply_layer_mode(values: np.ndarray, mode: str, qparams: QParams | None) -> np.ndarray:
    if mode == "fp32" or qparams is None:
        return values.astype(np.float32)
    if mode == "fp16":
        return values.astype(np.float16).astype(np.float32)
    bit_width = int(mode.replace("int", ""))
    return quantize_dequantize(values, qparams, bit_width=bit_width).astype(np.float32)


def _apply_layer_mode_tensor(values: tf.Tensor, mode: str, qparams: QParams | None) -> tf.Tensor:
    if mode == "fp32" or qparams is None:
        return tf.cast(values, tf.float32)
    if mode == "fp16":
        return tf.cast(tf.cast(values, tf.float16), tf.float32)
    bit_width = int(mode.replace("int", ""))
    qmin, qmax = _int_bounds(bit_width)
    scale = tf.cast(tf.convert_to_tensor(qparams.scale), tf.float32)
    zp = tf.cast(tf.convert_to_tensor(qparams.zero_point), tf.float32)
    if qparams.axis is not None:
        shape = [1] * len(values.shape)
        shape[qparams.axis] = -1
        scale = tf.reshape(scale, shape)
        zp = tf.reshape(zp, shape)
    q = tf.round(values / scale + zp)
    q = tf.clip_by_value(q, qmin, qmax)
    return (q - zp) * scale


def _patch_layer_activation_modes(
    model: tf.keras.Model,
    activation_mode_map: dict[str, str],
    activation_qparams: dict[str, QParams],
) -> None:
    for ref in iter_leaf_layers(model):
        layer = ref.layer
        mode = activation_mode_map.get(ref.name, "int8")
        qparams = activation_qparams.get(ref.name)
        original_call = layer.call

        def _wrapped_call(self, *args, __orig_call=original_call, __mode=mode, __qparams=qparams, **kwargs):
            out = __orig_call(*args, **kwargs)
            if isinstance(out, dict):
                return {k: _apply_layer_mode_tensor(v, __mode, __qparams) for k, v in out.items()}
            if isinstance(out, (list, tuple)):
                return type(out)(_apply_layer_mode_tensor(o, __mode, __qparams) for o in out)
            return _apply_layer_mode_tensor(out, __mode, __qparams)

        layer.call = types.MethodType(_wrapped_call, layer)


def _sample_channel_values(values: np.ndarray, axis: int | None) -> dict[str, list[float]]:
    if axis is None or values.ndim == 1:
        vals = values.reshape(-1)
        sampled = vals[:MAX_DIST_SAMPLES] if vals.size > MAX_DIST_SAMPLES else vals
        return {"0": sampled.astype(np.float32).tolist()}
    moved = np.moveaxis(values, axis, -1)
    channels = moved.shape[-1]
    out: dict[str, list[float]] = {}
    for c in range(channels):
        vals = moved[..., c].reshape(-1)
        sampled = vals[:MAX_DIST_SAMPLES] if vals.size > MAX_DIST_SAMPLES else vals
        out[str(c)] = sampled.astype(np.float32).tolist()
    return out


def _infer_weight_axis(weights: np.ndarray) -> int | None:
    if weights.ndim in (2, 4, 5):
        return -1
    return None


def _pick_bad_layers(model: tf.keras.Model) -> set[str]:
    refs = iter_leaf_layers(model)
    conv_layer = next((ref.name for ref in refs if isinstance(ref.layer, tf.keras.layers.Conv2D)), None)
    dense_layer = next((ref.name for ref in refs if isinstance(ref.layer, tf.keras.layers.Dense)), None)
    selected = {name for name in (conv_layer, dense_layer) if name is not None}
    return selected


def _corrupt_qparams(qparams: QParams) -> QParams:
    # Deliberately distort qparams to simulate calibration mistakes.
    scale = np.asarray(qparams.scale, dtype=np.float32) * 25.0
    zero_point = np.asarray(qparams.zero_point, dtype=np.int32) + 60
    zero_point = np.clip(zero_point, INT8_MIN, INT8_MAX)
    return QParams(
        scale=scale,
        zero_point=zero_point,
        min_val=qparams.min_val,
        max_val=qparams.max_val,
        axis=qparams.axis,
    )


def _metric_error_values(diff: np.ndarray, metric: str) -> np.ndarray:
    if metric == "rmse":
        return np.square(diff)
    return np.abs(diff)


def _to_primary_tensor(out: Any) -> tf.Tensor:
    if isinstance(out, dict):
        return next(iter(out.values()))
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def _build_model_feed(model: tf.keras.Model, batch: np.ndarray) -> Any:
    # Feed by input name to avoid Keras structure warnings for named inputs.
    if len(model.inputs) == 1:
        name = model.inputs[0].name.split(":")[0]
        return {name: batch}
    return batch


def _capture_layer_outputs(
    model: tf.keras.Model,
    layer_refs: list[LayerRef],
    dataset_batches: list[np.ndarray],
) -> dict[str, np.ndarray]:
    per_layer: dict[str, list[np.ndarray]] = {ref.name: [] for ref in layer_refs}
    originals: list[tuple[tf.keras.layers.Layer, Any]] = []

    try:
        for ref in layer_refs:
            layer = ref.layer
            original_call = layer.call
            originals.append((layer, original_call))

            def _wrapped_call(self, *args, __orig_call=original_call, __name=ref.name, **kwargs):
                out = __orig_call(*args, **kwargs)
                tensor = _to_primary_tensor(out)
                per_layer[__name].append(np.array(tensor))
                return out

            layer.call = types.MethodType(_wrapped_call, layer)

        for batch in dataset_batches:
            model(_build_model_feed(model, batch), training=False)
    finally:
        for layer, original_call in originals:
            layer.call = original_call

    return {name: np.concatenate(chunks, axis=0) for name, chunks in per_layer.items() if chunks}


def collect_activation_ranges(
    model: tf.keras.Model,
    dataset_batches: list[np.ndarray],
    layer_refs: list[LayerRef],
    range_mode: str,
    activation_modes: dict[str, str],
) -> tuple[dict[str, QParams], dict[str, np.ndarray]]:
    fp32_outputs = _capture_layer_outputs(model, layer_refs, dataset_batches)
    activation_qparams: dict[str, QParams] = {}
    for name, merged in fp32_outputs.items():
        axis = -1 if merged.ndim > 1 else None
        mode = activation_modes.get(name, "int8")
        if mode.startswith("int"):
            bit_width = int(mode.replace("int", ""))
            activation_qparams[name] = calc_qparams(merged, axis=axis, range_mode=range_mode, bit_width=bit_width)
    return activation_qparams, fp32_outputs


def quantize_model_weights(
    model: tf.keras.Model,
    intentionally_bad_layers: set[str] | None = None,
    range_mode: str = "minmax",
    weight_modes: dict[str, str] | None = None,
) -> tuple[tf.keras.Model, dict[str, Any]]:
    cloned = tf.keras.models.clone_model(model)
    cloned.set_weights(model.get_weights())
    weight_info: dict[str, Any] = {}
    bad_layers = intentionally_bad_layers or set()
    mode_map = weight_modes or {}

    for ref in iter_leaf_layers(cloned):
        layer = ref.layer
        weights = layer.get_weights()
        if not weights:
            continue
        kernel = weights[0]
        axis = _infer_weight_axis(kernel)
        mode = mode_map.get(ref.name, "int8")
        q = None
        if mode.startswith("int"):
            bit_width = int(mode.replace("int", ""))
            q = calc_qparams(kernel, axis=axis, range_mode=range_mode, bit_width=bit_width)
            if ref.name in bad_layers:
                q = _corrupt_qparams(q)
            qdq = quantize_dequantize(kernel, q, bit_width=bit_width).astype(np.float32)
        elif mode == "fp16":
            qdq = kernel.astype(np.float16).astype(np.float32)
        else:
            qdq = kernel.astype(np.float32)
        bias_info = None
        if len(weights) > 1:
            bias = weights[1].astype(np.float32)
            bias_qdq = bias
            bias_qparams = None
            if mode.startswith("int"):
                # Bias simulation in integer pipeline: use int32 QDQ.
                bias_qparams = calc_qparams(bias, axis=None, range_mode=range_mode, bit_width=32)
                bias_qdq = quantize_dequantize(bias, bias_qparams, bit_width=32).astype(np.float32)
            new_weights = [qdq, bias_qdq, *weights[2:]]
            bias_info = {
                "original": bias,
                "qdq": bias_qdq,
                "qparams": bias_qparams,
                "axis": None,
            }
        else:
            new_weights = [qdq, *weights[1:]]
        layer.set_weights(new_weights)
        weight_info[ref.name] = {
            "original": kernel.astype(np.float32),
            "qdq": qdq,
            "qparams": q,
            "axis": axis,
            "mode": mode,
            "bias": bias_info,
        }
    return cloned, weight_info


def run_fake_quant_pipeline(
    model: tf.keras.Model,
    dataset_batches: list[np.ndarray],
    intentionally_bad_layers_count: int = 0,
    range_mode: str = "minmax",
    global_weight_mode: str = "int8",
    global_activation_mode: str = "int8",
    layer_weight_modes: dict[str, str] | None = None,
    layer_activation_modes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if global_weight_mode not in LAYER_MODES:
        raise ValueError(f"Unsupported global weight mode: {global_weight_mode}")
    if global_activation_mode not in LAYER_MODES:
        raise ValueError(f"Unsupported global activation mode: {global_activation_mode}")
    if not model.inputs:
        # Ensure functional graph tensors exist for loaded Sequential variants.
        model(_build_model_feed(model, dataset_batches[0]), training=False)
    tracked_refs = iter_leaf_layers(model)
    tracked_names = [ref.name for ref in tracked_refs]
    weight_mode_map = {name: global_weight_mode for name in tracked_names}
    activation_mode_map = {name: global_activation_mode for name in tracked_names}
    if layer_weight_modes:
        for k, v in layer_weight_modes.items():
            if v in LAYER_MODES and k in weight_mode_map:
                weight_mode_map[k] = v
    if layer_activation_modes:
        for k, v in layer_activation_modes.items():
            if v in LAYER_MODES and k in activation_mode_map:
                activation_mode_map[k] = v
    bad_layers = _pick_bad_layers(model) if intentionally_bad_layers_count > 0 else set()
    activation_qparams, fp32_outputs = collect_activation_ranges(
        model,
        dataset_batches,
        tracked_refs,
        range_mode,
        activation_mode_map,
    )
    for layer_name in bad_layers:
        if layer_name in activation_qparams and activation_mode_map.get(layer_name, "int8").startswith("int"):
            activation_qparams[layer_name] = _corrupt_qparams(activation_qparams[layer_name])

    quant_model, weight_info = quantize_model_weights(
        model,
        intentionally_bad_layers=bad_layers,
        range_mode=range_mode,
        weight_modes=weight_mode_map,
    )
    if not quant_model.inputs:
        quant_model(_build_model_feed(quant_model, dataset_batches[0]), training=False)
    quant_tracked_refs = iter_leaf_layers(quant_model)

    _patch_layer_activation_modes(quant_model, activation_mode_map, activation_qparams)
    q_outputs = _capture_layer_outputs(quant_model, quant_tracked_refs, dataset_batches)

    dequant_outputs: dict[str, np.ndarray] = {}
    activation_distributions: dict[str, dict[str, dict[str, list[float]]]] = {}
    for name, merged in q_outputs.items():
        # Already mode-applied inside forward pass; this is the true propagated output.
        deq = merged.astype(np.float32)
        dequant_outputs[name] = deq
        act_errors = fp32_outputs[name] - deq
        axis = activation_qparams[name].axis if name in activation_qparams else (-1 if deq.ndim > 1 else None)
        activation_distributions[name] = {
            "pre": _sample_channel_values(fp32_outputs[name], axis),
            "post": _sample_channel_values(deq, axis),
            "error_mae": _sample_channel_values(_metric_error_values(act_errors, "mae"), axis),
            "error_rmse": _sample_channel_values(_metric_error_values(act_errors, "rmse"), axis),
        }

    weight_distributions: dict[str, dict[str, dict[str, list[float]]]] = {}
    bias_distributions: dict[str, dict[str, dict[str, list[float]]]] = {}
    for name, info in weight_info.items():
        w_err = info["original"] - info["qdq"]
        axis = info["axis"]
        weight_distributions[name] = {
            "pre": _sample_channel_values(info["original"], axis),
            "post": _sample_channel_values(info["qdq"], axis),
            "error_mae": _sample_channel_values(_metric_error_values(w_err, "mae"), axis),
            "error_rmse": _sample_channel_values(_metric_error_values(w_err, "rmse"), axis),
        }
        bias = info.get("bias")
        if bias is not None:
            b_err = bias["original"] - bias["qdq"]
            bias_distributions[name] = {
                "pre": _sample_channel_values(bias["original"], None),
                "post": _sample_channel_values(bias["qdq"], None),
                "error_mae": _sample_channel_values(_metric_error_values(b_err, "mae"), None),
                "error_rmse": _sample_channel_values(_metric_error_values(b_err, "rmse"), None),
            }

    return {
        "activation_qparams": activation_qparams,
        "fp32_outputs": fp32_outputs,
        "dequant_outputs": dequant_outputs,
        "weight_info": weight_info,
        "activation_distributions": activation_distributions,
        "weight_distributions": weight_distributions,
        "bias_distributions": bias_distributions,
        "layer_modes": {"weight": weight_mode_map, "activation": activation_mode_map},
    }
