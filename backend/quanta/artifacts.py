from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .model_inspect import iter_model_layers
from .model_ir import LayerMeta
from .quantization import QParams


@dataclass(frozen=True)
class RawEdge:
    source: str
    target: str
    source_tensor_index: int | None = None
    target_input_index: int | None = None


def _layer_input_sources(
    layer: tf.keras.layers.Layer, op_to_name: dict[int, str]
) -> list[tuple[str, int | None, int]]:
    names: list[tuple[str, int | None, int]] = []
    try:
        tensors = layer.input if isinstance(layer.input, list) else [layer.input]
        for input_idx, tensor in enumerate(tensors):
            history = getattr(tensor, "_keras_history", None)
            if history is not None and hasattr(history, "operation"):
                src = op_to_name.get(id(history.operation))
                if src is not None:
                    tensor_idx = getattr(history, "tensor_index", None)
                    names.append((src, tensor_idx, input_idx))
    except Exception:
        names = []
    if names:
        return names

    # Nested Functional models often expose internal placeholder tensors via
    # `layer.input`; recover true external parents from inbound node tensors.
    inbound_nodes = getattr(layer, "_inbound_nodes", [])
    if inbound_nodes:
        node = inbound_nodes[0]
        node_inputs = getattr(node, "input_tensors", None)
        if node_inputs is not None:
            tensors = node_inputs if isinstance(node_inputs, list) else [node_inputs]
            for input_idx, tensor in enumerate(tensors):
                history = getattr(tensor, "_keras_history", None)
                if history is not None and hasattr(history, "operation"):
                    src = op_to_name.get(id(history.operation))
                    if src is not None:
                        tensor_idx = getattr(history, "tensor_index", None)
                        names.append((src, tensor_idx, input_idx))
    return names


def build_graph(
    model: tf.keras.Model,
    metadata: list[LayerMeta],
    layer_metrics: dict[str, Any],
    layer_estimates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    layer_meta_map = {m.name: m for m in metadata}
    refs = iter_model_layers(model)
    ref_by_name = {ref.name: ref for ref in refs}
    container_names = {ref.name for ref in refs if ref.is_container}
    input_nodes: list[dict[str, Any]] = []
    input_op_to_name: dict[int, str] = {}

    for tensor in model.inputs:
        history = getattr(tensor, "_keras_history", None)
        op = getattr(history, "operation", None) if history is not None else None
        if op is None:
            continue
        input_name = op.name
        input_nodes.append(
            {
                "id": input_name,
                "name": input_name,
                "type": "InputLayer",
                "input_shape": str(getattr(op, "input_shape", None)),
                "output_shape": str(getattr(tensor, "shape", None)),
                "params": 0,
                "has_weights": False,
                "metrics": {"mae": 0.0, "rmse": 0.0, "kl": 0.0},
                "estimates": {},
                "parent": None,
                "is_container": False,
            }
        )
        input_op_to_name[id(op)] = input_name

    nodes.extend(input_nodes)

    for ref in refs:
        if ref.is_container:
            continue
        layer_meta = layer_meta_map.get(ref.name)
        nodes.append(
            {
                "id": ref.name,
                "name": ref.name,
                "type": ref.layer.__class__.__name__,
                "input_shape": str(getattr(ref.layer, "input_shape", None)),
                "output_shape": str(getattr(ref.layer, "output_shape", None)),
                "params": (layer_meta.params if layer_meta else 0),
                "has_weights": (
                    layer_meta.has_weights if layer_meta else bool(ref.layer.weights)
                ),
                "metrics": layer_metrics.get(
                    ref.name, {"mae": 0.0, "rmse": 0.0, "kl": 0.0}
                ),
                "estimates": (layer_estimates or {}).get(ref.name, {}),
                "parent": ref.parent,
                "is_container": False,
            }
        )

    op_to_name = {id(ref.layer): ref.name for ref in refs}
    op_to_name.update(input_op_to_name)
    raw_edges: list[RawEdge] = []
    for ref in refs:
        for src, src_tensor_idx, target_input_idx in _layer_input_sources(
            ref.layer, op_to_name
        ):
            raw_edges.append(
                RawEdge(
                    source=src,
                    target=ref.name,
                    source_tensor_index=src_tensor_idx,
                    target_input_index=target_input_idx,
                )
            )

    outgoing: dict[str, set[str]] = {}
    incoming: dict[str, set[str]] = {}
    for edge in raw_edges:
        outgoing.setdefault(edge.source, set()).add(edge.target)
        incoming.setdefault(edge.target, set()).add(edge.source)

    expanded_edges: set[tuple[str, str]] = {(e.source, e.target) for e in raw_edges}
    for container in container_names:
        descendants = {
            name
            for name, ref in ref_by_name.items()
            if (name.startswith(f"{container}/") and not ref.is_container)
        }
        if not descendants:
            continue
        container_ref = ref_by_name[container]
        if not isinstance(container_ref.layer, tf.keras.Model):
            continue

        # Map container input port idx -> inner entry leaf nodes fed from that port.
        in_port_ops: dict[int, int] = {}
        layer_inputs = getattr(container_ref.layer, "inputs", None)
        if layer_inputs is None:
            single_in = getattr(container_ref.layer, "input", None)
            layer_inputs = [single_in] if single_in is not None else []
        for idx, t in enumerate(layer_inputs):
            history = getattr(t, "_keras_history", None)
            op = getattr(history, "operation", None) if history is not None else None
            if op is not None:
                in_port_ops[id(op)] = idx
        entry_by_port: dict[int, set[str]] = {}
        for name in descendants:
            ref = ref_by_name[name]
            try:
                tensors = (
                    ref.layer.input
                    if isinstance(ref.layer.input, list)
                    else [ref.layer.input]
                )
            except Exception:
                tensors = []
            for tensor in tensors:
                history = getattr(tensor, "_keras_history", None)
                op = (
                    getattr(history, "operation", None) if history is not None else None
                )
                if op is None:
                    continue
                port_idx = in_port_ops.get(id(op))
                if port_idx is not None:
                    entry_by_port.setdefault(port_idx, set()).add(name)

        # Map container output port idx -> inner producing leaf node.
        sink_by_port: dict[int, str] = {}
        layer_outputs = getattr(container_ref.layer, "outputs", None)
        if layer_outputs is None:
            single_out = getattr(container_ref.layer, "output", None)
            layer_outputs = [single_out] if single_out is not None else []
        for idx, t in enumerate(layer_outputs):
            history = getattr(t, "_keras_history", None)
            op = getattr(history, "operation", None) if history is not None else None
            if op is None:
                continue
            src_name = op_to_name.get(id(op))
            if src_name in descendants:
                sink_by_port[idx] = src_name

        incoming_edges = [e for e in raw_edges if e.target == container]
        outgoing_edges = [e for e in raw_edges if e.source == container]

        # Remove direct container boundary edges.
        expanded_edges = {
            e
            for e in expanded_edges
            if not ((e[0] == container) or (e[1] == container))
        }

        # Reconnect predecessors to exact nested entry points by input index.
        for e in incoming_edges:
            pred = e.source
            if pred in descendants or pred == container:
                continue
            target_port = e.target_input_index or 0
            entry_targets = entry_by_port.get(target_port, set())
            if not entry_targets:
                # Fallback: connect to all descendants with no internal predecessors.
                entry_targets = {
                    d
                    for d in descendants
                    if not ((incoming.get(d, set())) & descendants)
                }
            for t in entry_targets:
                expanded_edges.add((pred, t))

        # Reconnect nested output producers to exact external consumers by output index.
        for e in outgoing_edges:
            succ = e.target
            if succ in descendants or succ == container:
                continue
            src_port = e.source_tensor_index or 0
            producer = sink_by_port.get(src_port)
            if producer is None:
                # Fallback: connect from all nested sinks.
                sinks = {
                    d
                    for d in descendants
                    if not ((outgoing.get(d, set())) & descendants)
                }
                for s in sinks:
                    expanded_edges.add((s, succ))
            else:
                expanded_edges.add((producer, succ))

    edges = [
        {"source": src, "target": dst}
        for src, dst in sorted(expanded_edges)
        if src not in container_names and dst not in container_names
    ]
    edge_set = {(e["source"], e["target"]) for e in edges}
    incoming_targets = {e["target"] for e in edges}
    input_node_names = {n["id"] for n in input_nodes}

    # Safety net: ensure input fanout is visible for expanded nested leaves.
    for ref in refs:
        if ref.is_container:
            continue
        if ref.name in incoming_targets:
            continue
        try:
            tensors = (
                ref.layer.input
                if isinstance(ref.layer.input, list)
                else [ref.layer.input]
            )
        except Exception:
            tensors = []
        for t in tensors:
            history = getattr(t, "_keras_history", None)
            op = getattr(history, "operation", None) if history is not None else None
            op_name = getattr(op, "name", None)
            if op_name in input_node_names and (op_name, ref.name) not in edge_set:
                edges.append({"source": op_name, "target": ref.name})
                edge_set.add((op_name, ref.name))
                incoming_targets.add(ref.name)

    return {"nodes": nodes, "edges": edges}


def _qparams_to_json(qparams: QParams) -> dict[str, Any]:
    return {
        "scale": np.asarray(qparams.scale).tolist(),
        "zero_point": np.asarray(qparams.zero_point).tolist(),
        "min_val": np.asarray(qparams.min_val).tolist(),
        "max_val": np.asarray(qparams.max_val).tolist(),
        "axis": qparams.axis,
    }


def write_artifacts(
    output_dir: Path,
    graph: dict[str, Any],
    metrics: dict[str, Any],
    activation_distributions: dict[str, Any],
    weight_distributions: dict[str, Any],
    bias_distributions: dict[str, Any],
    activation_qparams: dict[str, QParams],
    weight_info: dict[str, Any],
    layer_modes: dict[str, Any] | None = None,
    estimates: dict[str, Any] | None = None,
    include_distributions: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graph.json").write_text(json.dumps(graph, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if include_distributions:
        (output_dir / "distributions.json").write_text(
            json.dumps(
                {
                    "activations": activation_distributions,
                    "weights": weight_distributions,
                    "biases": bias_distributions,
                },
                indent=2,
            )
        )
    qparams_payload = {
        "activations": {k: _qparams_to_json(v) for k, v in activation_qparams.items()},
        "weights": {
            k: _qparams_to_json(v["qparams"])
            for k, v in weight_info.items()
            if v.get("qparams") is not None
        },
        "biases": {
            k: _qparams_to_json(v["bias"]["qparams"])
            for k, v in weight_info.items()
            if v.get("bias") is not None and v["bias"].get("qparams") is not None
        },
        "layer_modes": layer_modes or {"weight": {}, "activation": {}},
    }
    (output_dir / "qparams.json").write_text(json.dumps(qparams_payload, indent=2))
    (output_dir / "estimates.json").write_text(json.dumps(estimates or {}, indent=2))


def finalize_run_artifacts(
    tmp_dir: Path,
    final_dir: Path,
    profile: str,
    run_id: str,
    started_at: str,
    completed_at: str,
) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)

    required_files = (
        "graph.json",
        "metrics.json",
        "qparams.json",
        "estimates.json",
        "distributions.json",
    )
    optional_files: tuple[str, ...] = ()
    copied_files: list[str] = []

    for name in required_files + optional_files:
        src = tmp_dir / name
        if not src.exists():
            if name in required_files:
                raise FileNotFoundError(
                    f"Required artifact is missing in tmp run: {src}"
                )
            continue
        dst = final_dir / name
        dst.write_text(src.read_text())
        copied_files.append(name)

    run_meta = {
        "status": "complete",
        "run_id": run_id,
        "profile": profile,
        "started_at": started_at,
        "completed_at": completed_at,
        "artifacts": copied_files,
    }
    (final_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
