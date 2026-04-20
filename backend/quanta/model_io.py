from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import tensorflow as tf

from .model_inspect import iter_leaf_layers
from .model_ir import LayerMeta, UnifiedModel


@dataclass
class TorchScriptModel:
    model: Any
    path: Path


class ConversionStageError(RuntimeError):
    """Error tagged with conversion stage for actionable diagnostics."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


def _enable_unsafe_deserialization_if_available() -> None:
    if hasattr(tf.keras, "config") and hasattr(
        tf.keras.config, "enable_unsafe_deserialization"
    ):
        tf.keras.config.enable_unsafe_deserialization()
    try:
        from tf_keras.src.saving import serialization_lib

        if hasattr(serialization_lib, "enable_unsafe_deserialization"):
            serialization_lib.enable_unsafe_deserialization()
    except Exception:
        # Optional: tf_keras may not be installed for pure TF flows.
        pass


class ModelLoader:
    """Loader step: read supported model artifacts (.keras / TorchScript .pt)."""

    def load(
        self, model_path: str, custom_objects: dict[str, Any] | None = None
    ) -> Any:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file/folder not found: {path}")

        if path.is_dir() and (path / "saved_model.pb").exists():
            raise ValueError(
                "SavedModel format is no longer supported in Quanta. "
                "Please provide a Keras model file (for example .keras)."
            )

        if path.suffix.lower() == ".pt":
            return self._load_torchscript(path)

        # Quanta performs inference-only analysis; skip compile-time objects
        # (optimizer/loss/metrics) to reduce custom object requirements.
        model = tf.keras.models.load_model(
            path, custom_objects=custom_objects, compile=False
        )
        return _ensure_unique_layer_names(model)

    def _load_torchscript(self, path: Path) -> TorchScriptModel:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - dependency edge
            raise ConversionStageError(
                "pt_load",
                "PyTorch dependency is required for .pt models. Install torch.",
            ) from exc
        try:
            scripted = torch.jit.load(str(path), map_location="cpu")
            scripted.eval()
        except Exception as exc:
            raise ConversionStageError(
                "pt_load",
                f"Failed to load TorchScript model from '{path}': {exc}",
            ) from exc
        return TorchScriptModel(model=scripted, path=path)


class ModelConverter:
    """Converter step: convert loaded Keras model into UnifiedModel IR."""

    def convert(
        self, model: Any, sample_shape: tuple[int, ...] | None = None
    ) -> UnifiedModel:
        if isinstance(model, TorchScriptModel):
            _enable_unsafe_deserialization_if_available()
            keras_model = self._convert_torchscript_to_keras(
                model, sample_shape=sample_shape
            )
            return UnifiedModel(
                model=_ensure_unique_layer_names(keras_model),
                metadata=self._metadata_from_keras(keras_model),
                graph=None,
            )
        return UnifiedModel(
            model=model, metadata=self._metadata_from_keras(model), graph=None
        )

    def _metadata_from_keras(self, model: tf.keras.Model) -> list[LayerMeta]:
        meta: list[LayerMeta] = []
        for ref in iter_leaf_layers(model):
            layer = ref.layer
            meta.append(
                LayerMeta(
                    name=ref.name,
                    layer_type=layer.__class__.__name__,
                    input_shape=getattr(layer, "input_shape", None),
                    output_shape=getattr(layer, "output_shape", None),
                    params=layer.count_params(),
                    has_weights=bool(layer.weights),
                )
            )
        return meta

    def _convert_torchscript_to_keras(
        self, ts_model: TorchScriptModel, sample_shape: tuple[int, ...] | None = None
    ) -> tf.keras.Model:
        try:
            import onnx
        except Exception as exc:  # pragma: no cover - dependency edge
            raise ConversionStageError(
                "onnx_export",
                "ONNX dependency is required for .pt conversion. Install onnx.",
            ) from exc
        try:
            import torch
        except Exception as exc:  # pragma: no cover - dependency edge
            raise ConversionStageError(
                "onnx_export",
                "PyTorch dependency is required for .pt conversion. Install torch.",
            ) from exc
        try:
            from onnx2keras import onnx_to_keras
        except Exception as exc:  # pragma: no cover - dependency edge
            raise ConversionStageError(
                "onnx_to_tf",
                "onnx2keras dependency is required for ONNX->TF conversion. Install onnx2keras.",
            ) from exc
        _enable_unsafe_deserialization_if_available()

        input_name = "input_0"
        input_shape: tuple[int, ...] = ()
        if sample_shape:
            input_shape = (1, *sample_shape)
        else:
            input_name, input_shape = self._infer_torchscript_input(ts_model.model)
        if not input_shape:
            raise ConversionStageError(
                "onnx_export",
                "Could not infer TorchScript input shape. Provide a representative dataset with static sample shape.",
            )

        with tempfile.TemporaryDirectory() as td:
            onnx_path = Path(td) / "model.onnx"
            try:
                dummy = torch.randn(*input_shape)
                export_kwargs = {
                    "input_names": [input_name],
                    "output_names": ["output"],
                    "opset_version": 18,
                    "do_constant_folding": True,
                }
                try:
                    # Prefer TorchScript-compatible export path. Newer PyTorch
                    # defaults to torch.export (dynamo=True), which rejects ScriptModule.
                    torch.onnx.export(
                        ts_model.model,
                        dummy,
                        str(onnx_path),
                        dynamo=False,
                        **export_kwargs,
                    )
                except TypeError:
                    # Older PyTorch versions may not support the `dynamo` kwarg.
                    torch.onnx.export(
                        ts_model.model,
                        dummy,
                        str(onnx_path),
                        **export_kwargs,
                    )
            except Exception as exc:
                raise ConversionStageError(
                    "onnx_export",
                    f"Failed exporting TorchScript -> ONNX: {exc}",
                ) from exc

            try:
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
            except Exception as exc:
                raise ConversionStageError(
                    "onnx_validate",
                    f"Exported ONNX model is invalid: {exc}",
                ) from exc

            self._sanitize_onnx_names(onnx_model)

            try:
                keras_model = onnx_to_keras(
                    onnx_model,
                    [input_name],
                    change_ordering=True,
                    name_policy="short",
                )
            except Exception as exc:
                # Some ONNX graphs still leak "/" into generated layer names
                # through internal helper layers (e.g. *_pad). Retry with full
                # renumeration to guarantee Keras-safe names.
                if "cannot contain character '/'" in str(exc):
                    try:
                        keras_model = onnx_to_keras(
                            onnx_model,
                            [input_name],
                            change_ordering=True,
                            name_policy="renumerate",
                        )
                    except Exception as exc2:
                        raise ConversionStageError(
                            "onnx_to_tf",
                            f"Failed converting ONNX -> TF/Keras after renumeration fallback: {exc2}",
                        ) from exc2
                elif "Unrecognized keyword arguments passed to Conv2D" in str(exc) and (
                    "weights" in str(exc)
                ):
                    # Keras 3 rejects `weights=` in layer constructors, but
                    # onnx2keras still uses this legacy style for some ops.
                    # Retry conversion in a clean subprocess with legacy Keras.
                    try:
                        keras_model = self._convert_onnx_via_legacy_keras_subprocess(
                            onnx_path, input_name
                        )
                    except Exception as exc2:
                        raise ConversionStageError(
                            "onnx_to_tf",
                            f"Failed converting ONNX -> TF/Keras with legacy subprocess fallback: {exc2}",
                        ) from exc2
                else:
                    raise ConversionStageError(
                        "onnx_to_tf",
                        f"Failed converting ONNX -> TF/Keras: {exc}",
                    ) from exc

        return keras_model

    def _sanitize_onnx_names(self, onnx_model: Any) -> None:
        """Normalize ONNX names to Keras-safe identifiers.

        onnx2keras can create helper layer names based on raw ONNX node names
        (e.g. "<node>_pad"). If node names contain "/" then Keras rejects them.
        """

        def _sanitize(value: str) -> str:
            if not value:
                return value
            cleaned = value.replace("::", "__")
            cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", cleaned)
            cleaned = re.sub(r"_+", "_", cleaned).strip("_")
            if not cleaned:
                cleaned = "unnamed"
            if cleaned[0].isdigit():
                cleaned = f"n_{cleaned}"
            return cleaned

        rename: dict[str, str] = {}

        # Build rename map for every graph value name.
        graph = onnx_model.graph
        for tensor in list(graph.input) + list(graph.output) + list(graph.value_info):
            old = tensor.name
            new = _sanitize(old)
            if new != old:
                # keep unique mapping
                suffix = 1
                base = new
                while new in rename.values():
                    suffix += 1
                    new = f"{base}_{suffix}"
                rename[old] = new
                tensor.name = new

        for init in graph.initializer:
            old = init.name
            new = rename.get(old, _sanitize(old))
            if new != old:
                suffix = 1
                base = new
                while new in rename.values():
                    suffix += 1
                    new = f"{base}_{suffix}"
                rename[old] = new
                init.name = new

        # Rename node names and their input/output references.
        for node in graph.node:
            if node.name:
                node.name = _sanitize(node.name)
            for i, inp in enumerate(node.input):
                if inp in rename:
                    node.input[i] = rename[inp]
            for i, out in enumerate(node.output):
                new_out = rename.get(out, _sanitize(out))
                if new_out != out:
                    rename[out] = new_out
                    node.output[i] = new_out

    def _convert_onnx_via_legacy_keras_subprocess(
        self, onnx_path: Path, input_name: str
    ) -> tf.keras.Model:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "converted.keras"
            script = r"""
import os
import sys
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf
from tf_keras.src.saving import serialization_lib

if hasattr(tf.keras, "config") and hasattr(tf.keras.config, "enable_unsafe_deserialization"):
    tf.keras.config.enable_unsafe_deserialization()
if hasattr(serialization_lib, "enable_unsafe_deserialization"):
    serialization_lib.enable_unsafe_deserialization()

onnx_path = sys.argv[1]
input_name = sys.argv[2]
out_path = sys.argv[3]

model = onnx.load(onnx_path)
k = onnx_to_keras(model, [input_name], change_ordering=True, name_policy="renumerate")
k.save(out_path)
"""
            env = dict(os.environ)
            env.setdefault("TF_USE_LEGACY_KERAS", "1")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(onnx_path),
                    input_name,
                    str(out_path),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0 or not out_path.exists():
                raise RuntimeError(
                    f"legacy subprocess failed (code={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
                )
            try:
                import tf_keras
            except Exception as exc:
                raise RuntimeError(
                    "Legacy conversion produced a tf_keras model, but tf_keras is not importable."
                ) from exc
            return tf_keras.models.load_model(out_path, compile=False, safe_mode=False)

    def _infer_torchscript_input(self, scripted: Any) -> tuple[str, tuple[int, ...]]:
        """Best-effort TorchScript shape inference (may fail on some JIT graphs)."""
        try:
            graph = scripted.graph
            for idx, inp in enumerate(graph.inputs()):
                name = inp.debugName()
                if name == "self":
                    continue
                t = inp.type()
                sizes = getattr(t, "sizes", lambda: None)()
                if not sizes:
                    continue
                shape: list[int] = []
                for dim in sizes:
                    if dim is None or dim <= 0:
                        shape.append(1)
                    else:
                        shape.append(int(dim))
                if shape:
                    if len(shape) < 2:
                        shape = [1, 1]
                    return f"input_{idx}", tuple(shape)
        except Exception:
            return "input_0", ()
        return "input_0", ()


def _ensure_unique_layer_names(model: tf.keras.Model) -> tf.keras.Model:
    # Always normalize names via clone to avoid edge cases where duplicate
    # operation names are produced by externally serialized models.
    seen: dict[str, int] = {}

    def _clone_with_name(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        config = layer.get_config()
        base_name = config.get("name", layer.name)
        idx = seen.get(base_name, 0)
        seen[base_name] = idx + 1
        config["name"] = base_name if idx == 0 else f"{base_name}_{idx}"
        return layer.__class__.from_config(config)

    try:
        cloned = tf.keras.models.clone_model(model, clone_function=_clone_with_name)
        cloned.set_weights(model.get_weights())
        return cloned
    except Exception:
        # Keep original model if cloning is not possible for a custom object.
        return model
