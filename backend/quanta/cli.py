from __future__ import annotations

import argparse
import atexit
import importlib
import importlib.util
from importlib import resources
import json
import os
from pathlib import Path
import sys
import webbrowser
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quanta quantization visualizer")
    parser.add_argument(
        "--model", required=False, help="Path to Keras model file (.keras)"
    )
    parser.add_argument(
        "--dataset", required=False, help="Path to representative dataset .npy"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with bundled MNIST CNN test assets (overrides --model/--dataset)",
    )
    parser.add_argument(
        "--test-bad",
        action="store_true",
        help="Intentionally quantize two layers badly (debug mode)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--range-mode",
        choices=["minmax", "clip99_99", "clip99_999"],
        default="minmax",
        help="Dynamic range aggregation mode",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["int2", "int4", "int8", "int12", "int16", "fp16", "fp32"],
        default="int8",
        help="Global weight precision mode",
    )
    parser.add_argument(
        "--activation-mode",
        choices=["int2", "int4", "int8", "int12", "int16", "fp16", "fp32"],
        default="int8",
        help="Global activation precision mode",
    )
    parser.add_argument(
        "--final-artifact-profile",
        choices=["minimal", "full"],
        default="minimal",
        help="Storage profile for finalized run artifacts",
    )
    parser.add_argument(
        "--store-distributions",
        action="store_true",
        help="Store distributions in finalized run artifacts (equivalent to --final-artifact-profile full)",
    )
    parser.add_argument(
        "--max-cached-runs",
        type=int,
        default=3,
        help="How many finalized run_* directories to retain",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU execution by disabling visible GPU devices",
    )
    parser.add_argument(
        "--custom-objects",
        default=None,
        help='Inline JSON mapping, e.g. \'{"MyLayer":"mypkg.layers:MyLayer"}\'',
    )
    parser.add_argument(
        "--custom-objects-file",
        default=None,
        help="Path to JSON file with custom object mapping: name -> module:attribute",
    )
    parser.add_argument(
        "--custom-objects-module",
        default=None,
        help="Python module exposing CUSTOM_OBJECTS dict or get_custom_objects()",
    )
    parser.add_argument("--no-ui", action="store_true", help="Do not auto-open browser")
    return parser


def _bundled_test_assets() -> tuple[str, str]:
    model_resource = resources.files("quanta.assets").joinpath("mnist_cnn.keras")
    dataset_resource = resources.files("quanta.assets").joinpath("mnist_test.npy")
    if not model_resource.is_file() or not dataset_resource.is_file():
        raise FileNotFoundError(
            "Bundled test assets are missing. Expected quanta.assets/mnist_cnn.keras and mnist_test.npy"
        )
    return str(model_resource), str(dataset_resource)


def _parse_custom_object_specs(mapping: dict[str, Any], source: str) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise SystemExit(
                f"{source}: custom object key must be a string, got {type(key).__name__}"
            )
        if not isinstance(value, str) or ":" not in value:
            raise SystemExit(
                f"{source}: value for '{key}' must be a string in format 'module:attribute', got {value!r}"
            )
        module_name, attr_name = value.split(":", 1)
        if not module_name or not attr_name:
            raise SystemExit(f"{source}: invalid import spec for '{key}': {value!r}")
        try:
            module = importlib.import_module(module_name)
        except Exception:
            # Fallback for local repo files without Python packaging (__init__.py).
            # Supports "path/to/file.py:Symbol" and "path.to.file:Symbol" by resolving a local file.
            candidate_paths = []
            if module_name.endswith(".py"):
                candidate_paths.append(Path(module_name))
            candidate_paths.append(Path(module_name.replace(".", "/") + ".py"))

            module = None
            first_exc: Exception | None = None
            for candidate in candidate_paths:
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                if not candidate.exists():
                    continue
                try:
                    synthetic_name = f"quanta_user_custom_{candidate.stem}_{abs(hash(str(candidate)))}"
                    spec = importlib.util.spec_from_file_location(
                        synthetic_name, str(candidate)
                    )
                    if spec is None or spec.loader is None:
                        continue
                    loaded = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(loaded)
                    module = loaded
                    break
                except Exception as exc:
                    first_exc = exc
            if module is None:
                detail = f": {first_exc}" if first_exc else ""
                raise SystemExit(
                    f"{source}: failed to import '{module_name}' for key '{key}'. "
                    f"Use 'package.module:Symbol' or local file path like 'dir/file.py:Symbol'{detail}"
                )
        try:
            resolved[key] = getattr(module, attr_name)
        except Exception as exc:
            raise SystemExit(
                f"{source}: module '{module_name}' has no attribute '{attr_name}' for key '{key}'"
            ) from exc
    return resolved


def _load_custom_objects(args: argparse.Namespace) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    if args.custom_objects_module:
        try:
            module = importlib.import_module(args.custom_objects_module)
        except Exception as exc:
            raise SystemExit(f"--custom-objects-module import failed: {exc}") from exc
        module_objects: dict[str, Any] | None = None
        if hasattr(module, "CUSTOM_OBJECTS"):
            module_objects = getattr(module, "CUSTOM_OBJECTS")
        elif hasattr(module, "get_custom_objects"):
            module_objects = getattr(module, "get_custom_objects")()
        else:
            raise SystemExit(
                "--custom-objects-module must expose CUSTOM_OBJECTS dict or get_custom_objects()"
            )
        if not isinstance(module_objects, dict):
            raise SystemExit(
                "--custom-objects-module object provider must return a dict"
            )
        for key, value in module_objects.items():
            if not isinstance(key, str):
                raise SystemExit("--custom-objects-module keys must be strings")
            merged[key] = value

    if args.custom_objects_file:
        path = Path(args.custom_objects_file)
        if not path.exists():
            raise SystemExit(f"--custom-objects-file path does not exist: {path}")
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            raise SystemExit(
                f"--custom-objects-file must contain valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise SystemExit("--custom-objects-file JSON must be an object")
        merged.update(_parse_custom_object_specs(payload, "--custom-objects-file"))

    if args.custom_objects:
        try:
            payload = json.loads(args.custom_objects)
        except Exception as exc:
            raise SystemExit(f"--custom-objects must be valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit("--custom-objects JSON must be an object")
        merged.update(_parse_custom_object_specs(payload, "--custom-objects"))

    return merged


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be > 0 when provided")
    if args.max_cached_runs <= 0:
        raise SystemExit("--max-cached-runs must be > 0")
    if args.cpu_only or args.test:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    custom_objects = _load_custom_objects(args)
    final_artifact_profile = (
        "full" if args.store_distributions else args.final_artifact_profile
    )
    output_root = ".quanta"

    def _cleanup_tmp_runs() -> None:
        root = Path(output_root)
        if not root.exists():
            return
        for tmp_dir in root.glob("tmp_run_*"):
            if tmp_dir.is_dir():
                try:
                    import shutil

                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    continue

    atexit.register(_cleanup_tmp_runs)

    # Keep imports lazy so `quanta --help` works even if heavy ML deps are not installed yet.
    from .pipeline import PipelineConfig, run_pipeline

    try:
        if args.test:
            model_path, dataset_path = _bundled_test_assets()
            if args.max_samples is None:
                args.max_samples = 64
        else:
            if not args.model or not args.dataset:
                raise SystemExit(
                    "--model and --dataset are required unless --test is provided"
                )
            if not Path(args.model).exists():
                raise SystemExit(f"Model path does not exist: {args.model}")
            if not Path(args.dataset).exists():
                raise SystemExit(f"Dataset path does not exist: {args.dataset}")
            if not str(args.dataset).endswith(".npy"):
                raise SystemExit("--dataset must point to a .npy file")
            model_path, dataset_path = args.model, args.dataset

        run_dir = run_pipeline(
            PipelineConfig(
                model_path=model_path,
                dataset_path=dataset_path,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                intentionally_bad_layers_count=2 if args.test_bad else 0,
                range_mode=args.range_mode,
                global_weight_mode=args.weight_mode,
                global_activation_mode=args.activation_mode,
                custom_objects=custom_objects,
                final_artifact_profile=final_artifact_profile,
                max_cached_runs=args.max_cached_runs,
                output_root=output_root,
            )
        )
    except Exception as exc:
        print(f"Quanta failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    from .server import create_app

    app = create_app(
        str(run_dir),
        runtime_config={
            "model_path": model_path,
            "dataset_path": dataset_path,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "intentionally_bad_layers_count": 2 if args.test_bad else 0,
            "output_root": output_root,
            "range_mode": args.range_mode,
            "global_weight_mode": args.weight_mode,
            "global_activation_mode": args.activation_mode,
            "final_artifact_profile": final_artifact_profile,
            "max_cached_runs": args.max_cached_runs,
            "layer_weight_modes": {},
            "layer_activation_modes": {},
        },
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Artifacts generated in: {run_dir}")
    print(f"Quanta API running at: {url}")
    if not args.no_ui:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
