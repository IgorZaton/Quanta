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
from typing import Any, Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quanta quantization visualizer")
    parser.add_argument(
        "--model",
        required=False,
        help="Path to model file (.keras for TensorFlow or TorchScript .pt)",
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
    parser.add_argument(
        "--load-run",
        required=False,
        help="Load and serve an existing run directory without recomputation",
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


LoadRunCheck = Callable[[Path, dict[str, Any]], None]


def _check_run_dir_exists(run_dir: Path, context: dict[str, Any]) -> None:
    del context
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(
            f"--load-run path does not exist or is not a directory: {run_dir}"
        )


def _check_required_artifact_files(run_dir: Path, context: dict[str, Any]) -> None:
    required_artifacts = context["required_artifacts"]
    missing = [name for name in required_artifacts if not (run_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"--load-run directory is missing required artifacts: {', '.join(missing)}"
        )


def _check_run_meta_json(run_dir: Path, context: dict[str, Any]) -> None:
    run_meta_path = run_dir / "run_meta.json"
    if not run_meta_path.exists():
        raise SystemExit("--load-run directory is missing required run_meta.json")
    try:
        run_meta = json.loads(run_meta_path.read_text())
    except Exception as exc:
        raise SystemExit(f"run_meta.json is not valid JSON: {exc}") from exc
    if not isinstance(run_meta, dict):
        raise SystemExit("run_meta.json must be a JSON object")
    context["run_meta"] = run_meta


def _check_required_run_meta_fields(run_dir: Path, context: dict[str, Any]) -> None:
    del run_dir
    run_meta = context["run_meta"]
    required_meta_fields = context["required_meta_fields"]
    missing_meta = [name for name in required_meta_fields if name not in run_meta]
    if missing_meta:
        raise SystemExit(
            f"run_meta.json is missing required fields: {', '.join(missing_meta)}"
        )
    if not isinstance(run_meta.get("artifacts"), list):
        raise SystemExit("run_meta.json field 'artifacts' must be a list")


def _check_run_meta_artifacts_compatible(
    run_dir: Path, context: dict[str, Any]
) -> None:
    del run_dir
    required_artifacts = context["required_artifacts"]
    artifacts = set(context["run_meta"]["artifacts"])
    missing_declared = [name for name in required_artifacts if name not in artifacts]
    if missing_declared:
        raise SystemExit(
            "run_meta.json artifacts list is incompatible; missing required entries: "
            + ", ".join(missing_declared)
        )


def _check_optional_artifacts(run_dir: Path, context: dict[str, Any]) -> None:
    warnings = context["warnings"]
    for name in context.get("optional_artifacts", ()):
        if not (run_dir / name).exists():
            warnings.append(
                f"Optional artifact missing: {name} (some features may be unavailable)"
            )
    if "latest_state" not in context.get("run_meta", {}):
        warnings.append(
            "run_meta.json has no latest_state; UI will start with default settings"
        )


def _validate_load_run_dir(load_run: str) -> tuple[Path, dict[str, Any], list[str]]:
    run_dir = Path(load_run).expanduser().resolve()
    context: dict[str, Any] = {
        "required_artifacts": (
            "graph.json",
            "metrics.json",
            "qparams.json",
            "estimates.json",
        ),
        "optional_artifacts": ("distributions.json",),
        "required_meta_fields": ("run_id", "profile", "artifacts"),
        "warnings": [],
    }
    checks: tuple[LoadRunCheck, ...] = (
        _check_run_dir_exists,
        _check_required_artifact_files,
        _check_run_meta_json,
        _check_required_run_meta_fields,
        _check_run_meta_artifacts_compatible,
        _check_optional_artifacts,
    )
    for check in checks:
        check(run_dir, context)
    return run_dir, context["run_meta"], context["warnings"]


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
    has_compute_args = bool(args.model or args.dataset or args.test)
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

    from .strategy_cache import StrategyCache

    def _resolve_compute_args() -> tuple[str, str]:
        """Return (model_path, dataset_path) from --test or --model/--dataset."""
        if args.test:
            mp, dp = _bundled_test_assets()
            if args.max_samples is None:
                args.max_samples = 64
            return mp, dp
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
        return args.model, args.dataset

    def _build_runtime_config(
        model_path: str, dataset_path: str, **overrides: Any
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {
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
        }
        cfg.update(overrides)
        return cfg

    def _build_source_config(model_path: str, dataset_path: str) -> dict[str, Any]:
        """Serializable config stored in run_meta.json for session restore."""
        cfg: dict[str, Any] = {
            "model_path": str(Path(model_path).resolve()),
            "dataset_path": str(Path(dataset_path).resolve()),
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "output_root": output_root,
            "max_cached_runs": args.max_cached_runs,
            "final_artifact_profile": final_artifact_profile,
        }
        if args.custom_objects_file:
            cfg["custom_objects_file"] = str(Path(args.custom_objects_file).resolve())
        if args.custom_objects_module:
            cfg["custom_objects_module"] = args.custom_objects_module
        if args.custom_objects:
            cfg["custom_objects_inline"] = args.custom_objects
        return cfg

    def _restore_source_config(
        sc: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """From a stored source_config, rebuild runtime_config + custom_objects.

        Raises SystemExit if referenced files no longer exist.
        """
        mp = sc.get("model_path", "")
        dp = sc.get("dataset_path", "")
        if not mp or not dp:
            raise SystemExit(
                "Stored source_config is missing model_path / dataset_path — "
                "provide --model and --dataset explicitly"
            )
        if not Path(mp).exists():
            raise SystemExit(f"Stored model_path no longer exists: {mp}")
        if not Path(dp).exists():
            raise SystemExit(f"Stored dataset_path no longer exists: {dp}")

        rc: dict[str, Any] = {
            "model_path": mp,
            "dataset_path": dp,
            "batch_size": sc.get("batch_size", args.batch_size),
            "max_samples": sc.get("max_samples", args.max_samples),
            "intentionally_bad_layers_count": 0,
            "output_root": sc.get("output_root", output_root),
            "range_mode": args.range_mode,
            "global_weight_mode": args.weight_mode,
            "global_activation_mode": args.activation_mode,
            "final_artifact_profile": sc.get(
                "final_artifact_profile", final_artifact_profile
            ),
            "max_cached_runs": sc.get("max_cached_runs", args.max_cached_runs),
            "layer_weight_modes": {},
            "layer_activation_modes": {},
        }

        co: dict[str, Any] = {}
        co_file = sc.get("custom_objects_file")
        if co_file and Path(co_file).exists():
            payload = json.loads(Path(co_file).read_text())
            co.update(
                _parse_custom_object_specs(payload, "source_config.custom_objects_file")
            )
        co_module = sc.get("custom_objects_module")
        if co_module:
            try:
                mod = importlib.import_module(co_module)
                if hasattr(mod, "CUSTOM_OBJECTS"):
                    co.update(getattr(mod, "CUSTOM_OBJECTS"))
                elif hasattr(mod, "get_custom_objects"):
                    co.update(getattr(mod, "get_custom_objects")())
            except Exception:
                print(
                    f"Warning: could not reload custom_objects_module: {co_module}",
                    file=sys.stderr,
                )
        co_inline = sc.get("custom_objects_inline")
        if co_inline:
            try:
                payload = json.loads(co_inline)
                co.update(
                    _parse_custom_object_specs(
                        payload, "source_config.custom_objects_inline"
                    )
                )
            except Exception:
                print(
                    "Warning: could not reload inline custom_objects", file=sys.stderr
                )

        return rc, co

    try:
        if args.load_run:
            run_dir, run_meta, load_warnings = _validate_load_run_dir(args.load_run)
            for w in load_warnings:
                print(f"Warning: {w}", file=sys.stderr)
            loaded_from_run = True
            strategy_cache = StrategyCache.from_finalized_run(run_dir)

            source_config = run_meta.get("source_config", {})
            if has_compute_args:
                model_path, dataset_path = _resolve_compute_args()
                runtime_config = _build_runtime_config(model_path, dataset_path)
            elif source_config:
                runtime_config, custom_objects = _restore_source_config(source_config)
            else:
                runtime_config = None

            if runtime_config:
                print(
                    f"Loaded run from {run_dir} (interactive: recompute enabled)",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Loaded run from {run_dir} (read-only: no compute config available)",
                    file=sys.stderr,
                )
        else:
            if not has_compute_args:
                raise SystemExit(
                    "--model and --dataset (or --test) are required unless --load-run is provided"
                )
            from .pipeline import PipelineConfig, run_pipeline

            model_path, dataset_path = _resolve_compute_args()
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
            runtime_config = _build_runtime_config(model_path, dataset_path)
            loaded_from_run = False
            strategy_cache = StrategyCache(Path(output_root))
            strategy_cache.set_source_config(
                _build_source_config(model_path, dataset_path)
            )
    except Exception as exc:
        print(f"Quanta failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    from .server import create_app

    app = create_app(
        str(run_dir),
        runtime_config=runtime_config,
        loaded_from_run=loaded_from_run,
        strategy_cache=strategy_cache,
        custom_objects=custom_objects or None,
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
