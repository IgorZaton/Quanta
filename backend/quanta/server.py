from __future__ import annotations

import json
import os
import signal
import threading
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .strategy_cache import (
    StrategyCache,
    normalize_strategy_key,
    strategy_key_from_state,
)


class ActiveStatePayload(BaseModel):
    range_mode: str = "minmax"
    global_weight_mode: str = "int8"
    global_activation_mode: str = "int8"
    layer_weight_modes: dict[str, str] = {}
    layer_activation_modes: dict[str, str] = {}


def create_app(
    artifact_dir: str,
    runtime_config: dict[str, Any] | None = None,
    loaded_from_run: bool = False,
    strategy_cache: StrategyCache | None = None,
    custom_objects: dict[str, Any] | None = None,
) -> FastAPI:
    base_root = Path(artifact_dir)
    app = FastAPI(title="Quanta API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    cache = strategy_cache
    if cache is None:
        cache = StrategyCache(base_root.parent)

    initial_state: dict[str, Any] = {
        "range_mode": runtime_config.get("range_mode", "minmax")
        if runtime_config
        else "minmax",
        "global_weight_mode": runtime_config.get("global_weight_mode", "int8")
        if runtime_config
        else "int8",
        "global_activation_mode": runtime_config.get("global_activation_mode", "int8")
        if runtime_config
        else "int8",
        "layer_weight_modes": runtime_config.get("layer_weight_modes", {})
        if runtime_config
        else {},
        "layer_activation_modes": runtime_config.get("layer_activation_modes", {})
        if runtime_config
        else {},
    }
    initial_key = strategy_key_from_state(initial_state)
    if cache.get_strategy_root(initial_key) is None:
        cache.upsert_strategy(initial_key, base_root)
    if not cache.get_latest_state():
        cache.set_latest_state(initial_state)

    shutting_down = False

    def _strategy_key(
        strategy: str,
        weight_mode: str,
        activation_mode: str,
        layer_weight_modes_json: str,
        layer_activation_modes_json: str,
    ) -> str:
        try:
            lw = json.loads(layer_weight_modes_json) if layer_weight_modes_json else {}
        except Exception:
            lw = {}
        try:
            la = (
                json.loads(layer_activation_modes_json)
                if layer_activation_modes_json
                else {}
            )
        except Exception:
            la = {}
        return normalize_strategy_key(strategy, weight_mode, activation_mode, lw, la)

    def _ensure_strategy(
        strategy: str,
        weight_mode: str,
        activation_mode: str,
        layer_weight_modes_json: str,
        layer_activation_modes_json: str,
    ) -> Path:
        key = _strategy_key(
            strategy,
            weight_mode,
            activation_mode,
            layer_weight_modes_json,
            layer_activation_modes_json,
        )
        existing = cache.get_strategy_root(key)
        if existing is not None:
            return existing
        if runtime_config is None:
            raise HTTPException(
                status_code=400, detail=f"Configuration {key} is not available"
            )
        if shutting_down:
            raise HTTPException(status_code=503, detail="Server is shutting down")
        if strategy not in {"minmax", "clip99_99", "clip99_999"}:
            raise HTTPException(
                status_code=400, detail=f"Unsupported range mode: {strategy}"
            )
        if weight_mode not in {
            "int2",
            "int4",
            "int8",
            "int12",
            "int16",
            "fp16",
            "fp32",
        }:
            raise HTTPException(
                status_code=400, detail=f"Unsupported weight mode: {weight_mode}"
            )
        if activation_mode not in {
            "int2",
            "int4",
            "int8",
            "int12",
            "int16",
            "fp16",
            "fp32",
        }:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported activation mode: {activation_mode}",
            )
        try:
            layer_weight_modes = (
                json.loads(layer_weight_modes_json) if layer_weight_modes_json else {}
            )
            layer_activation_modes = (
                json.loads(layer_activation_modes_json)
                if layer_activation_modes_json
                else {}
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Invalid layer mode JSON"
            ) from exc
        if not isinstance(layer_weight_modes, dict) or not isinstance(
            layer_activation_modes, dict
        ):
            raise HTTPException(
                status_code=400, detail="layer mode payloads must be JSON objects"
            )
        from .pipeline import PipelineConfig, run_pipeline

        run_dir = run_pipeline(
            PipelineConfig(
                model_path=runtime_config["model_path"],
                dataset_path=runtime_config["dataset_path"],
                batch_size=runtime_config["batch_size"],
                max_samples=runtime_config["max_samples"],
                output_root=runtime_config.get("output_root", ".quanta"),
                intentionally_bad_layers_count=runtime_config.get(
                    "intentionally_bad_layers_count", 0
                ),
                range_mode=strategy,
                global_weight_mode=weight_mode,
                global_activation_mode=activation_mode,
                layer_weight_modes=layer_weight_modes,
                layer_activation_modes=layer_activation_modes,
                final_artifact_profile=runtime_config.get(
                    "final_artifact_profile", "minimal"
                ),
                max_cached_runs=runtime_config.get("max_cached_runs", 3),
                custom_objects=custom_objects,
                skip_finalize=True,
            )
        )
        cache.upsert_strategy(key, run_dir)
        return run_dir

    def _read_json(
        name: str,
        strategy: str,
        weight_mode: str,
        activation_mode: str,
        layer_weight_modes_json: str,
        layer_activation_modes_json: str,
    ):
        root = _ensure_strategy(
            strategy,
            weight_mode,
            activation_mode,
            layer_weight_modes_json,
            layer_activation_modes_json,
        )
        path = root / name
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"{name} not found")
        return json.loads(path.read_text())

    @app.get("/api/graph")
    def get_graph(
        range_mode: str = "minmax",
        weight_mode: str = "int8",
        activation_mode: str = "int8",
        layer_weight_modes: str = "{}",
        layer_activation_modes: str = "{}",
    ):
        return _read_json(
            "graph.json",
            range_mode,
            weight_mode,
            activation_mode,
            layer_weight_modes,
            layer_activation_modes,
        )

    @app.get("/api/metrics")
    def get_metrics(
        metric: str = "mae",
        range_mode: str = "minmax",
        weight_mode: str = "int8",
        activation_mode: str = "int8",
        layer_weight_modes: str = "{}",
        layer_activation_modes: str = "{}",
    ):
        payload = _read_json(
            "metrics.json",
            range_mode,
            weight_mode,
            activation_mode,
            layer_weight_modes,
            layer_activation_modes,
        )
        if metric not in {"mae", "rmse", "kl"}:
            raise HTTPException(status_code=400, detail="Unsupported metric")
        return payload

    @app.get("/api/layers/{layer_name:path}/distributions")
    def get_layer_distributions(
        layer_name: str,
        range_mode: str = "minmax",
        weight_mode: str = "int8",
        activation_mode: str = "int8",
        layer_weight_modes: str = "{}",
        layer_activation_modes: str = "{}",
    ):
        root = _ensure_strategy(
            range_mode,
            weight_mode,
            activation_mode,
            layer_weight_modes,
            layer_activation_modes,
        )
        dist_path = root / "distributions.json"
        if not dist_path.exists():
            raise HTTPException(
                status_code=409,
                detail="Distributions are not available for this finalized run profile",
            )
        payload = _read_json(
            "distributions.json",
            range_mode,
            weight_mode,
            activation_mode,
            layer_weight_modes,
            layer_activation_modes,
        )
        act = payload.get("activations", {}).get(layer_name)
        wgt = payload.get("weights", {}).get(layer_name)
        bias = payload.get("biases", {}).get(layer_name)
        if act is None and wgt is None and bias is None:
            raise HTTPException(
                status_code=404, detail=f"No distributions for {layer_name}"
            )
        return {"activations": act or {}, "weights": wgt or {}, "biases": bias or {}}

    @app.get("/api/layers/{layer_name:path}/qparams")
    def get_layer_qparams(
        layer_name: str,
        range_mode: str = "minmax",
        weight_mode: str = "int8",
        activation_mode: str = "int8",
        layer_weight_modes: str = "{}",
        layer_activation_modes: str = "{}",
    ):
        payload = _read_json(
            "qparams.json",
            range_mode,
            weight_mode,
            activation_mode,
            layer_weight_modes,
            layer_activation_modes,
        )
        act = payload.get("activations", {}).get(layer_name)
        wgt = payload.get("weights", {}).get(layer_name)
        bias = payload.get("biases", {}).get(layer_name)
        if act is None and wgt is None and bias is None:
            raise HTTPException(status_code=404, detail=f"No qparams for {layer_name}")
        return {"activations": act, "weights": wgt, "biases": bias}

    @app.get("/api/estimates")
    def get_estimates(
        range_mode: str = "minmax",
        weight_mode: str = "int8",
        activation_mode: str = "int8",
        layer_weight_modes: str = "{}",
        layer_activation_modes: str = "{}",
    ):
        return _read_json(
            "estimates.json",
            range_mode,
            weight_mode,
            activation_mode,
            layer_weight_modes,
            layer_activation_modes,
        )

    @app.get("/api/runtime")
    def get_runtime():
        latest = cache.get_latest_state()
        return {
            "run_id": cache.run_id,
            "loaded_from_run": loaded_from_run,
            "runtime_recompute_enabled": runtime_config is not None,
            "finalized": cache.finalized,
            "available_strategies": cache.strategy_keys,
            "latest_state": latest,
        }

    @app.post("/api/runtime/active-state")
    def set_active_state(payload: ActiveStatePayload):
        state = payload.model_dump()
        cache.set_latest_state(state)
        return {"status": "ok"}

    @app.post("/api/run/finalize")
    def finalize_run():
        nonlocal shutting_down
        if cache.finalized:
            return {
                "status": "already_finalized",
                "run_id": cache.run_id,
            }
        try:
            final_dir = cache.materialize_finalized_snapshot()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "status": "finalized",
            "run_id": cache.run_id,
            "finalized_path": str(final_dir),
        }

    @app.post("/api/run/close")
    def close_run():
        nonlocal shutting_down
        if not cache.finalized:
            raise HTTPException(
                status_code=409,
                detail="Cannot close: run has not been finalized yet",
            )
        shutting_down = True

        def _deferred_shutdown():
            import time

            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_deferred_shutdown, daemon=True).start()
        return {"status": "closing"}

    bundled_web = Path(str(resources.files("quanta.web")))
    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="ui")
    elif bundled_web.exists() and (bundled_web / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(bundled_web), html=True), name="ui")
    else:

        @app.get("/", response_class=HTMLResponse)
        def ui_missing():
            return (
                "<h2>Quanta API is running</h2>"
                "<p>Frontend build not found. Build UI with:</p>"
                "<pre>cd quanta/frontend && npm install && npm run build</pre>"
            )

    return app
