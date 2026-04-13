from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


def create_app(artifact_dir: str, runtime_config: dict[str, Any] | None = None) -> FastAPI:
    base_root = Path(artifact_dir)
    strategy_roots: dict[str, Path] = {}
    initial_key = (
        f"{runtime_config.get('range_mode','minmax')}|{runtime_config.get('global_weight_mode','int8')}|{runtime_config.get('global_activation_mode','int8')}|{{}}|{{}}"
        if runtime_config
        else "minmax|int8|int8|{}|{}"
    )
    strategy_roots[initial_key] = base_root
    app = FastAPI(title="Quanta API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _strategy_key(strategy: str, weight_mode: str, activation_mode: str, layer_weight_modes_json: str, layer_activation_modes_json: str) -> str:
        return f"{strategy}|{weight_mode}|{activation_mode}|{layer_weight_modes_json}|{layer_activation_modes_json}"

    def _ensure_strategy(strategy: str, weight_mode: str, activation_mode: str, layer_weight_modes_json: str, layer_activation_modes_json: str) -> Path:
        key = _strategy_key(strategy, weight_mode, activation_mode, layer_weight_modes_json, layer_activation_modes_json)
        if key in strategy_roots:
            existing = strategy_roots[key]
            if existing.exists():
                return existing
            # Cache entry may point to pruned run directory.
            strategy_roots.pop(key, None)
        if runtime_config is None:
            raise HTTPException(status_code=400, detail=f"Configuration {key} is not available")
        if strategy not in {"minmax", "clip99_99", "clip99_999"}:
            raise HTTPException(status_code=400, detail=f"Unsupported range mode: {strategy}")
        if weight_mode not in {"int2", "int4", "int8", "int12", "int16", "fp16", "fp32"}:
            raise HTTPException(status_code=400, detail=f"Unsupported weight mode: {weight_mode}")
        if activation_mode not in {"int2", "int4", "int8", "int12", "int16", "fp16", "fp32"}:
            raise HTTPException(status_code=400, detail=f"Unsupported activation mode: {activation_mode}")
        try:
            layer_weight_modes = json.loads(layer_weight_modes_json) if layer_weight_modes_json else {}
            layer_activation_modes = json.loads(layer_activation_modes_json) if layer_activation_modes_json else {}
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid layer mode JSON") from exc
        if not isinstance(layer_weight_modes, dict) or not isinstance(layer_activation_modes, dict):
            raise HTTPException(status_code=400, detail="layer mode payloads must be JSON objects")
        from .pipeline import PipelineConfig, run_pipeline

        run_dir = run_pipeline(
            PipelineConfig(
                model_path=runtime_config["model_path"],
                dataset_path=runtime_config["dataset_path"],
                batch_size=runtime_config["batch_size"],
                max_samples=runtime_config["max_samples"],
                output_root=runtime_config.get("output_root", ".quanta"),
                intentionally_bad_layers_count=runtime_config.get("intentionally_bad_layers_count", 0),
                range_mode=strategy,
                global_weight_mode=weight_mode,
                global_activation_mode=activation_mode,
                layer_weight_modes=layer_weight_modes,
                layer_activation_modes=layer_activation_modes,
            )
        )
        strategy_roots[key] = run_dir
        return run_dir

    def _read_json(
        name: str,
        strategy: str,
        weight_mode: str,
        activation_mode: str,
        layer_weight_modes_json: str,
        layer_activation_modes_json: str,
    ):
        root = _ensure_strategy(strategy, weight_mode, activation_mode, layer_weight_modes_json, layer_activation_modes_json)
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
        return _read_json("graph.json", range_mode, weight_mode, activation_mode, layer_weight_modes, layer_activation_modes)

    @app.get("/api/metrics")
    def get_metrics(
        metric: str = "mae",
        range_mode: str = "minmax",
        weight_mode: str = "int8",
        activation_mode: str = "int8",
        layer_weight_modes: str = "{}",
        layer_activation_modes: str = "{}",
    ):
        payload = _read_json("metrics.json", range_mode, weight_mode, activation_mode, layer_weight_modes, layer_activation_modes)
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
        payload = _read_json("distributions.json", range_mode, weight_mode, activation_mode, layer_weight_modes, layer_activation_modes)
        act = payload.get("activations", {}).get(layer_name)
        wgt = payload.get("weights", {}).get(layer_name)
        bias = payload.get("biases", {}).get(layer_name)
        if act is None and wgt is None and bias is None:
            raise HTTPException(status_code=404, detail=f"No distributions for {layer_name}")
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
        payload = _read_json("qparams.json", range_mode, weight_mode, activation_mode, layer_weight_modes, layer_activation_modes)
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
        return _read_json("estimates.json", range_mode, weight_mode, activation_mode, layer_weight_modes, layer_activation_modes)

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
