import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, { Background, Controls, Edge, Node } from "react-flow-renderer";
import { LayerDetail } from "./components/LayerDetail";
import { FinalizeResponse, GraphPayload, MetricKey, RuntimePayload } from "./types";

function layerTypeColor(layerType: string): string {
  const palette: Record<string, string> = {
    InputLayer: "#dbeafe",
    Conv2D: "#bfdbfe",
    Dense: "#c7d2fe",
    BatchNormalization: "#fde68a",
    ReLU: "#bbf7d0",
    Activation: "#bbf7d0",
    MaxPooling2D: "#fecaca",
    AveragePooling2D: "#fed7aa",
    Flatten: "#e5e7eb",
    Add: "#f5d0fe",
    Dropout: "#ddd6fe",
  };
  return palette[layerType] ?? "#e2e8f0";
}

function errorSeverity(value: number, maxError: number, goodThreshold: number): number {
  if (maxError <= goodThreshold || value <= goodThreshold) return 0;
  return Math.min(1, Math.max(0, (value - goodThreshold) / (maxError - goodThreshold)));
}

function absoluteThresholds(metric: MetricKey): { warn: number; bad: number; critical: number } {
  if (metric === "rmse") {
    return { warn: 2e-3, bad: 6e-3, critical: 2e-2 };
  }
  if (metric === "kl") {
    return { warn: 1e-2, bad: 5e-2, critical: 1e-1 };
  }
  return { warn: 1e-3, bad: 3e-3, critical: 1e-2 };
}

function absoluteSeverity(value: number, t: { warn: number; bad: number; critical: number }): number {
  if (value < t.warn) return 0;
  if (value < t.bad) return 0.45;
  if (value < t.critical) return 0.75;
  return 1;
}

function combinedSeverity(
  value: number,
  maxError: number,
  goodThreshold: number,
  metric: MetricKey
): number {
  const absT = absoluteThresholds(metric);
  if (value < absT.warn) return 0;
  const rel = errorSeverity(value, maxError, goodThreshold);
  const abs = absoluteSeverity(value, absT);
  return Math.max(rel, abs);
}

function isIntegerMode(mode: string): boolean {
  return mode.startsWith("int");
}

type PrecisionMode = "int2" | "int4" | "int8" | "int12" | "int16" | "fp16" | "fp32";
type RangeMode = "minmax" | "clip99_99" | "clip99_999";

export default function App() {
  const [metric, setMetric] = useState<MetricKey>("mae");
  const [rangeMode, setRangeMode] = useState<RangeMode>("minmax");
  const [globalWeightPrecision, setGlobalWeightPrecision] = useState<PrecisionMode>("int8");
  const [globalActivationPrecision, setGlobalActivationPrecision] = useState<PrecisionMode>("int8");
  const [layerWeightModes, setLayerWeightModes] = useState<Record<string, string>>({});
  const [layerActivationModes, setLayerActivationModes] = useState<Record<string, string>>({});
  const [graph, setGraph] = useState<GraphPayload>({ nodes: [], edges: [] });
  const [selectedLayer, setSelectedLayer] = useState<string>("");
  const [distributions, setDistributions] = useState<any>({});
  const [qparams, setQparams] = useState<any>({});
  const [estimates, setEstimates] = useState<any>({});
  const [loadingGraph, setLoadingGraph] = useState<boolean>(true);
  const [loadingLayer, setLoadingLayer] = useState<boolean>(false);
  const [runtimeInfo, setRuntimeInfo] = useState<RuntimePayload | null>(null);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
  const [finalizeResult, setFinalizeResult] = useState<FinalizeResponse | null>(null);
  const [closing, setClosing] = useState(false);
  const [distributionsUnavailable, setDistributionsUnavailable] = useState(false);

  const canRecompute = runtimeInfo?.runtime_recompute_enabled ?? true;

  // -- Hydrate controls from backend latest_state on first load ---------------
  useEffect(() => {
    fetch("/api/runtime")
      .then(async (r) => {
        if (!r.ok) throw new Error("Failed to fetch runtime");
        return r.json();
      })
      .then((payload: RuntimePayload) => {
        setRuntimeInfo(payload);
        if (payload.latest_state) {
          setRangeMode(payload.latest_state.range_mode as RangeMode);
          setGlobalWeightPrecision(payload.latest_state.global_weight_mode as PrecisionMode);
          setGlobalActivationPrecision(payload.latest_state.global_activation_mode as PrecisionMode);
          setLayerWeightModes(payload.latest_state.layer_weight_modes ?? {});
          setLayerActivationModes(payload.latest_state.layer_activation_modes ?? {});
        }
        setHydrated(true);
      })
      .catch((err) => {
        setRuntimeError(err instanceof Error ? err.message : "Runtime fetch failed");
        setHydrated(true);
      });
  }, []);

  const layerWeightModesJson = useMemo(() => JSON.stringify(layerWeightModes), [layerWeightModes]);
  const layerActivationModesJson = useMemo(() => JSON.stringify(layerActivationModes), [layerActivationModes]);
  const strategyQuery = useMemo(
    () =>
      `range_mode=${rangeMode}&weight_mode=${globalWeightPrecision}&activation_mode=${globalActivationPrecision}&layer_weight_modes=${encodeURIComponent(layerWeightModesJson)}&layer_activation_modes=${encodeURIComponent(layerActivationModesJson)}`,
    [rangeMode, globalWeightPrecision, globalActivationPrecision, layerWeightModesJson, layerActivationModesJson]
  );

  // -- Debounced active-state persistence -------------------------------------
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const persistActiveState = useCallback(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetch("/api/runtime/active-state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          range_mode: rangeMode,
          global_weight_mode: globalWeightPrecision,
          global_activation_mode: globalActivationPrecision,
          layer_weight_modes: layerWeightModes,
          layer_activation_modes: layerActivationModes,
        }),
      }).catch(() => {});
    }, 300);
  }, [rangeMode, globalWeightPrecision, globalActivationPrecision, layerWeightModes, layerActivationModes]);

  useEffect(() => {
    if (!hydrated) return;
    persistActiveState();
  }, [hydrated, persistActiveState]);

  // -- Fetch graph + estimates when strategy changes --------------------------
  useEffect(() => {
    if (!hydrated) return;
    setLoadingGraph(true);
    fetch(`/api/graph?${strategyQuery}`)
      .then((r) => {
        if (!r.ok) return null;
        return r.json();
      })
      .then((data) => { if (data) setGraph(data); })
      .finally(() => setLoadingGraph(false));
    fetch(`/api/estimates?${strategyQuery}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => { if (data) setEstimates(data); });
  }, [hydrated, strategyQuery]);

  // -- Refresh runtime info when strategy changes -----------------------------
  useEffect(() => {
    if (!hydrated) return;
    fetch("/api/runtime")
      .then(async (r) => {
        if (!r.ok) throw new Error("Runtime fetch failed");
        return r.json();
      })
      .then(setRuntimeInfo)
      .catch((e) => setRuntimeError(e instanceof Error ? e.message : "Unknown error"));
  }, [hydrated, strategyQuery]);

  // -- Fetch layer detail when selected layer / strategy changes --------------
  useEffect(() => {
    if (!hydrated || !selectedLayer) return;
    setLoadingLayer(true);
    const selectedLayerPath = encodeURIComponent(selectedLayer);
    const selectedNode = graph.nodes.find((n) => n.id === selectedLayer);
    const selectedWeightMode = layerWeightModes[selectedLayer] ?? globalWeightPrecision;
    const selectedActivationMode = layerActivationModes[selectedLayer] ?? globalActivationPrecision;
    const shouldFetchQparams =
      isIntegerMode(selectedActivationMode) || (Boolean(selectedNode?.has_weights) && isIntegerMode(selectedWeightMode));
    fetch(`/api/layers/${selectedLayerPath}/distributions?${strategyQuery}`)
      .then((r) => {
        if (!r.ok) {
          setDistributionsUnavailable(true);
          return { activations: {}, weights: {}, biases: {} };
        }
        setDistributionsUnavailable(false);
        return r.json();
      })
      .then(setDistributions);
    if (shouldFetchQparams) {
      fetch(`/api/layers/${selectedLayerPath}/qparams?${strategyQuery}`)
        .then((r) => (r.ok ? r.json() : {}))
        .then(setQparams)
        .finally(() => setLoadingLayer(false));
    } else {
      setQparams({});
      setLoadingLayer(false);
    }
  }, [hydrated, selectedLayer, strategyQuery, graph.nodes, layerWeightModes, layerActivationModes, globalWeightPrecision, globalActivationPrecision]);

  // -- Finalize and close handlers --------------------------------------------
  const handleFinalize = useCallback(async () => {
    if (finalizing) return;
    setFinalizing(true);
    setFinalizeResult(null);
    try {
      const resp = await fetch("/api/run/finalize", { method: "POST" });
      const data: FinalizeResponse = await resp.json();
      if (!resp.ok) throw new Error((data as any).detail ?? "Finalize failed");
      setFinalizeResult(data);
      const rtResp = await fetch("/api/runtime");
      if (rtResp.ok) setRuntimeInfo(await rtResp.json());
    } catch (e) {
      setRuntimeError(e instanceof Error ? e.message : "Finalize error");
    } finally {
      setFinalizing(false);
    }
  }, [finalizing]);

  const [serverStopped, setServerStopped] = useState(false);

  const handleClose = useCallback(async () => {
    if (closing) return;
    setClosing(true);
    try {
      await fetch("/api/run/close", { method: "POST" });
    } catch {
      // expected: server shuts down
    }
    setServerStopped(true);
    window.close();
  }, [closing]);

  const nodes = useMemo<Node[]>(() => {
    const values = graph.nodes.map((n) => n.metrics[metric] ?? 0);
    const max = Math.max(...values, 0);
    const goodThreshold = max > 0 ? max * 0.2 : 0;
    const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));
    const indegree: Record<string, number> = {};
    const outgoing: Record<string, string[]> = {};
    const incoming: Record<string, string[]> = {};
    graph.nodes.forEach((n) => {
      indegree[n.id] = 0;
      outgoing[n.id] = [];
      incoming[n.id] = [];
    });
    graph.edges.forEach((e) => {
      if (e.source in outgoing && e.target in indegree) {
        outgoing[e.source].push(e.target);
        incoming[e.target].push(e.source);
        indegree[e.target] += 1;
      }
    });
    const queue = Object.keys(indegree).filter((id) => indegree[id] === 0);
    const orderedIds: string[] = [];
    while (queue.length > 0) {
      const cur = queue.shift()!;
      orderedIds.push(cur);
      for (const nxt of outgoing[cur]) {
        indegree[nxt] -= 1;
        if (indegree[nxt] === 0) queue.push(nxt);
      }
    }
    graph.nodes.forEach((n) => {
      if (!orderedIds.includes(n.id)) orderedIds.push(n.id);
    });

    const depth: Record<string, number> = {};
    orderedIds.forEach((id) => {
      const preds = incoming[id] ?? [];
      if (preds.length === 0) {
        depth[id] = 0;
      } else {
        depth[id] = Math.max(...preds.map((p) => depth[p] ?? 0)) + 1;
      }
    });
    const byDepth = new Map<number, string[]>();
    orderedIds.forEach((id) => {
      const d = depth[id] ?? 0;
      const bucket = byDepth.get(d) ?? [];
      bucket.push(id);
      byDepth.set(d, bucket);
    });
    const xPos: Record<string, number> = {};
    const yPos: Record<string, number> = {};
    const laneGap = 260;
    const depthGap = 150;
    Array.from(byDepth.entries())
      .sort((a, b) => a[0] - b[0])
      .forEach(([d, ids]) => {
        ids.sort((a, b) => {
          const ac = incoming[a] ?? [];
          const bc = incoming[b] ?? [];
          const ax = ac.length ? ac.reduce((s, p) => s + (xPos[p] ?? 0), 0) / ac.length : 0;
          const bx = bc.length ? bc.reduce((s, p) => s + (xPos[p] ?? 0), 0) / bc.length : 0;
          return ax - bx;
        });
        const startX = -((ids.length - 1) * laneGap) / 2;
        ids.forEach((id, idx) => {
          xPos[id] = startX + idx * laneGap;
          yPos[id] = d * depthGap;
        });
      });

    return orderedIds.map((id) => {
      const node = nodeMap.get(id)!;
      return {
      id: node.id,
      data: {
        label: `${node.name}\nW:${layerWeightModes[node.id] ?? globalWeightPrecision} A:${layerActivationModes[node.id] ?? globalActivationPrecision}\n${metric}: ${node.metrics[metric]?.toFixed(5)}`,
      },
      position: { x: xPos[id] ?? 0, y: yPos[id] ?? 0 },
      style: {
        background: layerTypeColor(node.type),
        border: `${combinedSeverity(node.metrics[metric] ?? 0, max, goodThreshold, metric) > 0 ? 2 : 1}px solid ${
          combinedSeverity(node.metrics[metric] ?? 0, max, goodThreshold, metric) > 0 ? "#dc2626" : "#334155"
        }`,
        boxShadow:
          combinedSeverity(node.metrics[metric] ?? 0, max, goodThreshold, metric) > 0
            ? `0 0 ${10 + 28 * combinedSeverity(node.metrics[metric] ?? 0, max, goodThreshold, metric)}px rgba(220,38,38,${
                0.35 + 0.45 * combinedSeverity(node.metrics[metric] ?? 0, max, goodThreshold, metric)
              })`
            : "none",
        borderRadius: 8,
        width: 190,
        whiteSpace: "pre-line",
      },
    };
    });
  }, [graph, metric, layerWeightModes, layerActivationModes, globalWeightPrecision, globalActivationPrecision]);

  const colorLegend = useMemo(() => {
    const values = graph.nodes.map((n) => n.metrics[metric] ?? 0);
    const max = Math.max(...values, 0);
    const good = max > 0 ? max * 0.2 : 0;
    const abs = absoluteThresholds(metric);
    return {
      max,
      good,
      abs,
    };
  }, [graph, metric]);

  const edges = useMemo<Edge[]>(
    () =>
      graph.edges.map((e, idx) => ({
        id: `e_${idx}`,
        source: e.source,
        target: e.target,
        animated: false,
      })),
    [graph]
  );

  const selectedMetrics = graph.nodes.find((n) => n.id === selectedLayer)?.metrics;

  return (
    <div className="app">
      {(loadingGraph || loadingLayer) && (
        <div className="loading-overlay">
          <img src="/quanta-manta-icon.svg" alt="Quanta icon" width={72} height={72} />
          <div className="loading-title">Quanta</div>
          <div className="spinner" />
        </div>
      )}
      {(closing || serverStopped) && (
        <div className="loading-overlay">
          <div className="loading-title">{serverStopped ? "Server stopped" : "Shutting down..."}</div>
          {serverStopped && <div className="stopped-hint">You can close this tab.</div>}
        </div>
      )}
      <header className="toolbar">
        <h2 className="brand">
          <img src="/quanta-manta-icon.svg" alt="Quanta icon" width={26} height={26} />
          Quanta
        </h2>
        <label>
          Error metric:
          <select value={metric} onChange={(e) => setMetric(e.target.value as MetricKey)}>
            <option value="mae">MAE</option>
            <option value="rmse">RMSE</option>
            <option value="kl">KL</option>
          </select>
        </label>
        <label>
          Range mode:
          <select value={rangeMode} disabled={!canRecompute} onChange={(e) => setRangeMode(e.target.value as RangeMode)}>
            <option value="minmax">Min/Max</option>
            <option value="clip99_99">99.99% clip</option>
            <option value="clip99_999">99.999% clip</option>
          </select>
        </label>
        <label>
          Global W precision:
          <select value={globalWeightPrecision} disabled={!canRecompute} onChange={(e) => {
            const next = e.target.value as PrecisionMode;
            setGlobalWeightPrecision(next);
            setLayerWeightModes((prev) => {
              const cleaned: Record<string, string> = {};
              for (const [k, v] of Object.entries(prev)) if (v !== next) cleaned[k] = v;
              return cleaned;
            });
          }}>
            <option value="int2">int2</option>
            <option value="int4">int4</option>
            <option value="int8">int8</option>
            <option value="int12">int12</option>
            <option value="int16">int16</option>
            <option value="fp16">fp16</option>
            <option value="fp32">fp32</option>
          </select>
        </label>
        <label>
          Global A precision:
          <select value={globalActivationPrecision} disabled={!canRecompute} onChange={(e) => {
            const next = e.target.value as PrecisionMode;
            setGlobalActivationPrecision(next);
            setLayerActivationModes((prev) => {
              const cleaned: Record<string, string> = {};
              for (const [k, v] of Object.entries(prev)) if (v !== next) cleaned[k] = v;
              return cleaned;
            });
          }}>
            <option value="int2">int2</option>
            <option value="int4">int4</option>
            <option value="int8">int8</option>
            <option value="int12">int12</option>
            <option value="int16">int16</option>
            <option value="fp16">fp16</option>
            <option value="fp32">fp32</option>
          </select>
        </label>
        <button
          type="button"
          disabled={!canRecompute}
          onClick={() => {
            setLayerWeightModes({});
            setLayerActivationModes({});
          }}
        >
          Reset to Global
        </button>
        {canRecompute && (
          <div className="finalize-group">
            {!runtimeInfo?.finalized ? (
              <button
                type="button"
                className="finalize-btn"
                disabled={finalizing}
                onClick={handleFinalize}
              >
                {finalizing ? "Finalizing..." : "Finalize Run"}
              </button>
            ) : (
              <button
                type="button"
                className="close-btn-toolbar"
                disabled={closing}
                onClick={handleClose}
              >
                {closing ? "Closing..." : "Close"}
              </button>
            )}
            {finalizeResult && (
              <span className="finalize-result">
                {finalizeResult.status === "finalized"
                  ? `Saved: ${finalizeResult.finalized_path}`
                  : finalizeResult.status}
              </span>
            )}
          </div>
        )}
        <div className="legend">
          <span className="runtime-indicator">
            {runtimeInfo?.loaded_from_run ? "Loaded" : "Live"} | {runtimeInfo?.run_id ?? "unknown"}
            {runtimeInfo?.finalized ? " | Finalized" : ""}
          </span>
          {runtimeError && <span className="runtime-error">{runtimeError}</span>}
          <span>Good {"<="} {colorLegend.good.toExponential(2)}</span>
          <span>Max: {colorLegend.max.toExponential(2)}</span>
          <span>
            Abs thresholds ({metric.toUpperCase()}): {colorLegend.abs.warn.toExponential(1)} / {colorLegend.abs.bad.toExponential(1)} / {colorLegend.abs.critical.toExponential(1)}
          </span>
          <span>
            Global size: {estimates?.global?.size_kb?.toFixed?.(2) ?? "-"}KB / {estimates?.global?.size_kb_fp32?.toFixed?.(2) ?? "-"}KB
          </span>
          <span>
            Global latency: {estimates?.global?.latency_ms?.toFixed?.(3) ?? "-"}ms / {estimates?.global?.latency_ms_fp32?.toFixed?.(3) ?? "-"}ms
          </span>
          <span>Glow: ON above threshold</span>
        </div>
      </header>
      <div className="body">
        <div className="graph">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodeClick={(_, node) => setSelectedLayer(node.id)}
            fitView
          >
            <Background />
            <Controls />
          </ReactFlow>
        </div>
        <div className="panel">
          {selectedLayer ? (
            <>
              {distributionsUnavailable && (
                <div className="distributions-warning">
                  Distributions unavailable for this run. Plots will be empty.
                  Use <code>--final-artifact-profile full</code> or <b>Finalize Run</b> to include them.
                </div>
              )}
              <LayerDetail
                layerName={selectedLayer}
                layerType={graph.nodes.find((n) => n.id === selectedLayer)?.type}
                metricValues={selectedMetrics}
                activations={distributions.activations ?? {}}
                weights={distributions.weights ?? {}}
                biases={distributions.biases ?? {}}
                qparams={qparams}
                metric={metric}
                hasWeights={Boolean(graph.nodes.find((n) => n.id === selectedLayer)?.has_weights)}
                layerWeightMode={layerWeightModes[selectedLayer] ?? globalWeightPrecision}
                layerActivationMode={layerActivationModes[selectedLayer] ?? globalActivationPrecision}
                layerEstimate={estimates?.layers?.[selectedLayer] ?? graph.nodes.find((n) => n.id === selectedLayer)?.estimates}
                readOnly={!canRecompute}
                onLayerWeightModeChange={(mode) =>
                  setLayerWeightModes((prev) => {
                    if (mode === globalWeightPrecision) {
                      const { [selectedLayer]: _, ...rest } = prev;
                      return rest;
                    }
                    return { ...prev, [selectedLayer]: mode };
                  })
                }
                onLayerActivationModeChange={(mode) =>
                  setLayerActivationModes((prev) => {
                    if (mode === globalActivationPrecision) {
                      const { [selectedLayer]: _, ...rest } = prev;
                      return rest;
                    }
                    return { ...prev, [selectedLayer]: mode };
                  })
                }
              />
            </>
          ) : (
            <p>Select a node to view plots and qparams.</p>
          )}
        </div>
      </div>
    </div>
  );
}
