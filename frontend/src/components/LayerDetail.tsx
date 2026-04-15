import { useMemo, useState } from "react";
import Plot from "react-plotly.js";

type DistMap = Record<string, number[]>;
type DistBundle = {
  pre?: DistMap;
  post?: DistMap;
  error?: DistMap;
  error_mae?: DistMap;
  error_rmse?: DistMap;
};

interface LayerDetailProps {
  layerName: string;
  layerType?: string;
  metricValues: Record<string, number> | undefined;
  activations: DistMap | DistBundle;
  weights: DistMap | DistBundle;
  biases: DistMap | DistBundle;
  qparams: any;
  metric: "mae" | "rmse" | "kl";
  layerWeightMode: string;
  layerActivationMode: string;
  hasWeights: boolean;
  readOnly?: boolean;
  onLayerWeightModeChange: (mode: string) => void;
  onLayerActivationModeChange: (mode: string) => void;
  layerEstimate?: {
    size_kb?: number;
    size_kb_fp32?: number;
    latency_ms?: number;
    latency_ms_fp32?: number;
  };
}

function normalizeBundle(payload: DistMap | DistBundle): DistBundle {
  if ("pre" in payload || "post" in payload || "error" in payload || "error_mae" in payload || "error_rmse" in payload) {
    return payload as DistBundle;
  }
  // Backward compatibility with old payloads that only contained error values.
  return { error: payload as DistMap };
}

function compareTraces(pre: DistMap = {}, post: DistMap = {}) {
  const channels = Array.from(new Set([...Object.keys(pre), ...Object.keys(post)])).sort(
    (a, b) => Number(a) - Number(b)
  );
  const traces: any[] = [];
  channels.forEach((channel) => {
    const preVals = pre[channel] ?? [];
    const postVals = post[channel] ?? [];
    traces.push({
      type: "violin",
      orientation: "h" as const,
      y: preVals.map(() => channel),
      x: preVals,
      side: "negative" as const,
      name: "Pre-Quant (FP32)",
      legendgroup: "pre",
      scalegroup: `ch_${channel}`,
      line: { color: "#1d4ed8" },
      fillcolor: "rgba(29, 78, 216, 0.35)",
      points: false,
      meanline: { visible: true },
      showlegend: channel === channels[0],
    });
    traces.push({
      type: "violin",
      orientation: "h" as const,
      y: postVals.map(() => channel),
      x: postVals,
      side: "positive" as const,
      name: "Post-Quant (Dequantized)",
      legendgroup: "post",
      scalegroup: `ch_${channel}`,
      line: { color: "#b91c1c" },
      fillcolor: "rgba(185, 28, 28, 0.35)",
      points: false,
      meanline: { visible: true },
      showlegend: channel === channels[0],
    });
  });
  return traces;
}

function errorTraces(dist: DistMap = {}) {
  return Object.entries(dist).map(([channel, values]) => ({
    type: "violin",
    orientation: "h" as const,
    y: values.map(() => channel),
    x: values,
    name: `ch_${channel} error`,
    points: false,
    box: { visible: false },
    meanline: { visible: true },
    line: { color: "#6b21a8" },
    fillcolor: "rgba(107, 33, 168, 0.3)",
    showlegend: false,
  }));
}

function buildPlot(
  traces: any[],
  title: string,
  xTitle: string,
  channelCount: number,
  expanded: boolean,
  onExpand: () => void
) {
  const baseHeight = Math.max(340, Math.min(1400, channelCount * (expanded ? 28 : 18) + 180));
  return (
    <div className="plot-card">
      <div className="plot-head">
        <h4>{title}</h4>
        <button type="button" onClick={onExpand}>
          Enlarge
        </button>
      </div>
      <Plot
        data={traces}
        layout={{
          title,
          height: baseHeight,
          margin: { l: 70, r: 20, t: 56, b: 50 },
          yaxis: { title: "Channel (index)" },
          xaxis: { title: xTitle },
          violinmode: "overlay",
        }}
        style={{ width: "100%" }}
        config={{ displayModeBar: true, responsive: true }}
      />
    </div>
  );
}

export function LayerDetail({
  layerName,
  layerType,
  metricValues,
  activations,
  weights,
  biases,
  qparams,
  metric,
  layerWeightMode,
  layerActivationMode,
  hasWeights,
  readOnly,
  onLayerWeightModeChange,
  onLayerActivationModeChange,
  layerEstimate,
}: LayerDetailProps) {
  const [expandedPlot, setExpandedPlot] = useState<
    null | "weightsCompare" | "weightsError" | "actsCompare" | "actsError" | "biasCompare" | "biasError"
  >(null);
  const weightBundle = normalizeBundle(weights);
  const activationBundle = normalizeBundle(activations);
  const biasBundle = normalizeBundle(biases);

  const weightCompare = useMemo(
    () => compareTraces(weightBundle.pre, weightBundle.post),
    [weightBundle.pre, weightBundle.post]
  );
  const selectedWeightError =
    metric === "rmse"
      ? (weightBundle.error_rmse ?? weightBundle.error)
      : (weightBundle.error_mae ?? weightBundle.error);
  const weightError = useMemo(() => errorTraces(selectedWeightError), [selectedWeightError]);
  const actCompare = useMemo(
    () => compareTraces(activationBundle.pre, activationBundle.post),
    [activationBundle.pre, activationBundle.post]
  );
  const selectedActError =
    metric === "rmse"
      ? (activationBundle.error_rmse ?? activationBundle.error)
      : (activationBundle.error_mae ?? activationBundle.error);
  const actError = useMemo(() => errorTraces(selectedActError), [selectedActError]);
  const biasCompare = useMemo(
    () => compareTraces(biasBundle.pre, biasBundle.post),
    [biasBundle.pre, biasBundle.post]
  );
  const selectedBiasError =
    metric === "rmse"
      ? (biasBundle.error_rmse ?? biasBundle.error)
      : (biasBundle.error_mae ?? biasBundle.error);
  const biasError = useMemo(() => errorTraces(selectedBiasError), [selectedBiasError]);

  const weightChannels = Object.keys(weightBundle.pre ?? weightBundle.post ?? selectedWeightError ?? {}).length;
  const actChannels = Object.keys(activationBundle.pre ?? activationBundle.post ?? selectedActError ?? {}).length;
  const biasChannels = Object.keys(biasBundle.pre ?? biasBundle.post ?? selectedBiasError ?? {}).length;

  return (
    <div className="detail">
      <div className="plot-head">
        <div>
          <h3>{layerName}</h3>
          {layerType ? <div className="layer-subtype">{layerType}</div> : null}
        </div>
        <label>
          Activation precision:
          <select value={layerActivationMode} disabled={readOnly} onChange={(e) => onLayerActivationModeChange(e.target.value)}>
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
          Weight precision:
          <select
            value={layerWeightMode}
            onChange={(e) => onLayerWeightModeChange(e.target.value)}
            disabled={readOnly || !hasWeights}
          >
            <option value="int2">int2</option>
            <option value="int4">int4</option>
            <option value="int8">int8</option>
            <option value="int12">int12</option>
            <option value="int16">int16</option>
            <option value="fp16">fp16</option>
            <option value="fp32">fp32</option>
          </select>
        </label>
      </div>
      <div className="metrics">
        <span>MAE: {metricValues?.mae?.toFixed(6) ?? "-"}</span>
        <span>RMSE: {metricValues?.rmse?.toFixed(6) ?? "-"}</span>
        <span>KL: {metricValues?.kl?.toFixed(6) ?? "-"}</span>
      </div>
      <div className="metrics">
        <span>
          Size: {layerEstimate?.size_kb?.toFixed(2) ?? "-"}KB / {layerEstimate?.size_kb_fp32?.toFixed(2) ?? "-"}KB
        </span>
        <span>
          Latency: {layerEstimate?.latency_ms?.toFixed(3) ?? "-"}ms / {layerEstimate?.latency_ms_fp32?.toFixed(3) ?? "-"}ms
        </span>
      </div>
      <div className="qparams">
        <h4>QParams</h4>
        <pre>{JSON.stringify(qparams ?? {}, null, 2)}</pre>
      </div>
      <div className="plots">
        {buildPlot(
          actCompare,
          "Activations: Pre-Quant (FP32) vs Post-Quant (Dequantized)",
          "Activation Value",
          actChannels,
          expandedPlot === "actsCompare",
          () => setExpandedPlot("actsCompare")
        )}
        {buildPlot(
          actError,
          `Activations: ${metric.toUpperCase()} Error Distribution`,
          metric === "rmse"
            ? "Squared Error ((FP32 - Dequantized)^2)"
            : metric === "kl"
              ? "Absolute Error proxy for KL visualization"
              : "Absolute Error (|FP32 - Dequantized|)",
          actChannels,
          expandedPlot === "actsError",
          () => setExpandedPlot("actsError")
        )}
        {buildPlot(
          weightCompare,
          "Weights: Pre-Quant (FP32) vs Post-Quant (Dequantized)",
          "Weight Value",
          weightChannels,
          expandedPlot === "weightsCompare",
          () => setExpandedPlot("weightsCompare")
        )}
        {buildPlot(
          weightError,
          `Weights: ${metric.toUpperCase()} Error Distribution`,
          metric === "rmse"
            ? "Squared Error ((FP32 - Dequantized)^2)"
            : metric === "kl"
              ? "Absolute Error proxy for KL visualization"
              : "Absolute Error (|FP32 - Dequantized|)",
          weightChannels,
          expandedPlot === "weightsError",
          () => setExpandedPlot("weightsError")
        )}
        {biasChannels > 0 &&
          buildPlot(
            biasCompare,
            "Bias: Pre-Quant (FP32) vs Post-Quant (Dequantized int32)",
            "Bias Value",
            biasChannels,
            expandedPlot === "biasCompare",
            () => setExpandedPlot("biasCompare")
          )}
        {biasChannels > 0 &&
          buildPlot(
            biasError,
            `Bias: ${metric.toUpperCase()} Error Distribution`,
            metric === "rmse"
              ? "Squared Error ((FP32 - Dequantized)^2)"
              : metric === "kl"
                ? "Absolute Error proxy for KL visualization"
                : "Absolute Error (|FP32 - Dequantized|)",
            biasChannels,
            expandedPlot === "biasError",
            () => setExpandedPlot("biasError")
          )}
      </div>
      {expandedPlot && (
        <div className="plot-modal" onClick={() => setExpandedPlot(null)}>
          <div className="plot-modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" type="button" onClick={() => setExpandedPlot(null)}>
              Close
            </button>
            {expandedPlot === "weightsCompare" &&
              buildPlot(
                weightCompare,
                "Weights: Pre-Quant (FP32) vs Post-Quant (Dequantized)",
                "Weight Value",
                weightChannels,
                true,
                () => undefined
              )}
            {expandedPlot === "weightsError" &&
              buildPlot(
                weightError,
                `Weights: ${metric.toUpperCase()} Error Distribution`,
                metric === "rmse"
                  ? "Squared Error ((FP32 - Dequantized)^2)"
                  : metric === "kl"
                    ? "Absolute Error proxy for KL visualization"
                    : "Absolute Error (|FP32 - Dequantized|)",
                weightChannels,
                true,
                () => undefined
              )}
            {expandedPlot === "actsCompare" &&
              buildPlot(
                actCompare,
                "Activations: Pre-Quant (FP32) vs Post-Quant (Dequantized)",
                "Activation Value",
                actChannels,
                true,
                () => undefined
              )}
            {expandedPlot === "actsError" &&
              buildPlot(
                actError,
                `Activations: ${metric.toUpperCase()} Error Distribution`,
                metric === "rmse"
                  ? "Squared Error ((FP32 - Dequantized)^2)"
                  : metric === "kl"
                    ? "Absolute Error proxy for KL visualization"
                    : "Absolute Error (|FP32 - Dequantized|)",
                actChannels,
                true,
                () => undefined
              )}
            {expandedPlot === "biasCompare" &&
              buildPlot(
                biasCompare,
                "Bias: Pre-Quant (FP32) vs Post-Quant (Dequantized int32)",
                "Bias Value",
                biasChannels,
                true,
                () => undefined
              )}
            {expandedPlot === "biasError" &&
              buildPlot(
                biasError,
                `Bias: ${metric.toUpperCase()} Error Distribution`,
                metric === "rmse"
                  ? "Squared Error ((FP32 - Dequantized)^2)"
                  : metric === "kl"
                    ? "Absolute Error proxy for KL visualization"
                    : "Absolute Error (|FP32 - Dequantized|)",
                biasChannels,
                true,
                () => undefined
              )}
          </div>
        </div>
      )}
    </div>
  );
}
