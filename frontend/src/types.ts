export type MetricKey = "mae" | "rmse" | "kl";

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  has_weights?: boolean;
  metrics: Record<MetricKey, number>;
  estimates?: {
    mode?: string;
    size_kb?: number;
    latency_ms?: number;
  };
}

export interface GraphEdge {
  source: string;
  target: string;
}

export interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface LatestState {
  range_mode: string;
  global_weight_mode: string;
  global_activation_mode: string;
  layer_weight_modes: Record<string, string>;
  layer_activation_modes: Record<string, string>;
}

export interface RuntimePayload {
  run_id: string | null;
  loaded_from_run: boolean;
  runtime_recompute_enabled: boolean;
  finalized: boolean;
  available_strategies: string[];
  latest_state: LatestState | null;
}

export interface FinalizeResponse {
  status: string;
  run_id: string;
  finalized_path?: string;
}
