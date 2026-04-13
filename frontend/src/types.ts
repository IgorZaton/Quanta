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
