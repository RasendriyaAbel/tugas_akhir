export interface NilmData {
  voltage: number;
  current: number;
  power: number;
  energy: number;
  frequency: number;
  power_factor: number;
  device_detected: string;
  confidence: number;
  model_version: string;
  timestamp: string;
}

export type DataSource = "blynk" | "thingsboard" | "dummy";

export type NavigationSection =
  | "dashboard"
  | "device-detection"
  | "energy-analytics"
  | "system-status"
  | "settings";

export interface LatestBlynkResponse {
  success: boolean;
  data: NilmData | null;
  source: DataSource;
  last_updated: string;
  error?: string;
}

export interface DashboardSettings {
  tariff: number;
  refreshInterval: number;
}

export interface PowerHistoryPoint {
  timestamp: string;
  power: number;
  source: DataSource;
}

export interface TrainedModelInfo {
  model_name: string;
  keras_version: string;
  saved_at: string;
  input_shape: number[];
  output_units: number | null;
  output_activation: string | null;
  total_layers: number;
  architecture: string[];
  notes: string[];
}

export interface TrainedModelInfoResponse {
  success: boolean;
  data: TrainedModelInfo | null;
  error?: string;
}
