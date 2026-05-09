import { readFile } from "node:fs/promises";
import path from "node:path";

import type { TrainedModelInfo } from "@/types/nilm";

interface ModelLayerConfig {
  class_name?: string;
  config?: {
    name?: string;
    batch_shape?: Array<number | null>;
    units?: number;
    activation?: string;
  };
}

interface RawModelConfig {
  config?: {
    name?: string;
    layers?: ModelLayerConfig[];
  };
}

interface RawMetadata {
  keras_version?: string;
  date_saved?: string;
}

interface RawLabels {
  labels?: string[];
}

interface RawMetaNilm {
  model_version?: string;
  keras_version?: string;
  date_saved?: string;
  feature_cols?: string[];
  n_features?: number;
  window_size?: number;
  stride?: number;
  n_devices?: number;
  n_classes?: number;
  threshold?: number;
  devices?: string[];
  classes?: string[];
}

function getModelDir() {
  return process.env.NILM_MODEL_DIR?.trim() || path.join(process.cwd(), "src", "nilm_models_v9");
}

export async function readModelLabels(): Promise<string[]> {
  const modelDir = getModelDir();

  try {
    const labelsRaw = await readFile(path.join(modelDir, "labels.json"), "utf8");
    const labels = (JSON.parse(labelsRaw) as RawLabels).labels;

    if (!Array.isArray(labels) || labels.some((label) => typeof label !== "string")) {
      throw new Error("labels.json tidak valid. Field 'labels' harus berupa array string.");
    }

    return labels.map((label) => label.trim()).filter(Boolean);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);

    if (!message.includes("ENOENT")) {
      throw error;
    }
  }

  const metaNilm = await readMetaNilm(modelDir);
  if (metaNilm?.devices?.length) {
    return metaNilm.devices.map((label) => label.trim()).filter(Boolean);
  }

  if (metaNilm?.classes?.length) {
    return metaNilm.classes.map((label) => label.trim()).filter(Boolean);
  }

  const configRaw = await readFile(path.join(modelDir, "config.json"), "utf8");
  const config = JSON.parse(configRaw) as RawModelConfig;
  const layers = config.config?.layers ?? [];
  const outputLayer = [...layers]
    .reverse()
    .find((layer) => layer.class_name === "Dense" && layer.config?.activation === "softmax");
  const outputUnits = outputLayer?.config?.units ?? 0;

  if (!outputUnits) {
    throw new Error("labels.json tidak ditemukan dan output layer model tidak dapat dibaca.");
  }

  return Array.from({ length: outputUnits }, (_, index) => `unknown_${index}`);
}

async function readMetaNilm(modelDir: string): Promise<RawMetaNilm | null> {
  try {
    const raw = await readFile(path.join(modelDir, "meta_nilm.json"), "utf8");
    const meta = JSON.parse(raw) as RawMetaNilm;
    return meta;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!message.includes("ENOENT")) {
      throw error;
    }
  }

  return null;
}

export async function readTrainedModelInfo(): Promise<TrainedModelInfo> {
  const modelDir = getModelDir();
  const metaNilm = await readMetaNilm(modelDir);

  if (metaNilm) {
    const labels = await readModelLabels();
    const usesDeviceOutputs = Array.isArray(metaNilm.devices) && metaNilm.devices.length > 0;
    const outputUnits = usesDeviceOutputs
      ? metaNilm.n_devices ?? labels.length
      : metaNilm.n_classes ?? labels.length;
    const outputActivation = usesDeviceOutputs ? "sigmoid" : "softmax";
    const featureCols = Array.isArray(metaNilm.feature_cols)
      ? metaNilm.feature_cols.filter((value): value is string => typeof value === "string" && value.trim().length > 0)
      : [];

    return {
      model_name: metaNilm.model_version ?? path.basename(modelDir),
      keras_version: metaNilm.keras_version ?? "unknown",
      saved_at: metaNilm.date_saved ?? "unknown",
      input_shape: [metaNilm.window_size ?? 0, metaNilm.n_features ?? 0].filter(
        (value): value is number => typeof value === "number" && value > 0,
      ),
      output_units: outputUnits,
      output_activation: outputActivation,
      total_layers: 0,
      architecture: [],
      notes: [
        "Model NILM dibaca dari file meta_nilm.json.",
        `Input model memakai window_size=${metaNilm.window_size ?? "unknown"}, n_features=${metaNilm.n_features ?? "unknown"}, stride=${metaNilm.stride ?? "unknown"}.`,
        usesDeviceOutputs
          ? `Output model bersifat multi-label dengan ${outputUnits} device output dan aktivasi ${outputActivation}.`
          : `Output model memiliki ${outputUnits} kelas dengan aktivasi ${outputActivation}.`,
        `Label runtime:${labels.length > 0 ? ` ${labels.join(", ")}` : " (placeholder unknown_*)"}`,
        featureCols.length > 0
          ? `Urutan fitur input: ${featureCols.join(", ")}.`
          : "Urutan fitur input tidak tersedia di metadata.",
        typeof metaNilm.threshold === "number"
          ? `Threshold aktivasi device di metadata: ${metaNilm.threshold}.`
          : "Threshold aktivasi device tidak tersedia di metadata.",
        "Pastikan inferensi menggunakan preprocessing fitur dan window size yang sama seperti saat training.",
      ],
    };
  }

  const [metadataRaw, configRaw, labels] = await Promise.all([
    readFile(path.join(modelDir, "metadata.json"), "utf8"),
    readFile(path.join(modelDir, "config.json"), "utf8"),
    readModelLabels(),
  ]);

  const metadata = JSON.parse(metadataRaw) as RawMetadata;
  const config = JSON.parse(configRaw) as RawModelConfig;
  const layers = config.config?.layers ?? [];

  const inputLayer = layers.find((layer) => layer.class_name === "InputLayer");
  const outputLayer = [...layers]
    .reverse()
    .find((layer) => layer.class_name === "Dense" && layer.config?.activation === "softmax");
  const outputUnits = outputLayer?.config?.units ?? null;
  const configuredOnly = typeof outputUnits === "number" && labels.length !== outputUnits;

  return {
    model_name: config.config?.name ?? "unknown_model",
    keras_version: metadata.keras_version ?? "unknown",
    saved_at: metadata.date_saved ?? "unknown",
    input_shape: (inputLayer?.config?.batch_shape ?? []).filter(
      (value): value is number => typeof value === "number",
    ),
    output_units: outputUnits,
    output_activation: outputLayer?.config?.activation ?? null,
    total_layers: layers.length,
    architecture: layers.map((layer) => layer.class_name ?? "UnknownLayer"),
    notes: [
      "Model berhasil dibaca dari metadata dan config Keras.",
      "Model memiliki input sequence 99 timestep dengan 8 fitur per timestep.",
      `Output layer menggunakan softmax dengan ${outputUnits ?? "unknown"} kelas.`,
      configuredOnly
        ? `File label hanya mengonfigurasi ${labels.length} label notebook: ${labels.join(", ")}.`
        : `Label output model: ${labels.join(", ")}.`,
      "Untuk inferensi penuh di aplikasi, masih dibutuhkan label mapping dan preprocessing yang sama seperti saat training.",
    ],
  };
}
