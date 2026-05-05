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

function getModelDir() {
  return process.env.NILM_MODEL_DIR?.trim() || path.join(process.cwd(), "src", "best_nilm_model (1).keras");
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

export async function readTrainedModelInfo(): Promise<TrainedModelInfo> {
  const modelDir = getModelDir();
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
