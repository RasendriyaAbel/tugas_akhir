import { readModelLabels, readTrainedModelInfo } from "@/lib/modelInfo";
import type { NilmData } from "@/types/nilm";

interface DeviceProfile {
  powerBase: number;
  powerSwing: number;
  powerNoise: number;
  powerFactorMin: number;
  powerFactorMax: number;
  confidenceMin: number;
  confidenceMax: number;
  holdMin: number;
  holdMax: number;
}

interface MockContext {
  labels: string[];
  modelVersion: string;
}

interface SimulatorState {
  seed: number;
  phase: number;
  energy: number;
  lastTimestampMs: number;
  currentPower: number;
  currentLabel: string;
  currentSegmentLength: number;
  currentSegmentStep: number;
  sequenceIndex: number;
}

const FALLBACK_LABELS = [
  "charger_hp",
  "rice_cooker",
  "fan_room",
  "laptop_charger",
  "desk_lamp",
  "uncertain",
  "other",
  "idle",
];

const state: SimulatorState = {
  seed: 20260504,
  phase: 0,
  energy: 0.32,
  lastTimestampMs: 0,
  currentPower: 12,
  currentLabel: "idle",
  currentSegmentLength: 0,
  currentSegmentStep: 0,
  sequenceIndex: 0,
};

let contextPromise: Promise<MockContext> | null = null;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function round(value: number, digits = 1) {
  return Number(value.toFixed(digits));
}

function nextRandom() {
  state.seed = (1664525 * state.seed + 1013904223) >>> 0;
  return state.seed / 4294967296;
}

function randomBetween(min: number, max: number) {
  return min + nextRandom() * (max - min);
}

function randomInt(min: number, max: number) {
  return Math.round(randomBetween(min, max));
}

function normalizeLabel(label: string) {
  return label.trim().toLowerCase();
}

function findSpecialLabel(labels: string[], keyword: string) {
  const normalizedKeyword = keyword.toLowerCase();
  return (
    labels.find((label) => normalizeLabel(label) === normalizedKeyword) ??
    labels.find((label) => normalizeLabel(label).includes(normalizedKeyword))
  );
}

function buildLabelSequence(labels: string[]) {
  const idleLabel = findSpecialLabel(labels, "idle");
  const uncertainLabel = findSpecialLabel(labels, "uncertain");
  const otherLabel = findSpecialLabel(labels, "other");
  const regularLabels = labels.filter(
    (label) => ![idleLabel, uncertainLabel, otherLabel].includes(label),
  );

  const sequence: string[] = [];

  if (idleLabel) {
    sequence.push(idleLabel);
  }

  regularLabels.forEach((label, index) => {
    sequence.push(label);

    if (idleLabel && index % 2 === 1) {
      sequence.push(idleLabel);
    }
  });

  if (otherLabel) {
    sequence.push(otherLabel);
  }

  if (idleLabel) {
    sequence.push(idleLabel);
  }

  if (uncertainLabel) {
    sequence.push(uncertainLabel);
  }

  if (idleLabel) {
    sequence.push(idleLabel);
  }

  return sequence.length > 0 ? sequence : labels;
}

function getProfile(label: string): DeviceProfile {
  const normalized = normalizeLabel(label);

  if (normalized.includes("idle")) {
    return {
      powerBase: 8,
      powerSwing: 4,
      powerNoise: 1.5,
      powerFactorMin: 0.55,
      powerFactorMax: 0.72,
      confidenceMin: 93,
      confidenceMax: 99.4,
      holdMin: 2,
      holdMax: 4,
    };
  }

  if (normalized.includes("uncertain")) {
    return {
      powerBase: 120,
      powerSwing: 55,
      powerNoise: 18,
      powerFactorMin: 0.68,
      powerFactorMax: 0.83,
      confidenceMin: 35,
      confidenceMax: 59,
      holdMin: 2,
      holdMax: 3,
    };
  }

  if (normalized.includes("other")) {
    return {
      powerBase: 150,
      powerSwing: 70,
      powerNoise: 16,
      powerFactorMin: 0.72,
      powerFactorMax: 0.9,
      confidenceMin: 58,
      confidenceMax: 78,
      holdMin: 2,
      holdMax: 4,
    };
  }

  if (normalized.includes("rice") || normalized.includes("cooker")) {
    return {
      powerBase: 320,
      powerSwing: 85,
      powerNoise: 14,
      powerFactorMin: 0.9,
      powerFactorMax: 0.98,
      confidenceMin: 86,
      confidenceMax: 98,
      holdMin: 3,
      holdMax: 6,
    };
  }

  if (normalized.includes("fan")) {
    return {
      powerBase: 62,
      powerSwing: 18,
      powerNoise: 5,
      powerFactorMin: 0.74,
      powerFactorMax: 0.88,
      confidenceMin: 78,
      confidenceMax: 94,
      holdMin: 3,
      holdMax: 5,
    };
  }

  if (normalized.includes("lamp") || normalized.includes("light")) {
    return {
      powerBase: 18,
      powerSwing: 6,
      powerNoise: 2,
      powerFactorMin: 0.76,
      powerFactorMax: 0.89,
      confidenceMin: 82,
      confidenceMax: 97,
      holdMin: 2,
      holdMax: 4,
    };
  }

  if (normalized.includes("charger") || normalized.includes("phone") || normalized.includes("laptop")) {
    return {
      powerBase: 44,
      powerSwing: 14,
      powerNoise: 4,
      powerFactorMin: 0.78,
      powerFactorMax: 0.91,
      confidenceMin: 79,
      confidenceMax: 96,
      holdMin: 3,
      holdMax: 5,
    };
  }

  return {
    powerBase: 92,
    powerSwing: 26,
    powerNoise: 8,
    powerFactorMin: 0.74,
    powerFactorMax: 0.9,
    confidenceMin: 72,
    confidenceMax: 92,
    holdMin: 2,
    holdMax: 4,
  };
}

async function readMockContext(): Promise<MockContext> {
  if (!contextPromise) {
    contextPromise = Promise.all([readModelLabels(), readTrainedModelInfo()])
      .then(([labels, modelInfo]) => ({
        labels: labels.length > 0 ? labels : FALLBACK_LABELS,
        modelVersion: modelInfo.model_name || "best_nilm_model",
      }))
      .catch(() => ({
        labels: FALLBACK_LABELS,
        modelVersion: "best_nilm_model",
      }));
  }

  return contextPromise;
}

function beginNextSegment(labels: string[]) {
  const sequence = buildLabelSequence(labels);
  state.currentLabel = sequence[state.sequenceIndex % sequence.length] ?? labels[0] ?? "idle";
  state.currentSegmentLength = randomInt(
    getProfile(state.currentLabel).holdMin,
    getProfile(state.currentLabel).holdMax,
  );
  state.currentSegmentStep = 0;
  state.sequenceIndex += 1;
}

export async function getNextMockBlynkData(): Promise<NilmData> {
  const context = await readMockContext();

  if (state.currentSegmentLength === 0 || state.currentSegmentStep >= state.currentSegmentLength) {
    beginNextSegment(context.labels);
  }

  const profile = getProfile(state.currentLabel);
  const now = Date.now();
  const elapsedMs = state.lastTimestampMs === 0 ? 1000 : clamp(now - state.lastTimestampMs, 900, 2500);
  const progress = state.currentSegmentStep / Math.max(state.currentSegmentLength - 1, 1);

  state.phase += 0.45 + randomBetween(0.05, 0.18);

  const voltage = round(220 + Math.sin(state.phase / 3) * 3.2 + randomBetween(-1.1, 1.1), 1);
  const rawPower =
    profile.powerBase +
    Math.sin(progress * Math.PI) * profile.powerSwing * 0.35 +
    Math.sin(state.phase) * profile.powerSwing * 0.55 +
    randomBetween(-profile.powerNoise, profile.powerNoise);

  state.currentPower = clamp((state.currentPower * 0.72) + (rawPower * 0.28), 4, 450);

  const powerFactor = round(
    clamp(
      profile.powerFactorMin +
        ((Math.sin(state.phase / 4) + 1) / 2) * (profile.powerFactorMax - profile.powerFactorMin),
      0.5,
      0.99,
    ),
    2,
  );
  const current = round(state.currentPower / Math.max(voltage * powerFactor, 1), 2);
  const frequency = round(50 + Math.sin(state.phase / 5) * 0.08 + randomBetween(-0.03, 0.03), 1);

  state.energy += (state.currentPower * elapsedMs) / 3600000000;
  state.lastTimestampMs = now;
  state.currentSegmentStep += 1;

  const confidence = round(
    clamp(
      profile.confidenceMin +
        ((Math.sin((state.phase / 2) + progress) + 1) / 2) *
          (profile.confidenceMax - profile.confidenceMin),
      0,
      99.9,
    ),
    1,
  );

  return {
    voltage,
    current,
    power: round(state.currentPower, 1),
    energy: round(state.energy, 3),
    frequency,
    power_factor: powerFactor,
    device_detected: state.currentLabel,
    confidence,
    model_version: context.modelVersion,
    timestamp: new Date(now).toISOString(),
  };
}
