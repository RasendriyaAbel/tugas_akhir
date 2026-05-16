import { inferFromMlService } from "@/lib/mlService";
import type { NilmData } from "@/types/nilm";

const TELEMETRY_KEY_MAP = {
  voltage: process.env.THINGSBOARD_KEY_VOLTAGE?.trim() || "tegangan",
  current: process.env.THINGSBOARD_KEY_CURRENT?.trim() || "arus",
  power: process.env.THINGSBOARD_KEY_POWER?.trim() || "daya",
  energy: process.env.THINGSBOARD_KEY_ENERGY?.trim() || "kwh",
  frequency: process.env.THINGSBOARD_KEY_FREQUENCY?.trim() || "frekuensi",
  power_factor: process.env.THINGSBOARD_KEY_POWER_FACTOR?.trim() || "power_factor",
} as const;

type MetricField = keyof typeof TELEMETRY_KEY_MAP;
type TelemetryKey = (typeof TELEMETRY_KEY_MAP)[MetricField];

type ThingsBoardAuthResponse = {
  token?: string;
  refreshToken?: string;
};

type ThingsBoardDeviceInfo = {
  id?: {
    id?: string;
  };
  name?: string;
};

type TelemetryPoint = {
  ts?: number;
  value?: string | number;
};

type ThingsBoardTelemetry = Partial<Record<TelemetryKey, TelemetryPoint[]>>;

let cachedJwt: string | null = null;

function getThingsBoardBaseUrl() {
  return (process.env.THINGSBOARD_BASE_URL || "").trim().replace(/\/$/, "");
}

function getThingsBoardDeviceToken() {
  return process.env.THINGSBOARD_ACCESS_TOKEN?.trim() || "";
}

function getThingsBoardJwtToken() {
  return process.env.THINGSBOARD_JWT_TOKEN?.trim() || "";
}

function getThingsBoardUsername() {
  return process.env.THINGSBOARD_USERNAME?.trim() || "";
}

function getThingsBoardPassword() {
  return process.env.THINGSBOARD_PASSWORD?.trim() || "";
}

function getThingsBoardDeviceId() {
  return process.env.THINGSBOARD_DEVICE_ID?.trim() || "";
}

async function parseThingsBoardJson<T>(response: Response, endpoint: string): Promise<T> {
  const rawText = await response.text();
  const trimmed = rawText.trim();

  if (!trimmed) {
    throw new Error(`ThingsBoard endpoint ${endpoint} mengembalikan response kosong.`);
  }

  try {
    return JSON.parse(trimmed) as T;
  } catch {
    throw new Error(`ThingsBoard endpoint ${endpoint} tidak mengembalikan JSON valid.`);
  }
}

async function loginThingsBoard(forceRefresh = false) {
  const envJwt = getThingsBoardJwtToken();

  if (envJwt && !forceRefresh) {
    cachedJwt = envJwt;
    return envJwt;
  }

  if (cachedJwt && !forceRefresh) {
    return cachedJwt;
  }

  const baseUrl = getThingsBoardBaseUrl();
  const username = getThingsBoardUsername();
  const password = getThingsBoardPassword();

  if (!baseUrl) {
    throw new Error("THINGSBOARD_BASE_URL belum diatur.");
  }

  if (!username || !password) {
    throw new Error("THINGSBOARD_JWT_TOKEN belum diatur, dan fallback THINGSBOARD_USERNAME/THINGSBOARD_PASSWORD juga belum lengkap.");
  }

  const endpoint = `${baseUrl}/api/auth/login`;
  const response = await fetch(endpoint, {
    method: "POST",
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      username,
      password,
    }),
  });

  const payload = await parseThingsBoardJson<ThingsBoardAuthResponse>(response, endpoint);

  if (!response.ok || !payload.token) {
    throw new Error("Login ThingsBoard gagal. Periksa username/password tenant.");
  }

  cachedJwt = payload.token;
  return cachedJwt;
}

async function thingsBoardFetchJson<T>(path: string, init?: RequestInit, retry = true): Promise<T> {
  const baseUrl = getThingsBoardBaseUrl();
  const token = await loginThingsBoard();
  const endpoint = `${baseUrl}${path}`;

  const response = await fetch(endpoint, {
    ...init,
    cache: "no-store",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      "X-Authorization": `Bearer ${token}`,
      ...(init?.headers ?? {}),
    },
  });

  if ((response.status === 401 || response.status === 403) && retry) {
    await loginThingsBoard(true);
    return thingsBoardFetchJson<T>(path, init, false);
  }

  return parseThingsBoardJson<T>(response, endpoint);
}

async function resolveDeviceId() {
  const deviceId = getThingsBoardDeviceId();
  if (deviceId) {
    return deviceId;
  }

  const accessToken = getThingsBoardDeviceToken();
  if (!accessToken) {
    throw new Error(
      "THINGSBOARD_DEVICE_ID atau THINGSBOARD_ACCESS_TOKEN belum diatur. " +
      "Gunakan deviceId atau access token agar ThingsBoard dapat menemukan perangkat."
    );
  }

  const payload = await thingsBoardFetchJson<ThingsBoardDeviceInfo>(
    `/api/device/info?deviceToken=${encodeURIComponent(accessToken)}`,
  );
  const resolvedId = payload.id?.id;

  if (!resolvedId) {
    throw new Error("Device ID ThingsBoard tidak berhasil ditemukan dari access token.");
  }

  return resolvedId;
}

function getLatestValue(points: TelemetryPoint[] | undefined, key: MetricField) {
  const rawValue = points?.[0]?.value;
  const numericValue = Number(rawValue);

  if (Number.isNaN(numericValue)) {
    throw new Error(`nilai ${key} dari ThingsBoard tidak valid: ${String(rawValue)}`);
  }

  return numericValue;
}

export async function fetchLatestThingsBoardDataWithMeta(): Promise<{ data: NilmData; notice?: string }> {
  const baseUrl = getThingsBoardBaseUrl();
  const envJwt = getThingsBoardJwtToken();

  if (!baseUrl) {
    throw new Error("THINGSBOARD_BASE_URL belum diatur.");
  }

  const deviceId = await resolveDeviceId();
  const endTs = Date.now();
  const startTs = endTs - 300000;
  const query = new URLSearchParams({
    keys: Object.values(TELEMETRY_KEY_MAP).join(","),
    startTs: String(startTs),
    endTs: String(endTs),
    limit: "1",
    agg: "NONE",
  });

  const telemetry = await thingsBoardFetchJson<ThingsBoardTelemetry>(
    `/api/plugins/telemetry/DEVICE/${deviceId}/values/timeseries?${query.toString()}`,
  );

  const sample = {
    voltage: getLatestValue(telemetry[TELEMETRY_KEY_MAP.voltage], "voltage"),
    current: getLatestValue(telemetry[TELEMETRY_KEY_MAP.current], "current"),
    power: getLatestValue(telemetry[TELEMETRY_KEY_MAP.power], "power"),
    energy: getLatestValue(telemetry[TELEMETRY_KEY_MAP.energy], "energy"),
    frequency: getLatestValue(telemetry[TELEMETRY_KEY_MAP.frequency], "frequency"),
    power_factor: getLatestValue(telemetry[TELEMETRY_KEY_MAP.power_factor], "power_factor"),
  };

  let inference;
  const notices: string[] = [];

  try {
    inference = await inferFromMlService(sample);

    if (inference.label_source?.startsWith("meta_nilm.json")) {
      notices.push("Label inferensi berasal dari file meta_nilm.json.");
    } else if (inference.label_source === "generated") {
      notices.push("Label model masih placeholder unknown_* karena mapping label final belum tersedia.");
    }

    if (inference.buffer?.status && inference.buffer.status !== "READY") {
      notices.push(`Buffer inferensi model masih ${inference.buffer.status.toLowerCase()}. Prediksi awal dapat berubah.`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown ML service error";
    notices.push(`ML service tidak dapat diakses (${message}). Menggunakan fallback label unknown dan confidence 0.`);
    inference = {
      success: false,
      label: "unknown",
      confidence: 0,
      model_version: "N/A",
      timestamp: new Date().toISOString(),
      label_source: "fallback",
    };
  }

  if (inference.buffer?.status && inference.buffer.status !== "READY") {
    notices.push(`Buffer inferensi model masih ${inference.buffer.status.toLowerCase()}. Prediksi awal dapat berubah.`);
  }

  notices.push(
    `Dashboard membaca telemetry live dari ThingsBoard${baseUrl ? ` di ${baseUrl}` : ""} dan meneruskan sample ke ML service untuk inferensi.${envJwt ? " Autentikasi memakai JWT manual." : " Autentikasi memakai login tenant otomatis."}`,
  );

  return {
    data: {
      ...sample,
      device_detected: inference.label ?? "unknown",
      confidence: inference.confidence ?? 0,
      model_version: inference.model_version ?? "N/A",
      timestamp: inference.timestamp ?? new Date().toISOString(),
    },
    notice: notices.join(" "),
  };
}
