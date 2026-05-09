import { clsx, type ClassValue } from "clsx";

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

export function formatNumber(value: number, digits = 1) {
  return new Intl.NumberFormat("id-ID", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatCurrency(value: number) {
  return new Intl.NumberFormat("id-ID", {
    style: "currency",
    currency: "IDR",
    maximumFractionDigits: 2,
  }).format(value);
}

export function parseTimestamp(timestamp?: string | null) {
  if (!timestamp) {
    return null;
  }

  const normalized = timestamp.includes("T") ? timestamp : timestamp.replace(" ", "T");
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export function formatTimestamp(timestamp?: string | null) {
  const parsed = parseTimestamp(timestamp);
  if (!parsed) {
    return "Waiting for sensor data...";
  }

  return parsed.toLocaleString("id-ID", {
    dateStyle: "medium",
    timeStyle: "medium",
  });
}

export function formatChartTime(timestamp: string) {
  const parsed = parseTimestamp(timestamp);
  if (!parsed) {
    return timestamp;
  }

  return parsed.toLocaleTimeString("id-ID", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatRelativeTime(timestamp?: string | null) {
  const parsed = parseTimestamp(timestamp);
  if (!parsed) {
    return "Belum ada data";
  }

  const seconds = Math.max(0, Math.round((Date.now() - parsed.getTime()) / 1000));

  if (seconds < 5) {
    return "baru saja";
  }

  if (seconds < 60) {
    return `${seconds} detik lalu`;
  }

  const minutes = Math.round(seconds / 60);

  if (minutes < 60) {
    return `${minutes} menit lalu`;
  }

  const hours = Math.round(minutes / 60);
  return `${hours} jam lalu`;
}

export function isDataStale(timestamp?: string | null, refreshInterval = 3000) {
  const parsed = parseTimestamp(timestamp);
  if (!parsed) {
    return true;
  }

  const staleThreshold = Math.max(refreshInterval * 2, 15000);
  return Date.now() - parsed.getTime() > staleThreshold;
}

export function normalizeDeviceName(device: string) {
  return device
    .replaceAll("_", " ")
    .replaceAll("+", " + ")
    .split(/\s+/)
    .filter(Boolean)
    .map((token) => {
      if (token.toLowerCase() === "hp") {
        return "HP";
      }
      return token.charAt(0).toUpperCase() + token.slice(1).toLowerCase();
    })
    .join(" ");
}

export function formatDeviceLabel(device: string) {
  return normalizeDeviceName(device);
}


export function getConfidenceLabel(confidence: number) {
  if (confidence >= 85) {
    return "High Confidence";
  }

  if (confidence >= 60) {
    return "Medium Confidence";
  }

  return "Low Confidence";
}

export function getConfidenceTone(confidence: number) {
  if (confidence >= 85) {
    return "blue";
  }

  if (confidence >= 60) {
    return "orange";
  }

  return "soft-orange";
}

export function getEfficiencyRecommendation(power: number) {
  if (power < 50) {
    return "Konsumsi rendah, perangkat tergolong hemat.";
  }

  if (power <= 300) {
    return "Konsumsi sedang, pantau durasi penggunaan.";
  }

  return "Konsumsi tinggi, gunakan seperlunya untuk efisiensi energi.";
}
