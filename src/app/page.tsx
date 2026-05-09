"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BadgePercent,
  BatteryCharging,
  Cpu,
  Gauge,
  Radio,
  TriangleAlert,
  Waypoints,
  Zap,
} from "lucide-react";

import { DashboardCard } from "@/components/DashboardCard";
import { DeviceDetectionPanel } from "@/components/DeviceDetectionPanel";
import { EfficiencyRecommendation } from "@/components/EfficiencyRecommendation";
import { EnergyCostCard } from "@/components/EnergyCostCard";
import { Header } from "@/components/Header";
import { PowerChart } from "@/components/PowerChart";
import { SettingsPanel } from "@/components/SettingsPanel";
import { Sidebar } from "@/components/Sidebar";
import { SystemStatus } from "@/components/SystemStatus";
import {
  formatNumber,
  formatDeviceLabel,
  formatTimestamp,
  isDataStale,
} from "@/lib/utils";
import type {
  DataSource,
  DashboardSettings,
  LatestBlynkResponse,
  NavigationSection,
  NilmData,
  PowerHistoryPoint,
} from "@/types/nilm";

const DEFAULT_SETTINGS: DashboardSettings = {
  tariff: 1444.7,
  refreshInterval: Number(process.env.NEXT_PUBLIC_REFRESH_INTERVAL ?? 1000),
};

const SETTINGS_STORAGE_KEY = "nilm-dashboard-settings";
const HISTORY_STORAGE_KEY = "nilm-power-history";
const HISTORY_LIMIT = 24;

function loadInitialSettings() {
  if (typeof window === "undefined") {
    return DEFAULT_SETTINGS;
  }

  try {
    const storedSettings = localStorage.getItem(SETTINGS_STORAGE_KEY);

    if (!storedSettings) {
      return DEFAULT_SETTINGS;
    }

    const parsedSettings = JSON.parse(storedSettings) as Partial<DashboardSettings>;

    return {
      tariff: parsedSettings.tariff ?? DEFAULT_SETTINGS.tariff,
      refreshInterval: parsedSettings.refreshInterval ?? DEFAULT_SETTINGS.refreshInterval,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function loadInitialHistory() {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    const storedHistory = localStorage.getItem(HISTORY_STORAGE_KEY);

    if (!storedHistory) {
      return [];
    }

    const parsed = JSON.parse(storedHistory) as Array<Partial<PowerHistoryPoint>>;

    return parsed
      .filter((point) => typeof point?.timestamp === "string" && typeof point?.power === "number")
      .map((point) => ({
        timestamp: point.timestamp!,
        power: point.power!,
        source: "dummy" as const,
      }))
      .slice(-HISTORY_LIMIT);
  } catch {
    return [];
  }
}

function SectionCard({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-3xl border border-blue-100/80 bg-white/95 p-6 shadow-sm shadow-blue-100/70">
      <h2 className="text-xl font-semibold text-slate-900">{title}</h2>
      <p className="mt-1 text-sm text-slate-500">{description}</p>
      <div className="mt-5">{children}</div>
    </section>
  );
}

export default function HomePage() {
  const [activeSection, setActiveSection] = useState<NavigationSection>("dashboard");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [settings, setSettings] = useState<DashboardSettings>(loadInitialSettings);
  const [data, setData] = useState<NilmData | null>(null);
  const [history, setHistory] = useState<PowerHistoryPoint[]>(loadInitialHistory);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [source, setSource] = useState<DataSource>("thingsboard");
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history.slice(-HISTORY_LIMIT)));
  }, [history]);

  const fetchLatest = useCallback(async () => {
    const hasPreviousData = Boolean(data);

    if (hasPreviousData) {
      setIsRefreshing(true);
    } else {
      setIsLoading(true);
    }

    try {
      const response = await fetch("/api/blynk/latest", {
        cache: "no-store",
      });
      const payload = (await response.json()) as LatestBlynkResponse;

      if (!response.ok || !payload.success || !payload.data) {
        setError(payload.error ?? "Koneksi sumber data gagal");
        setNotice(null);
        return;
      }

      setData(payload.data);
      setSource(payload.source);
      setLastUpdated(payload.last_updated);
      setError(null);
      setNotice(payload.error ?? null);
      setHistory((previous) => {
        const nextPoint: PowerHistoryPoint = {
          timestamp: payload.data!.timestamp,
          power: payload.data!.power,
          source: payload.source,
        };

        if (previous.at(-1)?.timestamp === nextPoint.timestamp) {
          return previous;
        }

        return [...previous, nextPoint].slice(-HISTORY_LIMIT);
      });
    } catch {
      setError("Koneksi sumber data gagal");
      setNotice(null);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [data]);

  useEffect(() => {
    const initialFetch = window.setTimeout(() => {
      void fetchLatest();
    }, 0);
    const interval = window.setInterval(
      () => void fetchLatest(),
      Math.max(1000, settings.refreshInterval),
    );

    return () => {
      window.clearTimeout(initialFetch);
      window.clearInterval(interval);
    };
  }, [fetchLatest, settings.refreshInterval]);

  const isStale = useMemo(
    () => isDataStale(data?.timestamp ?? lastUpdated, settings.refreshInterval),
    [data?.timestamp, lastUpdated, settings.refreshInterval],
  );

  const statusLabel = !error && !isStale ? "Online" : "Offline";

  const summaryCards = [
    {
      title: "Voltage",
      value: data ? formatNumber(data.voltage, 1) : "--",
      unit: "Volt",
      description: "Tegangan RMS dari sensor PZEM-004T.",
      icon: Gauge,
      accent: "blue" as const,
    },
    {
      title: "Current",
      value: data ? formatNumber(data.current, 2) : "--",
      unit: "Ampere",
      description: "Arus beban yang terukur saat ini.",
      icon: Activity,
      accent: "emerald" as const,
    },
    {
      title: "Active Power",
      value: data ? formatNumber(data.power, 1) : "--",
      unit: "Watt",
      description: "Daya aktif real-time untuk analitik konsumsi.",
      icon: Zap,
      accent: "violet" as const,
    },
    {
      title: "Energy",
      value: data ? formatNumber(data.energy, 2) : "--",
      unit: "kWh",
      description: "Energi kumulatif yang telah terpakai.",
      icon: BatteryCharging,
      accent: "amber" as const,
    },
    {
      title: "Frequency",
      value: data ? formatNumber(data.frequency, 1) : "--",
      unit: "Hz",
      description: "Frekuensi sistem listrik terukur.",
      icon: Radio,
      accent: "blue" as const,
    },
    {
      title: "Power Factor",
      value: data ? formatNumber(data.power_factor, 2) : "--",
      description: "Faktor daya untuk evaluasi kualitas beban.",
      icon: Waypoints,
      accent: "emerald" as const,
    },
    {
      title: "Detected Device",
      value: data ? formatDeviceLabel(data.device_detected) : "--",
      description: "Perangkat yang sedang diidentifikasi model NILM.",
      icon: Cpu,
      accent: "violet" as const,
    },
    {
      title: "Confidence",
      value: data ? formatNumber(data.confidence, 1) : "--",
      unit: "%",
      description: "Probabilitas keyakinan model terhadap deteksi.",
      icon: BadgePercent,
      accent: "amber" as const,
    },
  ];

  const handleSettingsChange = <K extends keyof DashboardSettings>(
    key: K,
    value: DashboardSettings[K],
  ) => {
    setSettings((previous) => {
      if (key === "refreshInterval") {
        return {
          ...previous,
          refreshInterval: Math.max(1000, Number(value)),
        };
      }

      if (key === "tariff") {
        return {
          ...previous,
          tariff: Math.max(0, Number(value)),
        };
      }

      return {
        ...previous,
        [key]: value,
      };
    });
  };

  const topBanners = (
    <div className="space-y-3">
      {error ? (
        <div className="flex items-start gap-3 rounded-3xl border border-orange-200 bg-orange-50 px-4 py-3 text-sm text-orange-800">
          <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      ) : null}

      {notice ? (
        <div className="rounded-3xl border border-blue-100 bg-blue-50/80 px-4 py-3 text-sm text-blue-800">
          {notice}
        </div>
      ) : null}

      {isStale && data ? (
        <div className="rounded-3xl border border-orange-100 bg-gradient-to-r from-orange-50 to-white px-4 py-3 text-sm text-slate-700">
          Status: stale data. Data terakhir tercatat pada {formatTimestamp(data.timestamp)}.
        </div>
      ) : null}

      <div className="flex flex-wrap gap-3">
        <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-blue-700 ring-1 ring-blue-100">
          Source: {source}
        </span>
        <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-orange-700 ring-1 ring-orange-100">
          Refresh: {settings.refreshInterval} ms
        </span>
      </div>
    </div>
  );

  const dashboardContent = (
    <div className="space-y-6">
      {topBanners}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {summaryCards.map((card) => (
          <DashboardCard key={card.title} {...card} />
        ))}
      </div>

      <PowerChart history={history} isLoading={isLoading || isRefreshing} hasError={Boolean(error)} />

      <div className="grid gap-6 xl:grid-cols-2">
        <DeviceDetectionPanel
          device={data?.device_detected ?? "idle"}
          confidence={data?.confidence ?? 0}
          modelVersion={data?.model_version ?? "N/A"}
        />
        <EnergyCostCard
          energy={data?.energy ?? 0}
          power={data?.power ?? 0}
          tariff={settings.tariff}
        />
      </div>

      <EfficiencyRecommendation power={data?.power ?? 0} />
    </div>
  );

  const deviceContent = (
    <div className="space-y-6">
      {topBanners}
      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <DeviceDetectionPanel
          device={data?.device_detected ?? "idle"}
          confidence={data?.confidence ?? 0}
          modelVersion={data?.model_version ?? "N/A"}
        />
        <SectionCard
          title="Detection Metadata"
          description="Detail tambahan untuk membantu interpretasi hasil inferensi."
        >
          <div className="space-y-4">
            <div className="rounded-3xl border border-blue-100 bg-blue-50/70 p-4">
              <p className="text-sm font-medium text-slate-500">Timestamp</p>
              <p className="mt-2 text-base font-semibold text-slate-900">
                {data ? formatTimestamp(data.timestamp) : "Waiting for sensor data..."}
              </p>
            </div>
            <div className="rounded-3xl border border-orange-100 bg-orange-50/70 p-4">
              <p className="text-sm font-medium text-slate-500">Model Version</p>
              <p className="mt-2 text-base font-semibold text-slate-900">{data?.model_version ?? "N/A"}</p>
            </div>
            <div className="rounded-3xl border border-blue-100 bg-white p-4">
              <p className="text-sm font-medium text-slate-500">Inference Note</p>
              <p className="mt-2 text-base leading-7 text-slate-700">
                {data
                  ? "Model NILM multi-label memetakan pola konsumsi agregat menjadi satu atau lebih perangkat aktif."
                  : "Data inferensi akan tampil setelah sensor mengirim pembacaan terbaru."}
              </p>
            </div>
          </div>
        </SectionCard>
      </div>
    </div>
  );

  const analyticsContent = (
    <div className="space-y-6">
      {topBanners}
      <PowerChart history={history} isLoading={isLoading || isRefreshing} hasError={Boolean(error)} />
      <div className="grid gap-6 xl:grid-cols-2">
        <EnergyCostCard
          energy={data?.energy ?? 0}
          power={data?.power ?? 0}
          tariff={settings.tariff}
        />
        <EfficiencyRecommendation power={data?.power ?? 0} />
      </div>
    </div>
  );

  return (
    <div className="min-h-screen">
      <Sidebar
        activeSection={activeSection}
        onChange={setActiveSection}
        mobileOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
      />

      <div className="lg:pl-72">
        <Header
          statusLabel={statusLabel}
          lastUpdated={data?.timestamp ?? lastUpdated}
          isStale={isStale}
          onOpenMenu={() => setMobileOpen(true)}
        />

        <main className="space-y-6 px-4 py-6 sm:px-6 xl:px-8">
          {activeSection === "dashboard" ? dashboardContent : null}
          {activeSection === "device-detection" ? deviceContent : null}
          {activeSection === "energy-analytics" ? analyticsContent : null}
          {activeSection === "system-status" ? (
            <SystemStatus
              data={data}
              source={source}
              error={error}
              isStale={isStale}
              settings={settings}
              lastUpdated={lastUpdated}
            />
          ) : null}
          {activeSection === "settings" ? (
            <SettingsPanel settings={settings} onChange={handleSettingsChange} />
          ) : null}
        </main>
      </div>
    </div>
  );
}
