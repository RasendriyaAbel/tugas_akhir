import { Activity, Cloud, Cpu, RefreshCcw, Server, ShieldCheck } from "lucide-react";

import { formatRelativeTime, formatTimestamp } from "@/lib/utils";
import type { DashboardSettings, DataSource, NilmData } from "@/types/nilm";

interface SystemStatusProps {
  data: NilmData | null;
  source: DataSource;
  error: string | null;
  isStale: boolean;
  settings: DashboardSettings;
  lastUpdated?: string;
}

function StatusItem({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: typeof Cloud;
}) {
  return (
    <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-5 shadow-sm shadow-blue-100/70">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-slate-500">{label}</p>
          <p className="mt-2 text-xl font-semibold text-slate-900">{value}</p>
          <p className="mt-2 text-sm text-slate-500">{detail}</p>
        </div>
        <div className="rounded-2xl bg-gradient-to-br from-blue-50 to-orange-50 p-3 text-blue-700 ring-1 ring-blue-100">
          <Icon className="h-5 w-5" />
        </div>
      </div>
    </div>
  );
}

export function SystemStatus({
  data,
  source,
  error,
  isStale,
  settings,
  lastUpdated,
}: SystemStatusProps) {
  const telemetryStatus = error ? "Error" : "Connected";
  const sensorStatus = !data ? "Waiting" : data.voltage > 0 ? "Active" : "No Reading";
  const esp32Status = isStale ? "Stale Data" : "Online";

  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-900">System Status Page</h2>
        <p className="mt-1 text-sm text-slate-500">Ringkasan konektivitas platform, sensor, dan status refresh data.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <StatusItem
          label="Telemetry Source Status"
          value={telemetryStatus}
          detail={error ?? `Koneksi backend ke sumber telemetry (${source}) berjalan normal.`}
          icon={Cloud}
        />
        <StatusItem
          label="Sensor Status"
          value={sensorStatus}
          detail={data ? `Tegangan terakhir ${data.voltage} V.` : "Waiting for sensor data..."}
          icon={Activity}
        />
        <StatusItem
          label="ESP32 Status"
          value={esp32Status}
          detail={isStale ? "Data belum diperbarui sesuai interval refresh." : "ESP32 mengirim data secara berkala."}
          icon={Cpu}
        />
        <StatusItem
          label="Last Data Update"
          value={formatRelativeTime(data?.timestamp ?? lastUpdated)}
          detail={formatTimestamp(data?.timestamp ?? lastUpdated)}
          icon={RefreshCcw}
        />
        <StatusItem
          label="Model Version"
          value={data?.model_version ?? "N/A"}
          detail="Versi model NILM yang aktif pada proses inferensi."
          icon={ShieldCheck}
        />
        <StatusItem
          label="Data Refresh Interval"
          value={`${settings.refreshInterval / 1000} detik`}
          detail={`Source saat ini: ${source}.`}
          icon={Server}
        />
      </div>
    </section>
  );
}
