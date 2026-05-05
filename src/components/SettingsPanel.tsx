import { Save, Wifi } from "lucide-react";

import type { DashboardSettings } from "@/types/nilm";

interface SettingsPanelProps {
  settings: DashboardSettings;
  onChange: <K extends keyof DashboardSettings>(key: K, value: DashboardSettings[K]) => void;
}

const telemetryKeys = [
  "tegangan -> voltage",
  "arus -> current",
  "daya -> power",
  "kwh -> energy",
  "frekuensi -> frequency",
  "power_factor -> power_factor",
  "device_detected",
  "confidence",
  "model_version",
  "timestamp",
];

export function SettingsPanel({ settings, onChange }: SettingsPanelProps) {
  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-slate-900">Settings Page</h2>
        <p className="mt-1 text-sm text-slate-500">Atur tarif listrik dan interval refresh dashboard.</p>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-6 shadow-sm shadow-blue-100/70">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Dashboard Preferences</h3>
              <p className="mt-1 text-sm text-slate-500">Semua pengaturan disimpan di browser melalui localStorage.</p>
            </div>
            <div className="rounded-2xl bg-orange-50 p-3 text-orange-600 ring-1 ring-orange-100">
              <Save className="h-5 w-5" />
            </div>
          </div>

          <div className="mt-6 grid gap-5 md:grid-cols-2">
            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-600">Tarif listrik per kWh</span>
              <input
                type="number"
                min="0"
                step="0.1"
                value={settings.tariff}
                onChange={(event) => onChange("tariff", Number(event.target.value))}
                className="w-full rounded-2xl border border-blue-100 bg-blue-50/40 px-4 py-3 text-slate-900 outline-none ring-0 transition focus:border-orange-300"
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-600">Refresh interval (ms)</span>
              <input
                type="number"
                min="1000"
                step="500"
                value={settings.refreshInterval}
                onChange={(event) => onChange("refreshInterval", Number(event.target.value))}
                className="w-full rounded-2xl border border-blue-100 bg-blue-50/40 px-4 py-3 text-slate-900 outline-none ring-0 transition focus:border-orange-300"
              />
            </label>
          </div>

          <div className="mt-6 rounded-3xl border border-blue-100 bg-blue-50/50 px-5 py-4 text-sm text-slate-600">
            Dashboard menarik telemetry live dari ThingsBoard, lalu backend mengirim sample sensor ke model NILM untuk menghasilkan label perangkat.
          </div>
        </div>

        <div className="space-y-6">
          <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-6 shadow-sm shadow-blue-100/70">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">Telemetry Keys</h3>
                <p className="mt-1 text-sm text-slate-500">Referensi key sensor dan output inferensi yang dipakai backend ThingsBoard dan dashboard.</p>
              </div>
              <div className="rounded-2xl bg-blue-50 p-3 text-blue-600 ring-1 ring-blue-100">
                <Wifi className="h-5 w-5" />
              </div>
            </div>

            <div className="mt-5 space-y-3">
              {telemetryKeys.map((item) => (
                <div key={item} className="rounded-2xl border border-orange-100 bg-orange-50/70 px-4 py-3 text-sm font-medium text-slate-700">
                  {item}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-6 shadow-sm shadow-blue-100/70">
            <h3 className="text-lg font-semibold text-slate-900">Environment Variable</h3>
            <div className="mt-4 space-y-3 text-sm text-slate-600">
              <p className="rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3">NILM_DATA_SOURCE=thingsboard</p>
              <p className="rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3">THINGSBOARD_ACCESS_TOKEN=token-device-esp32</p>
              <p className="rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3">THINGSBOARD_BASE_URL=https://your-thingsboard-host</p>
              <p className="rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3">THINGSBOARD_JWT_TOKEN=jwt-token-thingsboard</p>
              <p className="rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3">THINGSBOARD_DEVICE_ID=opsional-device-id</p>
              <p className="rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3">THINGSBOARD_USERNAME=opsional-fallback-login</p>
              <p className="rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3">THINGSBOARD_PASSWORD=opsional-fallback-login</p>
              <p className="rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3">ML_SERVICE_URL=http://127.0.0.1:5001</p>
              <p className="rounded-2xl border border-blue-100 bg-blue-50/50 px-4 py-3">NEXT_PUBLIC_REFRESH_INTERVAL=3000</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
