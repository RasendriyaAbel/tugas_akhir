import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { PowerHistoryPoint } from "@/types/nilm";
import { formatChartTime, formatNumber } from "@/lib/utils";

interface PowerChartProps {
  history: PowerHistoryPoint[];
  isLoading: boolean;
  hasError: boolean;
}

export function PowerChart({ history, isLoading, hasError }: PowerChartProps) {
  const hasData = history.length > 0;

  return (
    <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-5 shadow-sm shadow-blue-100/70">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Real-Time Power Chart</h2>
          <p className="text-sm text-slate-500">Grafik daya aktif yang diperbarui setiap beberapa detik.</p>
        </div>
        <div className="rounded-full bg-orange-50 px-3 py-1 text-xs font-semibold text-orange-700">
          Recharts
        </div>
      </div>

      {!hasData ? (
        <div className="mt-6 flex h-80 items-center justify-center rounded-3xl border border-dashed border-blue-100 bg-gradient-to-br from-blue-50/70 to-orange-50/70 text-sm text-slate-500">
          {hasError
            ? "Blynk connection error"
            : isLoading
              ? "Loading chart data..."
              : "Waiting for sensor data..."}
        </div>
      ) : (
        <div className="mt-6 h-80">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
              <CartesianGrid stroke="#dbeafe" strokeDasharray="4 4" vertical={false} />
              <XAxis
                dataKey="timestamp"
                tickFormatter={formatChartTime}
                tick={{ fill: "#64748b", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tickFormatter={(value: number) => `${formatNumber(value, 0)} W`}
                tick={{ fill: "#64748b", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                width={72}
              />
              <Tooltip
                formatter={(value) => [`${formatNumber(Number(value ?? 0), 1)} W`, "Active Power"]}
                labelFormatter={(label) => `Timestamp: ${label}`}
                contentStyle={{
                  borderRadius: 16,
                  border: "1px solid #dbeafe",
                  boxShadow: "0 10px 30px rgba(37, 99, 235, 0.12)",
                }}
              />
              <Line
                type="monotone"
                dataKey="power"
                stroke="#f97316"
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 5, fill: "#2563eb" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
