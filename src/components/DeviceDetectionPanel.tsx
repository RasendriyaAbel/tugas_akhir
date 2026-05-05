import { BrainCircuit, TriangleAlert } from "lucide-react";

import { cn, formatDeviceLabel, formatNumber, getConfidenceLabel, getConfidenceTone } from "@/lib/utils";

interface DeviceDetectionPanelProps {
  device: string;
  confidence: number;
  modelVersion: string;
}

const toneStyles = {
  blue: {
    bar: "bg-blue-600",
    badge: "bg-blue-50 text-blue-700",
  },
  orange: {
    bar: "bg-orange-500",
    badge: "bg-orange-50 text-orange-700",
  },
  "soft-orange": {
    bar: "bg-orange-300",
    badge: "bg-orange-100 text-orange-800",
  },
};

export function DeviceDetectionPanel({
  device,
  confidence,
  modelVersion,
}: DeviceDetectionPanelProps) {
  const tone = getConfidenceTone(confidence) as keyof typeof toneStyles;
  const confidenceLabel = getConfidenceLabel(confidence);
  const isWarmingUp = confidence <= 0.1 && (device === "idle" || device === "uncertain");

  return (
    <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-5 shadow-sm shadow-blue-100/70">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Device Detection Panel</h2>
          <p className="text-sm text-slate-500">Prediksi perangkat hasil inferensi model deep learning.</p>
        </div>
        <div className="rounded-2xl bg-orange-50 p-3 text-orange-600 ring-1 ring-orange-100">
          <BrainCircuit className="h-5 w-5" />
        </div>
      </div>

      <div className="mt-6 rounded-3xl border border-blue-50 bg-gradient-to-r from-blue-50 to-orange-50 p-4">
        <p className="text-sm font-medium text-slate-600">Detected Device</p>
        <p className="mt-2 text-2xl font-semibold text-slate-900">{formatDeviceLabel(device)}</p>
        <p className="mt-1 text-sm text-slate-500">{isWarmingUp ? "Buffer sedang mengisi, prediksi akan makin stabil." : `Model version: ${modelVersion}`}</p>
      </div>

      <div className="mt-5">
        <div className="mb-2 flex items-center justify-between gap-3">
          <p className="text-sm font-medium text-slate-600">Confidence</p>
          <p className="text-sm font-semibold text-slate-900">{formatNumber(confidence, 1)}%</p>
        </div>
        <div className="h-3 overflow-hidden rounded-full bg-orange-100/60">
          <div
            className={cn("h-full rounded-full transition-all", toneStyles[tone].bar)}
            style={{ width: `${Math.min(confidence, 100)}%` }}
          />
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <span className={cn("rounded-full px-3 py-1 text-xs font-semibold", toneStyles[tone].badge)}>
          {confidenceLabel}
        </span>
        {confidence < 60 ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-orange-100 px-3 py-1 text-xs font-medium text-orange-800">
            <TriangleAlert className="h-3.5 w-3.5" />
            Confidence rendah, verifikasi hasil model.
          </span>
        ) : null}
      </div>
    </div>
  );
}
