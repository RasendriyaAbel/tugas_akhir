import { Lightbulb } from "lucide-react";

import { getEfficiencyRecommendation } from "@/lib/utils";

interface EfficiencyRecommendationProps {
  power: number;
}

export function EfficiencyRecommendation({ power }: EfficiencyRecommendationProps) {
  const recommendation = getEfficiencyRecommendation(power);

  return (
    <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-5 shadow-sm shadow-blue-100/70">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Efficiency Recommendation</h2>
          <p className="text-sm text-slate-500">Insight otomatis berdasarkan konsumsi daya saat ini.</p>
        </div>
        <div className="rounded-2xl bg-orange-50 p-3 text-orange-600 ring-1 ring-orange-100">
          <Lightbulb className="h-5 w-5" />
        </div>
      </div>

      <div className="mt-6 rounded-3xl border border-orange-100 bg-gradient-to-r from-blue-50 to-orange-50 p-4">
        <p className="text-sm font-medium text-blue-800">Rekomendasi sistem</p>
        <p className="mt-2 text-base leading-7 text-slate-700">{recommendation}</p>
      </div>
    </div>
  );
}
