import { ReceiptText } from "lucide-react";

import { formatCurrency, formatNumber } from "@/lib/utils";

interface EnergyCostCardProps {
  energy: number;
  power: number;
  tariff: number;
}

export function EnergyCostCard({ energy, power, tariff }: EnergyCostCardProps) {
  const currentCost = energy * tariff;
  const monthlyCost = (power / 1000) * 8 * 30 * tariff;

  return (
    <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-5 shadow-sm shadow-blue-100/70">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Energy Cost Estimation</h2>
          <p className="text-sm text-slate-500">Estimasi biaya listrik berbasis energi kumulatif dan daya saat ini.</p>
        </div>
        <div className="rounded-2xl bg-orange-50 p-3 text-orange-600 ring-1 ring-orange-100">
          <ReceiptText className="h-5 w-5" />
        </div>
      </div>

      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        <div className="rounded-3xl border border-blue-100 bg-blue-50/70 p-4">
          <p className="text-sm font-medium text-slate-500">Biaya saat ini</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{formatCurrency(currentCost)}</p>
          <p className="mt-2 text-sm text-slate-500">
            Berdasarkan energi {formatNumber(energy, 2)} kWh dan tarif {formatCurrency(tariff)}/kWh
          </p>
        </div>
        <div className="rounded-3xl border border-orange-100 bg-orange-50/80 p-4">
          <p className="text-sm font-medium text-slate-500">Estimasi bulanan</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{formatCurrency(monthlyCost)}</p>
          <p className="mt-2 text-sm text-slate-500">Asumsi pemakaian 8 jam/hari selama 30 hari.</p>
        </div>
      </div>
    </div>
  );
}
