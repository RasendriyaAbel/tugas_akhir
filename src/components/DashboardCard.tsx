import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

type Accent = "blue" | "emerald" | "violet" | "amber";

interface DashboardCardProps {
  title: string;
  value: string;
  unit?: string;
  description: string;
  icon: LucideIcon;
  accent?: Accent;
}

const accentStyles: Record<Accent, string> = {
  blue: "bg-blue-50 text-blue-700 ring-blue-100",
  emerald: "bg-orange-50 text-orange-700 ring-orange-100",
  violet: "bg-blue-100 text-blue-800 ring-blue-200",
  amber: "bg-orange-100 text-orange-800 ring-orange-200",
};

export function DashboardCard({
  title,
  value,
  unit,
  description,
  icon: Icon,
  accent = "blue",
}: DashboardCardProps) {
  return (
    <div className="rounded-3xl border border-blue-100/80 bg-white/95 p-5 shadow-sm shadow-blue-100/70 backdrop-blur">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-slate-500">{title}</p>
          <div className="mt-3 flex items-end gap-2">
            <p className="text-3xl font-semibold tracking-tight text-slate-900">{value}</p>
            {unit ? <span className="pb-1 text-sm font-medium text-orange-500">{unit}</span> : null}
          </div>
        </div>
        <div className={cn("rounded-2xl p-3 ring-1", accentStyles[accent])}>
          <Icon className="h-5 w-5" />
        </div>
      </div>
      <p className="mt-4 text-sm text-slate-500">{description}</p>
    </div>
  );
}
