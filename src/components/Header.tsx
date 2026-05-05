import { Clock3, Menu, Router } from "lucide-react";

import { cn, formatTimestamp } from "@/lib/utils";

interface HeaderProps {
  statusLabel: string;
  lastUpdated?: string;
  onOpenMenu: () => void;
  isStale: boolean;
}

export function Header({ statusLabel, lastUpdated, onOpenMenu, isStale }: HeaderProps) {
  const isOnline = statusLabel.toLowerCase().includes("online");

  return (
    <header className="sticky top-0 z-20 border-b border-blue-100/80 bg-white/90 backdrop-blur">
      <div className="flex flex-col gap-4 px-4 py-4 sm:px-6 xl:px-8">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <button
              type="button"
              className="rounded-2xl border border-blue-100 bg-white p-2 text-blue-700 shadow-sm shadow-blue-100/60 lg:hidden"
              onClick={onOpenMenu}
              aria-label="Buka menu"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-blue-700">NILM Energy Monitoring Dashboard</p>
              <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">
                Real-Time Smart Energy Monitoring using Deep Learning
              </h2>
            </div>
          </div>

          <div className="flex flex-col items-start gap-2 sm:items-end">
            <span
              className={cn(
                "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold",
                isOnline && !isStale ? "bg-blue-50 text-blue-700 ring-1 ring-blue-100" : "bg-orange-50 text-orange-700 ring-1 ring-orange-100",
              )}
            >
              <Router className="h-3.5 w-3.5" />
              {statusLabel}
            </span>
            <span className="inline-flex items-center gap-2 text-sm text-slate-500">
              <Clock3 className="h-4 w-4" />
              Last updated: {formatTimestamp(lastUpdated)}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
