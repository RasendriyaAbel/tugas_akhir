import {
  Activity,
  BarChart3,
  Cpu,
  LayoutDashboard,
  Settings,
  X,
} from "lucide-react";

import { cn } from "@/lib/utils";
import type { NavigationSection } from "@/types/nilm";

interface SidebarProps {
  activeSection: NavigationSection;
  onChange: (section: NavigationSection) => void;
  mobileOpen: boolean;
  onClose: () => void;
}

const navItems: Array<{ key: NavigationSection; label: string; icon: typeof LayoutDashboard }> = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "device-detection", label: "Device Detection", icon: Cpu },
  { key: "energy-analytics", label: "Energy Analytics", icon: BarChart3 },
  { key: "system-status", label: "System Status", icon: Activity },
  { key: "settings", label: "Settings", icon: Settings },
];

export function Sidebar({ activeSection, onChange, mobileOpen, onClose }: SidebarProps) {
  return (
    <>
      <div
        className={cn(
          "fixed inset-0 z-30 bg-blue-950/30 transition lg:hidden",
          mobileOpen ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={onClose}
      />
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 w-72 border-r border-blue-100 bg-gradient-to-b from-white via-blue-50/40 to-orange-50/40 px-5 py-6 shadow-xl shadow-blue-100/70 transition lg:translate-x-0 lg:shadow-none",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-blue-700">NILM Dashboard</p>
            <h1 className="mt-2 text-xl font-semibold text-slate-900">Energy Monitoring</h1>
            <p className="mt-2 text-sm leading-6 text-slate-500">Clean dashboard untuk monitoring energi dan inferensi NILM.</p>
          </div>
          <button
            type="button"
            className="rounded-2xl border border-blue-100 bg-white p-2 text-blue-700 lg:hidden"
            onClick={onClose}
            aria-label="Tutup menu"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <nav className="mt-8 space-y-2">
          {navItems.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              type="button"
              onClick={() => {
                onChange(key);
                onClose();
              }}
              className={cn(
                "flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm font-medium transition",
                activeSection === key
                  ? "bg-gradient-to-r from-blue-600 to-orange-500 text-white shadow-lg shadow-blue-200/60"
                  : "text-slate-700 hover:bg-white hover:text-blue-700 hover:shadow-sm hover:shadow-blue-100/60",
              )}
            >
              <Icon className="h-4 w-4" />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="mt-8 rounded-3xl border border-orange-100 bg-orange-50 p-4">
          <p className="text-sm font-semibold text-orange-800">ThingsBoard Telemetry</p>
          <p className="mt-2 text-sm leading-6 text-orange-700">
            Backend membaca telemetry sensor lalu meneruskan sample ke model NILM untuk menghasilkan label perangkat.
          </p>
        </div>
      </aside>
    </>
  );
}
