"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const groups = [
  {
    label: "Home",
    items: [{ label: "Dashboard", href: "/" }],
  },
  {
    label: "F1",
    items: [
      { label: "Preview", href: "/f1/preview" },
      { label: "Qualifying", href: "/f1/qualifying" },
      { label: "Race", href: "/f1/race" },
      { label: "Review", href: "/f1/review" },
    ],
  },
  {
    label: "Football",
    items: [
      { label: "Preview", href: "/football/preview" },
      { label: "Match", href: "/football/match" },
      { label: "Review", href: "/football/review" },
    ],
  },
  {
    label: "Lab",
    items: [
      { label: "Runs", href: "/runs" },
      { label: "Sweeps", href: "/sweeps" },
      { label: "Compare", href: "/compare" },
      { label: "Diagnostics", href: "/diagnostics" },
      { label: "Research", href: "/research" },
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-title">Sport Lab</div>
        <div className="brand-subtitle">Prediction Engine</div>
      </div>
      <div className="sidebar-groups">
        {groups.map((group) => (
          <div className="nav-group" key={group.label}>
            <div className="nav-group-title">{group.label}</div>
            <div className="nav-group-links">
              {group.items.map((item) => {
                const active =
                  item.href === "/"
                    ? pathname === "/"
                    : pathname === item.href || pathname.startsWith(item.href + "/");
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`nav-item ${active ? "active" : ""}`}
                  >
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="sidebar-footer">
        Local mode &middot; No data leaves device
      </div>
    </aside>
  );
}
