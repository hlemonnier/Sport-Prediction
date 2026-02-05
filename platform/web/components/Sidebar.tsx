"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const groups = [
  {
    label: "Accueil",
    items: [{ label: "Dashboard", href: "/" }],
  },
  {
    label: "F1",
    items: [
      { label: "Apercu", href: "/f1/preview" },
      { label: "Qualif", href: "/f1/qualifying" },
      { label: "Course", href: "/f1/race" },
      { label: "Analyse", href: "/f1/review" },
    ],
  },
  {
    label: "Football",
    items: [
      { label: "Apercu", href: "/football/preview" },
      { label: "Match", href: "/football/match" },
      { label: "Analyse", href: "/football/review" },
    ],
  },
  {
    label: "Lab",
    items: [
      { label: "Comparaison", href: "/compare" },
      { label: "Diagnostics", href: "/diagnostics" },
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-title">Sport Lab</div>
        <div className="brand-subtitle">Recherche Quant</div>
      </div>
      <div className="sidebar-groups">
        {groups.map((group) => (
          <div className="nav-group" key={group.label}>
            <div className="nav-group-title">{group.label}</div>
            <div className="nav-group-links">
              {group.items.map((item) => {
                const active = pathname === item.href || pathname.startsWith(item.href + "/");
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
        Local uniquement. Aucune donnee ne sort.
      </div>
    </aside>
  );
}
