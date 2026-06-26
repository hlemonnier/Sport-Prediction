"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { readUiPreferences, subscribeUiPreferences, type SportModule } from "@/lib/uiPreferences";

const dashboardItem = { label: "Dashboard", href: "/" };

type NavItem = {
  label: string;
  href: string;
};

type NavGroup = {
  label: string;
  sport: SportModule | null;
  hideWhenSport?: SportModule[];
  items: NavItem[];
};

const labItems: NavItem[] = [
  { label: "Runs", href: "/runs" },
  { label: "Sweeps", href: "/sweeps" },
  { label: "Compare", href: "/compare" },
  { label: "Diagnostics", href: "/diagnostics" },
  { label: "Research Library", href: "/research" },
];

const groups: NavGroup[] = [
  {
    label: "Insights",
    sport: "F1" as const,
    items: [
      { label: "Home", href: "/f1/insights" },
      { label: "Live Dashboard", href: "/f1/insights/live" },
      { label: "Session Analysis", href: "/f1/insights/session-analysis" },
      { label: "Engineer", href: "/f1/insights/engineer" },
      { label: "Season", href: "/f1/insights/season" },
      { label: "Standings", href: "/f1/insights/standings" },
      { label: "Driver Ranking", href: "/f1/insights/driver-ranking" },
      { label: "Power Units", href: "/f1/insights/power-units" },
    ],
  },
  {
    label: "Research Lab",
    sport: "F1" as const,
    items: [
      { label: "Overview", href: "/f1/research" },
      { label: "Preview", href: "/f1/research/preview" },
      { label: "Qualifying", href: "/f1/research/qualifying" },
      { label: "Review", href: "/f1/research/review" },
      ...labItems,
    ],
  },
  {
    label: "Football",
    sport: "Football" as const,
    items: [
      { label: "Preview", href: "/football/preview" },
      { label: "Match", href: "/football/match" },
      { label: "Review", href: "/football/review" },
    ],
  },
  {
    label: "Lab",
    sport: null,
    hideWhenSport: ["F1"],
    items: labItems,
  },
];

function sportFromPath(pathname: string): SportModule | null {
  if (pathname.startsWith("/f1")) return "F1";
  if (pathname.startsWith("/football")) return "Football";
  return null;
}

function isSectionIndex(href: string): boolean {
  return href === "/f1/research" || href === "/f1/insights" || href === "/football";
}

function isActiveHref(pathname: string, href: string): boolean {
  if (href === "/" || isSectionIndex(href)) {
    return pathname === href;
  }
  return pathname === href || pathname.startsWith(href + "/");
}

export default function Sidebar() {
  const pathname = usePathname();
  const storageKey = "sidebar:collapsed-groups";
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [autoExpandActiveGroup, setAutoExpandActiveGroup] = useState(true);
  const [autoCollapseNonActiveGroups, setAutoCollapseNonActiveGroups] = useState(false);
  const [rememberSidebarState, setRememberSidebarState] = useState(true);
  const [selectedSport, setSelectedSport] = useState<SportModule>("Football");
  const routeSport = sportFromPath(pathname);
  const activeSport = routeSport ?? selectedSport;

  const visibleGroups = useMemo(
    () =>
      groups.filter((group) => {
        if (group.sport !== null) return group.sport === activeSport;
        return !group.hideWhenSport?.includes(activeSport);
      }),
    [activeSport]
  );
  const activeGroup = useMemo(
    () =>
      visibleGroups.find((group) =>
        group.items.some((item) => isActiveHref(pathname, item.href))
      )?.label,
    [pathname, visibleGroups]
  );

  useEffect(() => {
    const syncPreferences = () => {
      const prefs = readUiPreferences();
      setAutoExpandActiveGroup(prefs.autoExpandActiveGroup);
      setAutoCollapseNonActiveGroups(prefs.autoCollapseNonActiveGroups);
      setRememberSidebarState(prefs.rememberSidebarState);
      setSelectedSport(routeSport ?? prefs.defaultSportModule);
    };
    syncPreferences();
    return subscribeUiPreferences(syncPreferences);
  }, [routeSport]);

  useEffect(() => {
    if (typeof window === "undefined" || !rememberSidebarState) return;
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (!raw) return;
      const parsed = JSON.parse(raw) as Record<string, boolean>;
      setCollapsed(parsed);
    } catch {
      setCollapsed({});
    }
  }, [rememberSidebarState]);

  useEffect(() => {
    if (!rememberSidebarState) {
      setCollapsed({});
    }
  }, [rememberSidebarState]);

  useEffect(() => {
    if (typeof window === "undefined" || !rememberSidebarState) return;
    window.localStorage.setItem(storageKey, JSON.stringify(collapsed));
  }, [collapsed, rememberSidebarState]);

  useEffect(() => {
    if (!autoExpandActiveGroup) return;
    if (!activeGroup) return;
    setCollapsed((prev) => {
      const next = { ...prev };
      if (autoCollapseNonActiveGroups) {
        groups.forEach((group) => {
          next[group.label] = group.label !== activeGroup;
        });
      } else {
        next[activeGroup] = false;
      }
      return next;
    });
  }, [activeGroup, autoExpandActiveGroup, autoCollapseNonActiveGroups]);

  const toggleGroup = (label: string) => {
    setCollapsed((prev) => {
      const next = { ...prev, [label]: !prev[label] };
      if (autoCollapseNonActiveGroups && next[label] === false) {
        groups.forEach((group) => {
          if (group.label !== label) {
            next[group.label] = true;
          }
        });
      }
      return next;
    });
  };

  const dashboardActive = pathname === "/";
  const settingsActive = pathname === "/settings" || pathname.startsWith("/settings/");

  return (
    <aside className="sidebar">
      <div className="sidebar-groups">
        <div className="nav-dashboard">
          <Link
            href={dashboardItem.href}
            className={`nav-item ${dashboardActive ? "active" : ""}`}
          >
            {dashboardItem.label}
          </Link>
        </div>

        {visibleGroups.map((group) => (
          <div
            className={`nav-group ${group.sport === activeSport ? "current-sport" : ""}`}
            key={group.label}
          >
            <button
              type="button"
              className="nav-group-toggle"
              onClick={() => toggleGroup(group.label)}
              aria-expanded={!collapsed[group.label]}
            >
              <span className="nav-group-title">{group.label}</span>
              <span className={`nav-group-chevron ${collapsed[group.label] ? "collapsed" : ""}`}>
                ▾
              </span>
            </button>
            <div className={`nav-group-links ${collapsed[group.label] ? "collapsed" : ""}`}>
              {group.items.map((item) => {
                const active = isActiveHref(pathname, item.href);
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
      <div className="sidebar-bottom-actions">
        <Link
          href="/settings"
          className={`nav-item ${settingsActive ? "active" : ""}`}
        >
          Settings
        </Link>
      </div>
      <div className="sidebar-footer">
        Runtime local &middot; Public F1 APIs enabled
      </div>
    </aside>
  );
}
