"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { readUiPreferences, subscribeUiPreferences } from "@/lib/uiPreferences";

const dashboardItem = { label: "Dashboard", href: "/" };

const groups = [
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
  const storageKey = "sidebar:collapsed-groups";
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [autoExpandActiveGroup, setAutoExpandActiveGroup] = useState(true);
  const [autoCollapseNonActiveGroups, setAutoCollapseNonActiveGroups] = useState(false);
  const [rememberSidebarState, setRememberSidebarState] = useState(true);

  const activeGroup = useMemo(
    () =>
      groups.find((group) =>
        group.items.some((item) => pathname === item.href || pathname.startsWith(item.href + "/"))
      )?.label,
    [pathname]
  );

  useEffect(() => {
    const syncPreferences = () => {
      const prefs = readUiPreferences();
      setAutoExpandActiveGroup(prefs.autoExpandActiveGroup);
      setAutoCollapseNonActiveGroups(prefs.autoCollapseNonActiveGroups);
      setRememberSidebarState(prefs.rememberSidebarState);
    };
    syncPreferences();
    return subscribeUiPreferences(syncPreferences);
  }, []);

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
      <div className="sidebar-brand">
        <div className="brand-title">Sport Lab</div>
        <div className="brand-subtitle">Prediction Engine</div>
      </div>
      <div className="sidebar-groups">
        <div className="nav-dashboard">
          <Link
            href={dashboardItem.href}
            className={`nav-item ${dashboardActive ? "active" : ""}`}
          >
            {dashboardItem.label}
          </Link>
        </div>

        {groups.map((group) => (
          <div className="nav-group" key={group.label}>
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
      <div className="sidebar-bottom-actions">
        <Link
          href="/settings"
          className={`nav-item ${settingsActive ? "active" : ""}`}
        >
          Settings
        </Link>
      </div>
      <div className="sidebar-footer">
        Local mode &middot; No data leaves device
      </div>
    </aside>
  );
}
