"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import {
  readUiPreferences,
  subscribeUiPreferences,
  updateUiPreferences,
} from "@/lib/uiPreferences";

type F1Context = {
  season: string;
  round: string;
  session: "Preview" | "Qualifying" | "Race" | "Review";
};

type FootballContext = {
  league: string;
  season: string;
  match: string;
};

const F1_SESSIONS = ["Preview", "Qualifying", "Race", "Review"] as const;

const defaultF1: F1Context = {
  season: "2026",
  round: "1",
  session: "Preview",
};

const defaultFootball: FootballContext = {
  league: "EPL",
  season: "2026",
  match: "Next",
};

function readLocal<T>(key: string, fallback: T): T {
  if (typeof window === "undefined") {
    return fallback;
  }
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    return { ...fallback, ...(JSON.parse(raw) as Partial<T>) } as T;
  } catch {
    return fallback;
  }
}

export default function TopBar() {
  const pathname = usePathname();
  const isF1 = pathname.startsWith("/f1");
  const isFootball = pathname.startsWith("/football");

  const [f1Context, setF1Context] = useState<F1Context>(defaultF1);
  const [footballContext, setFootballContext] = useState<FootballContext>(defaultFootball);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    const storedF1 = readLocal("context:f1", defaultF1);
    const normalizedF1 = F1_SESSIONS.includes(storedF1.session as typeof F1_SESSIONS[number])
      ? storedF1
      : { ...storedF1, session: defaultF1.session };
    setF1Context(normalizedF1);
    setFootballContext(readLocal("context:football", defaultFootball));
  }, []);

  useEffect(() => {
    const syncPreferences = () => {
      const prefs = readUiPreferences();
      setSidebarCollapsed(prefs.sidebarCollapsed);
    };
    syncPreferences();
    return subscribeUiPreferences(syncPreferences);
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("context:f1", JSON.stringify(f1Context));
    }
  }, [f1Context]);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("context:football", JSON.stringify(footballContext));
    }
  }, [footballContext]);

  return (
    <div className="topbar">
      <div className="topbar-title">Sport Lab Prediction Engine</div>
      {isF1 ? (
        <div className="context-grid">
          <div className="context-field">
            <label>Season</label>
            <input
              value={f1Context.season}
              onChange={(event) =>
                setF1Context((prev) => ({ ...prev, season: event.target.value }))
              }
            />
          </div>
          <div className="context-field">
            <label>Round</label>
            <input
              value={f1Context.round}
              onChange={(event) =>
                setF1Context((prev) => ({ ...prev, round: event.target.value }))
              }
            />
          </div>
          <div className="context-field">
            <label>Session</label>
            <select
              value={f1Context.session}
              onChange={(event) =>
                setF1Context((prev) => ({
                  ...prev,
                  session: event.target.value as F1Context["session"],
                }))
              }
            >
              {F1_SESSIONS.map((session) => (
                <option key={session} value={session}>
                  {session}
                </option>
              ))}
            </select>
          </div>
        </div>
      ) : isFootball ? (
        <div className="context-grid">
          <div className="context-field">
            <label>League</label>
            <input
              value={footballContext.league}
              onChange={(event) =>
                setFootballContext((prev) => ({ ...prev, league: event.target.value }))
              }
            />
          </div>
          <div className="context-field">
            <label>Season</label>
            <input
              value={footballContext.season}
              onChange={(event) =>
                setFootballContext((prev) => ({ ...prev, season: event.target.value }))
              }
            />
          </div>
          <div className="context-field">
            <label>Match</label>
            <input
              value={footballContext.match}
              onChange={(event) =>
                setFootballContext((prev) => ({ ...prev, match: event.target.value }))
              }
            />
          </div>
        </div>
      ) : (
        <div className="context-hint">Select a sport module to set context</div>
      )}
      <div className="topbar-actions">
        <button
          type="button"
          className="button secondary button-sm topbar-sidebar-toggle"
          onClick={() => updateUiPreferences({ sidebarCollapsed: !sidebarCollapsed })}
        >
          {sidebarCollapsed ? "Show sidebar" : "Hide sidebar"}
        </button>
        <span className="chip">
          <span className="chip-led green" />
          Local
        </span>
      </div>
    </div>
  );
}
