"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import {
  readUiPreferences,
  subscribeUiPreferences,
  type SportModule,
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
  session: "Race",
};

const defaultFootball: FootballContext = {
  league: "EPL",
  season: "2026",
  match: "Next",
};

const sportModules: SportModule[] = ["F1", "Football"];

function sportFromPath(pathname: string): SportModule | null {
  if (pathname.startsWith("/f1")) return "F1";
  if (pathname.startsWith("/football")) return "Football";
  return null;
}

function f1SessionFromPath(pathname: string): F1Context["session"] | null {
  if (pathname.startsWith("/f1/research/preview") || pathname.startsWith("/f1/preview")) {
    return "Preview";
  }
  if (pathname.startsWith("/f1/research/qualifying") || pathname.startsWith("/f1/qualifying")) {
    return "Qualifying";
  }
  if (pathname.startsWith("/f1/research/review") || pathname.startsWith("/f1/review")) {
    return "Review";
  }
  if (pathname.startsWith("/f1/insights") || pathname.startsWith("/f1/race")) {
    return "Race";
  }
  return null;
}

function equivalentSportHref(sport: SportModule, pathname: string): string {
  const page = pathname.split("/")[2] ?? "";
  if (sport === "F1") {
    if (page === "preview") return "/f1/research/preview";
    if (page === "review") return "/f1/research/review";
    return "/f1";
  }
  if (page === "preview") return "/football/preview";
  if (page === "review") return "/football/review";
  return "/football/match";
}

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
  const router = useRouter();
  const routeSport = sportFromPath(pathname);
  const routeF1Session = f1SessionFromPath(pathname);

  const [f1Context, setF1Context] = useState<F1Context>({
    ...defaultF1,
    session: routeF1Session ?? defaultF1.session,
  });
  const [footballContext, setFootballContext] = useState<FootballContext>(defaultFootball);
  const [selectedSport, setSelectedSport] = useState<SportModule>(routeSport ?? "F1");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    const storedF1 = readLocal("context:f1", defaultF1);
    const normalizedF1 = F1_SESSIONS.includes(storedF1.session as typeof F1_SESSIONS[number])
      ? storedF1
      : { ...storedF1, session: defaultF1.session };
    setF1Context({ ...normalizedF1, session: routeF1Session ?? normalizedF1.session });
    setFootballContext(readLocal("context:football", defaultFootball));
  }, [routeF1Session]);

  useEffect(() => {
    const syncPreferences = () => {
      const prefs = readUiPreferences();
      setSidebarCollapsed(prefs.sidebarCollapsed);
      setSelectedSport(routeSport ?? prefs.defaultSportModule);
      if (routeSport && prefs.defaultSportModule !== routeSport) {
        updateUiPreferences({ defaultSportModule: routeSport });
      }
    };
    syncPreferences();
    return subscribeUiPreferences(syncPreferences);
  }, [routeSport]);

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

  const selectSport = (sport: SportModule) => {
    setSelectedSport(sport);
    updateUiPreferences({ defaultSportModule: sport });
    if (routeSport && routeSport !== sport) {
      router.push(equivalentSportHref(sport, pathname));
    }
  };

  return (
    <div className="topbar">
      <div className="topbar-brand" aria-label="Sport Lab Prediction Engine">
        <span className="topbar-brand-dot" />
        <span className="topbar-brand-text">Sport Lab</span>
        <span className="topbar-brand-sub">{selectedSport} Module</span>
      </div>
      <div className="sport-switcher" role="tablist" aria-label="Sport module">
        {sportModules.map((sport) => (
          <button
            key={sport}
            type="button"
            role="tab"
            aria-selected={selectedSport === sport}
            className={`sport-switcher-item ${selectedSport === sport ? "active" : ""}`}
            onClick={() => selectSport(sport)}
          >
            {sport}
          </button>
        ))}
      </div>
      <div className="topbar-title">{selectedSport} Context</div>
      {selectedSport === "F1" ? (
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
      ) : (
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
