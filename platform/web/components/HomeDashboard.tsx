"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

type DataStatus = {
  football: {
    teams: { path: string; format: string; exists: boolean };
    matches: { path: string; format: string; exists: boolean };
    fixtures: { path: string; format: string; exists: boolean };
  };
};

type Fixture = {
  matchId: string;
  date: string;
  season: string;
  league: string;
  homeTeamId: string;
  awayTeamId: string;
};

type FixtureResponse = {
  fixtures: Fixture[];
  warning?: string | null;
};

export default function HomeDashboard() {
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [fixtures, setFixtures] = useState<FixtureResponse | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/data-status`);
        if (res.ok) {
          setStatus((await res.json()) as DataStatus);
        }
      } catch {
        setStatus(null);
      }
    };
    const fetchFixtures = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/football/fixtures?limit=5`);
        if (res.ok) {
          setFixtures((await res.json()) as FixtureResponse);
        }
      } catch {
        setFixtures(null);
      }
    };
    const fetchRuns = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/runs`);
        if (res.ok) {
          setRuns((await res.json()) as RunSummary[]);
        }
      } catch {
        setRuns([]);
      }
    };
    fetchStatus();
    fetchFixtures();
    fetchRuns();
  }, []);

  const footballReady = status?.football.matches.exists && status?.football.fixtures.exists;

  return (
    <div className="stack">
      <div>
        <h1 className="section-title">Research Cockpit</h1>
        <p className="section-subtitle">
          Ton tableau de bord personnel pour préparer, lancer et revoir les prédictions.
        </p>
      </div>

      <div className="grid-three">
        <div className="card stat">
          <span className="stat-label">Football data</span>
          <span className="stat-value">{footballReady ? "Ready" : "Missing"}</span>
          <span className="section-subtitle" style={{ marginTop: 8 }}>
            {status
              ? `Teams: ${status.football.teams.exists ? "OK" : "Missing"} · Matches: ${
                  status.football.matches.exists ? "OK" : "Missing"
                } · Fixtures: ${status.football.fixtures.exists ? "OK" : "Missing"}`
              : "Backend offline"}
          </span>
        </div>
        <div className="card stat">
          <span className="stat-label">Latest runs</span>
          <span className="stat-value">{runs.length}</span>
          <span className="section-subtitle" style={{ marginTop: 8 }}>
            {runs.length > 0 ? "Recent runs available" : "No runs yet"}
          </span>
        </div>
        <div className="card stat">
          <span className="stat-label">Next context</span>
          <span className="stat-value">Manual</span>
          <span className="section-subtitle" style={{ marginTop: 8 }}>
            Use the top bar to set F1/Football context.
          </span>
        </div>
      </div>

      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Upcoming Football Fixtures</h2>
          <p className="module-subtitle">
            {fixtures?.warning ?? "Loaded from data/football/fixtures.*"}
          </p>
          {fixtures && fixtures.fixtures.length > 0 ? (
            <table className="table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>League</th>
                  <th>Home</th>
                  <th>Away</th>
                </tr>
              </thead>
              <tbody>
                {fixtures.fixtures.map((fixture) => (
                  <tr key={fixture.matchId}>
                    <td>{fixture.date}</td>
                    <td>{fixture.league}</td>
                    <td>{fixture.homeTeamId}</td>
                    <td>{fixture.awayTeamId}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="section-subtitle">No fixtures loaded yet.</p>
          )}
        </div>
        <div className="card">
          <h2 className="module-title">Next F1 Event</h2>
          <p className="module-subtitle">Use the context selector for season/round.</p>
          <div className="stack">
            <span className="pill">No schedule source connected yet.</span>
            <span className="section-subtitle">
              Add data or connect FastF1/OpenF1 in the prediction pipeline.
            </span>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="module-title">Recent Runs</h2>
        {runs.length === 0 ? (
          <p className="section-subtitle">Aucune run enregistrée pour le moment.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Sport</th>
                <th>Project</th>
                <th>Status</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {runs.slice(0, 6).map((run) => (
                <tr key={run.id}>
                  <td>{run.id.slice(0, 8)}</td>
                  <td>{run.sport}</td>
                  <td>{run.project}</td>
                  <td>{run.status}</td>
                  <td>{new Date(run.createdAt).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
