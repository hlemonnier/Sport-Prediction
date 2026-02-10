"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

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

export default function FootballPreview() {
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [fixtures, setFixtures] = useState<FixtureResponse | null>(null);

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
        const res = await fetch(`${API_BASE}/api/football/fixtures?limit=8`);
        if (res.ok) {
          setFixtures((await res.json()) as FixtureResponse);
        }
      } catch {
        setFixtures(null);
      }
    };
    fetchStatus();
    fetchFixtures();
  }, []);

  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Football Preview</h1>
        <p className="page-status">Upcoming fixtures and pre-match signals</p>
      </div>

      {/* Data Health strip */}
      <div className="status-strip">
        <div className="status-item">
          <span className={`status-dot ${status?.football.teams.exists ? "ok" : "miss"}`} />
          Teams
        </div>
        <div className="status-item">
          <span className={`status-dot ${status?.football.matches.exists ? "ok" : "miss"}`} />
          Matches
        </div>
        <div className="status-item">
          <span className={`status-dot ${status?.football.fixtures.exists ? "ok" : "miss"}`} />
          Fixtures
        </div>
      </div>

      <div className="grid-two">
        {/* Left column: Fixtures + Context */}
        <div className="stack">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">Upcoming Matches</h2>
                <span className="module-subtitle">
                  {fixtures?.warning ?? "Source: data/football/fixtures.*"}
                </span>
              </div>
            </div>
            <div className="panel-body" style={{ padding: 0 }}>
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
                        <td className="mono" style={{ fontSize: 12 }}>{fixture.date}</td>
                        <td>{fixture.league}</td>
                        <td>{fixture.homeTeamId}</td>
                        <td>{fixture.awayTeamId}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="panel-body">
                  <div className="empty-state">
                    <span className="empty-state-text">No fixtures loaded</span>
                    <span className="empty-state-hint">Expected file: fixtures.parquet</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Key factors */}
          <div className="panel">
            <div className="panel-header">
              <h2 className="module-title">Key Factors</h2>
            </div>
            <div className="panel-body">
              <div className="data-health">
                <div className="data-health-row">
                  <span className="status-dot" />
                  <span className="data-health-label">Form</span>
                  <span className="data-health-hint">Awaiting data</span>
                </div>
                <div className="data-health-row">
                  <span className="status-dot" />
                  <span className="data-health-label">Home Advantage</span>
                  <span className="data-health-hint">Awaiting data</span>
                </div>
                <div className="data-health-row">
                  <span className="status-dot" />
                  <span className="data-health-label">Strength</span>
                  <span className="data-health-hint">Awaiting data</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right column: Predictions */}
        <div className="stack">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">1X2 Probabilities</h2>
                <span className="module-subtitle">Nearest match</span>
              </div>
            </div>
            <div className="panel-body">
              <div className="grid-three">
                <div className="stat" style={{ textAlign: "center" }}>
                  <span className="stat-label">Home</span>
                  <span className="stat-value-lg mono">—</span>
                </div>
                <div className="stat" style={{ textAlign: "center" }}>
                  <span className="stat-label">Draw</span>
                  <span className="stat-value-lg mono">—</span>
                </div>
                <div className="stat" style={{ textAlign: "center" }}>
                  <span className="stat-label">Away</span>
                  <span className="stat-value-lg mono">—</span>
                </div>
              </div>
            </div>
            <div className="panel-footer">No prediction available yet</div>
          </div>

          {/* Calibration */}
          <div className="panel">
            <div className="panel-header">
              <h2 className="module-title">Calibration Hint</h2>
            </div>
            <div className="panel-body">
              <div className="empty-state">
                <span className="empty-state-text">No calibration data</span>
                <span className="empty-state-hint">Run a backtest to see calibration metrics</span>
              </div>
            </div>
          </div>

          {/* Baseline comparison */}
          <div className="panel">
            <div className="panel-header">
              <h2 className="module-title">vs Baseline</h2>
            </div>
            <div className="panel-body">
              <div className="empty-state">
                <span className="empty-state-text">No baseline set</span>
                <span className="empty-state-hint">Configure Elo or odds as baseline</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
