"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
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
  const lastRun = runs.length > 0 ? runs[0] : null;

  return (
    <div className="stack-lg">
      {/* Page header — compact, functional */}
      <div>
        <h1 className="page-title">Dashboard</h1>
        <p className="page-status">
          {runs.length} run{runs.length !== 1 ? "s" : ""} recorded
          {lastRun ? ` · Last: ${new Date(lastRun.createdAt).toLocaleDateString()}` : ""}
          {footballReady ? " · Football data ready" : ""}
        </p>
      </div>

      {/* Top row: two wide panels */}
      <div className="grid-two">
        {/* Next Football Matches */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Upcoming Matches</h2>
              <span className="module-subtitle">Football fixtures</span>
            </div>
            <div className="panel-header-actions">
              <span className="chip">
                <span className={`chip-led ${footballReady ? "green" : "red"}`} />
                {footballReady ? "Data: OK" : "Data: Missing"}
              </span>
            </div>
          </div>
          <div className="panel-body">
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
              <div className="data-health">
                <div className="data-health-row">
                  <span className={`status-dot ${status?.football.fixtures.exists ? "ok" : "miss"}`} />
                  <span className="data-health-label">Fixtures</span>
                  <span className="data-health-hint">
                    {status?.football.fixtures.exists ? "Loaded" : "Expected: fixtures.parquet"}
                  </span>
                </div>
                <div className="data-health-row">
                  <span className={`status-dot ${status?.football.matches.exists ? "ok" : "miss"}`} />
                  <span className="data-health-label">Matches</span>
                  <span className="data-health-hint">
                    {status?.football.matches.exists ? "Loaded" : "Expected: matches.parquet"}
                  </span>
                </div>
              </div>
            )}
          </div>
          {fixtures?.warning && (
            <div className="panel-footer">{fixtures.warning}</div>
          )}
        </div>

        {/* F1 Context */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">F1 Overview</h2>
              <span className="module-subtitle">Next event context</span>
            </div>
            <div className="panel-header-actions">
              <span className="chip">
                <span className="chip-led amber" />
                No source
              </span>
            </div>
          </div>
          <div className="panel-body">
            <div className="data-health">
              <div className="data-health-row">
                <span className="status-dot miss" />
                <span className="data-health-label">FastF1</span>
                <span className="data-health-hint">Not connected</span>
              </div>
              <div className="data-health-row">
                <span className="status-dot miss" />
                <span className="data-health-label">Calendar</span>
                <span className="data-health-hint">Connect FastF1/OpenF1 to enable preview</span>
              </div>
            </div>
          </div>
          <div className="panel-footer">
            Use the top bar to set season &amp; round
          </div>
        </div>
      </div>

      {/* Second row: three panels */}
      <div className="grid-three">
        {/* Data Health */}
        <div className="panel">
          <div className="panel-header">
            <h2 className="module-title">Data Health</h2>
          </div>
          <div className="panel-body">
            <div className="data-health">
              <div className="data-health-row">
                <span className={`status-dot ${status?.football.teams.exists ? "ok" : "miss"}`} />
                <span className="data-health-label">Teams</span>
                <span className="data-health-hint">
                  {status?.football.teams.exists ? "OK" : "Missing"}
                </span>
              </div>
              <div className="data-health-row">
                <span className={`status-dot ${status?.football.matches.exists ? "ok" : "miss"}`} />
                <span className="data-health-label">Matches</span>
                <span className="data-health-hint">
                  {status?.football.matches.exists ? "OK" : "Missing"}
                </span>
              </div>
              <div className="data-health-row">
                <span className={`status-dot ${status?.football.fixtures.exists ? "ok" : "miss"}`} />
                <span className="data-health-label">Fixtures</span>
                <span className="data-health-hint">
                  {status?.football.fixtures.exists ? "OK" : "Missing"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Latest Runs */}
        <div className="panel">
          <div className="panel-header">
            <h2 className="module-title">Latest Runs</h2>
            <span className="chip">
              <span className={`chip-led ${runs.length > 0 ? "green" : "amber"}`} />
              {runs.length}
            </span>
          </div>
          <div className="panel-body">
            {runs.length === 0 ? (
              <div className="empty-state">
                <span className="empty-state-text">No runs recorded yet</span>
                <span className="empty-state-hint">Launch a run from F1 or Football modules</span>
              </div>
            ) : (
              <div className="stack-sm">
                {runs.slice(0, 4).map((run) => (
                  <Link
                    key={run.id}
                    href={`/runs/${run.id}`}
                    className="data-health-row"
                    style={{ textDecoration: "none" }}
                  >
                    <span className={`status-dot ${run.status === "done" ? "ok" : run.status === "error" ? "miss" : "warn"}`} />
                    <span className="data-health-label">{run.project}</span>
                    <span className="data-health-hint">{run.id.slice(0, 8)}</span>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Quick Compare */}
        <div className="panel">
          <div className="panel-header">
            <h2 className="module-title">Quick Compare</h2>
          </div>
          <div className="panel-body">
            {runs.length < 2 ? (
              <div className="empty-state">
                <span className="empty-state-text">Need at least 2 runs to compare</span>
                <span className="empty-state-hint">Run multiple iterations to track drift</span>
              </div>
            ) : (
              <div className="stack-sm">
                <div className="data-health-row">
                  <span className="data-health-label">Last</span>
                  <span className="data-health-hint">{runs[0].id.slice(0, 8)} — {runs[0].status}</span>
                </div>
                <div className="data-health-row">
                  <span className="data-health-label">Previous</span>
                  <span className="data-health-hint">{runs[1].id.slice(0, 8)} — {runs[1].status}</span>
                </div>
                <Link href="/compare" className="button secondary button-sm" style={{ alignSelf: "flex-start", marginTop: 4 }}>
                  Open Compare
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Runs Table */}
      {runs.length > 0 && (
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Recent Runs</h2>
              <span className="module-subtitle">{runs.length} total</span>
            </div>
            <Link href="/runs" className="button secondary button-sm">
              View All
            </Link>
          </div>
          <div className="panel-body" style={{ padding: 0 }}>
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Sport</th>
                  <th>Project</th>
                  <th>Status</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {runs.slice(0, 8).map((run) => (
                  <tr key={run.id}>
                    <td className="mono" style={{ fontSize: 12 }}>
                      <Link href={`/runs/${run.id}`} className="text-accent">
                        {run.id.slice(0, 8)}
                      </Link>
                    </td>
                    <td>{run.sport}</td>
                    <td>{run.project}</td>
                    <td>
                      <span className="chip">
                        <span className={`chip-led ${run.status === "done" ? "green" : run.status === "error" ? "red" : "amber"}`} />
                        {run.status}
                      </span>
                    </td>
                    <td className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>
                      {new Date(run.createdAt).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
