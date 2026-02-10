"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ReactECharts from "echarts-for-react";
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

const SAMPLE_FIXTURES: Fixture[] = [
  {
    matchId: "sample-epl-1",
    date: "2026-02-14 15:00",
    season: "2026",
    league: "EPL",
    homeTeamId: "Arsenal",
    awayTeamId: "Liverpool",
  },
  {
    matchId: "sample-epl-2",
    date: "2026-02-14 17:30",
    season: "2026",
    league: "EPL",
    homeTeamId: "Chelsea",
    awayTeamId: "Newcastle",
  },
  {
    matchId: "sample-ligue1-1",
    date: "2026-02-15 20:45",
    season: "2026",
    league: "Ligue 1",
    homeTeamId: "PSG",
    awayTeamId: "Monaco",
  },
];

export default function HomeDashboard() {
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [fixtures, setFixtures] = useState<FixtureResponse | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [usingSampleFixtures, setUsingSampleFixtures] = useState(false);

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
          setUsingSampleFixtures(false);
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

  const footballReady = Boolean(status?.football.matches.exists && status?.football.fixtures.exists);
  const lastRun = runs.length > 0 ? runs[0] : null;
  const doneRuns = runs.filter((run) => run.status === "done").length;
  const failedRuns = runs.filter((run) => run.status === "error").length;
  const pendingRuns = runs.length - doneRuns - failedRuns;
  const readyDatasets = status
    ? [status.football.teams, status.football.matches, status.football.fixtures].filter((item) => item.exists).length
    : 0;

  const dailyCounts = (() => {
    const buckets: Record<string, number> = {};
    const labels: string[] = [];
    for (let offset = 6; offset >= 0; offset -= 1) {
      const day = new Date();
      day.setHours(0, 0, 0, 0);
      day.setDate(day.getDate() - offset);
      const key = day.toISOString().slice(0, 10);
      buckets[key] = 0;
      labels.push(key);
    }
    runs.forEach((run) => {
      const key = new Date(run.createdAt).toISOString().slice(0, 10);
      if (key in buckets) {
        buckets[key] += 1;
      }
    });
    return labels.map((key) => ({
      label: key.slice(5),
      count: buckets[key],
    }));
  })();

  const statusChartOption = {
    grid: { left: 44, right: 20, top: 12, bottom: 24 },
    xAxis: {
      type: "category",
      data: ["Done", "Pending", "Error"],
      axisLabel: { color: "#58554f", fontSize: 11 },
      axisLine: { lineStyle: { color: "#e5e4e2" } },
    },
    yAxis: {
      type: "value",
      minInterval: 1,
      axisLabel: { color: "#58554f", fontSize: 11 },
      splitLine: { lineStyle: { color: "#ecebe9" } },
    },
    series: [
      {
        type: "bar",
        data: [
          { value: doneRuns, itemStyle: { color: "#16a34a" } },
          { value: pendingRuns, itemStyle: { color: "#d97706" } },
          { value: failedRuns, itemStyle: { color: "#dc2626" } },
        ],
        barMaxWidth: 40,
        itemStyle: { borderRadius: [4, 4, 0, 0] },
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: "#ffffff",
      borderColor: "#e5e4e2",
      textStyle: { color: "#1a1a19", fontSize: 11 },
    },
  };

  const trendChartOption = {
    grid: { left: 42, right: 16, top: 12, bottom: 24 },
    xAxis: {
      type: "category",
      data: dailyCounts.map((item) => item.label),
      axisLabel: { color: "#58554f", fontSize: 11 },
      axisLine: { lineStyle: { color: "#e5e4e2" } },
    },
    yAxis: {
      type: "value",
      minInterval: 1,
      axisLabel: { color: "#58554f", fontSize: 11 },
      splitLine: { lineStyle: { color: "#ecebe9" } },
    },
    series: [
      {
        type: "line",
        smooth: true,
        data: dailyCounts.map((item) => item.count),
        itemStyle: { color: "#dc2626" },
        lineStyle: { color: "#dc2626", width: 2 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(220, 38, 38, 0.18)" },
              { offset: 1, color: "rgba(220, 38, 38, 0)" },
            ],
          },
        },
        symbolSize: 6,
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: "#ffffff",
      borderColor: "#e5e4e2",
      textStyle: { color: "#1a1a19", fontSize: 11 },
    },
  };

  const loadSampleFixtures = () => {
    setUsingSampleFixtures(true);
    setFixtures({
      fixtures: SAMPLE_FIXTURES,
      warning: "Sample fixtures loaded locally for UI exploration.",
    });
  };

  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Dashboard</h1>
        <p className="page-status">Snapshot of data readiness, recent activity, and next actions.</p>
      </div>

      <div className="dashboard-kpis">
        <div className="kpi-card primary">
          <span className="kpi-label">Runs in workspace</span>
          <span className="kpi-value">{runs.length}</span>
          <span className="kpi-meta">
            {doneRuns} completed · {failedRuns} failed
          </span>
          <div className="kpi-actions">
            <Link href={runs.length > 0 ? "/runs" : "/f1/qualifying"} className="button button-sm">
              {runs.length > 0 ? "Open runs" : "Run first prediction"}
            </Link>
          </div>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Football data readiness</span>
          <span className="kpi-value">{readyDatasets}/3</span>
          <span className="kpi-meta">
            {footballReady ? "Ready for preview and match workflows." : "Teams, matches, and fixtures are required."}
          </span>
          <div className="kpi-actions">
            <Link href="/football/preview" className="button secondary button-sm">
              Open football preview
            </Link>
          </div>
        </div>

        <div className="kpi-card">
          <span className="kpi-label">Latest run</span>
          <span className="kpi-value mono">{lastRun ? lastRun.id.slice(0, 8) : "—"}</span>
          <span className="kpi-meta">
            {lastRun
              ? `${lastRun.project} · ${new Date(lastRun.createdAt).toLocaleDateString()}`
              : "No run executed yet in this workspace."}
          </span>
          <div className="kpi-actions">
            <Link href={lastRun ? `/runs/${lastRun.id}` : "/football/match"} className="button secondary button-sm">
              {lastRun ? "Inspect run" : "Run first prediction"}
            </Link>
          </div>
        </div>
      </div>

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Run Status Distribution</h2>
              <span className="module-subtitle">Done vs pending vs failed</span>
            </div>
          </div>
          <div className="panel-body">
            {runs.length === 0 ? (
              <div className="empty-state compact">
                <span className="empty-state-text">No run data to chart yet.</span>
                <span className="empty-state-hint">Launch a prediction to start visual tracking.</span>
              </div>
            ) : (
              <ReactECharts option={statusChartOption} style={{ height: 220 }} />
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Runs Trend (7 days)</h2>
              <span className="module-subtitle">Daily execution volume</span>
            </div>
          </div>
          <div className="panel-body">
            {runs.length === 0 ? (
              <div className="empty-state compact">
                <span className="empty-state-text">No recent trend to display.</span>
                <span className="empty-state-hint">The curve appears after your first runs.</span>
              </div>
            ) : (
              <ReactECharts option={trendChartOption} style={{ height: 220 }} />
            )}
          </div>
        </div>
      </div>

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Upcoming Matches</h2>
              <span className="module-subtitle">
                {usingSampleFixtures ? "Sample fixtures loaded locally" : "Football fixtures"}
              </span>
            </div>
            <div className="panel-header-actions">
              <span className="chip">
                <span
                  className={`chip-led ${
                    fixtures && fixtures.fixtures.length > 0
                      ? usingSampleFixtures
                        ? "amber"
                        : "green"
                      : "red"
                  }`}
                />
                {fixtures && fixtures.fixtures.length > 0
                  ? usingSampleFixtures
                    ? "Sample"
                    : "Live"
                  : "Missing"}
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
              <div className="stack-sm">
                <div className="empty-state compact">
                  <span className="empty-state-text">No fixtures file was found for preview.</span>
                  <span className="empty-state-hint">Expected source: `data/football/fixtures.parquet`.</span>
                  <div className="empty-state-actions">
                    <button type="button" className="button button-sm" onClick={loadSampleFixtures}>
                      Load sample fixtures
                    </button>
                    <Link href="/football/preview" className="button secondary button-sm">
                      Open football preview
                    </Link>
                  </div>
                </div>

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
              </div>
            )}
          </div>
          {fixtures?.warning && <div className="panel-footer">{fixtures.warning}</div>}
        </div>

        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">F1 Overview</h2>
              <span className="module-subtitle">Next event context and provider state</span>
            </div>
            <div className="panel-header-actions">
              <span className="chip">
                <span className="chip-led amber" />
                Provider missing
              </span>
            </div>
          </div>
          <div className="panel-body">
            <div className="empty-state compact">
              <span className="empty-state-text">No live provider is configured for F1 context yet.</span>
              <span className="empty-state-hint">
                Connect FastF1/OpenF1 to unlock circuit, weather, and session-aware preview.
              </span>
              <div className="empty-state-actions">
                <Link href="/diagnostics" className="button secondary button-sm">
                  Connect provider
                </Link>
              </div>
            </div>
            <div className="data-health" style={{ marginTop: 4 }}>
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
            <span>Use the top bar to set season and round.</span>
            <Link href="/f1/preview" className="button secondary button-sm">
              Open F1 preview
            </Link>
          </div>
        </div>
      </div>

      <div className="grid-three">
        <div className="panel">
          <div className="panel-header">
            <h2 className="module-title">Data Health</h2>
            <span className="module-subtitle">{readyDatasets}/3 ready</span>
          </div>
          <div className="panel-body">
            <div className="data-health">
              <div className="data-health-row">
                <span className={`status-dot ${status?.football.teams.exists ? "ok" : "miss"}`} />
                <span className="data-health-label">Teams</span>
                <span className="data-health-hint">{status?.football.teams.exists ? "Ready" : "Missing"}</span>
              </div>
              <div className="data-health-row">
                <span className={`status-dot ${status?.football.matches.exists ? "ok" : "miss"}`} />
                <span className="data-health-label">Matches</span>
                <span className="data-health-hint">{status?.football.matches.exists ? "Ready" : "Missing"}</span>
              </div>
              <div className="data-health-row">
                <span className={`status-dot ${status?.football.fixtures.exists ? "ok" : "miss"}`} />
                <span className="data-health-label">Fixtures</span>
                <span className="data-health-hint">{status?.football.fixtures.exists ? "Ready" : "Missing"}</span>
              </div>
            </div>
          </div>
        </div>

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
              <div className="empty-state compact">
                <span className="empty-state-text">No runs recorded yet</span>
                <span className="empty-state-hint">Start with one prediction run, then compare iterations.</span>
                <div className="empty-state-actions">
                  <Link href="/f1/qualifying" className="button button-sm">
                    Run first prediction
                  </Link>
                </div>
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

        <div className="panel">
          <div className="panel-header">
            <h2 className="module-title">Quick Compare</h2>
          </div>
          <div className="panel-body">
            {runs.length < 2 ? (
              <div className="empty-state compact">
                <span className="empty-state-text">Need at least 2 runs to compare</span>
                <span className="empty-state-hint">
                  Run another prediction to unlock side-by-side comparison.
                </span>
                <div className="empty-state-actions">
                  <Link href={runs.length === 0 ? "/football/match" : "/f1/race"} className="button button-sm">
                    {runs.length === 0 ? "Run first prediction" : "Run another prediction"}
                  </Link>
                </div>
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
