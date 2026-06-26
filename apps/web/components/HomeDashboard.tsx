"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import ReactECharts from "echarts-for-react";
import { getDataStatus, getFootballFixtures, listRuns } from "@/lib/api";
import type { Fixture, FixtureResponse, RunSummary, DataStatus } from "@/lib/types";
import {
  defaultUiPreferences,
  readUiPreferences,
  subscribeUiPreferences,
} from "@/lib/uiPreferences";

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
  const [uiPrefs, setUiPrefs] = useState(defaultUiPreferences);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        setStatus(await getDataStatus());
      } catch {
        setStatus(null);
      }
    };
    const fetchFixtures = async () => {
      try {
        setFixtures(await getFootballFixtures(5));
        setUsingSampleFixtures(false);
      } catch {
        setFixtures(null);
      }
    };
    const fetchRuns = async () => {
      try {
        setRuns(await listRuns());
      } catch {
        setRuns([]);
      }
    };
    fetchStatus();
    fetchFixtures();
    fetchRuns();
  }, []);

  useEffect(() => {
    const syncPreferences = () => {
      setUiPrefs(readUiPreferences());
    };
    syncPreferences();
    return subscribeUiPreferences(syncPreferences);
  }, []);

  const footballReady = Boolean(status?.football.matches.exists && status?.football.fixtures.exists);
  const activeSport = uiPrefs.defaultSportModule;
  const sportRuns = runs.filter((run) => run.sport === activeSport);
  const latestSportRun = sportRuns.length > 0 ? sportRuns[0] : null;
  const doneRuns = sportRuns.filter((run) => run.status === "done").length;
  const failedRuns = sportRuns.filter((run) => run.status === "error").length;
  const pendingRuns = sportRuns.length - doneRuns - failedRuns;
  const readyDatasets = status
    ? [status.football.teams, status.football.matches, status.football.fixtures].filter((item) => item.exists).length
    : 0;
  const showActiveSportModule =
    activeSport === "Football" ? uiPrefs.showUpcomingMatches : uiPrefs.showF1Overview;
  const upperModuleCount = showActiveSportModule ? 1 : 0;
  const lowerModuleCount =
    (uiPrefs.showDataHealth ? 1 : 0) +
    (uiPrefs.showLatestRuns ? 1 : 0) +
    (uiPrefs.showQuickCompare ? 1 : 0);
  const chartRangeDays = uiPrefs.dashboardChartRange === "30d" ? 30 : 7;
  const firstRunHref = uiPrefs.defaultSportModule === "F1" ? "/f1/research/qualifying" : "/football/match";
  const nextRunHref = uiPrefs.defaultSportModule === "F1" ? "/f1/insights/live" : "/football/match";
  const activeSportLabel = activeSport === "F1" ? "F1" : "Football";

  const cssValue = (name: string, fallback: string) => {
    if (typeof window === "undefined") return fallback;
    const value = getComputedStyle(document.body).getPropertyValue(name).trim();
    return value || fallback;
  };

  const toRgba = (color: string, alpha: number) => {
    const hex = color.replace("#", "").trim();
    if (/^[0-9a-fA-F]{6}$/.test(hex)) {
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    if (/^[0-9a-fA-F]{3}$/.test(hex)) {
      const r = parseInt(hex[0] + hex[0], 16);
      const g = parseInt(hex[1] + hex[1], 16);
      const b = parseInt(hex[2] + hex[2], 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    return color;
  };

  const inkColor = cssValue("--ink", "#fefefe");
  const mutedColor = cssValue("--muted", "#a7a7a7");
  const borderColor = cssValue("--border", "#292929");
  const accentColor = cssValue("--accent", "#ff6363");
  const panelColor = cssValue("--panel", "#151515");
  const accentAreaTop = toRgba(accentColor, 0.2);
  const accentAreaBottom = toRgba(accentColor, 0);

  const dailyCounts = (() => {
    const buckets: Record<string, number> = {};
    const labels: string[] = [];
    for (let offset = chartRangeDays - 1; offset >= 0; offset -= 1) {
      const day = new Date();
      day.setHours(0, 0, 0, 0);
      day.setDate(day.getDate() - offset);
      const key = day.toISOString().slice(0, 10);
      buckets[key] = 0;
      labels.push(key);
    }
    sportRuns.forEach((run) => {
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
      axisLabel: { color: mutedColor, fontSize: 11 },
      axisLine: { lineStyle: { color: borderColor } },
    },
    yAxis: {
      type: "value",
      minInterval: 1,
      axisLabel: { color: mutedColor, fontSize: 11 },
      splitLine: { lineStyle: { color: borderColor } },
    },
    series: [
      {
        type: "bar",
        data: [
          { value: doneRuns, itemStyle: { color: "#16a34a" } },
          { value: pendingRuns, itemStyle: { color: "#d97706" } },
          { value: failedRuns, itemStyle: { color: accentColor } },
        ],
        barMaxWidth: 40,
        itemStyle: { borderRadius: [4, 4, 0, 0] },
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: panelColor,
      borderColor: borderColor,
      textStyle: { color: inkColor, fontSize: 11 },
    },
  };

  const trendChartOption = {
    grid: { left: 42, right: 16, top: 12, bottom: 24 },
    xAxis: {
      type: "category",
      data: dailyCounts.map((item) => item.label),
      axisLabel: { color: mutedColor, fontSize: 11 },
      axisLine: { lineStyle: { color: borderColor } },
    },
    yAxis: {
      type: "value",
      minInterval: 1,
      axisLabel: { color: mutedColor, fontSize: 11 },
      splitLine: { lineStyle: { color: borderColor } },
    },
    series: [
      {
        type: uiPrefs.dashboardChartStyle,
        smooth: true,
        data: dailyCounts.map((item) => item.count),
        itemStyle: { color: accentColor },
        lineStyle: { color: accentColor, width: 2 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: accentAreaTop },
              { offset: 1, color: accentAreaBottom },
            ],
          },
        },
        symbolSize: uiPrefs.dashboardChartStyle === "line" ? 6 : 0,
        barMaxWidth: uiPrefs.dashboardChartStyle === "bar" ? 24 : undefined,
      },
    ],
    tooltip: {
      trigger: "axis",
      backgroundColor: panelColor,
      borderColor: borderColor,
      textStyle: { color: inkColor, fontSize: 11 },
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
        <h1 className="page-title">{activeSportLabel} Dashboard</h1>
        <p className="page-status">Snapshot of selected-sport readiness, recent activity, and next actions.</p>
      </div>

      {uiPrefs.showKpiCards ? (
        <div className="dashboard-kpis">
          <div className="kpi-card primary">
            <span className="kpi-label">{activeSportLabel} runs</span>
            <span className="kpi-value">{sportRuns.length}</span>
            <span className="kpi-meta">
              {doneRuns} completed · {failedRuns} failed
            </span>
            <div className="kpi-actions">
              <Link href={sportRuns.length > 0 ? "/runs" : firstRunHref} className="button button-sm">
                {sportRuns.length > 0 ? "Open runs" : "Run first prediction"}
              </Link>
            </div>
          </div>

          <div className="kpi-card">
            {activeSport === "Football" ? (
              <>
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
              </>
            ) : (
              <>
                <span className="kpi-label">F1 platform module</span>
                <span className="kpi-value">Live</span>
                <span className="kpi-meta">OpenF1 transport, FastF1 batch jobs, and prediction service boundary.</span>
                <div className="kpi-actions">
                  <Link href="/f1/insights/live" className="button secondary button-sm">
                    Open F1 insights
                  </Link>
                </div>
              </>
            )}
          </div>

          <div className="kpi-card">
            <span className="kpi-label">Latest {activeSportLabel} run</span>
            <span className="kpi-value mono">{latestSportRun ? latestSportRun.id.slice(0, 8) : "—"}</span>
            <span className="kpi-meta">
              {latestSportRun
                ? `${latestSportRun.project} · ${new Date(latestSportRun.createdAt).toLocaleDateString()}`
                : `No ${activeSportLabel} run executed yet in this workspace.`}
            </span>
            <div className="kpi-actions">
              <Link href={latestSportRun ? `/runs/${latestSportRun.id}` : firstRunHref} className="button secondary button-sm">
                {latestSportRun ? "Inspect run" : "Run first prediction"}
              </Link>
            </div>
          </div>
        </div>
      ) : (
        <div className="panel">
          <div className="panel-body">
            <div className="empty-state compact">
              <span className="empty-state-text">KPI cards are hidden in your settings.</span>
              <span className="empty-state-hint">You can re-enable them from the Settings page.</span>
              <div className="empty-state-actions">
                <Link href="/settings" className="button secondary button-sm">
                  Open settings
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}

      {uiPrefs.showDashboardCharts ? (
        <div className="grid-two">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">{activeSportLabel} Run Status</h2>
                <span className="module-subtitle">Done vs pending vs failed</span>
              </div>
            </div>
            <div className="panel-body">
              {sportRuns.length === 0 ? (
                <div className="empty-state compact">
                  <span className="empty-state-text">No {activeSportLabel} run data to chart yet.</span>
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
                <h2 className="module-title">{activeSportLabel} Trend ({chartRangeDays} days)</h2>
                <span className="module-subtitle">Daily execution volume</span>
              </div>
            </div>
            <div className="panel-body">
              {sportRuns.length === 0 ? (
                <div className="empty-state compact">
                  <span className="empty-state-text">No recent {activeSportLabel} trend to display.</span>
                  <span className="empty-state-hint">The curve appears after your first runs.</span>
                </div>
              ) : (
                <ReactECharts option={trendChartOption} style={{ height: 220 }} />
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="panel">
          <div className="panel-body">
            <div className="empty-state compact">
              <span className="empty-state-text">Dashboard charts are hidden in your settings.</span>
              <span className="empty-state-hint">Enable charts again from the Settings page at any time.</span>
              <div className="empty-state-actions">
                <Link href="/settings" className="button secondary button-sm">
                  Open settings
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}

      {upperModuleCount > 0 ? (
        <div className={`dashboard-grid-two ${upperModuleCount === 1 ? "single" : ""}`}>
          {activeSport === "Football" && uiPrefs.showUpcomingMatches ? (
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
          ) : null}

          {activeSport === "F1" && uiPrefs.showF1Overview ? (
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
                <Link href="/f1/research/preview" className="button secondary button-sm">
                  Open F1 preview
                </Link>
              </div>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="panel">
          <div className="panel-body">
            <div className="empty-state compact">
              <span className="empty-state-text">{activeSportLabel} dashboard module is hidden.</span>
              <span className="empty-state-hint">
                Re-enable the selected sport module from Settings.
              </span>
              <div className="empty-state-actions">
                <Link href="/settings" className="button secondary button-sm">
                  Open settings
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}

      {lowerModuleCount > 0 ? (
        <div
          className={`dashboard-grid-three ${
            lowerModuleCount === 1 ? "single" : lowerModuleCount === 2 ? "double" : ""
          }`}
        >
          {uiPrefs.showDataHealth ? (
            <div className="panel">
              <div className="panel-header">
                <h2 className="module-title">{activeSportLabel} Data Health</h2>
                <span className="module-subtitle">
                  {activeSport === "Football" ? `${readyDatasets}/3 ready` : "Platform services"}
                </span>
              </div>
              <div className="panel-body">
                {activeSport === "Football" ? (
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
                ) : (
                  <div className="data-health">
                    <div className="data-health-row">
                      <span className="status-dot ok" />
                      <span className="data-health-label">F1 platform API</span>
                      <span className="data-health-hint">Snapshot, replay, ingest and stream contracts</span>
                    </div>
                    <div className="data-health-row">
                      <span className="status-dot ok" />
                      <span className="data-health-label">Prediction service</span>
                      <span className="data-health-hint">Separate model boundary with fallback</span>
                    </div>
                    <div className="data-health-row">
                      <span className="status-dot warn" />
                      <span className="data-health-label">Live credentials</span>
                      <span className="data-health-hint">OpenF1 auth remains backend-only</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : null}

          {uiPrefs.showLatestRuns ? (
            <div className="panel">
              <div className="panel-header">
                <h2 className="module-title">Latest {activeSportLabel} Runs</h2>
                <span className="chip">
                  <span className={`chip-led ${sportRuns.length > 0 ? "green" : "amber"}`} />
                  {sportRuns.length}
                </span>
              </div>
              <div className="panel-body">
                {sportRuns.length === 0 ? (
                  <div className="empty-state compact">
                    <span className="empty-state-text">No {activeSportLabel} runs recorded yet</span>
                    <span className="empty-state-hint">Start with one prediction run, then compare iterations.</span>
                    <div className="empty-state-actions">
                      <Link href={firstRunHref} className="button button-sm">
                        Run first prediction
                      </Link>
                    </div>
                  </div>
                ) : (
                  <div className="stack-sm">
                    {sportRuns.slice(0, 4).map((run) => (
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
          ) : null}

          {uiPrefs.showQuickCompare ? (
            <div className="panel">
              <div className="panel-header">
                <h2 className="module-title">{activeSportLabel} Quick Compare</h2>
              </div>
              <div className="panel-body">
                {sportRuns.length < 2 ? (
                  <div className="empty-state compact">
                    <span className="empty-state-text">Need at least 2 runs to compare</span>
                    <span className="empty-state-hint">
                      Run another prediction to unlock side-by-side comparison.
                    </span>
                    <div className="empty-state-actions">
                      <Link href={sportRuns.length === 0 ? firstRunHref : nextRunHref} className="button button-sm">
                        {sportRuns.length === 0 ? "Run first prediction" : "Run another prediction"}
                      </Link>
                    </div>
                  </div>
                ) : (
                  <div className="stack-sm">
                    <div className="data-health-row">
                      <span className="data-health-label">Last</span>
                      <span className="data-health-hint">{sportRuns[0].id.slice(0, 8)} — {sportRuns[0].status}</span>
                    </div>
                    <div className="data-health-row">
                      <span className="data-health-label">Previous</span>
                      <span className="data-health-hint">{sportRuns[1].id.slice(0, 8)} — {sportRuns[1].status}</span>
                    </div>
                    <Link href="/compare" className="button secondary button-sm" style={{ alignSelf: "flex-start", marginTop: 4 }}>
                      Open Compare
                    </Link>
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </div>
      ) : (
        <div className="panel">
          <div className="panel-body">
            <div className="empty-state compact">
              <span className="empty-state-text">Secondary dashboard modules are hidden.</span>
              <span className="empty-state-hint">
                Re-enable Data Health, Latest Runs, or Quick Compare from Settings.
              </span>
              <div className="empty-state-actions">
                <Link href="/settings" className="button secondary button-sm">
                  Open settings
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}

      {uiPrefs.showRecentRuns ? (
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Recent {activeSportLabel} Runs</h2>
              <span className="module-subtitle">{sportRuns.length} total</span>
            </div>
            {sportRuns.length > 0 ? (
              <Link href="/runs" className="button secondary button-sm">
                View All
              </Link>
            ) : null}
          </div>
          {sportRuns.length > 0 ? (
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
                  {sportRuns.slice(0, 8).map((run) => (
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
          ) : (
            <div className="panel-body">
              <div className="empty-state compact">
                <span className="empty-state-text">No {activeSportLabel} runs to list yet.</span>
                <span className="empty-state-hint">Run your first prediction to populate this table.</span>
                <div className="empty-state-actions">
                  <Link href={firstRunHref} className="button button-sm">
                    Run first prediction
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
