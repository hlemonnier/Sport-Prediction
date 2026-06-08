"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getDataStatus, getFootballFixtures } from "@/lib/api";
import type { DataStatus, Fixture, FixtureResponse } from "@/lib/types";

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

export default function FootballPreview() {
  const [status, setStatus] = useState<DataStatus | null>(null);
  const [fixtures, setFixtures] = useState<FixtureResponse | null>(null);
  const [usingSampleFixtures, setUsingSampleFixtures] = useState(false);

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
        setFixtures(await getFootballFixtures(8));
        setUsingSampleFixtures(false);
      } catch {
        setFixtures(null);
      }
    };
    fetchStatus();
    fetchFixtures();
  }, []);

  const fixtureRows = fixtures?.fixtures ?? [];
  const hasFixtures = fixtureRows.length > 0;
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
                  {usingSampleFixtures
                    ? "Sample fixtures loaded locally"
                    : fixtures?.warning ?? "Source: data/football/fixtures.*"}
                </span>
              </div>
            </div>
            <div className="panel-body" style={{ padding: 0 }}>
              {hasFixtures ? (
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
                    {fixtureRows.map((fixture) => (
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
                  <div className="empty-state compact">
                    <span className="empty-state-text">No fixtures loaded yet.</span>
                    <span className="empty-state-hint">
                      Add `fixtures.parquet` or use a local sample to continue UI checks.
                    </span>
                    <div className="empty-state-actions">
                      <button type="button" className="button button-sm" onClick={loadSampleFixtures}>
                        Load sample fixtures
                      </button>
                      <Link href="/football/match" className="button secondary button-sm">
                        Run first prediction
                      </Link>
                    </div>
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
              <div className="empty-state compact">
                <span className="empty-state-text">No prediction available yet.</span>
                <span className="empty-state-hint">
                  Launch a match run to populate home/draw/away probabilities.
                </span>
                <div className="empty-state-actions">
                  <Link href="/football/match" className="button button-sm">
                    Run first prediction
                  </Link>
                </div>
              </div>
            </div>
          </div>

          {/* Calibration */}
          <div className="panel">
            <div className="panel-header">
              <h2 className="module-title">Calibration Hint</h2>
            </div>
            <div className="panel-body">
              <div className="empty-state compact">
                <span className="empty-state-text">No calibration data</span>
                <span className="empty-state-hint">Run a backtest to see calibration metrics</span>
                <div className="empty-state-actions">
                  <Link href="/football/review" className="button secondary button-sm">
                    Open review
                  </Link>
                </div>
              </div>
            </div>
          </div>

          {/* Baseline comparison */}
          <div className="panel">
            <div className="panel-header">
              <h2 className="module-title">vs Baseline</h2>
            </div>
            <div className="panel-body">
              <div className="empty-state compact">
                <span className="empty-state-text">No baseline set</span>
                <span className="empty-state-hint">Configure Elo or odds as baseline</span>
                <div className="empty-state-actions">
                  <Link href="/diagnostics" className="button secondary button-sm">
                    Configure baseline
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
