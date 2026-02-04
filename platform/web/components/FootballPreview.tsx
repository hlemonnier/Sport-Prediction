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
    <div className="stack">
      <div className="card">
        <h2 className="module-title">Data Readiness</h2>
        <p className="module-subtitle">
          Teams: {status?.football.teams.exists ? "OK" : "Missing"} · Matches: {" "}
          {status?.football.matches.exists ? "OK" : "Missing"} · Fixtures: {" "}
          {status?.football.fixtures.exists ? "OK" : "Missing"}
        </p>
        <p className="section-subtitle">Populate data/football to unlock predictions.</p>
      </div>
      <div className="card">
        <h2 className="module-title">Upcoming Fixtures</h2>
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
    </div>
  );
}
