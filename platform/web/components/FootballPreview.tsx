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
      <div className="status-strip">
        <div className="status-item">
          <span className={`status-dot ${status?.football.teams.exists ? "ok" : "miss"}`} />
          Equipes
        </div>
        <div className="status-item">
          <span className={`status-dot ${status?.football.matches.exists ? "ok" : "miss"}`} />
          Matchs
        </div>
        <div className="status-item">
          <span className={`status-dot ${status?.football.fixtures.exists ? "ok" : "miss"}`} />
          Fixtures
        </div>
      </div>
      <div className="card">
        <h2 className="module-title">Prochains matchs</h2>
        <p className="module-subtitle">
          {fixtures?.warning ?? "Source: data/football/fixtures.*"}
        </p>
        {fixtures && fixtures.fixtures.length > 0 ? (
          <table className="table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Ligue</th>
                  <th>Domicile</th>
                  <th>Exterieur</th>
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
          <p className="section-subtitle">Aucun fixture charge pour le moment.</p>
        )}
      </div>
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Proba 1X2</h2>
          <p className="module-subtitle">Match le plus proche.</p>
          <p className="section-subtitle">Aucune prediction pour l'instant.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Facteurs cles</h2>
          <p className="module-subtitle">Forme, home advantage, strength.</p>
          <p className="section-subtitle">Signaux en attente de donnees.</p>
        </div>
      </div>
    </div>
  );
}
