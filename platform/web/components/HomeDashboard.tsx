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
        <h1 className="section-title">Cockpit de Recherche</h1>
        <p className="section-subtitle">
          Tableau de bord personnel pour préparer, lancer et revoir les prédictions.
        </p>
      </div>

      <div className="status-strip">
        <div className="status-item">
          <span className={`status-dot ${footballReady ? "ok" : "miss"}`} />
          Donnees football
        </div>
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
        <div className="status-item">
          <span className={`status-dot ${runs.length > 0 ? "ok" : "warn"}`} />
          Runs {runs.length}
        </div>
      </div>

      <div className="grid-two">
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
        <div className="card">
          <h2 className="module-title">F1 contexte</h2>
          <p className="module-subtitle">Utilise la topbar pour choisir la saison et le round.</p>
          <div className="stack">
            <span className="pill">Pas de source calendrier connectee.</span>
            <span className="section-subtitle">
              Branche FastF1/OpenF1 dans la pipeline pour activer la preview.
            </span>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="module-title">Runs recents</h2>
        {runs.length === 0 ? (
          <p className="section-subtitle">Aucune run enregistree.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Sport</th>
                <th>Projet</th>
                <th>Statut</th>
                <th>Date</th>
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
