import RunsCompare from "@/components/RunsCompare";
import type { RunSummary } from "@/lib/types";
import { API_BASE } from "@/lib/api";

async function fetchRuns(): Promise<RunSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/api/runs`, { cache: "no-store" });
    if (!res.ok) return [];
    return (await res.json()) as RunSummary[];
  } catch {
    return [];
  }
}

export default async function FootballReviewPage() {
  const runs = await fetchRuns();
  return (
    <div className="stack">
      <div>
        <h1 className="section-title">Football Analyse</h1>
        <p className="section-subtitle">
          Calibration, erreurs couteuses, et comparaison baseline.
        </p>
      </div>
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Calibration</h2>
          <p className="module-subtitle">Logloss · Brier · bins de calibration</p>
          <p className="section-subtitle">Pas de calibration pour l'instant.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Zones d'erreur</h2>
          <p className="module-subtitle">Matchs ou le modele etait confiant et faux.</p>
          <p className="section-subtitle">En attente de donnees.</p>
        </div>
      </div>
      <RunsCompare runs={runs} />
    </div>
  );
}
