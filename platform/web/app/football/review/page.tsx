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
        <h1 className="section-title">Football Review</h1>
        <p className="section-subtitle">
          Calibration, erreurs couteuses, et comparaison baseline.
        </p>
      </div>
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Calibration</h2>
          <p className="module-subtitle">Logloss · Brier · reliability bins</p>
          <p className="section-subtitle">No calibration run yet.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Error Hotspots</h2>
          <p className="module-subtitle">Matches where the model was most confident and wrong.</p>
          <p className="section-subtitle">Awaiting evaluation data.</p>
        </div>
      </div>
      <RunsCompare runs={runs} />
    </div>
  );
}
