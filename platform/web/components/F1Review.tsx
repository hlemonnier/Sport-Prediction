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

export default async function F1Review() {
  const runs = await fetchRuns();
  return (
    <div className="stack">
      <div>
        <h1 className="section-title">F1 Review</h1>
        <p className="section-subtitle">
          Analyse post-session : écarts, surprises, stabilité du classement.
        </p>
      </div>
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Surprises</h2>
          <p className="module-subtitle">Top écarts entre ranking prédit et réel.</p>
          <p className="section-subtitle">No evaluation data available yet.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Ranking Metrics</h2>
          <p className="module-subtitle">Spearman · NDCG@10 · Top-10 hit rate</p>
          <p className="section-subtitle">Backtest not run yet.</p>
        </div>
      </div>
      <RunsCompare runs={runs} />
    </div>
  );
}
