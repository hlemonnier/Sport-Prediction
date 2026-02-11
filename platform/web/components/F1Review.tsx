import RunsCompare from "@/components/RunsCompare";
import type { RunSummary } from "@/lib/types";
import { listRuns } from "@/lib/api";

async function fetchRuns(): Promise<RunSummary[]> {
  try {
    return await listRuns();
  } catch {
    return [];
  }
}

export default async function F1Review() {
  const runs = await fetchRuns();
  return (
    <div className="stack">
      <div>
        <h1 className="section-title">F1 Analyse</h1>
        <p className="section-subtitle">
          Analyse post-session : ecarts, surprises, stabilite du classement.
        </p>
      </div>
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Surprises</h2>
          <p className="module-subtitle">Top ecarts entre ranking predit et reel.</p>
          <p className="section-subtitle">Aucune evaluation disponible.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Metrics classement</h2>
          <p className="module-subtitle">Spearman · NDCG@10 · Top-10 hit rate</p>
          <p className="section-subtitle">Backtest non lance.</p>
        </div>
      </div>
      <RunsCompare runs={runs} />
    </div>
  );
}
