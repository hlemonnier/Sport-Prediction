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
    <div className="stack-lg">
      <div>
        <h1 className="page-title">F1 Review</h1>
        <p className="page-status">Post-session analysis: gaps, surprises, ranking stability</p>
      </div>

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Surprises</h2>
              <span className="module-subtitle">Top gaps between predicted and actual</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="empty-state">
              <span className="empty-state-text">No evaluation available</span>
              <span className="empty-state-hint">Run a backtest with actual results to see surprises</span>
            </div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Ranking Metrics</h2>
              <span className="module-subtitle">Spearman &middot; NDCG@10 &middot; Top-10 hit rate</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="grid-three">
              <div className="stat">
                <span className="stat-label">Spearman</span>
                <span className="stat-value mono">—</span>
              </div>
              <div className="stat">
                <span className="stat-label">NDCG@10</span>
                <span className="stat-value mono">—</span>
              </div>
              <div className="stat">
                <span className="stat-label">Top-10</span>
                <span className="stat-value mono">—</span>
              </div>
            </div>
          </div>
          <div className="panel-footer">Backtest not launched</div>
        </div>
      </div>

      <RunsCompare runs={runs} />
    </div>
  );
}
