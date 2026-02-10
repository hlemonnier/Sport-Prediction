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
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Football Review</h1>
        <p className="page-status">Calibration, costly errors, and baseline comparison</p>
      </div>

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Calibration</h2>
              <span className="module-subtitle">Logloss &middot; Brier &middot; calibration bins</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="grid-three">
              <div className="stat">
                <span className="stat-label">Logloss</span>
                <span className="stat-value mono">—</span>
              </div>
              <div className="stat">
                <span className="stat-label">Brier</span>
                <span className="stat-value mono">—</span>
              </div>
              <div className="stat">
                <span className="stat-label">Bins</span>
                <span className="stat-value mono">—</span>
              </div>
            </div>
          </div>
          <div className="panel-footer">No calibration data yet</div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Error Zones</h2>
              <span className="module-subtitle">Matches where the model was confident and wrong</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="empty-state">
              <span className="empty-state-text">Awaiting data</span>
              <span className="empty-state-hint">Run backtests to identify systematic errors</span>
            </div>
          </div>
        </div>
      </div>

      <RunsCompare runs={runs} />
    </div>
  );
}
