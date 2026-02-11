import { API_BASE } from "@/lib/api";
import type { RunDetail } from "@/lib/types";
import RunResult from "@/components/RunResult";

type RunDetailPageProps = {
  params: Promise<{ id: string }>;
};

export default async function RunDetailPage({ params }: RunDetailPageProps) {
  const { id } = await params;
  let run: RunDetail | null = null;
  try {
    const res = await fetch(`${API_BASE}/api/runs/${id}`, { cache: "no-store" });
    if (res.ok) {
      run = (await res.json()) as RunDetail;
    }
  } catch {
    run = null;
  }
  if (!run) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h1 className="module-title">Run Not Found</h1>
        </div>
        <div className="panel-body">
          <div className="empty-state">
            <span className="empty-state-text">This run does not exist or could not be loaded</span>
            <span className="empty-state-hint">Check the run ID and try again</span>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Run Detail</h1>
        <p className="page-status">{run.id}</p>
      </div>
      <RunResult run={run} />
    </div>
  );
}
