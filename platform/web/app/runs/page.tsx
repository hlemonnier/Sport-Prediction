import { listRuns } from "@/lib/api";
import type { RunSummary } from "@/lib/types";
import RunsCompare from "@/components/RunsCompare";

async function fetchRuns(): Promise<RunSummary[]> {
  try {
    return await listRuns();
  } catch {
    return [];
  }
}

export default async function RunsPage() {
  const runs = await fetchRuns();
  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Runs</h1>
        <p className="page-status">{runs.length} run{runs.length !== 1 ? "s" : ""} recorded</p>
      </div>
      <RunsCompare runs={runs} />
    </div>
  );
}
