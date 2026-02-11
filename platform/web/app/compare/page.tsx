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

export default async function ComparePage() {
  const runs = await fetchRuns();
  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Compare</h1>
        <p className="page-status">Select two runs to inspect what changed and by how much</p>
      </div>
      <RunsCompare runs={runs} />
    </div>
  );
}
