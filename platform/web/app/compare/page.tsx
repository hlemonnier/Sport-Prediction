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

export default async function ComparePage() {
  const runs = await fetchRuns();
  return (
    <div className="stack">
      <div>
        <h1 className="section-title">Compare</h1>
        <p className="section-subtitle">
          Compare deux runs pour voir qui bouge, et de combien.
        </p>
      </div>
      <RunsCompare runs={runs} />
    </div>
  );
}
