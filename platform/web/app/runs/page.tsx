import { API_BASE } from "@/lib/api";
import type { RunSummary } from "@/lib/types";
import RunsCompare from "@/components/RunsCompare";

async function fetchRuns(): Promise<RunSummary[]> {
  try {
    const res = await fetch(`${API_BASE}/api/runs`, { cache: "no-store" });
    if (!res.ok) {
      return [];
    }
    return (await res.json()) as RunSummary[];
  } catch {
    return [];
  }
}

export default async function RunsPage() {
  const runs = await fetchRuns();
  return <RunsCompare runs={runs} />;
}
