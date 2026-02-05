import { API_BASE } from "@/lib/api";
import type { RunDetail } from "@/lib/types";
import RunResult from "@/components/RunResult";

export default async function RunDetailPage({ params }: { params: { id: string } }) {
  let run: RunDetail | null = null;
  try {
    const res = await fetch(`${API_BASE}/api/runs/${params.id}`, { cache: "no-store" });
    if (res.ok) {
      run = (await res.json()) as RunDetail;
    }
  } catch {
    run = null;
  }
  if (!run) {
    return (
      <div className="card">
        <h1 className="section-title">Run introuvable</h1>
      </div>
    );
  }
  return <RunResult run={run} />;
}
