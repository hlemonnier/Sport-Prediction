import { API_BASE } from "@/lib/api";
import type { RunDetail } from "@/lib/types";
import RunResult from "@/components/RunResult";

export default async function RunDetailPage({ params }: { params: { id: string } }) {
  const res = await fetch(`${API_BASE}/api/runs/${params.id}`, { cache: "no-store" });
  if (!res.ok) {
    return (
      <div className="card">
        <h1 className="section-title">Run not found</h1>
      </div>
    );
  }
  const run = (await res.json()) as RunDetail;
  return <RunResult run={run} />;
}
