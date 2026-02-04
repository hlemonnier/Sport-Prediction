import { API_BASE } from "@/lib/api";
import type { SweepDetail } from "@/lib/types";
import SweepChart from "@/components/SweepChart";

export default async function SweepDetailPage({ params }: { params: { id: string } }) {
  const res = await fetch(`${API_BASE}/api/sweeps/${params.id}`, { cache: "no-store" });
  if (!res.ok) {
    return (
      <div className="card">
        <h1 className="section-title">Sweep not found</h1>
      </div>
    );
  }
  const sweep = (await res.json()) as SweepDetail;
  return (
    <div className="stack">
      <div className="card">
        <h1 className="section-title">Sweep {sweep.id}</h1>
        <p className="section-subtitle">
          {sweep.sport} — {sweep.project} | Param: {sweep.param}
        </p>
      </div>
      <div className="card">
        <h2 className="section-title">Summary</h2>
        <SweepChart summary={sweep.summary} />
      </div>
      <div className="card">
        <h2 className="section-title">Runs</h2>
        <table className="table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Status</th>
              <th>Param Value</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {sweep.runs.map((run) => (
              <tr key={run.id}>
                <td>{run.id.slice(0, 8)}</td>
                <td>{run.status}</td>
                <td>{run.paramValue}</td>
                <td>{new Date(run.createdAt).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
