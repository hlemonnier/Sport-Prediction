import { getSweep } from "@/lib/api";
import type { SweepDetail } from "@/lib/types";
import SweepChart from "@/components/SweepChart";

type SweepDetailPageProps = {
  params: Promise<{ id: string }>;
};

export default async function SweepDetailPage({ params }: SweepDetailPageProps) {
  const { id } = await params;
  let sweep: SweepDetail | null = null;
  try {
    sweep = await getSweep(id);
  } catch {
    sweep = null;
  }
  if (!sweep) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h1 className="module-title">Sweep Not Found</h1>
        </div>
        <div className="panel-body">
          <div className="empty-state">
            <span className="empty-state-text">This sweep does not exist or could not be loaded</span>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Sweep Detail</h1>
        <p className="page-status">
          {sweep.sport} &middot; {sweep.project} &middot; Param: {sweep.param}
        </p>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div className="panel-header-left">
            <h2 className="module-title">Summary</h2>
            <span className="module-subtitle">{sweep.id}</span>
          </div>
        </div>
        <div className="panel-body">
          <SweepChart summary={sweep.summary} />
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div className="panel-header-left">
            <h2 className="module-title">Runs</h2>
            <span className="module-subtitle">{sweep.runs.length} total</span>
          </div>
        </div>
        <div className="panel-body" style={{ padding: 0 }}>
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
                  <td className="mono" style={{ fontSize: 12 }}>{run.id.slice(0, 8)}</td>
                  <td>
                    <span className="chip">
                      <span className={`chip-led ${run.status === "done" ? "green" : run.status === "error" ? "red" : "amber"}`} />
                      {run.status}
                    </span>
                  </td>
                  <td className="mono" style={{ fontSize: 12 }}>{run.paramValue}</td>
                  <td className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>
                    {new Date(run.createdAt).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
