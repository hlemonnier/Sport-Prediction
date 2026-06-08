import { getCatalog, listSweeps } from "@/lib/api";
import type { CatalogProject, SweepRow } from "@/lib/types";
import SweepForm from "@/components/SweepForm";
import Link from "next/link";

async function fetchCatalog(): Promise<CatalogProject[]> {
  try {
    const data = await getCatalog();
    return data.projects;
  } catch {
    return [];
  }
}

async function fetchSweeps(): Promise<SweepRow[]> {
  try {
    return await listSweeps();
  } catch {
    return [];
  }
}

export default async function SweepsPage() {
  const [projects, sweeps] = await Promise.all([fetchCatalog(), fetchSweeps()]);

  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Sweeps</h1>
        <p className="page-status">Parameter sweeps and hyperparameter search</p>
      </div>

      <SweepForm projects={projects} />

      <div className="panel">
        <div className="panel-header">
          <div className="panel-header-left">
            <h2 className="module-title">Sweep History</h2>
            <span className="module-subtitle">{sweeps.length} total</span>
          </div>
        </div>
        <div className="panel-body" style={{ padding: 0 }}>
          {sweeps.length === 0 ? (
            <div className="panel-body">
              <div className="empty-state">
                <span className="empty-state-text">No sweeps recorded yet</span>
                <span className="empty-state-hint">Launch a sweep above to get started</span>
              </div>
            </div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Sport</th>
                  <th>Project</th>
                  <th>Param</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {sweeps.map((sweep) => (
                  <tr key={sweep.id}>
                    <td className="mono" style={{ fontSize: 12 }}>
                      <Link href={`/sweeps/${sweep.id}`} className="text-accent">
                        {sweep.id.slice(0, 8)}
                      </Link>
                    </td>
                    <td>{sweep.sport}</td>
                    <td>{sweep.project}</td>
                    <td className="mono" style={{ fontSize: 12 }}>{sweep.param}</td>
                    <td>
                      <span className="chip">
                        <span className={`chip-led ${sweep.status === "done" ? "green" : sweep.status === "error" ? "red" : "amber"}`} />
                        {sweep.status}
                      </span>
                    </td>
                    <td className="mono" style={{ fontSize: 12, color: "var(--muted)" }}>
                      {new Date(sweep.createdAt).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
