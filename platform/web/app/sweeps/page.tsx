import { API_BASE } from "@/lib/api";
import type { CatalogResponse, CatalogProject, SweepRow } from "@/lib/types";
import SweepForm from "@/components/SweepForm";
import Link from "next/link";

async function fetchCatalog(): Promise<CatalogProject[]> {
  try {
    const res = await fetch(`${API_BASE}/api/catalog`, { cache: "no-store" });
    if (!res.ok) {
      return [];
    }
    const data = (await res.json()) as CatalogResponse;
    return data.projects;
  } catch {
    return [];
  }
}

async function fetchSweeps(): Promise<SweepRow[]> {
  try {
    const res = await fetch(`${API_BASE}/api/sweeps`, { cache: "no-store" });
    if (!res.ok) {
      return [];
    }
    return (await res.json()) as SweepRow[];
  } catch {
    return [];
  }
}

export default async function SweepsPage() {
  const [projects, sweeps] = await Promise.all([fetchCatalog(), fetchSweeps()]);

  return (
    <div className="stack">
      <SweepForm projects={projects} />
      <div className="card">
        <h2 className="section-title">Sweeps</h2>
        {sweeps.length === 0 ? (
          <p className="section-subtitle">Aucun sweep pour l'instant.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Sport</th>
                <th>Projet</th>
                <th>Param</th>
                <th>Statut</th>
                <th>Creation</th>
              </tr>
            </thead>
            <tbody>
              {sweeps.map((sweep) => (
                <tr key={sweep.id}>
                  <td>
                    <Link href={`/sweeps/${sweep.id}`}>{sweep.id.slice(0, 8)}</Link>
                  </td>
                  <td>{sweep.sport}</td>
                  <td>{sweep.project}</td>
                  <td>{sweep.param}</td>
                  <td>{sweep.status}</td>
                  <td>{new Date(sweep.createdAt).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
