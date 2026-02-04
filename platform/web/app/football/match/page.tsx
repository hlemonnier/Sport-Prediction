import { API_BASE } from "@/lib/api";
import type { CatalogProject, CatalogResponse } from "@/lib/types";
import RunForm from "@/components/RunForm";

async function fetchProject(): Promise<CatalogProject | null> {
  try {
    const res = await fetch(`${API_BASE}/api/catalog`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = (await res.json()) as CatalogResponse;
    return data.projects.find((project) => project.sport === "Football") ?? null;
  } catch {
    return null;
  }
}

export default async function FootballMatchPage() {
  const project = await fetchProject();
  if (!project) {
    return (
      <div className="card">
        <h1 className="section-title">Football Match</h1>
        <p className="section-subtitle">Project Football introuvable.</p>
      </div>
    );
  }

  return (
    <div className="stack">
      <RunForm
        project={project}
        title="Football Match Prediction"
        description="Predire le resultat 1X2 et comparer aux baselines."
        defaults={{ mode: "match_result" }}
        locked={["mode"]}
      />
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Scoreline Distribution</h2>
          <p className="module-subtitle">Distribution 0-0 / 1-0 / 1-1 (placeholder).</p>
          <p className="section-subtitle">Enable once scoreline model is ready.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Baseline Compare</h2>
          <p className="module-subtitle">Elo or odds vs model output.</p>
          <p className="section-subtitle">Awaiting baseline integration.</p>
        </div>
      </div>
    </div>
  );
}
