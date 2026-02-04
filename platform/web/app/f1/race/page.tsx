import { API_BASE } from "@/lib/api";
import type { CatalogProject, CatalogResponse } from "@/lib/types";
import RunForm from "@/components/RunForm";

async function fetchProject(): Promise<CatalogProject | null> {
  try {
    const res = await fetch(`${API_BASE}/api/catalog`, { cache: "no-store" });
    if (!res.ok) return null;
    const data = (await res.json()) as CatalogResponse;
    return data.projects.find((project) => project.sport === "F1") ?? null;
  } catch {
    return null;
  }
}

export default async function F1RacePage() {
  const project = await fetchProject();
  if (!project) {
    return (
      <div className="card">
        <h1 className="section-title">F1 Race</h1>
        <p className="section-subtitle">Project F1 introuvable.</p>
      </div>
    );
  }

  return (
    <div className="stack">
      <RunForm
        project={project}
        title="F1 Race Prediction"
        description="Prédire le classement course avant le départ."
        defaults={{ mode: "race" }}
        locked={["mode"]}
      />
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Top 10 Focus</h2>
          <p className="module-subtitle">Synthese des positions attendues.</p>
          <p className="section-subtitle">No ranking summary yet.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Scenario Notes</h2>
          <p className="module-subtitle">Course propre vs chaotique (optionnel).</p>
          <p className="section-subtitle">Add scenario heuristics later.</p>
        </div>
      </div>
    </div>
  );
}
