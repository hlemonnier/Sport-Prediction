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

export default async function F1QualifyingPage() {
  const project = await fetchProject();
  if (!project) {
    return (
      <div className="card">
        <h1 className="section-title">F1 Qualifying</h1>
        <p className="section-subtitle">Project F1 introuvable.</p>
      </div>
    );
  }

  return (
    <div className="stack">
      <RunForm
        project={project}
        title="F1 Qualifying Prediction"
        description="Prédire la grille avant la session de qualification."
        defaults={{ mode: "qualifying" }}
        locked={["mode"]}
      />
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Position Distribution</h2>
          <p className="module-subtitle">Range probable par pilote (placeholder).</p>
          <p className="section-subtitle">Connect data to display distributions.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Compare</h2>
          <p className="module-subtitle">Model A vs Model B / yesterday vs today.</p>
          <p className="section-subtitle">Use Compare page for detailed diff.</p>
        </div>
      </div>
    </div>
  );
}
