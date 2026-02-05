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
        <h1 className="section-title">F1 Course</h1>
        <p className="section-subtitle">Projet F1 introuvable.</p>
      </div>
    );
  }

  return (
    <div className="stack">
      <div className="context-bar">
        Saison 2026 / Manche 1 / Session Course / Modele v0.1 / Baseline v0
      </div>
      <RunForm
        project={project}
        title="F1 Course"
        description="Predire le classement course avant le depart."
        defaults={{ mode: "race" }}
        locked={["mode"]}
      />
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Focus top 10</h2>
          <p className="module-subtitle">Synthese des positions attendues.</p>
          <p className="section-subtitle">Pas de resume pour l'instant.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Notes scenarios</h2>
          <p className="module-subtitle">Course propre vs chaotique (optionnel).</p>
          <p className="section-subtitle">Ajouter des heuristiques plus tard.</p>
        </div>
      </div>
    </div>
  );
}
