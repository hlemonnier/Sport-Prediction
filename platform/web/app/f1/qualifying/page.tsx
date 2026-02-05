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
        <h1 className="section-title">F1 Qualif</h1>
        <p className="section-subtitle">Projet F1 introuvable.</p>
      </div>
    );
  }

  return (
    <div className="stack">
      <div className="context-bar">
        Saison 2026 / Manche 1 / Session Qualif / Modele v0.1 / Baseline v0
      </div>
      <RunForm
        project={project}
        title="F1 Qualif"
        description="Predire la grille avant la session de qualification."
        defaults={{ mode: "qualifying" }}
        locked={["mode"]}
      />
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Distribution position</h2>
          <p className="module-subtitle">Range probable par pilote (placeholder).</p>
          <p className="section-subtitle">Connecte des donnees pour afficher les ranges.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Comparaison</h2>
          <p className="module-subtitle">Modele A vs Modele B / hier vs aujourd'hui.</p>
          <p className="section-subtitle">Utilise la page Compare pour le detail.</p>
        </div>
      </div>
    </div>
  );
}
