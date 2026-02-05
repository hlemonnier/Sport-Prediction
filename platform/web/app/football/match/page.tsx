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
        <h1 className="section-title">Match Football</h1>
        <p className="section-subtitle">Projet Football introuvable.</p>
      </div>
    );
  }

  return (
    <div className="stack">
      <div className="context-bar">
        Ligue EPL / Saison 2026 / Match prochain / Modele v0.1 / Baseline v0
      </div>
      <RunForm
        project={project}
        title="Match Football"
        description="Predire le resultat 1X2 et comparer aux baselines."
        defaults={{ mode: "match_result" }}
        locked={["mode"]}
      />
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Distribution score</h2>
          <p className="module-subtitle">0-0 / 1-0 / 1-1 (placeholder).</p>
          <p className="section-subtitle">Active quand le modele scoreline est pret.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Baseline</h2>
          <p className="module-subtitle">Elo ou odds vs modele.</p>
          <p className="section-subtitle">Baseline en attente.</p>
        </div>
      </div>
    </div>
  );
}
