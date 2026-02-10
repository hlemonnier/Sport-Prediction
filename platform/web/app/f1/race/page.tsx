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
      <div className="panel">
        <div className="panel-header">
          <h1 className="module-title">F1 Race</h1>
        </div>
        <div className="panel-body">
          <div className="empty-state">
            <span className="empty-state-text">F1 project not found in catalog</span>
            <span className="empty-state-hint">Check that the backend is running and the F1 project is registered</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">F1 Race</h1>
        <p className="page-status">Predict the race classification before lights out</p>
      </div>

      <RunForm
        project={project}
        title="Run Console"
        description="Configure parameters, launch the pipeline, inspect results"
        defaults={{ mode: "race" }}
        locked={["mode"]}
      />

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Top 10 Focus</h2>
              <span className="module-subtitle">Expected position summary</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="empty-state">
              <span className="empty-state-text">No summary available</span>
              <span className="empty-state-hint">Run a prediction to populate this panel</span>
            </div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Scenario Notes</h2>
              <span className="module-subtitle">Clean race vs chaotic</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="empty-state">
              <span className="empty-state-text">No heuristics configured</span>
              <span className="empty-state-hint">Add scenario rules to enable this panel</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
