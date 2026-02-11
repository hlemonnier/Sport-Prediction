import { getCatalog } from "@/lib/api";
import type { CatalogProject } from "@/lib/types";
import RunForm from "@/components/RunForm";

async function fetchProject(): Promise<CatalogProject | null> {
  try {
    const data = await getCatalog();
    return data.projects.find((project) => project.sport === "Football") ?? null;
  } catch {
    return null;
  }
}

export default async function FootballMatchPage() {
  const project = await fetchProject();
  if (!project) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h1 className="module-title">Football Match</h1>
        </div>
        <div className="panel-body">
          <div className="empty-state">
            <span className="empty-state-text">Football project not found in catalog</span>
            <span className="empty-state-hint">Check that the backend is running and the Football project is registered</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Football Match</h1>
        <p className="page-status">Predict 1X2 outcome and compare to baselines</p>
      </div>

      <RunForm
        project={project}
        title="Run Console"
        description="Configure match parameters, launch prediction, inspect results"
        defaults={{ mode: "match_result" }}
        locked={["mode"]}
      />

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Score Distribution</h2>
              <span className="module-subtitle">0-0 / 1-0 / 1-1 probabilities</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="empty-state">
              <span className="empty-state-text">Scoreline model not ready</span>
              <span className="empty-state-hint">Enable scoreline predictions to populate this panel</span>
            </div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Baseline</h2>
              <span className="module-subtitle">Elo or odds vs model</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="empty-state">
              <span className="empty-state-text">No baseline configured</span>
              <span className="empty-state-hint">Set up Elo or market odds as a reference</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
