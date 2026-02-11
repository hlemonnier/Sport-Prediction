import { getCatalog } from "@/lib/api";
import type { CatalogProject } from "@/lib/types";
import RunForm from "@/components/RunForm";

async function fetchProject(): Promise<CatalogProject | null> {
  try {
    const data = await getCatalog();
    return data.projects.find((project) => project.sport === "F1") ?? null;
  } catch {
    return null;
  }
}

export default async function F1QualifyingPage() {
  const project = await fetchProject();
  if (!project) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h1 className="module-title">F1 Qualifying</h1>
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
        <h1 className="page-title">F1 Qualifying</h1>
        <p className="page-status">Predict the grid before the qualifying session</p>
      </div>

      <RunForm
        project={project}
        title="Run Console"
        description="Configure parameters, launch the pipeline, inspect results"
        defaults={{ mode: "qualifying" }}
        locked={["mode"]}
      />

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <h2 className="module-title">Position Distribution</h2>
          </div>
          <div className="panel-body">
            <div className="empty-state">
              <span className="empty-state-text">Awaiting data</span>
              <span className="empty-state-hint">Connect data sources to display position ranges per driver</span>
            </div>
          </div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <h2 className="module-title">Comparison</h2>
          </div>
          <div className="panel-body">
            <div className="empty-state">
              <span className="empty-state-text">No comparison available</span>
              <span className="empty-state-hint">Use the Compare page for detailed model A vs B overlay</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
