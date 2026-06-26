import Link from "next/link";

const researchModules = [
  {
    title: "Weekend Preview",
    href: "/f1/research/preview",
    status: "Context",
    body: "Circuit, weather, history, and pre-session model signals.",
  },
  {
    title: "Qualifying Prediction",
    href: "/f1/research/qualifying",
    status: "Run console",
    body: "Launch the F1 pipeline and inspect position distributions.",
  },
  {
    title: "Prediction Review",
    href: "/f1/research/review",
    status: "Backtest",
    body: "Post-session gaps, ranking metrics, and model comparison.",
  },
  {
    title: "Experiment Runs",
    href: "/runs",
    status: "Lab",
    body: "Inspect local F1 pipeline runs, artifacts, and execution state.",
  },
  {
    title: "Sweeps",
    href: "/sweeps",
    status: "Lab",
    body: "Track parameter sweeps, model variants, and search results.",
  },
  {
    title: "Compare",
    href: "/compare",
    status: "Lab",
    body: "Compare runs and model outputs before promoting research changes.",
  },
  {
    title: "Diagnostics",
    href: "/diagnostics",
    status: "Lab",
    body: "Check service health, data availability, and pipeline readiness.",
  },
  {
    title: "Research Library",
    href: "/research",
    status: "Library",
    body: "Keep papers, notebooks, and references attached to the lab workflow.",
  },
];

export default function F1ResearchHome() {
  return (
    <div className="stack-lg">
      <section className="f1-mode-header">
        <div>
          <h1 className="page-title">Research Lab</h1>
          <p className="page-status">F1 prediction workflows, experiments, diagnostics, and research library in one section</p>
        </div>
        <div className="f1-mode-header-actions">
          <Link href="/research" className="button secondary button-sm">
            Research library
          </Link>
          <Link href="/f1/insights" className="button button-sm">
            Open insights
          </Link>
        </div>
      </section>

      <section className="f1-mode-grid">
        {researchModules.map((module) => (
          <Link href={module.href} className="f1-mode-card compact" key={module.title}>
            <span className="f1-mode-eyebrow">{module.status}</span>
            <div className="f1-mode-card-title">
              <h2>{module.title}</h2>
              <span>Open</span>
            </div>
            <p>{module.body}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
