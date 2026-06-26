import Link from "next/link";

const f1Modes = [
  {
    title: "Research Lab",
    eyebrow: "Prediction and experiments",
    href: "/f1/research",
    cta: "Open lab",
    summary: "Models, backtests, qualifying prediction, research library, experiment runs, and diagnostics.",
    metrics: ["Preview", "Qualifying", "Runs"],
  },
  {
    title: "Insights",
    eyebrow: "Session hub",
    href: "/f1/insights",
    cta: "Open insights",
    summary: "Live timing, session analysis, track zones, telemetry, and engineer dashboards.",
    metrics: ["Live", "Analysis", "Telemetry"],
  },
];

export default function F1ModeDashboard() {
  return (
    <div className="stack-lg">
      <section className="f1-mode-header">
        <div>
          <h1 className="page-title">F1 Platform</h1>
          <p className="page-status">Research lab and race insights stay separated, with lab tooling grouped under research.</p>
        </div>
        <div className="f1-mode-header-actions">
          <Link href="/f1/research/qualifying" className="button secondary button-sm">
            Run prediction
          </Link>
          <Link href="/f1/insights/live" className="button button-sm">
            Live dashboard
          </Link>
        </div>
      </section>

      <section className="f1-mode-grid">
        {f1Modes.map((mode) => (
          <Link href={mode.href} className="f1-mode-card" key={mode.title}>
            <span className="f1-mode-eyebrow">{mode.eyebrow}</span>
            <div className="f1-mode-card-title">
              <h2>{mode.title}</h2>
              <span>{mode.cta}</span>
            </div>
            <p>{mode.summary}</p>
            <div className="f1-mode-metrics">
              {mode.metrics.map((metric) => (
                <span key={metric}>{metric}</span>
              ))}
            </div>
          </Link>
        ))}
      </section>
    </div>
  );
}
