import Link from "next/link";

const insightModules = [
  {
    title: "Live Dashboard",
    href: "/f1/insights/live",
    status: "Timing",
    body: "Standings, sectors, tyres, race-control messages, and track position.",
  },
  {
    title: "Session Analysis",
    href: "/f1/insights/session-analysis",
    status: "Analysis",
    body: "Lap chart, stint timeline, pace analysis, predictions, and micro-sector review.",
  },
  {
    title: "Engineer Dashboard",
    href: "/f1/insights/engineer",
    status: "Telemetry",
    body: "FastF1 centerline, track zones, driver comparison, and engineering artifacts.",
  },
  {
    title: "Season Overview",
    href: "/f1/insights/season",
    status: "Season",
    body: "Round context, session winners, championship form, and race-weekend calendar.",
  },
  {
    title: "Standings",
    href: "/f1/insights/standings",
    status: "Tables",
    body: "Driver and constructor points from imported race results.",
  },
  {
    title: "Driver Ranking",
    href: "/f1/insights/driver-ranking",
    status: "Ratings",
    body: "Explainable driver performance scores from laps, sectors, and race execution.",
  },
  {
    title: "Power Units",
    href: "/f1/insights/power-units",
    status: "Telemetry",
    body: "Power-unit comparison surfaces backed by speed and deployment traces.",
  },
];

export default function F1InsightsHome() {
  return (
    <div className="stack-lg">
      <section className="f1-mode-header">
        <div>
          <h1 className="page-title">Insights</h1>
          <p className="page-status">F1 timing, session analysis, telemetry, and season context inside the standard Sport Lab interface</p>
        </div>
        <div className="f1-mode-header-actions">
          <Link href="/f1/research" className="button secondary button-sm">
            Open research lab
          </Link>
          <Link href="/f1/insights/live" className="button button-sm">
            Live dashboard
          </Link>
        </div>
      </section>

      <section className="f1-mode-grid">
        {insightModules.map((module) => (
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
