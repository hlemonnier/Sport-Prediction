type InsightCard = {
  label: string;
  value: string;
  detail: string;
};

type F1InsightsStaticPageProps = {
  title: string;
  eyebrow: string;
  description: string;
  cards: InsightCard[];
};

export default function F1InsightsStaticPage({
  title,
  eyebrow,
  description,
  cards,
}: F1InsightsStaticPageProps) {
  return (
    <div className="stack-lg">
      <section className="f1-mode-header">
        <div>
          <span className="f1-mode-eyebrow">{eyebrow}</span>
          <h1 className="page-title">{title}</h1>
          <p className="page-status">{description}</p>
        </div>
      </section>

      <section className="grid-three">
        {cards.map((card) => (
          <div className="panel" key={card.label}>
            <div className="panel-header">
              <h2 className="module-title">{card.label}</h2>
            </div>
            <div className="panel-body">
              <div className="kpi-value">{card.value}</div>
              <p className="kpi-subtext">{card.detail}</p>
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
