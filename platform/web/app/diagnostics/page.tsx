export default function DiagnosticsPage() {
  return (
    <div className="stack">
      <div>
        <h1 className="section-title">Diagnostics</h1>
        <p className="section-subtitle">
          Suivi global: stabilite du modele, erreurs recurentes, baselines.
        </p>
      </div>
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Model Stability</h2>
          <p className="module-subtitle">Drift, variance, feature shifts.</p>
          <p className="section-subtitle">No diagnostics run yet.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Baseline Gap</h2>
          <p className="module-subtitle">Comparaison baseline vs modele principal.</p>
          <p className="section-subtitle">Awaiting backtest results.</p>
        </div>
      </div>
    </div>
  );
}
