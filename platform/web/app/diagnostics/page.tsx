export default function DiagnosticsPage() {
  return (
    <div className="stack">
      <div>
        <h1 className="section-title">Diagnostics</h1>
        <p className="section-subtitle">
          Suivi global: stabilite du modele, erreurs recurrentes, baselines.
        </p>
      </div>
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Stabilite modele</h2>
          <p className="module-subtitle">Drift, variance, shifts de features.</p>
          <p className="section-subtitle">Pas de diagnostic pour l'instant.</p>
        </div>
        <div className="card">
          <h2 className="module-title">Ecart baseline</h2>
          <p className="module-subtitle">Comparaison baseline vs modele principal.</p>
          <p className="section-subtitle">En attente de backtest.</p>
        </div>
      </div>
    </div>
  );
}
