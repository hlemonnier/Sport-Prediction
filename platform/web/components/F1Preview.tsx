export default function F1Preview() {
  return (
    <div className="stack">
      <div className="context-bar">
        Saison 2026 / Manche 1 / Session Apercu / Modele v0.1 / Baseline v0
      </div>
      <div>
        <h1 className="section-title">F1 Apercu</h1>
        <p className="section-subtitle">
          Contexte du week-end et signaux cles avant la session.
        </p>
      </div>
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Contexte</h2>
          <p className="module-subtitle">Circuit, meteo, historique.</p>
          <span className="pill">Aucune source connectee</span>
        </div>
        <div className="card">
          <h2 className="module-title">Snapshot force</h2>
          <p className="module-subtitle">Proxy team/driver + incertitude.</p>
          <p className="section-subtitle">En attente de donnees.</p>
        </div>
      </div>
      <div className="card">
        <h2 className="module-title">Signaux modele</h2>
        <p className="module-subtitle">Top 3-5 facteurs globaux.</p>
        <div className="grid-three">
          {Array.from({ length: 3 }).map((_, idx) => (
            <div className="card" key={`signal-${idx}`}>
              <span className="stat-label">Signal {idx + 1}</span>
              <span className="stat-value">N/A</span>
              <p className="section-subtitle">Data manquante.</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
