export default function F1Preview() {
  return (
    <div className="stack">
      <div>
        <h1 className="section-title">F1 Preview</h1>
        <p className="section-subtitle">
          Contexte du week-end et signaux clés avant la session.
        </p>
      </div>
      <div className="grid-two">
        <div className="card">
          <h2 className="module-title">Context</h2>
          <p className="module-subtitle">
            Circuit, météo, historiques — branche les sources pour enrichir.
          </p>
          <span className="pill">No context source connected yet.</span>
        </div>
        <div className="card">
          <h2 className="module-title">Strength Snapshot</h2>
          <p className="module-subtitle">
            Classements proxy par team/driver, avec incertitude.
          </p>
          <p className="section-subtitle">Awaiting data.</p>
        </div>
      </div>
      <div className="card">
        <h2 className="module-title">Model Signals</h2>
        <p className="module-subtitle">Top 3-5 facteurs globaux (proxy).</p>
        <div className="grid-three">
          {Array.from({ length: 3 }).map((_, idx) => (
            <div className="card" key={`signal-${idx}`}>
              <span className="stat-label">Signal {idx + 1}</span>
              <span className="stat-value">Pending</span>
              <p className="section-subtitle">Connect data to compute.</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
