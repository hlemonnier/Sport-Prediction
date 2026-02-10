export default function F1Preview() {
  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">F1 Preview</h1>
        <p className="page-status">Weekend context and key signals before the session</p>
      </div>

      <div className="grid-two">
        {/* Left column */}
        <div className="stack">
          {/* Context panel */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">Context</h2>
                <span className="module-subtitle">Circuit, weather, history</span>
              </div>
              <span className="chip">
                <span className="chip-led amber" />
                No source
              </span>
            </div>
            <div className="panel-body">
              <div className="data-health">
                <div className="data-health-row">
                  <span className="status-dot miss" />
                  <span className="data-health-label">Circuit data</span>
                  <span className="data-health-hint">Connect FastF1 to load</span>
                </div>
                <div className="data-health-row">
                  <span className="status-dot miss" />
                  <span className="data-health-label">Weather</span>
                  <span className="data-health-hint">Requires API key</span>
                </div>
                <div className="data-health-row">
                  <span className="status-dot miss" />
                  <span className="data-health-label">Historical</span>
                  <span className="data-health-hint">Previous sessions not loaded</span>
                </div>
              </div>
            </div>
          </div>

          {/* Strength Snapshot */}
          <div className="panel">
            <div className="panel-header">
              <h2 className="module-title">Strength Snapshot</h2>
            </div>
            <div className="panel-body">
              <div className="empty-state">
                <span className="empty-state-text">Awaiting data</span>
                <span className="empty-state-hint">Team/driver proxy + uncertainty will appear here</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right column */}
        <div className="stack">
          {/* Model Signals */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">Model Signals</h2>
                <span className="module-subtitle">Top 3-5 global factors</span>
              </div>
            </div>
            <div className="panel-body">
              <div className="stack-sm">
                {Array.from({ length: 3 }).map((_, idx) => (
                  <div className="data-health-row" key={`signal-${idx}`}>
                    <span className="status-dot" />
                    <span className="data-health-label">Signal {idx + 1}</span>
                    <span className="mono" style={{ color: "var(--muted)", fontSize: 13 }}>N/A</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="panel-footer">Data source required</div>
          </div>

          {/* Lab Notes */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">Lab Notes</h2>
                <span className="module-subtitle">Your session observations</span>
              </div>
            </div>
            <div className="panel-body">
              <div className="empty-state">
                <span className="empty-state-text">No notes yet</span>
                <span className="empty-state-hint">Notes will persist per session context</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
