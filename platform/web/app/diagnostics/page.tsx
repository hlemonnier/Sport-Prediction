export default function DiagnosticsPage() {
  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Diagnostics</h1>
        <p className="page-status">Global tracking: model stability, recurring errors, baselines</p>
      </div>

      <div className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Model Stability</h2>
              <span className="module-subtitle">Drift, variance, feature shifts</span>
            </div>
            <span className="chip">
              <span className="chip-led amber" />
              Pending
            </span>
          </div>
          <div className="panel-body">
            <div className="data-health">
              <div className="data-health-row">
                <span className="status-dot" />
                <span className="data-health-label">Prediction drift</span>
                <span className="data-health-hint">No data yet</span>
              </div>
              <div className="data-health-row">
                <span className="status-dot" />
                <span className="data-health-label">Feature variance</span>
                <span className="data-health-hint">No data yet</span>
              </div>
              <div className="data-health-row">
                <span className="status-dot" />
                <span className="data-health-label">Distribution shift</span>
                <span className="data-health-hint">No data yet</span>
              </div>
            </div>
          </div>
          <div className="panel-footer">Run multiple iterations to track stability</div>
        </div>
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Baseline Gap</h2>
              <span className="module-subtitle">Comparison: baseline vs main model</span>
            </div>
            <span className="chip">
              <span className="chip-led amber" />
              Pending
            </span>
          </div>
          <div className="panel-body">
            <div className="data-health">
              <div className="data-health-row">
                <span className="status-dot" />
                <span className="data-health-label">Accuracy delta</span>
                <span className="data-health-hint">No data yet</span>
              </div>
              <div className="data-health-row">
                <span className="status-dot" />
                <span className="data-health-label">Calibration gap</span>
                <span className="data-health-hint">No data yet</span>
              </div>
            </div>
          </div>
          <div className="panel-footer">Awaiting backtest results</div>
        </div>
      </div>
    </div>
  );
}
