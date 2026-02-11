"use client";

import { useEffect, useMemo, useState } from "react";
import {
  applyUiPreferences,
  defaultUiPreferences,
  readUiPreferences,
  type UiPreferences,
  writeUiPreferences,
} from "@/lib/uiPreferences";

function ToggleRow({
  label,
  hint,
  enabled,
  onToggle,
  onLabel = "Enabled",
  offLabel = "Disabled",
}: {
  label: string;
  hint?: string;
  enabled: boolean;
  onToggle: () => void;
  onLabel?: string;
  offLabel?: string;
}) {
  return (
    <div className="settings-row">
      <div className="settings-row-copy">
        <span className="data-health-label">{label}</span>
        {hint ? <span className="empty-state-hint">{hint}</span> : null}
      </div>
      <button
        type="button"
        className={`button button-sm ${enabled ? "" : "secondary"}`}
        onClick={onToggle}
      >
        {enabled ? onLabel : offLabel}
      </button>
    </div>
  );
}

export default function SettingsPage() {
  const [preferences, setPreferences] = useState<UiPreferences>(defaultUiPreferences);
  const [savedSnapshot, setSavedSnapshot] = useState<UiPreferences>(defaultUiPreferences);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  useEffect(() => {
    const initial = readUiPreferences();
    setPreferences(initial);
    setSavedSnapshot(initial);
    applyUiPreferences(initial);
  }, []);

  useEffect(() => {
    applyUiPreferences(preferences);
  }, [preferences]);

  useEffect(() => {
    return () => {
      applyUiPreferences(readUiPreferences());
    };
  }, []);

  const hasChanges = useMemo(
    () => JSON.stringify(preferences) !== JSON.stringify(savedSnapshot),
    [preferences, savedSnapshot]
  );

  const savePreferences = () => {
    writeUiPreferences(preferences);
    setSavedSnapshot(preferences);
    setSavedAt(new Date().toLocaleTimeString());
  };

  const resetPreferences = () => {
    setPreferences(defaultUiPreferences);
    setSavedSnapshot(defaultUiPreferences);
    writeUiPreferences(defaultUiPreferences);
    applyUiPreferences(defaultUiPreferences);
    setSavedAt(new Date().toLocaleTimeString());
  };

  return (
    <div className="stack-lg">
      <div>
        <h1 className="page-title">Settings</h1>
        <p className="page-status">
          Customize theme, accent, dashboard visibility, and sidebar behavior.
        </p>
      </div>

      <div className="grid-two">
        <section className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Appearance</h2>
              <span className="module-subtitle">Theme, accent palette, and visual density</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="stack-sm">
              <div className="settings-row">
                <div className="settings-row-copy">
                  <span className="data-health-label">Theme mode</span>
                  <span className="empty-state-hint">Choose how the interface follows light/dark mode.</span>
                </div>
                <div className="segmented">
                  <button
                    type="button"
                    className={`segmented-item ${preferences.themeMode === "system" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, themeMode: "system" }))}
                  >
                    System
                  </button>
                  <button
                    type="button"
                    className={`segmented-item ${preferences.themeMode === "light" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, themeMode: "light" }))}
                  >
                    Light
                  </button>
                  <button
                    type="button"
                    className={`segmented-item ${preferences.themeMode === "dark" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, themeMode: "dark" }))}
                  >
                    Dark
                  </button>
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-copy">
                  <span className="data-health-label">Accent color</span>
                  <span className="empty-state-hint">Pick a brand-safe accent preset.</span>
                </div>
                <div className="segmented">
                  <button
                    type="button"
                    className={`segmented-item ${preferences.accentPreset === "red" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, accentPreset: "red" }))}
                  >
                    Red
                  </button>
                  <button
                    type="button"
                    className={`segmented-item ${preferences.accentPreset === "blue" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, accentPreset: "blue" }))}
                  >
                    Blue
                  </button>
                  <button
                    type="button"
                    className={`segmented-item ${preferences.accentPreset === "emerald" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, accentPreset: "emerald" }))}
                  >
                    Emerald
                  </button>
                  <button
                    type="button"
                    className={`segmented-item ${preferences.accentPreset === "amber" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, accentPreset: "amber" }))}
                  >
                    Amber
                  </button>
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-copy">
                  <span className="data-health-label">Density</span>
                  <span className="empty-state-hint">Comfortable spacing or compact layout.</span>
                </div>
                <div className="segmented">
                  <button
                    type="button"
                    className={`segmented-item ${preferences.density === "comfortable" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, density: "comfortable" }))}
                  >
                    Comfortable
                  </button>
                  <button
                    type="button"
                    className={`segmented-item ${preferences.density === "dense" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, density: "dense" }))}
                  >
                    Dense
                  </button>
                </div>
              </div>

              <ToggleRow
                label="Background grid"
                hint="Enable or hide the global layout grid."
                enabled={preferences.showBackgroundGrid}
                onLabel="Visible"
                offLabel="Hidden"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    showBackgroundGrid: !prev.showBackgroundGrid,
                  }))
                }
              />

              <div className="settings-row slider-row">
                <div className="settings-row-copy">
                  <span className="data-health-label">Grid intensity</span>
                  <span className="empty-state-hint">Increase contrast of the background grid.</span>
                </div>
                <div className="settings-slider">
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={preferences.gridIntensity}
                    disabled={!preferences.showBackgroundGrid}
                    onChange={(event) =>
                      setPreferences((prev) => ({
                        ...prev,
                        gridIntensity: Number(event.target.value),
                      }))
                    }
                  />
                  <span className="chip">{preferences.gridIntensity}%</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Sidebar</h2>
              <span className="module-subtitle">Navigation grouping and collapse preferences</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="stack-sm">
              <ToggleRow
                label="Auto-open active group"
                hint="Expand the current section based on page route."
                enabled={preferences.autoExpandActiveGroup}
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    autoExpandActiveGroup: !prev.autoExpandActiveGroup,
                  }))
                }
              />
              <ToggleRow
                label="Auto-collapse non-active groups"
                hint="Keep only one expanded group at a time."
                enabled={preferences.autoCollapseNonActiveGroups}
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    autoCollapseNonActiveGroups: !prev.autoCollapseNonActiveGroups,
                  }))
                }
              />
              <ToggleRow
                label="Remember collapsed state"
                hint="Persist group open/closed state in local storage."
                enabled={preferences.rememberSidebarState}
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    rememberSidebarState: !prev.rememberSidebarState,
                  }))
                }
              />
              <ToggleRow
                label="Compact labels"
                hint="Use tighter label spacing and typography."
                enabled={preferences.compactSidebarLabels}
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    compactSidebarLabels: !prev.compactSidebarLabels,
                  }))
                }
              />
              <ToggleRow
                label="Sidebar visibility"
                hint="Collapse the full sidebar layout."
                enabled={!preferences.sidebarCollapsed}
                onLabel="Visible"
                offLabel="Collapsed"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    sidebarCollapsed: !prev.sidebarCollapsed,
                  }))
                }
              />

              <div className="settings-row">
                <div className="settings-row-copy">
                  <span className="data-health-label">Default sport module</span>
                  <span className="empty-state-hint">Used by dashboard CTAs for first/next run.</span>
                </div>
                <div className="segmented">
                  <button
                    type="button"
                    className={`segmented-item ${preferences.defaultSportModule === "Football" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, defaultSportModule: "Football" }))}
                  >
                    Football
                  </button>
                  <button
                    type="button"
                    className={`segmented-item ${preferences.defaultSportModule === "F1" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, defaultSportModule: "F1" }))}
                  >
                    F1
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Dashboard</h2>
              <span className="module-subtitle">Charts and high-level content behavior</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="stack-sm">
              <ToggleRow
                label="Show KPI cards"
                hint="Top summary cards with key workspace metrics."
                enabled={preferences.showKpiCards}
                onLabel="Visible"
                offLabel="Hidden"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    showKpiCards: !prev.showKpiCards,
                  }))
                }
              />
              <ToggleRow
                label="Show charts"
                hint="Run status distribution and trend cards."
                enabled={preferences.showDashboardCharts}
                onLabel="Visible"
                offLabel="Hidden"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    showDashboardCharts: !prev.showDashboardCharts,
                  }))
                }
              />

              <div className="settings-row">
                <div className="settings-row-copy">
                  <span className="data-health-label">Default chart range</span>
                  <span className="empty-state-hint">Trend chart time window.</span>
                </div>
                <div className="segmented">
                  <button
                    type="button"
                    className={`segmented-item ${preferences.dashboardChartRange === "7d" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, dashboardChartRange: "7d" }))}
                  >
                    7d
                  </button>
                  <button
                    type="button"
                    className={`segmented-item ${preferences.dashboardChartRange === "30d" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, dashboardChartRange: "30d" }))}
                  >
                    30d
                  </button>
                </div>
              </div>

              <div className="settings-row">
                <div className="settings-row-copy">
                  <span className="data-health-label">Trend chart style</span>
                  <span className="empty-state-hint">Choose between line and bar visualization.</span>
                </div>
                <div className="segmented">
                  <button
                    type="button"
                    className={`segmented-item ${preferences.dashboardChartStyle === "line" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, dashboardChartStyle: "line" }))}
                  >
                    Line
                  </button>
                  <button
                    type="button"
                    className={`segmented-item ${preferences.dashboardChartStyle === "bar" ? "active" : ""}`}
                    onClick={() => setPreferences((prev) => ({ ...prev, dashboardChartStyle: "bar" }))}
                  >
                    Bar
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Dashboard Modules</h2>
              <span className="module-subtitle">Show or hide dashboard sections</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="settings-module-grid">
              <ToggleRow
                label="Upcoming Matches"
                enabled={preferences.showUpcomingMatches}
                onLabel="Visible"
                offLabel="Hidden"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    showUpcomingMatches: !prev.showUpcomingMatches,
                  }))
                }
              />
              <ToggleRow
                label="F1 Overview"
                enabled={preferences.showF1Overview}
                onLabel="Visible"
                offLabel="Hidden"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    showF1Overview: !prev.showF1Overview,
                  }))
                }
              />
              <ToggleRow
                label="Data Health"
                enabled={preferences.showDataHealth}
                onLabel="Visible"
                offLabel="Hidden"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    showDataHealth: !prev.showDataHealth,
                  }))
                }
              />
              <ToggleRow
                label="Latest Runs"
                enabled={preferences.showLatestRuns}
                onLabel="Visible"
                offLabel="Hidden"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    showLatestRuns: !prev.showLatestRuns,
                  }))
                }
              />
              <ToggleRow
                label="Quick Compare"
                enabled={preferences.showQuickCompare}
                onLabel="Visible"
                offLabel="Hidden"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    showQuickCompare: !prev.showQuickCompare,
                  }))
                }
              />
              <ToggleRow
                label="Recent Runs"
                enabled={preferences.showRecentRuns}
                onLabel="Visible"
                offLabel="Hidden"
                onToggle={() =>
                  setPreferences((prev) => ({
                    ...prev,
                    showRecentRuns: !prev.showRecentRuns,
                  }))
                }
              />
            </div>
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="panel-body">
          <div className="row-between">
            <div className="stack-sm">
              <span className="module-title">Preferences</span>
              <span className="empty-state-hint">
                Settings are saved in local storage for this browser.
                {" "}
                {savedAt
                  ? `Last saved at ${savedAt}.`
                  : hasChanges
                    ? "You have unsaved changes."
                    : "No unsaved changes."}
              </span>
            </div>
            <div className="row">
              <button type="button" className="button secondary" onClick={resetPreferences}>
                Reset defaults
              </button>
              <button type="button" className="button" onClick={savePreferences} disabled={!hasChanges}>
                Save changes
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
