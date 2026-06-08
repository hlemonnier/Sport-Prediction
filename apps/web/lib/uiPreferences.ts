export const UI_PREFERENCES_STORAGE_KEY = "settings:ui-preferences";
export const UI_PREFERENCES_EVENT = "ui-preferences:changed";
export const USER_SAVINGS_STORAGE_KEY = "settings:user-savings";
export const USER_SAVINGS_EVENT = "user-savings:changed";

export type ThemeMode = "system" | "light" | "dark";
export type AccentPreset = "red" | "blue" | "emerald" | "amber";
export type DensityMode = "comfortable" | "dense";
export type SportModule = "F1" | "Football";
export type ChartRange = "7d" | "30d";
export type ChartStyle = "line" | "bar";

export type UiPreferences = {
  themeMode: ThemeMode;
  accentPreset: AccentPreset;
  density: DensityMode;
  showBackgroundGrid: boolean;
  gridIntensity: number;
  showKpiCards: boolean;
  showDashboardCharts: boolean;
  showUpcomingMatches: boolean;
  showF1Overview: boolean;
  showDataHealth: boolean;
  showLatestRuns: boolean;
  showQuickCompare: boolean;
  showRecentRuns: boolean;
  autoExpandActiveGroup: boolean;
  autoCollapseNonActiveGroups: boolean;
  rememberSidebarState: boolean;
  compactSidebarLabels: boolean;
  sidebarCollapsed: boolean;
  defaultSportModule: SportModule;
  dashboardChartRange: ChartRange;
  dashboardChartStyle: ChartStyle;
};

export type UserSavings = {
  bankroll: number;
  monthlySavingsTarget: number;
  reserveBalance: number;
  defaultStake: number;
  maxStakePercent: number;
  autoSaveProfit: boolean;
};

export const defaultUiPreferences: UiPreferences = {
  themeMode: "system",
  accentPreset: "red",
  density: "comfortable",
  showBackgroundGrid: true,
  gridIntensity: 25,
  showKpiCards: true,
  showDashboardCharts: true,
  showUpcomingMatches: true,
  showF1Overview: true,
  showDataHealth: true,
  showLatestRuns: true,
  showQuickCompare: true,
  showRecentRuns: true,
  autoExpandActiveGroup: true,
  autoCollapseNonActiveGroups: false,
  rememberSidebarState: true,
  compactSidebarLabels: false,
  sidebarCollapsed: false,
  defaultSportModule: "Football",
  dashboardChartRange: "7d",
  dashboardChartStyle: "line",
};

export const defaultUserSavings: UserSavings = {
  bankroll: 0,
  monthlySavingsTarget: 0,
  reserveBalance: 0,
  defaultStake: 10,
  maxStakePercent: 2,
  autoSaveProfit: true,
};

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function coerceNumber(
  value: unknown,
  fallback: number,
  min = 0,
  max = Number.POSITIVE_INFINITY
): number {
  const asNumber =
    typeof value === "number" && Number.isFinite(value)
      ? value
      : typeof value === "string"
        ? Number.parseFloat(value)
        : Number.NaN;

  if (!Number.isFinite(asNumber)) {
    return fallback;
  }

  return Math.min(max, Math.max(min, asNumber));
}

function sanitizePreferences(raw: Partial<UiPreferences>): UiPreferences {
  const themeMode =
    raw.themeMode === "light" || raw.themeMode === "dark" || raw.themeMode === "system"
      ? raw.themeMode
      : defaultUiPreferences.themeMode;
  const accentPreset =
    raw.accentPreset === "blue" ||
    raw.accentPreset === "emerald" ||
    raw.accentPreset === "amber" ||
    raw.accentPreset === "red"
      ? raw.accentPreset
      : defaultUiPreferences.accentPreset;
  const density = raw.density === "dense" || raw.density === "comfortable"
    ? raw.density
    : defaultUiPreferences.density;
  const defaultSportModule = raw.defaultSportModule === "F1" || raw.defaultSportModule === "Football"
    ? raw.defaultSportModule
    : defaultUiPreferences.defaultSportModule;
  const dashboardChartRange = raw.dashboardChartRange === "30d" || raw.dashboardChartRange === "7d"
    ? raw.dashboardChartRange
    : defaultUiPreferences.dashboardChartRange;
  const dashboardChartStyle = raw.dashboardChartStyle === "bar" || raw.dashboardChartStyle === "line"
    ? raw.dashboardChartStyle
    : defaultUiPreferences.dashboardChartStyle;
  const gridIntensity =
    typeof raw.gridIntensity === "number" && Number.isFinite(raw.gridIntensity)
      ? Math.min(100, Math.max(0, raw.gridIntensity))
      : defaultUiPreferences.gridIntensity;

  return {
    ...defaultUiPreferences,
    ...raw,
    themeMode,
    accentPreset,
    density,
    gridIntensity,
    defaultSportModule,
    dashboardChartRange,
    dashboardChartStyle,
    showBackgroundGrid: raw.showBackgroundGrid ?? defaultUiPreferences.showBackgroundGrid,
    showKpiCards: raw.showKpiCards ?? defaultUiPreferences.showKpiCards,
    showDashboardCharts: raw.showDashboardCharts ?? defaultUiPreferences.showDashboardCharts,
    showUpcomingMatches: raw.showUpcomingMatches ?? defaultUiPreferences.showUpcomingMatches,
    showF1Overview: raw.showF1Overview ?? defaultUiPreferences.showF1Overview,
    showDataHealth: raw.showDataHealth ?? defaultUiPreferences.showDataHealth,
    showLatestRuns: raw.showLatestRuns ?? defaultUiPreferences.showLatestRuns,
    showQuickCompare: raw.showQuickCompare ?? defaultUiPreferences.showQuickCompare,
    showRecentRuns: raw.showRecentRuns ?? defaultUiPreferences.showRecentRuns,
    autoExpandActiveGroup: raw.autoExpandActiveGroup ?? defaultUiPreferences.autoExpandActiveGroup,
    autoCollapseNonActiveGroups:
      raw.autoCollapseNonActiveGroups ?? defaultUiPreferences.autoCollapseNonActiveGroups,
    rememberSidebarState: raw.rememberSidebarState ?? defaultUiPreferences.rememberSidebarState,
    compactSidebarLabels: raw.compactSidebarLabels ?? defaultUiPreferences.compactSidebarLabels,
    sidebarCollapsed: raw.sidebarCollapsed ?? defaultUiPreferences.sidebarCollapsed,
  };
}

function sanitizeUserSavings(raw: Partial<UserSavings>): UserSavings {
  return {
    bankroll: coerceNumber(raw.bankroll, defaultUserSavings.bankroll, 0),
    monthlySavingsTarget: coerceNumber(
      raw.monthlySavingsTarget,
      defaultUserSavings.monthlySavingsTarget,
      0
    ),
    reserveBalance: coerceNumber(raw.reserveBalance, defaultUserSavings.reserveBalance, 0),
    defaultStake: coerceNumber(raw.defaultStake, defaultUserSavings.defaultStake, 0),
    maxStakePercent: coerceNumber(raw.maxStakePercent, defaultUserSavings.maxStakePercent, 0, 100),
    autoSaveProfit:
      typeof raw.autoSaveProfit === "boolean"
        ? raw.autoSaveProfit
        : defaultUserSavings.autoSaveProfit,
  };
}

export function coerceUiPreferences(raw: unknown): UiPreferences {
  if (!isObject(raw)) {
    return defaultUiPreferences;
  }

  return sanitizePreferences(raw as Partial<UiPreferences>);
}

export function coerceUserSavings(raw: unknown): UserSavings {
  if (!isObject(raw)) {
    return defaultUserSavings;
  }

  return sanitizeUserSavings(raw as Partial<UserSavings>);
}

export function readUiPreferences(): UiPreferences {
  if (typeof window === "undefined") return defaultUiPreferences;
  try {
    const raw = window.localStorage.getItem(UI_PREFERENCES_STORAGE_KEY);
    if (!raw) return defaultUiPreferences;
    return coerceUiPreferences(JSON.parse(raw));
  } catch {
    return defaultUiPreferences;
  }
}

export function readUserSavings(): UserSavings {
  if (typeof window === "undefined") return defaultUserSavings;
  try {
    const raw = window.localStorage.getItem(USER_SAVINGS_STORAGE_KEY);
    if (!raw) return defaultUserSavings;
    return coerceUserSavings(JSON.parse(raw));
  } catch {
    return defaultUserSavings;
  }
}

export function resolveThemeMode(mode: ThemeMode): "light" | "dark" {
  if (mode === "light" || mode === "dark") return mode;
  if (typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    return "dark";
  }
  return "light";
}

export function applyUiPreferences(prefs: UiPreferences): void {
  if (typeof document === "undefined") return;
  const body = document.body;

  const resolvedTheme = resolveThemeMode(prefs.themeMode);
  body.classList.remove("theme-light", "theme-dark");
  body.classList.add(resolvedTheme === "dark" ? "theme-dark" : "theme-light");

  body.classList.remove("accent-red", "accent-blue", "accent-emerald", "accent-amber");
  body.classList.add(`accent-${prefs.accentPreset}`);

  body.classList.toggle("dense", prefs.density === "dense");
  body.classList.toggle("no-grid", !prefs.showBackgroundGrid);
  body.classList.toggle("sidebar-collapsed", prefs.sidebarCollapsed);
  body.classList.toggle("sidebar-compact", prefs.compactSidebarLabels);

  const gridOpacity = (prefs.gridIntensity / 100) * 0.12;
  body.style.setProperty("--grid-opacity", gridOpacity.toFixed(3));
}

export function writeUiPreferences(prefs: UiPreferences): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(UI_PREFERENCES_STORAGE_KEY, JSON.stringify(prefs));
  window.dispatchEvent(new CustomEvent(UI_PREFERENCES_EVENT));
}

export function writeUserSavings(savings: UserSavings): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(USER_SAVINGS_STORAGE_KEY, JSON.stringify(savings));
  window.dispatchEvent(new CustomEvent(USER_SAVINGS_EVENT));
}

export function updateUiPreferences(partial: Partial<UiPreferences>): UiPreferences {
  const next = sanitizePreferences({
    ...readUiPreferences(),
    ...partial,
  });
  applyUiPreferences(next);
  writeUiPreferences(next);
  return next;
}

export function subscribeUiPreferences(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  const handleStorage = (event: StorageEvent) => {
    if (event.key && event.key !== UI_PREFERENCES_STORAGE_KEY) return;
    onChange();
  };
  window.addEventListener("storage", handleStorage);
  window.addEventListener(UI_PREFERENCES_EVENT, onChange as EventListener);
  return () => {
    window.removeEventListener("storage", handleStorage);
    window.removeEventListener(UI_PREFERENCES_EVENT, onChange as EventListener);
  };
}

export function subscribeUserSavings(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  const handleStorage = (event: StorageEvent) => {
    if (event.key && event.key !== USER_SAVINGS_STORAGE_KEY) return;
    onChange();
  };
  window.addEventListener("storage", handleStorage);
  window.addEventListener(USER_SAVINGS_EVENT, onChange as EventListener);
  return () => {
    window.removeEventListener("storage", handleStorage);
    window.removeEventListener(USER_SAVINGS_EVENT, onChange as EventListener);
  };
}
