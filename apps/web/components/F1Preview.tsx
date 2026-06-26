"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  F1_PLATFORM_API_BASE,
  getF1PlatformSessions,
  getF1SessionStatus,
  getF1WeatherForecast,
  getFastF1Artifacts,
  getFastF1Schedule,
  type F1SessionInfo,
  type F1SessionResolution,
  type F1SessionSummary,
  type F1WeatherForecastResponse,
  type FastF1ArtifactRecord,
  type FastF1ScheduleResponse,
} from "@/lib/f1Platform";

type PreviewData = {
  resolution: F1SessionResolution | null;
  schedule: FastF1ScheduleResponse | null;
  weather: F1WeatherForecastResponse | null;
  sessions: F1SessionSummary[];
  artifacts: FastF1ArtifactRecord[];
};

export default function F1Preview() {
  const [data, setData] = useState<PreviewData>({
    resolution: null,
    schedule: null,
    weather: null,
    sessions: [],
    artifacts: [],
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const year = new Date().getFullYear();
    const load = async () => {
      setLoading(true);
      setError(null);
      const resolutionPromise = getF1SessionStatus().catch(() => null);
      const schedulePromise = getFastF1Schedule(year).catch(() => null);
      const sessionsPromise = getF1PlatformSessions().catch(() => []);
      const artifactsPromise = getFastF1Artifacts({ limit: 80 }).then((response) => response.artifacts).catch(() => []);
      const weatherPromise = getF1WeatherForecast({ year, forecastDays: 7 }).catch(() => null);
      const [resolution, schedule, sessions, artifacts, weather] = await Promise.all([
        resolutionPromise,
        schedulePromise,
        sessionsPromise,
        artifactsPromise,
        weatherPromise,
      ]);
      if (cancelled) return;
      setData({ resolution, schedule, sessions, artifacts, weather });
      if (!schedule && !weather && !resolution) {
        setError("F1 preview data could not be loaded from the local platform API.");
      }
      setLoading(false);
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const preview = useMemo(() => buildPreviewModel(data), [data]);

  return (
    <div className="stack-lg">
      <section className="f1-mode-header">
        <div>
          <h1 className="page-title">F1 Preview</h1>
          <p className="page-status">Weekend context and key signals before the session</p>
        </div>
        <div className="f1-mode-header-actions">
          <span className="chip">
            <span className={`chip-led ${loading ? "amber" : error ? "red" : "green"}`} />
            {loading ? "Loading" : error ? "Limited" : "FastF1 + Open-Meteo"}
          </span>
          <Link href="/f1/insights/season" className="button secondary button-sm">
            Season context
          </Link>
        </div>
      </section>

      {error ? <div className="f1-inline-notice error">{error}</div> : null}

      <div className="grid-two">
        <div className="stack">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">Context</h2>
                <span className="module-subtitle">Circuit, weather, history</span>
              </div>
              <span className="chip">
                <span className={`chip-led ${preview.ready ? "green" : "amber"}`} />
                {preview.ready ? "Connected" : "Partial"}
              </span>
            </div>
            <div className="panel-body">
              <div className="f1-insight-focus">
                <strong>{preview.sessionTitle}</strong>
                <span>{preview.circuitLine}</span>
                <span>{preview.sessionTimeLine}</span>
              </div>
              <div className="data-health">
                <HealthRow
                  ok={Boolean(data.schedule && preview.nextSession)}
                  label="Circuit data"
                  hint={data.schedule ? `${preview.circuitName} from FastF1 schedule` : "FastF1 schedule pending"}
                />
                <HealthRow
                  ok={Boolean(data.weather)}
                  label="Weather"
                  hint={data.weather ? `Open-Meteo, no API key · ${preview.weatherLine}` : "Open-Meteo forecast pending"}
                />
                <HealthRow
                  ok={Boolean(preview.previousSessionCount)}
                  label="Historical"
                  hint={`${preview.previousSessionCount} previous FastF1 sessions indexed · ${data.sessions.length} local replay session${data.sessions.length === 1 ? "" : "s"}`}
                />
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <h2 className="module-title">Strength Snapshot</h2>
            </div>
            <div className="panel-body">
              <div className="grid-two">
                <MiniMetric label="Schedule" value={`${data.schedule?.roundCount ?? "-"} rounds`} detail={`${data.schedule?.sessionCount ?? 0} sessions`} />
                <MiniMetric label="Artifacts" value={String(data.artifacts.length)} detail="FastF1 local files indexed" />
                <MiniMetric label="Weather" value={formatTemperature(data.weather?.current.temperature_2m)} detail={formatWind(data.weather)} />
                <MiniMetric label="Risk" value={preview.rainRisk} detail="Open-Meteo precipitation proxy" />
              </div>
            </div>
          </div>
        </div>

        <div className="stack">
          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">Model Signals</h2>
                <span className="module-subtitle">Current preview inputs</span>
              </div>
            </div>
            <div className="panel-body">
              <div className="stack-sm">
                <SignalRow label="Circuit prior" value={preview.circuitName} detail="Resolved from FastF1 event/session metadata" />
                <SignalRow label="Weather prior" value={preview.weatherLine} detail="Free Open-Meteo forecast, no key required" />
                <SignalRow label="Historical context" value={`${preview.previousSessionCount} sessions`} detail="FastF1 calendar sessions before now" />
                <SignalRow label="Local reducer" value={`${data.sessions.length} sessions`} detail={preview.bestLocalSession} />
              </div>
            </div>
            <div className="panel-footer">Source: {F1_PLATFORM_API_BASE}</div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">Historical Coverage</h2>
                <span className="module-subtitle">Loaded vs available</span>
              </div>
            </div>
            <div className="panel-body">
              <div className="f1-calendar-list">
                {preview.recentHistoricalSessions.map((session, index) => (
                  <div className="f1-calendar-row" key={previewSessionKey(session, index)}>
                    <span>R{session.round_number ?? "-"}</span>
                    <strong>{session.session_name ?? "Session"} · {session.location ?? "-"}</strong>
                    <small>{formatDateTime(session.date_start)} · FastF1 schedule</small>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function HealthRow({ ok, label, hint }: { ok: boolean; label: string; hint: string }) {
  return (
    <div className="data-health-row">
      <span className={`status-dot ${ok ? "ok" : "warn"}`} />
      <span className="data-health-label">{label}</span>
      <span className="data-health-hint">{hint}</span>
    </div>
  );
}

function SignalRow({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="data-health-row">
      <span className="status-dot ok" />
      <span className="data-health-label">{label}</span>
      <span className="data-health-hint">{value}</span>
      <span className="mono" style={{ color: "var(--muted)", fontSize: 12 }}>{detail}</span>
    </div>
  );
}

function MiniMetric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="f1-session-resolution-cell">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function buildPreviewModel(data: PreviewData) {
  const nextSession = data.resolution?.session ?? data.resolution?.nextSession ?? null;
  const scheduleSessions = (data.schedule?.rounds ?? [])
    .flatMap((round) => round.sessions)
    .sort((a, b) => timestamp(a.date_start) - timestamp(b.date_start));
  const now = Date.now();
  const previousSessions = scheduleSessions.filter((session) => timestamp(session.date_start) < now);
  const recentHistoricalSessions = previousSessions.slice(-5).reverse();
  const bestLocal = [...data.sessions].sort((a, b) => (b.eventCount ?? 0) - (a.eventCount ?? 0))[0];
  const circuitName = data.weather?.circuit.name ?? formatSessionLocation(nextSession) ?? "Circuit pending";
  const sessionTitle = nextSession ? formatSessionTitle(nextSession) : "No FastF1 session resolved";
  const sessionTimeLine = nextSession?.date_start ? `Start ${formatDateTime(nextSession.date_start)}` : data.resolution?.message ?? "Waiting for FastF1 schedule";
  return {
    ready: Boolean(data.schedule && data.weather && nextSession),
    nextSession,
    sessionTitle,
    circuitName,
    circuitLine: data.weather
      ? `${data.weather.circuit.name} · ${data.weather.circuit.latitude.toFixed(3)}, ${data.weather.circuit.longitude.toFixed(3)}`
      : formatSessionLocation(nextSession) || "Circuit coordinates pending",
    sessionTimeLine,
    weatherLine: formatWeatherLine(data.weather),
    rainRisk: formatRainRisk(data.weather),
    previousSessionCount: previousSessions.length,
    recentHistoricalSessions,
    bestLocalSession: bestLocal ? `${bestLocal.sessionKey} · ${bestLocal.eventCount} events` : "No local replay loaded",
  };
}

function formatSessionTitle(session?: F1SessionInfo | null): string {
  if (!session) return "F1 session";
  const year = typeof session.year === "number" ? session.year : "";
  const name = session.session_name ?? session.session_type ?? "Session";
  const location = formatSessionLocation(session);
  return `${year} ${name}${location ? ` · ${location}` : ""}`.trim();
}

function formatSessionLocation(session?: F1SessionInfo | null): string {
  if (!session) return "";
  const track = session.circuit_short_name || session.location || null;
  const country = session.country_name || session.country_code || null;
  if (track && country && track !== country) return `${track}, ${country}`;
  return track || country || "";
}

function formatWeatherLine(weather?: F1WeatherForecastResponse | null): string {
  if (!weather) return "Weather pending";
  const temp = formatTemperature(weather.current.temperature_2m);
  const wind = formatWind(weather);
  return `${temp} · ${wind}`;
}

function formatTemperature(value: unknown): string {
  const parsed = numberOrNull(value);
  return parsed === null ? "-" : `${parsed.toFixed(1)} C`;
}

function formatWind(weather?: F1WeatherForecastResponse | null): string {
  const speed = numberOrNull(weather?.current.wind_speed_10m);
  const gust = numberOrNull(weather?.current.wind_gusts_10m);
  if (speed === null && gust === null) return "Wind pending";
  if (gust !== null) return `${Math.round(speed ?? 0)} km/h wind · ${Math.round(gust)} gust`;
  return `${Math.round(speed ?? 0)} km/h wind`;
}

function formatRainRisk(weather?: F1WeatherForecastResponse | null): string {
  const probability = numberOrNull(weather?.summary.maxPrecipitationProbability);
  const rain = numberOrNull(weather?.summary.maxRainMm);
  if (probability !== null) return `${Math.round(probability)}%`;
  if (rain !== null) return `${rain.toFixed(1)} mm`;
  return "-";
}

function formatDateTime(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  }).format(date);
}

function timestamp(value?: string | null): number {
  if (!value) return Number.POSITIVE_INFINITY;
  const parsed = new Date(value).getTime();
  return Number.isNaN(parsed) ? Number.POSITIVE_INFINITY : parsed;
}

function previewSessionKey(session: F1SessionInfo, index: number): string {
  return [
    session.session_key,
    session.schedule_event_key,
    session.date_start,
    session.round_number,
    session.session_name,
    index,
  ]
    .filter((part) => part !== null && part !== undefined && part !== "")
    .map(String)
    .join("::");
}

function numberOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
