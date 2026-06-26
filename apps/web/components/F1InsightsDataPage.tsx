"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  F1_SEASON_API_BASE,
  F1_PLATFORM_API_BASE,
  getFastF1Schedule,
  getF1PlatformSessions,
  getF1PlatformSnapshot,
  getF1SeasonSummary,
  getF1SessionStatus,
  type F1DriverState,
  type F1SeasonConstructorRow,
  type F1SeasonSummaryResponse,
  type F1SessionResolution,
  type F1SessionSnapshot,
  type F1SessionSummary,
  type FastF1ScheduleResponse,
  type FastF1ScheduleRound,
} from "@/lib/f1Platform";

type F1InsightsKind = "season" | "standings" | "driver-ranking" | "power-units";

type F1InsightsDataPageProps = {
  kind: F1InsightsKind;
};

type LoadedInsightsData = {
  resolution: F1SessionResolution | null;
  schedule: FastF1ScheduleResponse | null;
  seasonSummary: F1SeasonSummaryResponse | null;
  sessions: F1SessionSummary[];
  snapshot: F1SessionSnapshot | null;
};

const PAGE_COPY: Record<F1InsightsKind, { eyebrow: string; title: string; description: string }> = {
  season: {
    eyebrow: "FastF1 schedule",
    title: "Season Overview",
    description: "Calendar, current weekend context, and upcoming sessions from the free FastF1 schedule path.",
  },
  standings: {
    eyebrow: "Season standings",
    title: "Standings",
    description: "Driver and constructor championship tables from the current Jolpica Ergast-compatible season feed.",
  },
  "driver-ranking": {
    eyebrow: "Explainable rating",
    title: "Driver Ranking",
    description: "A local performance score built from position, lap pace, speed trace, and model confidence.",
  },
  "power-units": {
    eyebrow: "Telemetry proxy",
    title: "Power Units",
    description: "Power-unit group comparison using current speed and deployment traces from the reducer.",
  },
};

const POINTS_BY_POSITION = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1];

export default function F1InsightsDataPage({ kind }: F1InsightsDataPageProps) {
  const copy = PAGE_COPY[kind];
  const [data, setData] = useState<LoadedInsightsData>({
    resolution: null,
    schedule: null,
    seasonSummary: null,
    sessions: [],
    snapshot: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const year = new Date().getFullYear();
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const resolutionPromise = getF1SessionStatus().catch(() => null);
        const schedulePromise = getFastF1Schedule(year).catch(() => null);
        const seasonSummaryPromise = getF1SeasonSummary(year).catch(() => null);
        const sessions = await getF1PlatformSessions().catch(() => []);
        const sessionToOpen = selectBestSession(sessions);
        const snapshotPromise = sessionToOpen ? getF1PlatformSnapshot(sessionToOpen.sessionKey).catch(() => null) : Promise.resolve(null);
        const [resolution, schedule, seasonSummary, snapshot] = await Promise.all([
          resolutionPromise,
          schedulePromise,
          seasonSummaryPromise,
          snapshotPromise,
        ]);
        if (cancelled) return;
        setData({ resolution, schedule, seasonSummary, sessions, snapshot });
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unable to load F1 insights data");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const content = useMemo(() => {
    if (kind === "season") return <SeasonView data={data} />;
    if (kind === "standings") return <StandingsView seasonSummary={data.seasonSummary} snapshot={data.snapshot} />;
    if (kind === "driver-ranking") return <DriverRankingView snapshot={data.snapshot} />;
    return <PowerUnitsView snapshot={data.snapshot} />;
  }, [data, kind]);

  return (
    <div className="stack-lg">
      <section className="f1-mode-header">
        <div>
          <span className="f1-mode-eyebrow">{copy.eyebrow}</span>
          <h1 className="page-title">{copy.title}</h1>
          <p className="page-status">{copy.description}</p>
        </div>
        <div className="f1-mode-header-actions">
          <span className="status-item">
            <span className={`status-dot ${error ? "miss" : loading ? "warn" : "ok"}`} />
            {loading ? "Loading" : error ? "Limited" : "Live local"}
          </span>
          <span className="status-item">{data.resolution?.source ?? "resolver pending"}</span>
        </div>
      </section>

      {error ? <div className="f1-inline-notice error">{error}</div> : null}
      {content}

      <section className="status-strip">
        <span className="status-item">Season API {F1_SEASON_API_BASE}</span>
        <span className="status-item">Runtime {F1_PLATFORM_API_BASE}</span>
        <span className="status-item">Sessions {data.sessions.length}</span>
        <span className="status-item">Snapshot {data.snapshot?.sessionKey ?? "pending"}</span>
        <span className="status-item">Schedule {data.schedule ? `${data.schedule.roundCount} rounds` : "pending"}</span>
      </section>
    </div>
  );
}

function SeasonView({ data }: { data: LoadedInsightsData }) {
  const schedule = data.schedule;
  const nextSession = data.resolution?.nextSession ?? data.resolution?.session ?? null;
  const sessions = schedule ? flattenSchedule(schedule) : [];
  const now = Date.now();
  const completed = sessions.filter((session) => timestamp(session.date_end ?? session.date_start) < now).length;
  const upcoming = sessions.filter((session) => timestamp(session.date_start) >= now).slice(0, 10);
  const currentRound = schedule?.rounds.find((round) =>
    round.sessions.some((session) => session.session_key === nextSession?.session_key)
  );

  return (
    <>
      <section className="grid-four">
        <MetricCard label="Season" value={String(schedule?.year ?? new Date().getFullYear())} detail={schedule?.source ?? "FastF1 schedule"} />
        <MetricCard label="Rounds" value={String(schedule?.roundCount ?? "-")} detail={`${schedule?.sessionCount ?? 0} timed sessions`} />
        <MetricCard label="Completed" value={String(completed)} detail="Sessions before current clock" />
        <MetricCard label="Next" value={nextSession?.session_name ?? "-"} detail={formatSessionLocation(nextSession) || currentRound?.eventName || "Schedule pending"} />
      </section>

      <section className="grid-two">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Current Weekend</h2>
              <span className="module-subtitle">{currentRound?.eventName ?? "Next scheduled event"}</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="f1-insight-focus">
              <strong>{nextSession ? formatSessionTitle(nextSession) : "No session resolved"}</strong>
              <span>{nextSession?.date_start ? formatDateTime(nextSession.date_start) : "Waiting for schedule"}</span>
              <span>{data.resolution?.message ?? "FastF1 schedule resolver pending."}</span>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div className="panel-header-left">
              <h2 className="module-title">Calendar Coverage</h2>
              <span className="module-subtitle">FastF1 rounds loaded locally</span>
            </div>
          </div>
          <div className="panel-body">
            <div className="f1-calendar-list">
              {(schedule?.rounds ?? []).slice(0, 8).map((round, index) => (
                <div className="f1-calendar-row" key={scheduleRoundKey(round, index)}>
                  <span>R{round.roundNumber ?? "-"}</span>
                  <strong>{round.eventName ?? "Event"}</strong>
                  <small>{round.location ?? "-"} · {round.sessions.length} sessions</small>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <ScheduleTable sessions={upcoming} />
    </>
  );
}

function StandingsView({
  seasonSummary,
  snapshot,
}: {
  seasonSummary: F1SeasonSummaryResponse | null;
  snapshot: F1SessionSnapshot | null;
}) {
  const driverRows = buildDriverStandings(seasonSummary, snapshot);
  const constructorRows = buildConstructorStandings(seasonSummary?.constructorStandings, driverRows);
  const sourceLabel = seasonSummary?.source ?? snapshot?.source ?? "pending";
  const sourceDetail = seasonSummary?.round ? `${seasonSummary.year} Round ${seasonSummary.round}` : `Seq ${snapshot?.seq ?? 0}`;

  return (
    <>
      <section className="grid-four">
        <MetricCard label="Drivers" value={String(driverRows.length)} detail={seasonSummary ? "Championship table" : "Current reducer fallback"} />
        <MetricCard label="Leader" value={driverRows[0]?.acronym ?? "-"} detail={driverRows[0]?.teamName ?? "No reduced session"} />
        <MetricCard label="Constructors" value={String(constructorRows.length)} detail={seasonSummary ? "Official table" : "Aggregated from driver rows"} />
        <MetricCard label="Source" value={sourceLabel} detail={sourceDetail} />
      </section>

      <section className="grid-two">
        <TablePanel title="Driver Standings" subtitle={seasonSummary ? "Current season championship order" : "Fallback session points from current classification"}>
          <table className="table">
            <thead>
              <tr>
                <th>Pos</th>
                <th>Driver</th>
                <th>Team</th>
                <th>Pts</th>
                <th>Wins</th>
                <th>Code</th>
              </tr>
            </thead>
            <tbody>
              {driverRows.map((row) => (
                <tr key={row.driverNumber}>
                  <td>{row.position ?? "-"}</td>
                  <td><strong>{row.driverName ?? row.acronym}</strong></td>
                  <td>{row.teamName}</td>
                  <td>{formatPoints(row.points)}</td>
                  <td>{row.wins ?? "-"}</td>
                  <td>{row.acronym}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TablePanel>

        <TablePanel title="Constructor Standings" subtitle={seasonSummary ? "Current constructor championship order" : "Team aggregation from active result set"}>
          <table className="table">
            <thead>
              <tr>
                <th>Team</th>
                <th>Pts</th>
                <th>Wins</th>
                <th>Drivers</th>
              </tr>
            </thead>
            <tbody>
              {constructorRows.map((row) => (
                <tr key={row.teamName}>
                  <td><strong>{row.teamName}</strong></td>
                  <td>{formatPoints(row.points)}</td>
                  <td>{row.wins ?? "-"}</td>
                  <td>{row.drivers.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TablePanel>
      </section>
    </>
  );
}

function DriverRankingView({ snapshot }: { snapshot: F1SessionSnapshot | null }) {
  const rankings = buildDriverRankings(snapshot);

  return (
    <>
      <section className="grid-four">
        <MetricCard label="Rated" value={String(rankings.length)} detail="Drivers in reducer snapshot" />
        <MetricCard label="Top Score" value={rankings[0] ? rankings[0].score.toFixed(1) : "-"} detail={rankings[0]?.label ?? "No driver data"} />
        <MetricCard label="Pace Base" value={formatLapTime(rankings[0]?.bestLap)} detail="Best available lap" />
        <MetricCard label="Model" value="Local" detail="Position, pace, speed, prediction confidence" />
      </section>

      <TablePanel title="Explainable Driver Ranking" subtitle="Transparent score, not a hidden black-box rating">
        <table className="table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Driver</th>
              <th>Team</th>
              <th>Score</th>
              <th>Position</th>
              <th>Pace</th>
              <th>Signal</th>
            </tr>
          </thead>
          <tbody>
            {rankings.map((row, index) => (
              <tr key={row.driverNumber}>
                <td>{index + 1}</td>
                <td><strong>{row.label}</strong></td>
                <td>{row.teamName}</td>
                <td>
                  <div className="f1-score-cell">
                    <span>{row.score.toFixed(1)}</span>
                    <span className="f1-score-bar"><span style={{ width: `${Math.max(4, row.score)}%` }} /></span>
                  </div>
                </td>
                <td>{row.position ?? "-"}</td>
                <td>{formatLapTime(row.bestLap)}</td>
                <td>{row.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TablePanel>
    </>
  );
}

function PowerUnitsView({ snapshot }: { snapshot: F1SessionSnapshot | null }) {
  const groups = buildPowerUnitGroups(snapshot);

  return (
    <>
      <section className="grid-four">
        <MetricCard label="Groups" value={String(groups.length)} detail="Team/supplier buckets" />
        <MetricCard label="Top Speed" value={groups[0] ? `${Math.round(groups[0].maxSpeed)} km/h` : "-"} detail={groups[0]?.supplier ?? "No speed trace"} />
        <MetricCard label="DRS Avg" value={groups[0] ? groups[0].avgDrs.toFixed(1) : "-"} detail="Deployment proxy" />
        <MetricCard label="Source" value={snapshot?.source ?? "pending"} detail="Car-data reducer state" />
      </section>

      <TablePanel title="Power-Unit Telemetry Proxy" subtitle="Grouped by mapped supplier and current speed/deployment traces">
        <table className="table">
          <thead>
            <tr>
              <th>Supplier</th>
              <th>Teams</th>
              <th>Drivers</th>
              <th>Avg Speed</th>
              <th>Max Speed</th>
              <th>DRS</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((group) => (
              <tr key={group.supplier}>
                <td><strong>{group.supplier}</strong></td>
                <td>{group.teams.join(", ")}</td>
                <td>{group.drivers.join(", ")}</td>
                <td>{Math.round(group.avgSpeed)} km/h</td>
                <td>
                  <div className="f1-score-cell">
                    <span>{Math.round(group.maxSpeed)} km/h</span>
                    <span className="f1-score-bar"><span style={{ width: `${Math.min(100, (group.maxSpeed / 335) * 100)}%` }} /></span>
                  </div>
                </td>
                <td>{group.avgDrs.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TablePanel>
    </>
  );
}

function ScheduleTable({ sessions }: { sessions: F1ScheduleSession[] }) {
  return (
    <TablePanel title="Upcoming Sessions" subtitle="Next FastF1 calendar sessions">
      <table className="table">
        <thead>
          <tr>
            <th>Round</th>
            <th>Session</th>
            <th>Location</th>
            <th>Start</th>
            <th>Source Key</th>
          </tr>
        </thead>
        <tbody>
          {sessions.map((session, index) => (
            <tr key={scheduleSessionKey(session, index)}>
              <td>{session.round_number ?? "-"}</td>
              <td><strong>{session.session_name ?? "Session"}</strong></td>
              <td>{formatSessionLocation(session)}</td>
              <td>{formatDateTime(session.date_start)}</td>
              <td>{session.session_key}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </TablePanel>
  );
}

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h2 className="module-title">{label}</h2>
      </div>
      <div className="panel-body">
        <div className="kpi-value">{value}</div>
        <p className="kpi-subtext">{detail}</p>
      </div>
    </div>
  );
}

function TablePanel({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-header-left">
          <h2 className="module-title">{title}</h2>
          <span className="module-subtitle">{subtitle}</span>
        </div>
      </div>
      <div className="panel-body panel-body-dense">
        <div className="f1-table-wrap">{children}</div>
      </div>
    </div>
  );
}

type F1ScheduleSession = NonNullable<FastF1ScheduleRound["sessions"][number]>;

function selectBestSession(sessions: F1SessionSummary[]): F1SessionSummary | null {
  return [...sessions].sort((a, b) => {
    const eventDelta = (b.eventCount ?? 0) - (a.eventCount ?? 0);
    if (eventDelta !== 0) return eventDelta;
    return (b.drivers ?? 0) - (a.drivers ?? 0);
  })[0] ?? null;
}

function flattenSchedule(schedule: FastF1ScheduleResponse): F1ScheduleSession[] {
  return schedule.rounds
    .flatMap((round) => round.sessions)
    .sort((a, b) => timestamp(a.date_start) - timestamp(b.date_start));
}

function scheduleRoundKey(round: FastF1ScheduleRound, index: number): string {
  return [
    round.scheduleKey,
    round.officialEventName,
    round.eventDate,
    round.roundNumber,
    round.eventName,
    index,
  ]
    .filter((part) => part !== null && part !== undefined && part !== "")
    .map(String)
    .join("::");
}

function scheduleSessionKey(session: F1ScheduleSession, index: number): string {
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

type DriverStandingRow = {
  driverNumber: number;
  acronym: string;
  driverName?: string | null;
  teamName: string;
  position: number | null;
  points: number;
  wins?: number | null;
  gap: string;
  bestLap: number | null;
};

type ConstructorStandingRow = {
  teamName: string;
  points: number;
  wins?: number | null;
  drivers: string[];
  bestPosition: number | null;
};

function buildDriverStandings(
  seasonSummary: F1SeasonSummaryResponse | null,
  snapshot: F1SessionSnapshot | null
): DriverStandingRow[] {
  if (seasonSummary?.driverStandings.length) {
    return seasonSummary.driverStandings
      .map((row, index) => ({
        driverNumber: row.driverNumber ?? row.position ?? index + 1,
        acronym: row.code ?? String(row.driverNumber ?? row.position ?? index + 1),
        driverName: row.fullName ?? ([row.givenName, row.familyName].filter(Boolean).join(" ") || null),
        teamName: row.constructorName ?? "Unknown team",
        position: row.position ?? index + 1,
        points: numberOrZero(row.points),
        wins: row.wins ?? null,
        gap: "-",
        bestLap: null,
      }))
      .sort((a, b) => (a.position ?? 10_000) - (b.position ?? 10_000));
  }
  if (!snapshot?.drivers.length) return [];
  const results = new Map<number, Record<string, unknown>>();
  for (const row of snapshot.sessionResults ?? []) {
    const driverNumber = numberOrNull(row.driver_number);
    if (driverNumber !== null) results.set(driverNumber, row);
  }
  return snapshot.drivers
    .map((driver) => {
      const result = results.get(driver.driver_number);
      const position = numberOrNull(result?.position) ?? driver.position ?? null;
      return {
        driverNumber: driver.driver_number,
        acronym: driver.acronym ?? String(driver.driver_number),
        driverName: driver.full_name ?? driver.acronym ?? String(driver.driver_number),
        teamName: driver.team_name ?? "Unknown team",
        position,
        points: position ? POINTS_BY_POSITION[position - 1] ?? 0 : 0,
        wins: null,
        gap: stringValue(result?.gap_to_leader ?? driver.gap_to_leader ?? driver.interval ?? "-"),
        bestLap: driver.best_lap_time ?? driver.last_lap_time ?? null,
      };
    })
    .sort((a, b) => (a.position ?? 10_000) - (b.position ?? 10_000));
}

function buildConstructorStandings(
  seasonRows: F1SeasonConstructorRow[] | undefined,
  rows: DriverStandingRow[]
): ConstructorStandingRow[] {
  if (seasonRows?.length) {
    const driversByTeam = rows.reduce((teams, row) => {
      const drivers = teams.get(row.teamName) ?? [];
      drivers.push(row.acronym);
      teams.set(row.teamName, drivers);
      return teams;
    }, new Map<string, string[]>());
    return seasonRows
      .map((row, index) => {
        const teamName = row.constructorName ?? row.constructorId ?? `Team ${index + 1}`;
        return {
          teamName,
          points: numberOrZero(row.points),
          wins: row.wins ?? null,
          drivers: driversByTeam.get(teamName) ?? [],
          bestPosition: row.position ?? index + 1,
        };
      })
      .sort((a, b) => b.points - a.points || (a.bestPosition ?? 10_000) - (b.bestPosition ?? 10_000));
  }
  const teams = new Map<string, ConstructorStandingRow>();
  for (const row of rows) {
    const existing = teams.get(row.teamName) ?? { teamName: row.teamName, points: 0, drivers: [], bestPosition: null };
    existing.points += row.points;
    existing.drivers.push(row.acronym);
    existing.bestPosition = existing.bestPosition === null ? row.position : Math.min(existing.bestPosition, row.position ?? 10_000);
    teams.set(row.teamName, existing);
  }
  return Array.from(teams.values()).sort((a, b) => b.points - a.points || (a.bestPosition ?? 10_000) - (b.bestPosition ?? 10_000));
}

function buildDriverRankings(snapshot: F1SessionSnapshot | null) {
  const drivers = snapshot?.drivers ?? [];
  if (!drivers.length) return [];
  const bestLap = Math.min(...drivers.map((driver) => driver.best_lap_time ?? driver.last_lap_time ?? Infinity));
  const maxSpeed = Math.max(...drivers.map((driver) => driver.last_speed ?? 0), 1);
  const predictions = new Map((snapshot?.predictions ?? []).map((prediction) => [prediction.driver_number, prediction]));

  return drivers
    .map((driver) => {
      const lap = driver.best_lap_time ?? driver.last_lap_time ?? null;
      const positionScore = Math.max(0, 42 - ((driver.position ?? drivers.length + 1) - 1) * 4.2);
      const paceScore = lap && Number.isFinite(bestLap) ? Math.max(0, 28 - (lap - bestLap) * 14) : 0;
      const speedScore = ((driver.last_speed ?? 0) / maxSpeed) * 14;
      const prediction = predictions.get(driver.driver_number);
      const predictionScore = prediction ? prediction.confidence * 8 + prediction.podium_probability * 8 : 4;
      const score = Math.min(100, positionScore + paceScore + speedScore + predictionScore);
      return {
        driverNumber: driver.driver_number,
        label: driver.acronym ?? String(driver.driver_number),
        teamName: driver.team_name ?? "Unknown team",
        position: driver.position ?? null,
        bestLap: lap,
        score,
        reason: rankingReason(driver, lap, bestLap),
      };
    })
    .sort((a, b) => b.score - a.score);
}

function buildPowerUnitGroups(snapshot: F1SessionSnapshot | null) {
  const groups = new Map<string, { supplier: string; teams: Set<string>; drivers: string[]; speeds: number[]; drs: number[] }>();
  for (const driver of snapshot?.drivers ?? []) {
    const supplier = supplierForTeam(driver.team_name);
    const group = groups.get(supplier) ?? { supplier, teams: new Set<string>(), drivers: [], speeds: [], drs: [] };
    if (driver.team_name) group.teams.add(driver.team_name);
    group.drivers.push(driver.acronym ?? String(driver.driver_number));
    if (typeof driver.last_speed === "number") group.speeds.push(driver.last_speed);
    if (typeof driver.drs === "number") group.drs.push(driver.drs);
    groups.set(supplier, group);
  }
  return Array.from(groups.values())
    .map((group) => ({
      supplier: group.supplier,
      teams: Array.from(group.teams).sort(),
      drivers: group.drivers,
      avgSpeed: average(group.speeds),
      maxSpeed: Math.max(...group.speeds, 0),
      avgDrs: average(group.drs),
    }))
    .sort((a, b) => b.maxSpeed - a.maxSpeed);
}

function supplierForTeam(teamName?: string | null): string {
  const team = String(teamName ?? "").toLowerCase();
  if (team.includes("mercedes") || team.includes("mclaren") || team.includes("williams")) return "Mercedes";
  if (team.includes("ferrari") || team.includes("haas")) return "Ferrari";
  if (team.includes("red bull") || team.includes("racing bulls")) return "Red Bull Powertrains";
  if (team.includes("alpine")) return "Renault";
  if (team.includes("aston")) return "Honda";
  if (team.includes("audi") || team.includes("sauber")) return "Audi";
  return teamName || "Unknown";
}

function rankingReason(driver: F1DriverState, lap: number | null, bestLap: number): string {
  const parts = [`P${driver.position ?? "-"}`];
  if (lap && Number.isFinite(bestLap)) parts.push(`${(lap - bestLap).toFixed(3)}s off best`);
  if (driver.last_speed) parts.push(`${Math.round(driver.last_speed)} km/h trace`);
  return parts.join(" · ");
}

function formatSessionTitle(session?: F1ScheduleSession | null): string {
  if (!session) return "F1 session";
  const year = typeof session.year === "number" ? session.year : "";
  const name = session.session_name ?? session.session_type ?? "Session";
  const location = formatSessionLocation(session);
  return `${year} ${name}${location ? ` · ${location}` : ""}`.trim();
}

function formatSessionLocation(session?: F1ScheduleSession | null): string {
  if (!session) return "";
  const track = session.circuit_short_name || session.location || null;
  const country = session.country_name || session.country_code || null;
  if (track && country && track !== country) return `${track}, ${country}`;
  return track || country || "";
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

function formatLapTime(value?: number | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(3).padStart(6, "0")}`;
}

function numberOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function numberOrZero(value?: number | null): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function formatPoints(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function stringValue(value: unknown): string {
  return value === null || value === undefined ? "-" : String(value);
}

function average(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}
