"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  getF1CircuitHistory,
  getFastF1Schedule,
  getF1SeasonSummary,
  getF1PlatformSessions,
  getF1PlatformSnapshot,
  getF1SessionStatus,
  type F1CircuitHistoryResponse,
  type F1DriverState,
  type F1SeasonConstructorRow,
  type F1SeasonDriverRow,
  type F1SeasonQualifyingResultRow,
  type F1SeasonRaceResultRow,
  type F1SeasonSummaryResponse,
  type F1SessionInfo,
  type F1SessionResolution,
  type F1SessionSnapshot,
  type F1SessionSummary,
  type FastF1ScheduleResponse,
  type FastF1ScheduleRound,
} from "@/lib/f1Platform";
import {
  getF1TrackLayoutProfile,
  type F1TrackCornerKind,
  type F1TrackLayoutProfile,
} from "@/lib/f1TrackLayouts";

type LoadedInsightsHomeData = {
  circuitHistory: F1CircuitHistoryResponse | null;
  resolution: F1SessionResolution | null;
  schedule: FastF1ScheduleResponse | null;
  seasonSummary: F1SeasonSummaryResponse | null;
  sessions: F1SessionSummary[];
  snapshot: F1SessionSnapshot | null;
  trackLayout: F1TrackLayoutProfile | null;
};

type ScheduleSessionWithRound = F1SessionInfo & {
  round: FastF1ScheduleRound;
};

type DriverBoardRow = {
  driverNumber: number;
  acronym: string;
  name: string;
  teamName: string;
  teamColour: string;
  position: number;
  points: number;
  bestLap: number | null;
  gap?: string | null;
};

type ConstructorBoardRow = {
  teamName: string;
  points: number;
  wins?: number | null;
  teamColour?: string | null;
};

type LastYearTab = "qualifying" | "race";

type LastYearResultRow = {
  position: number;
  driverNumber: number;
  acronym: string;
  name: string;
  teamName: string;
  teamColour: string;
  metric: string;
};

const POINTS_BY_POSITION = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1];

const COUNTRY_FLAG_CODES: Record<string, string> = {
  australia: "AU",
  bahrain: "BH",
  belgium: "BE",
  brazil: "BR",
  canada: "CA",
  china: "CN",
  hungary: "HU",
  italy: "IT",
  japan: "JP",
  mexico: "MX",
  monaco: "MC",
  netherlands: "NL",
  qatar: "QA",
  singapore: "SG",
  spain: "ES",
  austria: "AT",
  azerbaijan: "AZ",
  "saudi arabia": "SA",
  "united arab emirates": "AE",
  "united kingdom": "GB",
  "great britain": "GB",
  "united states": "US",
  "united states of america": "US",
  usa: "US",
};

const EVENT_FLAG_CODES: Record<string, string> = {
  "abu dhabi": "AE",
  azerbaijan: "AZ",
  australian: "AU",
  austrian: "AT",
  bahrain: "BH",
  belgian: "BE",
  brazilian: "BR",
  british: "GB",
  canadian: "CA",
  chinese: "CN",
  dutch: "NL",
  "emilia romagna": "IT",
  hungarian: "HU",
  italian: "IT",
  japanese: "JP",
  "las vegas": "US",
  miami: "US",
  "mexico city": "MX",
  monaco: "MC",
  qatar: "QA",
  "saudi arabian": "SA",
  singapore: "SG",
  spanish: "ES",
  "united states": "US",
};

const insightModules = [
  { title: "Live Dashboard", href: "/f1/insights/live", status: "Timing" },
  { title: "Session Analysis", href: "/f1/insights/session-analysis", status: "Analysis" },
  { title: "Engineer", href: "/f1/insights/engineer", status: "Telemetry" },
  { title: "Standings", href: "/f1/insights/standings", status: "Tables" },
  { title: "Power Units", href: "/f1/insights/power-units", status: "Power" },
];

export default function F1InsightsHome() {
  const [data, setData] = useState<LoadedInsightsHomeData>({
    circuitHistory: null,
    resolution: null,
    schedule: null,
    seasonSummary: null,
    sessions: [],
    snapshot: null,
    trackLayout: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState<number | null>(null);
  const [lastYearTab, setLastYearTab] = useState<LastYearTab>("qualifying");

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
        const circuitHistory = await loadCircuitHistoryForHome(schedule, resolution, year).catch(() => null);
        const trackLayout = circuitHistory
          ? await getF1TrackLayoutProfile({
              circuitId: circuitHistory.circuitId,
              circuitName: circuitHistory.circuitName,
              eventName: circuitHistory.raceName,
            }).catch(() => null)
          : null;
        if (cancelled) return;
        setData({ circuitHistory, resolution, schedule, seasonSummary, sessions, snapshot, trackLayout });
        if (!trackLayout && !circuitHistory && !resolution && !schedule && !seasonSummary && !snapshot) {
          setError("F1 insights data could not be loaded from the season or runtime APIs.");
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unable to load F1 insights home");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const model = useMemo(() => buildInsightsHomeModel(data, now), [data, now]);
  const lastYearRows = lastYearTab === "qualifying" ? model.lastYearQualifyingRows : model.lastYearRaceRows;
  const historyLoading = loading && !data.circuitHistory;

  return (
    <div className="stack-lg f1-insights-home">
      <section className="f1-insights-race-hero">
        <div className="f1-insights-race-main">
          <div className="f1-race-overline">
            <span>Round {model.currentRound?.roundNumber ?? "-"}</span>
            <strong>{model.sessionStatusLabel}</strong>
          </div>
          <h1>{model.eventName}</h1>
          <p>{model.locationLine}</p>
          <div className="f1-session-pill-row">
            {model.currentRound?.sessions.map((session, index) => (
              <span
                className={`f1-session-pill ${session.session_key === model.focusSession?.session_key ? "active" : ""}`}
                key={homeScheduleSessionKey(session, index)}
              >
                <strong>{sessionDisplayName(session.session_name)}</strong>
                <span>{formatShortDate(session.date_start)}</span>
              </span>
            ))}
          </div>
        </div>
        <div className="f1-countdown-block">
          <span>{model.countdownTitle}</span>
          <strong className="f1-next-session-label">{model.focusSessionLine}</strong>
          <div className="f1-countdown-grid">
            <CountdownCell label="Days" value={model.countdown.days} />
            <CountdownCell label="Hrs" value={model.countdown.hours} />
            <CountdownCell label="Min" value={model.countdown.minutes} />
            <CountdownCell label="Sec" value={model.countdown.seconds} />
          </div>
        </div>
      </section>

      <section className="f1-calendar-panel">
        <div className="f1-section-heading">
          <h2>{data.schedule?.year ?? new Date().getFullYear()} Calendar</h2>
          <span>{loading ? "Loading" : error ? "Limited" : "FastF1 schedule"}</span>
        </div>
        <div className="f1-calendar-strip" aria-label="F1 season calendar">
          {model.calendarRounds.map((round, index) => (
            <div
              className={`f1-calendar-card ${round.scheduleKey === model.currentRound?.scheduleKey ? "active" : ""}`}
              key={homeScheduleRoundKey(round, index)}
            >
              <span>R{round.roundNumber ?? "-"}</span>
              <strong className="f1-calendar-flag">{countryFlag(round.country, round.eventName)}</strong>
              <small>{shortEventName(round.eventName)}</small>
              <em>{formatShortDate(round.eventDate)}</em>
            </div>
          ))}
        </div>
      </section>

      {error ? <div className="f1-inline-notice error">{error}</div> : null}

      <section className="f1-home-grid two">
        <div className="f1-home-panel">
          <div className="f1-home-panel-header">
            <div>
              <h2>Previous Race</h2>
              <span>{model.previousRaceTitle}</span>
            </div>
          </div>
          <div className="f1-podium">
            {model.previousRacePodium.map((driver, index) => (
              <div className={`f1-podium-step pos-${index + 1}`} key={`${driver.position}-${driver.driverNumber}-${driver.acronym}`}>
                <span style={{ background: driver.teamColour }} />
                <strong>P{index + 1}</strong>
                <h3>{driver.acronym}</h3>
                <small>{driver.teamName}</small>
              </div>
            ))}
          </div>
          <DriverResultList drivers={model.previousRaceRows.slice(0, 10)} />
        </div>

        <div className="f1-home-panel">
          <div className="f1-home-panel-header inline">
            <div>
              <h2>Championship</h2>
              <span>{model.championshipTitle}</span>
            </div>
            <Link href="/f1/insights/standings">Full standings</Link>
          </div>
          <ChampionshipList rows={model.championshipRows.slice(0, 5)} />
          <ConstructorList rows={model.constructors.slice(0, 3)} />
          <Link className="f1-full-width-action" href="/f1/insights/standings">
            View Full Standings
          </Link>
        </div>
      </section>

      <section className="f1-home-grid two">
        <div className="f1-home-panel">
          <div className="f1-home-panel-header">
            <div>
              <h2>Last Year At This Circuit</h2>
              <span>{model.lastYearTitle}</span>
            </div>
          </div>
          <div className="f1-tab-row">
            <button
              className={lastYearTab === "qualifying" ? "active" : ""}
              type="button"
              aria-pressed={lastYearTab === "qualifying"}
              disabled={historyLoading}
              onClick={() => setLastYearTab("qualifying")}
            >
              <span>Qualifying</span>
              <small>{historyLoading ? "..." : model.lastYearQualifyingRows.length}</small>
            </button>
            <button
              className={lastYearTab === "race" ? "active" : ""}
              type="button"
              aria-pressed={lastYearTab === "race"}
              disabled={historyLoading}
              onClick={() => setLastYearTab("race")}
            >
              <span>Race</span>
              <small>{historyLoading ? "..." : model.lastYearRaceRows.length}</small>
            </button>
          </div>
          <LastYearResultList rows={lastYearRows} mode={lastYearTab} pending={historyLoading} />
          <div className="f1-panel-actions">
            <Link href="/f1/insights/session-analysis" className="button secondary button-sm">
              Session Summary
            </Link>
            <Link href="/f1/insights/engineer" className="button button-sm">
              Quick Compare - Top 4
            </Link>
          </div>
        </div>

        <div className="f1-home-panel">
          <div className="f1-home-panel-header">
            <div>
              <h2>Track Profile</h2>
              <span>{model.eventName}</span>
            </div>
          </div>
          <TrackProfileCard profile={model.trackProfile} />
        </div>
      </section>

      <section className="f1-module-strip" aria-label="Insights shortcuts">
        {insightModules.map((module) => (
          <Link href={module.href} className="f1-module-shortcut" key={module.href}>
            <span>{module.status}</span>
            <strong>{module.title}</strong>
          </Link>
        ))}
      </section>
    </div>
  );
}

function CountdownCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="f1-countdown-cell">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function DriverResultList({ drivers }: { drivers: DriverBoardRow[] }) {
  return (
    <div className="f1-result-list">
      {drivers.map((driver) => (
        <div className="f1-result-row" key={`${driver.position}-${driver.driverNumber}-${driver.acronym}`}>
          <span className="f1-team-rail" style={{ background: driver.teamColour }} />
          <strong>{driver.position}</strong>
          <b>{driver.acronym}</b>
          <span>{driver.name}</span>
          <em>{driver.gap ?? "-"}</em>
        </div>
      ))}
    </div>
  );
}

function ChampionshipList({ rows }: { rows: DriverBoardRow[] }) {
  const maxPoints = Math.max(...rows.map((row) => row.points), 1);
  return (
    <div className="f1-championship-section">
      <h3>Drivers</h3>
      {rows.map((row) => (
        <div className="f1-championship-row" key={row.driverNumber}>
          <span>{row.position}</span>
          <i style={{ background: row.teamColour }} />
          <strong>{row.name}</strong>
          <div className="f1-points-bar"><span style={{ width: `${Math.max(8, (row.points / maxPoints) * 100)}%` }} /></div>
          <em>{formatPoints(row.points)}</em>
        </div>
      ))}
    </div>
  );
}

function ConstructorList({ rows }: { rows: ConstructorBoardRow[] }) {
  return (
    <div className="f1-championship-section constructors">
      <h3>Constructors</h3>
      {rows.map((row, index) => (
        <div className="f1-constructor-row" key={row.teamName}>
          <span>{index + 1}</span>
          <strong>{row.teamName}</strong>
          <em>{formatPoints(row.points)}</em>
        </div>
      ))}
    </div>
  );
}

function LastYearResultList({ rows, mode, pending }: { rows: LastYearResultRow[]; mode: LastYearTab; pending: boolean }) {
  if (pending) {
    return (
      <div className="f1-history-empty">
        Loading historical {mode === "qualifying" ? "qualifying" : "race"} classification
      </div>
    );
  }

  if (!rows.length) {
    return (
      <div className="f1-history-empty">
        Historical {mode === "qualifying" ? "qualifying" : "race"} classification pending
      </div>
    );
  }
  return (
    <div className="f1-history-result-list">
      {rows.slice(0, 10).map((row) => (
        <div className="f1-history-result-row" key={`${mode}-${row.position}-${row.driverNumber}-${row.acronym}`}>
          <span className="f1-team-rail" style={{ background: row.teamColour }} />
          <strong>{row.position}</strong>
          <b>{row.acronym}</b>
          <span className="f1-history-driver">{row.name}</span>
          <span className="f1-history-team">{row.teamName}</span>
          <em>{row.metric}</em>
        </div>
      ))}
    </div>
  );
}

function TrackProfileCard({ profile }: { profile: TrackProfile }) {
  return (
    <div className="f1-track-profile">
      <div className="f1-track-canvas" aria-label="Track map">
        <svg viewBox="0 0 640 360" role="img">
          {profile.pathD ? (
            <>
              <path className="track-shadow" d={profile.pathD} />
              <path className="track-road" d={profile.pathD} />
              <path className="track-line" d={profile.pathD} />
              {profile.corners.map((corner) => (
                <g className={`track-corner ${corner.kind}`} key={`${corner.number}-${corner.x}-${corner.y}`}>
                  <circle cx={corner.x} cy={corner.y} r="8" />
                  <text x={corner.x} y={corner.y + 3}>{corner.number}</text>
                </g>
              ))}
            </>
          ) : (
            <text className="track-empty-label" x="320" y="180">Track layout loading</text>
          )}
        </svg>
      </div>
      <div className="f1-track-type-row">
        <span className="f1-track-type">{profile.type}</span>
        <span className="f1-track-layout-source">{profile.source}</span>
      </div>
      <div className="f1-corner-legend" aria-label="Corner speed legend">
        <span><i className="slow" />Slow</span>
        <span><i className="medium" />Medium</span>
        <span><i className="fast" />Fast</span>
      </div>
      <div className="f1-corner-type-bar">
        <span className="slow" style={{ flexGrow: profile.slow }} />
        <span className="medium" style={{ flexGrow: profile.medium }} />
        <span className="fast" style={{ flexGrow: profile.fast }} />
      </div>
      <div className="f1-track-stats">
        <TrackStat value={String(profile.slow)} label="Slow" detail="<130 km/h" tone="slow" />
        <TrackStat value={String(profile.medium)} label="Medium" detail="130-220" tone="medium" />
        <TrackStat value={String(profile.fast)} label="Fast" detail=">220 km/h" tone="fast" />
      </div>
      <dl className="f1-track-profile-meta">
        <div><dt>Circuit Length</dt><dd>{profile.lengthKm} km</dd></div>
        <div><dt>Reference Pole</dt><dd>{profile.referencePole}</dd></div>
        <div><dt>Total Corners</dt><dd>{profile.totalCorners}</dd></div>
      </dl>
    </div>
  );
}

function TrackStat({ value, label, detail, tone }: { value: string; label: string; detail: string; tone: string }) {
  return (
    <div className={`f1-track-stat ${tone}`}>
      <strong>{value}</strong>
      <span>{label}</span>
      <small>{detail}</small>
    </div>
  );
}

function buildInsightsHomeModel(data: LoadedInsightsHomeData, now: number | null) {
  const calendarRounds = (data.schedule?.rounds ?? []).filter((round) => (round.roundNumber ?? 0) > 0);
  const scheduleSessions = flattenScheduleWithRounds(calendarRounds);
  const resolvedLiveSession = data.resolution?.status === "live" ? data.resolution.session ?? null : null;
  const focusSession = resolvedLiveSession ?? data.resolution?.nextSession ?? nextFromSchedule(scheduleSessions, now);
  const currentRound = findRoundForSession(calendarRounds, focusSession) ?? findCurrentRound(calendarRounds, now) ?? calendarRounds[0] ?? null;
  const previousRound = findPreviousRound(calendarRounds, currentRound);
  const snapshotRows = buildDriverBoardRows(data.snapshot);
  const previousRaceRows = buildRaceResultRows(data.seasonSummary?.latestRace?.results, snapshotRows);
  const championshipRows = buildSeasonDriverRows(data.seasonSummary?.driverStandings, snapshotRows);
  const constructorRows = buildSeasonConstructorRows(data.seasonSummary?.constructorStandings, buildConstructors(snapshotRows));
  const lastYearQualifyingRows = buildQualifyingHistoryRows(data.circuitHistory?.qualifyingResults);
  const lastYearRaceRows = buildRaceHistoryRows(data.circuitHistory?.raceResults);
  const eventName = currentRound?.eventName ?? stringValue(focusSession?.fastf1_event_name) ?? "F1 Grand Prix";
  const sessionIsLive = Boolean(resolvedLiveSession);
  const countdownTarget = sessionIsLive ? focusSession?.date_end ?? focusSession?.date_start : focusSession?.date_start;
  const latestRace = data.seasonSummary?.latestRace;
  const referencePole = data.circuitHistory?.qualifyingResults[0]
    ? qualifyingDisplayTime(data.circuitHistory.qualifyingResults[0])
    : null;
  return {
    calendarRounds,
    currentRound,
    previousRound,
    focusSession,
    sessionStatusLabel: sessionIsLive ? "Current session" : "Next session",
    countdownTitle: sessionIsLive ? "Time left in session" : "Time until next session",
    focusSessionLine: formatFocusSessionLine(focusSession),
    eventName,
    locationLine: formatRoundLocation(currentRound, focusSession),
    countdown: splitCountdown(countdownTarget, now),
    driverRows: snapshotRows,
    previousRaceRows,
    previousRacePodium: previousRaceRows.slice(0, 3),
    previousRaceTitle: latestRace?.raceName ?? previousRound?.eventName ?? "Latest race result",
    championshipRows,
    championshipTitle: data.seasonSummary?.round ? `${data.seasonSummary.year} Round ${data.seasonSummary.round}` : "Current season",
    constructors: constructorRows,
    lastYearQualifyingRows,
    lastYearRaceRows,
    lastYearTitle: `${data.circuitHistory?.year ?? (data.schedule?.year ?? new Date().getFullYear()) - 1} ${data.circuitHistory?.raceName ?? eventName}`,
    trackProfile: trackProfileForEvent(eventName, referencePole, data.trackLayout),
  };
}

async function loadCircuitHistoryForHome(
  schedule: FastF1ScheduleResponse | null,
  resolution: F1SessionResolution | null,
  year: number
): Promise<F1CircuitHistoryResponse | null> {
  const calendarRounds = (schedule?.rounds ?? []).filter((round) => (round.roundNumber ?? 0) > 0);
  if (!calendarRounds.length) return null;
  const scheduleSessions = flattenScheduleWithRounds(calendarRounds);
  const resolvedLiveSession = resolution?.status === "live" ? resolution.session ?? null : null;
  const focusSession = resolvedLiveSession ?? resolution?.nextSession ?? nextFromSchedule(scheduleSessions, Date.now());
  const currentRound = findRoundForSession(calendarRounds, focusSession) ?? findCurrentRound(calendarRounds, Date.now()) ?? calendarRounds[0] ?? null;
  const roundNumber = currentRound?.roundNumber ?? focusSession?.round_number ?? null;
  if (!roundNumber) return null;
  return getF1CircuitHistory({ year, roundNumber });
}

function flattenScheduleWithRounds(rounds: FastF1ScheduleRound[]): ScheduleSessionWithRound[] {
  return rounds
    .flatMap((round) => round.sessions.map((session) => ({ ...session, round })))
    .sort((a, b) => timestamp(a.date_start) - timestamp(b.date_start));
}

function nextFromSchedule(sessions: ScheduleSessionWithRound[], now: number | null): ScheduleSessionWithRound | null {
  const instant = now ?? Date.now();
  return sessions.find((session) => timestamp(session.date_start) >= instant) ?? null;
}

function findRoundForSession(rounds: FastF1ScheduleRound[], session?: F1SessionInfo | null): FastF1ScheduleRound | null {
  if (!session) return null;
  return rounds.find((round) => round.sessions.some((candidate) => candidate.session_key === session.session_key)) ?? null;
}

function findCurrentRound(rounds: FastF1ScheduleRound[], now: number | null): FastF1ScheduleRound | null {
  const instant = now ?? Date.now();
  return rounds.find((round) => {
    const starts = round.sessions.map((session) => timestamp(session.date_start)).filter(Number.isFinite);
    if (!starts.length) return false;
    const minStart = Math.min(...starts);
    const maxEnd = Math.max(...round.sessions.map((session) => timestamp(session.date_end ?? session.date_start)).filter(Number.isFinite));
    return minStart <= instant && instant <= maxEnd + 3 * 60 * 60 * 1000;
  }) ?? rounds.find((round) => round.sessions.some((session) => timestamp(session.date_start) >= instant)) ?? null;
}

function findPreviousRound(rounds: FastF1ScheduleRound[], current: FastF1ScheduleRound | null): FastF1ScheduleRound | null {
  if (!current) return null;
  const currentRound = current.roundNumber ?? 0;
  return [...rounds]
    .filter((round) => (round.roundNumber ?? 0) < currentRound)
    .sort((a, b) => (b.roundNumber ?? 0) - (a.roundNumber ?? 0))[0] ?? null;
}

function selectBestSession(sessions: F1SessionSummary[]): F1SessionSummary | null {
  return [...sessions].sort((a, b) => {
    const eventDelta = (b.eventCount ?? 0) - (a.eventCount ?? 0);
    if (eventDelta !== 0) return eventDelta;
    return (b.drivers ?? 0) - (a.drivers ?? 0);
  })[0] ?? null;
}

function buildDriverBoardRows(snapshot: F1SessionSnapshot | null): DriverBoardRow[] {
  return (snapshot?.drivers ?? [])
    .map((driver, index) => {
      const position = driver.position ?? index + 1;
      return {
        driverNumber: driver.driver_number,
        acronym: driver.acronym ?? String(driver.driver_number),
        name: driver.full_name ?? driver.acronym ?? `Driver ${driver.driver_number}`,
        teamName: driver.team_name ?? "Unknown",
        teamColour: teamColor(driver),
        position,
        points: POINTS_BY_POSITION[position - 1] ?? 0,
        bestLap: driver.best_lap_time ?? driver.last_lap_time ?? null,
      };
    })
    .sort((a, b) => a.position - b.position);
}

function buildRaceResultRows(rows: F1SeasonRaceResultRow[] | undefined, fallback: DriverBoardRow[]): DriverBoardRow[] {
  if (!rows?.length) return fallback;
  return rows
    .map((row, index) => seasonDriverToBoardRow(row, index, row.gap ?? row.status ?? null))
    .sort((a, b) => a.position - b.position);
}

function buildQualifyingHistoryRows(rows: F1SeasonQualifyingResultRow[] | undefined): LastYearResultRow[] {
  if (!rows?.length) return [];
  return rows
    .map((row, index) => seasonHistoryToResultRow(row, index, qualifyingDisplayTime(row)))
    .sort((a, b) => a.position - b.position);
}

function buildRaceHistoryRows(rows: F1SeasonRaceResultRow[] | undefined): LastYearResultRow[] {
  if (!rows?.length) return [];
  return rows
    .map((row, index) => seasonHistoryToResultRow(row, index, row.gap ?? row.time ?? row.status ?? "-"))
    .sort((a, b) => a.position - b.position);
}

function seasonHistoryToResultRow(row: F1SeasonDriverRow, index: number, metric: string | null): LastYearResultRow {
  const position = row.position ?? index + 1;
  const driverNumber = row.driverNumber ?? position;
  const acronym = row.code ?? String(driverNumber);
  return {
    position,
    driverNumber,
    acronym,
    name: row.fullName ?? ([row.givenName, row.familyName].filter(Boolean).join(" ") || acronym),
    teamName: row.constructorName ?? "Unknown",
    teamColour: normalizeTeamColour(row.teamColour),
    metric: metric || "-",
  };
}

function qualifyingDisplayTime(row: F1SeasonQualifyingResultRow): string {
  return row.q3 ?? row.q2 ?? row.q1 ?? row.time ?? "-";
}

function buildSeasonDriverRows(rows: F1SeasonDriverRow[] | undefined, fallback: DriverBoardRow[]): DriverBoardRow[] {
  if (!rows?.length) return fallback;
  return rows
    .map((row, index) => seasonDriverToBoardRow(row, index, null))
    .sort((a, b) => a.position - b.position);
}

function seasonDriverToBoardRow(row: F1SeasonDriverRow, index: number, gap: string | null): DriverBoardRow {
  const position = row.position ?? index + 1;
  const driverNumber = row.driverNumber ?? position;
  const acronym = row.code ?? String(driverNumber);
  return {
    driverNumber,
    acronym,
    name: row.fullName ?? ([row.givenName, row.familyName].filter(Boolean).join(" ") || acronym),
    teamName: row.constructorName ?? "Unknown",
    teamColour: normalizeTeamColour(row.teamColour),
    position,
    points: numberOrZero(row.points),
    bestLap: null,
    gap,
  };
}

function buildConstructors(rows: DriverBoardRow[]): ConstructorBoardRow[] {
  const teams = new Map<string, { teamName: string; points: number }>();
  for (const row of rows) {
    const team = teams.get(row.teamName) ?? { teamName: row.teamName, points: 0 };
    team.points += row.points;
    teams.set(row.teamName, team);
  }
  return [...teams.values()].sort((a, b) => b.points - a.points);
}

function buildSeasonConstructorRows(
  rows: F1SeasonConstructorRow[] | undefined,
  fallback: ConstructorBoardRow[]
): ConstructorBoardRow[] {
  if (!rows?.length) return fallback;
  return rows
    .map((row) => ({
      teamName: row.constructorName ?? row.constructorId ?? "Unknown",
      points: numberOrZero(row.points),
      wins: row.wins ?? null,
      teamColour: normalizeTeamColour(row.teamColour),
    }))
    .sort((a, b) => b.points - a.points);
}

function splitCountdown(target?: string | null, now?: number | null): { days: string; hours: string; minutes: string; seconds: string } {
  const targetTime = timestamp(target);
  if (!Number.isFinite(targetTime) || now === null || now === undefined) return { days: "--", hours: "--", minutes: "--", seconds: "--" };
  const totalSeconds = Math.max(0, Math.floor((targetTime - now) / 1000));
  const days = Math.floor(totalSeconds / 86400);
  const hours = Math.floor((totalSeconds % 86400) / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return {
    days: pad2(days),
    hours: pad2(hours),
    minutes: pad2(minutes),
    seconds: pad2(seconds),
  };
}

function homeScheduleRoundKey(round: FastF1ScheduleRound, index: number): string {
  return [round.scheduleKey, round.officialEventName, round.eventDate, round.roundNumber, round.eventName, index]
    .filter((part) => part !== null && part !== undefined && part !== "")
    .map(String)
    .join("::");
}

function homeScheduleSessionKey(session: F1SessionInfo, index: number): string {
  return [session.session_key, session.schedule_event_key, session.date_start, session.round_number, session.session_name, index]
    .filter((part) => part !== null && part !== undefined && part !== "")
    .map(String)
    .join("::");
}

function sessionDisplayName(name?: string | null): string {
  const normalized = String(name ?? "").trim();
  return normalized || "Session";
}

function formatFocusSessionLine(session?: F1SessionInfo | null): string {
  if (!session) return "Schedule pending";
  const date = formatShortDate(session.date_start);
  const time = formatClockTime(session.date_start);
  return `${sessionDisplayName(session.session_name)} · ${date}${time !== "-" ? ` · ${time}` : ""}`;
}

function formatRoundLocation(round?: FastF1ScheduleRound | null, session?: F1SessionInfo | null): string {
  const location = round?.location ?? session?.location ?? session?.circuit_short_name ?? "-";
  const country = round?.country ?? session?.country_name ?? session?.country_code ?? "";
  if (country && location !== country) return `${location}, ${country}`;
  return location;
}

function countryFlag(country?: string | null, eventName?: string | null): string {
  const countryKey = normalizeFlagKey(country);
  const eventKey = normalizeFlagKey(shortEventName(eventName));
  const code = countryCodeFromValue(country) ?? COUNTRY_FLAG_CODES[countryKey] ?? EVENT_FLAG_CODES[eventKey];
  return code ? flagEmoji(code) : "--";
}

function countryCodeFromValue(value?: string | null): string | null {
  const normalized = String(value ?? "").trim().toUpperCase();
  return /^[A-Z]{2}$/.test(normalized) ? normalized : null;
}

function flagEmoji(countryCode: string): string {
  const normalized = countryCode.trim().toUpperCase();
  if (!/^[A-Z]{2}$/.test(normalized)) return "--";
  return Array.from(normalized)
    .map((letter) => String.fromCodePoint(127397 + letter.charCodeAt(0)))
    .join("");
}

function normalizeFlagKey(value?: string | null): string {
  return String(value ?? "")
    .toLowerCase()
    .replace(/grand prix/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function shortEventName(name?: string | null): string {
  return String(name ?? "Grand Prix").replace(/\s+Grand Prix$/i, "");
}

function formatShortDate(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
}

function formatClockTime(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", timeZoneName: "short" }).format(date);
}

function formatLapGap(value: number | null, leader: number | null): string {
  if (value === null || !Number.isFinite(value)) return "-";
  if (leader === null || !Number.isFinite(leader) || value === leader) return formatLapTime(value);
  return `+${(value - leader).toFixed(3)}`;
}

function formatLapTime(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "-";
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(3).padStart(6, "0")}`;
}

function timestamp(value?: string | null): number {
  if (!value) return Number.POSITIVE_INFINITY;
  const parsed = new Date(value).getTime();
  return Number.isNaN(parsed) ? Number.POSITIVE_INFINITY : parsed;
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function teamColor(driver: F1DriverState): string {
  const raw = String(driver.team_colour ?? "").trim();
  return normalizeTeamColour(raw);
}

function normalizeTeamColour(value?: string | null): string {
  const raw = String(value ?? "").trim();
  return raw ? `#${raw.replace(/^#/, "")}` : "#ff6363";
}

function numberOrZero(value?: number | null): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function formatPoints(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function stringValue(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed || null;
}

type TrackProfile = {
  type: string;
  slow: number;
  medium: number;
  fast: number;
  totalCorners: number;
  lengthKm: string;
  referencePole: string;
  pathD: string;
  source: string;
  corners: Array<{ number: number; x: number; y: number; kind: F1TrackCornerKind }>;
};

function trackProfileForEvent(
  eventName: string,
  referencePole: string | null,
  layout: F1TrackLayoutProfile | null
): TrackProfile {
  if (layout) {
    return {
      type: `${layout.circuitName} layout`,
      slow: layout.slow,
      medium: layout.medium,
      fast: layout.fast,
      totalCorners: layout.totalCorners,
      lengthKm: layout.lengthKm ?? "-",
      referencePole: referencePole ?? "-",
      pathD: layout.pathD,
      source: layout.source,
      corners: layout.corners,
    };
  }
  return {
    type: eventName,
    slow: 0,
    medium: 0,
    fast: 0,
    totalCorners: 0,
    lengthKm: "-",
    referencePole: referencePole ?? "-",
    pathD: "",
    source: "layout pending",
    corners: [],
  };
}
