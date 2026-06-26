export const F1_PLATFORM_API_BASE =
  process.env.NEXT_PUBLIC_F1_PLATFORM_API_URL || "http://127.0.0.1:8001";

export const F1_SEASON_API_BASE =
  process.env.NEXT_PUBLIC_F1_SEASON_API_URL || "https://api.jolpi.ca/ergast/f1";

const F1_TEAM_COLOURS: Record<string, string> = {
  alpine: "00A1E8",
  aston_martin: "006F62",
  audi: "C0C0C0",
  cadillac: "B8B8B8",
  ferrari: "F91536",
  haas: "B6BABD",
  mclaren: "FF8700",
  mercedes: "27F4D2",
  racing_bulls: "6692FF",
  rb: "6692FF",
  red_bull: "3671C6",
  sauber: "00E701",
  williams: "64C4FF",
};

export type F1DriverState = {
  driver_number: number;
  acronym?: string | null;
  full_name?: string | null;
  team_name?: string | null;
  team_colour?: string | null;
  position?: number | null;
  interval?: string | null;
  gap_to_leader?: string | null;
  current_lap?: number | null;
  last_lap_time?: number | null;
  best_lap_time?: number | null;
  sector_times?: Record<string, number | null>;
  current_compound?: string | null;
  tyre_age?: number | null;
  stint_number?: number | null;
  pit_status?: string | null;
  track_status?: string | null;
  last_speed?: number | null;
  last_location?: Record<string, number> | null;
  track_progress?: number | null;
  drs?: number | null;
  last_update_seq: number;
};

export type F1LapPoint = {
  lap: number;
  driver_number: number;
  value: number;
};

export type F1StintSegment = {
  driver_number: number;
  stint_number: number;
  compound: string;
  start_lap: number;
  end_lap?: number | null;
  tyre_age_start: number;
};

export type F1PredictionSnapshot = {
  model_version: string;
  prediction_time: string;
  source_event_sequence: number;
  features_version: string;
  driver_number: number;
  expected_position?: number | null;
  position_p10?: number | null;
  position_p90?: number | null;
  position_distribution: Record<string, number>;
  win_probability: number;
  podium_probability: number;
  points_probability: number;
  dnf_probability: number;
  confidence: number;
};

export type F1CustomMicroSectorPassage = {
  driver_number: number;
  lap?: number | null;
  sector_index: number;
  sector_count: number;
  progress_start: number;
  progress_end: number;
  passage_time: number;
  personal_best_delta?: number | null;
  session_best_delta?: number | null;
  car_ahead_delta?: number | null;
  teammate_delta?: number | null;
  label: string;
  source: string;
  event_time?: string | null;
  seq: number;
};

export type F1WeatherSample = {
  date?: string | null;
  event_time?: string | null;
  air_temperature?: number | null;
  track_temperature?: number | null;
  rainfall?: number | null;
  wind_speed?: number | null;
  source_id?: number | null;
  source_key?: string | null;
  [key: string]: unknown;
};

export type F1SessionSnapshot = {
  sessionKey: string | number;
  seq: number;
  generatedAt: string;
  source: string;
  sessionInfo?: Record<string, unknown> | null;
  drivers: F1DriverState[];
  lapChart: F1LapPoint[];
  strategyTimeline: F1StintSegment[];
  raceControl: Array<Record<string, unknown>>;
  pitStops: Array<Record<string, unknown>>;
  overtakes: Array<Record<string, unknown>>;
  sessionResults: Array<Record<string, unknown>>;
  customMicroSectors: F1CustomMicroSectorPassage[];
  weather?: Record<string, unknown> | null;
  weatherSamples: F1WeatherSample[];
  predictions: F1PredictionSnapshot[];
  topicWatermarks: Record<string, number>;
  replay: Record<string, unknown>;
};

export type F1TrackGeometryPoint = {
  distance: number;
  progress: number;
  x: number;
  y: number;
  z?: number | null;
};

export type F1TrackGeometryResponse = {
  sessionKey: string;
  artifactId: string;
  source: string;
  pointCount: number;
  sampledPointCount: number;
  points: F1TrackGeometryPoint[];
};

export type F1SessionSummary = {
  sessionKey: string | number;
  seq: number;
  source: string;
  meetingKey?: number | null;
  sessionName?: string | null;
  sessionType?: string | null;
  eventName?: string | null;
  location?: string | null;
  countryName?: string | null;
  dateStart?: string | null;
  dateEnd?: string | null;
  year?: number | null;
  drivers: number;
  eventCount: number;
};

export type F1StreamUpdate = {
  seq: number;
  type: string;
  eventTime?: string | null;
  driverNumber?: number | null;
  payload: Record<string, unknown>;
};

export type OpenF1ImportRequest = {
  year?: number | null;
  meeting_key?: number | null;
  session_key?: string | number | null;
  session_name?: string;
  include_telemetry?: boolean;
  limit_per_topic?: number;
};

export type OpenF1ImportResponse = {
  imported: boolean;
  sessionKey: string | number;
  eventCount: number;
  topicCounts: Record<string, number>;
  replayPath: string;
  snapshot: F1SessionSnapshot;
};

export type F1SessionInfo = {
  session_key?: string | number | null;
  schedule_event_key?: string | number | null;
  meeting_key?: number | null;
  round_number?: number | null;
  session_name?: string | null;
  session_type?: string | null;
  date_start?: string | null;
  date_end?: string | null;
  location?: string | null;
  country_name?: string | null;
  country_code?: string | null;
  circuit_short_name?: string | null;
  year?: number | null;
  gmt_offset?: string | null;
  is_cancelled?: boolean | null;
  [key: string]: unknown;
};

export type F1SessionResolution = {
  status: "live" | "upcoming" | "unavailable";
  source: string;
  resolvedAt: string;
  message: string;
  session?: F1SessionInfo | null;
  nextSession?: F1SessionInfo | null;
  secondsUntilStart?: number | null;
  secondsUntilEnd?: number | null;
  fallbackReason?: string | null;
};

export type OpenF1SessionInfo = F1SessionInfo;
export type OpenF1SessionResolution = F1SessionResolution;

export type F1ReplayStatus = {
  sessionKey: string;
  state: "idle" | "starting" | "running" | "finished" | "stopped" | "error";
  speed: number;
  eventCount: number;
  cursor: number;
  replayPath?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  error?: string | null;
};

export type FastF1ImportRequest = {
  year: number;
  event: string | number;
  session_name?: string;
  drivers?: string[];
  include_telemetry?: boolean;
  telemetry_laps_per_driver?: number;
  distance_step_meters?: number;
  output_format?: "jsonl" | "parquet";
  map_to_session_key?: string | number | null;
};

export type FastF1ArtifactRecord = {
  kind: string;
  path: string;
  format: string;
  row_count?: number | null;
  metadata: Record<string, unknown>;
  artifactId?: string | null;
  relativePath?: string | null;
};

export type FastF1ImportResponse = {
  imported: boolean;
  sessionKey: string;
  generatedAt: string;
  artifacts: FastF1ArtifactRecord[];
  notes: string[];
};

export type FastF1ArtifactListResponse = {
  artifacts: FastF1ArtifactRecord[];
  count: number;
  limit: number;
};

export type FastF1ScheduleRound = {
  scheduleKey?: string | null;
  roundNumber?: number | null;
  eventName?: string | null;
  officialEventName?: string | null;
  eventDate?: string | null;
  country?: string | null;
  location?: string | null;
  eventFormat?: string | null;
  f1ApiSupport?: boolean | null;
  sessions: F1SessionInfo[];
};

export type FastF1ScheduleResponse = {
  year: number;
  source: string;
  roundCount: number;
  sessionCount: number;
  rounds: FastF1ScheduleRound[];
};

export type F1SeasonDriverRow = {
  position?: number | null;
  points?: number | null;
  wins?: number | null;
  driverId?: string | null;
  driverNumber?: number | null;
  code?: string | null;
  givenName?: string | null;
  familyName?: string | null;
  fullName?: string | null;
  nationality?: string | null;
  constructorId?: string | null;
  constructorName?: string | null;
  constructorNationality?: string | null;
  teamColour?: string | null;
};

export type F1SeasonConstructorRow = {
  position?: number | null;
  points?: number | null;
  wins?: number | null;
  constructorId?: string | null;
  constructorName?: string | null;
  constructorNationality?: string | null;
  teamColour?: string | null;
};

export type F1SeasonRaceResultRow = F1SeasonDriverRow & {
  positionText?: string | null;
  grid?: number | null;
  laps?: number | null;
  status?: string | null;
  time?: string | null;
  gap?: string | null;
  fastestLap?: string | null;
};

export type F1SeasonQualifyingResultRow = F1SeasonDriverRow & {
  q1?: string | null;
  q2?: string | null;
  q3?: string | null;
  time?: string | null;
  gap?: string | null;
};

export type F1SeasonSummaryResponse = {
  source: string;
  sourceUrl?: string | null;
  generatedAt: string;
  year: number;
  round?: number | null;
  latestRace?: {
    season?: number | null;
    round?: number | null;
    raceName?: string | null;
    date?: string | null;
    time?: string | null;
    circuitId?: string | null;
    circuitName?: string | null;
    location?: string | null;
    country?: string | null;
    results: F1SeasonRaceResultRow[];
  } | null;
  driverStandings: F1SeasonDriverRow[];
  constructorStandings: F1SeasonConstructorRow[];
};

export type F1CircuitHistoryResponse = {
  source: string;
  sourceUrl?: string | null;
  generatedAt: string;
  currentYear: number;
  year: number;
  round?: number | null;
  circuitId?: string | null;
  circuitName?: string | null;
  raceName?: string | null;
  date?: string | null;
  qualifyingResults: F1SeasonQualifyingResultRow[];
  raceResults: F1SeasonRaceResultRow[];
};

export type F1WeatherForecastResponse = {
  source: "open-meteo" | string;
  authentication: "none" | string;
  requiresApiKey: boolean;
  url?: string;
  matchedBy?: string | null;
  circuit: {
    id: string;
    name: string;
    latitude: number;
    longitude: number;
    timezone: string;
    aliases?: string[];
  };
  session?: F1SessionInfo | null;
  sessionResolution?: {
    status?: string | null;
    source?: string | null;
    message?: string | null;
  } | null;
  current: Record<string, unknown>;
  summary: Record<string, unknown>;
  hourly: Array<Record<string, unknown>>;
  rawUnits?: Record<string, unknown>;
};

export type FastF1ArtifactRowsResponse = {
  artifact: FastF1ArtifactRecord;
  columns: string[];
  rows: Array<Record<string, unknown>>;
  limit: number;
  truncated: boolean;
};

export type FastF1EngineeringSummary = {
  sessionKey?: string | null;
  generatedAt: string;
  telemetryDelta?: {
    artifact: FastF1ArtifactRecord;
    driverA?: string | null;
    driverB?: string | null;
    lapA?: number | null;
    lapB?: number | null;
    sampleCount: number;
    limit: number;
    truncated: boolean;
    distanceStart?: number | null;
    distanceEnd?: number | null;
    finalDeltaSeconds?: number | null;
    maxGainDriverASeconds?: number | null;
    maxGainDriverBSeconds?: number | null;
    maxSpeedDeltaKmh?: number | null;
    series: Array<{
      distance?: number | null;
      deltaSeconds?: number | null;
      speedA?: number | null;
      speedB?: number | null;
      throttleA?: number | null;
      throttleB?: number | null;
      brakeA?: number | null;
      brakeB?: number | null;
      gearA?: number | null;
      gearB?: number | null;
      drsA?: number | null;
      drsB?: number | null;
    }>;
  } | null;
  cornerMetrics: Array<{
    artifact: FastF1ArtifactRecord;
    driver?: string | null;
    lapNumber?: number | null;
    cornerCount: number;
    limit: number;
    truncated: boolean;
    fastestCornerTimeSeconds?: number | null;
    slowestMinimumSpeedKmh?: number | null;
    corners: Array<{
      cornerIndex?: number | null;
      entryDistance?: number | null;
      apexDistance?: number | null;
      exitDistance?: number | null;
      entrySpeed?: number | null;
      minimumSpeed?: number | null;
      exitSpeed?: number | null;
      brakeStartDistance?: number | null;
      throttleReapplicationDistance?: number | null;
      fullThrottlePercent?: number | null;
      brakingDurationSeconds?: number | null;
      cornerTimeSeconds?: number | null;
      exitAccelerationKmhPer100m?: number | null;
    }>;
  }>;
  artifactCounts: {
    telemetryDelta: number;
    cornerMetrics: number;
  };
};

export type F1TyreDegradationCompound = {
  compound: string;
  cleanLapCount: number;
  driverCount: number;
  minTyreAge: number;
  maxTyreAge: number;
  medianAdjustedPace?: number | null;
  slopeSecondsPerTyreLap?: number | null;
  slopeConfidenceInterval95?: { lower: number | null; upper: number | null } | null;
  projectedLossNext5Laps?: number | null;
  projectedLossNext10Laps?: number | null;
  tyreCliffProbability?: number | null;
  byTyreAge: Array<{
    tyreAge: number;
    sampleCount: number;
    rawLapTimeMedian: number;
    adjustedPaceMean: number;
    adjustedPaceMedian: number;
    confidenceInterval95: { lower: number | null; upper: number | null };
  }>;
};

export type F1TyreDegradationAnalytics = {
  version: number;
  generatedAt: string;
  status: string;
  method: string;
  adjustments: string[];
  sampleCount: number;
  excludedCount: number;
  filters: Record<string, number>;
  compounds: F1TyreDegradationCompound[];
  compoundCrossovers: Array<Record<string, unknown>>;
  cleanLapSample: Array<Record<string, unknown>>;
};

export type F1WeatherEvolutionAnalytics = {
  version: number;
  generatedAt: string;
  status: string;
  sampleCount: number;
  latest?: {
    eventTime?: string | null;
    airTemperature?: number | null;
    trackTemperature?: number | null;
    rainfall?: number | null;
    windSpeed?: number | null;
    sourceId?: number | null;
  } | null;
  trackTemperatureDelta?: number | null;
  airTemperatureDelta?: number | null;
  windSpeedDelta?: number | null;
  rainfallDetected: boolean;
  maxRainfall?: number | null;
  series: Array<{
    index: number;
    eventTime?: string | null;
    airTemperature?: number | null;
    trackTemperature?: number | null;
    rainfall?: number | null;
    windSpeed?: number | null;
    sourceId?: number | null;
  }>;
};

export type F1PaceAnalysisAnalytics = {
  version: number;
  generatedAt: string;
  status: string;
  method?: string;
  driverCount: number;
  fieldSeries: Array<{
    lap: number;
    sampleCount: number;
    medianLapTime?: number | null;
    spreadSeconds?: number | null;
  }>;
  drivers: Array<{
    driverNumber: number;
    acronym?: string | null;
    teamName?: string | null;
    position?: number | null;
    compound?: string | null;
    lapCount: number;
    firstLap: number;
    lastLap: number;
    bestLapTime?: number | null;
    lastLapTime?: number | null;
    averageLapTime?: number | null;
    medianLapTime?: number | null;
    rollingMedianLast3?: number | null;
    consistencyStdSeconds?: number | null;
    trendLastVsFirst?: number | null;
    lapTimes: Array<{ lap: number; lapTime?: number | null }>;
  }>;
};

export type F1BattleDriverRef = {
  driverNumber: number;
  acronym?: string | null;
  teamName?: string | null;
  position?: number | null;
  compound?: string | null;
  tyreAge?: number | null;
};

export type F1BattleDashboardAnalytics = {
  version: number;
  generatedAt: string;
  status: string;
  method?: string;
  battleCount: number;
  activeOvertakeWindows: number;
  battles: Array<{
    ahead: F1BattleDriverRef;
    chaser: F1BattleDriverRef;
    gapSeconds?: number | null;
    recentPaceDeltaSeconds?: number | null;
    chaserDrsActive: boolean;
    tyreAgeDelta?: number | null;
    overtakeWindowProbability: number;
    windowState: "active" | "building" | "distant" | string;
    reason: string;
  }>;
  drsTrains: Array<{
    size: number;
    leader: F1BattleDriverRef;
    drivers: F1BattleDriverRef[];
    maxAdjacentGapSeconds?: number | null;
  }>;
};

export type F1AnalyticsResponse = {
  sessionKey: string | number;
  enabled: boolean;
  kind?: string;
  analytics: {
    projection_summary?: Record<string, unknown>;
    tyre_degradation_v1?: F1TyreDegradationAnalytics;
    weather_evolution_v1?: F1WeatherEvolutionAnalytics;
    pace_analysis_v1?: F1PaceAnalysisAnalytics;
    battle_dashboard_v1?: F1BattleDashboardAnalytics;
  };
  updatedAt?: Record<string, string>;
};

export async function getF1PlatformSessions(): Promise<F1SessionSummary[]> {
  const response = await fetch(`${F1_PLATFORM_API_BASE}/api/f1/sessions`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`F1 platform sessions failed: ${response.status}`);
  }
  const payload = (await response.json()) as { sessions?: F1SessionSummary[] };
  return payload.sessions ?? [];
}

export async function getF1PlatformSnapshot(
  sessionKey: string | number
): Promise<F1SessionSnapshot> {
  const response = await fetch(
    `${F1_PLATFORM_API_BASE}/api/f1/sessions/${encodeURIComponent(String(sessionKey))}/snapshot`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    throw new Error(`F1 platform snapshot failed: ${response.status}`);
  }
  return (await response.json()) as F1SessionSnapshot;
}

export async function getF1SessionAnalytics(
  sessionKey: string | number
): Promise<F1AnalyticsResponse> {
  const response = await fetch(
    `${F1_PLATFORM_API_BASE}/api/f1/sessions/${encodeURIComponent(String(sessionKey))}/analytics`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    throw new Error(`F1 session analytics failed: ${response.status}`);
  }
  return (await response.json()) as F1AnalyticsResponse;
}

export async function getF1TrackGeometry(
  sessionKey: string | number,
  centerlineSessionKey?: string | null
): Promise<F1TrackGeometryResponse> {
  const params = new URLSearchParams({ limit: "900" });
  if (centerlineSessionKey?.trim()) {
    params.set("centerline_session_key", centerlineSessionKey.trim());
  }
  const response = await fetch(
    `${F1_PLATFORM_API_BASE}/api/f1/sessions/${encodeURIComponent(String(sessionKey))}/track-geometry?${params.toString()}`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    throw new Error(`F1 track geometry failed: ${response.status}`);
  }
  return (await response.json()) as F1TrackGeometryResponse;
}

export async function resetF1PlatformReplay(
  sessionKey: string | number
): Promise<F1SessionSnapshot> {
  const response = await fetch(
    `${F1_PLATFORM_API_BASE}/api/f1/sessions/${encodeURIComponent(String(sessionKey))}/replay/reset`,
    { method: "POST" }
  );
  if (!response.ok) {
    throw new Error(`F1 replay reset failed: ${response.status}`);
  }
  return (await response.json()) as F1SessionSnapshot;
}

export async function importOpenF1Session(
  payload: OpenF1ImportRequest
): Promise<OpenF1ImportResponse> {
  const response = await fetch(`${F1_PLATFORM_API_BASE}/api/f1/openf1/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response, `OpenF1 import failed: ${response.status}`));
  }
  return (await response.json()) as OpenF1ImportResponse;
}

export async function getOpenF1SessionStatus(params: {
  year?: number | null;
  now?: string | null;
} = {}): Promise<OpenF1SessionResolution> {
  const query = new URLSearchParams();
  if (params.year) query.set("year", String(params.year));
  if (params.now) query.set("now", params.now);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  const response = await fetch(`${F1_PLATFORM_API_BASE}/api/f1/openf1/session-status${suffix}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response, `OpenF1 session status failed: ${response.status}`));
  }
  return (await response.json()) as OpenF1SessionResolution;
}

export async function getF1SessionStatus(params: {
  year?: number | null;
  now?: string | null;
} = {}): Promise<F1SessionResolution> {
  const query = new URLSearchParams();
  if (params.year) query.set("year", String(params.year));
  if (params.now) query.set("now", params.now);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  const response = await fetch(`${F1_PLATFORM_API_BASE}/api/f1/session-status${suffix}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response, `F1 session status failed: ${response.status}`));
  }
  return (await response.json()) as F1SessionResolution;
}

export async function importFastF1Session(payload: FastF1ImportRequest): Promise<FastF1ImportResponse> {
  const response = await fetch(`${F1_PLATFORM_API_BASE}/api/f1/fastf1/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `FastF1 import failed: ${response.status}`);
  }
  return (await response.json()) as FastF1ImportResponse;
}

export async function getFastF1Artifacts(params: {
  sessionKey?: string | number | null;
  kind?: string | null;
  limit?: number;
} = {}): Promise<FastF1ArtifactListResponse> {
  const query = new URLSearchParams();
  if (params.sessionKey) query.set("session_key", String(params.sessionKey));
  if (params.kind) query.set("kind", params.kind);
  if (params.limit) query.set("limit", String(params.limit));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  const response = await fetch(`${F1_PLATFORM_API_BASE}/api/f1/fastf1/artifacts${suffix}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`FastF1 artifacts failed: ${response.status}`);
  }
  return (await response.json()) as FastF1ArtifactListResponse;
}

export async function getFastF1Schedule(year?: number | null): Promise<FastF1ScheduleResponse> {
  const query = new URLSearchParams();
  if (year) query.set("year", String(year));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  const response = await fetch(`${F1_PLATFORM_API_BASE}/api/f1/fastf1/schedule${suffix}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response, `FastF1 schedule failed: ${response.status}`));
  }
  return (await response.json()) as FastF1ScheduleResponse;
}

export async function getF1SeasonSummary(year?: number | null): Promise<F1SeasonSummaryResponse> {
  const selectedYear = year ?? new Date().getFullYear();
  const baseUrl = F1_SEASON_API_BASE.replace(/\/+$/, "");
  const seasonUrl = `${baseUrl}/${encodeURIComponent(String(selectedYear))}`;
  const [driverPayload, constructorPayload, resultPayload] = await Promise.all([
    fetchF1SeasonJson(`${seasonUrl}/driverstandings.json`),
    fetchF1SeasonJson(`${seasonUrl}/constructorstandings.json`),
    fetchF1SeasonJson(`${seasonUrl}/last/results.json`),
  ]);
  const latestRace = f1SeasonLatestRace(resultPayload);
  const round =
    f1SeasonOptionalInteger(seasonNested(driverPayload, "MRData", "StandingsTable", "round")) ??
    latestRace?.round ??
    null;

  return {
    source: "jolpica-ergast",
    sourceUrl: `${seasonUrl}/`,
    generatedAt: new Date().toISOString(),
    year: selectedYear,
    round,
    latestRace,
    driverStandings: f1SeasonStandingsList(driverPayload, "DriverStandings").map(f1SeasonDriverStanding),
    constructorStandings: f1SeasonStandingsList(constructorPayload, "ConstructorStandings").map(f1SeasonConstructorStanding),
  };
}

export async function getF1CircuitHistory(params: {
  year?: number | null;
  roundNumber?: number | null;
} = {}): Promise<F1CircuitHistoryResponse> {
  const selectedYear = params.year ?? new Date().getFullYear();
  const roundNumber = params.roundNumber;
  if (!roundNumber) {
    throw new Error("F1 circuit history requires a current round number");
  }

  const baseUrl = F1_SEASON_API_BASE.replace(/\/+$/, "");
  const currentRacePayload = await fetchF1SeasonJson(
    `${baseUrl}/${encodeURIComponent(String(selectedYear))}/${encodeURIComponent(String(roundNumber))}.json`
  );
  const currentRace = f1SeasonRaceFromPayload(currentRacePayload);
  const circuitId = currentRace.circuitId;
  if (!circuitId) {
    throw new Error(`F1 circuit history could not resolve circuit for ${selectedYear} round ${roundNumber}`);
  }

  const historyYear = selectedYear - 1;
  const circuitUrl = `${baseUrl}/${encodeURIComponent(String(historyYear))}/circuits/${encodeURIComponent(circuitId)}`;
  const [qualifyingPayload, racePayload] = await Promise.all([
    fetchF1SeasonJson(`${circuitUrl}/qualifying.json`),
    fetchF1SeasonJson(`${circuitUrl}/results.json`),
  ]);
  const qualifyingRace = f1SeasonRaceFromPayload(qualifyingPayload);
  const race = f1SeasonRaceFromPayload(racePayload);

  return {
    source: "jolpica-ergast",
    sourceUrl: `${circuitUrl}/`,
    generatedAt: new Date().toISOString(),
    currentYear: selectedYear,
    year: historyYear,
    round: race.round ?? qualifyingRace.round ?? null,
    circuitId,
    circuitName: currentRace.circuitName ?? race.circuitName ?? qualifyingRace.circuitName ?? null,
    raceName: race.raceName ?? qualifyingRace.raceName ?? currentRace.raceName ?? null,
    date: race.date ?? qualifyingRace.date ?? null,
    qualifyingResults: f1SeasonArray(qualifyingRace.raw?.QualifyingResults).map(f1SeasonQualifyingResult),
    raceResults: f1SeasonArray(race.raw?.Results).map(f1SeasonRaceResult),
  };
}

export async function getF1WeatherForecast(params: {
  location?: string | null;
  year?: number | null;
  forecastDays?: number;
} = {}): Promise<F1WeatherForecastResponse> {
  const query = new URLSearchParams();
  if (params.location) query.set("location", params.location);
  if (params.year) query.set("year", String(params.year));
  if (params.forecastDays) query.set("forecast_days", String(params.forecastDays));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  const response = await fetch(`${F1_PLATFORM_API_BASE}/api/f1/weather/forecast${suffix}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response, `F1 weather forecast failed: ${response.status}`));
  }
  return (await response.json()) as F1WeatherForecastResponse;
}

export async function getFastF1ArtifactRows(
  artifactId: string,
  limit = 120
): Promise<FastF1ArtifactRowsResponse> {
  const query = new URLSearchParams({ limit: String(limit) });
  const response = await fetch(
    `${F1_PLATFORM_API_BASE}/api/f1/fastf1/artifacts/${encodeURIComponent(artifactId)}/rows?${query.toString()}`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `FastF1 artifact preview failed: ${response.status}`);
  }
  return (await response.json()) as FastF1ArtifactRowsResponse;
}

export async function getFastF1EngineeringSummary(
  sessionKey?: string | number | null
): Promise<FastF1EngineeringSummary> {
  const query = new URLSearchParams();
  if (sessionKey !== null && sessionKey !== undefined && String(sessionKey).trim()) {
    query.set("session_key", String(sessionKey));
  }
  const suffix = query.toString() ? `?${query.toString()}` : "";
  const response = await fetch(`${F1_PLATFORM_API_BASE}/api/f1/fastf1/engineering-summary${suffix}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `FastF1 engineering summary failed: ${response.status}`);
  }
  return (await response.json()) as FastF1EngineeringSummary;
}

type F1SeasonJsonRecord = Record<string, unknown>;

async function fetchF1SeasonJson(url: string): Promise<F1SeasonJsonRecord> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response, `F1 season API failed: ${response.status}`));
  }
  const payload = (await response.json()) as unknown;
  if (!isF1SeasonRecord(payload)) {
    throw new Error("F1 season API returned malformed JSON");
  }
  return payload;
}

function f1SeasonStandingsList(payload: F1SeasonJsonRecord, key: "DriverStandings" | "ConstructorStandings"): F1SeasonJsonRecord[] {
  const lists = f1SeasonArray(seasonNested(payload, "MRData", "StandingsTable", "StandingsLists"));
  return f1SeasonArray(lists[0]?.[key]);
}

function f1SeasonLatestRace(payload: F1SeasonJsonRecord): F1SeasonSummaryResponse["latestRace"] {
  const races = f1SeasonArray(seasonNested(payload, "MRData", "RaceTable", "Races"));
  const race = races[0];
  if (!race) return null;

  const circuit = f1SeasonRecord(race.Circuit);
  const location = f1SeasonRecord(circuit.Location);

  return {
    season: f1SeasonOptionalInteger(race.season),
    round: f1SeasonOptionalInteger(race.round),
    raceName: f1SeasonOptionalString(race.raceName),
    date: f1SeasonOptionalString(race.date),
    time: f1SeasonOptionalString(race.time),
    circuitId: f1SeasonOptionalString(circuit.circuitId),
    circuitName: f1SeasonOptionalString(circuit.circuitName),
    location: f1SeasonOptionalString(location.locality),
    country: f1SeasonOptionalString(location.country),
    results: f1SeasonArray(race.Results).map(f1SeasonRaceResult),
  };
}

function f1SeasonRaceFromPayload(payload: F1SeasonJsonRecord): {
  raw: F1SeasonJsonRecord | null;
  season: number | null;
  round: number | null;
  raceName: string | null;
  date: string | null;
  time: string | null;
  circuitId: string | null;
  circuitName: string | null;
  location: string | null;
  country: string | null;
} {
  const race = f1SeasonArray(seasonNested(payload, "MRData", "RaceTable", "Races"))[0] ?? null;
  const circuit = f1SeasonRecord(race?.Circuit);
  const location = f1SeasonRecord(circuit.Location);

  return {
    raw: race,
    season: f1SeasonOptionalInteger(race?.season),
    round: f1SeasonOptionalInteger(race?.round),
    raceName: f1SeasonOptionalString(race?.raceName),
    date: f1SeasonOptionalString(race?.date),
    time: f1SeasonOptionalString(race?.time),
    circuitId: f1SeasonOptionalString(circuit.circuitId),
    circuitName: f1SeasonOptionalString(circuit.circuitName),
    location: f1SeasonOptionalString(location.locality),
    country: f1SeasonOptionalString(location.country),
  };
}

function f1SeasonDriverStanding(row: F1SeasonJsonRecord): F1SeasonDriverRow {
  const driver = f1SeasonRecord(row.Driver);
  const constructor = f1SeasonArray(row.Constructors)[0] ?? {};

  return {
    position: f1SeasonOptionalInteger(row.position),
    points: f1SeasonOptionalNumber(row.points),
    wins: f1SeasonOptionalInteger(row.wins),
    ...f1SeasonDriverPayload(driver),
    ...f1SeasonConstructorPayload(constructor),
  };
}

function f1SeasonConstructorStanding(row: F1SeasonJsonRecord): F1SeasonConstructorRow {
  const constructor = f1SeasonRecord(row.Constructor);

  return {
    position: f1SeasonOptionalInteger(row.position),
    points: f1SeasonOptionalNumber(row.points),
    wins: f1SeasonOptionalInteger(row.wins),
    ...f1SeasonConstructorPayload(constructor),
  };
}

function f1SeasonRaceResult(row: F1SeasonJsonRecord): F1SeasonRaceResultRow {
  const driver = f1SeasonRecord(row.Driver);
  const constructor = f1SeasonRecord(row.Constructor);
  const fastestLap = f1SeasonRecord(row.FastestLap);
  const fastestLapTime = f1SeasonRecord(fastestLap.Time);

  return {
    position: f1SeasonOptionalInteger(row.position),
    positionText: f1SeasonOptionalString(row.positionText),
    points: f1SeasonOptionalNumber(row.points),
    grid: f1SeasonOptionalInteger(row.grid),
    laps: f1SeasonOptionalInteger(row.laps),
    status: f1SeasonOptionalString(row.status),
    time: f1SeasonOptionalString(seasonNested(row, "Time", "time")),
    gap: f1SeasonResultGap(row),
    fastestLap: f1SeasonOptionalString(fastestLapTime.time),
    ...f1SeasonDriverPayload(driver),
    ...f1SeasonConstructorPayload(constructor),
  };
}

function f1SeasonQualifyingResult(row: F1SeasonJsonRecord): F1SeasonQualifyingResultRow {
  const driver = f1SeasonRecord(row.Driver);
  const constructor = f1SeasonRecord(row.Constructor);
  const q1 = f1SeasonOptionalString(row.Q1);
  const q2 = f1SeasonOptionalString(row.Q2);
  const q3 = f1SeasonOptionalString(row.Q3);
  const time = q3 ?? q2 ?? q1;

  return {
    position: f1SeasonOptionalInteger(row.position),
    q1,
    q2,
    q3,
    time,
    gap: time,
    ...f1SeasonDriverPayload(driver),
    ...f1SeasonConstructorPayload(constructor),
  };
}

function f1SeasonDriverPayload(driver: F1SeasonJsonRecord): Pick<
  F1SeasonDriverRow,
  "driverId" | "driverNumber" | "code" | "givenName" | "familyName" | "fullName" | "nationality"
> {
  const givenName = f1SeasonOptionalString(driver.givenName);
  const familyName = f1SeasonOptionalString(driver.familyName);
  const fullName = [givenName, familyName].filter(Boolean).join(" ").trim() || null;

  return {
    driverId: f1SeasonOptionalString(driver.driverId),
    driverNumber: f1SeasonOptionalInteger(driver.permanentNumber),
    code: f1SeasonOptionalString(driver.code) ?? f1SeasonCodeFromName(givenName, familyName),
    givenName,
    familyName,
    fullName,
    nationality: f1SeasonOptionalString(driver.nationality),
  };
}

function f1SeasonConstructorPayload(constructor: F1SeasonJsonRecord): Pick<
  F1SeasonConstructorRow,
  "constructorId" | "constructorName" | "constructorNationality" | "teamColour"
> {
  const constructorId = f1SeasonOptionalString(constructor.constructorId);

  return {
    constructorId,
    constructorName: f1SeasonOptionalString(constructor.name),
    constructorNationality: f1SeasonOptionalString(constructor.nationality),
    teamColour: F1_TEAM_COLOURS[constructorId ?? ""] ?? "4B6FD8",
  };
}

function f1SeasonResultGap(row: F1SeasonJsonRecord): string {
  const status = f1SeasonOptionalString(row.status);
  if (status && status !== "Finished") return status;
  return f1SeasonOptionalString(seasonNested(row, "Time", "time")) ?? status ?? "-";
}

function seasonNested(payload: unknown, ...path: string[]): unknown {
  let current = payload;
  for (const part of path) {
    if (!isF1SeasonRecord(current)) return null;
    current = current[part];
  }
  return current;
}

function f1SeasonRecord(value: unknown): F1SeasonJsonRecord {
  return isF1SeasonRecord(value) ? value : {};
}

function f1SeasonArray(value: unknown): F1SeasonJsonRecord[] {
  return Array.isArray(value) ? value.filter(isF1SeasonRecord) : [];
}

function isF1SeasonRecord(value: unknown): value is F1SeasonJsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function f1SeasonOptionalString(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  return text || null;
}

function f1SeasonOptionalInteger(value: unknown): number | null {
  const number = f1SeasonOptionalNumber(value);
  return number === null ? null : Math.trunc(number);
}

function f1SeasonOptionalNumber(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" && !value.trim()) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function f1SeasonCodeFromName(givenName: string | null, familyName: string | null): string | null {
  const source = familyName ?? givenName;
  return source ? source.slice(0, 3).toUpperCase() : null;
}

async function responseErrorMessage(response: Response, fallback: string): Promise<string> {
  const text = await response.text();
  if (!text) return fallback;
  try {
    const parsed = JSON.parse(text) as { detail?: unknown; message?: unknown; error?: unknown };
    for (const value of [parsed.detail, parsed.message, parsed.error]) {
      if (typeof value === "string" && value.trim()) {
        return value;
      }
    }
  } catch {
    return text;
  }
  return text;
}

export async function startF1TimedReplay(
  sessionKey: string | number,
  payload: { speed: number; max_delay_seconds?: number }
): Promise<F1ReplayStatus> {
  const response = await fetch(
    `${F1_PLATFORM_API_BASE}/api/f1/sessions/${encodeURIComponent(String(sessionKey))}/replay/start`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }
  );
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Timed replay start failed: ${response.status}`);
  }
  return (await response.json()) as F1ReplayStatus;
}

export async function stopF1TimedReplay(sessionKey: string | number): Promise<F1ReplayStatus> {
  const response = await fetch(
    `${F1_PLATFORM_API_BASE}/api/f1/sessions/${encodeURIComponent(String(sessionKey))}/replay/stop`,
    { method: "POST" }
  );
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Timed replay stop failed: ${response.status}`);
  }
  return (await response.json()) as F1ReplayStatus;
}

export async function getF1TimedReplayStatus(sessionKey: string | number): Promise<F1ReplayStatus> {
  const response = await fetch(
    `${F1_PLATFORM_API_BASE}/api/f1/sessions/${encodeURIComponent(String(sessionKey))}/replay/status`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    throw new Error(`Timed replay status failed: ${response.status}`);
  }
  return (await response.json()) as F1ReplayStatus;
}

export function f1PlatformStreamUrl(sessionKey: string | number): string {
  const configured = process.env.NEXT_PUBLIC_F1_PLATFORM_WS_URL;
  if (configured) {
    return `${configured.replace(/\/$/, "")}/api/f1/sessions/${encodeURIComponent(
      String(sessionKey)
    )}/stream`;
  }
  const base = F1_PLATFORM_API_BASE.replace(/^http/, "ws").replace(/\/$/, "");
  return `${base}/api/f1/sessions/${encodeURIComponent(String(sessionKey))}/stream`;
}
