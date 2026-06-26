"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  F1_PLATFORM_API_BASE,
  f1PlatformStreamUrl,
  getFastF1ArtifactRows,
  getFastF1Artifacts,
  getF1TimedReplayStatus,
  getF1SessionAnalytics,
  getF1PlatformSessions,
  getF1SessionStatus,
  getF1PlatformSnapshot,
  getF1TrackGeometry,
  getFastF1EngineeringSummary,
  importFastF1Session,
  importOpenF1Session,
  resetF1PlatformReplay,
  startF1TimedReplay,
  stopF1TimedReplay,
  type FastF1ArtifactRecord,
  type FastF1ArtifactRowsResponse,
  type FastF1EngineeringSummary,
  type F1AnalyticsResponse,
  type F1CustomMicroSectorPassage,
  type F1DriverState,
  type F1LapPoint,
  type F1PredictionSnapshot,
  type F1ReplayStatus,
  type F1SessionInfo,
  type F1SessionResolution,
  type F1SessionSummary,
  type F1SessionSnapshot,
  type F1StreamUpdate,
  type F1TrackGeometryResponse,
} from "@/lib/f1Platform";

const MEANINGFUL_REFRESH_EVENTS = new Set([
  "lap.updated",
  "position.updated",
  "interval.updated",
  "stint.updated",
  "pit.updated",
  "race_control.updated",
  "weather.updated",
  "session_result.updated",
]);

type F1RaceTab = "standings" | "lapChart" | "engineer";
type F1SessionMode = "live" | "selected";
type F1ConnectionStatus = "resolving" | "connecting" | "live" | "polling" | "scheduled" | "upcoming" | "error";

const F1_RACE_TABS: Array<{ id: F1RaceTab; label: string }> = [
  { id: "standings", label: "Standings" },
  { id: "lapChart", label: "Lap Chart" },
  { id: "engineer", label: "Engineer Dashboard" },
];

const F1_LIVE_RACE_TABS: Array<{ id: F1RaceTab; label: string }> = [
  { id: "standings", label: "Standings" },
  { id: "lapChart", label: "Lap Chart" },
  { id: "engineer", label: "Practice Lab" },
];

type LiveLapTarget = {
  driverNumber: number;
  lap: number;
  lapTime: number;
};

type LiveTelemetryMetric = "speed" | "delta" | "throttle" | "brake" | "gear";

type TelemetryStats = {
  topSpeed: number | null;
  avgSpeed: number | null;
  fullThrottlePercent: number | null;
  brakingEvents: number | null;
};

const LIVE_TELEMETRY_OPTIONS: Array<{ id: LiveTelemetryMetric; label: string; unit: string; height: number; stepped?: boolean }> = [
  { id: "speed", label: "Speed", unit: "km/h", height: 300 },
  { id: "delta", label: "Delta", unit: "s", height: 150 },
  { id: "throttle", label: "Throttle", unit: "%", height: 150 },
  { id: "brake", label: "Brake", unit: "state", height: 130, stepped: true },
  { id: "gear", label: "N Gear", unit: "gear", height: 150, stepped: true },
];

const DEFAULT_LIVE_TELEMETRY_VISIBLE: Record<LiveTelemetryMetric, boolean> = {
  speed: true,
  delta: true,
  throttle: true,
  brake: true,
  gear: true,
};

type F1LivePlatformDashboardProps = {
  initialTab?: F1RaceTab;
  surfaceTitle?: string;
  surfaceStatus?: string;
  showOperationsControls?: boolean;
  sessionMode?: F1SessionMode;
};

export default function F1LivePlatformDashboard({
  initialTab = "standings",
  surfaceTitle = "F1 Live Dashboard",
  surfaceStatus = "Near-live timing and session operations",
  showOperationsControls = true,
  sessionMode = "live",
}: F1LivePlatformDashboardProps = {}) {
  const useSelectedSessionMode = sessionMode === "selected";
  const [sessionKey, setSessionKey] = useState("");
  const [activeTab, setActiveTab] = useState<F1RaceTab>(initialTab);
  const [snapshot, setSnapshot] = useState<F1SessionSnapshot | null>(null);
  const [analytics, setAnalytics] = useState<F1AnalyticsResponse | null>(null);
  const [selectedDriver, setSelectedDriver] = useState<number | null>(null);
  const [status, setStatus] = useState<F1ConnectionStatus>(useSelectedSessionMode ? "polling" : "resolving");
  const [error, setError] = useState<string | null>(null);
  const [sessionResolution, setSessionResolution] = useState<F1SessionResolution | null>(null);
  const [resolvingSession, setResolvingSession] = useState(!useSelectedSessionMode);
  const [autoSessionMessage, setAutoSessionMessage] = useState<string | null>(null);
  const [availableSessions, setAvailableSessions] = useState<F1SessionSummary[]>([]);
  const [lastUpdate, setLastUpdate] = useState<F1StreamUpdate | null>(null);
  const [importYear, setImportYear] = useState(String(new Date().getFullYear()));
  const [importMeetingKey, setImportMeetingKey] = useState("");
  const [importSessionKey, setImportSessionKey] = useState("");
  const [importing, setImporting] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState("20");
  const [replayStatus, setReplayStatus] = useState<F1ReplayStatus | null>(null);
  const [replayBusy, setReplayBusy] = useState(false);
  const [fastF1Year, setFastF1Year] = useState("2024");
  const [fastF1Event, setFastF1Event] = useState("Austria");
  const [fastF1Session, setFastF1Session] = useState("R");
  const [fastF1Drivers, setFastF1Drivers] = useState("VER, RUS");
  const [fastF1Output, setFastF1Output] = useState<"jsonl" | "parquet">("jsonl");
  const [fastF1IncludeTelemetry, setFastF1IncludeTelemetry] = useState(true);
  const [artifactSessionKey, setArtifactSessionKey] = useState("");
  const [fastF1Artifacts, setFastF1Artifacts] = useState<FastF1ArtifactRecord[]>([]);
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const [artifactRows, setArtifactRows] = useState<FastF1ArtifactRowsResponse | null>(null);
  const [engineeringSummary, setEngineeringSummary] = useState<FastF1EngineeringSummary | null>(null);
  const [trackGeometry, setTrackGeometry] = useState<F1TrackGeometryResponse | null>(null);
  const [liveLapDriverNumbers, setLiveLapDriverNumbers] = useState<number[]>([]);
  const [liveLapTargets, setLiveLapTargets] = useState<LiveLapTarget[]>([]);
  const [liveTelemetryVisible, setLiveTelemetryVisible] = useState<Record<LiveTelemetryMetric, boolean>>(
    DEFAULT_LIVE_TELEMETRY_VISIBLE
  );
  const [fastF1Busy, setFastF1Busy] = useState(false);
  const [fastF1Error, setFastF1Error] = useState<string | null>(null);
  const refreshTimer = useRef<number | null>(null);
  const autoImportedSession = useRef<string | null>(null);

  useEffect(() => {
    setActiveTab(initialTab);
  }, [initialTab]);

  const clearResolvedSessionState = useCallback(() => {
    setSessionKey("");
    setSnapshot(null);
    setAnalytics(null);
    setEngineeringSummary(null);
    setTrackGeometry(null);
    setLastUpdate(null);
    setSelectedDriver(null);
  }, []);

  const primeFastF1ControlsFromSession = useCallback((session?: F1SessionInfo | null) => {
    if (!session) return;
    setFastF1Year(sessionYear(session));
    const round = typeof session.round_number === "number" ? session.round_number : null;
    const eventName = typeof session.fastf1_event_name === "string" ? session.fastf1_event_name : null;
    setFastF1Event(round !== null ? String(round) : eventName ?? session.location ?? "");
    const fastF1SessionName = typeof session.fastf1_session_name === "string" ? session.fastf1_session_name : null;
    setFastF1Session(fastF1SessionName ?? session.session_name ?? "R");
    const resolvedSessionKey = sessionKeyFromF1Session(session);
    if (resolvedSessionKey.startsWith("fastf1:")) {
      setArtifactSessionKey(resolvedSessionKey);
    }
  }, []);

  const useLocalEngineerSessionFallback = useCallback(async (message: string) => {
    if (activeTab !== "engineer") return false;
    const sessions = await getF1PlatformSessions().catch(() => []);
    setAvailableSessions(sessions);
    const fallback = [...sessions].sort((a, b) => {
      const eventDelta = (b.eventCount ?? 0) - (a.eventCount ?? 0);
      if (eventDelta !== 0) return eventDelta;
      return (b.drivers ?? 0) - (a.drivers ?? 0);
    })[0];
    if (!fallback) return false;
    setSessionKey(String(fallback.sessionKey));
    setStatus("polling");
    setAutoSessionMessage(`${message} Showing local session ${fallback.sessionKey}.`);
    return true;
  }, [activeTab]);

  const selectPlatformSession = useCallback((session: F1SessionSummary | null) => {
    if (!session) {
      clearResolvedSessionState();
      setSessionResolution(null);
      setStatus("error");
      setAutoSessionMessage("No imported or replay sessions are available yet.");
      return;
    }
    setSessionKey(String(session.sessionKey));
    setSessionResolution(null);
    setAutoSessionMessage(`Selected ${formatSessionSummaryTitle(session)}.`);
    setStatus("polling");
    setError(null);
  }, [clearResolvedSessionState]);

  const loadAvailablePlatformSessions = useCallback(async () => {
    const sessions = await getF1PlatformSessions();
    setAvailableSessions(sessions);
    return sessions;
  }, []);

  const resolveCurrentSession = useCallback(async () => {
    setResolvingSession(true);
    setStatus("resolving");
    setError(null);
    try {
      const resolution = await getF1SessionStatus();
      setSessionResolution(resolution);

      if (resolution.status === "live" && resolution.session) {
        primeFastF1ControlsFromSession(resolution.session);
        const key = sessionKeyFromF1Session(resolution.session);
        if (key && isOpenF1Source(resolution.source)) {
          setImportSessionKey(key);
          setImportMeetingKey(sessionMeetingKey(resolution.session));
          setImportYear(sessionYear(resolution.session));
          setAutoSessionMessage(`Connecting to ${formatF1SessionTitle(resolution.session)}.`);
          setStatus("connecting");
          return;
        }
        if (await useLocalEngineerSessionFallback(`Live session from ${sessionSourceLabel(resolution.source)}: ${formatF1SessionTitle(resolution.session)}.`)) {
          return;
        }
        clearResolvedSessionState();
        setStatus("scheduled");
        setAutoSessionMessage(`Live session from ${sessionSourceLabel(resolution.source)}: ${formatF1SessionTitle(resolution.session)}.`);
        return;
      }

      if (resolution.status === "upcoming" && resolution.nextSession) {
        primeFastF1ControlsFromSession(resolution.nextSession);
        if (await useLocalEngineerSessionFallback(`Next session: ${formatF1SessionTitle(resolution.nextSession)}.`)) {
          return;
        }
        clearResolvedSessionState();
        setStatus("upcoming");
        setAutoSessionMessage(`Next session: ${formatF1SessionTitle(resolution.nextSession)}.`);
        return;
      }

      clearResolvedSessionState();
      setStatus("error");
      setAutoSessionMessage(resolution.message);
      setError(resolution.message);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to resolve F1 live session";
      setSessionResolution({
        status: "unavailable",
        source: "f1-session-resolver",
        resolvedAt: new Date().toISOString(),
        message,
        session: null,
        nextSession: null,
        secondsUntilStart: null,
        secondsUntilEnd: null,
      });
      clearResolvedSessionState();
      setStatus("error");
      setAutoSessionMessage(message);
      setError(message);
    } finally {
      setResolvingSession(false);
    }
  }, [clearResolvedSessionState, primeFastF1ControlsFromSession, useLocalEngineerSessionFallback]);

  useEffect(() => {
    if (!useSelectedSessionMode) {
      void resolveCurrentSession();
    }
  }, [resolveCurrentSession, useSelectedSessionMode]);

  useEffect(() => {
    if (!useSelectedSessionMode) return;
    let cancelled = false;
    const loadSelectedSession = async () => {
      setResolvingSession(true);
      setStatus("polling");
      setError(null);
      try {
        const sessions = await getF1PlatformSessions();
        if (cancelled) return;
        setAvailableSessions(sessions);
        const selected = bestSessionSummary(sessions);
        selectPlatformSession(selected);
      } catch (err) {
        if (cancelled) return;
        clearResolvedSessionState();
        setStatus("error");
        setError(err instanceof Error ? err.message : "Unable to load F1 session list");
      } finally {
        if (!cancelled) setResolvingSession(false);
      }
    };
    void loadSelectedSession();
    return () => {
      cancelled = true;
    };
  }, [clearResolvedSessionState, selectPlatformSession, useSelectedSessionMode]);

  const loadSnapshot = useCallback(async () => {
    const normalizedSessionKey = sessionKey.trim();
    if (!normalizedSessionKey) {
      setSnapshot(null);
      setAnalytics(null);
      setEngineeringSummary(null);
      setTrackGeometry(null);
      setSelectedDriver(null);
      return;
    }
    try {
      const centerlineSessionKey = artifactSessionKey.trim() || null;
      const shouldLoadTrackGeometry = Boolean(
        centerlineSessionKey ||
          normalizedSessionKey.startsWith("fastf1:") ||
          (!useSelectedSessionMode && (showOperationsControls || (activeTab === "engineer" && initialTab !== "standings")))
      );
      const trackGeometryPromise =
        shouldLoadTrackGeometry
          ? getF1TrackGeometry(normalizedSessionKey, centerlineSessionKey).catch(() => null)
          : Promise.resolve(null);
      const [next, nextAnalytics, nextEngineeringSummary, nextTrackGeometry] = await Promise.all([
        getF1PlatformSnapshot(normalizedSessionKey),
        getF1SessionAnalytics(normalizedSessionKey).catch(() => null),
        getFastF1EngineeringSummary(artifactSessionKey.trim() || null).catch(() => null),
        trackGeometryPromise,
      ]);
      setSnapshot(next);
      if (useSelectedSessionMode) {
        const fallbackSummary = enrichSessionSummaryFromSnapshot(
          {
            sessionKey: normalizedSessionKey,
            seq: next.seq,
            source: next.source,
            drivers: next.drivers.length,
            eventCount: 0,
          },
          next
        );
        setAutoSessionMessage(`Selected ${formatSessionSummaryTitle(fallbackSummary)}.`);
        setAvailableSessions((current) =>
          current.map((session) =>
            String(session.sessionKey) === normalizedSessionKey ? enrichSessionSummaryFromSnapshot(session, next) : session
          )
        );
      }
      setAnalytics(nextAnalytics);
      setEngineeringSummary(nextEngineeringSummary);
      setTrackGeometry(nextTrackGeometry);
      setSelectedDriver((current) => current ?? next.drivers[0]?.driver_number ?? null);
      setError(null);
      setStatus((current) => (useSelectedSessionMode ? "polling" : current === "live" ? "live" : "polling"));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load F1 platform snapshot");
      setStatus("error");
    }
  }, [activeTab, artifactSessionKey, initialTab, sessionKey, showOperationsControls, useSelectedSessionMode]);

  const loadAnalyticsForSession = useCallback(async (key: string | number) => {
    try {
      setAnalytics(await getF1SessionAnalytics(key));
    } catch {
      setAnalytics(null);
    }
  }, []);

  useEffect(() => {
    if (useSelectedSessionMode) {
      return;
    }
    const liveSession = sessionResolution?.status === "live" ? sessionResolution.session : null;
    if (!liveSession || !isOpenF1Source(sessionResolution?.source)) {
      return;
    }
    const liveSessionKey = sessionKeyFromF1Session(liveSession);
    if (!liveSessionKey || autoImportedSession.current === liveSessionKey) {
      return;
    }
    autoImportedSession.current = liveSessionKey;
    let cancelled = false;
    const bootstrapLiveSession = async () => {
      setImporting(true);
      try {
        const imported = await importOpenF1Session({
          session_key: liveSessionKey,
          meeting_key: numberOrNull(sessionMeetingKey(liveSession)),
          year: numberOrNull(sessionYear(liveSession)),
          session_name: String(liveSession?.session_name ?? "Race"),
          include_telemetry: false,
          limit_per_topic: 5000,
        });
        if (cancelled) return;
        setSessionKey(String(imported.sessionKey));
        setSnapshot(imported.snapshot);
        setSelectedDriver(imported.snapshot.drivers[0]?.driver_number ?? null);
        void loadAnalyticsForSession(imported.sessionKey);
        setLastUpdate({
          seq: imported.snapshot.seq,
          type: "openf1.live_bootstrap",
          eventTime: imported.snapshot.generatedAt,
          driverNumber: null,
          payload: {
            eventCount: imported.eventCount,
            replayPath: imported.replayPath,
          },
        });
        setAutoSessionMessage(`Live OpenF1 session bootstrapped: ${formatF1SessionTitle(liveSession)}.`);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Unable to bootstrap current OpenF1 session");
        setStatus("error");
      } finally {
        if (!cancelled) {
          setImporting(false);
        }
      }
    };
    void bootstrapLiveSession();
    return () => {
      cancelled = true;
    };
  }, [loadAnalyticsForSession, sessionResolution, useSelectedSessionMode]);

  const loadFastF1Artifacts = useCallback(
    async (key = artifactSessionKey) => {
      try {
        let response = await getFastF1Artifacts({
          sessionKey: key.trim() || null,
          limit: 80,
        });
        if (!response.artifacts.length && key.trim()) {
          response = await getFastF1Artifacts({ limit: 80 });
        }
        setFastF1Artifacts(response.artifacts);
        setSelectedArtifactId((current) => {
          if (current && response.artifacts.some((artifact) => artifact.artifactId === current)) {
            return current;
          }
          return response.artifacts[0]?.artifactId ?? null;
        });
        if (!response.artifacts.length) {
          setArtifactRows(null);
        }
        const summary = await getFastF1EngineeringSummary(key.trim() || null).catch(() =>
          getFastF1EngineeringSummary(null).catch(() => null)
        );
        setEngineeringSummary(summary);
        setFastF1Error(null);
      } catch (err) {
        setFastF1Error(err instanceof Error ? err.message : "Unable to load FastF1 artifacts");
      }
    },
    [artifactSessionKey]
  );

  useEffect(() => {
    void loadSnapshot();
  }, [loadSnapshot]);

  useEffect(() => {
    void loadFastF1Artifacts();
  }, [loadFastF1Artifacts]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedArtifactId) {
      setArtifactRows(null);
      return;
    }
    const loadRows = async () => {
      try {
        const rows = await getFastF1ArtifactRows(selectedArtifactId, 80);
        if (!cancelled) {
          setArtifactRows(rows);
          setFastF1Error(null);
        }
      } catch (err) {
        if (!cancelled) {
          setArtifactRows(null);
          setFastF1Error(err instanceof Error ? err.message : "Unable to preview FastF1 artifact");
        }
      }
    };
    void loadRows();
    return () => {
      cancelled = true;
    };
  }, [selectedArtifactId]);

  useEffect(() => {
    if (!sessionKey.trim()) {
      return;
    }
    const socket = new WebSocket(f1PlatformStreamUrl(sessionKey));
    socket.onopen = () => {
      setStatus(useSelectedSessionMode ? "polling" : "live");
      setError(null);
    };
    socket.onerror = () => {
      setStatus("polling");
    };
    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as F1StreamUpdate | SnapshotStreamMessage;
        if (isSnapshotStreamMessage(message)) {
          setSnapshot(message.payload);
          setSelectedDriver((current) => current ?? message.payload.drivers[0]?.driver_number ?? null);
          return;
        }
        setLastUpdate(message);
        if (MEANINGFUL_REFRESH_EVENTS.has(message.type)) {
          if (refreshTimer.current !== null) {
            window.clearTimeout(refreshTimer.current);
          }
          refreshTimer.current = window.setTimeout(() => {
            void loadSnapshot();
          }, 180);
        }
      } catch {
        setStatus("polling");
      }
    };
    socket.onclose = () => {
      setStatus((current) => (current === "live" ? "polling" : current));
    };
    return () => {
      if (refreshTimer.current !== null) {
        window.clearTimeout(refreshTimer.current);
        refreshTimer.current = null;
      }
      socket.close();
    };
  }, [loadSnapshot, sessionKey, useSelectedSessionMode]);

  useEffect(() => {
    if (!sessionKey.trim()) {
      return;
    }
    const interval = window.setInterval(() => {
      if (status !== "live") {
        void loadSnapshot();
      }
    }, 8000);
    return () => window.clearInterval(interval);
  }, [loadSnapshot, status]);

  useEffect(() => {
    if (!sessionKey.trim()) {
      setReplayStatus(null);
      return;
    }
    let cancelled = false;
    const loadReplayStatus = async () => {
      try {
        const next = await getF1TimedReplayStatus(sessionKey);
        if (!cancelled) setReplayStatus(next);
      } catch {
        if (!cancelled) setReplayStatus(null);
      }
    };
    void loadReplayStatus();
    const interval = window.setInterval(() => {
      if (replayStatus?.state === "running" || replayStatus?.state === "starting") {
        void loadReplayStatus();
      }
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [replayStatus?.state, sessionKey]);

  const selected = useMemo(
    () => snapshot?.drivers.find((driver) => driver.driver_number === selectedDriver) ?? snapshot?.drivers[0] ?? null,
    [selectedDriver, snapshot]
  );
  const isWaitingForData = !snapshot && status !== "error" && status !== "scheduled" && status !== "upcoming";
  const showSessionResolutionPanel =
    !useSelectedSessionMode &&
    !snapshot &&
    (status === "upcoming" ||
      status === "scheduled" ||
      status === "error" ||
      sessionResolution?.status === "live" ||
      sessionResolution?.status === "upcoming" ||
      sessionResolution?.status === "unavailable");
  const blockSessionContent = showSessionResolutionPanel && activeTab !== "engineer";

  const latestPredictions = useMemo(() => {
    if (!snapshot?.predictions.length) return [];
    const byDriver = new Map<number, F1PredictionSnapshot>();
    for (const prediction of snapshot.predictions) {
      byDriver.set(prediction.driver_number, prediction);
    }
    return Array.from(byDriver.values()).sort((a, b) => b.win_probability - a.win_probability);
  }, [snapshot]);

  const selectedLapPoints = useMemo(() => {
    if (!snapshot || !selected) return [];
    return snapshot.lapChart.filter((point) => point.driver_number === selected.driver_number);
  }, [selected, snapshot]);

  const comparisonDrivers = useMemo(() => {
    const drivers = snapshot?.drivers ?? [];
    if (!drivers.length) return [];
    const selectedIndex = selected
      ? drivers.findIndex((driver) => driver.driver_number === selected.driver_number)
      : 0;
    const start = Math.max(0, selectedIndex - 1);
    const local = drivers.slice(start, start + 4);
    const unique = new Map<number, F1DriverState>();
    for (const driver of [drivers[0], ...local, ...drivers.slice(0, 4)]) {
      if (driver) unique.set(driver.driver_number, driver);
      if (unique.size >= 4) break;
    }
    return Array.from(unique.values());
  }, [selected, snapshot]);
  const useLiveSessionSurface = !useSelectedSessionMode && !showOperationsControls && initialTab === "standings";
  const raceTabs = useLiveSessionSurface ? F1_LIVE_RACE_TABS : F1_RACE_TABS;

  useEffect(() => {
    if (!useLiveSessionSurface || !snapshot?.drivers.length) return;
    const driverNumbers = new Set(snapshot.drivers.map((driver) => driver.driver_number));
    setLiveLapDriverNumbers((current) => {
      if (current.length && current.every((driverNumber) => driverNumbers.has(driverNumber))) return current;
      const valid = current.filter((driverNumber) => driverNumbers.has(driverNumber));
      if (valid.length) return valid;
      return sortedDrivers(snapshot.drivers).slice(0, 3).map((driver) => driver.driver_number);
    });
    setLiveLapTargets((current) => {
      const valid = current.filter((target) =>
        driverNumbers.has(target.driverNumber) &&
        snapshot.lapChart.some((point) => point.driver_number === target.driverNumber && point.lap === target.lap)
      );
      if (valid.length >= 2) return valid.slice(0, 2);
      return defaultLiveLapTargets(snapshot, liveLapDriverNumbers.length ? liveLapDriverNumbers : undefined);
    });
  }, [liveLapDriverNumbers, snapshot, useLiveSessionSurface]);

  const resetReplay = async () => {
    try {
      const next = await resetF1PlatformReplay(sessionKey);
      setSnapshot(next);
      setSelectedDriver(next.drivers[0]?.driver_number ?? null);
      void loadAnalyticsForSession(sessionKey);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to reset replay");
    }
  };

  const importOpenF1 = async () => {
    setImporting(true);
    try {
      const imported = await importOpenF1Session({
        year: numberOrNull(importYear),
        meeting_key: numberOrNull(importMeetingKey),
        session_key: importSessionKey.trim() || null,
        session_name: "Race",
        include_telemetry: false,
        limit_per_topic: 3000,
      });
      setSessionKey(String(imported.sessionKey));
      setSnapshot(imported.snapshot);
      setSelectedDriver(imported.snapshot.drivers[0]?.driver_number ?? null);
      void loadAnalyticsForSession(imported.sessionKey);
      setLastUpdate({
        seq: imported.snapshot.seq,
        type: "openf1.imported",
        eventTime: imported.snapshot.generatedAt,
        driverNumber: null,
        payload: {
          eventCount: imported.eventCount,
          replayPath: imported.replayPath,
        },
      });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to import OpenF1 session");
      setStatus("error");
    } finally {
      setImporting(false);
    }
  };

  const importFastF1 = async () => {
    setFastF1Busy(true);
    try {
      const eventValue = fastF1Event.trim();
      if (!eventValue) {
        throw new Error("FastF1 event or round is required");
      }
      const imported = await importFastF1Session({
        year: numberOrNull(fastF1Year) ?? new Date().getFullYear(),
        event: /^\d+$/.test(eventValue) ? Number.parseInt(eventValue, 10) : eventValue,
        session_name: fastF1Session.trim() || "R",
        drivers: fastF1Drivers
          .split(",")
          .map((driver) => driver.trim())
          .filter(Boolean),
        include_telemetry: fastF1IncludeTelemetry,
        telemetry_laps_per_driver: 1,
        distance_step_meters: 5,
        output_format: fastF1Output,
        map_to_session_key: sessionKey,
      });
      setArtifactSessionKey(imported.sessionKey);
      setFastF1Artifacts(imported.artifacts);
      setSelectedArtifactId(imported.artifacts[0]?.artifactId ?? null);
      setEngineeringSummary(await getFastF1EngineeringSummary(imported.sessionKey).catch(() => null));
      setFastF1Error(imported.notes[0] ?? null);
    } catch (err) {
      setFastF1Error(err instanceof Error ? err.message : "Unable to import FastF1 artifacts");
    } finally {
      setFastF1Busy(false);
    }
  };

  const startTimedReplay = async () => {
    setReplayBusy(true);
    try {
      const next = await startF1TimedReplay(sessionKey, {
        speed: numberOrNull(replaySpeed) ?? 20,
        max_delay_seconds: 2,
      });
      setReplayStatus(next);
      setLastUpdate({
        seq: snapshot?.seq ?? 0,
        type: "replay.started",
        eventTime: next.startedAt ?? null,
        driverNumber: null,
        payload: {
          speed: next.speed,
          eventCount: next.eventCount,
        },
      });
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start timed replay");
      setStatus("error");
    } finally {
      setReplayBusy(false);
    }
  };

  const stopTimedReplay = async () => {
    setReplayBusy(true);
    try {
      const next = await stopF1TimedReplay(sessionKey);
      setReplayStatus(next);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to stop timed replay");
      setStatus("error");
    } finally {
      setReplayBusy(false);
    }
  };

  return (
    <div className={useLiveSessionSurface ? "stack-lg f1-live-session-shell" : "stack-lg"}>
      {!useLiveSessionSurface ? (
        <section className="f1-platform-hero">
          <div>
            <h1 className="page-title">{surfaceTitle}</h1>
            <p className="page-status">{surfaceStatus}</p>
          </div>
          <div className="f1-platform-controls">
            {useSelectedSessionMode ? (
              <>
                <label className="f1-session-control wide">
                  <span>Analysis session</span>
                  <select
                    value={sessionKey}
                    disabled={resolvingSession}
                    onChange={(event) => {
                      const selectedSummary =
                        availableSessions.find((session) => String(session.sessionKey) === event.target.value) ?? null;
                      selectPlatformSession(selectedSummary);
                    }}
                  >
                    <option value="">{resolvingSession ? "Loading sessions" : "Select imported session"}</option>
                    {availableSessions.map((session) => (
                      <option value={String(session.sessionKey)} key={String(session.sessionKey)}>
                        {formatSessionSummaryOption(session)}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="button secondary button-sm"
                  onClick={async () => {
                    setResolvingSession(true);
                    try {
                      const sessions = await loadAvailablePlatformSessions();
                      if (!sessionKey.trim()) selectPlatformSession(bestSessionSummary(sessions));
                      setError(null);
                    } catch (err) {
                      setError(err instanceof Error ? err.message : "Unable to refresh session list");
                    } finally {
                      setResolvingSession(false);
                    }
                  }}
                  disabled={resolvingSession}
                >
                  {resolvingSession ? "Loading" : "Refresh sessions"}
                </button>
              </>
            ) : (
              <>
                <label className="f1-session-control">
                  <span>Session</span>
                  <input
                    value={sessionKey}
                    placeholder={resolvingSession ? "resolving live session" : "manual session key"}
                    onChange={(event) => {
                      setSessionKey(event.target.value);
                      setSessionResolution(null);
                      setAutoSessionMessage("Manual session key selected.");
                    }}
                  />
                </label>
                <button
                  type="button"
                  className="button secondary button-sm"
                  onClick={() => void resolveCurrentSession()}
                  disabled={resolvingSession}
                >
                  {resolvingSession ? "Checking live" : "Check live"}
                </button>
              </>
            )}
            {sessionKey.trim() ? (
              <button type="button" className="button secondary button-sm" onClick={() => void loadSnapshot()}>
                Reload session
              </button>
            ) : null}
            {showOperationsControls ? (
              <button type="button" className="button button-sm" onClick={() => void resetReplay()} disabled={!sessionKey.trim()}>
                Replay reset
              </button>
            ) : null}
          </div>
        </section>
      ) : null}

      {showOperationsControls ? (
        <>
          <section className="f1-import-bar">
            <label className="f1-session-control">
              <span>OpenF1 year</span>
              <input value={importYear} onChange={(event) => setImportYear(event.target.value)} />
            </label>
            <label className="f1-session-control">
              <span>Meeting key</span>
              <input value={importMeetingKey} onChange={(event) => setImportMeetingKey(event.target.value)} />
            </label>
            <label className="f1-session-control">
              <span>Session key</span>
              <input value={importSessionKey} onChange={(event) => setImportSessionKey(event.target.value)} />
            </label>
            <button type="button" className="button secondary button-sm" onClick={() => void importOpenF1()} disabled={importing}>
              {importing ? "Importing" : "Import OpenF1"}
            </button>
          </section>

          <section className="f1-import-bar">
            <label className="f1-session-control">
              <span>Replay speed</span>
              <input value={replaySpeed} onChange={(event) => setReplaySpeed(event.target.value)} />
            </label>
            <button
              type="button"
              className="button secondary button-sm"
              onClick={() => void startTimedReplay()}
              disabled={replayBusy || !sessionKey.trim()}
            >
              Start timed replay
            </button>
            <button
              type="button"
              className="button button-sm"
              onClick={() => void stopTimedReplay()}
              disabled={replayBusy || !sessionKey.trim()}
            >
              Stop replay
            </button>
            <span className="status-item">
              Replay {replayStatus ? `${replayStatus.state} ${replayStatus.cursor}/${replayStatus.eventCount}` : "idle"}
            </span>
          </section>
        </>
      ) : null}

      {!useLiveSessionSurface ? (
        <div className="status-strip">
          <span className="status-item">
            <span className={`status-dot ${statusDotClass(status)}`} />
            {status}
          </span>
          <span className="status-item">Seq {snapshot?.seq ?? 0}</span>
          <span className="status-item">Source {snapshot?.source ?? "pending"}</span>
          <span className="status-item">Session {formatSessionInfo(snapshot?.sessionInfo)}</span>
          <span className="status-item">Last {lastUpdate?.type ?? "snapshot"}</span>
          {autoSessionMessage ? <span className="status-item">{autoSessionMessage}</span> : null}
          {!snapshot ? <span className="status-item">API {F1_PLATFORM_API_BASE}</span> : null}
          {error ? <span className="status-item f1-status-error">{error}</span> : null}
        </div>
      ) : null}

      {showSessionResolutionPanel ? (
        <SessionResolutionPanel
          resolution={sessionResolution}
          error={error}
          onRefresh={useLiveSessionSurface ? () => void resolveCurrentSession() : undefined}
          refreshing={resolvingSession}
        />
      ) : null}

      {blockSessionContent ? null : (
        <>
      {useLiveSessionSurface ? (
        <LiveSessionOverview
          snapshot={snapshot}
          sessionResolution={sessionResolution}
          status={status}
          resolvingSession={resolvingSession}
          trackGeometry={trackGeometry}
          selectedDriver={selected?.driver_number ?? null}
          onSelect={setSelectedDriver}
          onCheckLive={() => void resolveCurrentSession()}
        />
      ) : (
        <section className="dashboard-kpis f1-kpis">
          <Metric label="Leader" value={driverLabel(snapshot?.drivers[0])} supporting={snapshot?.drivers[0]?.team_name ?? "No state"} />
          <Metric label="Drivers" value={String(snapshot?.drivers.length ?? 0)} supporting="Reduced current state" />
          <Metric label="Pit stops" value={String(snapshot?.pitStops.length ?? 0)} supporting="Strategy events" />
          <Metric label="Overtakes" value={String(snapshot?.overtakes.length ?? 0)} supporting="Battle timeline" />
          <Metric label="Results" value={String(snapshot?.sessionResults?.length ?? 0)} supporting="Final classifications" />
          <Metric label="Micro sectors" value={String(snapshot?.customMicroSectors?.length ?? 0)} supporting="Custom progress timing" />
          <Metric label="Weather" value={formatWeather(snapshot?.weather)} supporting={`${snapshot?.weatherSamples?.length ?? 0} retained samples`} />
        </section>
      )}

      <div className={`f1-race-tabs ${useLiveSessionSurface ? "f1-live-tabs" : ""}`} role="tablist" aria-label="F1 race views">
        {raceTabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`f1-race-tab ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === "standings" ? (
        <section className="f1-tab-panel">
          <div className="panel f1-live-feed-panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">Live Timing Feed</h2>
                <span className="module-subtitle">Updated {snapshot ? formatRelativeTime(snapshot.generatedAt) : "pending"}</span>
              </div>
              <span className="status-item">{snapshot?.sessionInfo?.location ? String(snapshot.sessionInfo.location) : "Race mode"}</span>
            </div>
            <div className="panel-body panel-body-dense">
              {isWaitingForData ? (
                <div className="f1-inline-notice">
                  Loading local F1 data from {F1_PLATFORM_API_BASE}
                </div>
              ) : null}
              {status === "error" && !snapshot ? (
                <div className="f1-inline-notice error">
                  F1 API is not reachable from this browser. Start the platform API on port 8001 or set NEXT_PUBLIC_F1_PLATFORM_API_URL.
                </div>
              ) : null}
              <TimingTable
                drivers={snapshot?.drivers ?? []}
                selectedDriver={selected?.driver_number ?? null}
                onSelect={setSelectedDriver}
              />
            </div>
          </div>

          <div className="f1-standings-lower-grid">
            <div className="panel">
              <div className="panel-header">
                <div className="panel-header-left">
                  <h2 className="module-title">Track Map</h2>
                  <span className="module-subtitle">Approximate progress and selected car</span>
                </div>
              </div>
              <div className="panel-body">
                <TrackMap
                  drivers={snapshot?.drivers ?? []}
                  selectedDriver={selected?.driver_number ?? null}
                  geometry={trackGeometry}
                  microSectors={snapshot?.customMicroSectors ?? []}
                  onSelect={setSelectedDriver}
                />
              </div>
            </div>

            <div className="panel">
              <div className="panel-header">
                <div className="panel-header-left">
                  <h2 className="module-title">Battle Dashboard</h2>
                  <span className="module-subtitle">Adjacent gaps and overtake windows</span>
                </div>
              </div>
              <div className="panel-body">
                <BattleDashboardPanel analytics={analytics} />
              </div>
            </div>

            <div className="panel">
              <div className="panel-header">
                <div className="panel-header-left">
                  <h2 className="module-title">Race Control</h2>
                  <span className="module-subtitle">Flags, restarts and control messages</span>
                </div>
              </div>
              <div className="panel-body">
                <RaceControl messages={snapshot?.raceControl ?? []} />
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {activeTab === "lapChart" ? (
        <section className="f1-tab-panel">
          {useLiveSessionSurface ? (
            <LiveLapComparisonPanel
              snapshot={snapshot}
              selectedDriverNumbers={liveLapDriverNumbers}
              compareTargets={liveLapTargets}
              telemetryVisible={liveTelemetryVisible}
              engineeringSummary={engineeringSummary}
              onSelectedDriverNumbersChange={setLiveLapDriverNumbers}
              onCompareTargetsChange={setLiveLapTargets}
              onTelemetryVisibleChange={setLiveTelemetryVisible}
            />
          ) : (
            <>
              <div className="panel">
                <div className="panel-header">
                  <div className="panel-header-left">
                    <h2 className="module-title">Lap Time Evolution</h2>
                    <span className="module-subtitle">Multi-driver line chart with selected comparison set</span>
                  </div>
                </div>
                <div className="panel-body">
                  <LapChart points={snapshot?.lapChart ?? selectedLapPoints} drivers={comparisonDrivers} />
                </div>
              </div>

              <div className="grid-two">
                <div className="stack">
                  <div className="panel">
                    <div className="panel-header">
                      <div className="panel-header-left">
                        <h2 className="module-title">Strategy Timeline</h2>
                        <span className="module-subtitle">Stints, compounds and tyre age</span>
                      </div>
                    </div>
                    <div className="panel-body">
                      <StrategyTimeline snapshot={snapshot} selectedDriver={selected?.driver_number ?? null} />
                    </div>
                  </div>

                  <div className="panel">
                    <div className="panel-header">
                      <div className="panel-header-left">
                        <h2 className="module-title">Pace Analysis</h2>
                        <span className="module-subtitle">Reduced lap pace, consistency and trend</span>
                      </div>
                    </div>
                    <div className="panel-body">
                      <PaceAnalysisPanel analytics={analytics} />
                    </div>
                  </div>

                  <div className="panel">
                    <div className="panel-header">
                      <div className="panel-header-left">
                        <h2 className="module-title">Tyre Degradation</h2>
                        <span className="module-subtitle">Adjusted clean-lap trend</span>
                      </div>
                    </div>
                    <div className="panel-body">
                      <TyreDegradationPanel analytics={analytics} />
                    </div>
                  </div>
                </div>

                <div className="stack">
                  <div className="panel">
                    <div className="panel-header">
                      <div className="panel-header-left">
                        <h2 className="module-title">Prediction Evolution</h2>
                        <span className="module-subtitle">Latest persisted model snapshots</span>
                      </div>
                    </div>
                    <div className="panel-body">
                      <PredictionList predictions={latestPredictions} drivers={snapshot?.drivers ?? []} />
                    </div>
                  </div>

                  <div className="panel">
                    <div className="panel-header">
                      <div className="panel-header-left">
                        <h2 className="module-title">Weather Evolution</h2>
                        <span className="module-subtitle">Track temperature, rain and wind trend</span>
                      </div>
                    </div>
                    <div className="panel-body">
                      <WeatherEvolutionPanel snapshot={snapshot} analytics={analytics} />
                    </div>
                  </div>

                  <div className="panel">
                    <div className="panel-header">
                      <div className="panel-header-left">
                        <h2 className="module-title">Custom Micro-Sectors</h2>
                        <span className="module-subtitle">Track-progress derived, not official mini-sectors</span>
                      </div>
                    </div>
                    <div className="panel-body">
                      <MicroSectorPanel snapshot={snapshot} selectedDriver={selected?.driver_number ?? null} />
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </section>
      ) : null}

      {activeTab === "engineer" ? (
        <section className="f1-tab-panel">
          {useLiveSessionSurface ? (
            <PracticeLabPanel snapshot={snapshot} />
          ) : (
            <>
          <EngineerDashboard
            snapshot={snapshot}
            selectedDriver={selected?.driver_number ?? null}
            comparisonDrivers={comparisonDrivers}
            engineeringSummary={engineeringSummary}
            trackGeometry={trackGeometry}
            onSelect={setSelectedDriver}
          />

          <div className="panel">
            <div className="panel-header">
              <div className="panel-header-left">
                <h2 className="module-title">Engineering Artifacts</h2>
                <span className="module-subtitle">FastF1 distance-aligned telemetry, centreline and delta previews</span>
              </div>
            </div>
            <div className="panel-body">
              <FastF1ArtifactsPanel
                engineeringSummary={engineeringSummary}
                year={fastF1Year}
                event={fastF1Event}
                session={fastF1Session}
                drivers={fastF1Drivers}
                output={fastF1Output}
                includeTelemetry={fastF1IncludeTelemetry}
                artifactSessionKey={artifactSessionKey}
                artifacts={fastF1Artifacts}
                selectedArtifactId={selectedArtifactId}
                artifactRows={artifactRows}
                busy={fastF1Busy}
                error={fastF1Error}
                onYearChange={setFastF1Year}
                onEventChange={setFastF1Event}
                onSessionChange={setFastF1Session}
                onDriversChange={setFastF1Drivers}
                onOutputChange={setFastF1Output}
                onIncludeTelemetryChange={setFastF1IncludeTelemetry}
                onArtifactSessionKeyChange={setArtifactSessionKey}
                onImport={() => void importFastF1()}
                onRefresh={() => void loadFastF1Artifacts()}
                onSelectArtifact={setSelectedArtifactId}
              />
            </div>
          </div>
            </>
          )}
        </section>
      ) : null}
        </>
      )}
    </div>
  );
}

type TelemetryDriverSeries = {
  name: string;
  color: string;
  speed: Array<[number, number]>;
  delta: Array<[number, number]>;
  throttle: Array<[number, number]>;
  brake: Array<[number, number]>;
  gear: Array<[number, number]>;
};

type TelemetryMetric = LiveTelemetryMetric;

function EngineerDashboard({
  snapshot,
  selectedDriver,
  comparisonDrivers,
  engineeringSummary,
  trackGeometry,
  onSelect,
}: {
  snapshot: F1SessionSnapshot | null;
  selectedDriver: number | null;
  comparisonDrivers: F1DriverState[];
  engineeringSummary: FastF1EngineeringSummary | null;
  trackGeometry: F1TrackGeometryResponse | null;
  onSelect: (driverNumber: number) => void;
}) {
  const drivers = comparisonDrivers.length ? comparisonDrivers : snapshot?.drivers.slice(0, 4) ?? [];
  const telemetry = buildTelemetrySeries(engineeringSummary, drivers);

  return (
    <div className="f1-engineer-dashboard">
      <div className="f1-engineer-top">
        <div className="f1-engineer-driver-grid">
          {drivers.map((driver) => (
            <button
              key={driver.driver_number}
              type="button"
              className={`f1-engineer-driver-card ${driver.driver_number === selectedDriver ? "selected" : ""}`}
              onClick={() => onSelect(driver.driver_number)}
            >
              <span className="f1-team-swatch" style={{ background: teamColor(driver) }} />
              <span className="f1-engineer-driver-main">
                <strong>{driverLabel(driver)}</strong>
                <small>{driver.team_name ?? "Unknown team"}</small>
              </span>
              <span className="f1-engineer-driver-metrics">
                <span>
                  <small>Lap</small>
                  <strong>{formatLap(driver.last_lap_time)}</strong>
                </span>
                <span>
                  <small>S1</small>
                  <strong>{formatSector(driver.sector_times?.sector_1)}</strong>
                </span>
                <span>
                  <small>S2</small>
                  <strong>{formatSector(driver.sector_times?.sector_2)}</strong>
                </span>
                <span>
                  <small>S3</small>
                  <strong>{formatSector(driver.sector_times?.sector_3)}</strong>
                </span>
                <span>
                  <small>Air</small>
                  <strong>{formatWeather(snapshot?.weather)}</strong>
                </span>
              </span>
            </button>
          ))}
        </div>

        <div className="panel f1-engineer-track-panel">
          <div className="panel-body">
            <TrackMap
              drivers={snapshot?.drivers ?? []}
              selectedDriver={selectedDriver}
              geometry={trackGeometry}
              microSectors={snapshot?.customMicroSectors ?? []}
              onSelect={onSelect}
            />
          </div>
        </div>
      </div>

      <div className="f1-engineer-chart-stack">
        <TelemetryChart title="Speed" unit="km/h" metric="speed" telemetry={telemetry} height={280} />
        <TelemetryChart title="Delta" unit="s" metric="delta" telemetry={telemetry} height={150} />
        <TelemetryChart title="Throttle" unit="%" metric="throttle" telemetry={telemetry} height={150} />
        <TelemetryChart title="Brake" unit="state" metric="brake" telemetry={telemetry} height={130} stepped />
        <TelemetryChart title="N Gear" unit="gear" metric="gear" telemetry={telemetry} height={150} stepped />
      </div>
    </div>
  );
}

function TelemetryChart({
  title,
  unit,
  metric,
  telemetry,
  height,
  stepped = false,
}: {
  title: string;
  unit: string;
  metric: TelemetryMetric;
  telemetry: TelemetryDriverSeries[];
  height: number;
  stepped?: boolean;
}) {
  if (!telemetry.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No telemetry series</span></div>;
  }
  const option = f1ChartOption({
    grid: { left: 56, right: 24, top: 28, bottom: 24 },
    title: {
      text: `${title} (${unit})`,
      left: 0,
      top: 0,
      textStyle: {
        color: "#fefefe",
        fontSize: 11,
        fontFamily: "JetBrains Mono",
        fontWeight: 700,
      },
    },
    xAxis: {
      type: "value",
      name: "Distance",
      axisLabel: { formatter: (value: number) => `${Math.round(value)}` },
    },
    yAxis: {
      type: "value",
      scale: metric !== "throttle" && metric !== "brake",
      min: metric === "throttle" || metric === "brake" ? 0 : undefined,
      max: metric === "throttle" ? 100 : metric === "brake" ? 1 : undefined,
    },
    tooltip: {
      trigger: "axis",
      valueFormatter: (value: number | null) => (isNumber(value) ? formatNumber(value) : "-"),
    },
    series: telemetry.map((driver) => ({
      name: driver.name,
      type: "line",
      data: driver[metric],
      showSymbol: false,
      step: stepped ? "middle" : false,
      lineStyle: { width: metric === "speed" ? 2 : 1.5, color: driver.color },
      itemStyle: { color: driver.color },
    })),
  });
  return (
    <div className="f1-chart-shell">
      <ReactECharts option={option} style={{ height, width: "100%" }} notMerge lazyUpdate />
    </div>
  );
}

function defaultLiveLapTargets(snapshot: F1SessionSnapshot, selectedDriverNumbers?: number[]): LiveLapTarget[] {
  const selected = new Set(selectedDriverNumbers ?? []);
  const candidatePoints = snapshot.lapChart
    .filter((point) => !selected.size || selected.has(point.driver_number))
    .filter((point) => isNumber(point.value));
  const fastestByDriver = new Map<number, F1LapPoint>();
  for (const point of candidatePoints) {
    const current = fastestByDriver.get(point.driver_number);
    if (!current || point.value < current.value) {
      fastestByDriver.set(point.driver_number, point);
    }
  }
  const targets = [...fastestByDriver.values()]
    .sort((left, right) => left.value - right.value)
    .slice(0, 2)
    .map((point) => ({ driverNumber: point.driver_number, lap: point.lap, lapTime: point.value }));
  if (targets.length >= 2) return targets;
  return candidatePoints
    .sort((left, right) => left.value - right.value)
    .slice(0, 2)
    .map((point) => ({ driverNumber: point.driver_number, lap: point.lap, lapTime: point.value }));
}

function nextLapTargets(current: LiveLapTarget[], target: LiveLapTarget): LiveLapTarget[] {
  const existingIndex = current.findIndex((item) => item.driverNumber === target.driverNumber && item.lap === target.lap);
  if (existingIndex >= 0) {
    return current.filter((_, index) => index !== existingIndex);
  }
  return [...current, target].slice(-2);
}

function fastestDriversByLap(points: F1LapPoint[], drivers: F1DriverState[]): F1DriverState[] {
  const bestByDriver = bestLapByDriver(points);
  return [...drivers].sort((left, right) => {
    const leftBest = bestByDriver.get(left.driver_number) ?? Number.POSITIVE_INFINITY;
    const rightBest = bestByDriver.get(right.driver_number) ?? Number.POSITIVE_INFINITY;
    if (leftBest !== rightBest) return leftBest - rightBest;
    return (left.position ?? 10_000) - (right.position ?? 10_000);
  });
}

function bestLapByDriver(points: F1LapPoint[]): Map<number, number> {
  const bestByDriver = new Map<number, number>();
  for (const point of points) {
    const current = bestByDriver.get(point.driver_number);
    if (!isNumber(current) || point.value < current) {
      bestByDriver.set(point.driver_number, point.value);
    }
  }
  return bestByDriver;
}

function liveLapTargetKey(target: Pick<LiveLapTarget, "driverNumber" | "lap">): string {
  return `${target.driverNumber}:${target.lap}`;
}

function liveLapTargetFromChartParam(params: unknown): LiveLapTarget | null {
  if (!isPlainObject(params) || !isPlainObject(params.data)) return null;
  const driverNumber = params.data.driverNumber;
  const lap = params.data.lap;
  const lapTime = params.data.lapTime;
  if (!isNumber(driverNumber) || !isNumber(lap) || !isNumber(lapTime)) return null;
  return { driverNumber, lap, lapTime };
}

function formatLiveLapTooltip(params: unknown, drivers: F1DriverState[]): string {
  const target = liveLapTargetFromChartParam(params);
  if (!target) return "";
  const driver = drivers.find((item) => item.driver_number === target.driverNumber);
  return [
    `<strong>${driver?.acronym ?? target.driverNumber} · Lap ${target.lap}</strong>`,
    `Lap time ${formatLap(target.lapTime)}`,
    driver?.team_name ? `Team ${driver.team_name}` : null,
  ].filter(Boolean).join("<br/>");
}

function telemetryStats(series?: TelemetryDriverSeries | null): TelemetryStats {
  if (!series) {
    return {
      topSpeed: null,
      avgSpeed: null,
      fullThrottlePercent: null,
      brakingEvents: null,
    };
  }
  const speeds = series.speed.map((point) => point[1]).filter(isNumber);
  const throttle = series.throttle.map((point) => point[1]).filter(isNumber);
  const brake = series.brake.map((point) => point[1]).filter(isNumber);
  return {
    topSpeed: speeds.length ? Math.max(...speeds) : null,
    avgSpeed: speeds.length ? speeds.reduce((sum, value) => sum + value, 0) / speeds.length : null,
    fullThrottlePercent: throttle.length
      ? (throttle.filter((value) => value >= 98).length / throttle.length) * 100
      : null,
    brakingEvents: brake.length ? countBrakeEvents(brake) : null,
  };
}

function countBrakeEvents(values: number[]): number {
  let events = 0;
  let braking = false;
  for (const value of values) {
    if (value > 0.2 && !braking) {
      events += 1;
      braking = true;
    } else if (value <= 0.2) {
      braking = false;
    }
  }
  return events;
}

function numericDelta(value: number | null, reference: number | null): number | null {
  return isNumber(value) && isNumber(reference) ? value - reference : null;
}

function formatTelemetryKmh(value?: number | null): string {
  return isNumber(value) ? `${value.toFixed(1)} km/h` : "N/A";
}

function formatTelemetryPercent(value?: number | null): string {
  return isNumber(value) ? `${value.toFixed(1)}%` : "N/A";
}

function formatTelemetryDelta(value: number | null, unit: string): string {
  return isNumber(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(1)} ${unit}` : "N/A";
}

function FastF1ArtifactsPanel({
  engineeringSummary,
  year,
  event,
  session,
  drivers,
  output,
  includeTelemetry,
  artifactSessionKey,
  artifacts,
  selectedArtifactId,
  artifactRows,
  busy,
  error,
  onYearChange,
  onEventChange,
  onSessionChange,
  onDriversChange,
  onOutputChange,
  onIncludeTelemetryChange,
  onArtifactSessionKeyChange,
  onImport,
  onRefresh,
  onSelectArtifact,
}: {
  engineeringSummary: FastF1EngineeringSummary | null;
  year: string;
  event: string;
  session: string;
  drivers: string;
  output: "jsonl" | "parquet";
  includeTelemetry: boolean;
  artifactSessionKey: string;
  artifacts: FastF1ArtifactRecord[];
  selectedArtifactId: string | null;
  artifactRows: FastF1ArtifactRowsResponse | null;
  busy: boolean;
  error: string | null;
  onYearChange: (value: string) => void;
  onEventChange: (value: string) => void;
  onSessionChange: (value: string) => void;
  onDriversChange: (value: string) => void;
  onOutputChange: (value: "jsonl" | "parquet") => void;
  onIncludeTelemetryChange: (value: boolean) => void;
  onArtifactSessionKeyChange: (value: string) => void;
  onImport: () => void;
  onRefresh: () => void;
  onSelectArtifact: (artifactId: string | null) => void;
}) {
  const previewColumns = artifactRows?.columns.slice(0, 8) ?? [];
  return (
    <div className="f1-artifact-workbench">
      <FastF1EngineeringSummaryPanel summary={engineeringSummary} />

      <div className="f1-artifact-controls">
        <label className="f1-session-control">
          <span>Year</span>
          <input value={year} onChange={(eventValue) => onYearChange(eventValue.target.value)} />
        </label>
        <label className="f1-session-control">
          <span>Event</span>
          <input value={event} onChange={(eventValue) => onEventChange(eventValue.target.value)} />
        </label>
        <label className="f1-session-control short">
          <span>Session</span>
          <input value={session} onChange={(eventValue) => onSessionChange(eventValue.target.value)} />
        </label>
        <label className="f1-session-control">
          <span>Drivers</span>
          <input value={drivers} onChange={(eventValue) => onDriversChange(eventValue.target.value)} />
        </label>
        <label className="f1-session-control short">
          <span>Format</span>
          <select value={output} onChange={(eventValue) => onOutputChange(eventValue.target.value as "jsonl" | "parquet")}>
            <option value="jsonl">jsonl</option>
            <option value="parquet">parquet</option>
          </select>
        </label>
        <label className="f1-checkbox-control">
          <input
            type="checkbox"
            checked={includeTelemetry}
            onChange={(eventValue) => onIncludeTelemetryChange(eventValue.target.checked)}
          />
          Telemetry
        </label>
        <button type="button" className="button secondary button-sm" onClick={onImport} disabled={busy}>
          {busy ? "Importing" : "Import FastF1"}
        </button>
      </div>

      <div className="f1-artifact-controls">
        <label className="f1-session-control wide">
          <span>Artifact session</span>
          <input
            value={artifactSessionKey}
            placeholder="fastf1:2026:austria:r"
            onChange={(eventValue) => onArtifactSessionKeyChange(eventValue.target.value)}
          />
        </label>
        <button type="button" className="button button-sm" onClick={onRefresh} disabled={busy}>
          Refresh artifacts
        </button>
        {error ? <span className="status-item f1-status-error">{error}</span> : null}
      </div>

      <div className="f1-artifact-grid">
        <div className="f1-artifact-list" aria-label="FastF1 artifact list">
          {artifacts.length ? (
            artifacts.map((artifact) => (
              <button
                type="button"
                key={artifact.artifactId ?? artifact.path}
                className={`f1-artifact-item ${artifact.artifactId === selectedArtifactId ? "selected" : ""}`}
                onClick={() => onSelectArtifact(artifact.artifactId ?? null)}
                disabled={!artifact.artifactId}
              >
                <strong>{artifactLabel(artifact)}</strong>
                <span>{artifact.metadata.sessionKey ? String(artifact.metadata.sessionKey) : "unknown session"}</span>
                <small>
                  {artifact.format} · {artifact.row_count ?? "?"} rows
                </small>
              </button>
            ))
          ) : (
            <div className="empty-state compact"><span className="empty-state-text">No FastF1 artifacts indexed</span></div>
          )}
        </div>

        <div className="f1-artifact-preview">
          {artifactRows && previewColumns.length ? (
            <>
              <div className="f1-artifact-preview-head">
                <span>{artifactLabel(artifactRows.artifact)}</span>
                <span>
                  {artifactRows.rows.length} rows{artifactRows.truncated ? " previewed" : ""}
                </span>
              </div>
              <div className="f1-table-wrap">
                <table className="table f1-artifact-table">
                  <thead>
                    <tr>
                      {previewColumns.map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {artifactRows.rows.slice(0, 12).map((row, rowIndex) => (
                      <tr key={`${artifactRows.artifact.artifactId}-${rowIndex}`}>
                        {previewColumns.map((column) => (
                          <td key={column}>{formatArtifactValue(row[column])}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="empty-state compact"><span className="empty-state-text">Select an artifact preview</span></div>
          )}
        </div>
      </div>
    </div>
  );
}

function FastF1EngineeringSummaryPanel({ summary }: { summary: FastF1EngineeringSummary | null }) {
  const telemetry = summary?.telemetryDelta;
  const corners = summary?.cornerMetrics ?? [];
  if (!telemetry && !corners.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No FastF1 engineering artifacts summarized</span></div>;
  }
  const series = telemetry?.series.filter((point) => isNumber(point.distance) && isNumber(point.deltaSeconds)) ?? [];
  const maxAbsDelta = Math.max(0.001, ...series.map((point) => Math.abs(point.deltaSeconds ?? 0)));
  return (
    <div className="f1-engineering-summary">
      <div className="f1-engineering-head">
        <span>{summary?.sessionKey ?? "FastF1 session"}</span>
        <span>
          {summary?.artifactCounts.telemetryDelta ?? 0} delta / {summary?.artifactCounts.cornerMetrics ?? 0} corner files
        </span>
      </div>

      {telemetry ? (
        <div className="f1-telemetry-summary">
          <div className="f1-telemetry-meta">
            <strong>
              {telemetry.driverA ?? "A"} L{telemetry.lapA ?? "-"} vs {telemetry.driverB ?? "B"} L{telemetry.lapB ?? "-"}
            </strong>
            <span>Final {formatSeconds(telemetry.finalDeltaSeconds)}</span>
            <span>A gain {formatSeconds(telemetry.maxGainDriverASeconds)}</span>
            <span>B gain {formatSeconds(telemetry.maxGainDriverBSeconds)}</span>
            <span>Speed delta {formatNumber(telemetry.maxSpeedDeltaKmh)} km/h</span>
          </div>
          <div className="f1-delta-strip" aria-label="Telemetry delta over distance">
            {series.map((point, index) => {
              const delta = point.deltaSeconds ?? 0;
              const magnitude = Math.max(8, (Math.abs(delta) / maxAbsDelta) * 100);
              return (
                <span
                  key={`${point.distance}-${index}`}
                  className={`f1-delta-cell ${delta <= 0 ? "driver-a" : "driver-b"}`}
                  style={{ height: `${magnitude}%` }}
                  title={`${formatNumber(point.distance)}m ${formatSeconds(delta)}`}
                />
              );
            })}
          </div>
        </div>
      ) : null}

      {corners.length ? (
        <div className="f1-corner-summary-list">
          {corners.slice(0, 4).map((corner) => (
            <div className="f1-corner-summary-row" key={corner.artifact.artifactId ?? `${corner.driver}-${corner.lapNumber}`}>
              <span>
                <strong>{corner.driver ?? "Driver"}</strong>
                <small>L{corner.lapNumber ?? "-"}</small>
              </span>
              <span>{corner.cornerCount} corners</span>
              <span>Min {formatNumber(corner.slowestMinimumSpeedKmh)} km/h</span>
              <span>Fast {formatNumber(corner.fastestCornerTimeSeconds)}s</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function SessionResolutionPanel({
  resolution,
  error,
  onRefresh,
  refreshing = false,
}: {
  resolution: F1SessionResolution | null;
  error: string | null;
  onRefresh?: () => void;
  refreshing?: boolean;
}) {
  const liveSession = resolution?.status === "live" ? resolution.session : null;
  const nextSession = resolution?.nextSession ?? null;
  const displaySession = liveSession ?? nextSession ?? null;
  const isUpcoming = resolution?.status === "upcoming" && nextSession;
  const isLive = resolution?.status === "live" && liveSession;
  const sourceLabel = sessionSourceLabel(resolution?.source);
  const title = isLive ? "Live session detected" : isUpcoming ? "No live session right now" : "Live session unavailable";
  const primary = isLive
    ? `Live session: ${formatF1SessionTitle(liveSession)}`
    : isUpcoming
      ? `Next session: ${formatF1SessionTitle(nextSession)}`
      : "F1 session status could not be resolved.";
  const start = displaySession?.date_start ?? null;
  const end = displaySession?.date_end ?? null;
  const countdown =
    isLive && resolution?.secondsUntilEnd !== null && resolution?.secondsUntilEnd !== undefined
      ? `Ends in ${formatDurationFromSeconds(resolution.secondsUntilEnd)}`
      : isUpcoming && resolution?.secondsUntilStart !== null && resolution?.secondsUntilStart !== undefined
        ? `Starts in ${formatDurationFromSeconds(resolution.secondsUntilStart)}`
        : "Schedule pending";
  const message = error ?? resolution?.message ?? `Waiting for ${sourceLabel} status.`;

  return (
    <section className="panel f1-session-resolution">
      <div className="panel-header">
        <div className="panel-header-left">
          <h2 className="module-title">{title}</h2>
          <span className="module-subtitle">{primary}</span>
        </div>
        <div className="f1-resolution-actions">
          <span className="status-item">
            <span className={`status-dot ${isLive ? "ok" : isUpcoming ? "warn" : "miss"}`} />
            {resolution?.status ?? "checking"}
          </span>
          {onRefresh ? (
            <button type="button" className="button secondary button-sm" onClick={onRefresh} disabled={refreshing}>
              {refreshing ? "Checking" : "Check live"}
            </button>
          ) : null}
        </div>
      </div>
      <div className="panel-body">
        <div className="f1-session-resolution-grid">
          <div className="f1-session-resolution-cell">
            <span>Countdown</span>
            <strong>{countdown}</strong>
          </div>
          <div className="f1-session-resolution-cell">
            <span>Start</span>
            <strong>{formatSessionDate(start)}</strong>
          </div>
          <div className="f1-session-resolution-cell">
            <span>End</span>
            <strong>{formatSessionDate(end)}</strong>
          </div>
          <div className="f1-session-resolution-cell">
            <span>Location</span>
            <strong>{formatF1SessionLocation(displaySession)}</strong>
          </div>
        </div>
        <div className={`f1-session-resolution-message ${error ? "error" : ""}`}>{message}</div>
      </div>
    </section>
  );
}

function Metric({ label, value, supporting }: { label: string; value: string; supporting: string }) {
  return (
    <div className="kpi-card">
      <span className="kpi-label">{label}</span>
      <strong className="kpi-value">{value}</strong>
      <span className="kpi-subtext">{supporting}</span>
    </div>
  );
}

function LiveSessionOverview({
  snapshot,
  sessionResolution,
  status,
  resolvingSession,
  trackGeometry,
  selectedDriver,
  onSelect,
  onCheckLive,
}: {
  snapshot: F1SessionSnapshot | null;
  sessionResolution: F1SessionResolution | null;
  status: F1ConnectionStatus;
  resolvingSession: boolean;
  trackGeometry: F1TrackGeometryResponse | null;
  selectedDriver: number | null;
  onSelect: (driverNumber: number) => void;
  onCheckLive: () => void;
}) {
  const drivers = sortedDrivers(snapshot?.drivers ?? []);
  const weather = latestWeatherRecord(snapshot);
  const title = formatLiveEventName(snapshot, sessionResolution);
  const location = formatLiveEventLocation(snapshot, sessionResolution);
  const sessionName = formatLiveSessionName(snapshot, sessionResolution);
  const sourceLabel = trackGeometry ? "FastF1 map" : "cached map";
  const timingLabel = snapshot ? formatRelativeTime(snapshot.generatedAt) : "now";
  const sessionMeta = [sessionName, liveConnectionLabel(status), sourceLabel, timingLabel].filter(Boolean).join(" · ");

  return (
    <section className="f1-live-session-card">
      <div className="f1-live-session-header">
        <div>
          <span className="f1-live-eyebrow">Session Info</span>
          <h1>{title}</h1>
          <p>{[location, sessionMeta].filter((item) => item && item !== "-").join(" · ")}</p>
        </div>
        <button type="button" className="f1-live-check-button" onClick={onCheckLive} disabled={resolvingSession}>
          <span aria-hidden="true">R</span>
          {resolvingSession ? "Checking" : "Check live"}
        </button>
      </div>

      <div className="f1-live-overview-grid">
        <div className="f1-driver-tracker-panel">
          <div className="f1-live-panel-title">
            <h2>
              Driver Tracker <strong>{drivers.length}</strong>
            </h2>
            <span className="f1-live-telemetry-pill">Live Telemetry</span>
          </div>
          <TrackMap
            drivers={drivers}
            selectedDriver={selectedDriver}
            geometry={trackGeometry}
            microSectors={snapshot?.customMicroSectors ?? []}
            onSelect={onSelect}
            mode="tracker"
          />
        </div>

        <div className="f1-live-metrics-grid">
          <LiveStatusTile label="Track" value={formatLiveTrackStatus(snapshot)} tone="ok" size="wide" />
          <LiveStatusTile label={`${sessionName} clock`} value={formatLiveSessionClock(snapshot, sessionResolution)} size="wide" />
          <LiveStatusTile label="Air" value={formatTemperature(weatherNumber(weather, "air_temperature", "airTemperature"))} />
          <LiveStatusTile label="Track" value={formatTemperature(weatherNumber(weather, "track_temperature", "trackTemperature"))} />
          <LiveStatusTile label="Humidity" value={formatHumidity(weatherNumber(weather, "humidity"))} />
          <LiveStatusTile label="Rain" value={formatRainState(weather)} />
          <div className="f1-live-wind-card">
            <span className="f1-live-wind-compass" aria-hidden="true">
              <i />
            </span>
            <div>
              <span>Wind</span>
              <strong>{formatWindSpeed(weather)}</strong>
              <small>{formatWindDirection(weather)}</small>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function LiveStatusTile({
  label,
  value,
  tone,
  size,
}: {
  label: string;
  value: string;
  tone?: "ok";
  size?: "wide";
}) {
  return (
    <div className={`f1-live-status-tile ${tone ? `tone-${tone}` : ""} ${size === "wide" ? "wide" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PracticeLabPanel({ snapshot }: { snapshot: F1SessionSnapshot | null }) {
  const drivers = sortedDrivers(snapshot?.drivers ?? []);
  const fastest = drivers
    .map((driver) => practiceLapTime(driver))
    .filter(isNumber)
    .sort((left, right) => left - right)[0] ?? null;
  const timedRuns = drivers.filter((driver) => isNumber(practiceLapTime(driver))).length;
  const raceRuns = drivers.filter((driver) => (driver.current_lap ?? 0) >= 4).length;

  return (
    <div className="panel f1-practice-lab-panel">
      <div className="panel-header">
        <div className="panel-header-left">
          <h2 className="module-title">Practice Lab</h2>
          <span className="module-subtitle">One-lap simulations, long-run pace, degradation, and stint shape from live practice laps.</span>
        </div>
        <span className="module-subtitle">Updated {snapshot ? formatRelativeTime(snapshot.generatedAt) : "pending"}</span>
      </div>
      <div className="panel-body">
        <div className="f1-practice-summary-strip">
          <div>
            <span>Clock</span>
            <strong>{formatLiveSessionClock(snapshot, null)}</strong>
            <small>{formatLiveTrackStatus(snapshot)}</small>
          </div>
          <div>
            <span>Quali pace</span>
            <strong>{fastest ? formatLap(fastest) : "N/A"}</strong>
            <small>{fastest ? "Fastest timed lap" : "No timed laps"}</small>
          </div>
          <div>
            <span>Programmes</span>
            <strong>{timedRuns} quali sims · {raceRuns} race sims</strong>
            <small>{formatWeather(snapshot?.weather)}</small>
          </div>
        </div>

        <div className="f1-table-wrap">
          <table className="table f1-practice-table">
            <thead>
              <tr>
                <th>Pos</th>
                <th>Driver</th>
                <th>Programme</th>
                <th>Quali pace</th>
                <th>Race pace</th>
                <th>Deg</th>
                <th>Runs</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {drivers.map((driver) => {
                const confidence = practiceConfidence(driver);
                return (
                  <tr key={driver.driver_number}>
                    <td>P{driver.position ?? "-"}</td>
                    <td>
                      <span className="f1-driver-cell">
                        <span className="f1-team-swatch" style={{ background: teamColor(driver) }} />
                        <span>
                          <strong>{driver.acronym ?? driver.driver_number}</strong>
                          <small>{driver.team_name ?? "Unknown team"}</small>
                        </span>
                      </span>
                    </td>
                    <td>
                      <span className="f1-practice-programme">{practiceProgramme(driver)}</span>
                    </td>
                    <td>
                      <strong>{formatLap(practiceLapTime(driver))}</strong>
                      <small>{practiceRankDelta(driver, fastest)}</small>
                    </td>
                    <td>{formatLap(driver.last_lap_time)}</td>
                    <td>{formatPracticeDeg(driver)}</td>
                    <td>{formatPracticeRuns(driver)}</td>
                    <td>
                      <span className="f1-practice-confidence">
                        <strong>{confidence}%</strong>
                        <span><i style={{ width: `${confidence}%` }} /></span>
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function TimingTable({
  drivers,
  selectedDriver,
  onSelect,
}: {
  drivers: F1DriverState[];
  selectedDriver: number | null;
  onSelect: (driverNumber: number) => void;
}) {
  return (
    <div className="f1-table-wrap">
      <table className="table f1-driver-table">
        <thead>
          <tr>
            <th>Status</th>
            <th>Pos</th>
            <th>Driver</th>
            <th>Interval</th>
            <th>Tyre</th>
            <th>Best</th>
            <th>Leader</th>
            <th>Last Lap</th>
            <th>Mini Sectors</th>
            <th>Last Sectors</th>
            <th>Speed</th>
          </tr>
        </thead>
        <tbody>
          {drivers.map((driver) => (
            <tr
              key={driver.driver_number}
              className={driver.driver_number === selectedDriver ? "selected" : ""}
              onClick={() => onSelect(driver.driver_number)}
            >
              <td><span className="f1-run-badge">{driver.track_status ?? "RUN"}</span></td>
              <td>{driver.position ?? "-"}</td>
              <td>
                <span className="f1-driver-cell">
                  <span className="f1-team-swatch" style={{ background: teamColor(driver) }} />
                  <span>
                    <strong>{driver.acronym ?? driver.driver_number}</strong>
                    <small>{driver.team_name ?? "Unknown team"}</small>
                  </span>
                </span>
              </td>
              <td>{driver.gap_to_leader ?? driver.interval ?? "-"}</td>
              <td>
                <span className={`f1-compound ${compoundClass(driver.current_compound)}`}>
                  {driver.current_compound ?? "-"} {driver.tyre_age ?? ""}
                </span>
              </td>
              <td>{formatLap(driver.best_lap_time)}</td>
              <td>{leaderGap(driver)}</td>
              <td>{formatLap(driver.last_lap_time)}</td>
              <td><MiniSectorStrip driver={driver} /></td>
              <td>
                <span className="f1-sector-values">
                  <span>{formatSector(driver.sector_times?.sector_1)}</span>
                  <span>{formatSector(driver.sector_times?.sector_2)}</span>
                  <span>{formatSector(driver.sector_times?.sector_3)}</span>
                </span>
              </td>
              <td>{driver.last_speed ? `${Math.round(driver.last_speed)} km/h` : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MiniSectorStrip({ driver }: { driver: F1DriverState }) {
  return (
    <span className="f1-mini-sector-strip" aria-label={`${driver.acronym ?? driver.driver_number} mini sectors`}>
      {Array.from({ length: 25 }, (_, index) => {
        const sector = (index % 3) + 1;
        const value = driver.sector_times?.[`sector_${sector}`];
        const className = isNumber(value)
          ? index % 11 === driver.driver_number % 11
            ? "purple"
            : index % 7 === driver.driver_number % 7
              ? "green"
              : "yellow"
          : "empty";
        return <span key={index} className={`f1-mini-sector ${className}`} />;
      })}
    </span>
  );
}

function LiveLapComparisonPanel({
  snapshot,
  selectedDriverNumbers,
  compareTargets,
  telemetryVisible,
  engineeringSummary,
  onSelectedDriverNumbersChange,
  onCompareTargetsChange,
  onTelemetryVisibleChange,
}: {
  snapshot: F1SessionSnapshot | null;
  selectedDriverNumbers: number[];
  compareTargets: LiveLapTarget[];
  telemetryVisible: Record<LiveTelemetryMetric, boolean>;
  engineeringSummary: FastF1EngineeringSummary | null;
  onSelectedDriverNumbersChange: (value: number[]) => void;
  onCompareTargetsChange: (value: LiveLapTarget[]) => void;
  onTelemetryVisibleChange: (value: Record<LiveTelemetryMetric, boolean>) => void;
}) {
  const [showSlowLaps, setShowSlowLaps] = useState(false);
  const drivers = sortedDrivers(snapshot?.drivers ?? []);
  const availableDriverNumbers = new Set(drivers.map((driver) => driver.driver_number));
  const selectedSet = new Set(selectedDriverNumbers.filter((driverNumber) => availableDriverNumbers.has(driverNumber)));
  const selectedDrivers = drivers.filter((driver) => selectedSet.has(driver.driver_number));
  const chartDrivers = selectedDrivers.length ? selectedDrivers : drivers.slice(0, 3);
  const lapPoints = snapshot?.lapChart ?? [];

  const toggleDriver = (driverNumber: number) => {
    const next = selectedSet.has(driverNumber)
      ? selectedDriverNumbers.filter((item) => item !== driverNumber)
      : [...selectedDriverNumbers, driverNumber];
    onSelectedDriverNumbersChange(next.filter((item, index, array) => array.indexOf(item) === index));
  };

  const selectFastestDrivers = () => {
    const fastest = fastestDriversByLap(lapPoints, drivers).slice(0, 3).map((driver) => driver.driver_number);
    onSelectedDriverNumbersChange(fastest.length ? fastest : drivers.slice(0, 3).map((driver) => driver.driver_number));
  };

  const addCompareTarget = (target: LiveLapTarget) => {
    onCompareTargetsChange(nextLapTargets(compareTargets, target));
  };

  return (
    <div className="f1-live-lap-workbench">
      <section className="panel f1-live-lap-panel">
        <div className="panel-header f1-live-lap-header">
          <div className="panel-header-left">
            <h2 className="module-title">Lap Chart</h2>
            <span className="module-subtitle">Select drivers, click a lap point, then compare lap and telemetry traces.</span>
          </div>
          <div className="f1-live-lap-actions">
            <a className="button secondary button-sm" href="#f1-live-telemetry-compare">Open live telemetry</a>
            <span className="module-subtitle">Updated {snapshot ? formatRelativeTime(snapshot.generatedAt) : "pending"}</span>
          </div>
        </div>

        <div className="panel-body">
          <div className="f1-live-driver-selector">
            <div className="f1-live-selector-head">
              <span>
                Drivers <strong>{chartDrivers.length} selected</strong>
              </span>
              <div>
                <button type="button" className="button secondary button-sm" onClick={() => onSelectedDriverNumbersChange(drivers.map((driver) => driver.driver_number))}>
                  Select all
                </button>
                <button type="button" className="button secondary button-sm" onClick={() => onSelectedDriverNumbersChange([])}>
                  Clear
                </button>
                <button type="button" className="button secondary button-sm" onClick={selectFastestDrivers}>
                  Pick fastest
                </button>
              </div>
            </div>

            <div className="f1-live-driver-grid">
              {drivers.map((driver) => {
                const selected = selectedSet.has(driver.driver_number);
                return (
                  <button
                    type="button"
                    key={driver.driver_number}
                    className={`f1-live-driver-pick ${selected ? "selected" : ""}`}
                    onClick={() => toggleDriver(driver.driver_number)}
                  >
                    <span>P{driver.position ?? "-"}</span>
                    <i style={{ background: teamColor(driver) }} />
                    <strong>{driver.acronym ?? driver.driver_number}</strong>
                    <small>{driver.track_status ?? "TRACK"}</small>
                    <em>#{driver.driver_number}</em>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="f1-live-compare-strip">
            <span>Compare</span>
            {compareTargets.length ? (
              compareTargets.map((target, index) => {
                const driver = drivers.find((item) => item.driver_number === target.driverNumber);
                return (
                  <button
                    type="button"
                    key={`${target.driverNumber}-${target.lap}-${index}`}
                    onClick={() => onCompareTargetsChange(compareTargets.filter((_, itemIndex) => itemIndex !== index))}
                  >
                    <strong>{driver?.acronym ?? target.driverNumber}</strong>
                    <span>L{target.lap} · {formatLap(target.lapTime)}</span>
                  </button>
                );
              })
            ) : (
              <small>Click chart points to select two laps.</small>
            )}
          </div>

          <LiveLapChart
            points={lapPoints}
            drivers={chartDrivers}
            compareTargets={compareTargets}
            showSlowLaps={showSlowLaps}
            onPointSelect={addCompareTarget}
          />

          <div className="f1-live-lap-footer">
            <button type="button" className="button button-sm" onClick={() => setShowSlowLaps((current) => !current)}>
              {showSlowLaps ? "Hide slow laps" : "Show slow laps (>=110%)"}
            </button>
          </div>
        </div>
      </section>

      <LiveTelemetryComparisonPanel
        drivers={drivers}
        compareTargets={compareTargets}
        engineeringSummary={engineeringSummary}
        telemetryVisible={telemetryVisible}
        onTelemetryVisibleChange={onTelemetryVisibleChange}
      />
    </div>
  );
}

function LiveLapChart({
  points,
  drivers,
  compareTargets,
  showSlowLaps,
  onPointSelect,
}: {
  points: F1LapPoint[];
  drivers: F1DriverState[];
  compareTargets: LiveLapTarget[];
  showSlowLaps: boolean;
  onPointSelect: (target: LiveLapTarget) => void;
}) {
  const driverNumbers = new Set(drivers.map((driver) => driver.driver_number));
  const selectedKeys = new Set(compareTargets.map((target) => liveLapTargetKey(target)));
  const filteredPoints = points.filter((point) => driverNumbers.has(point.driver_number));
  if (!filteredPoints.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No lap points for selected drivers</span></div>;
  }
  const bestByDriver = bestLapByDriver(filteredPoints);
  const visiblePoints = showSlowLaps
    ? filteredPoints
    : filteredPoints.filter((point) => {
        const best = bestByDriver.get(point.driver_number);
        return !isNumber(best) || point.value <= best * 1.1;
      });
  const series = drivers.map((driver) => {
    const driverPoints = visiblePoints
      .filter((point) => point.driver_number === driver.driver_number)
      .sort((left, right) => left.lap - right.lap);
    return {
      name: driver.acronym ?? String(driver.driver_number),
      type: "line",
      smooth: false,
      symbol: "circle",
      symbolSize: 8,
      connectNulls: true,
      data: driverPoints.map((point) => ({
        value: [point.lap, point.value],
        driverNumber: point.driver_number,
        lap: point.lap,
        lapTime: point.value,
        itemStyle: selectedKeys.has(liveLapTargetKey({ driverNumber: point.driver_number, lap: point.lap }))
          ? { borderColor: "#fefefe", borderWidth: 3, shadowBlur: 8, shadowColor: teamColor(driver) }
          : undefined,
      })),
      lineStyle: { width: 2, color: teamColor(driver) },
      itemStyle: { color: teamColor(driver), borderColor: "#fefefe", borderWidth: 1 },
    };
  });
  const laps = visiblePoints.map((point) => point.lap);
  const option = f1ChartOption({
    legend: { data: drivers.map((driver) => driver.acronym ?? String(driver.driver_number)) },
    grid: { left: 62, right: 24, top: 38, bottom: 48 },
    xAxis: {
      type: "value",
      name: "Lap Number",
      min: Math.min(...laps),
      max: Math.max(...laps),
      axisLabel: { formatter: (value: number) => `${Math.round(value)}` },
    },
    yAxis: {
      type: "value",
      name: "Lap Time",
      axisLabel: { formatter: (value: number) => formatLap(value) },
      scale: true,
    },
    tooltip: {
      trigger: "item",
      formatter: (params: unknown) => formatLiveLapTooltip(params, drivers),
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0 },
      { type: "inside", yAxisIndex: 0 },
    ],
    series,
  });
  return (
    <div className="f1-chart-shell f1-live-lap-chart-shell">
      <div className="f1-live-chart-hint">Click a point to compare telemetry. Drag to zoom.</div>
      <ReactECharts
        option={option}
        style={{ height: 430, width: "100%" }}
        notMerge
        lazyUpdate
        onEvents={{
          click: (params: unknown) => {
            const target = liveLapTargetFromChartParam(params);
            if (target) onPointSelect(target);
          },
        }}
      />
    </div>
  );
}

function LiveTelemetryComparisonPanel({
  drivers,
  compareTargets,
  engineeringSummary,
  telemetryVisible,
  onTelemetryVisibleChange,
}: {
  drivers: F1DriverState[];
  compareTargets: LiveLapTarget[];
  engineeringSummary: FastF1EngineeringSummary | null;
  telemetryVisible: Record<LiveTelemetryMetric, boolean>;
  onTelemetryVisibleChange: (value: Record<LiveTelemetryMetric, boolean>) => void;
}) {
  const compareDrivers = compareTargets
    .map((target) => drivers.find((driver) => driver.driver_number === target.driverNumber))
    .filter((driver): driver is F1DriverState => Boolean(driver))
    .slice(0, 2);
  const telemetry = buildTelemetrySeries(engineeringSummary, compareDrivers);
  const selectedOptions = LIVE_TELEMETRY_OPTIONS.filter((option) => telemetryVisible[option.id]);
  const hasFastF1Delta = Boolean(engineeringSummary?.telemetryDelta?.series?.length);

  return (
    <section className="panel f1-live-telemetry-compare" id="f1-live-telemetry-compare">
      <div className="panel-header">
        <div className="panel-header-left">
          <h2 className="module-title">Live Telemetry Compare</h2>
          <span className="module-subtitle">Compare two selected laps across speed, delta, throttle, brake, and gear traces.</span>
        </div>
        <span className={`f1-telemetry-confidence ${hasFastF1Delta ? "high" : "reduced"}`}>
          {hasFastF1Delta ? "Delta confidence high" : "FastF1 artifact pending"}
        </span>
      </div>

      <div className="panel-body">
        <TelemetryInsightsPanel
          drivers={drivers}
          targets={compareTargets.slice(0, 2)}
          telemetry={telemetry}
          hasFastF1Delta={hasFastF1Delta}
        />

        <div className="f1-live-chart-controls">
          <div>
            <strong>Chart Controls</strong>
            <span>Display Options</span>
          </div>
          <div className="f1-live-telemetry-toggles">
            {LIVE_TELEMETRY_OPTIONS.map((option) => (
              <label key={option.id}>
                <input
                  type="checkbox"
                  checked={telemetryVisible[option.id]}
                  onChange={(event) =>
                    onTelemetryVisibleChange({
                      ...telemetryVisible,
                      [option.id]: event.target.checked,
                    })
                  }
                />
                {option.label}
              </label>
            ))}
          </div>
        </div>

        {compareTargets.length < 2 ? (
          <div className="empty-state compact"><span className="empty-state-text">Select two lap points to compare telemetry</span></div>
        ) : !telemetry.length ? (
          <div className="empty-state compact">
            <span className="empty-state-text">FastF1 telemetry artifacts are required for selected-lap comparison</span>
          </div>
        ) : (
          <div className="f1-live-telemetry-stack">
            {selectedOptions.map((option) => (
              <TelemetryChart
                key={option.id}
                title={option.label}
                unit={option.unit}
                metric={option.id}
                telemetry={telemetry}
                height={option.height}
                stepped={option.stepped}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function TelemetryInsightsPanel({
  drivers,
  targets,
  telemetry,
  hasFastF1Delta,
}: {
  drivers: F1DriverState[];
  targets: LiveLapTarget[];
  telemetry: TelemetryDriverSeries[];
  hasFastF1Delta: boolean;
}) {
  if (targets.length < 2) {
    return (
      <div className="f1-live-insight-empty">
        Select one lap for each driver, or two laps from the same driver, to unlock telemetry insights.
      </div>
    );
  }
  if (!telemetry.length) {
    return (
      <div className="f1-live-insight-empty">
        Selected laps are ready. Import FastF1 telemetry artifacts for this session to calculate speed, throttle, brake,
        gear, and delta insights.
      </div>
    );
  }
  const primary = targets[0];
  const secondary = targets[1];
  const primaryDriver = drivers.find((driver) => driver.driver_number === primary.driverNumber);
  const secondaryDriver = drivers.find((driver) => driver.driver_number === secondary.driverNumber);
  const primaryStats = telemetryStats(telemetry[0]);
  const secondaryStats = telemetryStats(telemetry[1]);
  const lapDelta = secondary.lapTime - primary.lapTime;
  return (
    <div className="f1-live-insights-grid">
      <TelemetryInsightCard
        title={`${primaryDriver?.acronym ?? primary.driverNumber} · L${primary.lap}`}
        color={teamColor(primaryDriver)}
        lapTime={primary.lapTime}
        stats={primaryStats}
        note={hasFastF1Delta ? "FastF1 aligned reference lap." : "Reduced live trace until FastF1 delta is imported."}
      />
      <TelemetryInsightCard
        title={`${secondaryDriver?.acronym ?? secondary.driverNumber} · L${secondary.lap}`}
        color={teamColor(secondaryDriver)}
        lapTime={secondary.lapTime}
        stats={secondaryStats}
        delta={{
          lap: lapDelta,
          topSpeed: numericDelta(secondaryStats.topSpeed, primaryStats.topSpeed),
          avgSpeed: numericDelta(secondaryStats.avgSpeed, primaryStats.avgSpeed),
          fullThrottle: numericDelta(secondaryStats.fullThrottlePercent, primaryStats.fullThrottlePercent),
        }}
        note={lapDelta <= 0 ? "Selected lap is faster than the reference." : "Selected lap is slower than the reference."}
      />
    </div>
  );
}

function TelemetryInsightCard({
  title,
  color,
  lapTime,
  stats,
  note,
  delta,
}: {
  title: string;
  color: string;
  lapTime: number;
  stats: TelemetryStats;
  note: string;
  delta?: {
    lap: number;
    topSpeed: number | null;
    avgSpeed: number | null;
    fullThrottle: number | null;
  };
}) {
  return (
    <div className="f1-live-insight-card">
      <div className="f1-live-insight-card-head">
        <span>
          <i style={{ background: color }} />
          <strong>{title}</strong>
        </span>
        <em>{formatLap(lapTime)}</em>
      </div>
      <div className="f1-live-insight-metrics">
        <span><small>Top speed</small><strong>{formatTelemetryKmh(stats.topSpeed)}</strong></span>
        <span><small>Avg speed</small><strong>{formatTelemetryKmh(stats.avgSpeed)}</strong></span>
        <span><small>Full throttle</small><strong>{formatTelemetryPercent(stats.fullThrottlePercent)}</strong></span>
        <span><small>Braking events</small><strong>{stats.brakingEvents ?? "N/A"}</strong></span>
      </div>
      {delta ? (
        <div className="f1-live-delta-grid">
          <span><small>Lap delta</small><strong>{formatSeconds(delta.lap)}</strong></span>
          <span><small>Top speed</small><strong>{formatTelemetryDelta(delta.topSpeed, "km/h")}</strong></span>
          <span><small>Avg speed</small><strong>{formatTelemetryDelta(delta.avgSpeed, "km/h")}</strong></span>
          <span><small>Throttle</small><strong>{formatTelemetryDelta(delta.fullThrottle, "%")}</strong></span>
        </div>
      ) : null}
      <p>{note}</p>
    </div>
  );
}

function LapChart({ points, drivers }: { points: F1LapPoint[]; drivers: F1DriverState[] }) {
  if (!points.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No lap points</span></div>;
  }
  const lapNumbers = Array.from(new Set(points.map((point) => point.lap))).sort((a, b) => a - b);
  const visibleDrivers: F1DriverState[] = drivers.length
    ? drivers
    : Array.from(new Set(points.map((point) => point.driver_number))).slice(0, 4).map((driverNumber) => ({
        driver_number: driverNumber,
        acronym: String(driverNumber),
        last_update_seq: 0,
      }));
  const series = visibleDrivers.map((driver) => {
    const driverPoints = new Map(
      points
        .filter((point) => point.driver_number === driver.driver_number)
        .map((point) => [point.lap, point.value])
    );
    return {
      name: driver.acronym ?? String(driver.driver_number),
      type: "line",
      smooth: false,
      symbol: "circle",
      symbolSize: 7,
      connectNulls: true,
      data: lapNumbers.map((lap) => driverPoints.get(lap) ?? null),
      lineStyle: { width: 2, color: teamColor(driver) },
      itemStyle: { color: teamColor(driver), borderColor: "#fefefe", borderWidth: 1 },
    };
  });
  const option = f1ChartOption({
    legend: { data: visibleDrivers.map((driver) => driver.acronym ?? String(driver.driver_number)) },
    xAxis: {
      type: "category",
      name: "Lap Number",
      data: lapNumbers,
    },
    yAxis: {
      type: "value",
      name: "Lap Time",
      axisLabel: { formatter: (value: number) => formatLap(value) },
      inverse: false,
      scale: true,
    },
    tooltip: {
      trigger: "axis",
      valueFormatter: (value: number | null) => (isNumber(value) ? formatLap(value) : "-"),
    },
    series,
  });
  return (
    <div className="f1-chart-shell f1-lap-chart-shell">
      <ReactECharts option={option} style={{ height: 360, width: "100%" }} notMerge lazyUpdate />
    </div>
  );
}

type TrackSvgPoint = { x: number; y: number };

type TrackSegment = {
  index: number;
  sectorIndex: number;
  points: TrackSvgPoint[];
  leader: F1DriverState | null;
  ranking: F1DriverState[];
};

const TRACK_VIEWBOX = { width: 700, height: 500, padding: 36 };

const FALLBACK_TRACK_POINTS: TrackSvgPoint[] = [
  { x: 498, y: 60 },
  { x: 527, y: 94 },
  { x: 514, y: 132 },
  { x: 460, y: 147 },
  { x: 414, y: 148 },
  { x: 365, y: 113 },
  { x: 423, y: 106 },
  { x: 477, y: 90 },
  { x: 532, y: 87 },
  { x: 545, y: 120 },
  { x: 498, y: 170 },
  { x: 473, y: 196 },
  { x: 414, y: 256 },
  { x: 350, y: 319 },
  { x: 306, y: 390 },
  { x: 267, y: 440 },
  { x: 210, y: 463 },
  { x: 172, y: 465 },
  { x: 180, y: 378 },
  { x: 234, y: 391 },
  { x: 314, y: 390 },
  { x: 350, y: 319 },
  { x: 294, y: 310 },
  { x: 268, y: 341 },
  { x: 180, y: 378 },
  { x: 255, y: 271 },
  { x: 293, y: 246 },
  { x: 414, y: 148 },
  { x: 498, y: 60 },
];

const TURN_LABEL_PROGRESS = [0.04, 0.1, 0.18, 0.25, 0.32, 0.39, 0.46, 0.53, 0.6, 0.68, 0.75, 0.82, 0.9, 0.96];

function TrackMap({
  drivers,
  selectedDriver,
  geometry,
  microSectors,
  onSelect,
  mode = "default",
}: {
  drivers: F1DriverState[];
  selectedDriver: number | null;
  geometry: F1TrackGeometryResponse | null;
  microSectors: F1CustomMicroSectorPassage[];
  onSelect: (driverNumber: number) => void;
  mode?: "default" | "tracker";
}) {
  const [activeSegment, setActiveSegment] = useState<number | null>(null);
  const trackerMode = mode === "tracker";
  const trackPoints = useMemo(() => buildTrackPoints(geometry), [geometry]);
  const segments = useMemo(
    () => buildTrackSegments(trackPoints, drivers, microSectors),
    [drivers, microSectors, trackPoints]
  );
  const corners = useMemo(() => TURN_LABEL_PROGRESS.map((progress, index) => ({
    label: `T${index + 1}`,
    point: pointAtTrackProgress(trackPoints, progress),
  })), [trackPoints]);
  const pointString = useMemo(() => trackPoints.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" "), [trackPoints]);
  const selectedSegment = activeSegment === null ? null : segments[activeSegment] ?? null;
  const visibleDrivers = selectedSegment?.ranking.slice(0, 5) ?? sortedDrivers(drivers).slice(0, 5);
  const sourceLabel = geometry
    ? `FastF1 centerline · ${geometry.sampledPointCount}/${geometry.pointCount} pts`
    : "Fallback layout · import FastF1 telemetry for exact circuit";

  return (
    <div className={`f1-track-card ${trackerMode ? "tracker" : ""}`}>
      <div className="f1-track-stage">
        <svg
          className="f1-track-map"
          viewBox={`0 0 ${TRACK_VIEWBOX.width} ${TRACK_VIEWBOX.height}`}
          role="img"
          aria-label="F1 track map"
        >
          <polyline
            points={pointString}
            fill="none"
            stroke="var(--border-2)"
            strokeWidth="28"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <polyline
            points={pointString}
            fill="none"
            stroke="#101010"
            strokeWidth="18"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          {segments.map((segment) => {
            const leader = segment.leader;
            const active = activeSegment === segment.index;
            return (
              <polyline
                key={segment.index}
                points={segment.points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ")}
                fill="none"
                stroke={leader ? teamColor(leader) : "var(--accent)"}
                strokeWidth={active ? 12 : 8}
                strokeLinecap="round"
                strokeLinejoin="round"
                className="f1-track-zone"
                role="button"
                tabIndex={0}
                onMouseEnter={() => setActiveSegment(segment.index)}
                onFocus={() => setActiveSegment(segment.index)}
                onClick={() => setActiveSegment((current) => (current === segment.index ? null : segment.index))}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setActiveSegment((current) => (current === segment.index ? null : segment.index));
                  }
                }}
              >
                <title>{`Zone ${segment.index + 1}${leader ? ` · ${driverLabel(leader)}` : ""}`}</title>
              </polyline>
            );
          })}
          {corners.map((corner) => (
            <g key={corner.label} className="f1-track-corner-label">
              <rect x={corner.point.x - 13} y={corner.point.y - 18} width="26" height="17" rx="4" />
              <text x={corner.point.x} y={corner.point.y - 5} textAnchor="middle">
                {corner.label}
              </text>
            </g>
          ))}
          {drivers.slice(0, 12).map((driver, index) => {
            const rawProgress = driver.track_progress ?? index / Math.max(1, drivers.length);
            const point = pointAtTrackProgress(trackPoints, normalizedProgress(rawProgress));
            const selected = driver.driver_number === selectedDriver;
            return (
              <g
                key={driver.driver_number}
                onClick={() => onSelect(driver.driver_number)}
                className={`f1-track-car ${selected ? "selected" : ""}`}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelect(driver.driver_number);
                  }
                }}
              >
                <circle cx={point.x} cy={point.y} r={selected ? 12 : 9} fill={teamColor(driver)} />
                <text x={point.x} y={point.y + 4} textAnchor="middle">
                  {driver.acronym?.slice(0, 2) ?? driver.driver_number}
                </text>
              </g>
            );
          })}
        </svg>
        <span className="f1-track-source">{sourceLabel}</span>
      </div>

      {!trackerMode ? (
      <div className="f1-track-zone-panel">
        <div className="f1-track-zone-title">
          <span>{selectedSegment ? `Zone ${selectedSegment.index + 1}` : "Track Zones"}</span>
          <strong>{selectedSegment?.leader ? driverLabel(selectedSegment.leader) : "Field order"}</strong>
        </div>
        <div className="f1-track-zone-dots">
          {segments.map((segment) => (
            <button
              key={segment.index}
              type="button"
              aria-label={`Show track zone ${segment.index + 1}`}
              className={activeSegment === segment.index ? "active" : ""}
              style={{ background: segment.leader ? teamColor(segment.leader) : "var(--accent)" }}
              onClick={() => setActiveSegment((current) => (current === segment.index ? null : segment.index))}
            />
          ))}
        </div>
        <div className="f1-track-zone-list">
          {visibleDrivers.map((driver, index) => (
            <button
              type="button"
              key={driver.driver_number}
              className="f1-track-zone-row"
              onClick={() => onSelect(driver.driver_number)}
            >
              <span>{index + 1}</span>
              <span className="f1-team-swatch" style={{ background: teamColor(driver) }} />
              <strong>{driverLabel(driver)}</strong>
              <small>{driver.position ? `P${driver.position}` : driver.interval ?? "-"}</small>
            </button>
          ))}
        </div>
      </div>
      ) : null}
    </div>
  );
}

function buildTrackPoints(geometry: F1TrackGeometryResponse | null): TrackSvgPoint[] {
  const rawPoints = geometry?.points
    .filter((point) => isNumber(point.x) && isNumber(point.y))
    .map((point) => ({ x: point.x, y: point.y }));
  if (!rawPoints || rawPoints.length < 2) {
    return FALLBACK_TRACK_POINTS;
  }
  return thinTrackPoints(normalizeTrackPoints(rawPoints), 320);
}

function normalizeTrackPoints(points: TrackSvgPoint[]): TrackSvgPoint[] {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const rawWidth = Math.max(1, maxX - minX);
  const rawHeight = Math.max(1, maxY - minY);
  const availableWidth = TRACK_VIEWBOX.width - TRACK_VIEWBOX.padding * 2;
  const availableHeight = TRACK_VIEWBOX.height - TRACK_VIEWBOX.padding * 2;
  const scale = Math.min(availableWidth / rawWidth, availableHeight / rawHeight);
  const renderedWidth = rawWidth * scale;
  const renderedHeight = rawHeight * scale;
  const offsetX = (TRACK_VIEWBOX.width - renderedWidth) / 2;
  const offsetY = (TRACK_VIEWBOX.height - renderedHeight) / 2;
  return points.map((point) => ({
    x: offsetX + (point.x - minX) * scale,
    y: offsetY + (maxY - point.y) * scale,
  }));
}

function thinTrackPoints(points: TrackSvgPoint[], limit: number): TrackSvgPoint[] {
  if (points.length <= limit) return points;
  const step = (points.length - 1) / (limit - 1);
  return Array.from({ length: limit }, (_, index) => points[Math.min(points.length - 1, Math.round(index * step))]);
}

function buildTrackSegments(
  points: TrackSvgPoint[],
  drivers: F1DriverState[],
  microSectors: F1CustomMicroSectorPassage[]
): TrackSegment[] {
  const segmentCount = Math.min(14, Math.max(8, microSectors[0]?.sector_count ?? 10));
  const maxIndex = Math.max(1, points.length - 1);
  const sectorCount = Math.max(1, ...microSectors.map((sector) => sector.sector_count || 1));
  return Array.from({ length: segmentCount }, (_, index) => {
    const startIndex = Math.floor((index / segmentCount) * maxIndex);
    const endIndex = Math.max(startIndex + 1, Math.round(((index + 1) / segmentCount) * maxIndex));
    const sectorIndex = Math.min(sectorCount, Math.floor((index / segmentCount) * sectorCount) + 1);
    const ranking = rankDriversForTrackSector(drivers, microSectors, sectorIndex);
    return {
      index,
      sectorIndex,
      points: points.slice(startIndex, Math.min(points.length, endIndex + 1)),
      leader: ranking[0] ?? sortedDrivers(drivers)[index % Math.max(1, drivers.length)] ?? null,
      ranking,
    };
  });
}

function rankDriversForTrackSector(
  drivers: F1DriverState[],
  microSectors: F1CustomMicroSectorPassage[],
  sectorIndex: number
): F1DriverState[] {
  const driverByNumber = new Map(drivers.map((driver) => [driver.driver_number, driver]));
  const bestByDriver = new Map<number, F1CustomMicroSectorPassage>();
  for (const passage of microSectors) {
    if (passage.sector_index !== sectorIndex || !isNumber(passage.passage_time)) continue;
    const current = bestByDriver.get(passage.driver_number);
    if (!current || passage.passage_time < current.passage_time) {
      bestByDriver.set(passage.driver_number, passage);
    }
  }
  const ranked = [...bestByDriver.values()]
    .sort((left, right) => left.passage_time - right.passage_time)
    .map((passage) => driverByNumber.get(passage.driver_number))
    .filter((driver): driver is F1DriverState => Boolean(driver));
  return ranked.length ? ranked : sortedDrivers(drivers);
}

function sortedDrivers(drivers: F1DriverState[]): F1DriverState[] {
  return [...drivers].sort((left, right) => {
    const leftPosition = left.position ?? 10_000;
    const rightPosition = right.position ?? 10_000;
    if (leftPosition !== rightPosition) return leftPosition - rightPosition;
    return driverLabel(left).localeCompare(driverLabel(right));
  });
}

function pointAtTrackProgress(points: TrackSvgPoint[], progress: number): TrackSvgPoint {
  if (!points.length) return { x: TRACK_VIEWBOX.width / 2, y: TRACK_VIEWBOX.height / 2 };
  if (points.length === 1) return points[0];
  const target = normalizedProgress(progress) * trackLength(points);
  let accumulated = 0;
  for (let index = 1; index < points.length; index += 1) {
    const left = points[index - 1];
    const right = points[index];
    const segmentLength = distanceBetweenPoints(left, right);
    if (accumulated + segmentLength >= target) {
      const ratio = segmentLength <= 0 ? 0 : (target - accumulated) / segmentLength;
      return {
        x: left.x + (right.x - left.x) * ratio,
        y: left.y + (right.y - left.y) * ratio,
      };
    }
    accumulated += segmentLength;
  }
  return points[points.length - 1];
}

function trackLength(points: TrackSvgPoint[]): number {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) {
    length += distanceBetweenPoints(points[index - 1], points[index]);
  }
  return Math.max(1, length);
}

function distanceBetweenPoints(left: TrackSvgPoint, right: TrackSvgPoint): number {
  return Math.hypot(right.x - left.x, right.y - left.y);
}

function normalizedProgress(value: number): number {
  if (!isNumber(value)) return 0;
  return ((value % 1) + 1) % 1;
}

function StrategyTimeline({
  snapshot,
  selectedDriver,
}: {
  snapshot: F1SessionSnapshot | null;
  selectedDriver: number | null;
}) {
  const stints = (snapshot?.strategyTimeline ?? []).filter((segment) =>
    selectedDriver ? segment.driver_number === selectedDriver : true
  );
  if (!stints.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No stint data</span></div>;
  }
  const maxLap = Math.max(...stints.map((segment) => segment.end_lap ?? snapshot?.drivers[0]?.current_lap ?? segment.start_lap));
  return (
    <div className="f1-strategy-list">
      {stints.map((segment) => {
        const driver = snapshot?.drivers.find((item) => item.driver_number === segment.driver_number);
        const start = ((segment.start_lap - 1) / Math.max(1, maxLap)) * 100;
        const endLap = segment.end_lap ?? driver?.current_lap ?? maxLap;
        const width = Math.max(8, ((endLap - segment.start_lap + 1) / Math.max(1, maxLap)) * 100);
        return (
          <div className="f1-strategy-row" key={`${segment.driver_number}-${segment.stint_number}`}>
            <span>{driver?.acronym ?? segment.driver_number}</span>
            <div className="f1-strategy-track">
              <span
                className={`f1-strategy-bar ${compoundClass(segment.compound)}`}
                style={{ left: `${start}%`, width: `${width}%` }}
              >
                {segment.compound}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PredictionList({
  predictions,
  drivers,
}: {
  predictions: F1PredictionSnapshot[];
  drivers: F1DriverState[];
}) {
  if (!predictions.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No prediction snapshots</span></div>;
  }
  return (
    <div className="f1-prediction-list">
      {predictions.slice(0, 6).map((prediction) => {
        const driver = drivers.find((item) => item.driver_number === prediction.driver_number);
        return (
          <div className="f1-prediction-row" key={`${prediction.driver_number}-${prediction.source_event_sequence}`}>
            <span className="f1-driver-cell">
              <span className="f1-team-swatch" style={{ background: teamColor(driver) }} />
              <strong>{driver?.acronym ?? prediction.driver_number}</strong>
            </span>
            <span>Exp {formatPositionLabel(prediction.expected_position)}</span>
            <span>P10-90 {formatPositionRange(prediction)}</span>
            <span>Win {(prediction.win_probability * 100).toFixed(1)}%</span>
            <span>Podium {(prediction.podium_probability * 100).toFixed(1)}%</span>
            <span>Pts {(prediction.points_probability * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

function MicroSectorPanel({
  snapshot,
  selectedDriver,
}: {
  snapshot: F1SessionSnapshot | null;
  selectedDriver: number | null;
}) {
  const allPassages = snapshot?.customMicroSectors ?? [];
  const passages = selectedDriver
    ? allPassages.filter((passage) => passage.driver_number === selectedDriver)
    : allPassages;
  if (!allPassages.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No custom micro-sector passages</span></div>;
  }
  const sectorCount = allPassages[0]?.sector_count ?? 25;
  const latestBySector = new Map<number, F1CustomMicroSectorPassage>();
  for (const passage of passages) {
    const current = latestBySector.get(passage.sector_index);
    if (!current || passage.seq > current.seq) {
      latestBySector.set(passage.sector_index, passage);
    }
  }
  const recent = [...passages].sort((a, b) => b.seq - a.seq).slice(0, 6);
  return (
    <div className="f1-micro-sector-panel">
      <div className="f1-micro-sector-strip" aria-label="Custom micro-sector strip">
        {Array.from({ length: sectorCount }, (_, index) => {
          const sectorIndex = index + 1;
          const passage = latestBySector.get(sectorIndex);
          return (
            <span
              key={sectorIndex}
              className={`f1-micro-sector-cell ${microSectorClass(passage?.session_best_delta)}`}
              title={passage ? `S${sectorIndex}: ${passage.passage_time.toFixed(3)}s` : `S${sectorIndex}`}
            />
          );
        })}
      </div>
      <div className="f1-micro-sector-list">
        {recent.map((passage) => {
          const driver = snapshot?.drivers.find((item) => item.driver_number === passage.driver_number);
          return (
            <div className="f1-micro-sector-row" key={`${passage.driver_number}-${passage.lap}-${passage.sector_index}`}>
              <span className="f1-driver-cell">
                <span className="f1-team-swatch" style={{ background: teamColor(driver) }} />
                <strong>{driver?.acronym ?? passage.driver_number}</strong>
              </span>
              <span>S{passage.sector_index}/{passage.sector_count}</span>
              <strong>{passage.passage_time.toFixed(3)}s</strong>
              <span>Best {formatDelta(passage.personal_best_delta)}</span>
              <span>Ahead {formatDelta(passage.car_ahead_delta)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PaceAnalysisPanel({ analytics }: { analytics: F1AnalyticsResponse | null }) {
  const pace = analytics?.analytics.pace_analysis_v1;
  if (!pace || !pace.drivers.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No pace analysis</span></div>;
  }
  const medians = pace.drivers.map((driver) => driver.medianLapTime).filter(isNumber);
  const minMedian = medians.length ? Math.min(...medians) : 0;
  const maxMedian = medians.length ? Math.max(...medians) : 1;
  const span = Math.max(0.001, maxMedian - minMedian);
  return (
    <div className="f1-pace-panel">
      <div className="f1-degradation-summary">
        <span>{pace.driverCount} drivers</span>
        <span>{pace.fieldSeries.length} laps</span>
        <span>Status {pace.status}</span>
      </div>
      <div className="f1-pace-list">
        {pace.drivers.slice(0, 6).map((driver) => {
          const medianValue = isNumber(driver.medianLapTime) ? driver.medianLapTime : maxMedian;
          const width = 100 - ((medianValue - minMedian) / span) * 42;
          return (
            <div className="f1-pace-row" key={driver.driverNumber}>
              <span className="f1-driver-cell">
                <strong>{driver.acronym ?? driver.driverNumber}</strong>
              </span>
              <span className="f1-pace-bar-track">
                <span className="f1-pace-bar" style={{ width: `${Math.max(42, width)}%` }} />
              </span>
              <strong>{formatLap(driver.medianLapTime)}</strong>
              <span>Best {formatLap(driver.bestLapTime)}</span>
              <span>Trend {formatSigned(driver.trendLastVsFirst)}s</span>
              <span>Std {formatNumber(driver.consistencyStdSeconds)}s</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function BattleDashboardPanel({ analytics }: { analytics: F1AnalyticsResponse | null }) {
  const battle = analytics?.analytics.battle_dashboard_v1;
  if (!battle || !battle.battles.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No battle windows</span></div>;
  }
  return (
    <div className="f1-battle-panel">
      <div className="f1-degradation-summary">
        <span>{battle.battleCount} battles</span>
        <span>{battle.activeOvertakeWindows} active</span>
        <span>{battle.drsTrains.length} DRS trains</span>
      </div>
      <div className="f1-battle-list">
        {battle.battles.slice(0, 6).map((item) => (
          <div className="f1-battle-row" key={`${item.ahead.driverNumber}-${item.chaser.driverNumber}`}>
            <span className="f1-battle-pair">
              <strong>{item.chaser.acronym ?? item.chaser.driverNumber}</strong>
              <small>vs {item.ahead.acronym ?? item.ahead.driverNumber}</small>
            </span>
            <span className={`f1-window-state ${item.windowState}`}>{item.windowState}</span>
            <span>Gap {formatNumber(item.gapSeconds)}s</span>
            <span>Pace {formatSigned(item.recentPaceDeltaSeconds)}s</span>
            <strong>{formatPercent(item.overtakeWindowProbability)}</strong>
          </div>
        ))}
      </div>
      {battle.drsTrains.length ? (
        <div className="f1-drs-trains">
          {battle.drsTrains.slice(0, 3).map((train) => (
            <span key={train.drivers.map((driver) => driver.driverNumber).join("-")}>
              DRS train {train.drivers.map((driver) => driver.acronym ?? driver.driverNumber).join(" / ")}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function TyreDegradationPanel({ analytics }: { analytics: F1AnalyticsResponse | null }) {
  const degradation = analytics?.analytics.tyre_degradation_v1;
  if (!degradation || !degradation.compounds.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No degradation projection</span></div>;
  }
  return (
    <div className="f1-degradation">
      <div className="f1-degradation-summary">
        <span>Status {degradation.status}</span>
        <span>{degradation.sampleCount} clean</span>
        <span>{degradation.excludedCount} excluded</span>
      </div>
      <div className="f1-degradation-list">
        {degradation.compounds.map((compound) => (
          <div className="f1-degradation-row" key={compound.compound}>
            <span className={`f1-compound ${compoundClass(compound.compound)}`}>{compound.compound}</span>
            <span>{compound.cleanLapCount} laps</span>
            <span>Age {compound.minTyreAge}-{compound.maxTyreAge}</span>
            <strong>{formatSigned(compound.slopeSecondsPerTyreLap)}s/lap</strong>
            <span>5L {formatSigned(compound.projectedLossNext5Laps)}s</span>
            <span>Cliff {formatPercent(compound.tyreCliffProbability)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function WeatherEvolutionPanel({
  snapshot,
  analytics,
}: {
  snapshot: F1SessionSnapshot | null;
  analytics: F1AnalyticsResponse | null;
}) {
  const evolution = analytics?.analytics.weather_evolution_v1;
  const samples = evolution?.series?.length ? evolution.series : weatherSeriesFromSnapshot(snapshot);
  if (!samples.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No weather evolution samples</span></div>;
  }
  const trackValues = samples.map((sample) => sample.trackTemperature).filter(isNumber);
  const minTrack = trackValues.length ? Math.min(...trackValues) : 0;
  const maxTrack = trackValues.length ? Math.max(...trackValues) : 1;
  const span = Math.max(0.1, maxTrack - minTrack);
  return (
    <div className="f1-weather-evolution">
      <div className="f1-degradation-summary">
        <span>{samples.length} samples</span>
        <span>Track {formatSigned(evolution?.trackTemperatureDelta)} C</span>
        <span>Air {formatSigned(evolution?.airTemperatureDelta)} C</span>
        <span>{evolution?.rainfallDetected ? "Rain detected" : "Dry track"}</span>
      </div>
      <div className="f1-weather-chart" aria-label="Weather evolution chart">
        {samples.map((sample, index) => {
          const track = sample.trackTemperature;
          const height = isNumber(track) ? 24 + ((track - minTrack) / span) * 76 : 24;
          return (
            <div className="f1-weather-column" key={`${sample.eventTime ?? index}-${index}`}>
              <span className="f1-weather-bar" style={{ height: `${height}%` }} />
              <small>{formatWeatherTime(sample.eventTime)}</small>
              <strong>{isNumber(track) ? `${track.toFixed(1)} C` : "-"}</strong>
            </div>
          );
        })}
      </div>
      <div className="f1-weather-latest">
        <span>Wind {formatNumber(evolution?.latest?.windSpeed)} km/h</span>
        <span>Rain {formatNumber(evolution?.latest?.rainfall)}</span>
        <span>Max rain {formatNumber(evolution?.maxRainfall)}</span>
      </div>
    </div>
  );
}

function RaceControl({ messages }: { messages: Array<Record<string, unknown>> }) {
  if (!messages.length) {
    return <div className="empty-state compact"><span className="empty-state-text">No race-control messages</span></div>;
  }
  return (
    <div className="f1-race-control">
      {messages.slice(-6).map((message, index) => (
        <div className="data-health-row" key={`${String(message.message)}-${index}`}>
          <span className="status-dot ok" />
          <span className="data-health-label">{String(message.message ?? message.category ?? "Race control")}</span>
          <span className="data-health-hint">{String(message.flag ?? message.scope ?? "")}</span>
        </div>
      ))}
    </div>
  );
}

function artifactLabel(artifact: Pick<FastF1ArtifactRecord, "kind" | "metadata" | "relativePath">): string {
  const driver = typeof artifact.metadata.driver === "string" ? artifact.metadata.driver : null;
  const lap = typeof artifact.metadata.lapNumber === "number" ? artifact.metadata.lapNumber : null;
  const driverA = typeof artifact.metadata.driverA === "string" ? artifact.metadata.driverA : null;
  const driverB = typeof artifact.metadata.driverB === "string" ? artifact.metadata.driverB : null;
  if (artifact.kind === "fastf1_distance_aligned_telemetry" && driver) {
    return `${driver}${lap ? ` L${lap}` : ""} telemetry`;
  }
  if (artifact.kind === "fastf1_telemetry_delta" && driverA && driverB) {
    return `${driverA} vs ${driverB} delta`;
  }
  if (artifact.kind === "fastf1_centerline") return "Canonical centreline";
  if (artifact.kind === "fastf1_laps") return "Lap table";
  if (artifact.kind === "fastf1_weather") return "Weather samples";
  if (artifact.kind === "fastf1_race_control") return "Race control";
  return artifact.relativePath ?? artifact.kind;
}

function f1ChartOption(option: Record<string, unknown>): Record<string, unknown> {
  const base = {
    backgroundColor: "transparent",
    color: ["#00d2be", "#ff6363", "#f97316", "#fefefe", "#3b82f6", "#a855f7"],
    grid: { left: 60, right: 24, top: 34, bottom: 42 },
    legend: {
      bottom: 0,
      textStyle: {
        color: "#fefefe",
        fontFamily: "Inter",
        fontSize: 11,
      },
      itemWidth: 10,
      itemHeight: 10,
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#151515",
      borderColor: "#3a3a3a",
      textStyle: { color: "#fefefe", fontFamily: "JetBrains Mono", fontSize: 11 },
      axisPointer: { lineStyle: { color: "rgba(254,254,254,0.32)" } },
    },
    xAxis: {
      axisLine: { lineStyle: { color: "#3a3a3a" } },
      axisLabel: { color: "#a7a7a7", fontFamily: "JetBrains Mono", fontSize: 10 },
      nameTextStyle: { color: "#a7a7a7", fontFamily: "JetBrains Mono", fontSize: 10 },
      splitLine: { show: true, lineStyle: { color: "rgba(254,254,254,0.08)" } },
    },
    yAxis: {
      axisLine: { lineStyle: { color: "#3a3a3a" } },
      axisLabel: { color: "#a7a7a7", fontFamily: "JetBrains Mono", fontSize: 10 },
      nameTextStyle: { color: "#fefefe", fontFamily: "JetBrains Mono", fontSize: 10 },
      splitLine: { lineStyle: { color: "rgba(254,254,254,0.18)" } },
    },
  };
  return mergeChartOption(base, option);
}

function mergeChartOption(
  base: Record<string, unknown>,
  override: Record<string, unknown>
): Record<string, unknown> {
  const output: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(override)) {
    const current = output[key];
    output[key] =
      isPlainObject(current) && isPlainObject(value)
        ? mergeChartOption(current, value)
        : value;
  }
  return output;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function buildTelemetrySeries(
  summary: FastF1EngineeringSummary | null,
  drivers: F1DriverState[]
): TelemetryDriverSeries[] {
  const delta = summary?.telemetryDelta;
  if (delta?.series?.length) {
    const driverA = drivers[0];
    const driverB = drivers[1];
    const nameA = delta.driverA ?? driverA?.acronym ?? "Driver A";
    const nameB = delta.driverB ?? driverB?.acronym ?? "Driver B";
    const colorA = teamColor(driverA) === "var(--accent)" ? "#00d2be" : teamColor(driverA);
    const colorB = teamColor(driverB) === "var(--accent)" ? "#ff6363" : teamColor(driverB);
    return [
      {
        name: nameA,
        color: colorA,
        speed: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.speedA))
          .map((point): [number, number] => [point.distance as number, point.speedA as number]),
        delta: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.deltaSeconds))
          .map((point): [number, number] => [point.distance as number, point.deltaSeconds as number]),
        throttle: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.throttleA))
          .map((point): [number, number] => [point.distance as number, point.throttleA as number]),
        brake: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.brakeA))
          .map((point): [number, number] => [point.distance as number, point.brakeA as number]),
        gear: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.gearA))
          .map((point): [number, number] => [point.distance as number, point.gearA as number]),
      },
      {
        name: nameB,
        color: colorB,
        speed: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.speedB))
          .map((point): [number, number] => [point.distance as number, point.speedB as number]),
        delta: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.deltaSeconds))
          .map((point): [number, number] => [point.distance as number, -(point.deltaSeconds as number)]),
        throttle: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.throttleB))
          .map((point): [number, number] => [point.distance as number, point.throttleB as number]),
        brake: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.brakeB))
          .map((point): [number, number] => [point.distance as number, point.brakeB as number]),
        gear: delta.series
          .filter((point) => isNumber(point.distance) && isNumber(point.gearB))
          .map((point): [number, number] => [point.distance as number, point.gearB as number]),
      },
    ].filter((series) => series.speed.length);
  }

  return [];
}

function latestWeatherRecord(snapshot: F1SessionSnapshot | null): Record<string, unknown> | null {
  const samples = snapshot?.weatherSamples ?? [];
  const latestSample = samples.length ? samples[samples.length - 1] : null;
  if (!latestSample && !snapshot?.weather) return null;
  return {
    ...(latestSample ?? {}),
    ...(snapshot?.weather ?? {}),
  };
}

function formatLiveEventName(snapshot: F1SessionSnapshot | null, resolution: F1SessionResolution | null): string {
  const info = snapshot?.sessionInfo ?? {};
  const resolvedSession = resolution?.session ?? resolution?.nextSession ?? null;
  const resolvedInfo = resolvedSession ? (resolvedSession as Record<string, unknown>) : {};
  const name =
    stringFromSessionInfo(info, "event_name") ??
    stringFromSessionInfo(info, "meeting_name") ??
    stringFromSessionInfo(info, "fastf1_event_name") ??
    stringFromSessionInfo(resolvedInfo, "event_name") ??
    stringFromSessionInfo(resolvedInfo, "meeting_name") ??
    stringFromSessionInfo(resolvedInfo, "fastf1_event_name");
  if (name) return name;
  const country = stringFromSessionInfo(info, "country_name") ?? stringFromSessionInfo(resolvedInfo, "country_name");
  if (country) return `${country} Grand Prix`;
  const location = stringFromSessionInfo(info, "location") ?? stringFromSessionInfo(resolvedInfo, "location");
  return location ? `${location} Grand Prix` : "F1 Live Session";
}

function formatLiveEventLocation(snapshot: F1SessionSnapshot | null, resolution: F1SessionResolution | null): string {
  const info = snapshot?.sessionInfo ?? {};
  const resolvedSession = resolution?.session ?? resolution?.nextSession ?? null;
  const resolvedInfo = resolvedSession ? (resolvedSession as Record<string, unknown>) : {};
  const location = stringFromSessionInfo(info, "location") ?? stringFromSessionInfo(info, "circuit_short_name") ?? stringFromSessionInfo(resolvedInfo, "location") ?? stringFromSessionInfo(resolvedInfo, "circuit_short_name");
  const country = stringFromSessionInfo(info, "country_name") ?? stringFromSessionInfo(info, "country_code") ?? stringFromSessionInfo(resolvedInfo, "country_name") ?? stringFromSessionInfo(resolvedInfo, "country_code");
  if (location && country && location !== country) return `${location}, ${country}`;
  return location ?? country ?? "-";
}

function formatLiveSessionName(snapshot: F1SessionSnapshot | null, resolution: F1SessionResolution | null): string {
  const info = snapshot?.sessionInfo ?? {};
  const resolvedSession = resolution?.session ?? resolution?.nextSession ?? null;
  const resolvedInfo = resolvedSession ? (resolvedSession as Record<string, unknown>) : {};
  const name =
    stringFromSessionInfo(info, "session_name") ??
    stringFromSessionInfo(info, "session_type") ??
    stringFromSessionInfo(resolvedInfo, "session_name") ??
    stringFromSessionInfo(resolvedInfo, "session_type");
  return name ?? "Session";
}

function liveConnectionLabel(status: F1ConnectionStatus): string {
  if (status === "live") return "Live";
  if (status === "polling") return "Polling";
  if (status === "resolving" || status === "connecting") return "Connecting";
  if (status === "upcoming" || status === "scheduled") return "Scheduled";
  return "Unavailable";
}

function formatLiveTrackStatus(snapshot: F1SessionSnapshot | null): string {
  const latest = [...(snapshot?.raceControl ?? [])].reverse().find((message) => {
    const text = [
      message.flag,
      message.category,
      message.message,
      message.status,
      message.track_status,
    ].filter(Boolean).join(" ");
    return text.trim().length > 0;
  });
  if (!latest) return snapshot?.drivers.length ? "All Clear" : "Pending";
  const text = [
    latest.flag,
    latest.category,
    latest.message,
    latest.status,
    latest.track_status,
  ].filter(Boolean).join(" ");
  const normalized = text.toLowerCase();
  if (normalized.includes("green") || normalized.includes("clear")) return "All Clear";
  if (normalized.includes("yellow")) return "Yellow";
  if (normalized.includes("red")) return "Red Flag";
  if (normalized.includes("safety")) return "Safety Car";
  if (normalized.includes("vsc")) return "VSC";
  return text.length > 22 ? `${text.slice(0, 19)}...` : text;
}

function formatLiveSessionClock(snapshot: F1SessionSnapshot | null, resolution: F1SessionResolution | null): string {
  if (isNumber(resolution?.secondsUntilEnd)) return formatClockDuration(resolution.secondsUntilEnd);
  const end = stringFromSessionInfo(snapshot?.sessionInfo ?? {}, "date_end");
  if (!end) return "--:--";
  const remaining = Math.floor((new Date(end).getTime() - Date.now()) / 1000);
  if (!Number.isFinite(remaining)) return "--:--";
  return formatClockDuration(Math.max(0, remaining));
}

function formatClockDuration(value: number): string {
  const totalSeconds = Math.max(0, Math.floor(value));
  const days = Math.floor(totalSeconds / 86_400);
  const hours = Math.floor((totalSeconds % 86_400) / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  const seconds = totalSeconds % 60;
  if (days > 0) return `${days}d ${String(hours).padStart(2, "0")}h`;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function weatherNumber(weather: Record<string, unknown> | null, ...keys: string[]): number | null {
  if (!weather) return null;
  for (const key of keys) {
    const value = weather[key];
    if (isNumber(value)) return value;
  }
  return null;
}

function formatTemperature(value?: number | null): string {
  return isNumber(value) ? `${value.toFixed(1)} C` : "N/A";
}

function formatHumidity(value?: number | null): string {
  return isNumber(value) ? `${value.toFixed(1)}%` : "N/A";
}

function formatRainState(weather: Record<string, unknown> | null): string {
  const rainfall = weatherNumber(weather, "rainfall", "rain");
  if (!isNumber(rainfall)) return "N/A";
  return rainfall > 0 ? "Wet" : "Dry";
}

function formatWindSpeed(weather: Record<string, unknown> | null): string {
  const speed = weatherNumber(weather, "wind_speed", "windSpeed");
  return isNumber(speed) ? `${speed.toFixed(1)} km/h` : "N/A";
}

function formatWindDirection(weather: Record<string, unknown> | null): string {
  const direction = weatherNumber(weather, "wind_direction", "windDirection");
  if (!isNumber(direction)) return "Direction pending";
  const normalized = ((direction % 360) + 360) % 360;
  const labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  const label = labels[Math.round(normalized / 45) % labels.length];
  return `${label} ${Math.round(normalized)} deg`;
}

function practiceLapTime(driver: F1DriverState): number | null {
  return isNumber(driver.best_lap_time) ? driver.best_lap_time : isNumber(driver.last_lap_time) ? driver.last_lap_time : null;
}

function practiceProgramme(driver: F1DriverState): string {
  if (isNumber(driver.best_lap_time)) return (driver.current_lap ?? 0) >= 4 ? "Race run" : "Quali sim";
  if ((driver.current_lap ?? 0) > 0) return "Install only";
  return "No timed run";
}

function practiceConfidence(driver: F1DriverState): number {
  let score = 0;
  if (isNumber(driver.best_lap_time)) score += 35;
  if (isNumber(driver.last_lap_time)) score += 20;
  if (isNumber(driver.sector_times?.sector_1)) score += 12;
  if (isNumber(driver.sector_times?.sector_2)) score += 12;
  if (isNumber(driver.sector_times?.sector_3)) score += 12;
  score += Math.min(9, Math.max(0, driver.current_lap ?? 0));
  return Math.min(100, score);
}

function practiceRankDelta(driver: F1DriverState, fastest: number | null): string {
  const lap = practiceLapTime(driver);
  if (!isNumber(lap) || !isNumber(fastest)) return "#- --";
  return lap === fastest ? "#1 · 0.000s" : `${formatSigned(lap - fastest)}s`;
}

function formatPracticeDeg(driver: F1DriverState): string {
  if (!isNumber(driver.tyre_age) || (driver.current_lap ?? 0) < 4) return "N/A";
  return "Requires run fit";
}

function formatPracticeRuns(driver: F1DriverState): string {
  const laps = driver.current_lap ?? 0;
  if (!laps) return "No timed runs";
  const compound = driver.current_compound ? ` ${driver.current_compound}` : "";
  return `${laps} laps${compound}`;
}

function formatArtifactValue(value: unknown): string {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "-";
    return Number.isInteger(value) ? String(value) : value.toFixed(Math.abs(value) >= 100 ? 1 : 3);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return value.length > 42 ? `${value.slice(0, 39)}...` : value;
  return JSON.stringify(value);
}

function driverLabel(driver?: F1DriverState | null): string {
  if (!driver) return "Pending";
  return `${driver.position ? `P${driver.position} ` : ""}${driver.acronym ?? driver.driver_number}`;
}

function teamColor(driver?: Pick<F1DriverState, "team_colour"> | null): string {
  const colour = driver?.team_colour?.trim();
  if (!colour) return "var(--accent)";
  return colour.startsWith("#") ? colour : `#${colour}`;
}

function compoundClass(compound?: string | null): string {
  const normalized = String(compound ?? "").toLowerCase();
  if (normalized.includes("soft")) return "soft";
  if (normalized.includes("medium")) return "medium";
  if (normalized.includes("hard")) return "hard";
  if (normalized.includes("inter")) return "intermediate";
  if (normalized.includes("wet")) return "wet";
  return "unknown";
}

function formatLap(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(3).padStart(6, "0")}`;
}

function formatWeather(weather?: Record<string, unknown> | null): string {
  const track = typeof weather?.track_temperature === "number" ? weather.track_temperature : null;
  if (track === null) return "Pending";
  return `${track.toFixed(1)} C`;
}

function formatSessionInfo(sessionInfo?: Record<string, unknown> | null): string {
  const name = typeof sessionInfo?.session_name === "string" ? sessionInfo.session_name : null;
  const year = typeof sessionInfo?.year === "number" ? sessionInfo.year : null;
  if (name && year) return `${year} ${name}`;
  if (name) return name;
  return "Pending";
}

function bestSessionSummary(sessions: F1SessionSummary[]): F1SessionSummary | null {
  return [...sessions].sort((a, b) => {
    const eventDelta = (b.eventCount ?? 0) - (a.eventCount ?? 0);
    if (eventDelta !== 0) return eventDelta;
    const driverDelta = (b.drivers ?? 0) - (a.drivers ?? 0);
    if (driverDelta !== 0) return driverDelta;
    return String(b.sessionKey).localeCompare(String(a.sessionKey));
  })[0] ?? null;
}

function formatSessionSummaryTitle(session: F1SessionSummary): string {
  const place =
    cleanSessionLabel(session.location) ??
    cleanSessionLabel(session.eventName) ??
    (session.meetingKey ? `Meeting ${session.meetingKey}` : null) ??
    String(session.sessionKey);
  const name = cleanSessionLabel(session.sessionName) ?? cleanSessionLabel(session.sessionType) ?? "Session";
  return `${place} - ${name}`;
}

function formatSessionSummaryOption(session: F1SessionSummary): string {
  const title = formatSessionSummaryTitle(session);
  const year = session.year ? String(session.year) : null;
  const counts = `${session.drivers} drivers / ${session.eventCount} events`;
  return [title, year, counts].filter(Boolean).join(" · ");
}

function enrichSessionSummaryFromSnapshot(summary: F1SessionSummary, snapshot: F1SessionSnapshot): F1SessionSummary {
  const info = snapshot.sessionInfo ?? {};
  return {
    ...summary,
    seq: summary.seq || snapshot.seq,
    source: summary.source || snapshot.source,
    sessionName: summary.sessionName ?? stringFromSessionInfo(info, "session_name"),
    sessionType: summary.sessionType ?? stringFromSessionInfo(info, "session_type"),
    eventName: summary.eventName ?? stringFromSessionInfo(info, "event_name") ?? stringFromSessionInfo(info, "fastf1_event_name"),
    location: summary.location ?? stringFromSessionInfo(info, "location") ?? stringFromSessionInfo(info, "circuit_short_name"),
    countryName: summary.countryName ?? stringFromSessionInfo(info, "country_name"),
    dateStart: summary.dateStart ?? stringFromSessionInfo(info, "date_start"),
    dateEnd: summary.dateEnd ?? stringFromSessionInfo(info, "date_end"),
    year: summary.year ?? numberFromSessionInfo(info, "year"),
    drivers: summary.drivers || snapshot.drivers.length,
  };
}

function cleanSessionLabel(value?: string | number | null): string | null {
  if (value === null || value === undefined) return null;
  const text = String(value).trim();
  return text || null;
}

function stringFromSessionInfo(info: Record<string, unknown>, key: string): string | null {
  const value = info[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function numberFromSessionInfo(info: Record<string, unknown>, key: string): number | null {
  const value = info[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function sessionKeyFromF1Session(session?: F1SessionInfo | null): string {
  const value = session?.session_key;
  return value === null || value === undefined ? "" : String(value);
}

function sessionMeetingKey(session?: F1SessionInfo | null): string {
  const value = session?.meeting_key;
  return value === null || value === undefined ? "" : String(value);
}

function sessionYear(session?: F1SessionInfo | null): string {
  if (typeof session?.year === "number") return String(session.year);
  if (session?.date_start) {
    const date = new Date(session.date_start);
    if (!Number.isNaN(date.getTime())) return String(date.getUTCFullYear());
  }
  return String(new Date().getFullYear());
}

function formatF1SessionTitle(session?: F1SessionInfo | null): string {
  if (!session) return "F1 session";
  const year = sessionYear(session);
  const name = session.session_name || session.session_type || "Session";
  const location = formatF1SessionLocation(session);
  return location === "-" ? `${year} ${name}` : `${year} ${name} · ${location}`;
}

function formatF1SessionLocation(session?: F1SessionInfo | null): string {
  if (!session) return "-";
  const track = session.circuit_short_name || session.location || null;
  const country = session.country_name || session.country_code || null;
  if (track && country && track !== country) return `${track}, ${country}`;
  return track || country || "-";
}

function isOpenF1Source(source?: string | null): boolean {
  return String(source ?? "").toLowerCase().startsWith("openf1");
}

function sessionSourceLabel(source?: string | null): string {
  const normalized = String(source ?? "").toLowerCase();
  if (normalized.startsWith("fastf1")) return "FastF1 schedule";
  if (normalized.startsWith("openf1")) return "OpenF1";
  return "F1 session resolver";
}

function formatSessionDate(value?: string | null): string {
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

function formatDurationFromSeconds(value?: number | null): string {
  if (!isNumber(value)) return "-";
  const totalSeconds = Math.max(0, Math.floor(value));
  const days = Math.floor(totalSeconds / 86_400);
  const hours = Math.floor((totalSeconds % 86_400) / 3_600);
  const minutes = Math.floor((totalSeconds % 3_600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m`;
  return `${totalSeconds}s`;
}

function statusDotClass(value: string): "ok" | "warn" | "miss" {
  if (value === "live") return "ok";
  if (value === "error") return "miss";
  return "warn";
}

function formatSigned(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function formatSeconds(value?: number | null): string {
  const formatted = formatSigned(value);
  return formatted === "-" ? "-" : `${formatted}s`;
}

function formatPercent(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(0)}%`;
}

function formatPosition(value?: number | null): string {
  if (!isNumber(value)) return "-";
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function formatPositionLabel(value?: number | null): string {
  const formatted = formatPosition(value);
  return formatted === "-" ? "-" : `P${formatted}`;
}

function formatPositionRange(prediction: F1PredictionSnapshot): string {
  const lower = formatPositionLabel(prediction.position_p10);
  const upper = formatPositionLabel(prediction.position_p90);
  if (lower === "-" || upper === "-") return "-";
  return `${lower}-${upper}`;
}

function weatherSeriesFromSnapshot(snapshot: F1SessionSnapshot | null) {
  return (snapshot?.weatherSamples ?? []).map((sample, index) => ({
    index,
    eventTime: typeof sample.event_time === "string" ? sample.event_time : typeof sample.date === "string" ? sample.date : null,
    airTemperature: typeof sample.air_temperature === "number" ? sample.air_temperature : null,
    trackTemperature: typeof sample.track_temperature === "number" ? sample.track_temperature : null,
    rainfall: typeof sample.rainfall === "number" ? sample.rainfall : null,
    windSpeed: typeof sample.wind_speed === "number" ? sample.wind_speed : null,
    sourceId: typeof sample.source_id === "number" ? sample.source_id : null,
  }));
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatNumber(value?: number | null): string {
  if (!isNumber(value)) return "-";
  return Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
}

function formatWeatherTime(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value.slice(11, 16) || value;
  return date.toISOString().slice(11, 16);
}

function formatRelativeTime(value?: string | null): string {
  if (!value) return "pending";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const seconds = Math.max(0, Math.floor((Date.now() - date.getTime()) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return date.toLocaleString();
}

function formatSector(value?: number | null): string {
  if (!isNumber(value)) return "-";
  return value.toFixed(3);
}

function leaderGap(driver: F1DriverState): string {
  if (driver.position === 1) return "Leader";
  return driver.gap_to_leader ?? driver.interval ?? "-";
}

function formatDelta(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}s`;
}

function microSectorClass(delta?: number | null): string {
  if (delta === null || delta === undefined || Number.isNaN(delta)) return "neutral";
  if (delta <= -0.02) return "faster";
  if (delta >= 0.02) return "slower";
  return "equal";
}

function numberOrNull(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

type SnapshotStreamMessage = {
  type: "snapshot";
  payload: F1SessionSnapshot;
};

function isSnapshotStreamMessage(
  message: F1StreamUpdate | SnapshotStreamMessage
): message is SnapshotStreamMessage {
  return message.type === "snapshot";
}
