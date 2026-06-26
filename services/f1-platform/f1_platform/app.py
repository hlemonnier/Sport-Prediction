"""FastAPI application for the F1 platform service."""

from __future__ import annotations

import os
import asyncio
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .event_stream import event_stream_from_url
from .fastf1_analysis import (
    FastF1ArtifactService,
    FastF1ArtifactStore,
    request_from_payload as fastf1_request_from_payload,
)
from .fastf1_schedule import FastF1ScheduleClient
from .openf1_rest import OpenF1RestClient, request_from_payload
from .predictions import prediction_service_from_env
from .projections import projection_store_from_config
from .replay import load_jsonl_events
from .runtime import F1PlatformRuntime
from .sample_data import SAMPLE_SESSION_KEY, sample_events
from .schemas import F1Event
from .storage import JsonlEventStore
from .timed_replay import TimedReplayController
from .time import utc_now_iso
from .track_geometry import FastF1CenterlineProjector, parse_session_aliases
from .weather import fetch_open_meteo_forecast, resolve_circuit_location


def create_app() -> FastAPI:
    app = FastAPI(
        title="F1 Platform API",
        version="0.1.0",
        description="Near-live and replay API for the F1 platform.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    event_store = JsonlEventStore(os.environ.get("F1_PLATFORM_EVENT_STORE", str(_default_event_store_path())))
    websocket_snapshot_interval_seconds = _nonnegative_float(
        os.environ.get("F1_PLATFORM_WS_SNAPSHOT_INTERVAL_SECONDS"),
        default=15.0,
    )
    fastf1_artifacts = FastF1ArtifactService(
        FastF1ArtifactStore(os.environ.get("F1_PLATFORM_FASTF1_ARTIFACT_STORE", str(_default_fastf1_artifact_path()))),
        cache_dir=os.environ.get("F1_PLATFORM_FASTF1_CACHE", str(_default_fastf1_cache_path())),
    )
    track_projector = FastF1CenterlineProjector(
        fastf1_artifacts.artifact_store,
        session_aliases=parse_session_aliases(os.environ.get("F1_PLATFORM_CENTERLINE_SESSION_MAP")),
        fallback_to_latest=_truthy(os.environ.get("F1_PLATFORM_CENTERLINE_FALLBACK_TO_LATEST")),
    )
    projection_store = projection_store_from_config(
        os.environ.get("F1_PLATFORM_DATABASE_URL"),
        os.environ.get("F1_PLATFORM_SQLITE_PROJECTION_STORE", str(_default_projection_store_path())),
    )
    event_stream = event_stream_from_url(os.environ.get("F1_PLATFORM_REDIS_URL"))
    runtime = F1PlatformRuntime(
        prediction_service=prediction_service_from_env(),
        event_stream=event_stream,
        projection_store=projection_store,
        track_projector=track_projector,
    )
    replay_controller = TimedReplayController(runtime, event_store)
    app.state.runtime = runtime
    app.state.event_store = event_store
    app.state.fastf1_artifacts = fastf1_artifacts
    app.state.track_projector = track_projector
    app.state.event_stream = event_stream
    app.state.projection_store = projection_store
    app.state.replay_controller = replay_controller
    app.state.websocket_snapshot_interval_seconds = websocket_snapshot_interval_seconds
    app.state.fastf1_schedule = FastF1ScheduleClient(
        cache_dir=os.environ.get("F1_PLATFORM_FASTF1_CACHE", str(_default_fastf1_cache_path())),
    )
    app.state.f1_session_status_cache = {}
    app.state.f1_session_status_cache_ttl_seconds = _nonnegative_float(
        os.environ.get("F1_PLATFORM_SESSION_STATUS_CACHE_SECONDS"),
        default=20.0,
    )
    app.state.weather_forecast_cache = {}
    app.state.weather_forecast_cache_ttl_seconds = _nonnegative_float(
        os.environ.get("F1_PLATFORM_WEATHER_FORECAST_CACHE_SECONDS"),
        default=900.0,
    )
    app.state.openf1_session_status_cache = {}
    app.state.openf1_session_status_cache_ttl_seconds = _nonnegative_float(
        os.environ.get("F1_PLATFORM_OPENF1_SESSION_STATUS_CACHE_SECONDS"),
        default=20.0,
    )

    @app.on_event("startup")
    async def seed_sample() -> None:
        await runtime.reset_session(SAMPLE_SESSION_KEY, sample_events(SAMPLE_SESSION_KEY))

    @app.on_event("shutdown")
    async def shutdown_replays() -> None:
        await replay_controller.shutdown()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "f1-platform",
            "sessions": len(runtime.reducers),
            "projectionStore": projection_store.kind,
            "trackProjection": "fastf1-centerline",
            "websocketSnapshotIntervalSeconds": websocket_snapshot_interval_seconds,
            "time": utc_now_iso(),
        }

    @app.get("/api/f1/sessions")
    async def sessions() -> dict[str, Any]:
        return {"sessions": runtime.list_sessions()}

    @app.get("/api/f1/sessions/{session_key}/snapshot")
    async def snapshot(session_key: str) -> dict[str, Any]:
        state = await runtime.snapshot(session_key)
        return state.to_dict()

    @app.get("/api/f1/sessions/{session_key}/events")
    async def recent_events(session_key: str, count: int = 100) -> dict[str, Any]:
        bounded_count = max(1, min(1_000, count))
        return {"events": await runtime.recent_events(session_key, count=bounded_count)}

    @app.get("/api/f1/sessions/{session_key}/projection")
    async def projection(session_key: str) -> dict[str, Any]:
        return await _run_blocking(lambda: projection_store.session_counts(session_key))

    @app.get("/api/f1/sessions/{session_key}/analytics")
    async def analytics(session_key: str) -> dict[str, Any]:
        return await _run_blocking(lambda: projection_store.derived_analytics(session_key))

    @app.get("/api/f1/sessions/{session_key}/track-geometry")
    async def track_geometry(
        session_key: str,
        centerline_session_key: str | None = None,
        limit: int = 900,
    ) -> dict[str, Any]:
        centerline = await _run_blocking(
            lambda: app.state.track_projector.centerline_for_session(
                session_key,
                centerline_session_key=centerline_session_key or None,
            )
        )
        if centerline is None:
            raise HTTPException(status_code=404, detail=f"No FastF1 centerline found for session {session_key}")
        return _centerline_payload(centerline, limit=max(2, min(2_000, limit)))

    @app.post("/api/f1/sessions/{session_key}/events")
    async def ingest_event(session_key: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            event = F1Event.from_payload(body, received_at=utc_now_iso(), session_key=session_key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await _run_blocking(lambda: event_store.append(event))
        update = await runtime.ingest(event)
        if update is None:
            return {"accepted": False, "reason": "duplicate_or_stale"}
        return {"accepted": True, "update": update.to_dict()}

    @app.post("/api/f1/sessions/{session_key}/replay/reset")
    async def reset_replay(session_key: str) -> dict[str, Any]:
        state = await runtime.reset_session(session_key, sample_events(session_key))
        return state.to_dict()

    @app.post("/api/f1/openf1/import")
    async def import_openf1(body: dict[str, Any]) -> dict[str, Any]:
        request = request_from_payload(body)
        client = OpenF1RestClient.from_env()
        try:
            result = await _run_blocking(lambda: client.import_session(request))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"OpenF1 import failed: {exc}") from exc

        replay_path = event_store.replace(result.session_key, result.events)
        fastf1_session_key = _optional_str(body.get("fastf1_session_key", body.get("centerline_session_key")))
        if fastf1_session_key:
            app.state.track_projector.set_session_alias(result.session_key, fastf1_session_key)
        state = await runtime.reset_session(
            result.session_key,
            result.events,
            source="openf1-rest",
            replay_meta={
                "mode": "openf1-rest-import",
                "eventCount": len(result.events),
                "topicCounts": result.topic_counts,
                "replayPath": str(replay_path),
                "sourceUrl": result.source_url,
                "sessionName": result.session_name,
                "meetingKey": result.meeting_key,
                "fastf1SessionKey": fastf1_session_key,
            },
        )
        return {
            "imported": True,
            "sessionKey": result.session_key,
            "eventCount": len(result.events),
            "topicCounts": result.topic_counts,
            "replayPath": str(replay_path),
            "snapshot": state.to_dict(),
        }

    @app.get("/api/f1/session-status")
    async def f1_session_status(year: int | None = None, now: str | None = None) -> dict[str, Any]:
        cache_key = f"{year or ''}:{now or ''}"
        cache_ttl = app.state.f1_session_status_cache_ttl_seconds
        cache_entry = app.state.f1_session_status_cache.get(cache_key)
        monotonic_now = time.monotonic()
        if cache_entry and cache_ttl > 0:
            cached_at, cached_payload = cache_entry
            if monotonic_now - cached_at <= cache_ttl:
                return dict(cached_payload)

        fastf1_unavailable: dict[str, Any] | None = None
        fastf1_error: str | None = None
        try:
            fastf1_payload = await _run_blocking(
                lambda: app.state.fastf1_schedule.resolve_live_or_next_session(year=year, now=now)
            )
            if fastf1_payload.get("status") != "unavailable":
                app.state.f1_session_status_cache[cache_key] = (monotonic_now, dict(fastf1_payload))
                return fastf1_payload
            fastf1_unavailable = dict(fastf1_payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            fastf1_error = f"FastF1 schedule failed: {exc}"

        try:
            openf1_payload = await _run_blocking(lambda: OpenF1RestClient.from_env().resolve_live_or_next_session(year=year, now=now))
            if fastf1_error:
                openf1_payload = dict(openf1_payload)
                openf1_payload["fallbackReason"] = fastf1_error
            app.state.f1_session_status_cache[cache_key] = (monotonic_now, dict(openf1_payload))
            return openf1_payload
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            openf1_error = f"OpenF1 fallback failed: {exc}"

        if fastf1_unavailable is not None:
            fastf1_unavailable["message"] = f"{fastf1_unavailable.get('message', 'FastF1 schedule unavailable')} {openf1_error}"
            app.state.f1_session_status_cache[cache_key] = (monotonic_now, dict(fastf1_unavailable))
            return fastf1_unavailable

        payload = {
            "status": "unavailable",
            "source": "f1-session-resolver",
            "resolvedAt": utc_now_iso(),
            "message": "F1 session status could not be resolved. "
            + " ".join(message for message in (fastf1_error, openf1_error) if message),
            "session": None,
            "nextSession": None,
            "secondsUntilStart": None,
            "secondsUntilEnd": None,
        }
        app.state.f1_session_status_cache[cache_key] = (monotonic_now, dict(payload))
        return payload

    @app.get("/api/f1/openf1/session-status")
    async def openf1_session_status(year: int | None = None, now: str | None = None) -> dict[str, Any]:
        cache_key = f"{year or ''}:{now or ''}"
        cache_ttl = app.state.openf1_session_status_cache_ttl_seconds
        cache_entry = app.state.openf1_session_status_cache.get(cache_key)
        monotonic_now = time.monotonic()
        if cache_entry and cache_ttl > 0:
            cached_at, cached_payload = cache_entry
            if monotonic_now - cached_at <= cache_ttl:
                return dict(cached_payload)

        client = OpenF1RestClient.from_env()
        try:
            payload = await _run_blocking(lambda: client.resolve_live_or_next_session(year=year, now=now))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"OpenF1 session status failed: {exc}") from exc
        app.state.openf1_session_status_cache[cache_key] = (monotonic_now, dict(payload))
        return payload

    @app.get("/api/f1/fastf1/schedule")
    async def fastf1_schedule(year: int | None = None) -> dict[str, Any]:
        selected_year = year or _current_utc_year()
        try:
            return await _run_blocking(lambda: app.state.fastf1_schedule.season_schedule(year=selected_year))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"FastF1 schedule failed: {exc}") from exc

    @app.get("/api/f1/weather/forecast")
    async def f1_weather_forecast(
        location: str | None = None,
        year: int | None = None,
        forecast_days: int = 7,
    ) -> dict[str, Any]:
        try:
            resolution = await _run_blocking(lambda: app.state.fastf1_schedule.resolve_live_or_next_session(year=year))
        except Exception:
            resolution = None
        session = _weather_session_from_resolution(resolution) if isinstance(resolution, dict) else None
        circuit, matched_by = resolve_circuit_location(
            location,
            session.get("circuit_short_name") if session else None,
            session.get("location") if session else None,
            session.get("country_name") if session else None,
            session.get("fastf1_event_name") if session else None,
            session.get("official_event_name") if session else None,
        )
        if circuit is None:
            raise HTTPException(status_code=404, detail="No mapped F1 circuit coordinates found for the requested session/location")

        cache_key = f"{circuit.id}:{max(1, min(16, int(forecast_days)))}"
        cache_ttl = app.state.weather_forecast_cache_ttl_seconds
        cache_entry = app.state.weather_forecast_cache.get(cache_key)
        monotonic_now = time.monotonic()
        if cache_entry and cache_ttl > 0:
            cached_at, cached_payload = cache_entry
            if monotonic_now - cached_at <= cache_ttl:
                return dict(cached_payload)

        try:
            payload = await _run_blocking(lambda: fetch_open_meteo_forecast(circuit, forecast_days=forecast_days))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Open-Meteo forecast failed: {exc}") from exc
        payload["matchedBy"] = matched_by
        payload["session"] = session
        payload["sessionResolution"] = {
            "status": resolution.get("status"),
            "source": resolution.get("source"),
            "message": resolution.get("message"),
        } if isinstance(resolution, dict) else None
        app.state.weather_forecast_cache[cache_key] = (monotonic_now, dict(payload))
        return payload

    @app.post("/api/f1/fastf1/import")
    async def import_fastf1(body: dict[str, Any]) -> dict[str, Any]:
        try:
            request = fastf1_request_from_payload(body)
            result = await _run_blocking(lambda: app.state.fastf1_artifacts.import_session(request))
            app.state.track_projector.clear_cache()
            mapped_session_key = _optional_str(body.get("map_to_session_key", body.get("openf1_session_key")))
            if mapped_session_key:
                app.state.track_projector.set_session_alias(mapped_session_key, result.session_key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"FastF1 import failed: {exc}") from exc
        return {"imported": True, **result.to_dict()}

    @app.get("/api/f1/fastf1/artifacts")
    async def list_fastf1_artifacts(
        session_key: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        bounded_limit = max(1, min(1_000, limit))
        artifacts = await _run_blocking(
            lambda: app.state.fastf1_artifacts.list_artifacts(
                session_key=session_key or None,
                kind=kind or None,
                limit=bounded_limit,
            )
        )
        return {
            "artifacts": [artifact.to_dict() for artifact in artifacts],
            "count": len(artifacts),
            "limit": bounded_limit,
        }

    @app.get("/api/f1/fastf1/engineering-summary")
    async def fastf1_engineering_summary(session_key: str | None = None) -> dict[str, Any]:
        try:
            return await _run_blocking(
                lambda: app.state.fastf1_artifacts.engineering_summary(
                    session_key=session_key or None,
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/f1/fastf1/artifacts/{artifact_id}/rows")
    async def read_fastf1_artifact_rows(artifact_id: str, limit: int = 200) -> dict[str, Any]:
        try:
            return await _run_blocking(
                lambda: app.state.fastf1_artifacts.read_artifact_rows(
                    artifact_id,
                    limit=max(1, min(1_000, limit)),
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/f1/sessions/{session_key}/replay/load")
    async def load_replay(session_key: str) -> dict[str, Any]:
        replay_path = event_store.path_for_session(session_key)
        if not replay_path.exists():
            raise HTTPException(status_code=404, detail=f"No stored replay for session {session_key}")
        events = load_jsonl_events(replay_path, session_key=session_key)
        state = await runtime.reset_session(
            session_key,
            events,
            source="jsonl-replay",
            replay_meta={
                "mode": "jsonl-replay",
                "eventCount": len(events),
                "replayPath": str(replay_path),
            },
        )
        return state.to_dict()

    @app.post("/api/f1/sessions/{session_key}/replay/start")
    async def start_timed_replay(session_key: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = body or {}
        try:
            status = await replay_controller.start(
                session_key,
                speed=_positive_float(payload.get("speed"), default=1.0),
                max_delay_seconds=_nonnegative_float(payload.get("max_delay_seconds"), default=2.0),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return status.to_dict()

    @app.post("/api/f1/sessions/{session_key}/replay/stop")
    async def stop_timed_replay(session_key: str) -> dict[str, Any]:
        status = await replay_controller.stop(session_key)
        return status.to_dict()

    @app.get("/api/f1/sessions/{session_key}/replay/status")
    async def timed_replay_status(session_key: str) -> dict[str, Any]:
        return replay_controller.status(session_key).to_dict()

    @app.websocket("/api/f1/sessions/{session_key}/stream")
    async def stream(websocket: WebSocket, session_key: str) -> None:
        await websocket.accept()
        queue = await runtime.subscribe(session_key)
        try:
            await _send_stream_snapshot(websocket, runtime, session_key, reason="initial")
            loop = asyncio.get_running_loop()
            snapshot_interval = app.state.websocket_snapshot_interval_seconds
            next_snapshot_at = loop.time() + snapshot_interval if snapshot_interval > 0 else None
            while True:
                if next_snapshot_at is None:
                    update = await queue.get()
                else:
                    timeout = max(0.0, next_snapshot_at - loop.time())
                    try:
                        update = await asyncio.wait_for(queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        await _send_stream_snapshot(websocket, runtime, session_key, reason="periodic")
                        next_snapshot_at = loop.time() + snapshot_interval
                        continue
                await websocket.send_json(update.to_dict())
                if next_snapshot_at is not None and loop.time() >= next_snapshot_at:
                    await _send_stream_snapshot(websocket, runtime, session_key, reason="periodic")
                    next_snapshot_at = loop.time() + snapshot_interval
        except asyncio.CancelledError:
            return
        except WebSocketDisconnect:
            return
        finally:
            runtime.unsubscribe(session_key, queue)

    return app


async def _run_blocking(func):
    import asyncio

    return await asyncio.to_thread(func)


async def _send_stream_snapshot(
    websocket: WebSocket,
    runtime: F1PlatformRuntime,
    session_key: int | str,
    *,
    reason: str,
) -> None:
    state = await runtime.snapshot(session_key)
    await websocket.send_json({"type": "snapshot", "reason": reason, "payload": state.to_dict()})


def _positive_float(value: Any, *, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected a positive number") from exc
    if parsed <= 0:
        raise ValueError("Expected a positive number")
    return parsed


def _nonnegative_float(value: Any, *, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected a non-negative number") from exc
    if parsed < 0:
        raise ValueError("Expected a non-negative number")
    return parsed


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "latest"}


def _weather_session_from_resolution(resolution: dict[str, Any]) -> dict[str, Any] | None:
    session = resolution.get("session") or resolution.get("nextSession")
    return dict(session) if isinstance(session, dict) else None


def _current_utc_year() -> int:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).year


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _centerline_payload(centerline: Any, *, limit: int) -> dict[str, Any]:
    points = list(centerline.points)
    sampled = _sample_centerline_points(points, limit=limit)
    return {
        "sessionKey": centerline.session_key,
        "artifactId": centerline.artifact_id,
        "source": centerline.source,
        "pointCount": len(points),
        "sampledPointCount": len(sampled),
        "points": [
            {
                "distance": round(point.distance, 3),
                "progress": round(point.progress, 9),
                "x": round(point.x, 3),
                "y": round(point.y, 3),
                "z": round(point.z, 3),
            }
            for point in sampled
        ],
    }


def _sample_centerline_points(points: list[Any], *, limit: int) -> list[Any]:
    if len(points) <= limit:
        return points
    if limit <= 2:
        return [points[0], points[-1]]
    step = (len(points) - 1) / (limit - 1)
    sampled = [points[min(len(points) - 1, round(index * step))] for index in range(limit)]
    sampled[0] = points[0]
    sampled[-1] = points[-1]
    return sampled


def _default_event_store_path() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists() and (candidate / "data" / "raw" / "f1").exists():
            return candidate / "data" / "raw" / "f1" / "platform-events"
    return Path("data/raw/f1/platform-events")


def _default_projection_store_path() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists() and (candidate / "data" / "raw" / "f1").exists():
            return candidate / "data" / "raw" / "f1" / "platform-projections.sqlite"
    return Path("data/raw/f1/platform-projections.sqlite")


def _default_fastf1_artifact_path() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists() and (candidate / "data" / "raw" / "f1").exists():
            return candidate / "data" / "raw" / "f1" / "fastf1-artifacts"
    return Path("data/raw/f1/fastf1-artifacts")


def _default_fastf1_cache_path() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists() and (candidate / "data" / "raw" / "f1").exists():
            return candidate / "data" / "raw" / "f1" / "fastf1-cache"
    return Path("data/raw/f1/fastf1-cache")


app = create_app()
