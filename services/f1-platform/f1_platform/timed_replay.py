"""Timed replay runner for stored OpenF1 event fixtures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from .replay import load_jsonl_events
from .runtime import F1PlatformRuntime
from .schemas import F1Event, JsonObject
from .storage import JsonlEventStore
from .time import utc_now_iso

SleepFn = Callable[[float], Awaitable[None]]


@dataclass(slots=True)
class ReplayRunStatus:
    session_key: str
    state: str
    speed: float
    event_count: int
    cursor: int = 0
    replay_path: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    def to_dict(self) -> JsonObject:
        return {
            "sessionKey": self.session_key,
            "state": self.state,
            "speed": self.speed,
            "eventCount": self.event_count,
            "cursor": self.cursor,
            "replayPath": self.replay_path,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "error": self.error,
        }


class TimedReplayController:
    def __init__(
        self,
        runtime: F1PlatformRuntime,
        event_store: JsonlEventStore,
        *,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self.runtime = runtime
        self.event_store = event_store
        self.sleep = sleep
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._statuses: dict[str, ReplayRunStatus] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        session_key: int | str,
        *,
        speed: float = 1.0,
        max_delay_seconds: float = 2.0,
    ) -> ReplayRunStatus:
        session_id = str(session_key)
        if speed <= 0:
            raise ValueError("Replay speed must be positive")
        replay_path = self.event_store.path_for_session(session_id)
        if not replay_path.exists():
            raise FileNotFoundError(f"No stored replay for session {session_id}")
        events = load_jsonl_events(replay_path, session_key=session_id)
        status = ReplayRunStatus(
            session_key=session_id,
            state="starting",
            speed=float(speed),
            event_count=len(events),
            replay_path=str(replay_path),
            started_at=utc_now_iso(),
        )
        async with self._lock:
            existing = self._tasks.get(session_id)
            if existing is not None and not existing.done():
                raise RuntimeError(f"Replay already running for session {session_id}")
            self._statuses[session_id] = status
            task = asyncio.create_task(self._run(status, events, max_delay_seconds=max_delay_seconds))
            self._tasks[session_id] = task
        return status

    async def stop(self, session_key: int | str) -> ReplayRunStatus:
        session_id = str(session_key)
        async with self._lock:
            task = self._tasks.get(session_id)
            status = self._statuses.get(session_id)
            if task is not None and not task.done():
                task.cancel()
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
        if status is None:
            status = ReplayRunStatus(session_key=session_id, state="idle", speed=0.0, event_count=0)
            self._statuses[session_id] = status
        return status

    async def wait(self, session_key: int | str) -> ReplayRunStatus:
        session_id = str(session_key)
        task = self._tasks.get(session_id)
        if task is not None:
            await task
        return self.status(session_id)

    def status(self, session_key: int | str) -> ReplayRunStatus:
        session_id = str(session_key)
        return self._statuses.get(session_id) or ReplayRunStatus(
            session_key=session_id,
            state="idle",
            speed=0.0,
            event_count=0,
        )

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        for task in list(self._tasks.values()):
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _run(
        self,
        status: ReplayRunStatus,
        events: list[F1Event],
        *,
        max_delay_seconds: float,
    ) -> None:
        previous_event: F1Event | None = None
        try:
            status.state = "running"
            await self.runtime.reset_session(
                status.session_key,
                [],
                source="jsonl-timed-replay",
                replay_meta={
                    "mode": "timed-replay",
                    "speed": status.speed,
                    "eventCount": status.event_count,
                    "replayPath": status.replay_path,
                },
            )
            for event in events:
                delay = replay_delay_seconds(previous_event, event, speed=status.speed)
                if delay > 0:
                    await self.sleep(min(delay, max(0.0, max_delay_seconds)))
                await self.runtime.ingest(event)
                previous_event = event
                status.cursor += 1
            status.state = "finished"
        except asyncio.CancelledError:
            status.state = "stopped"
            raise
        except Exception as exc:
            status.state = "error"
            status.error = str(exc)
        finally:
            status.finished_at = utc_now_iso()


def replay_delay_seconds(previous: F1Event | None, current: F1Event, *, speed: float) -> float:
    if previous is None or speed <= 0:
        return 0.0
    previous_time = _parse_event_time(previous.event_time)
    current_time = _parse_event_time(current.event_time)
    if previous_time is None or current_time is None:
        return 0.0
    return max(0.0, (current_time - previous_time).total_seconds() / speed)


def _parse_event_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
