"""In-memory runtime used by FastAPI and tests."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable

from .event_stream import EventStream, InMemoryEventStream
from .predictions import HeuristicPredictionService, PredictionService
from .projections import NoopProjectionStore, ProjectionStore
from .reducer import F1StateReducer
from .schemas import F1Event, JsonObject, SessionSnapshot, StateUpdate
from .track_geometry import TrackProjectionProvider


class F1PlatformRuntime:
    def __init__(
        self,
        prediction_service: PredictionService | None = None,
        event_stream: EventStream | None = None,
        projection_store: ProjectionStore | None = None,
        track_projector: TrackProjectionProvider | None = None,
    ) -> None:
        self.reducers: dict[str, F1StateReducer] = {}
        self.prediction_service = prediction_service or HeuristicPredictionService()
        self.event_stream = event_stream or InMemoryEventStream()
        self.projection_store = projection_store or NoopProjectionStore()
        self.track_projector = track_projector
        self.prediction_history: dict[str, list] = defaultdict(list)
        self.subscribers: dict[str, set[asyncio.Queue[StateUpdate]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    def ensure_session(self, session_key: int | str) -> F1StateReducer:
        session_id = _session_id(session_key)
        if session_id not in self.reducers:
            self.reducers[session_id] = F1StateReducer(session_key=session_id, track_projector=self.track_projector)
        return self.reducers[session_id]

    async def reset_session(
        self,
        session_key: int | str,
        events: Iterable[F1Event],
        *,
        source: str = "openf1-replay",
        replay_meta: JsonObject | None = None,
    ) -> SessionSnapshot:
        session_id = _session_id(session_key)
        async with self._lock:
            reducer = F1StateReducer(session_key=session_id, source=source, track_projector=self.track_projector)
            updates = []
            for event in events:
                update = reducer.ingest(event)
                if update is not None:
                    updates.append(update)
            reducer.replay_meta.update(replay_meta or {"mode": "sample-replay"})
            self.reducers[session_id] = reducer
            base_snapshot = reducer.snapshot()
            self.prediction_history[session_id] = await self.prediction_service.predict_race(base_snapshot)
            snapshot = self._snapshot_with_history(reducer)
        for update in updates[-10:]:
            await self._publish(session_id, update)
        await self._project(snapshot)
        return snapshot

    async def ingest(self, event: F1Event) -> StateUpdate | None:
        session_id = _session_id(event.session_key)
        snapshot_to_project: SessionSnapshot | None = None
        await self.event_stream.append(event)
        async with self._lock:
            reducer = self.ensure_session(session_id)
            update = reducer.ingest(event)
            if update is not None and _is_meaningful_prediction_event(update.type):
                snapshot = reducer.snapshot()
                predictions = await self.prediction_service.predict_race(snapshot)
                self.prediction_history[session_id].extend(predictions)
            if update is not None:
                snapshot_to_project = self._snapshot_with_history(reducer)
        if update is not None:
            await self._publish(session_id, update)
        if snapshot_to_project is not None:
            await self._project(snapshot_to_project)
        return update

    async def recent_events(self, session_key: int | str, *, count: int = 100) -> list[JsonObject]:
        records = await self.event_stream.read_session(session_key, count=count)
        return [record.to_dict() for record in records]

    async def snapshot(self, session_key: int | str) -> SessionSnapshot:
        async with self._lock:
            reducer = self.ensure_session(session_key)
            return self._snapshot_with_history(reducer)

    def list_sessions(self) -> list[JsonObject]:
        summaries = []
        for session_key, reducer in sorted(self.reducers.items(), key=lambda item: str(item[0])):
            info = reducer.session_info or {}
            summaries.append(
                {
                    "sessionKey": session_key,
                    "seq": reducer.seq,
                    "source": reducer.source,
                    "meetingKey": info.get("meeting_key"),
                    "sessionName": info.get("session_name"),
                    "sessionType": info.get("session_type"),
                    "eventName": info.get("event_name") or info.get("fastf1_event_name") or info.get("meeting_name"),
                    "location": info.get("location") or info.get("circuit_short_name"),
                    "countryName": info.get("country_name"),
                    "dateStart": info.get("date_start"),
                    "dateEnd": info.get("date_end"),
                    "year": info.get("year"),
                    "drivers": len(reducer.drivers),
                    "eventCount": reducer.replay_meta.get("eventCount", 0),
                }
            )
        return summaries

    async def subscribe(self, session_key: int | str) -> asyncio.Queue[StateUpdate]:
        queue: asyncio.Queue[StateUpdate] = asyncio.Queue(maxsize=200)
        self.subscribers[_session_id(session_key)].add(queue)
        return queue

    def unsubscribe(self, session_key: int | str, queue: asyncio.Queue[StateUpdate]) -> None:
        self.subscribers[_session_id(session_key)].discard(queue)

    def _snapshot_with_history(self, reducer: F1StateReducer) -> SessionSnapshot:
        snapshot = reducer.snapshot()
        history = self.prediction_history[_session_id(reducer.session_key)]
        snapshot.predictions = history[-80:]
        return snapshot

    async def _project(self, snapshot: SessionSnapshot) -> None:
        await asyncio.to_thread(self.projection_store.project_snapshot, snapshot)

    async def _publish(self, session_key: int | str, update: StateUpdate) -> None:
        session_id = _session_id(session_key)
        stale = []
        for queue in self.subscribers.get(session_id, set()):
            try:
                queue.put_nowait(update)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            self.unsubscribe(session_id, queue)


def _is_meaningful_prediction_event(update_type: str) -> bool:
    return update_type in {
        "lap.updated",
        "pit.updated",
        "race_control.updated",
        "weather.updated",
        "position.updated",
        "stint.updated",
        "overtake.updated",
        "session_result.updated",
    }


def _session_id(session_key: int | str) -> str:
    return str(session_key)
