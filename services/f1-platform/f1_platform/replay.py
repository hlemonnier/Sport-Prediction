"""Replay support for deterministic live-pipeline development."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .reducer import F1StateReducer
from .schemas import F1Event, JsonObject, StateUpdate
from .time import utc_now_iso


def load_jsonl_events(path: str | Path, *, session_key: int | str | None = None) -> list[F1Event]:
    events: list[F1Event] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                events.append(F1Event.from_record(raw))
            except ValueError:
                events.append(F1Event.from_payload(raw, received_at=utc_now_iso(), session_key=session_key))
    return events


def run_replay(
    events: Iterable[F1Event],
    *,
    session_key: int | str,
    source: str = "openf1-replay",
) -> tuple[F1StateReducer, list[StateUpdate]]:
    reducer = F1StateReducer(session_key=session_key, source=source)
    updates: list[StateUpdate] = []
    for event in events:
        update = reducer.ingest(event)
        if update is not None:
            updates.append(update)
    reducer.replay_meta["mode"] = source
    return reducer, updates


def raw_event(
    source_id: int,
    topic: str,
    source_key: str,
    session_key: int | str,
    payload: JsonObject,
    *,
    driver_number: int | None = None,
    meeting_key: int | None = 2026001,
    event_time: str | None = None,
) -> F1Event:
    merged = dict(payload)
    if driver_number is not None:
        merged.setdefault("driver_number", driver_number)
    merged.setdefault("session_key", session_key)
    if meeting_key is not None:
        merged.setdefault("meeting_key", meeting_key)
    if event_time is not None:
        merged.setdefault("date", event_time)
    return F1Event.from_payload(
        {
            "topic": topic,
            "source_id": source_id,
            "source_key": source_key,
            "session_key": session_key,
            "meeting_key": meeting_key,
            "driver_number": driver_number,
            "event_time": event_time,
            "payload": merged,
        },
        received_at=utc_now_iso(),
    )
