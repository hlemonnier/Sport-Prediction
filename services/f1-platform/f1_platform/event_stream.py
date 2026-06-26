"""Ordered raw-event stream adapters.

The platform treats Redis Streams as the production event log, while tests and
local runs can use the in-memory implementation without starting Redis.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from .schemas import F1Event, JsonObject


@dataclass(slots=True)
class EventStreamRecord:
    id: str
    session_key: str
    event: F1Event

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "sessionKey": self.session_key,
            "event": self.event.to_dict(),
        }


class EventStream(Protocol):
    async def append(self, event: F1Event) -> str:
        ...

    async def read_session(self, session_key: int | str, *, count: int = 100) -> list[EventStreamRecord]:
        ...


class InMemoryEventStream:
    def __init__(self) -> None:
        self._records: dict[str, list[EventStreamRecord]] = defaultdict(list)
        self._next_id = 1

    async def append(self, event: F1Event) -> str:
        session_id = str(event.session_key)
        stream_id = f"{self._next_id}-0"
        self._next_id += 1
        self._records[session_id].append(EventStreamRecord(id=stream_id, session_key=session_id, event=event))
        return stream_id

    async def read_session(self, session_key: int | str, *, count: int = 100) -> list[EventStreamRecord]:
        records = self._records.get(str(session_key), [])
        return records[-max(0, count) :]


class RedisEventStream:
    def __init__(self, redis_url: str, *, key_prefix: str = "f1:events") -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise RuntimeError("Install the redis package to enable Redis Streams") from exc
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.key_prefix = key_prefix

    async def append(self, event: F1Event) -> str:
        return str(
            await self.redis.xadd(
                self._key(event.session_key),
                {"event": json.dumps(event.to_dict(), sort_keys=True)},
            )
        )

    async def read_session(self, session_key: int | str, *, count: int = 100) -> list[EventStreamRecord]:
        rows = await self.redis.xrevrange(self._key(session_key), count=max(0, count))
        records: list[EventStreamRecord] = []
        for stream_id, fields in reversed(rows):
            raw = json.loads(fields["event"])
            records.append(
                EventStreamRecord(
                    id=str(stream_id),
                    session_key=str(session_key),
                    event=F1Event.from_record(raw),
                )
            )
        return records

    def _key(self, session_key: int | str) -> str:
        return f"{self.key_prefix}:{session_key}"


def event_stream_from_url(redis_url: str | None) -> EventStream:
    if redis_url:
        return RedisEventStream(redis_url)
    return InMemoryEventStream()
