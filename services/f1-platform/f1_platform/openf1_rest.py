"""Historical OpenF1 REST importer.

The live platform reducer should not care whether events came from live MQTT,
REST bootstrap, or replayed JSONL. This adapter pulls bounded historical REST
records and converts them into the same `F1Event` stream consumed by the live
runtime.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .openf1 import OpenF1TokenManager
from .schemas import F1Event, JsonObject
from .time import utc_now_iso

DEFAULT_REST_TOPICS = (
    "sessions",
    "drivers",
    "position",
    "intervals",
    "laps",
    "stints",
    "pit",
    "race_control",
    "weather",
    "overtakes",
    "session_result",
)

TELEMETRY_TOPICS = ("car_data", "location")


@dataclass(slots=True)
class OpenF1ImportRequest:
    year: int | None = None
    meeting_key: int | None = None
    session_key: int | str | None = None
    session_name: str = "Race"
    topics: tuple[str, ...] = DEFAULT_REST_TOPICS
    include_telemetry: bool = False
    limit_per_topic: int = 2_000


@dataclass(slots=True)
class OpenF1ImportResult:
    session_key: int | str
    events: list[F1Event]
    meeting_key: int | None
    session_name: str
    topic_counts: dict[str, int]
    source_url: str


class OpenF1RestClient:
    def __init__(
        self,
        *,
        base_url: str = "https://api.openf1.org/v1",
        timeout_seconds: float = 20.0,
        request_interval_seconds: float = 0.35,
        auth_token: str | None = None,
        token_provider: Callable[[], str] | None = None,
        transport: Callable[[str], list[JsonObject]] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.request_interval_seconds = request_interval_seconds
        self._auth_token = auth_token
        self._token_provider = token_provider
        self._transport = transport

    @classmethod
    def from_env(cls) -> "OpenF1RestClient":
        token = os.environ.get("OPENF1_ACCESS_TOKEN") or os.environ.get("OPENF1_BEARER_TOKEN")
        if token:
            return cls(
                base_url=os.environ.get("OPENF1_REST_BASE_URL", "https://api.openf1.org/v1"),
                auth_token=token,
            )
        if os.environ.get("OPENF1_USERNAME") and os.environ.get("OPENF1_PASSWORD"):
            token_manager = OpenF1TokenManager.from_env()
            return cls(
                base_url=os.environ.get("OPENF1_REST_BASE_URL", "https://api.openf1.org/v1"),
                token_provider=lambda: _token_from_manager(token_manager),
            )
        return cls(base_url=os.environ.get("OPENF1_REST_BASE_URL", "https://api.openf1.org/v1"))

    def import_session(self, request: OpenF1ImportRequest) -> OpenF1ImportResult:
        session = self._resolve_session(request)
        session_key = session.get("session_key")
        if session_key is None:
            raise ValueError("OpenF1 session did not include session_key")
        meeting_key = _optional_int(session.get("meeting_key", request.meeting_key))
        topics = tuple(request.topics) + (TELEMETRY_TOPICS if request.include_telemetry else ())

        collected: list[tuple[str, JsonObject]] = []
        topic_counts: dict[str, int] = {}
        for topic in topics:
            normalized = topic.removeprefix("v1/")
            rows = self.fetch_json(normalized, {"session_key": session_key})
            bounded = rows[: max(0, int(request.limit_per_topic))]
            topic_counts[normalized] = len(bounded)
            collected.extend((normalized, dict(row)) for row in bounded)
            if self.request_interval_seconds > 0:
                time.sleep(self.request_interval_seconds)

        collected.sort(key=lambda item: (_event_sort_key(item[1]), _topic_order(item[0]), _source_key(item[0], item[1])))

        events: list[F1Event] = []
        for source_id, (topic, payload) in enumerate(collected, start=1):
            event_payload = dict(payload)
            event_payload.setdefault("session_key", session_key)
            if meeting_key is not None:
                event_payload.setdefault("meeting_key", meeting_key)
            events.append(
                F1Event.from_payload(
                    {
                        "source": "openf1-rest",
                        "topic": f"v1/{topic}",
                        "source_id": source_id,
                        "source_key": _source_key(topic, event_payload),
                        "session_key": session_key,
                        "meeting_key": meeting_key,
                        "driver_number": event_payload.get("driver_number"),
                        "event_time": event_payload.get("date", event_payload.get("date_start")),
                        "payload": event_payload,
                    },
                    received_at=utc_now_iso(),
                )
            )

        return OpenF1ImportResult(
            session_key=session_key,
            events=events,
            meeting_key=meeting_key,
            session_name=str(session.get("session_name") or request.session_name),
            topic_counts=topic_counts,
            source_url=self.base_url,
        )

    def fetch_json(self, endpoint: str, params: Mapping[str, Any] | None = None) -> list[JsonObject]:
        query = urlencode({key: value for key, value in (params or {}).items() if value is not None})
        url = f"{self.base_url}/{endpoint.removeprefix('/')}"
        if query:
            url = f"{url}?{query}"
        headers = {"User-Agent": "sport-prediction-f1-platform/0.1"}
        token = self._token_provider() if self._token_provider is not None else self._auth_token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(url, headers=headers)
        try:
            if self._transport is not None:
                payload = self._transport(url)
            else:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404 and _is_openf1_no_results(detail):
                return []
            if exc.code == 401 and not token:
                raise RuntimeError(
                    "OpenF1 REST API requires authentication; set OPENF1_ACCESS_TOKEN or OPENF1_USERNAME/OPENF1_PASSWORD on the F1 platform service."
                ) from exc
            raise RuntimeError(f"OpenF1 REST request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenF1 REST request failed: {exc.reason}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"OpenF1 endpoint {endpoint} did not return a list")
        return [dict(row) for row in payload if isinstance(row, dict)]

    def resolve_live_or_next_session(
        self,
        *,
        now: str | datetime | None = None,
        year: int | None = None,
    ) -> JsonObject:
        instant = _coerce_datetime(now)
        base_year = year or instant.year
        checked_years = [base_year]
        result = _resolve_session_candidates(self.fetch_json("sessions", {"year": base_year}), instant)
        if result is not None:
            return result

        if year is None:
            next_year = base_year + 1
            checked_years.append(next_year)
            result = _resolve_session_candidates(self.fetch_json("sessions", {"year": next_year}), instant)
            if result is not None:
                return result

        return {
            "status": "unavailable",
            "source": "openf1-rest",
            "resolvedAt": instant.isoformat().replace("+00:00", "Z"),
            "message": f"No live or upcoming OpenF1 sessions found for {', '.join(str(value) for value in checked_years)}.",
            "session": None,
            "nextSession": None,
            "secondsUntilStart": None,
            "secondsUntilEnd": None,
        }

    def _resolve_session(self, request: OpenF1ImportRequest) -> JsonObject:
        if request.session_key is not None:
            sessions = self.fetch_json("sessions", {"session_key": request.session_key})
            if not sessions:
                return {"session_key": request.session_key, "meeting_key": request.meeting_key, "session_name": request.session_name}
            return sessions[0]

        meeting_key = request.meeting_key
        if meeting_key is None:
            if request.year is None:
                raise ValueError("year, meeting_key, or session_key is required")
            meetings = self.fetch_json("meetings", {"year": request.year})
            if not meetings:
                raise ValueError(f"No OpenF1 meetings found for year={request.year}")
            meetings.sort(key=lambda row: str(row.get("date_start") or ""))
            meeting_key = _optional_int(meetings[-1].get("meeting_key"))

        sessions = self.fetch_json("sessions", {"meeting_key": meeting_key})
        if not sessions:
            raise ValueError(f"No OpenF1 sessions found for meeting_key={meeting_key}")
        match = _select_session(sessions, request.session_name)
        if match is None:
            available = ", ".join(str(row.get("session_name")) for row in sessions)
            raise ValueError(f"Session {request.session_name!r} not found. Available: {available}")
        return match


def request_from_payload(payload: Mapping[str, Any]) -> OpenF1ImportRequest:
    raw_topics = payload.get("topics")
    if isinstance(raw_topics, Iterable) and not isinstance(raw_topics, (str, bytes)):
        topics = tuple(str(topic).removeprefix("v1/") for topic in raw_topics if str(topic).strip())
    else:
        topics = DEFAULT_REST_TOPICS
    return OpenF1ImportRequest(
        year=_optional_int(payload.get("year")),
        meeting_key=_optional_int(payload.get("meeting_key")),
        session_key=payload.get("session_key"),
        session_name=str(payload.get("session_name") or "Race"),
        topics=topics,
        include_telemetry=bool(payload.get("include_telemetry", False)),
        limit_per_topic=max(1, min(50_000, _optional_int(payload.get("limit_per_topic")) or 2_000)),
    )


def _select_session(sessions: list[JsonObject], session_name: str) -> JsonObject | None:
    wanted = session_name.strip().lower()
    for session in sessions:
        if str(session.get("session_name") or "").strip().lower() == wanted:
            return session
    if wanted == "race":
        for session in sessions:
            if str(session.get("session_type") or "").strip().lower() == "race":
                return session
    return None


def _source_key(topic: str, payload: Mapping[str, Any]) -> str:
    driver = payload.get("driver_number")
    date = payload.get("date")
    if topic == "sessions":
        return f"session:{payload.get('session_key')}"
    if topic == "drivers":
        return f"driver:{driver}"
    if topic == "laps":
        return f"{driver}:lap:{payload.get('lap_number')}"
    if topic == "stints":
        return f"{driver}:stint:{payload.get('stint_number')}"
    if topic == "pit":
        return f"{driver}:pit:{payload.get('lap_number')}:{date or ''}"
    if topic in {"position", "intervals", "car_data", "location"}:
        return f"{driver}:{topic}:{date or payload.get('date_start') or payload.get('index', '')}"
    if topic == "race_control":
        return f"session:race_control:{date or ''}:{payload.get('message') or payload.get('category') or ''}"
    if topic == "weather":
        return f"session:weather:{date or payload.get('date_start') or ''}"
    if topic == "overtakes":
        overtaking = payload.get("overtaking_driver_number", payload.get("driver_overtaking", driver))
        overtaken = payload.get("overtaken_driver_number", payload.get("driver_overtaken"))
        lap = payload.get("lap_number", "")
        return f"{overtaking}:overtakes:{overtaken or ''}:{lap}:{date or payload.get('date_start') or ''}"
    if topic == "session_result":
        return f"{driver}:session_result"
    return f"{driver or 'session'}:{topic}:{date or ''}"


def _event_sort_key(payload: Mapping[str, Any]) -> str:
    if "session_name" in payload and "driver_number" not in payload:
        return ""
    for key in ("date", "date_start", "date_end"):
        value = payload.get(key)
        if value is not None:
            return str(value)
    if "number_of_laps" in payload and "position" in payload:
        return f"zzzz:session_result:{payload.get('position')}"
    lap = payload.get("lap_number")
    if lap is not None:
        return f"lap:{lap}"
    return ""


def _topic_order(topic: str) -> int:
    order = {
        "sessions": 0,
        "drivers": 1,
        "stints": 2,
        "position": 3,
        "intervals": 4,
        "laps": 5,
        "pit": 6,
        "race_control": 7,
        "weather": 8,
        "overtakes": 9,
        "session_result": 10,
        "car_data": 11,
        "location": 12,
    }
    return order.get(topic, 100)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _resolve_session_candidates(sessions: list[JsonObject], instant: datetime) -> JsonObject | None:
    candidates = sorted(
        (dict(session) for session in sessions if not _is_cancelled(session) and _parse_datetime(session.get("date_start"))),
        key=lambda session: _parse_datetime(session.get("date_start")) or datetime.max.replace(tzinfo=timezone.utc),
    )

    live_sessions: list[JsonObject] = []
    upcoming_sessions: list[JsonObject] = []
    for session in candidates:
        start = _parse_datetime(session.get("date_start"))
        if start is None:
            continue
        end = _parse_datetime(session.get("date_end")) or start
        if start <= instant <= end:
            live_sessions.append(session)
        elif start > instant:
            upcoming_sessions.append(session)

    if live_sessions:
        session = live_sessions[-1]
        end = _parse_datetime(session.get("date_end"))
        return {
            "status": "live",
            "source": "openf1-rest",
            "resolvedAt": instant.isoformat().replace("+00:00", "Z"),
            "message": "OpenF1 reports an ongoing session.",
            "session": session,
            "nextSession": upcoming_sessions[0] if upcoming_sessions else None,
            "secondsUntilStart": 0,
            "secondsUntilEnd": _seconds_between(instant, end),
        }

    if upcoming_sessions:
        next_session = upcoming_sessions[0]
        start = _parse_datetime(next_session.get("date_start"))
        return {
            "status": "upcoming",
            "source": "openf1-rest",
            "resolvedAt": instant.isoformat().replace("+00:00", "Z"),
            "message": "No OpenF1 session is live right now.",
            "session": None,
            "nextSession": next_session,
            "secondsUntilStart": _seconds_between(instant, start),
            "secondsUntilEnd": None,
        }

    return None


def _is_openf1_no_results(detail: str) -> bool:
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        return "no results found" in detail.lower()
    message = payload.get("detail") if isinstance(payload, dict) else None
    return isinstance(message, str) and message.strip().lower() == "no results found."


def _token_from_manager(token_manager: OpenF1TokenManager) -> str:
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(token_manager.token())
    if loop.is_running():
        token, expires_in = token_manager._request_token_sync()
        token_manager._token = token
        token_manager._expires_at_monotonic = time.monotonic() + max(
            1.0,
            float(expires_in) - float(token_manager.config.refresh_margin_seconds),
        )
        return token
    return loop.run_until_complete(token_manager.token())


def _coerce_datetime(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO timestamp: {value}") from exc
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _coerce_datetime(value)
    except ValueError:
        return None


def _is_cancelled(session: Mapping[str, Any]) -> bool:
    value = session.get("is_cancelled")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _seconds_between(start: datetime, end: datetime | None) -> int | None:
    if end is None:
        return None
    return max(0, int((end - start).total_seconds()))
