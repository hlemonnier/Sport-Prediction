"""FastF1-backed session schedule resolver.

FastF1 gives us the official event schedule without requiring OpenF1
credentials. This module resolves the current or next session from that
schedule and returns the same broad shape as the OpenF1 session-status
endpoint so the web app can stay source-agnostic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .schemas import JsonObject

ScheduleProvider = Callable[[int], Iterable[Mapping[str, Any]]]

SESSION_COLUMNS = tuple((f"Session{index}", f"Session{index}DateUtc") for index in range(1, 6))


class FastF1ScheduleClient:
    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        schedule_provider: ScheduleProvider | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._schedule_provider = schedule_provider

    def resolve_live_or_next_session(
        self,
        *,
        now: str | datetime | None = None,
        year: int | None = None,
    ) -> JsonObject:
        instant = _coerce_datetime(now)
        base_year = year or instant.year
        checked_years = [base_year]

        result = _resolve_session_candidates(self._sessions_for_year(base_year), instant)
        if result is not None:
            return result

        if year is None:
            next_year = base_year + 1
            checked_years.append(next_year)
            result = _resolve_session_candidates(self._sessions_for_year(next_year), instant)
            if result is not None:
                return result

        return {
            "status": "unavailable",
            "source": "fastf1-schedule",
            "resolvedAt": _iso_z(instant),
            "message": f"No live or upcoming FastF1 sessions found for {', '.join(str(value) for value in checked_years)}.",
            "session": None,
            "nextSession": None,
            "secondsUntilStart": None,
            "secondsUntilEnd": None,
        }

    def season_schedule(self, *, year: int) -> JsonObject:
        rows = self._schedule_rows(year)
        rounds: list[JsonObject] = []
        session_count = 0
        for row in rows:
            sessions = _sessions_from_schedule_row(row, year=year)
            session_count += len(sessions)
            round_number = _optional_int(row.get("RoundNumber"))
            event_name = _clean_text(row.get("EventName"))
            official_event_name = _clean_text(row.get("OfficialEventName"))
            event_date = _date_iso(row.get("EventDate"))
            event_format = _clean_text(row.get("EventFormat"))
            rounds.append(
                {
                    "scheduleKey": _schedule_event_key(
                        year=year,
                        round_number=round_number,
                        event_name=event_name,
                        official_event_name=official_event_name,
                        event_date=event_date,
                        event_format=event_format,
                    ),
                    "roundNumber": round_number,
                    "eventName": event_name,
                    "officialEventName": official_event_name,
                    "eventDate": event_date,
                    "country": _clean_text(row.get("Country")),
                    "location": _clean_text(row.get("Location")),
                    "eventFormat": event_format,
                    "f1ApiSupport": row.get("F1ApiSupport"),
                    "sessions": sessions,
                }
            )
        rounds.sort(key=lambda item: (item.get("roundNumber") is None, item.get("roundNumber") or 0))
        return {
            "year": year,
            "source": "fastf1-schedule",
            "roundCount": len(rounds),
            "sessionCount": session_count,
            "rounds": rounds,
        }

    def _sessions_for_year(self, year: int) -> list[JsonObject]:
        rows = self._schedule_rows(year)
        sessions: list[JsonObject] = []
        for row in rows:
            sessions.extend(_sessions_from_schedule_row(row, year=year))
        sessions.sort(key=lambda session: _parse_datetime(session.get("date_start")) or datetime.max.replace(tzinfo=timezone.utc))
        return sessions

    def _schedule_rows(self, year: int) -> list[JsonObject]:
        if self._schedule_provider is not None:
            return _records_from_any(self._schedule_provider(year))

        try:
            import fastf1
        except ImportError as exc:
            raise RuntimeError("Install fastf1 to resolve the F1 schedule without OpenF1") from exc

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache(str(self.cache_dir))

        try:
            schedule = fastf1.get_event_schedule(year, include_testing=True)
        except TypeError:
            schedule = fastf1.get_event_schedule(year)
        return _records_from_any(schedule)


def _sessions_from_schedule_row(row: Mapping[str, Any], *, year: int) -> list[JsonObject]:
    round_number = _optional_int(row.get("RoundNumber"))
    event_name = _clean_text(row.get("EventName")) or _clean_text(row.get("OfficialEventName")) or f"Round {round_number or '?'}"
    official_event_name = _clean_text(row.get("OfficialEventName"))
    location = _clean_text(row.get("Location"))
    country = _clean_text(row.get("Country"))
    event_format = _clean_text(row.get("EventFormat"))
    f1_api_support = row.get("F1ApiSupport")
    event_date = _date_iso(row.get("EventDate"))
    event_slug = _schedule_event_slug(
        event_name=event_name,
        official_event_name=official_event_name,
        event_date=event_date,
        event_format=event_format,
        round_number=round_number,
    )

    sessions: list[JsonObject] = []
    for index, (name_column, date_column) in enumerate(SESSION_COLUMNS, start=1):
        session_name = _clean_text(row.get(name_column))
        start = _parse_datetime(row.get(date_column))
        if not session_name or start is None:
            continue
        alias = _fastf1_session_alias(session_name)
        end = start + _estimated_duration(session_name)
        sessions.append(
            {
                "session_key": f"fastf1:{year}:{event_slug}:{_slug(alias)}",
                "meeting_key": round_number,
                "round_number": round_number,
                "session_name": session_name,
                "session_type": _session_type(session_name),
                "date_start": _iso_z(start),
                "date_end": _iso_z(end),
                "location": location,
                "country_name": country,
                "circuit_short_name": location,
                "year": year,
                "is_cancelled": False,
                "fastf1_event_name": event_name,
                "fastf1_session_name": alias,
                "official_event_name": official_event_name,
                "schedule_event_key": _schedule_event_key(
                    year=year,
                    round_number=round_number,
                    event_name=event_name,
                    official_event_name=official_event_name,
                    event_date=event_date,
                    event_format=event_format,
                ),
                "event_format": event_format,
                "f1_api_support": bool(f1_api_support) if isinstance(f1_api_support, bool) else f1_api_support,
                "schedule_index": index,
            }
        )
    return sessions


def _resolve_session_candidates(sessions: list[JsonObject], instant: datetime) -> JsonObject | None:
    live_sessions: list[JsonObject] = []
    upcoming_sessions: list[JsonObject] = []
    for session in sessions:
        start = _parse_datetime(session.get("date_start"))
        end = _parse_datetime(session.get("date_end"))
        if start is None:
            continue
        if start <= instant <= (end or start):
            live_sessions.append(session)
        elif start > instant:
            upcoming_sessions.append(session)

    if live_sessions:
        session = live_sessions[-1]
        end = _parse_datetime(session.get("date_end"))
        return {
            "status": "live",
            "source": "fastf1-schedule",
            "resolvedAt": _iso_z(instant),
            "message": "FastF1 schedule reports an ongoing session.",
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
            "source": "fastf1-schedule",
            "resolvedAt": _iso_z(instant),
            "message": "No FastF1 session is live right now.",
            "session": None,
            "nextSession": next_session,
            "secondsUntilStart": _seconds_between(instant, start),
            "secondsUntilEnd": None,
        }

    return None


def _records_from_any(value: Iterable[Mapping[str, Any]] | Any) -> list[JsonObject]:
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict("records")
            return [dict(record) for record in records if isinstance(record, Mapping)]
        except TypeError:
            pass
    return [dict(record) for record in value if isinstance(record, Mapping)]


def _estimated_duration(session_name: str) -> timedelta:
    normalized = session_name.strip().lower()
    if "race" in normalized:
        return timedelta(hours=3)
    if "testing" in normalized:
        return timedelta(hours=9)
    return timedelta(hours=1)


def _session_type(session_name: str) -> str:
    normalized = session_name.strip().lower()
    if "practice" in normalized:
        return "Practice"
    if "qualifying" in normalized or "shootout" in normalized:
        return "Qualifying"
    if "sprint" in normalized:
        return "Sprint"
    if "race" in normalized:
        return "Race"
    if "testing" in normalized:
        return "Testing"
    return session_name


def _fastf1_session_alias(session_name: str) -> str:
    normalized = session_name.strip().lower()
    aliases = {
        "practice 1": "FP1",
        "practice 2": "FP2",
        "practice 3": "FP3",
        "qualifying": "Q",
        "sprint": "S",
        "sprint qualifying": "SQ",
        "sprint shootout": "SS",
        "race": "R",
    }
    return aliases.get(normalized, session_name)


def _schedule_event_key(
    *,
    year: int,
    round_number: int | None,
    event_name: str | None,
    official_event_name: str | None,
    event_date: str | None,
    event_format: str | None,
) -> str:
    event_slug = _schedule_event_slug(
        event_name=event_name,
        official_event_name=official_event_name,
        event_date=event_date,
        event_format=event_format,
        round_number=round_number,
    )
    round_slug = "testing" if _is_testing_event(round_number, event_format) else str(round_number or "unknown")
    return f"fastf1:{year}:round:{round_slug}:{event_slug}"


def _schedule_event_slug(
    *,
    event_name: str | None,
    official_event_name: str | None,
    event_date: str | None,
    event_format: str | None,
    round_number: int | None,
) -> str:
    base = _slug(event_name or official_event_name or f"round-{round_number or 'unknown'}")
    if _is_testing_event(round_number, event_format):
        suffix = _slug(event_date or official_event_name or "testing")
        return f"{base}-{suffix}" if suffix and suffix != base else base
    return base


def _is_testing_event(round_number: int | None, event_format: str | None) -> bool:
    return round_number == 0 or str(event_format or "").strip().lower() == "testing"


def _coerce_datetime(value: str | datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = _parse_datetime(value)
    if parsed is None:
        raise ValueError(f"Invalid ISO timestamp: {value}")
    return parsed


def _parse_datetime(value: Any) -> datetime | None:
    if _is_missing(value):
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        normalized = normalized.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _clean_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    return text


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"nan", "nat", "none"}


def _optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _seconds_between(start: datetime, end: datetime | None) -> int | None:
    if end is None:
        return None
    return max(0, int((end - start).total_seconds()))


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _date_iso(value: Any) -> str | None:
    parsed = _parse_datetime(value)
    if parsed is not None:
        return parsed.date().isoformat()
    text = _clean_text(value)
    return text


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "session"
