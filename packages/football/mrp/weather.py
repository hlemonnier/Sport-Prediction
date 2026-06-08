"""Football weather helpers built on shared sports weather infra."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Any


def _ensure_shared_weather_path() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "sports_core"
        if candidate.exists():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return
        if parent.name == "packages":
            candidate = parent / "sports_core"
            if candidate.exists():
                candidate_text = str(candidate)
                if candidate_text not in sys.path:
                    sys.path.insert(0, candidate_text)
                return


_ensure_shared_weather_path()

from sports_weather import OpenMeteoWeatherProvider, WeatherLocation  # noqa: E402


def location_from_fixture(
    fixture: Any,
    *,
    fallback_latitude: float | None = None,
    fallback_longitude: float | None = None,
    fallback_timezone: str | None = None,
) -> WeatherLocation | None:
    latitude = getattr(fixture, "venue_latitude", None)
    longitude = getattr(fixture, "venue_longitude", None)
    timezone = getattr(fixture, "timezone", None) or fallback_timezone or "auto"
    name = getattr(fixture, "venue_name", None) or getattr(fixture, "match_id", None) or "football_fixture"
    if latitude is None or longitude is None:
        latitude = fallback_latitude
        longitude = fallback_longitude
    if latitude is None or longitude is None:
        return None
    return WeatherLocation(
        name=str(name),
        latitude=float(latitude),
        longitude=float(longitude),
        timezone=str(timezone or "auto"),
    )


def fetch_fixture_weather_summary(
    *,
    fixture: Any,
    cache_root: str | None,
    provider_name: str = "open_meteo",
    fallback_latitude: float | None = None,
    fallback_longitude: float | None = None,
    fallback_timezone: str | None = None,
    hours_before: float = 2.0,
    hours_after: float = 3.0,
) -> tuple[dict[str, object], list[str]]:
    notes: list[str] = []
    if str(provider_name or "open_meteo").strip().lower() != "open_meteo":
        notes.append(f"Weather provider unsupported: {provider_name}.")
        return {"weather_available": False, "weather_reason": "unsupported_provider"}, notes

    kickoff = getattr(fixture, "date", None)
    if kickoff is None:
        notes.append(f"Weather enabled but fixture {getattr(fixture, 'match_id', '')} has no kickoff date.")
        return {"weather_available": False, "weather_reason": "missing_kickoff"}, notes
    if not isinstance(kickoff, datetime):
        notes.append(f"Weather enabled but fixture {getattr(fixture, 'match_id', '')} kickoff is not datetime.")
        return {"weather_available": False, "weather_reason": "invalid_kickoff"}, notes

    location = location_from_fixture(
        fixture,
        fallback_latitude=fallback_latitude,
        fallback_longitude=fallback_longitude,
        fallback_timezone=fallback_timezone,
    )
    if location is None:
        notes.append(
            f"Weather enabled but fixture {getattr(fixture, 'match_id', '')} has no venue coordinates."
        )
        return {"weather_available": False, "weather_reason": "missing_location"}, notes

    start = kickoff - timedelta(hours=float(hours_before))
    end = kickoff + timedelta(hours=float(hours_after))
    provider = OpenMeteoWeatherProvider.from_cache_root(cache_root)
    snapshot = provider.fetch_event_weather(location, start=start, end=end)
    summary = snapshot.summary(start=start, end=end)
    summary["weather_match_id"] = str(getattr(fixture, "match_id", ""))
    if bool(summary.get("weather_available")):
        notes.append(
            "Open-Meteo fixture weather fetched: "
            f"{getattr(fixture, 'match_id', '')}, "
            f"wet_risk={float(summary.get('weather_wet_risk') or 0.0):.3f}."
        )
    else:
        notes.append(
            f"Open-Meteo weather fetched but no rows matched fixture {getattr(fixture, 'match_id', '')}."
        )
    return summary, notes
