"""F1 weather integration helpers built on shared sports weather infra."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from packages.f1.data.schemas.circuit import circuit_card_from_event


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

from packages.sports_core.weather import OpenMeteoWeatherProvider, WeatherLocation  # noqa: E402
from packages.sports_core.weather.locations import F1_CIRCUIT_LOCATIONS  # noqa: E402


def resolve_f1_weather_location(
    event_name: object,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
) -> WeatherLocation | None:
    if latitude is not None and longitude is not None:
        return WeatherLocation(
            name=str(event_name or "configured_f1_location"),
            latitude=float(latitude),
            longitude=float(longitude),
            timezone=timezone or "auto",
        )
    card = circuit_card_from_event(event_name)
    location = F1_CIRCUIT_LOCATIONS.get(card.card_id)
    if location is None:
        return None
    if timezone and timezone != location.timezone:
        return WeatherLocation(
            name=location.name,
            latitude=location.latitude,
            longitude=location.longitude,
            timezone=timezone,
        )
    return location


def fetch_f1_weather_summary(
    *,
    event_name: object,
    start: str | datetime | None,
    end: str | datetime | None,
    cache_root: str | None,
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
    provider_name: str = "open_meteo",
) -> tuple[dict[str, object], list[str]]:
    notes: list[str] = []
    if str(provider_name or "open_meteo").strip().lower() != "open_meteo":
        notes.append(f"Weather provider unsupported: {provider_name}.")
        return {"weather_available": False, "weather_reason": "unsupported_provider"}, notes
    if start is None or end is None:
        notes.append("Weather enabled but weather_start/weather_end are missing.")
        return {"weather_available": False, "weather_reason": "missing_event_window"}, notes

    location = resolve_f1_weather_location(
        event_name,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )
    if location is None:
        notes.append(f"Weather enabled but no circuit coordinates resolved for event={event_name}.")
        return {"weather_available": False, "weather_reason": "missing_location"}, notes

    weather_provider = OpenMeteoWeatherProvider.from_cache_root(cache_root)
    snapshot = weather_provider.fetch_event_weather(location, start=start, end=end)
    summary = snapshot.summary(start=start, end=end)
    summary["weather_event_name"] = str(event_name or "")
    if not bool(summary.get("weather_available")):
        notes.append(
            "Open-Meteo weather fetched but no hourly rows matched the requested event window."
        )
    else:
        notes.append(
            "Open-Meteo weather integrated: "
            f"{summary.get('weather_hour_count')} hourly rows, "
            f"wet_risk={float(summary.get('weather_wet_risk') or 0.0):.3f}."
        )
    return summary, notes


def apply_f1_weather_to_features(
    frame: pd.DataFrame,
    weather_summary: dict[str, object],
) -> pd.DataFrame:
    if frame.empty or not bool(weather_summary.get("weather_available")):
        return frame
    out = frame.copy()
    wet_risk = float(weather_summary.get("weather_wet_risk") or 0.0)
    out["open_meteo_wet_risk"] = wet_risk
    out["open_meteo_precipitation_sum"] = float(weather_summary.get("weather_precipitation_sum") or 0.0)
    out["open_meteo_rain_sum"] = float(weather_summary.get("weather_rain_sum") or 0.0)
    out["open_meteo_wind_gusts_10m_max"] = _optional_float(
        weather_summary.get("weather_wind_gusts_10m_max")
    )

    for column in ("track_weather_uncertainty", "track_weather_uncertainty_prior"):
        existing = (
            pd.to_numeric(out[column], errors="coerce")
            if column in out.columns
            else pd.Series(0.0, index=out.index, dtype=float)
        )
        out[column] = existing.fillna(0.0).clip(lower=0.0, upper=1.0).combine(
            pd.Series(wet_risk, index=out.index, dtype=float),
            max,
        )

    if "race_generation_variance_prior" in out.columns:
        variance = pd.to_numeric(out["race_generation_variance_prior"], errors="coerce").fillna(0.0)
        out["race_generation_variance_prior"] = (
            variance + (0.18 * wet_risk)
        ).clip(lower=0.0, upper=1.0)
    return out


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
