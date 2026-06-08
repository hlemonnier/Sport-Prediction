"""Shared weather feature transformations."""

from __future__ import annotations

from typing import Mapping


def wet_risk_bucket(summary: Mapping[str, object]) -> str:
    risk = float(summary.get("weather_wet_risk") or 0.0)
    if risk >= 0.66:
        return "high"
    if risk >= 0.33:
        return "medium"
    return "low"


def weather_feature_row(summary: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "weather_available",
        "weather_kind",
        "weather_temperature_2m_mean",
        "weather_precipitation_probability_max",
        "weather_precipitation_sum",
        "weather_rain_sum",
        "weather_wet_risk",
        "weather_wind_speed_10m_mean",
        "weather_wind_gusts_10m_max",
    )
    row = {key: summary.get(key) for key in keys if key in summary}
    row["weather_wet_risk_bucket"] = wet_risk_bucket(summary)
    return row
