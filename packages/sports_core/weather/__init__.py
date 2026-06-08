"""Shared weather provider contracts and implementations."""

from .open_meteo import (
    DEFAULT_HOURLY_VARIABLES,
    OpenMeteoWeatherProvider,
    WeatherLocation,
    WeatherSnapshot,
    summarize_hourly_weather,
)

__all__ = [
    "DEFAULT_HOURLY_VARIABLES",
    "OpenMeteoWeatherProvider",
    "WeatherLocation",
    "WeatherSnapshot",
    "summarize_hourly_weather",
]
