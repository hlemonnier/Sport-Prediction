"""Compatibility package for shared weather imports."""

try:
    from weather import (
        DEFAULT_HOURLY_VARIABLES,
        OpenMeteoWeatherProvider,
        WeatherLocation,
        WeatherSnapshot,
        summarize_hourly_weather,
    )
except ModuleNotFoundError:  # pragma: no cover - package-root import mode
    from ...weather import (
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
