"""Compatibility import path for shared sport weather infrastructure."""

try:
    from weather.open_meteo import (
        DEFAULT_HOURLY_VARIABLES,
        OpenMeteoWeatherProvider,
        WeatherLocation,
        WeatherSnapshot,
        summarize_hourly_weather,
    )
except ModuleNotFoundError:  # pragma: no cover - package-root import mode
    from ..weather.open_meteo import (
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
