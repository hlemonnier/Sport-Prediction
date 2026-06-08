"""Compatibility wrapper for the canonical Open-Meteo weather provider."""

try:
    from weather.open_meteo import *  # noqa: F403
except ModuleNotFoundError:  # pragma: no cover - package-root import mode
    from ...weather.open_meteo import *  # noqa: F403
