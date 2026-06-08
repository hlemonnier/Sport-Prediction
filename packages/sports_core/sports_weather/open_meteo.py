"""Compatibility wrapper for the canonical sports_core weather provider."""

try:
    from weather.open_meteo import *  # noqa: F403
except ModuleNotFoundError:  # pragma: no cover - package-root import mode
    from ..weather.open_meteo import *  # noqa: F403
