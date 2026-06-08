"""Compatibility wrapper for canonical sport weather locations."""

try:
    from weather.locations import *  # noqa: F403
except ModuleNotFoundError:  # pragma: no cover - package-root import mode
    from ..weather.locations import *  # noqa: F403
