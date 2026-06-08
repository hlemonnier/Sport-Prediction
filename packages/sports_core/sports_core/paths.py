"""Compatibility wrapper for canonical path helpers."""

try:
    from paths import *  # noqa: F403
except ModuleNotFoundError:  # pragma: no cover - package-root import mode
    from ..paths import *  # noqa: F403
