"""Shared feature helpers used across sport domains."""
"""Compatibility wrapper for canonical feature helpers."""

try:
    from features import *  # noqa: F403
except ModuleNotFoundError:  # pragma: no cover - package-root import mode
    from ...features import *  # noqa: F403
