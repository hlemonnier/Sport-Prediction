"""Shared data contracts, validation, and local-store infrastructure."""
"""Compatibility wrapper for canonical data contracts."""

try:
    from data import *  # noqa: F403
except ModuleNotFoundError:  # pragma: no cover - package-root import mode
    from ...data import *  # noqa: F403
