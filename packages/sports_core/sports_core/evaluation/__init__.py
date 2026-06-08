"""Shared backtest, calibration, and metric infrastructure."""
"""Compatibility wrapper for canonical evaluation helpers."""

try:
    from evaluation import *  # noqa: F403
except ModuleNotFoundError:  # pragma: no cover - package-root import mode
    from ...evaluation import *  # noqa: F403
