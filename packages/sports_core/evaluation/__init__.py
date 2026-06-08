"""Shared evaluation metrics and backtest contracts."""

from .backtest import BacktestResult
from .calibration import expected_calibration_error
from .probability_metrics import brier_score, log_loss
from .ranking_metrics import mean_absolute_rank_error

__all__ = [
    "BacktestResult",
    "brier_score",
    "expected_calibration_error",
    "log_loss",
    "mean_absolute_rank_error",
]
