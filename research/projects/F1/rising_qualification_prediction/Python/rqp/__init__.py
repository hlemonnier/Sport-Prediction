"""Rising Qualification Prediction package."""

from .config import PredictionConfig, PredictionResult
from .betting import BettingConfig, build_betting_recommendations
from .circuit_cards import CircuitCard, circuit_card_from_event

__all__ = [
    "BettingConfig",
    "CircuitCard",
    "PredictionConfig",
    "PredictionResult",
    "build_betting_recommendations",
    "circuit_card_from_event",
    "run_prediction",
]


def __getattr__(name: str):
    if name == "run_prediction":
        from .prediction import run_prediction

        return run_prediction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
