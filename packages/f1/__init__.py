"""F1 domain package root."""

from packages.f1.betting import BettingConfig, build_betting_recommendations
from packages.f1.data.schemas import CircuitCard, PredictionConfig, PredictionResult
from packages.f1.data.schemas.circuit import circuit_card_from_event
from packages.f1.orchestration.prediction import run_prediction

__all__ = [
    "BettingConfig",
    "CircuitCard",
    "PredictionConfig",
    "PredictionResult",
    "build_betting_recommendations",
    "circuit_card_from_event",
    "run_prediction",
]
