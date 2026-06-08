"""Match Result Prediction (football) package."""

from .config import PredictionConfig
from .prediction import PredictionResult, run_prediction

__all__ = ["PredictionConfig", "PredictionResult", "run_prediction"]
