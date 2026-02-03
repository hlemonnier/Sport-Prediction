"""Prediction orchestration (placeholder implementation)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import PredictionConfig


@dataclass(frozen=True)
class PredictionResult:
    version: str
    rows: list[dict[str, str]]
    notes: list[str]


def run_prediction(config: PredictionConfig) -> PredictionResult:
    _ = config
    return PredictionResult(
        version="0.1.0",
        rows=[],
        notes=[
            "Model not implemented yet. Populate mrp/data.py and mrp/training.py once research is complete."
        ],
    )
