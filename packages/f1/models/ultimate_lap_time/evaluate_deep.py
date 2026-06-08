"""Evaluation helper for the optional Ultimate Lap-Time deep model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from packages.f1.models.ultimate_lap_time.evaluate import (
    UltimateLapTimeEvaluationResult,
    evaluate_ultimate_lap_time_predictions,
    write_ultimate_lap_time_evaluation_report,
)
from packages.f1.models.ultimate_lap_time.schemas import UltimateLapTelemetryExample
from packages.f1.models.ultimate_lap_time.train_deep import (
    DeepTrainingResult,
    DeepUltimateLapTimeModel,
    predict_ultimate_lap_time_deep,
)


def evaluate_deep_ultimate_lap_time(
    model_or_result: DeepUltimateLapTimeModel | DeepTrainingResult,
    examples: Sequence[UltimateLapTelemetryExample],
    *,
    report_path: str | Path | None = None,
) -> UltimateLapTimeEvaluationResult | dict[str, Any]:
    """Evaluate a trained deep model, or return a skipped payload."""

    if isinstance(model_or_result, DeepTrainingResult):
        if not model_or_result.is_trained:
            return {
                "status": model_or_result.status,
                "reason": model_or_result.reason,
                "model_name": "ultimate_lap_time_distance_tcn",
            }
        model = model_or_result.model
        assert model is not None
    else:
        model = model_or_result

    predictions = predict_ultimate_lap_time_deep(model, examples)
    result = evaluate_ultimate_lap_time_predictions(
        examples,
        predictions,
        model_name="ultimate_lap_time_distance_tcn",
    )
    if report_path is not None:
        write_ultimate_lap_time_evaluation_report(result, report_path)
    return result


__all__ = ["evaluate_deep_ultimate_lap_time"]
