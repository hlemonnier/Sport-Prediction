"""Ultimate lap-time model package."""

from .datasets import (
    build_distance_normalized_telemetry,
    build_ultimate_lap_dataset,
    build_ultimate_lap_example,
    dataset_summary,
)
from .evaluate import evaluate_ultimate_lap_time_predictions
from .model import UltimateLapTimeConfig, UltimateLapTimeModel, UltimateLapTimeTrainingSummary
from .predict import predict_ultimate_lap_time
from .schemas import (
    DistanceNormalizedTelemetryTensor,
    UltimateLapMetadata,
    UltimateLapSplitKey,
    UltimateLapTargets,
    UltimateLapTelemetryBatch,
    UltimateLapTelemetryExample,
)
from .tabular_quantile import TabularQuantileConfig, fit_tabular_quantile_model
from .train import train_ultimate_lap_time

__all__ = [
    "DistanceNormalizedTelemetryTensor",
    "TabularQuantileConfig",
    "UltimateLapTimeConfig",
    "UltimateLapMetadata",
    "UltimateLapTimeModel",
    "UltimateLapTimeTrainingSummary",
    "UltimateLapSplitKey",
    "UltimateLapTargets",
    "UltimateLapTelemetryBatch",
    "UltimateLapTelemetryExample",
    "build_distance_normalized_telemetry",
    "build_ultimate_lap_dataset",
    "build_ultimate_lap_example",
    "dataset_summary",
    "evaluate_ultimate_lap_time_predictions",
    "fit_tabular_quantile_model",
    "predict_ultimate_lap_time",
    "train_ultimate_lap_time",
]
