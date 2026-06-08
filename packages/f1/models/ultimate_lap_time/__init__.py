"""Ultimate lap-time model package."""

from .datasets import (
    build_distance_normalized_telemetry,
    build_ultimate_lap_dataset,
    build_ultimate_lap_example,
    dataset_summary,
)
from .deep import DistanceTelemetryTCN, DistanceTelemetryTCNConfig, torch_available
from .evaluate import evaluate_ultimate_lap_time_predictions
from .evaluate_deep import evaluate_deep_ultimate_lap_time
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
from .train_deep import (
    DeepTrainingConfig,
    DeepTrainingResult,
    DeepUltimateLapTimeModel,
    examples_to_deep_numpy,
    predict_ultimate_lap_time_deep,
    train_ultimate_lap_time_deep,
)

__all__ = [
    "DeepTrainingConfig",
    "DeepTrainingResult",
    "DeepUltimateLapTimeModel",
    "DistanceTelemetryTCN",
    "DistanceTelemetryTCNConfig",
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
    "evaluate_deep_ultimate_lap_time",
    "evaluate_ultimate_lap_time_predictions",
    "examples_to_deep_numpy",
    "fit_tabular_quantile_model",
    "predict_ultimate_lap_time",
    "predict_ultimate_lap_time_deep",
    "torch_available",
    "train_ultimate_lap_time",
    "train_ultimate_lap_time_deep",
]
