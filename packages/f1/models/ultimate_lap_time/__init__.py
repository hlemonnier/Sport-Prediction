"""Ultimate lap-time model package."""

from .datasets import (
    build_distance_normalized_telemetry,
    build_ultimate_lap_dataset,
    build_ultimate_lap_example,
    dataset_summary,
)
from .deep import DistanceTelemetryTCN, DistanceTelemetryTCNConfig, torch_available
from .evaluate import (
    DETERMINISTIC_BASELINE_MODEL_NAME,
    evaluate_ultimate_lap_time_baseline_backtest,
    evaluate_ultimate_lap_time_predictions,
    write_ultimate_lap_time_baseline_backtest_report,
)
from .evaluate_deep import evaluate_deep_ultimate_lap_time
from .model import (
    UltimateLapTimeConfig,
    UltimateLapTimeModel,
    UltimateLapTimeTrainingSummary,
    aggregate_ideal_lap_holdout_targets,
)
from .predict import predict_ultimate_lap_time
from .schemas import (
    IDEAL_LAP_TARGET_CONTRACT,
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
    DeepFeatureNormalization,
    DeepTrainingConfig,
    DeepTrainingResult,
    DeepUltimateLapTimeModel,
    examples_to_deep_numpy,
    predict_ultimate_lap_time_deep,
    train_ultimate_lap_time_deep,
)

__all__ = [
    "DeepFeatureNormalization",
    "DeepTrainingConfig",
    "DeepTrainingResult",
    "DeepUltimateLapTimeModel",
    "DistanceTelemetryTCN",
    "DistanceTelemetryTCNConfig",
    "DistanceNormalizedTelemetryTensor",
    "DETERMINISTIC_BASELINE_MODEL_NAME",
    "IDEAL_LAP_TARGET_CONTRACT",
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
    "aggregate_ideal_lap_holdout_targets",
    "build_ultimate_lap_dataset",
    "build_ultimate_lap_example",
    "dataset_summary",
    "evaluate_deep_ultimate_lap_time",
    "evaluate_ultimate_lap_time_baseline_backtest",
    "evaluate_ultimate_lap_time_predictions",
    "examples_to_deep_numpy",
    "fit_tabular_quantile_model",
    "predict_ultimate_lap_time",
    "predict_ultimate_lap_time_deep",
    "torch_available",
    "train_ultimate_lap_time",
    "train_ultimate_lap_time_deep",
    "write_ultimate_lap_time_baseline_backtest_report",
]
