from __future__ import annotations

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.datasets import build_ultimate_lap_dataset
from packages.f1.models.ultimate_lap_time.deep import DistanceTelemetryTCN, torch_available
from packages.f1.models.ultimate_lap_time.evaluate_deep import evaluate_deep_ultimate_lap_time
from packages.f1.models.ultimate_lap_time.train_deep import (
    DeepTrainingConfig,
    examples_to_deep_numpy,
    train_ultimate_lap_time_deep,
)


def _examples():
    records = []
    telemetry = []
    for idx, driver in enumerate(["VER", "LEC", "NOR", "PIA"]):
        records.append(
            {
                "season": 2026,
                "event_key": "bahrain",
                "circuit_id": "bahrain",
                "driver_id": driver,
                "team_id": "team",
                "session": "Q",
                "lap_number": idx + 1,
                "split_name": "train",
                "lap_time_seconds": 89.0 + idx * 0.2,
                "sector1_seconds": 29.0 + idx * 0.05,
                "sector2_seconds": 30.0 + idx * 0.08,
                "sector3_seconds": 30.0 + idx * 0.07,
                "p05_target": 88.8 + idx * 0.2,
                "p50_target": 89.0 + idx * 0.2,
                "p90_target": 89.3 + idx * 0.2,
                "tyre_age": idx,
                "track_temp_c": 37.0,
            }
        )
        distance = np.linspace(0.0, 5000.0, num=10)
        telemetry.append(
            pd.DataFrame(
                {
                    "Distance": distance,
                    "Speed": np.linspace(100.0 + idx, 300.0 + idx, num=10),
                    "Throttle": np.linspace(0.1, 1.0, num=10),
                }
            )
        )
    return build_ultimate_lap_dataset(
        records,
        telemetry=telemetry,
        distance_bins=12,
        channel_names=("Speed", "Throttle"),
    )


def test_deep_numpy_contract_uses_batch_channels_bins_and_targets() -> None:
    examples = _examples()
    telemetry, static, target, static_names = examples_to_deep_numpy(examples)

    assert telemetry.shape == (4, 2, 12)
    assert static.shape[0] == 4
    assert target.shape == (4, 6)
    assert "track_temp_c" in static_names


def test_deep_training_skips_cleanly_without_torch_or_trains_when_available() -> None:
    examples = _examples()
    result = train_ultimate_lap_time_deep(
        examples,
        config=DeepTrainingConfig(epochs=1, batch_size=2, hidden_channels=8, head_hidden_dim=8),
    )

    if not torch_available():
        assert result.status == "skipped"
        assert result.reason == "PyTorch is not installed"
        skipped_eval = evaluate_deep_ultimate_lap_time(result, examples)
        assert skipped_eval["status"] == "skipped"
    else:
        assert result.status == "trained"
        assert result.model is not None
        assert result.history
        evaluation = evaluate_deep_ultimate_lap_time(result, examples)
        assert evaluation.row_count == 4


def test_tcn_constructor_raises_without_torch() -> None:
    if torch_available():
        return
    try:
        DistanceTelemetryTCN()
    except RuntimeError as exc:
        assert "PyTorch is not installed" in str(exc)
