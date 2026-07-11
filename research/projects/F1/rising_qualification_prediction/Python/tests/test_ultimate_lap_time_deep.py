from __future__ import annotations

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.datasets import build_ultimate_lap_dataset
from packages.f1.models.ultimate_lap_time.deep import (
    DistanceTelemetryTCN,
    fastest_lap_pairwise_rank_loss,
    torch,
    torch_available,
)
from packages.f1.models.ultimate_lap_time.evaluate_deep import evaluate_deep_ultimate_lap_time
from packages.f1.models.ultimate_lap_time.train_deep import (
    DeepTrainingConfig,
    examples_to_deep_numpy,
    train_ultimate_lap_time_deep,
)
from packages.f1.models.ultimate_lap_time.schemas import IDEAL_LAP_TARGET_CONTRACT


def _examples():
    records = []
    telemetry = []
    for event_idx, (event, circuit, split_name) in enumerate(
        (("2026-01-bahrain", "bahrain", "train"), ("2026-02-jeddah", "jeddah", "validation"))
    ):
        for idx, driver in enumerate(["VER", "LEC", "NOR", "PIA"]):
            records.append({
                "season": 2026,
                "event_key": event,
                "circuit_id": circuit,
                "driver_id": driver,
                "team_id": "team",
                "session": "Q",
                "lap_number": idx + 1,
                "split_name": split_name,
                "ideal_lap_time_seconds": 89.0 + event_idx + idx * 0.2,
                "target_contract": IDEAL_LAP_TARGET_CONTRACT,
                "sector1_seconds": 29.0 + idx * 0.05,
                "sector2_seconds": 30.0 + idx * 0.08,
                "sector3_seconds": 30.0 + idx * 0.07,
                "p05_target": 88.8 + event_idx + idx * 0.2,
                "p50_target": 89.0 + event_idx + idx * 0.2,
                "p90_target": 89.3 + event_idx + idx * 0.2,
                "tyre_age": idx,
                "track_temp_c": 37.0,
                "expected_lap_distance_m": 5000.0,
            })
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

    assert telemetry.shape == (8, 2, 12)
    assert static.shape[0] == 8
    assert target.shape == (8, 6)
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
        assert result.model.normalization.fitted_row_count == 4
        assert result.model.validation_group_keys
        assert all("validation_loss" in row for row in result.history)
        evaluation = evaluate_deep_ultimate_lap_time(result, examples)
        assert evaluation.row_count == 8


def test_deep_training_fails_closed_without_grouped_temporal_validation() -> None:
    if not torch_available():
        return
    one_group = [example for example in _examples() if example.metadata.split_key.split_name == "train"]

    result = train_ultimate_lap_time_deep(
        one_group,
        config=DeepTrainingConfig(epochs=1, batch_size=2, hidden_channels=8, head_hidden_dim=8),
    )

    assert result.status == "rejected"
    assert result.model is None
    assert "at least two" in str(result.reason)


def test_pairwise_rank_loss_never_compares_different_event_groups() -> None:
    if not torch_available():
        return
    prediction = torch.tensor([[0.0, 100.0, 0.0], [0.0, 80.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 80.0, 0.0], [0.0, 100.0, 0.0]], dtype=torch.float32)

    cross_event = fastest_lap_pairwise_rank_loss(prediction, target, torch.tensor([0, 1]))
    same_event = fastest_lap_pairwise_rank_loss(prediction, target, torch.tensor([0, 0]))

    assert float(cross_event.item()) == 0.0
    assert float(same_event.item()) > 1.0


def test_tcn_constructor_raises_without_torch() -> None:
    if torch_available():
        return
    try:
        DistanceTelemetryTCN()
    except RuntimeError as exc:
        assert "PyTorch is not installed" in str(exc)
