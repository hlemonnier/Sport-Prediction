from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from packages.f1.models.ultimate_lap_time.datasets import build_ultimate_lap_dataset
from packages.f1.models.ultimate_lap_time.deep import (
    DistanceTelemetryTCN,
    DistanceTelemetryTCNConfig,
    deep_output_contract_issues,
    fastest_lap_pairwise_rank_loss,
    pinball_loss_tensor,
    torch,
    torch_available,
)
from packages.f1.models.ultimate_lap_time.evaluate_deep import evaluate_deep_ultimate_lap_time
from packages.f1.models.ultimate_lap_time.train_deep import (
    DeepFeatureNormalization,
    DeepTrainingConfig,
    DeepUltimateLapTimeModel,
    examples_to_deep_numpy,
    predict_ultimate_lap_time_deep,
    train_ultimate_lap_time_deep,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
)


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
                "session": "SQ",
                "target_session": "Q",
                "feature_as_of": f"2026-03-{event_idx + 1:02d}T12:00:00Z",
                "target_as_of": f"2026-03-{event_idx + 1:02d}T15:00:00Z",
                "lap_number": idx + 1,
                "split_name": split_name,
                "achievable_session_end_lap_time_seconds": 89.0 + event_idx + idx * 0.2,
                "target_contract": ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
                "sector1_seconds": 29.0 + event_idx + idx * 0.05,
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
    np.testing.assert_allclose(target[:, 0], target[:, 1])
    np.testing.assert_allclose(target[:, 1], target[:, 2])
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


def test_deep_training_rejects_same_session_target_reconstruction() -> None:
    examples = _examples()
    leaked = [
        replace(
            example,
            metadata=replace(
                example.metadata,
                session="Q",
                target_session="Q",
                split_key=replace(example.metadata.split_key, session="Q"),
            ),
        )
        for example in examples
    ]

    result = train_ultimate_lap_time_deep(leaked)

    assert result.status == "rejected"
    assert "must differ" in str(result.reason)


def test_pairwise_rank_loss_never_compares_different_event_groups() -> None:
    if not torch_available():
        return
    prediction = torch.tensor([[0.0, 100.0, 0.0], [0.0, 80.0, 0.0]], dtype=torch.float32)
    # The canonical target vector repeats the realized lap in its first three
    # columns; the first column is the explicit scalar outcome consumed by the
    # rank and pinball losses.
    target = torch.tensor([[80.0, 80.0, 80.0], [100.0, 100.0, 100.0]], dtype=torch.float32)

    cross_event = fastest_lap_pairwise_rank_loss(prediction, target, torch.tensor([0, 1]))
    same_event = fastest_lap_pairwise_rank_loss(prediction, target, torch.tensor([0, 0]))

    assert float(cross_event.item()) == 0.0
    assert float(same_event.item()) > 1.0


def test_all_pinball_heads_use_the_same_realized_lap_outcome() -> None:
    if not torch_available():
        return
    prediction = torch.tensor([[90.0, 91.0, 92.0]], dtype=torch.float32)
    # Only the first column is the realized outcome. Different legacy values
    # in the other columns must not become fake observable quantile labels.
    target = torch.tensor([[91.0, 1.0, 500.0]], dtype=torch.float32)

    loss = pinball_loss_tensor(prediction, target)
    expected = (0.05 * 1.0 + 0.0 + 0.10 * 1.0) / 3.0

    assert float(loss.item()) == pytest.approx(expected)


def test_tcn_constructor_raises_without_torch() -> None:
    if torch_available():
        return
    try:
        DistanceTelemetryTCN()
    except RuntimeError as exc:
        assert "PyTorch is not installed" in str(exc)


def test_deep_training_rejects_unlabeled_inputs_before_any_training() -> None:
    unlabeled_inputs = [example.without_targets() for example in _examples()]

    result = train_ultimate_lap_time_deep(unlabeled_inputs)

    assert result.status == "rejected"
    assert result.model is None
    assert "requires labelled" in str(result.reason)


def test_tcn_outputs_are_positive_ordered_and_sector_sum_coherent() -> None:
    if not torch_available():
        return
    network = DistanceTelemetryTCN(
        DistanceTelemetryTCNConfig(
            input_channels=2,
            distance_bins=12,
            hidden_channels=8,
            head_hidden_dim=8,
            dropout=0.0,
        )
    )
    network.eval()

    with torch.no_grad():
        output = network(torch.randn(5, 2, 12))
    values = output.detach().cpu().numpy()

    assert np.all(values > 0.0)
    assert np.all(values[:, 0] <= values[:, 1])
    assert np.all(values[:, 1] <= values[:, 2])
    np.testing.assert_allclose(values[:, 1], values[:, 3:6].sum(axis=1), atol=1e-5)
    assert deep_output_contract_issues(values) == ()


def test_deep_prediction_accepts_genuinely_unlabeled_inputs() -> None:
    if not torch_available():
        return
    examples = _examples()
    unlabeled_inputs = [example.without_targets() for example in examples]
    telemetry, static, target, static_names = examples_to_deep_numpy(unlabeled_inputs)
    assert np.isnan(target).all()

    architecture = DistanceTelemetryTCNConfig(
        input_channels=telemetry.shape[1],
        distance_bins=telemetry.shape[2],
        static_feature_dim=static.shape[1],
        hidden_channels=8,
        head_hidden_dim=8,
        dropout=0.0,
    )
    model = DeepUltimateLapTimeModel(
        network=DistanceTelemetryTCN(architecture),
        architecture_config=architecture,
        training_config=DeepTrainingConfig(),
        channel_names=examples[0].telemetry.channel_names,
        static_feature_names=static_names,
        normalization=DeepFeatureNormalization(
            channel_names=examples[0].telemetry.channel_names,
            telemetry_mean=tuple(0.0 for _ in examples[0].telemetry.channel_names),
            telemetry_std=tuple(1.0 for _ in examples[0].telemetry.channel_names),
            static_feature_names=static_names,
            static_mean=tuple(0.0 for _ in static_names),
            static_std=tuple(1.0 for _ in static_names),
            fitted_row_count=len(examples),
        ),
        device_used="cpu",
        history=(),
        best_epoch=0,
        training_group_keys=(),
        validation_group_keys=(),
    )

    prediction = predict_ultimate_lap_time_deep(model, unlabeled_inputs)

    assert len(prediction) == len(unlabeled_inputs)
    assert set(prediction["target_contract"]) == {ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT}
    assert set(prediction["target_semantics"]) == {"achievable_session_end_lap"}
    assert deep_output_contract_issues(prediction.iloc[:, :6].to_numpy()) == ()
