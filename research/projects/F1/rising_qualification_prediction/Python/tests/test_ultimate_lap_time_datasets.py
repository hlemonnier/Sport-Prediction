from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packages.f1.models.ultimate_lap_time.datasets import (
    build_distance_normalized_telemetry,
    build_ultimate_lap_dataset,
    build_ultimate_lap_example,
    build_ultimate_lap_inference_input,
    dataset_summary,
    leakage_issues_for_examples,
    validate_ultimate_lap_examples,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    DistanceNormalizedTelemetryTensor,
    IDEAL_LAP_TARGET_CONTRACT,
    THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT,
    UltimateLapTargets,
    UltimateLapTelemetryBatch,
    UltimateLapTelemetryInput,
)


def _lap_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "season": 2026,
        "event_key": "2026-bahrain",
        "circuit_id": "bahrain",
        "driver_id": "VER",
        "team_id": "red_bull",
        "session": "Q",
        "lap_number": 7,
        "split_name": "train",
        "ideal_lap_time_seconds": 89.8,
        "target_contract": IDEAL_LAP_TARGET_CONTRACT,
        "sector1_seconds": 29.8,
        "sector2_seconds": 30.0,
        "sector3_seconds": 30.0,
        "p05_target": 89.6,
        "p50_target": 89.8,
        "p90_target": 90.1,
        "compound": "SOFT",
        "tyre_age": 2,
        "track_temp_c": 38.0,
        "expected_lap_distance_m": 5400.0,
    }
    record.update(overrides)
    return record


def _telemetry_frame(samples: int = 12) -> pd.DataFrame:
    distance = np.linspace(0.0, 5400.0, num=samples)
    return pd.DataFrame(
        {
            "Distance": distance,
            "Speed": np.linspace(80.0, 310.0, num=samples),
            "Throttle": np.linspace(0.2, 1.0, num=samples),
            "Brake": np.r_[np.ones(2), np.zeros(samples - 2)],
        }
    )


def test_distance_normalized_telemetry_resamples_channels_x_bins() -> None:
    tensor = build_distance_normalized_telemetry(
        _telemetry_frame(),
        distance_bins=16,
        channel_names=("Speed", "Throttle", "Brake"),
        expected_lap_distance=5400.0,
    )

    assert isinstance(tensor, DistanceNormalizedTelemetryTensor)
    assert tensor.shape == (3, 16)
    assert tensor.channel_names == ("Speed", "Throttle", "Brake")
    assert np.isfinite(tensor.values).all()
    assert tensor.values[0, -1] == pytest.approx(310.0)
    assert tensor.distance_coverage == pytest.approx(1.0)


def test_incomplete_physical_lap_is_rejected_instead_of_stretched() -> None:
    incomplete = _telemetry_frame()
    incomplete["Distance"] *= 0.70

    with pytest.raises(ValueError, match="incomplete lap distance coverage"):
        build_distance_normalized_telemetry(
            incomplete,
            distance_bins=16,
            channel_names=("Speed", "Throttle", "Brake"),
            expected_lap_distance=5400.0,
        )


def test_time_indexed_telemetry_is_rejected_without_distance_basis() -> None:
    time_only = pd.DataFrame(
        {
            "Time": pd.to_timedelta(np.arange(6), unit="s"),
            "Speed": np.linspace(90.0, 250.0, num=6),
            "Throttle": np.linspace(0.0, 1.0, num=6),
        }
    )

    with pytest.raises(ValueError, match="time-indexed"):
        build_distance_normalized_telemetry(time_only, distance_bins=8)


def test_build_example_batch_summary_and_split_leakage_checks() -> None:
    examples = build_ultimate_lap_dataset(
        pd.DataFrame(
            [
                _lap_record(driver_id="VER", lap_number=7, split_name="train"),
                _lap_record(driver_id="PER", lap_number=8, split_name="train", ideal_lap_time_seconds=90.2),
            ]
        ),
        telemetry=[_telemetry_frame(), _telemetry_frame()],
        distance_bins=20,
        channel_names=("Speed", "Throttle", "Brake"),
    )

    batch = validate_ultimate_lap_examples(examples)
    summary = dataset_summary(examples)

    assert isinstance(batch, UltimateLapTelemetryBatch)
    assert batch.telemetry.shape == (2, 3, 20)
    assert batch.target_matrix().shape == (2, 6)
    assert summary["row_count"] == 2
    assert summary["by_circuit"] == {"bahrain": 2}
    assert summary["target_availability"]["lap_time_seconds"] == 2
    assert summary["target_diagnostics"]["degenerate_quantile_target_rows"] == 0
    assert not summary["target_diagnostics"]["promotion_grade_validation_passed"]
    assert leakage_issues_for_examples(examples) == ()


def test_dataset_builder_handles_empty_records() -> None:
    assert build_ultimate_lap_dataset([]) == []
    empty_batch = validate_ultimate_lap_examples([])
    assert empty_batch.batch_size == 0


def test_split_fields_reject_target_leakage() -> None:
    with pytest.raises(ValueError, match="leakage-prone"):
        build_ultimate_lap_example(
            _lap_record(),
            _telemetry_frame(),
            distance_bins=12,
            split_fields=("event_key", "lap_time_seconds"),
        )


def test_ideal_target_contract_cannot_relabel_an_observed_raw_lap() -> None:
    mislabeled = _lap_record()
    mislabeled.pop("ideal_lap_time_seconds")
    mislabeled["lap_time_seconds"] = 89.8

    with pytest.raises(ValueError, match="requires an explicit ideal_lap_time_seconds"):
        build_ultimate_lap_example(
            mislabeled,
            _telemetry_frame(),
            distance_bins=12,
        )


def test_unlabeled_inference_input_does_not_invent_targets() -> None:
    unlabeled_record = _lap_record()
    for key in (
        "ideal_lap_time_seconds",
        "target_contract",
        "sector1_seconds",
        "sector2_seconds",
        "sector3_seconds",
        "p05_target",
        "p50_target",
        "p90_target",
    ):
        unlabeled_record.pop(key)

    model_input = build_ultimate_lap_inference_input(
        unlabeled_record,
        _telemetry_frame(),
        distance_bins=12,
        channel_names=("Speed", "Throttle", "Brake"),
    )

    assert isinstance(model_input, UltimateLapTelemetryInput)
    flat = model_input.as_flat_record()
    assert "lap_time_seconds" not in flat
    assert "ideal_lap_time_seconds" not in flat
    assert "target_contract" not in flat


def test_target_semantics_and_degenerate_quantiles_are_explicit() -> None:
    implicit_interval = UltimateLapTargets(
        lap_time_seconds=90.0,
        target_contract=ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    )
    valid_interval = UltimateLapTargets(
        lap_time_seconds=90.0,
        p05_target=89.5,
        p50_target=90.0,
        p90_target=90.8,
        target_contract=ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    )
    theoretical_floor = UltimateLapTargets(
        lap_time_seconds=89.5,
        p05_target=89.0,
        p50_target=89.5,
        p90_target=90.0,
        target_contract=THEORETICAL_SECTOR_FLOOR_TARGET_CONTRACT,
    )

    assert implicit_interval.quantile_targets_are_degenerate
    assert not implicit_interval.quantile_targets_were_explicit
    assert implicit_interval.quantile_target_semantics == "shared_realized_scalar"
    assert implicit_interval.promotion_grade_quantiles_valid
    assert "quantile_targets_are_degenerate" not in implicit_interval.promotion_blockers
    assert valid_interval.quantile_target_semantics == "legacy_per_row_quantiles_ignored"
    assert valid_interval.promotion_grade_quantiles_valid
    np.testing.assert_allclose(implicit_interval.target_vector()[:3], [90.0, 90.0, 90.0])
    assert theoretical_floor.target_semantics == "theoretical_sector_floor"
    assert not theoretical_floor.promotion_grade_quantiles_valid

    with pytest.raises(ValueError, match="multiple incompatible estimands"):
        UltimateLapTargets.from_mapping(
            {
                "theoretical_sector_floor_seconds": 89.5,
                "achievable_session_end_lap_time_seconds": 90.0,
            }
        )
