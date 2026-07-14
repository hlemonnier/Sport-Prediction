from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from packages.f1.data.providers.telemetry_cache import (
    NORMALIZED_TELEMETRY_CHANNELS,
    sha256_file,
)
from packages.f1.data.providers.telemetry_supervised import canonical_sha256
from packages.f1.models.ultimate_lap_time.deep import (
    DistanceTelemetryResidualTCN,
    DistanceTelemetryResidualTCNConfig,
    torch,
    torch_available,
    trainable_parameter_count,
)
from run_prequal_telemetry_tcn_research import (
    ANCHOR_LAP_TIME_TOLERANCE_SECONDS,
    PROFILE_DESIGN_STAGE_ORIGINAL_CONTROL,
    PROFILE_DESIGN_STAGE_POSTDEVELOPMENT,
    STATIC_FEATURE_NAMES,
    TCNBagDataset,
    TCNResearchConfig,
    TELEMETRY_INPUT_ZERO_ABLATION,
    _event_relative_static_features,
    _event_relative_tensor_features,
    _load_bag_tensor,
    _training_tensors,
    normalize_profile_design_provenance,
    run_expanding_tcn_benchmark,
)
from run_prequal_telemetry_residual_research import TelemetryResidualResearchError


def _tensor_evidence(
    tmp_path,
    *,
    filename: str,
    fill_value: float,
    lap_number: int,
    push_lap_rank: int | None,
    rehearsal_lap_time_seconds: float,
) -> dict[str, object]:
    path = tmp_path / filename
    np.savez_compressed(
        path,
        values=np.full(
            (len(NORMALIZED_TELEMETRY_CHANNELS), 8),
            fill_value,
            dtype=np.float32,
        ),
        channel_names=np.asarray(NORMALIZED_TELEMETRY_CHANNELS),
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "lap_number": lap_number,
        "push_lap_rank": push_lap_rank,
        "rehearsal_lap_time_seconds": rehearsal_lap_time_seconds,
    }


def _dataset(*, event_count: int = 6, drivers_per_event: int = 5) -> TCNBagDataset:
    bins = 24
    phase = np.linspace(0.0, 2.0 * np.pi, num=bins, endpoint=False)
    rows: list[dict[str, object]] = []
    telemetry: list[np.ndarray] = []
    static: list[np.ndarray] = []
    for event in range(1, event_count + 1):
        event_key = 202600 + event
        source = "Sprint Qualifying" if event % 2 == 0 else "Practice 3"
        source_shift = -0.2 if source == "Sprint Qualifying" else -0.7
        for driver in range(drivers_per_event):
            skill = (driver - (drivers_per_event - 1) / 2.0) / drivers_per_event
            reference = 80.0 + event * 1.8 + driver * 0.08
            correction = -0.35 * skill + 0.015 * np.sin(event)
            actual = reference + source_shift + correction
            rows.append(
                {
                    "event_key": event_key,
                    "round": event,
                    "event_name": f"Test Grand Prix {event}",
                    "driver_id": f"DRV{driver}",
                    "rehearsal_source": source,
                    "bag_sha256": f"{event_key * 100 + driver:064x}",
                    "tensor_count_raw": 2,
                    "rehearsal_reference_seconds": reference,
                    "actual_lap_time_seconds": actual,
                    "target_residual_seconds": actual - reference,
                    "lap_time_observed": True,
                }
            )
            channels = np.vstack(
                [
                    skill + 0.2 * np.sin(phase),
                    skill * 0.5 + 0.1 * np.cos(phase),
                    skill + np.sin(phase),
                    skill * 0.25 + np.cos(phase),
                    skill * np.ones_like(phase),
                    -skill * np.ones_like(phase),
                ]
            ).astype(np.float32)
            telemetry.append(channels)
            static.append(np.asarray([skill, 2.0 * driver / 4.0 - 1.0], dtype=np.float32))
    frame = pd.DataFrame(rows)
    values = np.stack(telemetry, axis=0)
    static_values = np.stack(static, axis=0)
    return TCNBagDataset(
        frame=frame,
        telemetry=values,
        static_features=static_values,
        channel_names=tuple(NORMALIZED_TELEMETRY_CHANNELS),
        distance_bins=bins,
        validated_tensor_count=len(frame) * 2,
        feature_set_sha256=canonical_sha256(
            {"telemetry_shape": list(values.shape), "static_shape": list(static_values.shape)}
        ),
    )


def _profile_design_provenance(*, informed: bool) -> dict[str, object]:
    return {
        "design_stage": (
            PROFILE_DESIGN_STAGE_POSTDEVELOPMENT
            if informed
            else PROFILE_DESIGN_STAGE_ORIGINAL_CONTROL
        ),
        "prior_outer_results_informed_profile_design": informed,
        "same_outer_evaluation_targets_seen_before_profile_freeze": informed,
        "hyperparameters_tuned_on_outer_targets": informed,
        "durable_matrix_results_used_to_select_profile": False,
        "promotion_eligible_from_profile_design": False,
    }


def test_profile_design_provenance_distinguishes_postdevelopment_tuning() -> None:
    control = normalize_profile_design_provenance(
        _profile_design_provenance(informed=False)
    )
    postdevelopment = normalize_profile_design_provenance(
        _profile_design_provenance(informed=True)
    )

    assert control["hyperparameters_tuned_on_outer_targets"] is False
    assert postdevelopment["hyperparameters_tuned_on_outer_targets"] is True
    assert postdevelopment[
        "same_outer_evaluation_targets_seen_before_profile_freeze"
    ] is True
    assert postdevelopment["promotion_eligible_from_profile_design"] is False


def test_profile_design_provenance_rejects_outer_target_contradiction() -> None:
    provenance = _profile_design_provenance(informed=True)
    provenance["hyperparameters_tuned_on_outer_targets"] = False

    with pytest.raises(
        TelemetryResidualResearchError,
        match="contradicts its outer-target history",
    ):
        normalize_profile_design_provenance(provenance)


def test_bag_adapter_selects_verified_fastest_anchor_not_path_or_manifest_order(
    tmp_path,
) -> None:
    alphabetically_first_slow = _tensor_evidence(
        tmp_path,
        filename="a_slow.npz",
        fill_value=1.0,
        lap_number=4,
        push_lap_rank=3,
        rehearsal_lap_time_seconds=91.2,
    )
    manifest_first_middle = _tensor_evidence(
        tmp_path,
        filename="m_middle.npz",
        fill_value=5.0,
        lap_number=7,
        push_lap_rank=2,
        rehearsal_lap_time_seconds=90.4,
    )
    alphabetically_last_fast = _tensor_evidence(
        tmp_path,
        filename="z_fast.npz",
        fill_value=9.0,
        lap_number=11,
        push_lap_rank=1,
        rehearsal_lap_time_seconds=89.8,
    )
    bag = {
        "feature": {
            "tensors": [
                manifest_first_middle,
                alphabetically_first_slow,
                alphabetically_last_fast,
            ]
        }
    }

    selected, audit = _load_bag_tensor(bag, root=tmp_path)

    assert selected == pytest.approx(np.full_like(selected, 9.0), abs=0.0)
    assert audit["selection_method"] == (
        "unique_push_lap_rank_1_verified_against_minimum_time"
    )
    assert audit["selected_tensor_lap_number"] == 11
    assert audit["selected_tensor_push_lap_rank"] == 1
    assert audit["selected_tensor_rehearsal_lap_time_seconds"] == pytest.approx(89.8)
    assert audit["selected_tensor_sha256"] == alphabetically_last_fast["sha256"]
    assert audit["correlated_tensor_count"] == 3
    assert {row["sha256"] for row in audit["correlated_tensors"]} == {
        alphabetically_first_slow["sha256"],
        manifest_first_middle["sha256"],
        alphabetically_last_fast["sha256"],
    }
    assert audit["anchor_time_tolerance_seconds"] == (
        ANCHOR_LAP_TIME_TOLERANCE_SECONDS
    )


def test_bag_adapter_fails_closed_when_rank_one_is_not_fastest(tmp_path) -> None:
    fastest_but_rank_two = _tensor_evidence(
        tmp_path,
        filename="fast_rank_two.npz",
        fill_value=2.0,
        lap_number=3,
        push_lap_rank=2,
        rehearsal_lap_time_seconds=88.0,
    )
    slower_rank_one = _tensor_evidence(
        tmp_path,
        filename="slow_rank_one.npz",
        fill_value=8.0,
        lap_number=9,
        push_lap_rank=1,
        rehearsal_lap_time_seconds=(
            88.0 + ANCHOR_LAP_TIME_TOLERANCE_SECONDS + 0.01
        ),
    )

    with pytest.raises(
        TelemetryResidualResearchError,
        match="rank=1 tensor time is inconsistent with the fastest rehearsal reference",
    ):
        _load_bag_tensor(
            {"feature": {"tensors": [slower_rank_one, fastest_but_rank_two]}},
            root=tmp_path,
        )


def test_bag_adapter_fails_closed_on_ambiguous_rank_one(tmp_path) -> None:
    first = _tensor_evidence(
        tmp_path,
        filename="first_rank_one.npz",
        fill_value=3.0,
        lap_number=2,
        push_lap_rank=1,
        rehearsal_lap_time_seconds=87.0,
    )
    second = _tensor_evidence(
        tmp_path,
        filename="second_rank_one.npz",
        fill_value=4.0,
        lap_number=5,
        push_lap_rank=1,
        rehearsal_lap_time_seconds=87.2,
    )

    with pytest.raises(
        TelemetryResidualResearchError,
        match="ambiguous duplicate push_lap_rank=1",
    ):
        _load_bag_tensor(
            {"feature": {"tensors": [first, second]}},
            root=tmp_path,
        )


@pytest.mark.skipif(not torch_available(), reason="optional torch dependency missing")
def test_residual_network_is_a_small_true_dilated_tcn() -> None:
    config = DistanceTelemetryResidualTCNConfig(
        input_channels=6,
        distance_bins=200,
        static_feature_dim=2,
        hidden_channels=4,
        dilations=(1, 2),
        head_hidden_dim=4,
        max_abs_correction_seconds=1.5,
    )
    network = DistanceTelemetryResidualTCN(config)

    output = network(torch.randn(7, 6, 200), torch.randn(7, 2))

    assert output.shape == (7,)
    assert torch.all(torch.abs(output) <= 1.5)
    assert network.blocks[0].net[0].dilation == (1,)
    assert network.blocks[1].net[0].dilation == (2,)
    assert trainable_parameter_count(network) == 274


@pytest.mark.skipif(not torch_available(), reason="optional torch dependency missing")
@pytest.mark.parametrize(
    "override, message",
    [
        ({"kernel_size": 4}, "positive odd"),
        ({"dilations": ()}, "non-empty"),
        ({"dilations": (1, 0)}, "positive integers"),
        ({"dilations": (1, 1)}, "increasing and unique"),
        ({"dropout": 1.0}, r"in \[0, 1\)"),
        ({"hidden_channels": 0}, "hidden_channels"),
        ({"head_hidden_dim": 0}, "head_hidden_dim"),
        ({"max_abs_correction_seconds": float("nan")}, "finite and positive"),
    ],
)
def test_residual_network_rejects_invalid_architecture_controls(
    override: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "input_channels": 6,
        "distance_bins": 200,
        "static_feature_dim": 2,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        DistanceTelemetryResidualTCN(DistanceTelemetryResidualTCNConfig(**values))


def test_event_relative_tensor_adapter_removes_common_circuit_shape_without_targets() -> None:
    drivers = 5
    base = np.arange(drivers * 6 * 12, dtype=np.float32).reshape(drivers, 6, 12)
    event1 = base.copy()
    event2 = base + np.linspace(100.0, 200.0, num=12, dtype=np.float32)[None, None, :]
    raw = np.concatenate([event1, event2], axis=0)
    keys = np.asarray([202601] * drivers + [202602] * drivers)

    normalized, audit = _event_relative_tensor_features(raw, keys)

    assert normalized[:drivers] == pytest.approx(normalized[drivers:], abs=1e-5)
    assert audit["target_values_used"] is False
    assert audit["available_before_qualifying"] is True


def test_event_relative_static_rank_spans_symmetric_endpoints_and_centers_singleton() -> None:
    frame = pd.DataFrame(
        {
            "event_key": [202601, 202601, 202601, 202602],
            "rehearsal_reference_seconds": [82.0, 80.0, 81.0, 90.0],
        }
    )

    features = _event_relative_static_features(frame)

    assert features[:3, 1].tolist() == pytest.approx([1.0, -1.0, 0.0])
    assert features[3, 1] == pytest.approx(0.0)


@pytest.mark.skipif(not torch_available(), reason="optional torch dependency missing")
def test_training_target_is_clipped_to_the_network_output_support() -> None:
    dataset = _dataset(event_count=4)
    frame = dataset.frame.copy(deep=True)
    frame.loc[0, "actual_lap_time_seconds"] += 10.0
    bounded = replace(dataset, frame=frame)
    config = TCNResearchConfig(max_abs_correction_seconds=0.1)

    _, _, target, _, audit = _training_tensors(
        bounded,
        np.arange(len(bounded.frame), dtype=int),
        config,
    )

    assert float(torch.max(torch.abs(target)).item()) <= 0.1000001
    assert audit["rows_clipped_to_model_output_bound"] > 0
    assert audit["training_target_matches_model_output_support"] is True


@pytest.mark.skipif(not torch_available(), reason="optional torch dependency missing")
def test_zero_telemetry_sham_keeps_identical_architecture_and_static_features() -> None:
    dataset = _dataset()
    base = TCNResearchConfig(
        minimum_train_events=3,
        maximum_epochs=2,
        early_stopping_patience=2,
        seed=9876,
    )

    observed = run_expanding_tcn_benchmark(dataset, config=base)
    sham = run_expanding_tcn_benchmark(
        dataset,
        config=replace(base, telemetry_input_mode=TELEMETRY_INPUT_ZERO_ABLATION),
    )

    assert observed["architecture"] == sham["architecture"]
    assert (
        observed["capacity"]["trainable_scalar_parameter_count"]
        == sham["capacity"]["trainable_scalar_parameter_count"]
        == 274
    )
    assert observed["model_input_ablation"][
        "observed_telemetry_values_passed_to_model"
    ] is True
    assert sham["model_input_ablation"][
        "telemetry_zeroed_after_target_free_normalization"
    ] is True
    assert sham["model_input_ablation"]["static_anchor_features_retained"] is True
    assert (
        observed["model_input_ablation"]["telemetry_model_input_sha256"]
        != sham["model_input_ablation"]["telemetry_model_input_sha256"]
    )


@pytest.mark.skipif(not torch_available(), reason="optional torch dependency missing")
def test_nested_expanding_tcn_is_deterministic_and_never_uses_outer_target() -> None:
    dataset = _dataset()
    config = TCNResearchConfig(
        minimum_train_events=3,
        maximum_epochs=3,
        early_stopping_patience=2,
        seed=1234,
    )

    first = run_expanding_tcn_benchmark(dataset, config=config)
    second = run_expanding_tcn_benchmark(dataset, config=config)

    assert first["prediction_set_sha256"] == second["prediction_set_sha256"]
    assert first["warmup_event_keys"] == [202601, 202602, 202603]
    assert first["scored_event_keys"] == [202604, 202605, 202606]
    assert first["capacity"]["trainable_scalar_parameter_count"] == 274
    assert first["capacity"]["correlated_tensor_count_treated_as_sample_size"] is False
    for fold in first["folds"]:
        target = fold["target_event_key"]
        assert target not in fold["prior_event_keys"]
        assert fold["target_event_used_for_training_or_selection"] is False
        inner = fold["inner_selection_and_early_stopping"]
        assert max(inner["inner_train_event_keys"]) < inner["inner_validation_event_key"]
        assert inner["inner_validation_event_key"] < target
        assert inner["target_event_used"] is False
        assert inner["initialization_seed"] == fold["outer_refit"][
            "initialization_seed"
        ]
        assert all(
            value == pytest.approx(1.0)
            for value in fold["outer_refit"]["event_weight_sums"].values()
        )
        assert (
            fold["outer_refit"]["tcn_feature_extractor_parameter_change_l2"]
            > 0.0
        )


@pytest.mark.skipif(not torch_available(), reason="optional torch dependency missing")
def test_target_event_truth_cannot_change_its_tcn_forecast() -> None:
    dataset = _dataset()
    config = TCNResearchConfig(
        minimum_train_events=3,
        maximum_epochs=2,
        early_stopping_patience=2,
        seed=4321,
    )
    original = run_expanding_tcn_benchmark(dataset, config=config)
    mutated_frame = dataset.frame.copy(deep=True)
    mask = mutated_frame["event_key"] == 202606
    mutated_frame.loc[mask, "actual_lap_time_seconds"] += 20.0
    mutated_frame.loc[mask, "target_residual_seconds"] += 20.0
    mutated = run_expanding_tcn_benchmark(
        TCNBagDataset(
            frame=mutated_frame,
            telemetry=dataset.telemetry.copy(),
            static_features=dataset.static_features.copy(),
            channel_names=dataset.channel_names,
            distance_bins=dataset.distance_bins,
            validated_tensor_count=dataset.validated_tensor_count,
            feature_set_sha256=dataset.feature_set_sha256,
        ),
        config=config,
    )
    before = next(fold for fold in original["folds"] if fold["target_event_key"] == 202606)
    after = next(fold for fold in mutated["folds"] if fold["target_event_key"] == 202606)

    assert len(before["predictions"]) == len(after["predictions"])
    for before_row, after_row in zip(before["predictions"], after["predictions"]):
        assert before_row["driver_id"] == after_row["driver_id"]
        for model in (
            "raw_baseline",
            "source_shift_baseline",
            "tcn_driver_correction",
            "locked_selected_policy",
        ):
            assert before_row[
                f"{model}_predicted_lap_time_seconds"
            ] == pytest.approx(
                after_row[f"{model}_predicted_lap_time_seconds"], abs=0.0
            )
