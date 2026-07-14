from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from packages.f1.data.providers.telemetry_cache import (
    NORMALIZED_TELEMETRY_CHANNELS,
    sha256_file,
)
from packages.f1.data.providers.telemetry_supervised import (
    LAP_TIME_CENSORED_STATUS,
    LAP_TIME_OBSERVED_STATUS,
    SUPERVISED_TELEMETRY_SCHEMA_VERSION,
    TARGET_CONTRACT,
    canonical_sha256,
)
from packages.sports_core.paths import find_repo_root
from run_prequal_telemetry_residual_research import (
    FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    TelemetryResidualResearchError,
    _implementation_manifest,
    aggregate_supervised_manifest,
    run_expanding_window_benchmark,
)


def test_residual_research_manifest_binds_all_repository_implementation_inputs() -> None:
    root = find_repo_root()
    manifest = _implementation_manifest(root)
    paths = {str(row["path"]) for row in manifest}

    assert {
        "packages/f1/data/providers/telemetry_cache.py",
        "packages/f1/data/providers/telemetry_supervised.py",
        "packages/sports_core/paths.py",
        (
            "research/projects/F1/rising_qualification_prediction/Python/"
            "repo_bootstrap.py"
        ),
        (
            "research/projects/F1/rising_qualification_prediction/Python/"
            "run_prequal_telemetry_residual_research.py"
        ),
    } == paths
    for row in manifest:
        path = root / str(row["path"])
        assert int(row["size_bytes"]) == path.stat().st_size
        assert str(row["sha256"]) == sha256_file(path)


def _tensor_values(*, event: int, driver: int, lap: int, bins: int = 24) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * np.pi, num=bins, endpoint=False)
    speed = 225.0 + event * 2.0 + driver * 4.0 + 35.0 * np.sin(phase)
    rpm = 9_500.0 + event * 40.0 + driver * 80.0 + 1_500.0 * np.sin(phase + 0.2)
    gear = np.clip(np.rint(4.5 + 2.5 * np.sin(phase)), 1.0, 8.0)
    throttle = np.clip(70.0 + driver * 2.0 + 35.0 * np.sin(phase + 0.5), 0.0, 100.0)
    brake = (np.sin(phase + 0.1) < -0.55).astype(float)
    drs = (np.cos(phase + lap * 0.05) > 0.65).astype(float)
    return np.vstack([speed, rpm, gear, throttle, brake, drs]).astype(np.float32)


def _write_tensor(
    root: Path,
    *,
    event: int,
    driver: int,
    lap: int,
) -> dict[str, object]:
    relative = Path(
        f"data/f1/telemetry/pre_qualifying/2026/round_{event:02d}/"
        f"DRV{driver}_lap_{lap:03d}.npz"
    )
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=_tensor_values(event=event, driver=driver, lap=lap),
        channel_names=np.asarray(NORMALIZED_TELEMETRY_CHANNELS),
    )
    reference = 83.0 + event * 2.2 - driver * 0.28
    return {
        "path": str(relative),
        "sha256": sha256_file(path),
        "lap_number": lap,
        "push_lap_rank": lap,
        "rehearsal_lap_time_seconds": reference + (lap - 1) * 0.17,
        "feature_as_of": f"2026-01-{event:02d}T10:00:00Z",
        "shape": [len(NORMALIZED_TELEMETRY_CHANNELS), 24],
        "channels": list(NORMALIZED_TELEMETRY_CHANNELS),
        "distance_bins": 24,
        "expected_lap_distance_m": 5_000.0,
        "distance_coverage": 0.99,
    }


def _fixture_manifest(
    root: Path,
    *,
    event_count: int = 6,
    drivers_per_event: int = 5,
) -> dict[str, object]:
    bags: list[dict[str, object]] = []
    for event in range(1, event_count + 1):
        event_key = 202600 + event
        rehearsal_source = "Sprint Qualifying" if event % 2 == 0 else "Practice 3"
        for driver in range(drivers_per_event):
            tensors = [
                _write_tensor(root, event=event, driver=driver, lap=lap)
                for lap in (1, 2)
            ]
            feature: dict[str, object] = {
                "rehearsal_source": rehearsal_source,
                "qualifying_start_utc": f"2026-01-{event:02d}T12:00:00Z",
                "telemetry_manifest_path": f"round_{event:02d}/telemetry_manifest.json",
                "telemetry_manifest_sha256": f"{event:064x}",
                "tensor_count": len(tensors),
                "tensors": tensors,
            }
            feature["feature_bag_sha256"] = canonical_sha256(feature)
            reference = min(
                float(tensor["rehearsal_lap_time_seconds"]) for tensor in tensors
            )
            source_shift = -0.15 if rehearsal_source == "Sprint Qualifying" else -0.75
            residual = source_shift - driver * 0.055 + event * 0.005
            target: dict[str, object] = {
                "target_contract": TARGET_CONTRACT,
                "has_legal_qualifying_lap": True,
                "lap_time_observed": True,
                "lap_time_target_status": LAP_TIME_OBSERVED_STATUS,
                "lap_time_seconds": reference + residual,
                "target_session": "Qualifying",
                "target_available_at": f"2026-01-{event:02d}T14:00:00Z",
                "target_source_manifest_sha256": f"{event + 100:064x}",
                "inference_eligible": False,
            }
            target["target_sha256"] = canonical_sha256(target)
            bag: dict[str, object] = {
                "event_key": event_key,
                "year": 2026,
                "round": event,
                "event_name": f"Test Grand Prix {event}",
                "driver_id": f"DRV{driver}",
                "row_unit": "driver_event_bag",
                "feature": feature,
                "target": target,
            }
            bag["bag_sha256"] = canonical_sha256(bag)
            bags.append(bag)
    payload: dict[str, object] = {
        "schema_version": SUPERVISED_TELEMETRY_SCHEMA_VERSION,
        "year": 2026,
        "targets_inference_eligible": False,
        "supervised_row_unit": "driver_event_bag",
        "independent_evaluation_unit": "event",
        "tensor_rows_are_independent_supervised_rows": False,
        "feature_input_manifest_sha256": "1" * 64,
        "target_input_manifest_sha256": "2" * 64,
        "bags": bags,
        "bag_set_sha256": canonical_sha256([bag["bag_sha256"] for bag in bags]),
    }
    return payload


def _rehash_target_mutation(manifest: dict[str, object], *, event_key: int) -> None:
    bags = manifest["bags"]
    assert isinstance(bags, list)
    for bag in bags:
        assert isinstance(bag, dict)
        if int(bag["event_key"]) != event_key:
            continue
        target = bag["target"]
        assert isinstance(target, dict)
        target.pop("target_sha256")
        target["lap_time_seconds"] = float(target["lap_time_seconds"]) + 20.0
        target["target_sha256"] = canonical_sha256(target)
        bag.pop("bag_sha256")
        bag["bag_sha256"] = canonical_sha256(bag)
    manifest["bag_set_sha256"] = canonical_sha256(
        [bag["bag_sha256"] for bag in bags]
    )


def _censor_driver_target(
    manifest: dict[str, object],
    *,
    event_key: int,
    driver_id: str,
) -> None:
    bags = manifest["bags"]
    assert isinstance(bags, list)
    mutated = False
    for bag in bags:
        assert isinstance(bag, dict)
        if int(bag["event_key"]) != event_key or bag["driver_id"] != driver_id:
            continue
        target = bag["target"]
        assert isinstance(target, dict)
        target.pop("target_sha256")
        target["has_legal_qualifying_lap"] = False
        target["lap_time_observed"] = False
        target["lap_time_target_status"] = LAP_TIME_CENSORED_STATUS
        target["lap_time_seconds"] = None
        target["target_sha256"] = canonical_sha256(target)
        bag.pop("bag_sha256")
        bag["bag_sha256"] = canonical_sha256(bag)
        mutated = True
    assert mutated
    manifest["bag_set_sha256"] = canonical_sha256(
        [bag["bag_sha256"] for bag in bags]
    )


def test_aggregation_keeps_driver_event_as_unit_not_each_tensor(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path, event_count=4, drivers_per_event=3)

    frame = aggregate_supervised_manifest(manifest, root=tmp_path)

    assert len(frame) == 12
    assert int(frame["tensor_count_raw"].sum()) == 24
    assert not frame.duplicated(["event_key", "driver_id"]).any()
    assert frame.loc[:, FEATURE_NAMES].shape == (12, len(FEATURE_NAMES))
    assert np.isfinite(frame.loc[:, FEATURE_NAMES].to_numpy()).all()
    assert frame.loc[:, TEMPORAL_FEATURE_NAMES].shape == (
        12,
        len(TEMPORAL_FEATURE_NAMES),
    )
    assert np.isfinite(frame.loc[:, TEMPORAL_FEATURE_NAMES].to_numpy()).all()
    first = frame.iloc[0]
    assert first["target_residual_seconds"] == pytest.approx(-0.745)
    assert first["event_relative_z__rehearsal_reference_seconds"] > 0.0
    assert first["event_rank__rehearsal_reference_seconds"] == pytest.approx(1.0)


def test_event_relative_features_remove_common_circuit_scale_without_targets(
    tmp_path: Path,
) -> None:
    manifest = _fixture_manifest(tmp_path, event_count=3, drivers_per_event=5)
    frame = aggregate_supervised_manifest(manifest, root=tmp_path)

    event1 = frame.loc[frame["event_key"] == 202601, FEATURE_NAMES].to_numpy()
    event2 = frame.loc[frame["event_key"] == 202602, FEATURE_NAMES].to_numpy()
    temporal1 = frame.loc[
        frame["event_key"] == 202601, TEMPORAL_FEATURE_NAMES
    ].to_numpy()
    temporal2 = frame.loc[
        frame["event_key"] == 202602, TEMPORAL_FEATURE_NAMES
    ].to_numpy()

    # The fixture adds common event offsets to speed, RPM, and lap time.  The
    # target-free within-event normalization must remove all of those aliases.
    assert event1 == pytest.approx(event2, abs=1e-10)
    # Source tensors are float32, so subtracting different common event offsets
    # can leave a few ppm of rounding noise after robust normalization.
    assert temporal1 == pytest.approx(temporal2, abs=1e-5)


def test_censored_bag_stays_in_manifest_but_is_reported_and_not_regressed(
    tmp_path: Path,
) -> None:
    manifest = _fixture_manifest(tmp_path)
    _censor_driver_target(manifest, event_key=202606, driver_id="DRV4")

    frame = aggregate_supervised_manifest(manifest, root=tmp_path)
    censored = frame.loc[
        (frame["event_key"] == 202606) & (frame["driver_id"] == "DRV4")
    ].iloc[0]
    assert len(frame) == 30
    assert censored["has_legal_qualifying_lap"] == False  # noqa: E712
    assert censored["lap_time_observed"] == False  # noqa: E712
    assert np.isnan(censored["actual_lap_time_seconds"])
    assert np.isnan(censored["target_residual_seconds"])
    assert np.isfinite(frame.loc[:, FEATURE_NAMES].to_numpy()).all()

    result = run_expanding_window_benchmark(
        frame,
        minimum_train_events=3,
        minimum_inner_train_events=2,
    )

    assert result["supervised_driver_event_count"] == 30
    assert result["lap_time_model_driver_event_count"] == 29
    assert result["censored_lap_time_driver_event_count"] == 1
    assert result["lap_time_target_counts_by_event"]["202606"] == {
        "driver_event_bag_count": 5,
        "observed_lap_time_target_count": 4,
        "censored_lap_time_target_count": 1,
    }
    event6 = next(
        fold for fold in result["folds"] if fold["target_event_key"] == 202606
    )
    assert event6["target_event_driver_event_bag_count"] == 5
    assert event6["target_event_observed_lap_time_target_count"] == 4
    assert event6["target_event_censored_lap_time_target_count"] == 1
    assert event6["scored_driver_event_count"] == 4
    assert {row["driver_id"] for row in event6["predictions"]} == {
        "DRV0",
        "DRV1",
        "DRV2",
        "DRV3",
    }


def test_nested_expanding_window_never_splits_or_trains_on_target_event(
    tmp_path: Path,
) -> None:
    manifest = _fixture_manifest(tmp_path)
    frame = aggregate_supervised_manifest(manifest, root=tmp_path)

    result = run_expanding_window_benchmark(
        frame,
        minimum_train_events=3,
        minimum_inner_train_events=2,
    )

    assert result["warmup_event_keys"] == [202601, 202602, 202603]
    assert [fold["target_event_key"] for fold in result["folds"]] == [
        202604,
        202605,
        202606,
    ]
    assert result["supervised_driver_event_count"] == 30
    assert result["validated_tensor_count"] == 60
    for fold in result["folds"]:
        target = fold["target_event_key"]
        assert fold["prior_event_keys"] == list(range(202601, target))
        assert target not in fold["ridge_fit_audit"]["training_event_keys"]
        assert target not in fold["huber_fit_audit"]["training_event_keys"]
        assert target not in fold["temporal_ridge_fit_audit"]["training_event_keys"]
        assert target not in fold["source_shift_audit"]["fit_event_keys"]
        assert fold["source_shift_audit"]["scored_sources"] == [
            fold["rehearsal_source"]
        ]
        assert fold["source_shift_audit"]["fallback_to_global_sources"] == []
        assert all(
            value == pytest.approx(1.0)
            for value in fold["ridge_fit_audit"]["event_weight_sums"].values()
        )
        assert all(
            value == pytest.approx(1.0)
            for value in fold["huber_fit_audit"]["event_weight_sums"].values()
        )
        for selection_name in (
            "ridge_selection",
            "huber_selection",
            "temporal_ridge_selection",
        ):
            selection = fold[selection_name]
            assert max(selection["selection_source_event_keys"]) < target
            assert any(
                candidate["candidate"]["candidate_id"]
                == "zero_telemetry_correction"
                for candidate in selection["candidates"]
            )
            for candidate in selection["candidates"]:
                for inner in candidate["folds"]:
                    assert max(inner["prior_event_keys"]) < inner["target_event_key"]
                    assert inner["target_event_key"] < target
        assert len(fold["predictions"]) == 5
        assert len(fold["prediction_set_sha256"]) == 64
    assert len(result["prediction_set_sha256"]) == 64
    assert result["summary"]["scored_event_count"] == 3
    assert all("event_relative" in name or "event_rank" in name for name in FEATURE_NAMES)
    assert all(name.startswith("event_relative_z__temporal_") for name in TEMPORAL_FEATURE_NAMES)
    assert not any(name.startswith("lap_median") for name in FEATURE_NAMES)
    assert "source_shift_baseline" in result["summary"]
    assert result["summary"]["source_shift_baseline"][
        "event_balanced_mae_seconds"
    ] < result["summary"]["raw_baseline"]["event_balanced_mae_seconds"]
    assert result["folds"][0]["ridge_selection"]["selected_candidate_id"] == (
        "zero_telemetry_correction"
    )
    assert result["folds"][1]["huber_selection"]["selected_candidate_id"] == (
        "zero_telemetry_correction"
    )
    ridge_alphas = [
        candidate["candidate"].get("alpha")
        for candidate in result["folds"][-1]["ridge_selection"]["candidates"]
        if candidate["candidate"]["family"] == "ridge"
    ]
    assert max(ridge_alphas) == pytest.approx(100_000.0)
    capacity = result["temporal_capacity_diagnostic"]
    assert capacity["fixed_minimum_event_count_claim_used"] is False
    assert capacity["true_tcn_evaluated"] is False
    assert capacity["true_tcn_runtime"]["minimum_supported_python"] == "3.9"
    assert capacity["true_tcn_runtime"]["python_version_supported"] is True
    assert isinstance(capacity["true_tcn_runtime"]["python_version_supported"], bool)
    assert isinstance(capacity["true_tcn_runtime"]["torch_dependency_available"], bool)
    assert capacity["representation_feature_count"] == len(TEMPORAL_FEATURE_NAMES)
    assert capacity["scored_fold_count"] == 3
    assert not capacity["promotion_eligible"]
    assert [
        row["training_event_count"]
        for row in capacity["chronological_capacity_trace"]
    ] == [3, 4, 5]


def test_future_truth_cannot_change_its_forecast_or_any_earlier_forecast(
    tmp_path: Path,
) -> None:
    manifest = _fixture_manifest(tmp_path)
    original = run_expanding_window_benchmark(
        aggregate_supervised_manifest(manifest, root=tmp_path),
        minimum_train_events=3,
        minimum_inner_train_events=2,
    )
    mutated_manifest = copy.deepcopy(manifest)
    _rehash_target_mutation(mutated_manifest, event_key=202606)
    mutated = run_expanding_window_benchmark(
        aggregate_supervised_manifest(mutated_manifest, root=tmp_path),
        minimum_train_events=3,
        minimum_inner_train_events=2,
    )

    assert len(original["folds"]) == len(mutated["folds"])
    for before_fold, after_fold in zip(original["folds"], mutated["folds"]):
        assert before_fold["target_event_key"] == after_fold["target_event_key"]
        assert before_fold["ridge_selection"]["selected_candidate_id"] == after_fold[
            "ridge_selection"
        ]["selected_candidate_id"]
        assert before_fold["huber_selection"]["selected_candidate_id"] == after_fold[
            "huber_selection"
        ]["selected_candidate_id"]
        assert len(before_fold["predictions"]) == len(after_fold["predictions"])
        for before, after in zip(
            before_fold["predictions"], after_fold["predictions"]
        ):
            for model_name in (
                "raw_baseline",
                "source_shift_baseline",
                "ridge_or_zero",
                "huber_or_zero",
                "temporal_ridge_or_zero",
            ):
                assert before[
                    f"{model_name}_predicted_lap_time_seconds"
                ] == pytest.approx(
                    after[f"{model_name}_predicted_lap_time_seconds"], abs=0.0
                )
    original_event6 = next(
        fold for fold in original["folds"] if fold["target_event_key"] == 202606
    )
    mutated_event6 = next(
        fold for fold in mutated["folds"] if fold["target_event_key"] == 202606
    )
    assert mutated_event6["predictions"][0]["actual_lap_time_seconds"] == pytest.approx(
        original_event6["predictions"][0]["actual_lap_time_seconds"] + 20.0
    )


def test_insufficient_independent_events_fail_closed(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path, event_count=3)
    frame = aggregate_supervised_manifest(manifest, root=tmp_path)

    with pytest.raises(TelemetryResidualResearchError, match="need more than 3"):
        run_expanding_window_benchmark(
            frame,
            minimum_train_events=3,
            minimum_inner_train_events=2,
        )


def test_changed_tensor_is_rejected_before_model_fit(tmp_path: Path) -> None:
    manifest = _fixture_manifest(tmp_path, event_count=4)
    first_bag = manifest["bags"][0]
    tensor = first_bag["feature"]["tensors"][0]
    tensor_path = tmp_path / str(tensor["path"])
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered")

    with pytest.raises(TelemetryResidualResearchError, match="tensor hash mismatch"):
        aggregate_supervised_manifest(manifest, root=tmp_path)
