from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

import run_best_estimated_lap_2026_backtest as best_runner

from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN,
    Q1_LAP_COLUMN,
    Q2_LAP_COLUMN,
    Q3_LAP_COLUMN,
    calibrate_achievable_best_lap_model,
    decompose_event_fastest_and_driver_gap,
    fit_achievable_best_lap_model,
    robust_huber_location,
    sample_joint_qualifying_laps,
    shared_qualifying_forecast_artifact,
    shared_point_predictor_sha256,
    summarize_joint_lap_samples,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
)
from run_best_estimated_lap_2026_backtest import (
    _build_shared_event_forecast as build_best_runner_shared_forecast,
    _locked_best_lap_partitions,
    run_backtest,
)
from run_qualifying_pairwise_challenger_backtest import (
    _build_shared_event_forecast as build_quali_runner_shared_forecast,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_key": [202601, 202601, 202602, 202602, 202602],
            "driver_id": ["a", "b", "a", "b", "c"],
            "rehearsal_source": [
                "practice_3",
                "practice_3",
                "practice_3",
                "practice_3",
                "sprint_qualifying",
            ],
            "rehearsal_lap_time_seconds": [80.0, 81.0, 90.0, 91.0, 89.0],
            ACTUAL_LAP_COLUMN: [79.0, 80.0, 88.0, 89.0, 88.8],
        }
    )


def test_model_learns_event_balanced_source_specific_shift() -> None:
    model = fit_achievable_best_lap_model(_history(), target_event_key=202603)
    inputs = pd.DataFrame(
        {
            "event_key": [202603, 202603],
            "driver_id": ["x", "y"],
            "rehearsal_source": ["practice_3", "sprint_qualifying"],
            "rehearsal_lap_time_seconds": [100.0, 100.0],
        }
    )

    result = model.predict(inputs)

    assert result["lap_p50"].tolist() == pytest.approx([98.5, 99.8])
    assert result["session_shift_seconds"].tolist() == pytest.approx([-1.5, -0.2])
    assert result["target_contract"].eq(ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT).all()
    assert (result["lap_p05"] <= result["lap_p50"]).all()
    assert (result["lap_p50"] <= result["lap_p90"]).all()
    assert result.loc[0, "interval_status"] == "diagnostic_no_disjoint_calibration_partition"
    assert result.loc[1, "interval_status"] == "diagnostic_no_disjoint_calibration_partition"


def test_cold_start_is_explicit_and_does_not_fake_quantiles() -> None:
    model = fit_achievable_best_lap_model(pd.DataFrame(), target_event_key=202601)
    result = model.predict(
        pd.DataFrame(
            {
                "event_key": [202601],
                "driver_id": ["x"],
                "rehearsal_source": ["fp3"],
                "rehearsal_lap_time_seconds": [80.5],
            }
        )
    )

    assert result.loc[0, "lap_p50"] == 80.5
    assert math.isnan(result.loc[0, "lap_p05"])
    assert math.isnan(result.loc[0, "lap_p90"])
    assert result.loc[0, "interval_status"] == "unavailable_no_same_source_history"


def test_training_and_inference_fail_closed_on_target_leakage() -> None:
    with pytest.raises(ValueError, match="strictly earlier"):
        fit_achievable_best_lap_model(_history(), target_event_key=202602)

    model = fit_achievable_best_lap_model(_history(), target_event_key=202603)
    leaked = pd.DataFrame(
        {
            "event_key": [202603],
            "driver_id": ["x"],
            "rehearsal_source": ["fp3"],
            "rehearsal_lap_time_seconds": [80.0],
            ACTUAL_LAP_COLUMN: [79.0],
        }
    )
    with pytest.raises(ValueError, match="target/outcome"):
        model.predict(leaked)


def test_best_runner_target_io_requires_frozen_event_forecast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    target_laps = tmp_path / "qualifying_laps.csv"
    target_results = tmp_path / "qualifying_results.csv"
    target_laps.write_text("target\n", encoding="utf-8")
    target_results.write_text("target\n", encoding="utf-8")
    calls: list[str] = []

    def read_target(_path):
        calls.append("target_read")
        return pd.Series({"AAA": 90.0})

    def hash_target(paths, *, root):
        calls.append("target_hash")
        return {str(path.relative_to(root)): "f" * 64 for path in paths}

    monkeypatch.setattr(best_runner, "_official_driver_best_laps", read_target)
    monkeypatch.setattr(best_runner, "_hash_manifest", hash_target)

    with pytest.raises(RuntimeError, match="requires a frozen forecast artifact"):
        best_runner._load_target_after_frozen_forecasts(
            target_laps,
            expected_event_key=202601,
            frozen_forecast_artifact={},
            root=tmp_path,
        )
    assert calls == []

    actual, manifest = best_runner._load_target_after_frozen_forecasts(
        target_laps,
        expected_event_key=202601,
        frozen_forecast_artifact={
            "event_key": 202601,
            "artifact_sha256": "a" * 64,
        },
        root=tmp_path,
    )

    assert calls == ["target_read", "target_hash"]
    assert actual.to_dict() == {"AAA": 90.0}
    assert set(manifest) == {"qualifying_laps.csv", "qualifying_results.csv"}


def test_best_runner_revalidates_telemetry_and_binds_all_inputs(tmp_path) -> None:
    event_dir = (
        tmp_path
        / "data/f1/telemetry/pre_qualifying/2026/round_01_test_grand_prix"
    )
    tensor_path = event_dir / "features/AAA_lap_001.npz"
    tensor_path.parent.mkdir(parents=True)
    feature_as_of = pd.Timestamp("2026-03-01T11:59:59Z")
    np.savez_compressed(
        tensor_path,
        schema_version=np.asarray("f1_prequal_distance_tensor_v1"),
        values=np.ones((6, 2), dtype=np.float32),
        channel_names=np.asarray(("Speed", "RPM", "nGear", "Throttle", "Brake", "DRS")),
        distance_grid_m=np.asarray([0.0, 1_000.0], dtype=np.float32),
        sample_timestamp_ns=np.asarray(
            [feature_as_of.value - 1_000_000_000, feature_as_of.value], dtype=np.int64
        ),
        feature_as_of_ns=np.asarray(feature_as_of.value, dtype=np.int64),
        expected_lap_distance_m=np.asarray(1_000.0, dtype=np.float64),
        distance_coverage=np.asarray(1.0, dtype=np.float64),
    )
    tensor_sha = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    tensor_relative = str(tensor_path.relative_to(tmp_path))
    record = {
        "event_key": 202601,
        "driver_id": "AAA",
        "lap_number": 1,
        "feature_as_of": "2026-03-01T11:59:59Z",
        "qualifying_start_utc": "2026-03-01T12:00:00Z",
        "telemetry_path": tensor_relative,
        "telemetry_sha256": tensor_sha,
        "tensor_schema_version": "f1_prequal_distance_tensor_v1",
        "distance_normalized": True,
        "telemetry_shape": [6, 2],
        "distance_bins": 2,
        "channels": ["Speed", "RPM", "nGear", "Throttle", "Brake", "DRS"],
        "expected_lap_distance_m": 1_000.0,
        "distance_coverage": 1.0,
        "minimum_distance_coverage": 0.95,
    }
    manifest_path = event_dir / "telemetry_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "f1_prequal_telemetry_cache_v2",
                "event_key": 202601,
                "year": 2026,
                "qualifying_start_utc": "2026-03-01T12:00:00Z",
                "feature_records": [record],
                "rejected_feature_records": [],
            }
        ),
        encoding="utf-8",
    )

    readiness, inputs = best_runner._telemetry_readiness_audit(
        root=tmp_path,
        year=2026,
        minimum_independent_events=2,
        minimum_drivers_per_event=1,
    )

    assert set(inputs) == {manifest_path, tensor_path}
    assert readiness["accessed_manifest_file_count"] == 1
    assert readiness["accessed_tensor_file_count"] == 1
    assert readiness["accessed_input_file_count"] == 2
    assert len(readiness["manifest_set_sha256"]) == 64
    assert len(readiness["accessed_input_manifest_sha256"]) == 64
    assert readiness["audit"]["record_count"] == 1
    assert readiness["audit"]["validated_tensor_count"] == 1
    assert readiness["audit"]["event_count"] == 1
    assert readiness["audit"]["driver_event_count"] == 1
    assert not readiness["audit"]["ready_for_deep_model"]
    assert readiness["audit"]["blockers"] == [
        "insufficient_independent_prequalifying_telemetry_events"
    ]
    input_manifest = best_runner._hash_manifest(inputs, root=tmp_path)
    assert set(input_manifest) == {
        str(manifest_path.relative_to(tmp_path)),
        tensor_relative,
    }
    decision = best_runner._deep_model_readiness_decision(readiness["audit"])
    assert decision["deep_model_telemetry_blockers"] == readiness["audit"]["blockers"]
    assert decision["deep_model_blockers"] == [
        "insufficient_independent_prequalifying_telemetry_events"
    ]
    assert "no_2026_distance_normalized_car_telemetry_cache" not in decision[
        "deep_model_blockers"
    ]


def test_best_runner_keeps_ready_but_unevaluated_deep_model_fail_closed() -> None:
    decision = best_runner._deep_model_readiness_decision(
        {"ready_for_deep_model": True, "blockers": []}
    )

    assert decision["deep_model_telemetry_blockers"] == []
    assert decision["deep_model_blockers"] == [
        "deep_model_challenger_not_trained_or_evaluated"
    ]


def test_inference_rejects_wrong_target_event() -> None:
    model = fit_achievable_best_lap_model(_history(), target_event_key=202603)
    inputs = pd.DataFrame(
        {
            "event_key": [202604],
            "driver_id": ["x"],
            "rehearsal_source": ["practice_3"],
            "rehearsal_lap_time_seconds": [80.0],
        }
    )

    with pytest.raises(ValueError, match="target_event_key"):
        model.predict(inputs)


def test_backtest_rejects_pre_2024_sprint_chronology(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"2024\+ weekend chronology"):
        run_backtest(
            weekends_dir=tmp_path,
            year=2023,
            rounds=[1],
            bootstrap_samples=10,
            bootstrap_seed=7,
        )


def test_huber_location_keeps_one_value_and_resists_large_outlier() -> None:
    assert robust_huber_location([1.25]) == pytest.approx(1.25)
    assert robust_huber_location([0.0, 0.1, -0.1, 20.0]) < 1.0


def test_event_fastest_and_driver_gap_decomposition_is_lossless() -> None:
    frame = pd.DataFrame(
        {
            "event_key": [1, 1, 2, 2],
            "anchor": [90.0, 91.0, 100.0, 102.0],
            ACTUAL_LAP_COLUMN: [89.0, 90.5, 98.0, 101.0],
        }
    )
    result = decompose_event_fastest_and_driver_gap(frame, anchor_column="anchor")

    reconstructed = result["anchor"] + result["decomposed_total_residual_seconds"]
    assert reconstructed.tolist() == pytest.approx(frame[ACTUAL_LAP_COLUMN].tolist())
    assert result.loc[0, "target_gap_to_event_fastest_seconds"] == 0.0
    assert result.loc[3, "target_gap_to_event_fastest_seconds"] == pytest.approx(3.0)


def test_explicit_valid_lap_and_stage_hurdle_is_event_balanced() -> None:
    history = _history().assign(
        team_id=["ta", "tb", "ta", "tb", "tc"],
        has_valid_qualifying_lap=[True, False, True, True, True],
        reached_q2=[True, False, True, False, True],
        reached_q3=[False, False, True, False, True],
    )
    model = fit_achievable_best_lap_model(history, target_event_key=202603)
    result = model.predict(
        pd.DataFrame(
            {
                "event_key": [202603],
                "driver_id": ["x"],
                "team_id": ["new"],
                "rehearsal_source": ["practice_3"],
                "rehearsal_lap_time_seconds": [90.0],
            }
        )
    )

    assert 0.0 < result.loc[0, "valid_lap_probability"] < 1.0
    assert 0.0 < result.loc[0, "q2_given_valid_probability"] < 1.0
    assert 0.0 < result.loc[0, "q3_given_q2_probability"] < 1.0
    assert result.loc[0, "stage_probability_status"] == "regularized_logistic_not_posthoc_calibrated"
    assert (
        result.loc[0, "no_valid_lap_probability"]
        + result.loc[0, "q1_only_probability"]
        + result.loc[0, "q2_only_probability"]
        + result.loc[0, "q3_probability"]
    ) == pytest.approx(1.0)


def test_quality_anchor_robust_residual_and_event_block_interval_are_exposed() -> None:
    history = _history().assign(
        quality_aware_anchor_seconds=[80.0, 81.0, 90.0, 91.0, 89.0],
        team_id=["ta", "tb", "ta", "tb", "tc"],
        evidence_coverage_rate=[1.0, 0.9, 0.8, 0.7, 1.0],
    )
    model = fit_achievable_best_lap_model(history, target_event_key=202603)
    result = model.predict(
        pd.DataFrame(
            {
                "event_key": [202603],
                "driver_id": ["x"],
                "team_id": ["ta"],
                "rehearsal_source": ["practice_3"],
                "quality_aware_anchor_seconds": [100.0],
                "anchor_source": ["valid_clean_rehearsal"],
                "anchor_uncertainty_seconds": [0.6],
                "evidence_coverage_rate": [0.8],
            }
        )
    )

    assert result.loc[0, "anchor_available"]
    assert result.loc[0, "interval_method"] == (
        "diagnostic_prequential_anchor_shift_residual_quantiles_"
        "not_final_predictor_calibration"
    )
    assert result.loc[0, "interval_nominal_mass"] == pytest.approx(0.85)
    assert result.loc[0, "lap_p05"] <= result.loc[0, "lap_p50"] <= result.loc[0, "lap_p90"]


def test_weak_transfer_prior_cannot_create_historical_team_or_driver_effect() -> None:
    history = pd.DataFrame(
        {
            "event_key": [202501, 202501, 202601, 202601],
            "driver_id": ["old_a", "old_b", "new_a", "new_b"],
            "team_id": ["OldTeam", "OldTeam", "NewTeam", "NewTeam"],
            "rehearsal_source": ["practice_3"] * 4,
            "quality_aware_anchor_seconds": [90.0, 91.0, 90.0, 91.0],
            ACTUAL_LAP_COLUMN: [95.0, 96.0, 89.5, 90.5],
            "history_weight": [0.1, 0.1, 1.0, 1.0],
            "weak_transfer_prior": [True, True, False, False],
        }
    )

    model = fit_achievable_best_lap_model(history, target_event_key=202602)

    assert "OldTeam" not in model.residual_model.team_effects
    assert "old_a" not in model.residual_model.driver_effects
    calibration = model.calibrations["practice_3"]
    assert calibration.event_keys == (202501, 202601)
    assert calibration.conformal_event_keys == (202601,)


def test_joint_samples_generate_coherent_fastest_and_top3_probabilities() -> None:
    predictions = pd.DataFrame(
        {
            "driver_id": ["fast", "slow", "invalid"],
            "lap_p05": [89.8, 91.8, 90.8],
            "lap_p50": [90.0, 92.0, 91.0],
            "lap_p90": [90.2, 92.2, 91.2],
            "valid_lap_probability": [1.0, 1.0, 0.0],
            "q2_given_valid_probability": [1.0, 1.0, 0.0],
            "q3_given_q2_probability": [1.0, 1.0, 0.0],
            "stage_q1_residual_sigma_seconds": [0.1, 0.1, 0.1],
            "stage_q2_residual_sigma_seconds": [0.1, 0.1, 0.1],
            "stage_q3_residual_sigma_seconds": [0.1, 0.1, 0.1],
        }
    )

    samples = sample_joint_qualifying_laps(predictions, samples=2_000, seed=7)
    summary = summarize_joint_lap_samples(samples).set_index("driver_id")

    assert samples.lap_seconds.shape == (2_000, 3)
    assert summary.loc["fast", "fastest_driver_probability"] > 0.99
    assert summary.loc["invalid", "valid_lap_probability_sampled"] == 0.0
    assert summary["fastest_driver_probability"].sum() == pytest.approx(1.0)


def test_joint_sampler_fails_closed_on_missing_hurdles_unless_diagnostic_is_explicit() -> None:
    predictions = pd.DataFrame(
        {
            "driver_id": ["fast", "slow"],
            "lap_p05": [89.8, 90.8],
            "lap_p50": [90.0, 91.0],
            "lap_p90": [90.2, 91.2],
            "valid_lap_probability": [1.0, float("nan")],
            "q2_given_valid_probability": [float("nan"), float("nan")],
            "q3_given_q2_probability": [float("nan"), float("nan")],
        }
    )

    with pytest.raises(ValueError, match="finite valid/Q2/Q3 hurdles"):
        sample_joint_qualifying_laps(predictions, samples=10, seed=1)

    diagnostic = sample_joint_qualifying_laps(
        predictions,
        samples=10,
        seed=1,
        allow_diagnostic_pace_fallback=True,
    )
    assert (
        diagnostic.stage_advancement_status
        == "diagnostic_pace_fallback_missing_hurdles"
    )


def test_joint_sampler_requires_learned_stage_dispersion_for_promotable_probabilities() -> None:
    predictions = pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "lap_p05": [89.8, 90.8],
            "lap_p50": [90.0, 91.0],
            "lap_p90": [90.2, 91.2],
            "valid_lap_probability": [1.0, 1.0],
            "q2_given_valid_probability": [0.8, 0.2],
            "q3_given_q2_probability": [0.7, 0.3],
        }
    )

    with pytest.raises(ValueError, match="learned stage residual dispersion"):
        sample_joint_qualifying_laps(predictions, samples=10, seed=1)

    diagnostic = sample_joint_qualifying_laps(
        predictions,
        samples=10,
        seed=1,
        allow_diagnostic_pace_fallback=True,
    )
    assert (
        diagnostic.stage_time_distribution_status
        == "diagnostic_missing_learned_stage_residual_dispersion"
    )


def _shared_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in range(202501, 202506):
        for index in range(20):
            stage = 3 if index < 10 else 2 if index < 15 else 1
            stage_effect = {1: 0.8, 2: 0.3, 3: -0.1}[stage]
            rows.append(
                {
                    "event_key": event,
                    "driver_id": f"d{index:02d}",
                    "team_id": f"t{index // 2:02d}",
                    "rehearsal_source": "practice_3",
                    "rehearsal_lap_time_seconds": 90.0 + index * 0.1,
                    ACTUAL_LAP_COLUMN: 89.0 + index * 0.1 + stage_effect,
                    Q1_LAP_COLUMN: 89.0 + index * 0.1 + 0.8,
                    Q2_LAP_COLUMN: 89.0 + index * 0.1 + 0.3 if stage >= 2 else float("nan"),
                    Q3_LAP_COLUMN: 89.0 + index * 0.1 - 0.1 if stage >= 3 else float("nan"),
                    "has_valid_qualifying_lap": 1,
                    "reached_q2": int(stage >= 2),
                    "reached_q3": int(stage >= 3),
                    "push_lap_count": 3,
                }
            )
    return pd.DataFrame(rows)


def test_shared_model_learns_stage_times_and_uses_true_p05_p90_semantics() -> None:
    history = _shared_history()
    model = fit_achievable_best_lap_model(
        history,
        target_event_key=202601,
        calibration_event_keys=(202504, 202505),
    )
    inference = history.loc[history["event_key"].eq(202505)].drop(
        columns=[
            ACTUAL_LAP_COLUMN,
            Q1_LAP_COLUMN,
            Q2_LAP_COLUMN,
            Q3_LAP_COLUMN,
            "has_valid_qualifying_lap",
            "reached_q2",
            "reached_q3",
        ]
    )
    inference["event_key"] = 202601
    result = model.predict(inference)

    assert model.stage_time_effects.fitted
    assert model.stage_time_effects.q1_only_seconds > model.stage_time_effects.q3_seconds
    assert model.stage_time_effects.q1_residual_sigma_seconds > 0.0
    assert model.stage_time_effects.q2_residual_sigma_seconds > 0.0
    assert model.stage_time_effects.q3_residual_sigma_seconds > 0.0
    assert result["lap_p05_quantile_probability"].eq(0.05).all()
    assert result["lap_p90_quantile_probability"].eq(0.90).all()
    assert result["interval_nominal_mass"].eq(0.85).all()
    assert result["interval_status"].eq(
        "diagnostic_calibration_rows_reused_for_model_fit"
    ).all()
    assert not result["calibration_partition_validated"].any()


def test_held_out_final_predictor_residuals_calibrate_without_changing_point_model() -> None:
    history = _shared_history()
    point_history = history.loc[history["event_key"].le(202503)].copy()
    calibration_rows: list[pd.DataFrame] = []
    for event_key in (202504, 202505):
        event_model = fit_achievable_best_lap_model(
            point_history,
            target_event_key=event_key,
        )
        event = history.loc[history["event_key"].eq(event_key)].copy()
        inference = event.drop(
            columns=[
                ACTUAL_LAP_COLUMN,
                Q1_LAP_COLUMN,
                Q2_LAP_COLUMN,
                Q3_LAP_COLUMN,
                "has_valid_qualifying_lap",
                "reached_q2",
                "reached_q3",
            ]
        )
        predicted = event_model.predict_qualifying(
            inference,
            samples=300,
            seed=event_key,
        ).lap_predictions[["event_key", "driver_id", "rehearsal_source", "lap_p50"]]
        predicted[ACTUAL_LAP_COLUMN] = event[ACTUAL_LAP_COLUMN].to_numpy(dtype=float)
        calibration_rows.append(predicted)
    audit_model = fit_achievable_best_lap_model(
        point_history,
        target_event_key=202506,
    )
    before = shared_point_predictor_sha256(audit_model)
    calibrated = calibrate_achievable_best_lap_model(
        audit_model,
        pd.concat(calibration_rows, ignore_index=True),
    )

    assert calibrated.calibration_partition_validated
    assert calibrated.calibration_event_keys == (202504, 202505)
    assert shared_point_predictor_sha256(calibrated) == before
    audit_inference = history.loc[history["event_key"].eq(202505)].drop(
        columns=[
            ACTUAL_LAP_COLUMN,
            Q1_LAP_COLUMN,
            Q2_LAP_COLUMN,
            Q3_LAP_COLUMN,
            "has_valid_qualifying_lap",
            "reached_q2",
            "reached_q3",
        ]
    )
    audit_inference["event_key"] = 202506
    result = calibrated.predict(audit_inference)
    assert result["interval_status"].eq("calibrated_disjoint_event_partition").all()


def test_best_lap_marginals_and_classification_use_the_exact_same_joint_samples() -> None:
    history = _shared_history()
    model = fit_achievable_best_lap_model(
        history,
        target_event_key=202601,
        calibration_event_keys=(202504, 202505),
    )
    inference = history.loc[history["event_key"].eq(202505)].drop(
        columns=[
            ACTUAL_LAP_COLUMN,
            Q1_LAP_COLUMN,
            Q2_LAP_COLUMN,
            Q3_LAP_COLUMN,
            "has_valid_qualifying_lap",
            "reached_q2",
            "reached_q3",
        ]
    )
    inference["event_key"] = 202601
    forecast = model.predict_qualifying(inference, samples=600, seed=19)
    predicted = forecast.lap_predictions.set_index("driver_id")
    marginals = forecast.position_marginals.set_index("driver_id")

    for driver_index, driver in enumerate(forecast.samples.driver_ids):
        finite = forecast.samples.lap_seconds[:, driver_index]
        finite = finite[pd.notna(finite)]
        assert predicted.loc[driver, "lap_p05"] == pytest.approx(
            float(pd.Series(finite).quantile(0.05))
        )
        assert predicted.loc[driver, "lap_p50"] == pytest.approx(
            float(pd.Series(finite).quantile(0.50))
        )
        assert predicted.loc[driver, "lap_p90"] == pytest.approx(
            float(pd.Series(finite).quantile(0.90))
        )
        assert marginals.loc[driver, "pole_probability"] == pytest.approx(
            float((forecast.samples.official_positions[:, driver_index] == 1).mean())
        )
        sampled_valid = float(forecast.samples.valid_mask[:, driver_index].mean())
        sampled_q2 = float((forecast.samples.deepest_stage[:, driver_index] >= 2).mean())
        sampled_q3 = float((forecast.samples.deepest_stage[:, driver_index] >= 3).mean())
        assert predicted.loc[driver, "valid_lap_probability"] == pytest.approx(
            sampled_valid
        )
        assert predicted.loc[driver, "q2_given_valid_probability"] == pytest.approx(
            sampled_q2 / sampled_valid if sampled_valid else 0.0
        )
        assert predicted.loc[driver, "q3_given_q2_probability"] == pytest.approx(
            sampled_q3 / sampled_q2 if sampled_q2 else 0.0
        )
    assert predicted["joint_sample_seed"].eq(19).all()
    assert predicted["distribution_source"].eq("shared_joint_qualifying_samples").all()
    assert predicted["stage_probability_status"].eq(
        "legal_field_constrained_joint_samples_uncalibrated"
    ).all()
    assert forecast.probability_calibration_status == "uncalibrated_joint_latent_samples"
    assert (
        forecast.samples.stage_time_distribution_status
        == "learned_stage_residual_dispersion"
    )
    partition = {"point_fit_event_keys": [202501, 202502, 202503, 202504, 202505]}
    artifact = shared_qualifying_forecast_artifact(
        forecast,
        model=model,
        training_partition_manifest=partition,
    )
    repeated = shared_qualifying_forecast_artifact(
        forecast,
        model=model,
        training_partition_manifest=partition,
    )
    assert artifact == repeated
    reindexed = replace(
        forecast,
        lap_predictions=forecast.lap_predictions.set_axis(
            range(100, 100 + len(forecast.lap_predictions))
        ),
        point_order=forecast.point_order.set_axis(
            range(200, 200 + len(forecast.point_order))
        ),
        position_marginals=forecast.position_marginals.set_axis(
            range(300, 300 + len(forecast.position_marginals))
        ),
    )
    assert shared_qualifying_forecast_artifact(
        reindexed,
        model=model,
        training_partition_manifest=partition,
    ) == artifact
    assert artifact["event_key"] == 202601
    assert artifact["joint_sample_count"] == 600
    assert artifact["shared_samples_drive_best_lap_and_qualifying"]
    assert len(str(artifact["artifact_sha256"])) == 64
    assert len(str(artifact["joint_samples_sha256"])) == 64
    assert len(str(artifact["model_sha256"])) == 64
    changed_partition = shared_qualifying_forecast_artifact(
        forecast,
        model=model,
        training_partition_manifest={"point_fit_event_keys": [202501]},
    )
    assert changed_partition["artifact_sha256"] != artifact["artifact_sha256"]
    shifted_history = history.copy()
    shifted_history[ACTUAL_LAP_COLUMN] = shifted_history[ACTUAL_LAP_COLUMN] + 0.5
    shifted_model = fit_achievable_best_lap_model(
        shifted_history,
        target_event_key=202601,
        calibration_event_keys=(202504, 202505),
    )
    assert shared_point_predictor_sha256(shifted_model) != artifact["model_sha256"]


def test_quali_and_best_runner_paths_emit_identical_shared_model_and_sample_hashes() -> None:
    history = _shared_history()
    inference = history.loc[history["event_key"].eq(202505)].drop(
        columns=[
            ACTUAL_LAP_COLUMN,
            Q1_LAP_COLUMN,
            Q2_LAP_COLUMN,
            Q3_LAP_COLUMN,
            "has_valid_qualifying_lap",
            "reached_q2",
            "reached_q3",
        ]
    )
    inference["event_key"] = 202601

    _, _, quali_artifact = build_quali_runner_shared_forecast(
        history,
        inference,
        target_event_key=202601,
    )
    _, _, best_artifact = build_best_runner_shared_forecast(
        history,
        inference,
        target_event_key=202601,
    )

    assert quali_artifact["model_sha256"] == best_artifact["model_sha256"]
    assert (
        quali_artifact["joint_samples_sha256"]
        == best_artifact["joint_samples_sha256"]
    )
    assert quali_artifact["artifact_sha256"] == best_artifact["artifact_sha256"]


def test_joint_sampler_enforces_legal_stage_blocks_and_hurdles_change_order() -> None:
    drivers = [f"d{index:02d}" for index in range(20)]
    predictions = pd.DataFrame(
        {
            "driver_id": drivers,
            "latent_lap_location_seconds": [90.0] * 20,
            "lap_p05": [89.5] * 20,
            "lap_p50": [90.0] * 20,
            "lap_p90": [90.5] * 20,
            "valid_lap_probability": [0.0] + [1.0] * 19,
            "q2_given_valid_probability": [0.0] + [0.99] * 14 + [0.01] * 5,
            "q3_given_q2_probability": [0.0] + [0.99] * 9 + [0.01] * 10,
            "stage_q1_time_effect_seconds": [0.7] * 20,
            "stage_q2_time_effect_seconds": [0.3] * 20,
            "stage_q3_time_effect_seconds": [-0.1] * 20,
            "stage_q1_residual_sigma_seconds": [0.1] * 20,
            "stage_q2_residual_sigma_seconds": [0.1] * 20,
            "stage_q3_residual_sigma_seconds": [0.1] * 20,
        }
    )
    samples = sample_joint_qualifying_laps(predictions, samples=300, seed=11)
    summary = summarize_joint_lap_samples(samples).set_index("driver_id")

    assert samples.official_positions is not None
    assert all(
        sorted(row.tolist()) == list(range(1, 21)) for row in samples.official_positions
    )
    for stages, positions in zip(samples.deepest_stage, samples.official_positions):
        q3_positions = positions[stages == 3]
        q2_positions = positions[stages == 2]
        q1_positions = positions[stages == 1]
        invalid_positions = positions[stages == 0]
        assert q3_positions.max(initial=0) < q2_positions.min(initial=21)
        assert q2_positions.max(initial=0) < q1_positions.min(initial=21)
        assert q1_positions.max(initial=0) < invalid_positions.min(initial=21)
    assert summary.loc["d00", "expected_qualifying_position"] == pytest.approx(20.0)
    assert summary.loc["d01", "reaches_q3_probability_sampled"] > 0.95


def test_joint_sampler_uses_frozen_2026_elimination_rule_for_22_car_field() -> None:
    predictions = pd.DataFrame(
        {
            "driver_id": [f"d{index:02d}" for index in range(22)],
            "lap_p05": [89.5] * 22,
            "lap_p50": [90.0] * 22,
            "lap_p90": [90.5] * 22,
            "valid_lap_probability": [1.0] * 22,
            "q2_given_valid_probability": [0.75] * 22,
            "q3_given_q2_probability": [0.625] * 22,
            "stage_q1_residual_sigma_seconds": [0.1] * 22,
            "stage_q2_residual_sigma_seconds": [0.1] * 22,
            "stage_q3_residual_sigma_seconds": [0.1] * 22,
        }
    )
    samples = sample_joint_qualifying_laps(predictions, samples=1, seed=3)
    counts = pd.Series(samples.deepest_stage[0]).value_counts().to_dict()

    assert counts == {1: 6, 2: 6, 3: 10}


def test_shared_model_rejects_stage_outcomes_at_inference() -> None:
    model = fit_achievable_best_lap_model(_shared_history(), target_event_key=202601)
    inference = pd.DataFrame(
        {
            "event_key": [202601],
            "driver_id": ["d00"],
            "rehearsal_source": ["practice_3"],
            "rehearsal_lap_time_seconds": [90.0],
            "reached_q3": [1],
        }
    )
    with pytest.raises(ValueError, match="target/outcome"):
        model.predict(inference)


def test_best_lap_partitions_are_disjoint_and_reserve_same_season_calibration() -> None:
    priors = [
        pd.DataFrame({"event_key": [202401, 202402, 202501, 202502]})
    ]
    partitions = _locked_best_lap_partitions(
        priors,
        target_event_keys=(202601, 202602, 202603, 202604, 202605, 202606),
        target_year=2026,
    )

    assert partitions["development"] == (202401, 202402, 202501, 202502)
    assert partitions["selection"] == (202601, 202602)
    assert partitions["calibration"] == (202603, 202604)
    assert partitions["audit"] == (202605, 202606)
