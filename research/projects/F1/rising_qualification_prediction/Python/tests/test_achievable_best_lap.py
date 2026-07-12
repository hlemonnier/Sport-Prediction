from __future__ import annotations

import math

import pandas as pd
import pytest

from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN,
    decompose_event_fastest_and_driver_gap,
    fit_achievable_best_lap_model,
    robust_huber_location,
    sample_joint_qualifying_laps,
    summarize_joint_lap_samples,
)
from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
)
from run_best_estimated_lap_2026_backtest import run_backtest


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
    assert result.loc[0, "interval_status"] == "calibrated_minimum_event_count_met"
    assert result.loc[1, "interval_status"] == "diagnostic_underpowered"


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
    assert result.loc[0, "stage_probability_status"] == "event_balanced_beta_binomial"
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
    assert result.loc[0, "interval_method"] == "rolling_equal_event_weight_conformal"
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
        }
    )

    samples = sample_joint_qualifying_laps(predictions, samples=2_000, seed=7)
    summary = summarize_joint_lap_samples(samples).set_index("driver_id")

    assert samples.lap_seconds.shape == (2_000, 3)
    assert summary.loc["fast", "fastest_driver_probability"] > 0.99
    assert summary.loc["invalid", "valid_lap_probability_sampled"] == 0.0
    assert summary["fastest_driver_probability"].sum() == pytest.approx(1.0)
