from __future__ import annotations

import math

import pandas as pd
import pytest

from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN,
    fit_achievable_best_lap_model,
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
