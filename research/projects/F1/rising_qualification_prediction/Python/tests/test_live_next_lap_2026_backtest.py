from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run_live_next_lap_2026_backtest import (
    _paired_bootstrap,
    _select_weight,
    _validated_matched_rows,
)


def _trace() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_key": [202601] * 7,
            "driver_id": ["1", "1", "1", "1", "2", "2", "2"],
            "lap_number": [2, 3, 4, 5, 2, 3, 4],
            "timestamp": [200.0, 300.0, 400.0, 500.0, 201.0, 301.0, 401.0],
            "timestamp_known": [True] * 7,
            "baseline_information_order": ["global_event_time"] * 7,
            "baseline_evidence_max_timestamp": [190.0, 290.0, 390.0, 490.0, 190.0, 290.0, 390.0],
            "baseline_evidence_row_count": [10, 20, 30, 40, 10, 20, 30],
            "eval_included": [True, True, False, True, True, True, True],
            "assim_laps_driver": [2, 3, 3, 4, 2, 3, 4],
            "lap_time_seconds": [99.0, 100.0, 160.0, 101.0, 100.0, 101.0, 102.0],
            "next_lap_mean": [99.5, 100.5, 120.0, 101.5, 100.5, 101.5, 102.5],
            "next_lap_mean_ssm": [99.5, 100.5, 120.0, 101.5, 100.5, 101.5, 102.5],
            "next_lap_mean_naive": [99.0, 100.0, 100.0, 101.0, 100.0, 101.0, 102.0],
            "next_lap_ssm_weight": [1.0] * 7,
        }
    )


def test_matched_rows_fail_closed_for_missing_comparator_and_noncausal_order() -> None:
    valid = _validated_matched_rows(_trace(), event_key=202601, warmup_laps=3)
    assert len(valid) == 2
    assert valid.attrs["eligible_issuance_rows"] == 4
    assert valid.attrs["issuances_without_next_eligible_target"] == 2
    driver_one = valid.loc[valid["driver_id"] == "1"].iloc[0]
    assert driver_one["issued_after_lap_number"] == 3
    assert driver_one["target_lap_number"] == 5
    assert driver_one["skipped_nonrepresentative_laps"] == 1
    assert driver_one["forecast_ssm_seconds"] == 100.5
    assert driver_one["lap_time_seconds"] == 101.0

    missing = _trace()
    missing.loc[1, "next_lap_mean_naive"] = np.nan
    with pytest.raises(ValueError, match="matched-population"):
        _validated_matched_rows(missing, event_key=202601, warmup_laps=3)

    noncausal = _trace()
    noncausal.loc[0, "baseline_information_order"] = "lap_number_fallback_not_global_time"
    with pytest.raises(ValueError, match="noncausal baseline"):
        _validated_matched_rows(noncausal, event_key=202601, warmup_laps=3)

    future_evidence = _trace()
    future_evidence.loc[1, "baseline_evidence_max_timestamp"] = 300.0
    with pytest.raises(ValueError, match="future baseline evidence"):
        _validated_matched_rows(future_evidence, event_key=202601, warmup_laps=3)


def test_matching_ignores_same_row_one_step_nowcasts() -> None:
    trace = _trace()
    trace["one_step_pred_mean"] = -999.0
    trace["one_step_pred_mean_ssm"] = -999.0
    trace["one_step_pred_mean_naive"] = -999.0
    rows = _validated_matched_rows(trace, event_key=202601, warmup_laps=3)
    assert (rows["forecast_ssm_seconds"] > 0.0).all()
    assert (rows["forecast_naive_seconds"] > 0.0).all()


def test_weight_selection_uses_prior_event_mean_not_target_or_pooled_rows() -> None:
    # One large event prefers naive, one tiny event prefers SSM.  Equal event
    # weighting therefore selects naive despite the pooled rows preferring SSM.
    large = pd.DataFrame(
        {
            "lap_time_seconds": [0.0] * 100,
            "forecast_ssm_seconds": [0.0] * 100,
            "forecast_naive_seconds": [1.0] * 100,
        }
    )
    tiny = pd.DataFrame(
        {
            "lap_time_seconds": [0.0],
            "forecast_ssm_seconds": [3.0],
            "forecast_naive_seconds": [0.0],
        }
    )
    selected, scores = _select_weight(
        [large, tiny],
        weight_grid=(0.0, 1.0),
        cold_start_weight=0.0,
    )
    assert selected == 0.0
    assert scores[0.0] == 0.5
    assert scores[1.0] == 1.5

    cold, no_scores = _select_weight(
        [],
        weight_grid=(0.0, 0.5, 1.0),
        cold_start_weight=0.5,
    )
    assert cold == 0.5
    assert no_scores == {}


def test_paired_event_bootstrap_is_fixed_seed_deterministic() -> None:
    left = _paired_bootstrap([1.0, 2.0, 3.0], [1.5, 2.5, 3.5], samples=1000, seed=7)
    right = _paired_bootstrap([1.0, 2.0, 3.0], [1.5, 2.5, 3.5], samples=1000, seed=7)
    assert left == right
    assert left["mean_delta_seconds"] == -0.5
    assert left["ci95_seconds"] == [-0.5, -0.5]
    assert left["bootstrap_probability_of_improvement"] == 1.0
