from __future__ import annotations

import math

import pandas as pd

from packages.f1.features.assembly import _add_latest_qualifying_rehearsal_features
from packages.f1.models.training import _candidate_models, _preferred_qualifying_baseline_column
from packages.f1.orchestration.prediction import (
    _qualifying_feature_sets,
    _scrub_current_weekend_practice,
)


def test_standard_weekend_uses_fp3_and_flags_driver_imputation() -> None:
    frame = pd.DataFrame(
        {
            "event_key": [202601, 202601, 202601],
            "driver_id": ["a", "b", "c"],
            "fp3_rank": pd.Series([2, 1, pd.NA], dtype="Int64"),
            "fp3_delta": [0.2, 0.0, math.nan],
            "fp_mean_rank": pd.Series([2, 1, 3], dtype="Int64"),
            "fp_mean_delta": [0.3, 0.1, 0.8],
        }
    )

    result = _add_latest_qualifying_rehearsal_features(frame)

    assert result["latest_qualifying_rehearsal_source"].tolist() == ["practice_3"] * 3
    assert result["latest_qualifying_rehearsal_rank"].tolist() == [2.0, 1.0, 3.0]
    assert result["latest_qualifying_rehearsal_delta"].tolist() == [0.2, 0.0, 0.8]
    assert result["latest_qualifying_rehearsal_imputed"].tolist() == [False, False, True]
    assert result["latest_qualifying_rehearsal_coverage"].tolist() == [2.0 / 3.0] * 3


def test_sprint_qualifying_is_target_aligned_priority_for_whole_event() -> None:
    frame = pd.DataFrame(
        {
            "event_key": [202602, 202602, 202603, 202603],
            "driver_id": ["a", "b", "a", "b"],
            "sq_rank": [2.0, 1.0, math.nan, math.nan],
            "sq_delta": [0.4, 0.0, math.nan, math.nan],
            "fp3_rank": [1.0, 2.0, 2.0, 1.0],
            "fp3_delta": [0.0, 0.2, 0.3, 0.0],
            "fp_mean_rank": [1.0, 2.0, 2.0, 1.0],
            "fp_mean_delta": [0.0, 0.2, 0.3, 0.0],
        }
    )

    result = _add_latest_qualifying_rehearsal_features(frame)

    assert result.loc[result["event_key"] == 202602, "latest_qualifying_rehearsal_source"].eq(
        "sprint_qualifying"
    ).all()
    assert result.loc[result["event_key"] == 202602, "latest_qualifying_rehearsal_rank"].tolist() == [2.0, 1.0]
    assert result.loc[result["event_key"] == 202603, "latest_qualifying_rehearsal_source"].eq("practice_3").all()
    assert result.loc[result["event_key"] == 202603, "latest_qualifying_rehearsal_rank"].tolist() == [2.0, 1.0]


def test_rehearsal_signal_is_scrubbed_before_practice_and_is_default_baseline() -> None:
    feature_cols, fallback_cols = _qualifying_feature_sets()
    assert "latest_qualifying_rehearsal_rank" in feature_cols
    assert "latest_qualifying_rehearsal_rank" in fallback_cols
    assert _preferred_qualifying_baseline_column(
        ["fp_mean_rank", "event_pace_index", "latest_qualifying_rehearsal_rank"]
    ) == "latest_qualifying_rehearsal_rank"

    scrubbed = _scrub_current_weekend_practice(
        pd.DataFrame(
            {
                "latest_qualifying_rehearsal_rank": [1.0],
                "latest_qualifying_rehearsal_source": ["practice_3"],
                "driver_form_3_fp_mean_delta": [0.2],
            }
        )
    )
    assert scrubbed["latest_qualifying_rehearsal_rank"].isna().all()
    assert scrubbed["latest_qualifying_rehearsal_source"].isna().all()
    assert scrubbed["driver_form_3_fp_mean_delta"].notna().all()


def test_baseline_only_family_does_not_silently_enable_ml_candidates() -> None:
    candidates, _ = _candidate_models(
        enable_dl_candidates=False,
        compare_families=["baseline"],
        dl_arch="mlp_tabular_v1",
        dl_hyperparams={},
        dl_seed=42,
        dl_device="cpu",
        requested_model="auto",
        notes=[],
    )

    assert candidates == []
