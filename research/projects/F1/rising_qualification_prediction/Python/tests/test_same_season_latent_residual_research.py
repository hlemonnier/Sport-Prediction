from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from packages.f1.models.ultimate_lap_time.achievable import ACTUAL_LAP_COLUMN
from run_same_season_latent_residual_research import (
    PAIRWISE_MODEL_ID,
    QUALIFYING_TARGET_COLUMN,
    RESIDUAL_FEATURE_ALLOWLIST,
    ResearchEvent,
    _event_balanced_row_weights,
    _fit_numeric_design,
    _stable_rank,
    run_event_stream,
)


def test_stable_rank_ties_use_provider_order_not_driver_identity() -> None:
    ranks = _stable_rank([1.0, 1.0, np.nan], ["ZED", "ALP", "MID"])

    assert ranks.tolist() == [1, 2, 3]


def _stable_positions(values: np.ndarray, drivers: list[str]) -> np.ndarray:
    order = np.lexsort((np.asarray(drivers), np.asarray(values, dtype=float)))
    positions = np.empty(len(values), dtype=int)
    positions[order] = np.arange(1, len(values) + 1)
    return positions


def _synthetic_events(
    *,
    count: int = 7,
    altered_final_target: bool = False,
    altered_target_round: int | None = None,
) -> tuple[tuple[ResearchEvent, ...], dict[int, pd.DataFrame]]:
    drivers = [f"d{index}" for index in range(6)]
    events: list[ResearchEvent] = []
    targets: dict[int, pd.DataFrame] = {}
    for round_number in range(1, count + 1):
        event_key = 202600 + round_number
        driver_index = np.arange(len(drivers), dtype=float)
        source = "sprint_qualifying" if round_number in {3, 6} else "practice_3"
        raw = 79.0 + 0.7 * round_number + 0.22 * driver_index
        shape = ((driver_index + round_number) % 4.0) - 1.5
        valid_minus_potential = 0.18 * shape
        missing_spread = 0.04 + 0.01 * driver_index
        if round_number % 2 == 0:
            missing_spread[1] = np.nan
        inference = pd.DataFrame(
            {
                "event_key": event_key,
                "driver_id": drivers,
                "rehearsal_source": source,
                "valid_clean_best_seconds": raw,
                "quality_aware_anchor_seconds": raw + 0.02,
                "field_relative_anchor_seconds": raw - float(raw.min()),
                "teammate_relative_anchor_seconds": np.tile([-0.11, 0.11], 3),
                "anchor_uncertainty_seconds": 0.12 + 0.01 * driver_index,
                "valid_minus_potential_seconds": valid_minus_potential,
                "best_two_spread_seconds": missing_spread,
                "deleted_potential_lap_count": (driver_index % 3.0 == 0.0).astype(float),
                "best_lap_session_progress": 0.55 + 0.04 * driver_index,
                "evidence_coverage_rate": 0.95 - 0.02 * (driver_index % 2.0),
            }
        )
        source_shift = -0.45 if source == "practice_3" else -0.18
        actual_lap = (
            raw
            + source_shift
            + 1.35 * valid_minus_potential
            + 0.06 * np.sin(driver_index + round_number)
        )
        should_alter = (
            altered_target_round is not None
            and round_number == int(altered_target_round)
        ) or (altered_final_target and round_number == count)
        if should_alter:
            actual_lap = actual_lap[::-1] + 4.0
        actual_position = _stable_positions(actual_lap, drivers)
        target = pd.DataFrame(
            {
                "driver_id": drivers,
                ACTUAL_LAP_COLUMN: actual_lap,
                QUALIFYING_TARGET_COLUMN: actual_position,
            }
        )
        events.append(
            ResearchEvent(
                event_key=event_key,
                inference=inference,
                event_info={"round": round_number, "event_name": f"synthetic-{round_number}"},
            )
        )
        targets[event_key] = target
    return tuple(events), targets


def _run_synthetic(
    *,
    altered_final_target: bool = False,
    altered_target_round: int | None = None,
    count: int = 7,
    audit_event_count: int = 1,
    prequential_audit_diagnostic: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    events, targets = _synthetic_events(
        count=count,
        altered_final_target=altered_final_target,
        altered_target_round=altered_target_round,
    )
    target_reads: list[dict[str, object]] = []

    def load_target(
        event: ResearchEvent,
        artifact: dict[str, object],
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        assert re.fullmatch(r"[0-9a-f]{64}", str(artifact["artifact_sha256"]))
        assert artifact["event_key"] == event.event_key
        assert artifact["target_columns_present_at_freeze"] == []
        assert ACTUAL_LAP_COLUMN not in event.inference.columns
        assert QUALIFYING_TARGET_COLUMN not in event.inference.columns
        target_reads.append(
            {
                "event_key": event.event_key,
                "training_event_keys": list(artifact["training_event_keys"]),
                "candidate_forecasts_sha256": artifact[
                    "candidate_forecasts_sha256"
                ],
                "selected_models": artifact[
                    "selected_models_frozen_before_target"
                ],
                "audit_forecast_block_sha256": artifact.get(
                    "audit_forecast_block_sha256"
                ),
                "audit_forecast_block_event_keys": artifact.get(
                    "audit_forecast_block_event_keys"
                ),
            }
        )
        return targets[event.event_key].copy(), {"loader": "synthetic_after_freeze"}

    payload = run_event_stream(
        events,
        target_loader=load_target,
        year=2026,
        audit_event_count=audit_event_count,
        seed=123,
        code_paths=(Path(__file__).resolve(),),
        prequential_audit_diagnostic=prequential_audit_diagnostic,
    )
    return payload, target_reads


def test_target_reads_follow_complete_frozen_expanding_forecasts() -> None:
    payload, target_reads = _run_synthetic()

    assert [row["event_key"] for row in target_reads] == list(range(202601, 202608))
    for offset, row in enumerate(target_reads):
        assert row["training_event_keys"] == list(range(202601, 202601 + offset))
    assert all(row["selected_models"] is None for row in target_reads[:-1])
    assert target_reads[-1]["selected_models"]["audit_outcomes_used"] is False

    protocol = payload["protocol"]
    assert protocol["prior_season_training_rows"] == 0
    assert protocol["development_event_keys"] == [202601, 202602, 202603, 202604]
    assert protocol["selection_event_keys"] == [202605, 202606]
    assert protocol["final_untouched_audit_event_keys"] == [202607]
    assert protocol["audit_training_mode"] == (
        "within_run_frozen_block_postdevelopment_replay"
    )
    assert protocol["audit_fitted_state_frozen_before_first_audit_target"] is True
    assert protocol["all_audit_forecasts_frozen_before_first_audit_target"] is True
    assert protocol["audit_outcomes_used_for_later_audit_fits"] is False
    assert protocol["audit_evidence_eligible_for_formal_promotion"] is False
    assert protocol["prospective_development_evidence"] is False
    assert protocol["within_run_target_isolation"] is True
    assert protocol["event_block_is_training_weight_and_selection_unit"] is True
    assert target_reads[-1]["audit_forecast_block_event_keys"] == [202607]
    assert payload["selection"]["audit_outcomes_used"] is False
    assert payload["per_round_prediction_vs_reality"]
    assert payload["per_round_metrics"]
    assert re.fullmatch(r"[0-9a-f]{64}", payload["result_sha256"])


def test_selection_and_audit_forecast_do_not_depend_on_final_target() -> None:
    original, original_reads = _run_synthetic()
    altered, altered_reads = _run_synthetic(altered_final_target=True)

    assert original["selection"]["selection_sha256"] == altered["selection"][
        "selection_sha256"
    ]
    assert original["selection"]["best_lap_selected_model_id"] == altered[
        "selection"
    ]["best_lap_selected_model_id"]
    assert original["selection"]["qualifying_selected_model_id"] == altered[
        "selection"
    ]["qualifying_selected_model_id"]
    assert original_reads[-1]["candidate_forecasts_sha256"] == altered_reads[-1][
        "candidate_forecasts_sha256"
    ]
    assert original["audit_metrics"] != altered["audit_metrics"]

    pairwise_failures = [
        row
        for row in original["failed_candidates"]
        if row["model_id"] == PAIRWISE_MODEL_ID
    ]
    assert [row["event_key"] for row in pairwise_failures] == [
        202601,
        202602,
        202603,
        202604,
    ]
    pairwise_predictions = [
        row
        for row in original["per_round_prediction_vs_reality"]
        if row["model_id"] == PAIRWISE_MODEL_ID
    ]
    assert min(row["event_key"] for row in pairwise_predictions) == 202605
    for trace_name in ("best_lap_trace", "qualifying_trace"):
        for row in original["selection"][trace_name]:
            if row["eligible"]:
                assert row["selection_event_keys"] == [202605, 202606]


def test_multi_event_audit_forecasts_are_frozen_before_first_audit_target() -> None:
    original, original_reads = _run_synthetic(count=9, audit_event_count=3)
    altered, altered_reads = _run_synthetic(
        count=9,
        audit_event_count=3,
        altered_target_round=7,
    )

    original_audit_reads = original_reads[-3:]
    altered_audit_reads = altered_reads[-3:]
    assert [row["event_key"] for row in original_audit_reads] == [
        202607,
        202608,
        202609,
    ]
    assert all(
        row["training_event_keys"] == list(range(202601, 202607))
        for row in original_audit_reads
    )
    assert all(
        row["audit_forecast_block_event_keys"] == [202607, 202608, 202609]
        for row in original_audit_reads
    )
    assert len(
        {row["audit_forecast_block_sha256"] for row in original_audit_reads}
    ) == 1
    assert [
        row["candidate_forecasts_sha256"] for row in original_audit_reads
    ] == [row["candidate_forecasts_sha256"] for row in altered_audit_reads]
    assert [
        row["audit_forecast_block_sha256"] for row in original_audit_reads
    ] == [row["audit_forecast_block_sha256"] for row in altered_audit_reads]
    assert original["selection"]["selection_sha256"] == altered["selection"][
        "selection_sha256"
    ]
    assert original["audit_metrics"] != altered["audit_metrics"]

    protocol = original["protocol"]
    assert protocol["final_untouched_audit_event_keys"] == [
        202607,
        202608,
        202609,
    ]
    assert protocol["audit_outcomes_used_for_later_audit_fits"] is False
    block = original["audit_forecast_block"]
    assert block["all_forecasts_materialized_before_first_audit_target"] is True
    assert block["formal_promotion_evidence"] is False
    assert block["prospective_development_evidence"] is False
    assert block["training_event_keys"] == list(range(202601, 202607))
    assert set(block["selected_fitted_state_manifest_sha256"]) == {
        original["selection"]["best_lap_selected_model_id"],
        original["selection"]["qualifying_selected_model_id"],
    }


def test_prequential_audit_is_explicit_opt_in_and_not_promotion_evidence() -> None:
    payload, target_reads = _run_synthetic(
        count=9,
        audit_event_count=3,
        prequential_audit_diagnostic=True,
    )

    audit_reads = target_reads[-3:]
    assert [row["training_event_keys"] for row in audit_reads] == [
        list(range(202601, 202607)),
        list(range(202601, 202608)),
        list(range(202601, 202609)),
    ]
    assert all(row["audit_forecast_block_sha256"] is None for row in audit_reads)
    protocol = payload["protocol"]
    assert protocol["audit_training_mode"] == (
        "opt_in_prequential_expanding_diagnostic"
    )
    assert protocol["final_untouched_audit_event_keys"] == []
    assert protocol["prequential_diagnostic_event_keys"] == [
        202607,
        202608,
        202609,
    ]
    assert protocol["audit_outcomes_used_for_later_audit_fits"] is True
    assert protocol["audit_evidence_eligible_for_formal_promotion"] is False
    assert payload["audit_forecast_block"]["formal_promotion_evidence"] is False


def test_event_weights_and_missingness_are_explicit_and_target_leaks_fail_closed() -> None:
    weights_frame = pd.DataFrame(
        {
            "event_key": [202601, 202601, 202602, 202602, 202602, 202602],
            **{
                column: [0.0, 1.0, np.nan, 2.0, 3.0, 4.0]
                for column in RESIDUAL_FEATURE_ALLOWLIST
            },
        }
    )
    weights = _event_balanced_row_weights(weights_frame)
    assert weights[:2].sum() == pytest.approx(1.0)
    assert weights[2:].sum() == pytest.approx(1.0)
    design, transformed = _fit_numeric_design(weights_frame, weights)
    assert transformed.shape == (
        len(weights_frame),
        2 * len(RESIDUAL_FEATURE_ALLOWLIST),
    )
    assert np.isfinite(transformed).all()
    assert all(
        f"{column}__missing" in design.feature_names
        for column in RESIDUAL_FEATURE_ALLOWLIST
    )

    events, targets = _synthetic_events()
    leaked = replace(
        events[0],
        inference=events[0].inference.assign(
            **{ACTUAL_LAP_COLUMN: targets[events[0].event_key][ACTUAL_LAP_COLUMN]}
        ),
    )
    target_loader_called = False

    def forbidden_loader(
        _event: ResearchEvent, _artifact: dict[str, object]
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        nonlocal target_loader_called
        target_loader_called = True
        return pd.DataFrame(), {}

    with pytest.raises(ValueError, match="contains evaluation targets"):
        run_event_stream(
            (leaked, *events[1:]),
            target_loader=forbidden_loader,
            code_paths=(Path(__file__).resolve(),),
        )
    assert target_loader_called is False

    prior_season = replace(events[0], event_key=202501)
    with pytest.raises(ValueError, match="prior/future-season"):
        run_event_stream(
            (prior_season, *events[1:]),
            target_loader=forbidden_loader,
            code_paths=(Path(__file__).resolve(),),
        )
