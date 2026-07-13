from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from packages.f1.features.qualifying_lap import build_quality_aware_rehearsal_features
from packages.f1.models.pre_quali.classification import (
    STAGE_PROBABILITY_STATUS,
    StageProbabilityConfig,
    fit_qualifying_stage_probability_model,
    quality_aware_stage_probability_config,
)
from packages.f1.models.pre_quali.evaluate import walk_forward_pairwise_qualifying
from packages.f1.models.pre_quali.pairwise import (
    UNCALIBRATED_JOINT_SAMPLES,
    PairwiseRankerConfig,
    build_event_pairwise_dataset,
    fit_pairwise_qualifying_ranker,
    quality_aware_pairwise_config,
)
from packages.f1.models.pre_quali.selection import (
    FrozenSelectorConfig,
    QualifyingModelEvidence,
    select_frozen_qualifying_model,
)
from run_qualifying_pairwise_challenger_backtest import (
    _event_frame,
    _locked_event_partitions,
    _pre_qualifying_roster,
    _qualifying_contract_gates,
)
def _rank_history(*, event_keys: tuple[int, ...] = (202601, 202602, 202603, 202604)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    drivers = ("a", "b", "c", "d", "e")
    for event_offset, event_key in enumerate(event_keys):
        sprint = event_offset % 2 == 1
        source = "sprint_qualifying" if sprint else "practice_3"
        for index, driver in enumerate(drivers):
            # Pace is the dominant causal signal. Small event shifts ensure the
            # ranker learns comparisons rather than absolute circuit seconds.
            pace = float(index) + (0.03 * event_offset)
            rows.append(
                {
                    "event_key": event_key,
                    "driver_id": driver,
                    "latest_qualifying_rehearsal_rank": [2, 1, 4, 3, 5][index],
                    "latest_qualifying_rehearsal_source": source,
                    "quality_anchor_delta": pace,
                    "valid_push_lap_count": float(5 - index),
                    "qualy_position": index + 1,
                }
            )
    return pd.DataFrame(rows)


def _rank_config(**overrides: object) -> PairwiseRankerConfig:
    values: dict[str, object] = {
        "feature_columns": ("quality_anchor_delta", "valid_push_lap_count"),
        "minimum_training_events": 3,
        "max_movement": 1,
    }
    values.update(overrides)
    return PairwiseRankerConfig(**values)


def _stage_history() -> pd.DataFrame:
    history = _rank_history()
    position = pd.to_numeric(history["qualy_position"], errors="raise")
    # Vary validity independently from pace so all three classifiers see both
    # outcomes. Q2/Q3 labels remain logically nested.
    invalid = ((history["event_key"] + history.groupby("event_key").cumcount()) % 7).eq(0)
    history["has_valid_qualifying_lap"] = (~invalid).astype(int)
    history["reached_q2"] = ((position <= 3) & ~invalid).astype(int)
    history["reached_q3"] = ((position == 1) & ~invalid).astype(int)
    return history


def test_pair_dataset_contains_only_within_event_pairs_and_equal_event_weight() -> None:
    history = _rank_history(event_keys=(202601, 202602, 202603))
    # One tie removes exactly two oriented rows in the first event.
    history.loc[
        (history["event_key"] == 202601) & history["driver_id"].isin(["a", "b"]),
        "qualy_position",
    ] = 1
    dataset = build_event_pairwise_dataset(history, config=_rank_config())

    row_event = history.set_index("driver_id")["event_key"].to_dict()
    # Driver ids repeat between events, so the dataset's explicit event id is
    # the boundary of truth; no pair may carry a different event label.
    assert set(dataset.event_keys) == {202601, 202602, 202603}
    assert dataset.pair_counts_by_event[202601] == 18
    assert dataset.pair_counts_by_event[202602] == 20
    assert dataset.pair_counts_by_event[202603] == 20
    _ = row_event
    event_array = np.asarray(dataset.event_keys)
    for event_key in sorted(set(dataset.event_keys)):
        assert dataset.sample_weight[event_array == event_key].sum() == pytest.approx(
            dataset.sample_weight[event_array == 202602].sum()
        )
    assert set(dataset.labels.tolist()) == {0, 1}


def test_quality_aware_feature_output_satisfies_canonical_model_adapters() -> None:
    laps = pd.DataFrame(
        {
            "event_key": [202601, 202601],
            "driver_id": ["a", "b"],
            "team_id": ["red", "blue"],
            "session": ["FP3", "FP3"],
            "lap_time_seconds": [90.0, 90.2],
            "is_official_classified": [True, True],
            "is_accurate": [True, True],
            "is_deleted": [False, False],
        }
    )
    features = build_quality_aware_rehearsal_features(laps)
    rank_config = quality_aware_pairwise_config(minimum_training_events=2)
    stage_config = quality_aware_stage_probability_config(minimum_training_events=2)

    assert set(rank_config.required_inference_columns).issubset(features.columns)
    assert set(stage_config.required_inference_columns).issubset(features.columns)


def test_pairwise_forecast_is_a_deterministic_capped_permutation_with_normalized_marginals() -> None:
    history = _rank_history(event_keys=(202601, 202602, 202603))
    current = _rank_history(event_keys=(202604,)).drop(columns="qualy_position")
    model = fit_pairwise_qualifying_ranker(
        history,
        config=_rank_config(),
        target_event_key=202604,
    )

    forecast = model.predict_event(current, samples=400, temperature=0.7, seed=9)
    point = forecast.point_order
    predicted = point["predicted_qualifying_position"].astype(int)
    assert sorted(predicted.tolist()) == [1, 2, 3, 4, 5]
    assert point["movement_from_baseline"].abs().le(1).all()
    assert tuple(model.required_inference_columns) == _rank_config().required_inference_columns

    marginal_columns = [f"p_position_{position}" for position in range(1, 6)]
    marginals = forecast.position_marginals
    assert np.allclose(marginals[marginal_columns].sum(axis=1), 1.0)
    assert np.allclose(marginals[marginal_columns].sum(axis=0), 1.0)
    assert set(marginals["probability_calibration_status"]) == {UNCALIBRATED_JOINT_SAMPLES}
    assert not marginals["position_marginals_calibrated"].any()
    assert forecast.probability_calibration_status == UNCALIBRATED_JOINT_SAMPLES
    assert forecast.position_marginals_calibrated is False

    repeated = model.predict_event(current.sample(frac=1.0, random_state=18), samples=400, temperature=0.7, seed=9)
    first_by_driver = point.set_index("driver_id")["predicted_qualifying_position"].sort_index()
    repeated_by_driver = repeated.point_order.set_index("driver_id")[
        "predicted_qualifying_position"
    ].sort_index()
    assert first_by_driver.equals(repeated_by_driver)


def test_tied_baseline_is_resolved_stably_and_outcome_features_are_rejected() -> None:
    history = _rank_history(event_keys=(202601, 202602, 202603))
    current = _rank_history(event_keys=(202604,)).drop(columns="qualy_position")
    current["latest_qualifying_rehearsal_rank"] = 1.0
    model = fit_pairwise_qualifying_ranker(
        history,
        config=_rank_config(max_movement=0),
        target_event_key=202604,
    )
    point = model.predict_event(current, samples=5).point_order.set_index("driver_id")
    assert point.loc["a", "predicted_qualifying_position"] == 1
    assert point.loc["e", "predicted_qualifying_position"] == 5

    leaky = _rank_config(feature_columns=("quality_anchor_delta", "qualy_position"))
    with pytest.raises(ValueError, match="forbidden"):
        fit_pairwise_qualifying_ranker(history, config=leaky)


def test_ranker_rejects_target_event_leakage_and_multi_event_inference() -> None:
    history = _rank_history()
    with pytest.raises(ValueError, match="strictly earlier"):
        fit_pairwise_qualifying_ranker(
            history,
            config=_rank_config(),
            target_event_key=202604,
        )
    model = fit_pairwise_qualifying_ranker(
        history.loc[history["event_key"] < 202604],
        config=_rank_config(),
        target_event_key=202604,
    )
    multi_event = _rank_history(event_keys=(202604, 202605)).drop(columns="qualy_position")
    with pytest.raises(ValueError, match="exactly one"):
        model.predict_event(multi_event)


def test_stage_models_are_separate_conditional_monotone_and_source_aware() -> None:
    history = _stage_history()
    config = StageProbabilityConfig(
        feature_columns=("quality_anchor_delta", "valid_push_lap_count"),
        minimum_training_events=3,
    )
    model = fit_qualifying_stage_probability_model(
        history.loc[history["event_key"] < 202604],
        config=config,
        target_event_key=202604,
    )
    current = history.loc[history["event_key"] == 202604].drop(columns=list(config.label_columns))
    result = model.predict_event(current)

    assert result["p_valid_qualifying_lap"].between(0.0, 1.0).all()
    assert result["p_reaches_q2"].le(result["p_valid_qualifying_lap"] + 1e-12).all()
    assert result["p_reaches_q3"].le(result["p_reaches_q2"] + 1e-12).all()
    assert np.allclose(
        result["p_reaches_q3"],
        result["p_valid_qualifying_lap"]
        * result["p_q2_given_valid"]
        * result["p_q3_given_q2"],
    )
    assert set(result["probability_calibration_status"]) == {STAGE_PROBABILITY_STATUS}
    assert model.q2_given_valid_model.observed_rows < model.valid_model.observed_rows
    assert model.q3_given_q2_model.observed_rows < model.q2_given_valid_model.observed_rows
    assert "quality_anchor_delta__x_sprint_rehearsal" in model.design.names


def test_stage_hurdles_are_driver_conditioned_after_partial_pooling() -> None:
    rows: list[dict[str, object]] = []
    outcomes = {
        "always_q3": (1, 1, 1),
        "q2_only": (1, 1, 0),
        "q1_only": (1, 0, 0),
        "invalid": (0, 0, 0),
    }
    for event in range(202501, 202507):
        for driver, (valid, q2, q3) in outcomes.items():
            rows.append(
                {
                    "event_key": event,
                    "driver_id": driver,
                    "latest_qualifying_rehearsal_source": "practice_3",
                    "same_feature": 1.0,
                    "has_valid_qualifying_lap": valid,
                    "reached_q2": q2,
                    "reached_q3": q3,
                }
            )
    history = pd.DataFrame(rows)
    config = StageProbabilityConfig(feature_columns=("same_feature",), minimum_training_events=3)
    model = fit_qualifying_stage_probability_model(
        history,
        config=config,
        target_event_key=202601,
    )
    inference = history.loc[history["event_key"].eq(202506)].drop(
        columns=list(config.label_columns)
    )
    inference["event_key"] = 202601
    predicted = model.predict_event(inference).set_index("driver_id")

    assert predicted.loc["always_q3", "p_valid_qualifying_lap"] > predicted.loc[
        "invalid", "p_valid_qualifying_lap"
    ]
    assert predicted.loc["always_q3", "p_q2_given_valid"] > predicted.loc[
        "q1_only", "p_q2_given_valid"
    ]
    assert predicted.loc["always_q3", "p_q3_given_q2"] > predicted.loc[
        "q2_only", "p_q3_given_q2"
    ]


def test_pre_q_roster_and_partitions_fail_closed_without_target_results() -> None:
    fp3 = pd.DataFrame(
        {"Driver": ["A", "B"], "Team": ["red", "blue"], "LapTime": [90.0, 91.0]}
    )
    fp2 = pd.DataFrame(
        {"Driver": ["C"], "Team": ["green"], "LapTime": [92.0]}
    )
    roster = _pre_qualifying_roster(fp3, fp2)
    assert roster["driver_id"].tolist() == ["A", "B", "C"]

    partitions = _locked_event_partitions(
        (
            202401,
            202402,
            202501,
            202502,
            202601,
            202602,
            202603,
            202604,
            202605,
            202606,
        ),
        audit_year=2026,
    )
    assert partitions["selection"] == (202501, 202502)
    assert partitions["point_fit"] == (202601, 202602)
    assert partitions["calibration"] == (202603, 202604)
    assert partitions["audit"] == (202605, 202606)


def _evidence(
    event_keys: tuple[int, ...],
    *,
    baseline_mae: float,
    challenger_mae: float,
    challenger_gates_passed: bool = True,
) -> tuple[QualifyingModelEvidence, QualifyingModelEvidence]:
    return (
        QualifyingModelEvidence("base", baseline_mae, event_keys),
        QualifyingModelEvidence(
            "challenger",
            challenger_mae,
            event_keys,
            promotion_gates_passed=challenger_gates_passed,
        ),
    )


def test_selector_requires_matched_minimum_evidence_and_freezes_switches() -> None:
    config = FrozenSelectorConfig(
        baseline_model_id="base",
        challenger_model_id="challenger",
        minimum_evidence_events=8,
        freeze_for_new_events=4,
        minimum_mae_improvement=0.15,
    )
    seven = tuple(range(1, 8))
    insufficient = select_frozen_qualifying_model(
        _evidence(seven, baseline_mae=2.0, challenger_mae=1.0),
        config=config,
    )
    assert insufficient.selected_model_id == "base"
    assert "insufficient" in insufficient.decision

    eight = tuple(range(1, 9))
    selected = select_frozen_qualifying_model(
        _evidence(eight, baseline_mae=2.0, challenger_mae=1.7),
        config=config,
    )
    assert selected.selected_model_id == "challenger"

    gates_failed = select_frozen_qualifying_model(
        _evidence(
            eight,
            baseline_mae=2.0,
            challenger_mae=1.0,
            challenger_gates_passed=False,
        ),
        config=config,
    )
    assert gates_failed.selected_model_id == "base"
    assert "promotion_gates_failed" in gates_failed.decision

    held = select_frozen_qualifying_model(
        _evidence(tuple(range(1, 10)), baseline_mae=1.8, challenger_mae=1.9),
        config=config,
        previous_state=selected,
    )
    assert held.selected_model_id == "challenger"
    assert "freeze" in held.decision

    switched = select_frozen_qualifying_model(
        _evidence(tuple(range(1, 13)), baseline_mae=1.8, challenger_mae=1.9),
        config=config,
        previous_state=selected,
    )
    assert switched.selected_model_id == "base"

    mismatched = list(_evidence(eight, baseline_mae=2.0, challenger_mae=1.7))
    mismatched[1] = QualifyingModelEvidence("challenger", 1.7, tuple(range(2, 10)))
    with pytest.raises(ValueError, match="identical event blocks"):
        select_frozen_qualifying_model(mismatched, config=config)


def test_qualifying_contract_gates_require_complete_legal_field_and_fail_close_probabilities() -> None:
    predictions = pd.DataFrame(
        {
            "event_key": [202601, 202601, 202601],
            "driver_id": ["a", "b", "c"],
            "predicted_qualifying_position": [1, 2, 3],
            "p_position_1": [1.0, 0.0, 0.0],
            "p_position_2": [0.0, 1.0, 0.0],
            "p_position_3": [0.0, 0.0, 1.0],
            "position_marginals_calibrated": [False, False, False],
            "probability_calibration_status": [
                "uncalibrated_joint_latent_samples"
            ]
            * 3,
        }
    )

    gates = _qualifying_contract_gates(
        predictions,
        event_info={
            202601: {
                "field_size": 3,
                "official_target_driver_ids": ["a", "b", "c"],
            }
        },
        event_keys=(202601,),
    )

    assert gates["pre_q_entrant_coverage_is_100_percent"]
    assert gates["every_event_is_legal_full_field_permutation"]
    assert not gates["position_probabilities_calibrated"]
    assert not gates["position_probability_outputs_promoted"]
    assert gates["uncalibrated_probability_outputs_fail_closed"]
    assert gates["point_contract_gates_passed"]

    target_mismatch = _qualifying_contract_gates(
        predictions,
        event_info={
            202601: {
                "field_size": 3,
                "official_target_driver_ids": ["a", "b", "substitute"],
            }
        },
        event_keys=(202601,),
    )
    assert not target_mismatch["pre_q_entrant_coverage_is_100_percent"]
    assert not target_mismatch["point_contract_gates_passed"]


def test_event_frame_uses_latest_rehearsal_roster_and_does_not_readd_fp1_reserve(
    tmp_path,
) -> None:
    event_dir = tmp_path / "round_01_test"
    event_dir.mkdir()
    pd.DataFrame(
        {
            "Driver": ["A"],
            "Team": ["Old Team"],
            "LapTime": [92.0],
            "Deleted": [False],
            "IsAccurate": [True],
        }
    ).to_csv(event_dir / "p1_laps.csv", index=False)
    pd.DataFrame(
        {
            "Abbreviation": ["A", "SUB"],
            "TeamName": ["Old Team", "Sub Team"],
        }
    ).to_csv(event_dir / "p1_results.csv", index=False)
    pd.DataFrame(
        {
            "Driver": ["A"],
            "Team": ["New Team"],
            "LapTime": [90.0],
            "Deleted": [False],
            "IsAccurate": [True],
        }
    ).to_csv(event_dir / "fp3_laps.csv", index=False)
    pd.DataFrame(
        {"Abbreviation": ["A"], "TeamName": ["New Team"]}
    ).to_csv(event_dir / "fp3_results.csv", index=False)
    pd.DataFrame(
        {
            "Abbreviation": ["A", "SUB"],
            "TeamName": ["New Team", "Sub Team"],
            "Position": [1, 2],
            "Q1": ["0:59.0", "1:00.0"],
            "Q2": ["0:58.5", "0:59.5"],
            "Q3": ["0:58.0", "0:59.0"],
        }
    ).to_csv(event_dir / "q_results.csv", index=False)
    metadata = {
        "year": 2026,
        "round_number": 1,
        "event_name": "Test GP",
        "event_format": "conventional",
        "sessions": [
            {
                "session_type": "practice",
                "session_name": "Practice 1",
                "session_order": 1,
                "completed": True,
                "laps_path": "p1_laps.csv",
                "results_path": "p1_results.csv",
            },
            {
                "session_type": "practice",
                "session_name": "Practice 3",
                "session_order": 3,
                "completed": True,
                "laps_path": "fp3_laps.csv",
                "results_path": "fp3_results.csv",
            },
            {
                "session_type": "qualifying",
                "session_name": "Qualifying",
                "session_order": 4,
                "completed": True,
                "results_path": "q_results.csv",
            },
        ],
    }
    (event_dir / "weekend_metadata.json").write_text(json.dumps(metadata))

    frame, info, _ = _event_frame(tmp_path, event_dir)
    indexed = frame.set_index("driver_id")

    assert set(indexed.index) == {"A"}
    assert indexed.loc["A", "team_id"] == "New Team"
    assert info["official_target_driver_ids"] == ["A", "SUB"]
    assert info["target_result_used_for_roster"] is False
    assert info["roster_source"] == "latest_target_aligned_pre_qualifying_session_only"


def test_walk_forward_helper_trains_only_on_prior_complete_events() -> None:
    frame = _rank_history(event_keys=(202601, 202602, 202603, 202604, 202605))
    result = walk_forward_pairwise_qualifying(
        frame,
        config=_rank_config(minimum_training_events=3),
        evaluation_event_keys=(202604, 202605),
    )

    assert result.skipped_event_keys == ()
    assert result.per_event_metrics["event_key"].tolist() == [202604, 202605]
    assert result.per_event_metrics["observed_targets"].tolist() == [5, 5]
    for _, event in result.predictions.groupby("event_key"):
        assert sorted(event["predicted_qualifying_position"].astype(int).tolist()) == [1, 2, 3, 4, 5]
        assert event["movement_from_baseline"].abs().le(1).all()
