from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import run_qualifying_pairwise_challenger_backtest as qualifying_runner

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
from packages.f1.models.ultimate_lap_time.achievable import (
    POINT_HEAD_DIRECT_PACE,
    POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS,
)
from run_qualifying_pairwise_challenger_backtest import (
    QUALIFYING_BASELINE_MODEL_ID,
    QUALIFYING_CHALLENGER_MODEL_ID,
    _apply_public_output_decision,
    _baseline_event_forecast,
    _build_argument_parser,
    _build_locked_final_fit_history,
    _build_shared_event_forecast,
    _causal_rehearsal_ranks,
    _event_frame,
    _event_inference_frame,
    _freeze_same_season_engine_from_selection_forecasts,
    _json_ints,
    _locked_event_partitions,
    _load_target_after_frozen_forecast,
    _pre_qualifying_roster,
    _qualifying_contract_gates,
    _qualifying_target_frame,
    _validate_same_season_cli_scope,
)


def test_rejected_selection_winner_is_not_exposed_as_public_output() -> None:
    predictions = pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "baseline_rank_prior": [2, 1],
            "selected_predicted_qualifying_position": [1, 2],
            "selected_output_model_id": [
                QUALIFYING_CHALLENGER_MODEL_ID,
                QUALIFYING_CHALLENGER_MODEL_ID,
            ],
        }
    )

    scored, public_model_id = _apply_public_output_decision(
        predictions,
        selection_block_winner_model_id=QUALIFYING_CHALLENGER_MODEL_ID,
        point_model_promoted=False,
    )

    assert public_model_id == QUALIFYING_BASELINE_MODEL_ID
    assert scored["research_selected_predicted_qualifying_position"].tolist() == [1, 2]
    assert scored["selected_predicted_qualifying_position"].tolist() == [2, 1]
    assert scored["public_output_predicted_qualifying_position"].tolist() == [2, 1]
    assert set(scored["selected_output_model_id"]) == {QUALIFYING_BASELINE_MODEL_ID}


def test_promoted_selection_winner_remains_public_output() -> None:
    predictions = pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "baseline_rank_prior": [2, 1],
            "selected_predicted_qualifying_position": [1, 2],
            "selected_output_model_id": [
                QUALIFYING_CHALLENGER_MODEL_ID,
                QUALIFYING_CHALLENGER_MODEL_ID,
            ],
        }
    )

    scored, public_model_id = _apply_public_output_decision(
        predictions,
        selection_block_winner_model_id=QUALIFYING_CHALLENGER_MODEL_ID,
        point_model_promoted=True,
    )

    assert public_model_id == QUALIFYING_CHALLENGER_MODEL_ID
    assert scored["selected_predicted_qualifying_position"].tolist() == [1, 2]
    assert set(scored["selected_output_model_id"]) == {
        QUALIFYING_CHALLENGER_MODEL_ID
    }


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


def test_json_ints_converts_numpy_event_keys_to_serializable_python_ints() -> None:
    values = _json_ints(pd.Index([np.int64(202601), np.int64(202602)]))

    assert values == [202601, 202602]
    assert all(type(value) is int for value in values)
    assert json.dumps({"event_keys": values})


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
            202601,
            202602,
            202603,
            202604,
            202605,
            202606,
            202607,
            202608,
        ),
        audit_year=2026,
    )
    assert partitions["point_fit"] == (202601, 202602)
    assert partitions["selection"] == (202603, 202604)
    assert partitions["calibration"] == (202605, 202606)
    assert partitions["audit"] == (202607, 202608)


def test_rehearsal_rank_ties_preserve_provider_order_not_driver_alphabet() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["ZED", "ALP", "MID"],
            "valid_clean_best_seconds": [90.0, 90.0, np.nan],
            "quality_aware_anchor_seconds": [90.0, 90.0, np.nan],
        },
        index=[20, 10, 30],
    )

    ranks = _causal_rehearsal_ranks(frame)

    assert ranks.to_dict() == {20: 1, 10: 2, 30: 3}


def test_qualifying_partitions_reject_prior_season_without_explicit_legacy_mode() -> None:
    keys = (
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
        202607,
    )
    with pytest.raises(ValueError, match="outside the target year"):
        _locked_event_partitions(keys, audit_year=2026)

    legacy = _locked_event_partitions(
        keys,
        audit_year=2026,
        same_season_only=False,
    )
    assert legacy["selection"] == (202501, 202502)
    assert legacy["point_fit"] == (202601, 202602)

    with pytest.raises(ValueError, match="exact R1-R6"):
        _locked_event_partitions(
            (202602, 202603, 202604, 202605, 202606, 202607, 202608),
            audit_year=2026,
        )


def test_qualifying_cli_defaults_to_one_2026_season_and_legacy_is_explicit() -> None:
    defaults = _build_argument_parser().parse_args([])
    assert defaults.years == (2026,)
    assert defaults.evaluation_years == (2026,)
    assert defaults.legacy_cross_season is False
    assert _validate_same_season_cli_scope(defaults.years, defaults.evaluation_years) == 2026

    legacy = _build_argument_parser().parse_args(
        [
            "--legacy-cross-season",
            "--years",
            "2025,2026",
            "--evaluation-years",
            "2026",
        ]
    )
    assert legacy.legacy_cross_season is True
    with pytest.raises(ValueError, match="one identical"):
        _validate_same_season_cli_scope(legacy.years, legacy.evaluation_years)


def _same_season_selection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_key": [202603, 202603, 202604, 202604],
            "driver_id": ["a", "b", "a", "b"],
            "qualy_position": [1, 2, 2, 1],
        }
    )


def test_same_season_selector_uses_robust_choice_and_material_gain_gate() -> None:
    rows = [
        {
            "event_key": event_key,
            "baseline_mae": 1.20,
            "location_mae": 1.10,
            "location_meal_mae": 1.00,
            "robust_mae": 0.90,
            "robust_meal_mae": 0.70,
            "baseline_forecast_artifact_sha256": "a" * 64,
            "location_forecast_artifact_sha256": "b" * 64,
            "robust_forecast_artifact_sha256": "c" * 64,
            "selection_contract_gates_passed": True,
            "forecast_frozen_before_target_read": True,
        }
        for event_key in (202603, 202604)
    ]
    frozen, state = _freeze_same_season_engine_from_selection_forecasts(
        rows,
        selection_frame=_same_season_selection_frame(),
    )
    assert frozen["selected_enable_robust_residual"] is True
    assert frozen["selected_challenger_variant"] == "robust"
    assert (
        frozen["selected_point_head"]
        == POINT_HEAD_MINIMUM_EXPECTED_ABSOLUTE_LOSS
    )
    assert state.selected_model_id == "shared_qualifying_latent_lap_v4"

    insufficient_gain = [
        {
            "event_key": event_key,
            "baseline_mae": 1.00,
            "location_mae": 1.00,
            "location_meal_mae": 0.99,
            "robust_mae": 0.90,
            "robust_meal_mae": 0.88,
            "baseline_forecast_artifact_sha256": "a" * 64,
            "location_forecast_artifact_sha256": "b" * 64,
            "robust_forecast_artifact_sha256": "c" * 64,
            "selection_contract_gates_passed": True,
            "forecast_frozen_before_target_read": True,
        }
        for event_key in (202603, 202604)
    ]
    retained, retained_state = _freeze_same_season_engine_from_selection_forecasts(
        insufficient_gain,
        selection_frame=_same_season_selection_frame(),
    )
    assert retained["selected_enable_robust_residual"] is True
    assert retained["selected_point_head"] == POINT_HEAD_DIRECT_PACE
    assert retained_state.selected_model_id == "qualifying_rehearsal_rank_baseline_v1"
    assert retained_state.decision == "baseline_retained_no_material_challenger_gain"

    failed_contract = [dict(row) for row in rows]
    failed_contract[1]["selection_contract_gates_passed"] = False
    failed, failed_state = _freeze_same_season_engine_from_selection_forecasts(
        failed_contract,
        selection_frame=_same_season_selection_frame(),
    )
    assert failed["selection_contract_gates_passed"] is False
    assert failed_state.selected_model_id == "qualifying_rehearsal_rank_baseline_v1"
    assert "promotion_gates_failed" in failed_state.decision


def test_same_season_selector_rejects_post_target_selection_evidence() -> None:
    rows = [
        {
            "event_key": event_key,
            "baseline_mae": 1.2,
            "location_mae": 1.0,
            "location_meal_mae": 0.9,
            "robust_mae": 0.9,
            "robust_meal_mae": 0.7,
            "baseline_forecast_artifact_sha256": "a" * 64,
            "location_forecast_artifact_sha256": "b" * 64,
            "robust_forecast_artifact_sha256": "c" * 64,
            "selection_contract_gates_passed": True,
            "forecast_frozen_before_target_read": event_key == 202603,
        }
        for event_key in (202603, 202604)
    ]
    with pytest.raises(ValueError, match="frozen before target"):
        _freeze_same_season_engine_from_selection_forecasts(
            rows,
            selection_frame=_same_season_selection_frame(),
        )


def test_same_season_point_head_requires_material_consistent_selection_gain() -> None:
    # The average MEAL gain is material, but it regresses on R4. The fixed
    # two-event consistency shield therefore retains direct pace.
    rows = [
        {
            "event_key": 202603,
            "baseline_mae": 1.20,
            "location_mae": 0.90,
            "location_meal_mae": 0.70,
            "robust_mae": 0.90,
            "robust_meal_mae": 0.70,
            "baseline_forecast_artifact_sha256": "a" * 64,
            "location_forecast_artifact_sha256": "b" * 64,
            "robust_forecast_artifact_sha256": "c" * 64,
            "selection_contract_gates_passed": True,
            "forecast_frozen_before_target_read": True,
        },
        {
            "event_key": 202604,
            "baseline_mae": 1.20,
            "location_mae": 0.90,
            "location_meal_mae": 0.92,
            "robust_mae": 0.90,
            "robust_meal_mae": 0.92,
            "baseline_forecast_artifact_sha256": "a" * 64,
            "location_forecast_artifact_sha256": "b" * 64,
            "robust_forecast_artifact_sha256": "c" * 64,
            "selection_contract_gates_passed": True,
            "forecast_frozen_before_target_read": True,
        },
    ]

    frozen, _state = _freeze_same_season_engine_from_selection_forecasts(
        rows,
        selection_frame=_same_season_selection_frame(),
    )

    assert frozen["selected_point_head"] == POINT_HEAD_DIRECT_PACE
    assert frozen["direct_pace_point_head_event_mean_mae"] == pytest.approx(0.90)
    assert frozen[
        "minimum_expected_absolute_loss_point_head_event_mean_mae"
    ] == pytest.approx(0.81)


def test_final_refit_uses_exactly_locked_rounds_one_to_four() -> None:
    partitions = {
        "point_fit": (202601, 202602),
        "selection": (202603, 202604),
        "calibration": (202605, 202606),
        "audit": (202607, 202608),
    }
    point_fit = pd.DataFrame(
        {"event_key": [202601, 202602], "driver_id": ["a", "a"]}
    )
    selection = pd.DataFrame(
        {"event_key": [202603, 202604], "driver_id": ["a", "a"]}
    )
    history, keys = _build_locked_final_fit_history(
        point_fit,
        selection,
        partitions=partitions,
    )
    assert keys == (202601, 202602, 202603, 202604)
    assert set(history["weak_transfer_prior"]) == {False}

    leaked = pd.concat(
        [selection, pd.DataFrame({"event_key": [202605], "driver_id": ["a"]})],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="exactly the locked R1-R4"):
        _build_locked_final_fit_history(point_fit, leaked, partitions=partitions)


def test_shared_forecast_adapter_forwards_frozen_robust_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_builder(history, inference, **kwargs):
        observed.update(kwargs)
        return "model", "forecast", {"artifact_sha256": "a" * 64}

    monkeypatch.setattr(
        qualifying_runner,
        "build_shared_qualifying_event_forecast",
        fake_builder,
    )
    result = _build_shared_event_forecast(
        pd.DataFrame(),
        pd.DataFrame(),
        target_event_key=202607,
        enable_robust_residual=True,
    )
    assert result[0] == "model"
    assert observed["target_event_key"] == 202607
    assert observed["enable_robust_residual"] is True


def test_point_fit_baseline_artifact_is_complete_and_target_free() -> None:
    inference = pd.DataFrame(
        {
            "driver_id": ["b", "a", "c"],
            "latest_qualifying_rehearsal_rank": [1, 1, 3],
        }
    )
    point, artifact = _baseline_event_forecast(
        inference,
        target_event_key=202601,
        phase="point_fit",
    )
    assert sorted(point["predicted_qualifying_position"].tolist()) == [1, 2, 3]
    assert artifact["event_key"] == 202601
    assert artifact["target_columns_present"] == []
    assert len(str(artifact["artifact_sha256"])) == 64


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
    monkeypatch: pytest.MonkeyPatch,
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

    target_reads: list[str] = []
    original_read_csv = pd.read_csv

    def tracked_read_csv(path, *args, **kwargs):
        if str(path).endswith("q_results.csv"):
            target_reads.append(str(path))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(qualifying_runner.pd, "read_csv", tracked_read_csv)

    inference, pre_target_info, _, target_path = _event_inference_frame(
        tmp_path,
        event_dir,
    )
    assert target_reads == []
    assert target_path.name == "q_results.csv"
    assert "qualy_position" not in inference.columns
    assert "official_target_driver_ids" not in pre_target_info

    frame, info, _ = _event_frame(tmp_path, event_dir)
    indexed = frame.set_index("driver_id")

    assert set(indexed.index) == {"A"}
    assert indexed.loc["A", "team_id"] == "New Team"
    assert info["official_target_driver_ids"] == ["A", "SUB"]
    assert info["target_result_used_for_roster"] is False
    assert info["roster_source"] == "latest_target_aligned_pre_qualifying_session_only"
    assert len(target_reads) == 1


def test_qualifying_runner_target_io_requires_frozen_event_forecast(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    target_path = tmp_path / "q_results.csv"
    calls: list[str] = []

    def read_target(path):
        calls.append(f"target_read:{path.name}")
        return pd.DataFrame({"driver_id": ["AAA"]}), {
            "official_target_driver_count": 1,
            "official_target_driver_ids": ["AAA"],
        }

    monkeypatch.setattr(qualifying_runner, "_qualifying_target_frame", read_target)

    with pytest.raises(RuntimeError, match="requires a frozen forecast artifact"):
        _load_target_after_frozen_forecast(
            target_path,
            expected_event_key=202601,
            frozen_forecast_artifact={},
        )
    assert calls == []

    target, info = _load_target_after_frozen_forecast(
        target_path,
        expected_event_key=202601,
        frozen_forecast_artifact={
            "event_key": 202601,
            "artifact_sha256": "b" * 64,
        },
    )

    assert calls == ["target_read:q_results.csv"]
    assert target["driver_id"].tolist() == ["AAA"]
    assert info["official_target_driver_ids"] == ["AAA"]


def test_official_advancement_is_separate_from_next_segment_time_presence(
    tmp_path,
) -> None:
    path = tmp_path / "qualifying_results.csv"
    positions = np.arange(1, 23)
    q2 = [90.0 + index / 100 for index in range(22)]
    q3 = [89.0 + index / 100 if position <= 10 else np.nan for index, position in enumerate(positions)]
    q2[15] = np.nan  # P16 officially reached Q2 but set no valid Q2 time.
    q3[9] = np.nan  # P10 officially reached Q3 but set no valid Q3 time.
    pd.DataFrame(
        {
            "Abbreviation": [f"D{index:02d}" for index in range(22)],
            "Position": positions,
            "Status": [np.nan] * 22,
            "Q1": [91.0 + index / 100 for index in range(22)],
            "Q2": q2,
            "Q3": q3,
        }
    ).to_csv(path, index=False)

    target, info = _qualifying_target_frame(path)
    by_position = target.set_index("qualy_position")

    assert by_position.loc[16, "reached_q2"] == 1
    assert by_position.loc[16, "has_valid_q2_lap"] == 0
    assert by_position.loc[10, "reached_q3"] == 1
    assert by_position.loc[10, "has_valid_q3_lap"] == 0
    # A later-stage time proves advancement even if a post-session penalty
    # moved the final classification outside the normal segment cutoff.
    assert by_position.loc[17, "reached_q2"] == 1
    assert by_position.loc[17, "has_valid_q2_lap"] == 1
    assert info["advanced_stage_time_validity_modeled_separately"] is True


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
