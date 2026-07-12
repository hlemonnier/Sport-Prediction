from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import packages.f1.orchestration.prediction as prediction_module
from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.features.assembly import _add_temporal_features_train, _attach_temporal_features_current
from packages.f1.features.weather import apply_f1_weather_to_features
from packages.f1.features.wet import add_f1_wet_pace_interactions
from packages.f1.models.race_probability import RACE_PROBABILITY_SCORE_LAYER, race_stochastic_score_layer
from packages.f1.models.training import (
    _fold_metrics,
    _infer_target_spec,
    _oof_scores_for_selection,
    _probability_calibration_audit_event_split,
    _probability_audit_from_oof,
    train_model,
)
from packages.f1.orchestration.prediction import (
    _apply_race_information_horizon,
    _merge_predicted_qualifying_context,
    _qualifying_feature_sets,
    _race_feature_sets,
    _race_input_evidence,
    _race_stochastic_score_layer,
    _resolve_race_information_horizon,
)


def test_current_temporal_state_includes_latest_completed_event_exactly_once() -> None:
    history = pd.DataFrame(
        {
            "driver_id": ["a", "a"],
            "team_name": ["red", "red"],
            "event_name": ["Round 1", "Round 2"],
            "event_key": [202501, 202502],
            "fp_mean_delta": [1.0, 3.0],
            "fp_weighted_delta": [1.0, 3.0],
        }
    )
    current = pd.DataFrame(
        {
            "driver_id": ["a"],
            "team_name": ["red"],
            "event_name": ["Round 3"],
            "event_key": [202503],
            "fp_mean_delta": [99.0],
            "fp_weighted_delta": [99.0],
        }
    )

    shifted_train = _add_temporal_features_train(history)
    attached = _attach_temporal_features_current(current, history)

    # Training remains causal: Round 2 sees only Round 1.
    assert shifted_train.loc[shifted_train["event_key"] == 202502, "driver_form_3_fp_mean_delta"].iloc[0] == 1.0
    # Current Round 3 sees Round 1 and the latest completed Round 2 once.
    assert attached["driver_form_3_fp_mean_delta"].iloc[0] == 2.0
    assert attached["driver_ewma_fp_mean_delta"].iloc[0] == 2.0
    assert attached["team_form_3_fp_mean_delta"].iloc[0] == 2.0


def _race_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": ["a", "b", "c"],
            "target": [2.0, 1.0, 3.0],
            "grid_position": [1.0, 2.0, 3.0],
            "grid_source": ["retrospective_results_grid"] * 3,
            "grid_status": ["grid"] * 3,
            "qualy_position": [1.0, 2.0, 3.0],
            "qualy_gap_to_best": [0.0, 0.2, 0.4],
            "qualy_pred_position": [0.3, 0.1, 0.5],
            "qualy_pred_rank": [2.0, 1.0, 3.0],
            "qualy_pred_vs_actual_gap": [1.0, 1.0, 0.0],
            "fp_mean_delta": [0.1, 0.2, 0.3],
            "fp_race_sim_delta": [0.2, 0.4, 0.6],
            "pace_sessions_available": [3.0, 3.0, 3.0],
            "event_pace_index": [0.1, 0.5, 0.9],
            "driver_form_3_fp_mean_delta": [0.4, 0.5, 0.6],
        }
    )


def _complete_official_grid_event(
    *,
    year: int,
    round_number: int,
    evidence_as_of: str,
) -> pd.DataFrame:
    field_size = 22 if year >= 2026 else 20
    positions = list(range(1, field_size + 1))
    return pd.DataFrame(
        {
            "event_key": [(year * 100) + round_number] * field_size,
            "event_year": [year] * field_size,
            "event_round": [round_number] * field_size,
            "driver_id": [f"{round_number}-{value}" for value in positions],
            "target": positions,
            "qualy_position": positions,
            "grid_position": positions,
            "grid_source": ["pre_race_official_grid"] * field_size,
            "grid_status": ["grid"] * field_size,
            "grid_revision_phase": ["final_pre_race"] * field_size,
            "grid_evidence_as_of": [evidence_as_of] * field_size,
            "grid_evidence_id": [f"fia-grid-r{round_number}-final"] * field_size,
            "grid_evidence_complete": [True] * field_size,
            "weekend_format_version": ["standard"] * field_size,
        }
    )


def test_prequal_race_horizon_replaces_retrospective_grid_and_scrubs_unavailable_qualifying() -> None:
    out = _apply_race_information_horizon(
        _race_rows(),
        horizon="post_fp_pre_qualifying",
        training=True,
    )

    assert out["grid_position"].tolist() == [2.0, 1.0, 3.0]
    assert set(out["grid_source"]) == {"oof_predicted_qualifying_grid"}
    assert out["race_delta_target"].tolist() == [0.0, 0.0, 0.0]
    assert out["qualy_position"].isna().all()
    assert out["qualy_gap_to_best"].isna().all()
    assert out["qualy_pred_vs_actual_gap"].isna().all()
    assert set(out["race_information_horizon"]) == {"post_fp_pre_qualifying"}


def test_postqual_and_official_grid_horizons_remain_distinct() -> None:
    rows = _race_rows()
    rows["grid_position"] = [3.0, 1.0, 2.0]

    postqual = _apply_race_information_horizon(rows, horizon="post_qualifying_pre_grid", training=True)
    official = _apply_race_information_horizon(rows, horizon="post_grid_pre_race", training=True)

    assert postqual["grid_position"].tolist() == [1.0, 2.0, 3.0]
    assert set(postqual["grid_source"]) == {"historical_qualifying_proxy"}
    assert not postqual["grid_evidence_complete"].any()
    assert official["grid_position"].isna().all()
    assert set(official["grid_source"]) == {"unresolved_final_grid"}
    assert not official["grid_evidence_complete"].any()


def test_post_grid_horizon_accepts_only_a_complete_timestamped_final_revision() -> None:
    rows = _complete_official_grid_event(
        year=2026,
        round_number=1,
        evidence_as_of="2026-03-08T13:00:00Z",
    )

    official = _apply_race_information_horizon(
        rows,
        horizon="post_grid_pre_race",
        training=True,
    )

    assert official["grid_position"].tolist() == [float(value) for value in range(1, 23)]
    assert set(official["grid_source"]) == {"pre_race_official_grid"}
    assert set(official["grid_resolution_status"]) == {"resolved"}
    assert official["grid_evidence_complete"].all()


def test_final_grid_evidence_after_prediction_cutoff_fails_closed() -> None:
    rows = _complete_official_grid_event(
        year=2026,
        round_number=1,
        evidence_as_of="2026-03-08T13:00:00Z",
    )

    assert _resolve_race_information_horizon(
        rows,
        "auto",
        prediction_as_of="2026-03-08T12:00:00Z",
    ) == "post_qualifying_pre_grid"
    result = _apply_race_information_horizon(
        rows,
        horizon="post_grid_pre_race",
        training=False,
        as_of="2026-03-08T12:00:00Z",
    )
    assert result["grid_position"].isna().all()
    assert set(result["grid_source"]) == {"unresolved_final_grid"}


def test_post_grid_training_resolves_each_event_independently() -> None:
    rows = pd.concat(
        [
            _complete_official_grid_event(
                year=2025,
                round_number=1,
                evidence_as_of="2025-03-16T03:00:00Z",
            ),
            _complete_official_grid_event(
                year=2025,
                round_number=2,
                evidence_as_of="2025-03-23T06:00:00Z",
            ),
        ],
        ignore_index=True,
    )
    result = _apply_race_information_horizon(
        rows,
        horizon="post_grid_pre_race",
        training=True,
    )
    assert len(result) == 40
    assert result["grid_evidence_complete"].all()
    assert set(result["grid_resolution_status"]) == {"resolved"}
    assert result.groupby("event_key")["grid_position"].count().eq(20).all()


@pytest.mark.parametrize("bad_position", [1.9, np.inf])
def test_final_grid_rejects_non_integral_or_nonfinite_positions(bad_position: float) -> None:
    rows = _complete_official_grid_event(
        year=2026,
        round_number=1,
        evidence_as_of="2026-03-08T13:00:00Z",
    )
    rows["grid_position"] = rows["grid_position"].astype(float)
    rows.loc[0, "grid_position"] = bad_position
    result = _apply_race_information_horizon(
        rows,
        horizon="post_grid_pre_race",
        training=False,
    )
    assert result["grid_position"].isna().all()


def test_final_grid_requires_provenance_on_every_row() -> None:
    rows = _complete_official_grid_event(
        year=2026,
        round_number=1,
        evidence_as_of="2026-03-08T13:00:00Z",
    )
    rows.loc[0, "grid_evidence_id"] = None
    result = _apply_race_information_horizon(
        rows,
        horizon="post_grid_pre_race",
        training=False,
    )
    assert result["grid_position"].isna().all()


def test_2021_2022_postqual_horizon_does_not_call_qualifying_the_gp_grid() -> None:
    rows = _race_rows()
    rows["weekend_format_version"] = "sprint_2021_2022"

    result = _apply_race_information_horizon(
        rows,
        horizon="post_qualifying_pre_grid",
        training=True,
    )

    assert result["grid_position"].tolist() == [1.0, 2.0, 3.0]
    assert set(result["grid_source"]) == {"historical_qualifying_proxy_pre_sprint"}
    assert set(result["grid_status"]) == {"gp_grid_unresolved_proxy"}
    assert not result["grid_evidence_complete"].any()


def test_missing_nested_qualifying_signal_still_enforces_race_horizon(monkeypatch) -> None:
    monkeypatch.setattr(prediction_module, "build_training_data", lambda **_: (pd.DataFrame(), []))
    monkeypatch.setattr(prediction_module, "build_current_features", lambda **_: (pd.DataFrame(), []))
    config = PredictionConfig(
        source="local",
        mode="race",
        year=2026,
        round_number=3,
        train_seasons=[2026],
        include_standings=False,
        cache_dir=None,
        meeting_name=None,
        country_name=None,
        weekends_dir=None,
        race_information_horizon="post_fp_pre_qualifying",
        qualifying_information_horizon="pre_qualifying",
    )

    training, current = _merge_predicted_qualifying_context(
        object(),
        config,
        _race_rows(),
        _race_rows().drop(columns=["target"]),
        [],
    )

    assert training["grid_position"].tolist() == [2.0, 1.0, 3.0]
    assert current["grid_position"].tolist() == [2.0, 1.0, 3.0]
    assert training["qualy_position"].isna().all()
    assert current["qualy_position"].isna().all()
    assert set(training["grid_source"]) == {"oof_predicted_qualifying_grid"}
    assert set(current["grid_source"]) == {"predicted_qualifying_grid"}


def test_pre_fp_horizon_scrubs_completed_weekend_practice_without_losing_prior_form() -> None:
    rows = _race_rows()

    pre_fp = _apply_race_information_horizon(rows, horizon="pre_fp_provisional", training=True)
    post_fp = _apply_race_information_horizon(rows, horizon="post_fp_pre_qualifying", training=True)

    for column in ("fp_mean_delta", "fp_race_sim_delta", "pace_sessions_available", "event_pace_index"):
        assert pre_fp[column].isna().all()
        assert post_fp[column].notna().all()
    assert pre_fp["driver_form_3_fp_mean_delta"].tolist() == [0.4, 0.5, 0.6]

    evidence = _race_input_evidence(
        config=PredictionConfig(
            source="local",
            mode="race",
            year=2026,
            round_number=1,
            train_seasons=[2025],
            include_standings=False,
            cache_dir=None,
            meeting_name=None,
            country_name=None,
            weekends_dir=None,
        ),
        features=pre_fp,
        feature_cols=["fp_mean_delta", "driver_form_3_fp_mean_delta"],
        fallback_cols=["event_pace_index", "driver_form_3_fp_mean_delta"],
        weather_summary={},
    )
    assert "fp_mean_delta" in evidence["model_features_present"]
    assert "fp_mean_delta" not in evidence["model_features_available"]
    assert "driver_form_3_fp_mean_delta" in evidence["model_features_available"]


def test_race_horizon_resolution_is_explicit_and_rejects_unknown_values() -> None:
    prequal = pd.DataFrame(
        {
            "pace_sessions_available": [1.0],
            "grid_position": [np.nan],
            "qualy_position": [np.nan],
            "grid_source": ["missing"],
        }
    )
    assert _resolve_race_information_horizon(prequal, "auto") == "post_fp_pre_qualifying"
    unproven_grid = prequal.assign(
        grid_position=[1.0],
        grid_source=["pre_race_official_grid"],
        grid_evidence_complete=[False],
        qualy_position=[1.0],
    )
    proven_but_incomplete_grid = unproven_grid.assign(grid_evidence_complete=[True])
    string_false_grid = unproven_grid.assign(grid_evidence_complete=["false"])
    assert _resolve_race_information_horizon(unproven_grid, "auto") == "post_qualifying_pre_grid"
    assert _resolve_race_information_horizon(string_false_grid, "auto") == "post_qualifying_pre_grid"
    assert _resolve_race_information_horizon(proven_but_incomplete_grid, "auto") == "post_qualifying_pre_grid"
    complete_grid = pd.DataFrame(
        {
            "driver_id": [str(value) for value in range(1, 23)],
            "grid_position": list(range(1, 23)),
            "grid_source": ["pre_race_official_grid"] * 22,
            "grid_status": ["grid"] * 22,
            "grid_revision_phase": ["final_pre_race"] * 22,
            "grid_evidence_as_of": ["2026-03-08T13:00:00Z"] * 22,
            "grid_evidence_id": ["fia-grid-r1-final"] * 22,
            "grid_evidence_complete": [True] * 22,
            "event_year": [2026] * 22,
        }
    )
    assert _resolve_race_information_horizon(complete_grid, "auto") == "post_grid_pre_race"
    assert _resolve_race_information_horizon(prequal, "post_grid") == "post_grid_pre_race"
    with pytest.raises(ValueError, match="Unknown race_information_horizon"):
        _resolve_race_information_horizon(prequal, "future_magic")


def test_race_probability_layer_is_shared_and_does_not_shrink_mobility_twice() -> None:
    low = pd.DataFrame(
        {
            "grid_position": [1.0, 2.0, 3.0],
            "track_finish_order_mobility": [0.05] * 3,
            "track_safety_car_prior": [0.2] * 3,
            "track_dnf_prior": [0.05] * 3,
            "track_strategy_variance_prior": [0.3] * 3,
            "track_weather_uncertainty_prior": [0.1] * 3,
            "race_generation_variance_prior": [0.2] * 3,
        }
    )
    high = low.assign(track_finish_order_mobility=0.90)
    predictions = pd.Series([2.5, 1.2, 2.8])

    low_shared = race_stochastic_score_layer(low, predictions)
    low_production = _race_stochastic_score_layer(low, predictions)
    high_shared = race_stochastic_score_layer(high, predictions)

    pd.testing.assert_frame_equal(low_shared, low_production)
    pd.testing.assert_series_equal(
        low_shared["race_stochastic_score"],
        high_shared["race_stochastic_score"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        low_shared["race_stochastic_pl_score"],
        high_shared["race_stochastic_pl_score"],
        check_names=False,
    )
    assert set(low_shared["race_stochastic_layer"]) == {RACE_PROBABILITY_SCORE_LAYER}


def test_race_oof_calibration_scores_are_the_exact_deployed_transformation() -> None:
    train = pd.DataFrame(
        {
            "event_key": [1, 1, 1, 2, 2, 2],
            "driver_id": ["a", "b", "c"] * 2,
            "grid_position": [1.0, 2.0, 3.0] * 2,
            "target": [1.0, 2.0, 3.0] * 2,
            "race_delta_target": [0.0] * 6,
            "track_dnf_prior": [0.05, 0.10, 0.15] * 2,
            "track_safety_car_prior": [0.2] * 6,
            "track_strategy_variance_prior": [0.3] * 6,
            "track_weather_uncertainty_prior": [0.1] * 6,
            "race_generation_variance_prior": [0.2] * 6,
        }
    )
    spec = _infer_target_spec(train)
    scores, actual, events = _oof_scores_for_selection(
        train=train,
        feature_cols=["grid_position"],
        selected_name="grid_only_baseline",
        candidate_lookup={},
        folds=[({1}, 2)],
        race_baseline_col="grid_position",
        target_spec=spec,
        apply_race_probability_layer=True,
    )
    val = train.loc[train["event_key"] == 2]
    expected = race_stochastic_score_layer(val, val["grid_position"])["race_stochastic_pl_score"]

    pd.testing.assert_series_equal(scores.sort_index(), expected.sort_index(), check_names=False)
    assert actual.tolist() == [1.0, 2.0, 3.0]
    assert events.tolist() == [2.0, 2.0, 2.0]


def test_race_auto_selection_allows_grid_only_to_win_and_locks_latest_event() -> None:
    rows = []
    for event in range(1, 15):
        for grid in range(1, 5):
            rows.append(
                {
                    "event_key": event,
                    "driver_id": f"d{grid}",
                    "grid_position": float(grid),
                    "target": float(grid),
                    "race_delta_target": 0.0,
                    "noise_feature": float((event * 7 + grid * 3) % 11),
                    "fp_race_sim_rank": float(5 - grid),
                    "track_finish_order_mobility": 0.8,
                }
            )
    train = pd.DataFrame(rows)

    result = train_model(
        train,
        ["grid_position", "noise_feature", "fp_race_sim_rank", "track_finish_order_mobility"],
        f1_model="auto",
        f1_pl_samples=200,
    )

    assert result.model_name == "grid_only_baseline"
    assert any(row["name"] == "strategic_race_delta" for row in result.candidate_leaderboard)
    assert result.selection_audit["available"] is True
    assert result.selection_audit["selection_event_keys"] == [5, 6, 7]
    assert result.selection_audit["probability_calibration_event_keys"] == [8, 9]
    assert result.selection_audit["locked_event_keys"] == [10, 11, 12, 13, 14]
    assert result.selection_audit["evaluation_event_disjoint"] is True
    assert result.selection_audit["chronological"] is True
    assert result.selection_audit["event_partition"]["final_audit_model_training_event_keys"] == list(range(1, 10))
    assert result.selection_audit["event_partition"]["final_audit_model_refit_within_block"] is False
    assert all(row["evaluation_scope"] == "selection_walk_forward" for row in result.candidate_leaderboard)
    assert result.probability_audit["selection_event_keys"] == [5, 6, 7]
    assert result.probability_audit["calibration_event_keys"] == [8, 9]
    assert result.probability_audit["audit_event_keys"] == [10, 11, 12, 13, 14]
    assert result.probability_audit["three_way_temporal_partition_available"] is True
    assert result.probability_audit["evaluation_event_disjoint"] is True
    assert result.probability_audit["final_audit_event_block_complete"] is True


def test_training_disables_probability_claims_when_three_way_history_is_too_short() -> None:
    train = pd.DataFrame(
        [
            {
                "event_key": event,
                "driver_id": f"d{grid}",
                "grid_position": float(grid),
                "target": float(grid),
                "race_delta_target": 0.0,
                "fp_race_sim_rank": float(grid),
            }
            for event in range(1, 9)
            for grid in range(1, 5)
        ],
    )

    result = train_model(
        train,
        ["grid_position", "fp_race_sim_rank"],
        f1_model="baseline",
        f1_pl_samples=100,
    )

    assert result.listwise_temperature is None
    assert result.selection_audit["available"] is False
    assert result.selection_audit["probability_calibration_event_keys"] == []
    assert result.selection_audit["locked_event_keys"] == []
    assert result.probability_audit["available"] is False
    assert result.probability_audit["passed"] is False
    assert result.probability_audit["reason"] == "insufficient_events_for_three_way_temporal_validation"


def test_candidate_metrics_explicitly_score_pole_top3_and_probability_quality() -> None:
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    event = pd.Series([1, 1, 1, 1])
    good = _fold_metrics(actual, pd.Series([1.0, 2.0, 3.0, 4.0]), event)
    bad = _fold_metrics(actual, pd.Series([4.0, 3.0, 2.0, 1.0]), event)

    assert good["pole_hit"] == 1.0
    assert good["top3_hit"] == 1.0
    assert good["win_brier"] < bad["win_brier"]
    assert bad["pole_hit"] == 0.0


def test_probability_audit_records_the_score_layer_contract() -> None:
    scores = pd.Series(list(range(1, 7)) * 5, dtype=float)
    actual = pd.Series(list(range(1, 7)) * 5, dtype=float)
    event = pd.Series(np.repeat(range(1, 6), 6), dtype=float)
    audit = _probability_audit_from_oof(
        scores,
        actual,
        event,
        temperature=1.0,
        temperature_audit={
            "available": True,
            "source": "walk_forward_oof",
            "evaluation_disjoint_from_temperature_fit": True,
            "fit_event_keys": [-1, 0],
            "audit_event_keys": [1, 2, 3, 4, 5],
        },
        samples=200,
        score_layer=RACE_PROBABILITY_SCORE_LAYER,
    )

    assert audit["schema_version"] == "pl_gumbel_probability_audit_v4_disjoint_calibration"
    assert audit["score_layer"] == RACE_PROBABILITY_SCORE_LAYER
    assert audit["same_probability_layer_as_production"] is True
    assert audit["evaluation_disjoint_from_temperature_fit"] is True


def test_probability_temperature_fit_and_audit_events_are_disjoint_and_temporal() -> None:
    event = pd.Series(np.repeat(range(202501, 202513), 2), dtype=float)

    split = _probability_calibration_audit_event_split(event)

    assert split["available"] is True
    assert set(split["selection_event_keys"]).isdisjoint(split["fit_event_keys"])
    assert set(split["selection_event_keys"]).isdisjoint(split["audit_event_keys"])
    assert set(split["fit_event_keys"]).isdisjoint(split["audit_event_keys"])
    assert max(split["selection_event_keys"]) < min(split["fit_event_keys"])
    assert max(split["fit_event_keys"]) < min(split["audit_event_keys"])
    assert split["audit_event_count"] >= 5


def test_probability_three_way_split_fails_closed_below_minimum_event_count() -> None:
    event = pd.Series(np.repeat(range(202501, 202509), 2), dtype=float)

    split = _probability_calibration_audit_event_split(event)

    assert split["available"] is False
    assert split["reason"] == "insufficient_events_for_three_way_temporal_validation"
    assert split["required_event_count"] == 9
    assert split["minimum_event_counts"] == {"selection": 2, "calibration": 2, "audit": 5}
    assert split["selection_event_keys"] == list(range(202501, 202509))
    assert split["calibration_event_keys"] == []
    assert split["audit_event_keys"] == []


def test_probability_audit_fails_closed_when_temperature_used_same_events() -> None:
    scores = pd.Series(list(range(1, 7)) * 5, dtype=float)
    actual = pd.Series(list(range(1, 7)) * 5, dtype=float)
    event = pd.Series(np.repeat(range(1, 6), 6), dtype=float)

    audit = _probability_audit_from_oof(
        scores,
        actual,
        event,
        temperature=1.0,
        temperature_audit={"available": True, "source": "walk_forward_oof"},
        samples=100,
    )

    assert audit["passed"] is False
    assert audit["evaluation_disjoint_from_temperature_fit"] is False
    assert "temperature_fit_and_audit_events_not_disjoint" in audit["reason"]


def test_wet_risk_interactions_are_model_features_only_when_causal_evidence_exists() -> None:
    base = pd.DataFrame(
        {
            "fp_wet_sim_delta": [0.2, 0.8],
            "fp_wet_sim_rank": [1.0, 2.0],
            "fp_wet_sim_laps": [12.0, 12.0],
            "wet_sim_sessions_available": [1.0, 1.0],
            "track_weather_uncertainty_prior": [0.1, 0.1],
        }
    )
    low_risk = add_f1_wet_pace_interactions(base)
    high_risk = apply_f1_weather_to_features(
        base,
        {"weather_available": True, "weather_wet_risk": 0.9},
    )
    qualifying_features, _ = _qualifying_feature_sets()
    race_features, _ = _race_feature_sets(include_standings=False)

    assert high_risk["fp_wet_sim_delta_weather_adj"].abs().sum() > low_risk["fp_wet_sim_delta_weather_adj"].abs().sum()
    for feature_set in (qualifying_features, race_features):
        assert "fp_wet_sim_delta_weather_adj" in feature_set
        assert "fp_wet_sim_rank_weather_adj" in feature_set
        assert "fp_wet_sim_delta" not in feature_set
