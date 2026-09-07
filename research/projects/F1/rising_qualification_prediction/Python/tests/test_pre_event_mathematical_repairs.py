"""Independent likelihood, causal feature, and classification counterexamples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import log_softmax

from packages.f1.features.race import (
    RACE_ORDER_FEATURE_COLUMNS,
    derive_race_practice_evidence,
    engineer_survival_aware_race_features,
)
from packages.f1.models.pre_race.joint import sample_fia_classification_order
from packages.f1.models.pre_race.ranking import BradleyTerryOrderRanker
from packages.f1.models.pre_race.status import TerminalStatus, reason_code_terminal_status
from packages.f1.models.pre_race.survival import (
    PartialPooledTerminalHazard,
    TerminalHazardConfig,
    _observation_encoding,
    _observed_hazard_terms,
)
from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN,
    fit_achievable_best_lap_model,
)
from packages.f1.models.ultimate_lap_time.tabular_quantile import (
    TabularQuantileConfig,
    fit_tabular_quantile_model,
)


def test_unknown_failure_time_is_integrated_and_gradient_matches_finite_difference() -> None:
    # Each cause/interval has a separate logit. Finite differences exercise the
    # observed likelihood, independently of the optimizer's implementation.
    logits = np.array([[[-2.0, -2.5], [-1.5, -2.0], [-2.2, -1.4]]])
    codes = np.array([1, 0, -1])
    fractions = np.array([np.nan, 0.5, 1.0])
    censor = np.array([3, 3, 3])

    def evaluate(parameters: np.ndarray):
        full = np.broadcast_to(parameters, (3, 3, 2))
        logs = log_softmax(np.concatenate([np.zeros((3, 3, 1)), full], axis=2), axis=2)
        hazards = np.exp(logs[:, :, 1:])
        likelihood, risk, event = _observed_hazard_terms(
            hazards, codes, fractions, censor,
            log_no_event=logs[:, :, 0], log_cause_hazard=logs[:, :, 1:],
        )
        gradient = (risk[:, :, None] * hazards - event).sum(axis=0, keepdims=True)
        return -likelihood.sum(), gradient, likelihood, risk, event, hazards

    value, analytic, likelihood, risk, events, hazards = evaluate(logits)
    h = hazards[0]
    expected_mass = h[0, 1] + (1 - h[0].sum()) * h[1, 1] + np.prod(1 - h[:2].sum(axis=1)) * h[2, 1]
    assert np.exp(likelihood[0]) == pytest.approx(expected_mass)
    assert events[0, :, 1].sum() == pytest.approx(1.0)
    assert risk[0, 0] == pytest.approx(1.0)
    assert np.isfinite(value)
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical = (evaluate(plus)[0] - evaluate(minus)[0]) / 2e-6
        assert analytic[index] == pytest.approx(numerical, abs=1e-8)


def test_extreme_multinomial_logits_keep_observed_likelihood_finite() -> None:
    logs = log_softmax(np.array([[[0.0, 700.0, -700.0]]] * 2), axis=2)
    likelihood, _, _ = _observed_hazard_terms(
        np.exp(logs[:, :, 1:]), np.array([1, -1]), np.array([np.nan, 1.0]), np.array([1, 1]),
        log_no_event=logs[:, :, 0], log_cause_hazard=logs[:, :, 1:],
    )
    assert likelihood == pytest.approx([-1400.0, -700.0])


@pytest.mark.parametrize("n", [200, 1000])
def test_untimed_terminal_risk_retains_observed_incidence_as_sample_grows(n: int) -> None:
    frame = pd.DataFrame({
        "driver_id": [str(i) for i in range(n)],
        "terminal_status": ["Finished"] * (4 * n // 5) + ["DNF"] * (n // 5),
        "retirement_fraction": [1.0] * (4 * n // 5) + [np.nan] * (n // 5),
    })
    model = PartialPooledTerminalHazard().fit(frame)
    result = model.predict_proba(pd.DataFrame({"driver_id": ["unseen"]})).iloc[0]
    assert result["p_non_classified"] == pytest.approx(0.20, abs=0.01)
    assert model.interval_censored_failure_rows == n // 5


def test_untimed_failures_teach_covariate_risk_without_inventing_event_times() -> None:
    n = 300
    high = np.repeat([False, True], n // 2)
    failure = np.arange(n) % 10 < np.where(high, 6, 1)
    history = pd.DataFrame({
        "driver_id": [str(i) for i in range(n)],
        "race_wet_probability": high.astype(float),
        "terminal_status": np.where(failure, "DNF", "Finished"),
        "retirement_fraction": np.where(failure, np.nan, 1.0),
    })
    model = PartialPooledTerminalHazard(TerminalHazardConfig(covariate_l2_c=1.0)).fit(history)
    result = model.predict_proba(pd.DataFrame({
        "driver_id": ["dry", "wet"], "race_wet_probability": [0.0, 1.0],
    }))
    assert result.loc[1, "p_non_classified"] > result.loc[0, "p_non_classified"] + 0.30
    assert model.interval_censored_failure_rows == int(failure.sum())


def test_exclusion_is_not_an_observed_running_failure() -> None:
    assert reason_code_terminal_status("Disqualified for fuel infringement") is TerminalStatus.DISQUALIFIED
    statuses = pd.Series([TerminalStatus.DISQUALIFIED, TerminalStatus.DISQUALIFIED, TerminalStatus.DNS_WITHDRAWAL])
    codes, _, censor = _observation_encoding(statuses, pd.Series([1.0, np.nan, 0.0]), 12)
    assert codes.tolist() == [-1, -1, -1]
    assert censor.tolist() == [12, 0, 0]
    history = pd.DataFrame({
        "terminal_status": ["Finished"] * 90 + ["Disqualified"] * 10,
        "retirement_fraction": np.ones(100),
    })
    model = PartialPooledTerminalHazard().fit(history)
    row = model.predict_proba(pd.DataFrame({"driver_id": ["unseen"]})).iloc[0]
    assert row["p_disqualified"] == pytest.approx(11 / 106)
    assert row["p_non_classified"] < 0.02
    assert np.isnan(row["expected_retirement_fraction_disqualified"])


def _classification(statuses, fractions, scores, laps=60.0):
    count = len(statuses)
    return sample_fia_classification_order(
        statuses=statuses, terminal_retirement_fraction=np.asarray(fractions),
        conditional_scores=np.asarray(scores), order_shocks=np.zeros(count),
        expected_lap_deficit=np.zeros(count), scheduled_laps=np.full(count, laps),
        grid_positions=np.arange(1.0, count + 1), driver_ids=np.array([str(i) for i in range(count)]),
    )


def test_full_distance_disqualification_is_excluded_even_with_fastest_running_score() -> None:
    order, distance = _classification(
        [TerminalStatus.DISQUALIFIED, TerminalStatus.CLASSIFIED_FINISH, TerminalStatus.NON_CLASSIFIED],
        [1.0, 1.0, 0.5], [100.0, 0.0, -10.0],
    )
    assert order.tolist() == [1, 2, 0]
    assert distance.tolist() == [60.0, 60.0, 30.0]


def test_retirees_with_equal_completed_laps_use_running_order_not_partial_laps() -> None:
    order, distance = _classification(
        [TerminalStatus.NON_CLASSIFIED] * 2, [30.9 / 60, 30.1 / 60], [0.0, 2.0],
    )
    assert distance.tolist() == [30.0, 30.0]
    assert order.tolist() == [1, 0]


def test_pair_probability_is_invariant_to_full_field_feature_context() -> None:
    roster = pd.DataFrame({"event_key": [202601] * 20, "driver_id": [str(i) for i in range(20)], "grid_position": np.arange(1, 21)})
    ranker = BradleyTerryOrderRanker().fit(roster.assign(finish_position=roster.grid_position))
    full_scores = ranker.score(roster)["conditional_order_score"].to_numpy()
    expected = 1 / (1 + np.exp(-(full_scores[0] - full_scores[1])))
    assert ranker.pairwise_probability(roster.iloc[0], roster.iloc[1], roster=roster) == pytest.approx(expected)
    engineered = engineer_survival_aware_race_features(roster)
    assert ranker.pairwise_probability(engineered.iloc[0], engineered.iloc[1]) == pytest.approx(expected)
    with pytest.raises(ValueError, match="complete roster"):
        ranker.pairwise_probability(roster.iloc[0], roster.iloc[1])
    with pytest.raises(ValueError, match="distinct"):
        ranker.pairwise_probability(engineered.iloc[0], engineered.iloc[0])


@pytest.mark.parametrize("event_key", [202603, 202604, np.nan, 202602.5])
def test_all_best_lap_components_reject_invalid_or_future_rows_before_label_filter(event_key) -> None:
    history = pd.DataFrame({
        "event_key": [202601, 202602, event_key], "driver_id": ["a", "a", "a"],
        "rehearsal_source": ["fp3"] * 3, "rehearsal_lap_time_seconds": [90.0] * 3,
        ACTUAL_LAP_COLUMN: [89.0, 89.0, np.nan], "valid_lap": [1, 1, 0],
    })
    with pytest.raises(ValueError):
        fit_achievable_best_lap_model(history, target_event_key=202603)


@pytest.mark.parametrize("component", ["stage_calibration", "stage_time_effects", "residual_model"])
def test_best_lap_inference_checks_chronology_of_each_attached_component(component: str) -> None:
    from dataclasses import replace

    model = fit_achievable_best_lap_model(pd.DataFrame(), target_event_key=202603)
    contaminated = replace(model, **{component: replace(getattr(model, component), event_keys=(202603,))})
    assert contaminated.training_event_keys == (202603,)
    inference = pd.DataFrame({"event_key": [202603], "driver_id": ["a"], "rehearsal_source": ["fp3"], "rehearsal_lap_time_seconds": [90.0]})
    with pytest.raises(ValueError, match="strictly earlier"):
        contaminated.predict(inference)


def test_explicit_quantile_target_cannot_fall_back_to_a_different_estimand() -> None:
    with pytest.raises(ValueError, match="configured target"):
        fit_tabular_quantile_model(
            pd.DataFrame({"season": [2026, 2026], "lap_time_seconds": [88.0, 89.0], "tyre_age": [1.0, 2.0]}),
            config=TabularQuantileConfig(backend="empirical", target_column="achievable_best_lap_seconds", feature_columns=("tyre_age",)),
        )


def _practice() -> pd.DataFrame:
    return pd.DataFrame([
        {"DriverNumber": driver, "LapNumber": lap + 1 + 10 * index, "LapTime": 90.0 + offset,
         "Time": 1000.0 + 100 * lap, "TyreLife": lap + 1, "Compound": "MEDIUM", "Stint": 1}
        for index, (driver, offset) in enumerate([("1", -1.0), ("2", 0.0), ("3", 0.0)])
        for lap in range(4)
    ])


@pytest.mark.parametrize("variant", ["single_driver", "missing_clock", "separate_clocks", "unknown_compound", "missing_tyre_age"])
def test_unsupported_practice_comparisons_remain_missing(variant: str) -> None:
    laps = _practice()
    if variant == "single_driver":
        laps = laps[laps.DriverNumber == "1"]
    elif variant == "missing_clock":
        laps = laps.drop(columns="Time")
    elif variant == "separate_clocks":
        laps["Time"] += laps.DriverNumber.astype(int) * 10000
    elif variant == "missing_tyre_age":
        laps = laps.drop(columns="TyreLife")
    else:
        laps["Compound"] = "UNKNOWN"
    evidence = derive_race_practice_evidence(laps, session_label="FP2")
    assert evidence.race_compound_pace_delta.isna().all()
    assert evidence.race_practice_uncertainty.isna().all()
    assert evidence.race_practice_evidence_count.eq(0).all()


def test_practice_matches_clock_time_across_different_driver_lap_numbers() -> None:
    evidence = derive_race_practice_evidence(_practice(), session_label="FP2").set_index("driver_id")
    assert evidence.loc["1", "race_compound_pace_delta"] == pytest.approx(-1.0)
    assert evidence.loc["1", "race_practice_evidence_count"] == 4
    assert evidence.loc["1", "race_practice_minimum_independent_peer_count"] == 2
    assert evidence.loc["1", "race_practice_uncertainty"] > 0.0
    assert evidence.race_tyre_age_pace_delta.isna().all()
    assert evidence.race_fuel_track_adjusted_degradation.isna().all()
    assert "race_degradation_score" not in RACE_ORDER_FEATURE_COLUMNS


def test_practice_relative_drift_is_measured_against_independent_peers() -> None:
    laps = _practice()
    mask = laps.DriverNumber.eq("1")
    laps.loc[mask, "LapTime"] += 0.2 * (laps.loc[mask, "TyreLife"] - 1)
    result = derive_race_practice_evidence(laps, session_label="FP2").set_index("driver_id")
    assert result.loc["1", "race_relative_pace_drift_seconds_per_tyre_lap"] == pytest.approx(0.2)


def test_native_fastf1_timedeltas_keep_seconds_units_in_practice_comparison() -> None:
    laps = _practice()
    laps["LapTime"] = pd.to_timedelta(laps.LapTime, unit="s")
    laps["Time"] = pd.to_timedelta(laps.Time, unit="s")
    result = derive_race_practice_evidence(laps, session_label="FP2").set_index("driver_id")
    assert result.loc["1", "race_compound_pace_delta"] == pytest.approx(-1.0)


def test_openf1_metadata_preserves_known_starting_age_for_every_stint() -> None:
    from packages.f1.features.race import _attach_openf1_stints

    laps = pd.DataFrame({"driver_number": [1, 1, 2, 2], "lap_number": [2, 3, 2, 3]})
    stints = pd.DataFrame({
        "driver_number": [1, 2], "stint_number": [1, 1],
        "lap_start": [2, 2], "lap_end": [3, 3], "compound": ["MEDIUM", "MEDIUM"],
        "tyre_age_at_start": [0, 20],
    })
    attached = _attach_openf1_stints(laps, stints)
    assert attached.tyre_life.tolist() == [0.0, 1.0, 20.0, 21.0]
    # Unknown starting age must not silently create a fresh tyre assumption.
    stints.loc[1, "tyre_age_at_start"] = np.nan
    attached = _attach_openf1_stints(laps, stints)
    assert attached.tyre_life.iloc[2:].isna().all()


def test_qualifying_optional_predictions_serialize_as_null_without_changing_known_values() -> None:
    import json
    from run_qualifying_pairwise_challenger_backtest import _nullable_prediction_rows

    rows = _nullable_prediction_rows([{"lap_p05": float("nan"), "predicted_position": 1}])
    assert json.loads(json.dumps(rows, allow_nan=False)) == [{"lap_p05": None, "predicted_position": 1}]
    with pytest.raises(ValueError, match="infinite"):
        _nullable_prediction_rows([{"lap_p50": float("inf")}])
