from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from packages.f1.data.providers.local_weekends import LocalWeekendProvider
from packages.f1.domain.starting_grid import (
    ClassificationEntry,
    ClassificationSnapshot,
    GridAdjustment,
    GridAdjustmentKind,
    GridEntryStatus,
    GridRevisionPhase,
    OfficialGridEntry,
    OfficialGridRevision,
    RacePredictionHorizon,
    build_race_grid_snapshot,
    resolve_grand_prix_start,
)
from packages.f1.domain.weekend import Session, build_weekend_contract
from packages.f1.features.race import (
    engineer_survival_aware_race_features,
    race_grid_snapshot_frame,
)
from packages.f1.models.pre_race import (
    BinaryTerminalCalibrator,
    BradleyTerryOrderRanker,
    PartialPooledTerminalHazard,
    SurvivalAwareRaceModel,
    TerminalStatus,
    add_reason_coded_terminal_targets,
    evaluate_terminal_status_probabilities,
    reason_code_terminal_status,
)
from packages.f1.models.pre_race.joint import (
    expected_classified_lap_deficit,
    minimum_expected_absolute_assignment,
    sample_fia_classification_order,
)


DRIVERS = tuple(f"DRV{number:02d}" for number in range(1, 21))


def _classification() -> ClassificationSnapshot:
    return ClassificationSnapshot(
        session=Session.QUALIFYING,
        entries=tuple(
            ClassificationEntry(driver_id=driver, position=position)
            for position, driver in enumerate(DRIVERS, start=1)
        ),
        as_of="2025-07-05T15:00:00Z",
        evidence_id="qualifying-classification",
    )


def _resolution(*, with_final_grid: bool):
    revisions = ()
    if with_final_grid:
        entries = []
        for position, driver in enumerate(DRIVERS, start=1):
            if position == 19:
                entries.append(
                    OfficialGridEntry(
                        driver_id=driver,
                        position=None,
                        status=GridEntryStatus.PIT_LANE,
                        adjustments=(GridAdjustmentKind.PIT_LANE_START,),
                    )
                )
            elif position == 20:
                entries.append(
                    OfficialGridEntry(
                        driver_id=driver,
                        position=None,
                        status=GridEntryStatus.WITHDRAWN,
                        adjustments=(GridAdjustmentKind.WITHDRAWAL,),
                    )
                )
            else:
                entries.append(OfficialGridEntry(driver_id=driver, position=position))
        revisions = (
            OfficialGridRevision(
                revision_id="fia-final-grid-v2",
                phase=GridRevisionPhase.FINAL_PRE_RACE,
                entries=tuple(entries),
                as_of="2025-07-06T12:30:00Z",
            ),
        )
    return resolve_grand_prix_start(
        build_weekend_contract(2025),
        as_of="2025-07-06T13:00:00Z",
        classifications=(_classification(),),
        official_grid_revisions=revisions,
    )


def _history() -> pd.DataFrame:
    records = []
    event_dates = (
        "2025-01-01T12:00:00Z",
        "2025-02-01T12:00:00Z",
        "2025-03-01T12:00:00Z",
    )
    statuses = (
        TerminalStatus.CLASSIFIED_FINISH.value,
        TerminalStatus.CLASSIFIED_FINISH.value,
        TerminalStatus.MECHANICAL_POWER_UNIT.value,
        TerminalStatus.COLLISION_INCIDENT.value,
    )
    for event, event_as_of in enumerate(event_dates, start=1):
        for grid, driver in enumerate(("A", "B", "C", "D"), start=1):
            records.append(
                {
                    "event_key": event,
                    "event_as_of": event_as_of,
                    "driver_id": driver,
                    "team_name": "TEAM_1" if grid <= 2 else "TEAM_2",
                    "power_unit": "PU_1" if grid <= 2 else "PU_2",
                    "circuit_id": f"CIRCUIT_{event}",
                    "grid_position": grid,
                    "grid_status": "grid",
                    "grid_starter_eligible": True,
                    "qualy_position": grid,
                    "qualy_pred_rank": grid + (1 if driver == "A" else 0),
                    "finish_position": grid,
                    "terminal_status": statuses[grid - 1],
                    "retirement_fraction": 1.0 if grid <= 2 else 0.45 + (0.05 * event),
                    "team_strength_score": 1.0 if grid <= 2 else 0.0,
                    "driver_strength_score": float(5 - grid),
                    "long_run_pace_delta": float(grid) / 10.0,
                    "longest_clean_stint_laps": 12 - grid,
                    "long_run_evidence_share": 0.8,
                    "long_run_uncertainty": 0.2,
                    "track_finish_order_mobility": 0.4,
                }
            )
    return pd.DataFrame.from_records(records)


def _final_roster() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": ["A", "B", "C", "D"],
            "team_name": ["TEAM_1", "TEAM_1", "TEAM_2", "TEAM_2"],
            "power_unit": ["PU_1", "PU_1", "PU_2", "PU_2"],
            "circuit_id": ["CURRENT"] * 4,
            "grid_position": [1, 2, np.nan, np.nan],
            "grid_status": ["grid", "grid", "pit_lane", "withdrawn"],
            "grid_starter_eligible": [True, True, True, False],
            "grid_pit_lane_start": [False, False, True, False],
            "grid_snapshot_available": [True] * 4,
            "grid_evidence_complete": [True] * 4,
            "grid_resolution_status": ["resolved"] * 4,
            "race_information_horizon": [RacePredictionHorizon.POST_GRID_PRE_RACE.value] * 4,
            "feature_as_of": ["2025-04-30T12:00:00Z"] * 4,
            "qualy_position": [1, 2, 3, 4],
            "qualy_pred_rank": [2, 2, 3, 4],
            "team_strength_score": [1.0, 1.0, 0.0, 0.0],
            "driver_strength_score": [4.0, 3.0, 2.0, 1.0],
            "long_run_pace_delta": [0.1, 0.2, 0.3, 0.4],
            "longest_clean_stint_laps": [12, 11, 10, 9],
            "long_run_evidence_share": [0.8] * 4,
            "long_run_uncertainty": [0.2] * 4,
            "track_finish_order_mobility": [0.4] * 4,
        }
    )


def test_final_grid_snapshot_is_immutable_provenance_and_starter_aware() -> None:
    snapshot = build_race_grid_snapshot(_resolution(with_final_grid=True))

    assert snapshot.horizon is RacePredictionHorizon.POST_GRID_PRE_RACE
    assert snapshot.available is True
    assert snapshot.publication_as_of == "2025-07-06T12:30:00Z"
    assert snapshot.revision_ids == ("fia-final-grid-v2",)
    by_driver = {entry.driver_id: entry for entry in snapshot.entries}
    assert by_driver[DRIVERS[18]].starter_eligible is True
    assert by_driver[DRIVERS[18]].pit_lane_start is True
    assert by_driver[DRIVERS[19]].starter_eligible is False
    assert by_driver[DRIVERS[19]].status is GridEntryStatus.WITHDRAWN
    frame = race_grid_snapshot_frame(snapshot)
    assert set(frame["race_information_horizon"]) == {"post_grid_pre_race"}


def test_missing_final_grid_fails_closed_but_postqual_proxy_stays_separate() -> None:
    resolution = _resolution(with_final_grid=False)
    final_snapshot = build_race_grid_snapshot(resolution)
    proxy_snapshot = build_race_grid_snapshot(
        resolution,
        horizon=RacePredictionHorizon.POST_QUALIFYING_PRE_GRID,
    )

    assert final_snapshot.available is False
    with pytest.raises(ValueError, match="post_grid_pre_race snapshot is unavailable"):
        final_snapshot.require_available()
    assert proxy_snapshot.available is True
    assert proxy_snapshot.horizon is RacePredictionHorizon.POST_QUALIFYING_PRE_GRID
    assert proxy_snapshot.source.value == "qualifying_classification"


def test_grid_penalty_reason_is_preserved_and_keeps_final_product_fail_closed() -> None:
    resolution = resolve_grand_prix_start(
        build_weekend_contract(2025),
        as_of="2025-07-06T13:00:00Z",
        classifications=(_classification(),),
        official_grid_revisions=(
            OfficialGridRevision(
                revision_id="grid-before-penalty",
                phase=GridRevisionPhase.FINAL_PRE_RACE,
                entries=tuple(
                    OfficialGridEntry(driver_id=driver, position=position)
                    for position, driver in enumerate(DRIVERS, start=1)
                ),
                as_of="2025-07-06T12:00:00Z",
            ),
        ),
        adjustments=(
            GridAdjustment(
                driver_id=DRIVERS[0],
                kind=GridAdjustmentKind.GRID_DROP,
                places=5,
                reason="power-unit element change",
                as_of="2025-07-06T12:30:00Z",
                evidence_id="fia-decision-42",
            ),
        ),
    )
    snapshot = build_race_grid_snapshot(resolution)
    entry = next(row for row in snapshot.entries if row.driver_id == DRIVERS[0])
    assert snapshot.available is False
    assert entry.penalty_evidence[0].reason == "power-unit element change"


def test_feature_engineering_preserves_signed_qualifying_surprise() -> None:
    features = engineer_survival_aware_race_features(
        pd.DataFrame(
            {
                "driver_id": ["A", "B"],
                "grid_position": [1, 2],
                "grid_status": ["grid", "grid"],
                "qualy_position": [1, 5],
                "qualy_pred_rank": [4, 2],
                "track_finish_order_mobility": [0.2, 0.2],
            }
        )
    )

    assert features["race_signed_qualifying_surprise"].tolist() == [-3, 3]
    assert features["race_signed_qualifying_surprise_score"].tolist() == [3, -3]
    assert features["race_grid_prior_score"].iloc[0] > features["race_grid_prior_score"].iloc[1]


def test_unknown_terminal_status_target_fails_closed() -> None:
    with pytest.raises(ValueError, match="failed closed"):
        add_reason_coded_terminal_targets(
            pd.DataFrame({"race_status_raw": ["Finished", "mystery_status"]})
        )


@pytest.mark.parametrize("missing", [None, pd.NA, np.nan])
def test_missing_terminal_status_scalar_returns_none_without_ambiguous_truth(missing: object) -> None:
    assert reason_code_terminal_status(missing) is None


@pytest.mark.parametrize("raw", ["Lapped", "+1 Lap", "+4 Laps", "Finished"])
def test_real_classified_status_variants_are_not_mislabeled_terminal(raw: str) -> None:
    assert reason_code_terminal_status(raw) is TerminalStatus.CLASSIFIED_FINISH


def test_partial_pooled_hazard_forces_published_withdrawal_to_dns() -> None:
    model = PartialPooledTerminalHazard().fit(
        _history(),
        cutoff="2025-04-01T00:00:00Z",
    )
    roster = _final_roster()
    probabilities = model.predict_proba(
        roster,
        prediction_as_of="2025-05-01T00:00:00Z",
    ).set_index("driver_id")

    assert probabilities.loc["D", "p_dns_withdrawal"] == pytest.approx(1.0)
    assert probabilities.loc["D", "p_classified_finish"] == pytest.approx(0.0)
    assert probabilities.loc["A", [f"p_{status.value}" for status in TerminalStatus]].sum() == pytest.approx(1.0)


def test_terminal_evaluation_reports_brier_logloss_calibration_and_reason_recall() -> None:
    model = PartialPooledTerminalHazard().fit(
        _history(),
        cutoff="2025-04-01T00:00:00Z",
    )
    roster = _final_roster()
    probabilities = model.predict_proba(
        roster,
        prediction_as_of="2025-05-01T00:00:00Z",
    )
    actual = pd.DataFrame(
        {
            "driver_id": ["A", "B", "C", "D"],
            "terminal_status": [
                "classified_finish",
                "classified_finish",
                "mechanical_power_unit",
                "dns_withdrawal",
            ],
        }
    )
    metrics = evaluate_terminal_status_probabilities(actual, probabilities)
    assert metrics["rows"] == 4
    assert float(metrics["multiclass_brier"]) >= 0.0
    assert float(metrics["terminal_brier"]) >= 0.0
    assert float(metrics["multiclass_log_loss"]) >= 0.0
    assert isinstance(metrics["terminal_calibration"], list)


def test_terminal_evaluation_separates_exact_coarse_and_retirement_timing() -> None:
    model = PartialPooledTerminalHazard().fit(
        _history(),
        cutoff="2025-04-01T00:00:00Z",
    )
    roster = _final_roster()
    probabilities = model.predict_proba(
        roster,
        prediction_as_of="2025-05-01T00:00:00Z",
    )
    actual = pd.DataFrame(
        {
            "driver_id": ["A", "B", "C", "D"],
            "terminal_status": [
                "classified_finish",
                "non_classified",
                "mechanical_power_unit",
                "dns_withdrawal",
            ],
            "terminal_label_granularity": [
                "classified",
                "coarse_terminal",
                "exact_cause",
                "prestart",
            ],
            "retirement_fraction": [1.0, 0.7, 0.4, 0.0],
        }
    )
    metrics = evaluate_terminal_status_probabilities(actual, probabilities)

    assert metrics["exact_reason_rows"] == 1
    assert metrics["coarse_terminal_rows"] == 1
    assert metrics["retirement_timing_rows"] == 2
    assert metrics["retirement_fraction_mae"] is not None


def test_joint_model_emits_probabilities_and_legal_permutation_with_dns_and_pitlane() -> None:
    model = SurvivalAwareRaceModel().fit(
        _history(),
        cutoff="2025-04-01T00:00:00Z",
    )
    forecast = model.predict_joint(
        _final_roster(),
        prediction_as_of="2025-05-01T00:00:00Z",
        simulations=300,
        seed=9,
    )

    point = forecast.point_classification.set_index("driver_id")
    assert sorted(point["predicted_position"].tolist()) == [1, 2, 3, 4]
    assert sorted(point["global_meal_position"].tolist()) == [1, 2, 3, 4]
    assert point["dns_constrained_meal_position"].tolist() == point[
        "predicted_position"
    ].tolist()
    assert point.loc["D", "predicted_terminal_status"] == "dns_withdrawal"
    assert int(point.loc["D", "predicted_position"]) == 4
    assert point.loc["C", "starter_eligible"]
    assert set(point["classified_lap_deficit_source"]) == {
        "zero_no_calibrated_evidence"
    }
    position_sums = forecast.position_probabilities.filter(like="p_position_").sum(axis=1)
    status_sums = forecast.status_probabilities.filter(regex=r"^p_(?!terminal$)").sum(axis=1)
    assert position_sums.tolist() == pytest.approx([1.0] * 4)
    assert status_sums.tolist() == pytest.approx([1.0] * 4)
    assert set(forecast.status_probabilities["status_probability_source"]) == {
        "empirical_post_shared_shock_joint_samples"
    }
    for status in TerminalStatus:
        empirical = np.mean(forecast.status_samples == status.value, axis=1)
        assert forecast.status_probabilities[f"p_{status.value}"].to_numpy() == pytest.approx(
            empirical
        )


def test_person_period_hazard_learns_regularized_causal_covariate_effects() -> None:
    rows = []
    for event in range(1, 13):
        for driver in range(8):
            mechanical = driver < 2
            incident = 2 <= driver < 4
            coarse = driver == 4
            status = (
                TerminalStatus.MECHANICAL_POWER_UNIT.value
                if mechanical
                else TerminalStatus.COLLISION_INCIDENT.value
                if incident
                else TerminalStatus.NON_CLASSIFIED.value
                if coarse
                else TerminalStatus.CLASSIFIED_FINISH.value
            )
            rows.append(
                {
                    "event_key": event,
                    "event_as_of": f"2024-{event:02d}-01T12:00:00Z",
                    "driver_id": f"D{driver}",
                    "team_name": "SAME_TEAM",
                    "power_unit": "SAME_PU",
                    "circuit_id": "SAME_TRACK",
                    "grid_position": driver + 1,
                    "grid_status": "grid",
                    "grid_starter_eligible": True,
                    "terminal_status": status,
                    "retirement_fraction": (
                        1.0 if status == "classified_finish" else 0.15 + 0.05 * driver
                    ),
                    "race_team_mechanical_rate": 0.9 if mechanical else 0.1,
                    "race_power_unit_mechanical_rate": 0.9 if mechanical else 0.1,
                    "race_driver_incident_rate": 0.9 if incident else 0.1,
                    "race_circuit_dnf_rate": 0.4,
                }
            )
    model = PartialPooledTerminalHazard().fit(pd.DataFrame(rows))
    card = model.model_card
    fitted = set(card["covariate_model"]["fitted_causes"])

    assert {
        "mechanical_power_unit",
        "collision_incident",
        "non_classified",
    }.issubset(fitted)
    assert (
        card["covariate_model"]["coefficients"]["mechanical_power_unit"][
            "race_team_mechanical_rate"
        ]
        > 0.0
    )
    predicted = model.predict_proba(pd.DataFrame(rows[:2]))
    assert len(predicted.filter(regex=r"^terminal_interval_hazard_").columns) == 12
    assert len(predicted.filter(regex=r"^survival_through_interval_").columns) == 12


def test_calibration_mapping_changes_terminal_hazard_without_breaking_probability_sum() -> None:
    model = PartialPooledTerminalHazard().fit(
        _history(),
        cutoff="2025-04-01T00:00:00Z",
    )
    roster = _final_roster().iloc[:3].copy()
    raw = model.predict_proba(
        roster,
        prediction_as_of="2025-05-01T00:00:00Z",
    )
    model.set_terminal_calibrator(
        BinaryTerminalCalibrator(
            intercept=-1.5,
            slope=0.7,
            calibration_rows=80,
            calibration_event_keys=("202501", "202502", "202503", "202504"),
        )
    )
    calibrated = model.predict_proba(
        roster,
        prediction_as_of="2025-05-01T00:00:00Z",
    )

    assert not np.allclose(raw["p_terminal"], calibrated["p_terminal"])
    assert calibrated.filter(regex=r"^p_(?!terminal$)").sum(axis=1).tolist() == pytest.approx(
        [1.0] * len(calibrated)
    )
    assert model.model_card["binary_terminal_calibration"]["calibration_rows"] == 80


def test_joint_all_terminal_roster_remains_a_legal_permutation() -> None:
    model = SurvivalAwareRaceModel().fit(
        _history(),
        cutoff="2025-04-01T00:00:00Z",
    )
    roster = _final_roster()
    roster["grid_status"] = "withdrawn"
    roster["grid_starter_eligible"] = False
    roster["grid_pit_lane_start"] = False
    roster["grid_position"] = np.nan
    forecast = model.predict_joint(
        roster,
        prediction_as_of="2025-05-01T00:00:00Z",
        simulations=100,
        seed=5,
    )

    point = forecast.point_classification
    assert sorted(point["predicted_position"].tolist()) == [1, 2, 3, 4]
    assert set(point["predicted_terminal_status"]) == {"dns_withdrawal"}
    assert np.all(np.sort(forecast.position_samples, axis=0) == np.arange(1, 5)[:, None])


def test_all_classified_distance_preserves_grid_prior_instead_of_random_beta_laps() -> None:
    statuses = [TerminalStatus.CLASSIFIED_FINISH] * 4
    classification, distance = sample_fia_classification_order(
        statuses=statuses,
        # Deliberately extreme values: classified finishers must ignore these
        # terminal-hazard draws completely.
        terminal_retirement_fraction=np.asarray([0.1, 0.9, 0.4, 0.7]),
        conditional_scores=np.asarray([4.0, 3.0, 2.0, 1.0]),
        order_shocks=np.zeros(4),
        expected_lap_deficit=np.zeros(4),
        scheduled_laps=np.full(4, 60.0),
        grid_positions=np.arange(1.0, 5.0),
        driver_ids=np.asarray(["A", "B", "C", "D"]),
    )

    assert classification.tolist() == [0, 1, 2, 3]
    assert distance.tolist() == [60.0, 60.0, 60.0, 60.0]


def test_late_retiree_can_rank_ahead_of_pace_coupled_lapped_finisher() -> None:
    classification, distance = sample_fia_classification_order(
        statuses=[
            TerminalStatus.CLASSIFIED_FINISH,
            TerminalStatus.MECHANICAL_POWER_UNIT,
            TerminalStatus.CLASSIFIED_FINISH,
        ],
        terminal_retirement_fraction=np.asarray([1.0, 0.99, 1.0]),
        conditional_scores=np.asarray([3.0, 2.0, 1.0]),
        order_shocks=np.zeros(3),
        expected_lap_deficit=np.asarray([0.0, 0.0, 2.0]),
        scheduled_laps=np.full(3, 60.0),
        grid_positions=np.arange(1.0, 4.0),
        driver_ids=np.asarray(["WIN", "RET", "LAP"]),
    )

    assert distance.tolist() == pytest.approx([60.0, 59.4, 58.0])
    assert classification.tolist() == [0, 1, 2]


def test_classified_lap_deficit_is_derived_from_accumulated_long_run_pace() -> None:
    features = pd.DataFrame(
        {
            "race_long_run_pace_delta": [0.0, 1.5, np.nan],
            "race_expected_lap_seconds": [90.0, 90.0, 90.0],
        }
    )
    deficit = expected_classified_lap_deficit(
        features,
        np.full(3, 60.0),
        allow_pace_implied=True,
    )
    assert deficit.tolist() == pytest.approx([0.0, 1.0, 0.0])


def test_expected_absolute_assignment_is_globally_optimal_and_deterministic() -> None:
    samples = np.asarray(
        [
            [1, 1, 3, 3],
            [1, 2, 2, 3],
            [2, 2, 3, 1],
        ]
    )
    assignment = minimum_expected_absolute_assignment(samples)
    cost = sum(np.mean(np.abs(samples[row] - assignment[row])) for row in range(3))
    brute_costs = [
        sum(np.mean(np.abs(samples[row] - candidate[row])) for row in range(3))
        for candidate in permutations((1, 2, 3))
    ]

    assert cost == pytest.approx(min(brute_costs))
    assert minimum_expected_absolute_assignment(samples).tolist() == assignment.tolist()


def test_pairwise_ties_are_ranked_deterministically_by_driver_id() -> None:
    ranker = BradleyTerryOrderRanker().fit(_history())
    tied = pd.DataFrame(
        {
            "driver_id": ["B", "A"],
            "grid_position": [1, 1],
            "grid_status": ["grid", "grid"],
            "grid_starter_eligible": [True, True],
            "qualy_position": [1, 1],
            "qualy_pred_rank": [1, 1],
        }
    )
    scored = ranker.score(tied).set_index("driver_id")
    assert scored.loc["A", "conditional_order_rank"] == 1
    assert scored.loc["B", "conditional_order_rank"] == 2


def test_local_provider_preserves_status_and_lap_targets(tmp_path) -> None:
    weekend = tmp_path / "2025" / "round_01_test_grand_prix"
    weekend.mkdir(parents=True)
    (weekend / "weekend_metadata.json").write_text(
        """{
          "round_number": 1,
          "event_name": "Test Grand Prix",
          "sessions": [{"session_type": "Race", "results_path": "race_results.csv"}]
        }""",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "DriverNumber": [1, 2, 3],
            "Position": [1, 2, 3],
            "GridPosition": [1, 2, 3],
            "Status": ["Finished", "+1 Lap", "Engine"],
            "Laps": [60, 59, 30],
        }
    ).to_csv(weekend / "race_results.csv", index=False)

    result = LocalWeekendProvider(weekends_dir=str(tmp_path)).get_race_results(2025, 1)
    assert result["race_status_raw"].tolist() == ["Finished", "+1 Lap", "Engine"]
    assert result["race_status_evidence_complete"].all()
    assert result["retirement_fraction"].tolist() == pytest.approx([1.0, 59 / 60, 0.5])
