from copy import deepcopy

import pytest

import f1_prediction_service.model as model_module
from f1_prediction_service.model import TARGET_MODELS, predict_from_snapshot


def test_predict_from_snapshot_returns_prediction_contract():
    result = predict_from_snapshot(_snapshot())

    assert result["modelVersion"] == TARGET_MODELS["race"]["model_version"]
    assert result["featuresVersion"] == TARGET_MODELS["race"]["features_version"]
    assert result["diagnostics"]["driverCount"] == 3
    assert len(result["predictions"]) == 3

    first = result["predictions"][0]
    assert first["driver_number"] == 1
    assert first["model_version"] == TARGET_MODELS["race"]["model_version"]
    assert first["features_version"] == TARGET_MODELS["race"]["features_version"]
    assert 1.0 <= first["expected_position"] <= 3.0
    assert 1 <= first["position_p10"] <= first["position_p90"] <= 3
    assert first["win_probability"] > 0.0
    assert first["podium_probability"] >= first["win_probability"]
    assert first["points_probability"] >= first["podium_probability"]
    assert first["strategy"]["recommendedAction"] in {"stay_out", "pit_next_lap", "pit_now"}
    assert first["strategy"]["safeToRecommend"] is True
    assert first["strategy"]["paceMode"] is None
    assert first["strategy"]["legalActionKey"] is None
    assert first["strategy"]["compatibleLegalActionKeys"]
    assert set(first["strategy"]["compatibleLegalActionKeys"]).issubset(
        first["strategy"]["legalActionMask"]["legal_action_keys"]
    )
    assert "conditional_classification_order" in first["position_semantics"]
    assert result["diagnostics"]["calibrationStatus"] == "uncalibrated_heuristic_not_validated_for_promotion"
    assert result["diagnostics"]["promotionStatus"] == "not_promoted"
    _assert_joint_invariants(result)


def test_twenty_driver_field_has_coherent_joint_marginals():
    result = predict_from_snapshot(_twenty_driver_snapshot())

    assert len(result["predictions"]) == 20
    _assert_joint_invariants(result, tolerance=1e-8)
    diagnostics = result["diagnostics"]["jointDistribution"]
    assert diagnostics["method"] == "sinkhorn_balanced_score_position_kernel"
    assert diagnostics["magnitudeSensitive"] is True
    assert diagnostics["rowMaxAbsError"] < 1e-8
    assert diagnostics["columnMaxAbsError"] < 1e-8


def test_joint_rank_probabilities_respond_to_score_gap_magnitude():
    close = deepcopy(_snapshot())
    wide = deepcopy(_snapshot())
    for driver, lap_time in zip(close["drivers"], (70.0, 70.02, 70.04)):
        driver["best_lap_time"] = lap_time
    for driver, lap_time in zip(wide["drivers"], (70.0, 71.0, 72.0)):
        driver["best_lap_time"] = lap_time

    close_result = predict_from_snapshot(close, prediction_kind="qualifying")
    wide_result = predict_from_snapshot(wide, prediction_kind="qualifying")
    close_by_driver = {row["driver_number"]: row for row in close_result["predictions"]}
    wide_by_driver = {row["driver_number"]: row for row in wide_result["predictions"]}

    assert wide_by_driver[1]["win_probability"] > close_by_driver[1]["win_probability"] + 0.25
    assert wide_by_driver[16]["win_probability"] < close_by_driver[16]["win_probability"]
    _assert_joint_invariants(close_result)
    _assert_joint_invariants(wide_result)

    close_race = deepcopy(_snapshot())
    wide_race = deepcopy(_snapshot())
    for driver, gap in zip(close_race["drivers"], (0.0, 0.1, 0.2)):
        driver["gap_to_leader"] = str(gap)
    for driver, gap in zip(wide_race["drivers"], (0.0, 30.0, 60.0)):
        driver["gap_to_leader"] = str(gap)
    close_race_result = predict_from_snapshot(close_race, prediction_kind="race")
    wide_race_result = predict_from_snapshot(wide_race, prediction_kind="race")
    close_race_by_driver = {row["driver_number"]: row for row in close_race_result["predictions"]}
    wide_race_by_driver = {row["driver_number"]: row for row in wide_race_result["predictions"]}
    assert wide_race_by_driver[1]["win_probability"] > close_race_by_driver[1]["win_probability"]
    _assert_joint_invariants(close_race_result)
    _assert_joint_invariants(wide_race_result)


def test_retired_driver_remains_race_classification_eligible_and_may_score():
    snapshot = _snapshot()
    snapshot["drivers"][0]["pit_status"] = "retired"
    snapshot["drivers"][0]["is_running"] = False

    result = predict_from_snapshot(snapshot, prediction_kind="race")
    by_driver = {row["driver_number"]: row for row in result["predictions"]}
    retired = by_driver[1]

    assert retired["participation_status"] == "retired_or_stopped"
    assert retired["eligibility_status"] == "classification_eligible_retired"
    assert retired["forecast_available"] is True
    assert retired["unavailable_reason"] is None
    assert set(retired["position_distribution"]) == {"1", "2", "3"}
    assert 1.0 <= retired["expected_position"] <= 3.0
    assert retired["win_probability"] > 0.0
    assert retired["podium_probability"] >= retired["win_probability"]
    assert retired["points_probability"] >= retired["podium_probability"]
    assert retired["dnf_probability"] == 1.0
    assert retired["strategy"]["safeToRecommend"] is False
    assert retired["strategy"]["unavailableReason"] == "known_not_running"
    assert result["diagnostics"]["jointDistribution"]["eligibleDriverCount"] == 3
    _assert_joint_invariants(result)


@pytest.mark.parametrize(
    ("raw_status", "participation_status", "reason"),
    [
        ("DNS", "dns", "classification_ineligible:dns"),
        ("Disqualified", "dsq", "classification_ineligible:dsq"),
        ("Excluded from classification", "dsq", "classification_ineligible:dsq"),
        ("Withdrawn", "withdrawn", "classification_ineligible:withdrawn"),
        ("Not classified", "classification_ineligible", "classification_ineligible:unclassified"),
    ],
)
def test_classification_exclusions_are_unavailable_not_fake_tail_rows(
    raw_status,
    participation_status,
    reason,
):
    snapshot = _snapshot()
    snapshot["drivers"][0]["status"] = raw_status

    result = predict_from_snapshot(snapshot, prediction_kind="race")
    excluded = next(row for row in result["predictions"] if row["driver_number"] == 1)

    assert excluded["participation_status"] == participation_status
    assert excluded["eligibility_status"] == "target_unavailable"
    assert excluded["forecast_available"] is False
    assert excluded["unavailable_reason"] == reason
    assert excluded["position_distribution"] == {}
    assert excluded["expected_position"] is None
    assert excluded["position_p10"] is None
    assert excluded["position_p90"] is None
    assert excluded["win_probability"] == 0.0
    assert excluded["podium_probability"] == 0.0
    assert excluded["points_probability"] == 0.0
    assert excluded["dnf_probability"] == 0.0
    _assert_joint_invariants(result)


def test_stopped_qualifying_driver_keeps_valid_lap_but_not_a_missing_lap():
    with_lap = _snapshot()
    with_lap["drivers"][0]["status"] = "Stopped"
    with_lap["drivers"][0]["is_running"] = False

    available = predict_from_snapshot(with_lap, prediction_kind="qualifying")
    stopped = next(row for row in available["predictions"] if row["driver_number"] == 1)
    assert stopped["participation_status"] == "retired_or_stopped"
    assert stopped["eligibility_status"] == "classification_eligible_retired"
    assert stopped["forecast_available"] is True
    assert stopped["position_distribution"]
    _assert_joint_invariants(available)

    without_lap = deepcopy(with_lap)
    without_lap["drivers"][0]["last_lap_time"] = None
    without_lap["drivers"][0]["best_lap_time"] = None
    unavailable = predict_from_snapshot(without_lap, prediction_kind="qualifying")
    stopped_without_lap = next(
        row for row in unavailable["predictions"] if row["driver_number"] == 1
    )
    assert stopped_without_lap["forecast_available"] is False
    assert stopped_without_lap["unavailable_reason"] == "qualifying_unavailable:no_valid_lap"
    assert stopped_without_lap["position_distribution"] == {}
    _assert_joint_invariants(unavailable)


def test_retired_and_finished_drivers_have_no_next_lap_forecast():
    snapshot = _snapshot()
    snapshot["drivers"][0]["status"] = "Retired"
    snapshot["drivers"][0]["is_running"] = False
    snapshot["drivers"][1]["status"] = "Finished"
    snapshot["drivers"][1]["is_running"] = False

    result = predict_from_snapshot(snapshot, prediction_kind="next-lap")
    by_driver = {row["driver_number"]: row for row in result["predictions"]}

    assert by_driver[1]["participation_status"] == "retired_or_stopped"
    assert by_driver[1]["unavailable_reason"] == "next_lap_unavailable:retired_or_stopped"
    assert by_driver[63]["participation_status"] == "finished"
    assert by_driver[63]["unavailable_reason"] == "next_lap_unavailable:finished"
    for number in (1, 63):
        assert by_driver[number]["forecast_available"] is False
        assert by_driver[number]["position_distribution"] == {}
        assert by_driver[number]["expected_position"] is None
    assert by_driver[16]["forecast_available"] is True
    assert by_driver[16]["position_distribution"] == {"1": 1.0}
    _assert_joint_invariants(result)


def test_next_lap_excludes_large_retired_block_and_stays_jointly_coherent():
    snapshot = _twenty_driver_snapshot()
    for driver in snapshot["drivers"][:13]:
        driver["pit_status"] = "retired"
        driver["is_running"] = False
        driver["current_lap"] -= driver["driver_number"] % 3
    snapshot["drivers"][13]["last_lap_time"] = 70.0
    snapshot["drivers"][14]["last_lap_time"] = 95.0

    result = predict_from_snapshot(snapshot, prediction_kind="next-lap")

    diagnostics = result["diagnostics"]["jointDistribution"]
    assert diagnostics["eligibleDriverCount"] == 7
    assert diagnostics["unavailableDriverCount"] == 13
    assert diagnostics["rowMaxAbsError"] < 1e-8
    assert diagnostics["columnMaxAbsError"] < 1e-8
    assert diagnostics["iterations"] < 1000
    _assert_joint_invariants(result, tolerance=1e-8)


def test_targets_use_distinct_models_and_inputs():
    snapshot = _snapshot()
    snapshot["drivers"][0]["best_lap_time"] = 70.0
    snapshot["drivers"][0]["last_lap_time"] = 70.2
    snapshot["drivers"][2]["best_lap_time"] = 67.5
    snapshot["drivers"][2]["last_lap_time"] = 67.7

    race = predict_from_snapshot(snapshot, prediction_kind="race")
    qualifying = predict_from_snapshot(snapshot, prediction_kind="qualifying")
    next_lap = predict_from_snapshot(snapshot, prediction_kind="next-lap")
    strategy = predict_from_snapshot(snapshot, prediction_kind="strategy")

    assert race["predictions"][0]["driver_number"] == 1
    assert qualifying["predictions"][0]["driver_number"] == 16
    assert next_lap["predictions"][0]["driver_number"] == 16
    model_versions = {
        race["modelVersion"],
        qualifying["modelVersion"],
        next_lap["modelVersion"],
        strategy["modelVersion"],
    }
    assert len(model_versions) == 4
    assert qualifying["predictions"][0]["strategy"] is None
    assert qualifying["diagnostics"]["strategyPolicyEnabled"] is False
    assert "packages_f1_live_strategy_v1" not in str((race, qualifying, next_lap, strategy))
    for result in (race, qualifying, next_lap, strategy):
        _assert_joint_invariants(result)


def test_provenance_is_honest_about_canonical_package_boundary():
    result = predict_from_snapshot(_snapshot(), prediction_kind="race")
    provenance = result["diagnostics"]["provenance"]

    assert provenance["modelType"] == "deterministic_uncalibrated_snapshot_heuristic"
    assert provenance["canonicalLiveRaceModelUsed"] is False
    assert "causal_lap_history_trace" in provenance["canonicalLiveRaceUnavailableReason"]
    assert provenance["promotionEligible"] is False
    if result["diagnostics"]["strategyPolicyEnabled"]:
        assert provenance["canonicalPackageComponents"] == [
            "packages.f1.models.live_race.strategy.BaselineStrategyPolicyAdapter"
        ]


def test_unsupported_prediction_kind_fails_closed():
    with pytest.raises(ValueError, match="unsupported prediction_kind"):
        predict_from_snapshot(_snapshot(), prediction_kind="one-model-for-everything")


def test_duplicate_driver_numbers_fail_closed():
    snapshot = _snapshot()
    snapshot["drivers"][1]["driver_number"] = snapshot["drivers"][0]["driver_number"]

    with pytest.raises(ValueError, match="duplicate driver_number"):
        predict_from_snapshot(snapshot)


def test_strategy_pressure_reduces_overaged_driver_ranking():
    snapshot = _snapshot()
    snapshot["drivers"][0]["tyre_age"] = 36
    snapshot["drivers"][0]["current_compound"] = "SOFT"

    result = predict_from_snapshot(snapshot)
    ranks = {item["driver_number"]: item["expected_position"] for item in result["predictions"]}
    strategy = {item["driver_number"]: item["strategy"] for item in result["predictions"]}

    assert ranks[1] > 1.0
    assert strategy[1]["pitUrgency"] is None or strategy[1]["pitUrgency"] > 0.0


def test_package_strategy_adapter_path_when_pandas_available():
    pytest.importorskip("pandas")

    result = predict_from_snapshot(_snapshot())

    assert result["diagnostics"]["strategyPolicyEnabled"] is True
    assert result["diagnostics"]["strategyPolicyError"] is None
    assert result["predictions"][0]["strategy"]["policyVersion"] == "deterministic_baseline_v1"


def test_strategy_adapter_failure_is_reported_and_uses_honest_fallback(monkeypatch):
    def unavailable():
        raise ImportError("canonical package unavailable in isolated service runtime")

    monkeypatch.setattr(model_module, "_load_strategy_policy_adapter", unavailable)

    result = predict_from_snapshot(_snapshot())

    assert result["diagnostics"]["strategyPolicyEnabled"] is False
    assert "ImportError" in result["diagnostics"]["strategyPolicyError"]
    assert result["diagnostics"]["provenance"]["fallbackStrategyPolicyUsed"] is True
    assert result["diagnostics"]["provenance"]["canonicalPackageComponents"] == []
    assert result["predictions"][0]["strategy"]["policyVersion"] == "fallback_strategy_policy_v1"
    _assert_joint_invariants(result)


def test_strategy_legality_evidence_covers_shared_mask_state():
    result = predict_from_snapshot(_snapshot(), prediction_kind="strategy")

    for prediction in result["predictions"]:
        strategy = prediction["strategy"]
        assert strategy["safeToRecommend"] is True
        assert strategy["paceMode"] is None
        assert strategy["legalActionKey"] is None
        assert strategy["compatibleLegalActionKeys"]
        assert set(strategy["compatibleLegalActionKeys"]).issubset(
            strategy["legalActionMask"]["legal_action_keys"]
        )
        assert strategy["legalityState"] == {
            "lapNumber": 23,
            "totalLaps": 57,
            "remainingLaps": 34,
            "tyreAge": 11,
            "stintNumber": 1,
            "currentCompound": next(
                driver["current_compound"]
                for driver in _snapshot()["drivers"]
                if driver["driver_number"] == prediction["driver_number"]
            ),
            "usedCompounds": next(
                driver["used_compounds"]
                for driver in _snapshot()["drivers"]
                if driver["driver_number"] == prediction["driver_number"]
            ),
            "availableCompounds": ["SOFT", "MEDIUM", "HARD"],
            "pitLaneOpen": True,
            "isRed": False,
            "isWetTrack": False,
            "isBoxLap": False,
        }
    assert result["diagnostics"]["forecastAvailable"] is False
    assert result["diagnostics"]["strategyRecommendationAvailableCount"] == 3


def test_explicit_pace_mode_is_the_only_path_to_a_singular_legal_action_key():
    driver = model_module._drivers(_snapshot())[0]
    legality_state, missing = model_module._strategy_legality_state(driver)
    assert missing == []

    coarse = model_module._apply_shared_legal_action_mask(
        {"recommended_action": "stay_out"},
        legality_state,
    )
    explicit = model_module._apply_shared_legal_action_mask(
        {"recommended_action": "stay_out", "action_mode": "aggressive"},
        legality_state,
    )

    assert coarse["pace_mode"] is None
    assert coarse["legal_action_key"] is None
    assert coarse["compatible_legal_action_keys"] == [
        "stay_out:conservative",
        "stay_out:aggressive",
    ]
    assert explicit["pace_mode"] == "aggressive"
    assert explicit["compatible_legal_action_keys"] == ["stay_out:aggressive"]
    assert explicit["legal_action_key"] == "stay_out:aggressive"


@pytest.mark.parametrize(
    ("missing_case", "missing_field"),
    [
        ("current_lap", "current_lap"),
        ("total_laps", "total_laps"),
        ("remaining_laps", "remaining_laps"),
        ("tyre_age", "tyre_age"),
        ("stint", "stint_number"),
        ("compound", "current_compound"),
        ("used_compounds", "used_compounds"),
        ("available_compounds", "available_compounds"),
        ("pit_lane", "pit_lane_open"),
        ("red", "red_flag_state"),
        ("wet", "wet_track_state"),
        ("box", "box_or_pit_state"),
    ],
)
def test_strategy_is_explicitly_unavailable_when_critical_legality_state_is_missing(
    missing_case,
    missing_field,
):
    snapshot = _snapshot()
    if missing_case == "current_lap":
        for driver in snapshot["drivers"]:
            driver.pop("current_lap")
    elif missing_case == "total_laps":
        snapshot.pop("total_laps")
    elif missing_case == "remaining_laps":
        snapshot.pop("remaining_laps")
    elif missing_case == "tyre_age":
        for driver in snapshot["drivers"]:
            driver.pop("tyre_age")
    elif missing_case == "stint":
        for driver in snapshot["drivers"]:
            driver.pop("stint_number")
    elif missing_case == "compound":
        for driver in snapshot["drivers"]:
            driver["current_compound"] = "UNKNOWN"
    elif missing_case == "used_compounds":
        for driver in snapshot["drivers"]:
            driver.pop("used_compounds")
            driver["stint_number"] = 2
    elif missing_case == "available_compounds":
        snapshot.pop("available_compounds")
    elif missing_case == "pit_lane":
        snapshot.pop("pit_lane_open")
    elif missing_case == "red":
        snapshot["raceControl"] = []
    elif missing_case == "wet":
        snapshot.pop("is_wet_track")
    elif missing_case == "box":
        for driver in snapshot["drivers"]:
            driver.pop("pit_status")

    result = predict_from_snapshot(snapshot, prediction_kind="strategy")

    for prediction in result["predictions"]:
        strategy = prediction["strategy"]
        assert strategy["recommendedAction"] is None
        assert strategy["safeToRecommend"] is False
        assert strategy["unavailableReason"] == "missing_critical_legality_state"
        assert missing_field in strategy["missingLegalityFields"]
        assert strategy["legalActionMask"] is None
        evidence_key = {
            "current_lap": "lapNumber",
            "total_laps": "totalLaps",
            "remaining_laps": "remainingLaps",
            "tyre_age": "tyreAge",
        }.get(missing_field)
        if evidence_key is not None:
            assert strategy["legalityState"][evidence_key] is None
    assert result["diagnostics"]["strategyRecommendationAvailableCount"] == 0
    assert result["diagnostics"]["strategyRecommendationUnavailableCount"] == 3


def test_illegal_policy_action_is_not_replaced_with_a_fabricated_recommendation():
    snapshot = _snapshot()
    snapshot["pit_lane_open"] = False
    snapshot["drivers"][0]["current_compound"] = "SOFT"
    snapshot["drivers"][0]["used_compounds"] = ["SOFT"]
    snapshot["drivers"][0]["tyre_age"] = 40

    result = predict_from_snapshot(snapshot, prediction_kind="strategy")
    by_driver = {row["driver_number"]: row for row in result["predictions"]}
    strategy = by_driver[1]["strategy"]

    assert strategy["recommendedAction"] is None
    assert strategy["originalRecommendedAction"] in {"pit_now", "pit_next_lap"}
    assert strategy["safeToRecommend"] is False
    assert strategy["unavailableReason"] == "policy_action_illegal:pit_lane_closed"
    assert strategy["legalActionMask"]["legal_action_keys"] == [
        "stay_out:conservative",
        "stay_out:aggressive",
    ]


def test_native_platform_pit_status_blocks_a_second_pit_recommendation():
    snapshot = _snapshot()
    snapshot["drivers"][0]["pit_status"] = "pit lap 23"
    snapshot["drivers"][0]["current_compound"] = "SOFT"
    snapshot["drivers"][0]["used_compounds"] = ["SOFT"]
    snapshot["drivers"][0]["tyre_age"] = 40

    result = predict_from_snapshot(snapshot, prediction_kind="strategy")
    strategy = next(row for row in result["predictions"] if row["driver_number"] == 1)["strategy"]

    assert strategy["recommendedAction"] is None
    assert strategy["originalRecommendedAction"] in {"pit_now", "pit_next_lap"}
    assert strategy["safeToRecommend"] is False
    assert strategy["unavailableReason"] == "policy_action_illegal:already_on_box_lap"
    assert strategy["legalityState"]["isBoxLap"] is True


def test_unknown_compound_inventory_is_not_replaced_by_default_dry_compounds():
    snapshot = _snapshot()
    snapshot["available_compounds"] = ["MYSTERY"]

    result = predict_from_snapshot(snapshot, prediction_kind="strategy")

    for prediction in result["predictions"]:
        strategy = prediction["strategy"]
        assert strategy["safeToRecommend"] is False
        assert strategy["unavailableReason"] == "missing_critical_legality_state"
        assert "available_compounds" in strategy["missingLegalityFields"]
        assert strategy["legalActionMask"] is None


def test_textual_platform_track_status_supplies_red_flag_state():
    snapshot = _snapshot()
    snapshot["raceControl"] = []
    for driver in snapshot["drivers"]:
        driver["track_status"] = "green"

    result = predict_from_snapshot(snapshot, prediction_kind="strategy")

    for prediction in result["predictions"]:
        strategy = prediction["strategy"]
        assert strategy["safeToRecommend"] is True
        assert strategy["legalityState"]["isRed"] is False


def _assert_joint_invariants(result, *, tolerance=2e-6):
    predictions = result["predictions"]
    total = len(predictions)
    assert total > 0
    available = [prediction for prediction in predictions if prediction["forecast_available"]]
    unavailable = [prediction for prediction in predictions if not prediction["forecast_available"]]
    eligible_total = len(available)
    for prediction in predictions:
        distribution = prediction["position_distribution"]
        if not prediction["forecast_available"]:
            assert prediction["eligibility_status"] == "target_unavailable"
            assert prediction["unavailable_reason"]
            assert distribution == {}
            assert prediction["expected_position"] is None
            assert prediction["position_p10"] is None
            assert prediction["position_p90"] is None
            assert prediction["win_probability"] == 0.0
            assert prediction["podium_probability"] == 0.0
            assert prediction["points_probability"] == 0.0
            continue
        assert prediction["eligibility_status"] in {
            "classification_eligible",
            "classification_eligible_retired",
        }
        assert prediction["unavailable_reason"] is None
        assert set(distribution) == {str(position) for position in range(1, eligible_total + 1)}
        assert abs(sum(distribution.values()) - 1.0) < tolerance
        assert prediction["win_probability"] == pytest.approx(distribution["1"], abs=tolerance)
        assert prediction["podium_probability"] == pytest.approx(
            sum(distribution[str(position)] for position in range(1, min(3, eligible_total) + 1)),
            abs=tolerance,
        )
        assert prediction["points_probability"] == pytest.approx(
            sum(distribution[str(position)] for position in range(1, min(10, eligible_total) + 1)),
            abs=tolerance,
        )
    for position in range(1, eligible_total + 1):
        assert sum(row["position_distribution"][str(position)] for row in available) == pytest.approx(
            1.0,
            abs=tolerance,
        )
    assert sum(row["win_probability"] for row in predictions) == pytest.approx(
        min(1, eligible_total),
        abs=tolerance,
    )
    assert sum(row["podium_probability"] for row in predictions) == pytest.approx(
        min(3, eligible_total),
        abs=tolerance,
    )
    assert sum(row["points_probability"] for row in predictions) == pytest.approx(
        min(10, eligible_total),
        abs=tolerance,
    )
    assert result["diagnostics"]["forecastAvailableCount"] == eligible_total
    assert result["diagnostics"]["forecastUnavailableCount"] == len(unavailable)


def _snapshot():
    return {
        "sessionKey": "sample-race",
        "seq": 42,
        "total_laps": 57,
        "remaining_laps": 34,
        "available_compounds": ["SOFT", "MEDIUM", "HARD"],
        "pit_lane_open": True,
        "is_wet_track": False,
        "raceControl": [{"flag": "GREEN", "message": "Track clear"}],
        "drivers": [
            {
                "driver_number": 1,
                "acronym": "VER",
                "full_name": "Max Verstappen",
                "team_name": "Red Bull Racing",
                "position": 1,
                "current_lap": 23,
                "last_lap_time": 68.322,
                "best_lap_time": 68.322,
                "current_compound": "MEDIUM",
                "tyre_age": 11,
                "stint_number": 1,
                "used_compounds": ["MEDIUM"],
                "pit_status": "on_track",
                "gap_to_leader": "0.000",
                "last_speed": 309,
            },
            {
                "driver_number": 63,
                "acronym": "RUS",
                "full_name": "George Russell",
                "team_name": "Mercedes",
                "position": 2,
                "current_lap": 23,
                "last_lap_time": 68.431,
                "best_lap_time": 68.431,
                "current_compound": "MEDIUM",
                "tyre_age": 11,
                "stint_number": 1,
                "used_compounds": ["MEDIUM"],
                "pit_status": "on_track",
                "gap_to_leader": "+1.842",
                "last_speed": 306,
            },
            {
                "driver_number": 16,
                "acronym": "LEC",
                "full_name": "Charles Leclerc",
                "team_name": "Ferrari",
                "position": 3,
                "current_lap": 23,
                "last_lap_time": 68.64,
                "best_lap_time": 68.64,
                "current_compound": "HARD",
                "tyre_age": 11,
                "stint_number": 1,
                "used_compounds": ["HARD"],
                "pit_status": "on_track",
                "gap_to_leader": "+2.753",
                "last_speed": 304,
            },
        ],
    }


def _twenty_driver_snapshot():
    return {
        "sessionKey": "twenty-car-race",
        "seq": 101,
        "total_laps": 70,
        "remaining_laps": 39,
        "available_compounds": ["SOFT", "MEDIUM", "HARD"],
        "pit_lane_open": True,
        "is_wet_track": False,
        "raceControl": [{"flag": "GREEN", "message": "Track clear"}],
        "drivers": [
            {
                "driver_number": number,
                "acronym": f"D{number}",
                "position": number,
                "current_lap": 31,
                "last_lap_time": 80.0 + (number * 0.07),
                "best_lap_time": 79.5 + (number * 0.05),
                "current_compound": "MEDIUM" if number % 2 else "HARD",
                "tyre_age": 8 + number,
                "stint_number": 1,
                "used_compounds": ["MEDIUM" if number % 2 else "HARD"],
                "pit_status": "on_track",
                "gap_to_leader": str((number - 1) * 1.2),
            }
            for number in range(1, 21)
        ],
    }
