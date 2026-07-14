import asyncio
import json

import pytest

from f1_platform.predictions import (
    HeuristicPredictionService,
    RemotePredictionConfig,
    RemotePredictionService,
    prediction_kind_for_session,
    prediction_service_from_env,
)
from f1_platform.reducer import F1StateReducer
from f1_platform.sample_data import SAMPLE_SESSION_KEY, sample_events
from f1_platform.schemas import DriverState, SessionSnapshot, StintSegment


def test_remote_prediction_service_maps_model_response_to_snapshots():
    reducer = F1StateReducer(SAMPLE_SESSION_KEY)
    for event in sample_events(SAMPLE_SESSION_KEY):
        reducer.ingest(event)
    snapshot = reducer.snapshot()
    seen = []

    def transport(request, timeout):
        seen.append((request, timeout))
        body = json.loads(request.data.decode("utf-8"))
        assert body["predictionKind"] == "race"
        assert body["snapshot"]["sessionKey"] == SAMPLE_SESSION_KEY
        drivers = sorted(body["snapshot"]["drivers"], key=lambda row: row.get("position") or 99)
        total = len(drivers)
        predictions = []
        for position, driver in enumerate(drivers, start=1):
            distribution = {str(place): 1.0 if place == position else 0.0 for place in range(1, total + 1)}
            predictions.append(
                {
                    "model_version": "remote_model_v1",
                    "prediction_time": "2026-06-25T00:00:00Z",
                    "source_event_sequence": body["snapshot"]["seq"],
                    "features_version": "features_v1",
                    "driver_number": driver["driver_number"],
                    "expected_position": float(position),
                    "position_distribution": distribution,
                    "win_probability": distribution["1"],
                    "podium_probability": sum(distribution[str(place)] for place in range(1, min(3, total) + 1)),
                    "points_probability": sum(distribution[str(place)] for place in range(1, min(10, total) + 1)),
                    "dnf_probability": 0.02,
                    "confidence": 0.71,
                    "strategy": {
                        "recommendedAction": "stay_out",
                        "policyVersion": "test_policy_v1",
                    },
                }
            )
        return {"predictionKind": "race", "predictions": predictions}

    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", timeout_seconds=3.5, fallback_on_error=False),
        transport=transport,
    )

    predictions = asyncio.run(service.predict_race(snapshot))

    assert len(predictions) == len(snapshot.drivers)
    assert seen[0][0].full_url == "http://prediction.local/api/f1/predict/race"
    assert seen[0][1] == 3.5
    mapped = {prediction.driver_number: prediction for prediction in predictions}
    assert mapped[63].model_version == "remote_model_v1"
    assert mapped[63].expected_position == 2.0
    assert mapped[63].position_distribution["2"] == 1.0
    assert mapped[63].position_p10 == 2.0
    assert mapped[63].position_p90 == 2.0
    assert mapped[63].prediction_kind == "race"
    assert mapped[63].position_semantics == "race_finish_order"
    assert mapped[63].strategy["safeToRecommend"] is False
    assert mapped[63].strategy["recommendedAction"] is None
    assert mapped[63].strategy["originalRecommendedAction"] == "stay_out"
    assert mapped[63].strategy["unavailableReason"] == "remote_strategy_not_explicitly_safe"
    _assert_snapshot_joint_invariants(predictions)


def test_remote_strategy_is_nonforecast_and_unverified_recommendation_fails_closed():
    snapshot = _twenty_driver_session_snapshot()
    seen = []

    def transport(request, _timeout):
        seen.append(request)
        body = json.loads(request.data.decode("utf-8"))
        assert body["predictionKind"] == "strategy"
        return {
            "predictionKind": "strategy",
            "predictions": [
                {
                    "model_version": "strategy_policy_v1",
                    "features_version": "strategy_features_v1",
                    "source_event_sequence": snapshot.seq,
                    "driver_number": driver.driver_number,
                    "forecast_available": False,
                    "unavailable_reason": "strategy_is_a_decision_target_not_a_position_forecast",
                    "eligibility_status": "target_unavailable",
                    "participation_status": "running_or_unknown",
                    "expected_position": None,
                    "position_distribution": {},
                    "win_probability": 0.0,
                    "podium_probability": 0.0,
                    "points_probability": 0.0,
                    "dnf_probability": 0.0,
                    "confidence": 0.0,
                    "strategy": {
                        "recommendedAction": "pit_next_lap",
                        "policyVersion": "strategy_policy_v1",
                    },
                }
                for driver in snapshot.drivers
            ],
        }

    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=transport,
    )

    predictions = asyncio.run(service.predict_strategy(snapshot))

    assert seen[0].full_url == "http://prediction.local/api/f1/predict/strategy"
    assert len(predictions) == len(snapshot.drivers)
    for prediction in predictions:
        assert prediction.prediction_kind == "strategy"
        assert prediction.position_semantics == "not_applicable"
        assert prediction.expected_position is None
        assert prediction.position_distribution == {}
        assert prediction.position_p10 is None
        assert prediction.position_p90 is None
        assert prediction.win_probability == 0.0
        assert prediction.podium_probability == 0.0
        assert prediction.points_probability == 0.0
        assert prediction.dnf_probability == 0.0
        assert prediction.forecast_available is False
        assert prediction.strategy["safeToRecommend"] is False
        assert prediction.strategy["recommendedAction"] is None
        assert prediction.strategy["originalRecommendedAction"] == "pit_next_lap"
        assert prediction.strategy["unavailableReason"] == "remote_strategy_not_explicitly_safe"


def test_remote_prediction_service_falls_back_on_error():
    reducer = F1StateReducer(SAMPLE_SESSION_KEY)
    for event in sample_events(SAMPLE_SESSION_KEY):
        reducer.ingest(event)
    snapshot = reducer.snapshot()

    def transport(_request, _timeout):
        raise RuntimeError("model service unavailable")

    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=True),
        transport=transport,
    )

    predictions = asyncio.run(service.predict_race(snapshot))

    assert predictions
    assert predictions[0].model_version == "platform_fallback_live_race_joint_baseline_v2"
    _assert_snapshot_joint_invariants(predictions)


def test_remote_prediction_service_rejects_incoherent_joint_response_and_falls_back():
    snapshot = _twenty_driver_session_snapshot()

    def transport(_request, _timeout):
        total = len(snapshot.drivers)
        all_win = {str(position): 1.0 if position == 1 else 0.0 for position in range(1, total + 1)}
        return {
            "predictions": [
                {
                    "model_version": "broken_independent_model",
                    "features_version": "broken_features",
                    "driver_number": driver.driver_number,
                    "position_distribution": all_win,
                    "win_probability": 1.0,
                    "podium_probability": 1.0,
                    "points_probability": 1.0,
                }
                for driver in snapshot.drivers
            ]
        }

    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=True),
        transport=transport,
    )

    predictions = asyncio.run(service.predict_race(snapshot))

    assert predictions[0].model_version == "platform_fallback_live_race_joint_baseline_v2"
    _assert_snapshot_joint_invariants(predictions)


def test_remote_prediction_service_keeps_retired_driver_in_classification_field():
    snapshot = _three_driver_session_snapshot(retired_driver=3)
    response = _three_driver_classification_contract_response(snapshot, retired_driver=3)

    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=lambda _request, _timeout: response,
    )

    predictions = asyncio.run(service.predict_race(snapshot))

    by_driver = {prediction.driver_number: prediction for prediction in predictions}
    assert by_driver[1].model_version == "prediction_service_classification_contract_v2"
    assert by_driver[3].eligibility_status == "classification_eligible_retired"
    assert by_driver[3].forecast_available is True
    assert by_driver[3].dnf_probability == 1.0
    assert by_driver[3].win_probability == pytest.approx(0.1)
    assert by_driver[3].podium_probability == pytest.approx(1.0)
    assert by_driver[3].points_probability == pytest.approx(1.0)
    for position in range(1, 4):
        assert sum(row.position_distribution[str(position)] for row in predictions) == pytest.approx(1.0)
    assert sum(row.win_probability for row in predictions) == pytest.approx(1.0)
    assert sum(row.podium_probability for row in predictions) == pytest.approx(3.0)
    assert sum(row.points_probability for row in predictions) == pytest.approx(3.0)


def test_platform_fallback_uses_target_specific_retirement_semantics():
    snapshot = _three_driver_session_snapshot(retired_driver=3)
    service = HeuristicPredictionService()

    race = asyncio.run(service.predict_race(snapshot))
    qualifying = asyncio.run(service.predict_qualifying(snapshot))
    next_lap = asyncio.run(service.predict_next_lap(snapshot))

    race_by_driver = {row.driver_number: row for row in race}
    qualifying_by_driver = {row.driver_number: row for row in qualifying}
    next_lap_by_driver = {row.driver_number: row for row in next_lap}
    assert race_by_driver[3].forecast_available is True
    assert race_by_driver[3].eligibility_status == "classification_eligible_retired"
    assert race_by_driver[3].dnf_probability == 1.0
    assert race_by_driver[3].points_probability > 0.0
    assert qualifying_by_driver[3].forecast_available is True
    assert qualifying_by_driver[3].eligibility_status == "classification_eligible_retired"
    assert next_lap_by_driver[3].forecast_available is False
    assert next_lap_by_driver[3].position_distribution == {}
    assert next_lap_by_driver[3].unavailable_reason == "next_lap_unavailable:retired_or_stopped"
    _assert_snapshot_joint_invariants(race)
    _assert_snapshot_joint_invariants(qualifying)
    _assert_snapshot_joint_invariants(next_lap)


def test_remote_prediction_service_rejects_retired_driver_as_target_unavailable():
    snapshot = _three_driver_session_snapshot(retired_driver=3)
    response = _three_driver_classification_contract_response(snapshot, retired_driver=3)
    response["predictions"][2].update(
        {
            "forecast_available": False,
            "unavailable_reason": "classification_ineligible:retired",
            "eligibility_status": "target_unavailable",
            "position_distribution": {},
            "expected_position": None,
            "position_p10": None,
            "position_p90": None,
            "win_probability": 0.0,
            "podium_probability": 0.0,
            "points_probability": 0.0,
        }
    )

    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=lambda _request, _timeout: response,
    )

    with pytest.raises(RuntimeError, match="disagrees with local target status"):
        asyncio.run(service.predict_race(snapshot))


def test_remote_prediction_service_rejects_incomplete_field_without_fallback():
    snapshot = _twenty_driver_session_snapshot()

    def transport(_request, _timeout):
        return {"predictions": []}

    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="incomplete field"):
        asyncio.run(service.predict_race(snapshot))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("expected_position", float("inf"), "expected_position disagrees"),
        ("expected_position", 99.0, "expected_position disagrees"),
        ("position_p10", -1.0, "position_p10 disagrees"),
        ("confidence", float("inf"), "invalid confidence"),
        ("dnf_probability", -0.01, "invalid dnf_probability"),
    ],
)
def test_remote_prediction_service_rejects_nonfinite_or_incoherent_scalars(field, value, error):
    snapshot = _three_driver_session_snapshot()
    response = _three_driver_classification_contract_response(snapshot)
    response["predictions"][0][field] = value
    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=lambda _request, _timeout: response,
    )

    with pytest.raises(RuntimeError, match=error):
        asyncio.run(service.predict_race(snapshot))


def test_remote_next_lap_accepts_explicit_unavailable_retired_row():
    snapshot = _three_driver_session_snapshot(retired_driver=3)
    distributions = {
        1: {"1": 0.7, "2": 0.3},
        2: {"1": 0.3, "2": 0.7},
    }
    rows = [
        _remote_prediction_row(
            number,
            distributions[number],
            eligibility="classification_eligible",
            participation="running_or_unknown",
            active_count=2,
        )
        for number in (1, 2)
    ]
    rows.append(
        {
            "model_version": "next_lap_availability_v1",
            "features_version": "next_lap_availability_features_v1",
            "driver_number": 3,
            "forecast_available": False,
            "unavailable_reason": "next_lap_unavailable:retired_or_stopped",
            "eligibility_status": "target_unavailable",
            "participation_status": "retired_or_stopped",
            "expected_position": None,
            "position_p10": None,
            "position_p90": None,
            "position_distribution": {},
            "win_probability": 0.0,
            "podium_probability": 0.0,
            "points_probability": 0.0,
            "dnf_probability": 0.0,
            "confidence": 0.0,
        }
    )
    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=lambda _request, _timeout: {"predictionKind": "next-lap", "predictions": rows},
    )

    predictions = asyncio.run(service.predict_next_lap(snapshot))

    by_driver = {row.driver_number: row for row in predictions}
    assert by_driver[3].forecast_available is False
    assert by_driver[3].position_distribution == {}
    assert by_driver[3].expected_position is None
    _assert_snapshot_joint_invariants(predictions)


def test_platform_fallback_excludes_dns_but_not_retired_from_classification():
    snapshot = _three_driver_session_snapshot(retired_driver=2, dns_driver=3)
    predictions = asyncio.run(HeuristicPredictionService().predict_race(snapshot))

    by_driver = {row.driver_number: row for row in predictions}
    assert by_driver[2].forecast_available is True
    assert by_driver[2].eligibility_status == "classification_eligible_retired"
    assert by_driver[2].points_probability > 0.0
    assert by_driver[3].forecast_available is False
    assert by_driver[3].participation_status == "dns"
    assert by_driver[3].position_distribution == {}
    _assert_snapshot_joint_invariants(predictions)


def test_pit_out_status_is_not_misclassified_as_retired():
    snapshot = _three_driver_session_snapshot()
    snapshot.drivers[0].pit_status = "PIT_OUT"

    next_lap = asyncio.run(HeuristicPredictionService().predict_next_lap(snapshot))

    by_driver = {row.driver_number: row for row in next_lap}
    assert by_driver[1].forecast_available is True
    assert by_driver[1].participation_status == "running_or_unknown"


def test_remote_safe_strategy_is_recomputed_from_local_legal_state():
    snapshot = _strategy_ready_snapshot()
    response = _strategy_response(snapshot, action="stay_out", safe=True)
    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=lambda _request, _timeout: response,
    )

    prediction = asyncio.run(service.predict_strategy(snapshot))[0]

    assert prediction.strategy["safeToRecommend"] is True
    assert prediction.strategy["recommendedAction"] == "stay_out"
    assert prediction.strategy["paceMode"] is None
    assert prediction.strategy["legalActionKey"] is None
    assert prediction.strategy["compatibleLegalActionKeys"] == [
        "stay_out:conservative",
        "stay_out:aggressive",
    ]
    assert prediction.strategy["legalityState"]["tyreAge"] == 12


def test_remote_explicit_pace_mode_produces_one_locally_verified_action_key():
    snapshot = _strategy_ready_snapshot()
    response = _strategy_response(snapshot, action="stay_out", safe=True, pace_mode="aggressive")
    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=lambda _request, _timeout: response,
    )

    prediction = asyncio.run(service.predict_strategy(snapshot))[0]

    assert prediction.strategy["safeToRecommend"] is True
    assert prediction.strategy["paceMode"] == "aggressive"
    assert prediction.strategy["compatibleLegalActionKeys"] == ["stay_out:aggressive"]
    assert prediction.strategy["legalActionKey"] == "stay_out:aggressive"


def test_remote_strategy_fails_closed_when_local_tyre_age_is_missing():
    snapshot = _strategy_ready_snapshot()
    snapshot.drivers[0].tyre_age = None
    response = _strategy_response(snapshot, action="stay_out", safe=True)
    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=lambda _request, _timeout: response,
    )

    prediction = asyncio.run(service.predict_strategy(snapshot))[0]

    assert prediction.strategy["safeToRecommend"] is False
    assert prediction.strategy["recommendedAction"] is None
    assert "tyre_age" in prediction.strategy["unavailableReason"]


def test_remote_embedded_strategy_fails_closed_when_local_pit_lane_is_closed():
    snapshot = _strategy_ready_snapshot(pit_lane_open=False)
    distribution = {"1": 1.0}
    row = _remote_prediction_row(
        1,
        distribution,
        eligibility="classification_eligible",
        participation="running_or_unknown",
        active_count=1,
    )
    row["strategy"] = {
        "recommendedAction": "pit_now",
        "nextCompound": "HARD",
        "safeToRecommend": True,
        "availability": "available",
    }
    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", fallback_on_error=False),
        transport=lambda _request, _timeout: {"predictionKind": "race", "predictions": [row]},
    )

    prediction = asyncio.run(service.predict_race(snapshot))[0]

    assert prediction.strategy["safeToRecommend"] is False
    assert prediction.strategy["recommendedAction"] is None
    assert "pit_lane_closed" in prediction.strategy["unavailableReason"]


def test_platform_fallback_twenty_driver_invariants_and_target_specificity():
    snapshot = _twenty_driver_session_snapshot()
    service = HeuristicPredictionService()

    race = asyncio.run(service.predict_race(snapshot))
    qualifying = asyncio.run(service.predict_qualifying(snapshot))
    next_lap = asyncio.run(service.predict_next_lap(snapshot))
    strategy = asyncio.run(service.predict_strategy(snapshot))

    model_versions = {
        race[0].model_version,
        qualifying[0].model_version,
        next_lap[0].model_version,
        strategy[0].model_version,
    }
    assert len(model_versions) == 4
    assert race[0].driver_number == 1
    assert qualifying[0].driver_number == 20
    assert next_lap[0].driver_number == 20
    for predictions in (race, qualifying, next_lap):
        _assert_snapshot_joint_invariants(predictions, tolerance=1e-8)
    assert {prediction.prediction_kind for prediction in race} == {"race"}
    assert {prediction.prediction_kind for prediction in qualifying} == {"qualifying"}
    assert {prediction.prediction_kind for prediction in next_lap} == {"next-lap"}
    for prediction in strategy:
        assert prediction.prediction_kind == "strategy"
        assert prediction.position_semantics == "not_applicable"
        assert prediction.expected_position is None
        assert prediction.position_distribution == {}
        assert prediction.win_probability == 0.0
        assert prediction.podium_probability == 0.0
        assert prediction.points_probability == 0.0
        assert prediction.dnf_probability == 0.0
        assert prediction.strategy is None


@pytest.mark.parametrize(
    ("session_info", "expected"),
    [
        ({"session_type": "Qualifying", "session_name": "Qualifying"}, "qualifying"),
        ({"session_type": "Race", "session_name": "Sprint Shootout"}, "qualifying"),
        ({"session_type": "Qualifying", "session_name": "SQ"}, "qualifying"),
        ({"session_type": "Practice", "session_name": "Practice 2"}, "next-lap"),
        ({"session_type": "Race", "session_name": "Sprint"}, "race"),
        (None, "race"),
    ],
)
def test_prediction_kind_for_session_routes_session_contract(session_info, expected):
    snapshot = SessionSnapshot(
        session_key="routing-test",
        seq=0,
        generated_at="2026-07-12T00:00:00Z",
        source="test",
        drivers=[],
        session_info=session_info,
    )

    assert prediction_kind_for_session(snapshot) == expected


def test_prediction_service_from_env_selects_remote(monkeypatch):
    monkeypatch.setenv("F1_PLATFORM_PREDICTION_URL", "http://prediction:8002")
    monkeypatch.setenv("F1_PLATFORM_PREDICTION_TIMEOUT_SECONDS", "2.25")

    service = prediction_service_from_env()

    assert isinstance(service, RemotePredictionService)
    assert service.config.base_url == "http://prediction:8002"
    assert service.config.timeout_seconds == 2.25


def _assert_snapshot_joint_invariants(predictions, *, tolerance=2e-6):
    assert predictions
    available = [prediction for prediction in predictions if prediction.forecast_available]
    total = len(available)
    for prediction in predictions:
        distribution = prediction.position_distribution
        if not prediction.forecast_available:
            assert distribution == {}
            assert prediction.expected_position is None
            assert prediction.position_p10 is None
            assert prediction.position_p90 is None
            assert prediction.win_probability == 0.0
            assert prediction.podium_probability == 0.0
            assert prediction.points_probability == 0.0
            assert prediction.unavailable_reason
            continue
        assert set(distribution) == {str(position) for position in range(1, total + 1)}
        assert sum(distribution.values()) == pytest.approx(1.0, abs=tolerance)
        assert prediction.win_probability == pytest.approx(distribution["1"], abs=tolerance)
        assert prediction.podium_probability == pytest.approx(
            sum(distribution[str(position)] for position in range(1, min(3, total) + 1)),
            abs=tolerance,
        )
        assert prediction.points_probability == pytest.approx(
            sum(distribution[str(position)] for position in range(1, min(10, total) + 1)),
            abs=tolerance,
        )
    for position in range(1, total + 1):
        assert sum(row.position_distribution[str(position)] for row in available) == pytest.approx(
            1.0,
            abs=tolerance,
        )
    assert sum(row.win_probability for row in predictions) == pytest.approx(min(1, total), abs=tolerance)
    assert sum(row.podium_probability for row in predictions) == pytest.approx(min(3, total), abs=tolerance)
    assert sum(row.points_probability for row in predictions) == pytest.approx(min(10, total), abs=tolerance)


def _twenty_driver_session_snapshot():
    drivers = [
        DriverState(
            driver_number=number,
            position=number,
            last_lap_time=82.0 - (number * 0.08),
            best_lap_time=81.5 - (number * 0.09),
            current_compound="MEDIUM" if number % 2 else "HARD",
            tyre_age=5 + number,
        )
        for number in range(1, 21)
    ]
    return SessionSnapshot(
        session_key="twenty-driver-test",
        seq=77,
        generated_at="2026-07-11T00:00:00Z",
        source="test",
        drivers=drivers,
    )


def _strategy_ready_snapshot(*, pit_lane_open=True):
    driver = DriverState(
        driver_number=1,
        position=1,
        current_lap=20,
        last_lap_time=82.0,
        best_lap_time=81.5,
        current_compound="MEDIUM",
        tyre_age=12,
        stint_number=1,
        pit_status="TRACK",
        track_status="green",
    )
    return SessionSnapshot(
        session_key="strategy-ready-test",
        seq=91,
        generated_at="2026-07-12T00:00:00Z",
        source="test",
        drivers=[driver],
        session_info={
            "session_type": "Race",
            "total_laps": 70,
            "remaining_laps": 50,
            "available_compounds": ["SOFT", "MEDIUM", "HARD"],
            "pit_lane_open": pit_lane_open,
        },
        strategy_timeline=[
            StintSegment(
                driver_number=1,
                stint_number=1,
                compound="MEDIUM",
                start_lap=1,
                end_lap=None,
                tyre_age_start=0,
            )
        ],
        weather={"rainfall": 0.0},
    )


def _strategy_response(snapshot, *, action, safe, pace_mode=None):
    return {
        "predictionKind": "strategy",
        "predictions": [
            {
                "model_version": "strategy_contract_v2",
                "features_version": "strategy_contract_features_v2",
                "source_event_sequence": snapshot.seq,
                "driver_number": 1,
                "forecast_available": False,
                "unavailable_reason": "strategy_has_no_position_forecast",
                "eligibility_status": "target_unavailable",
                "participation_status": "running_or_unknown",
                "expected_position": None,
                "position_p10": None,
                "position_p90": None,
                "position_distribution": {},
                "win_probability": 0.0,
                "podium_probability": 0.0,
                "points_probability": 0.0,
                "dnf_probability": 0.0,
                "confidence": 0.0,
                "strategy": {
                    "recommendedAction": action,
                    "safeToRecommend": safe,
                    "availability": "available" if safe else "unavailable",
                    "policyVersion": "strategy_contract_v2",
                    **({"paceMode": pace_mode} if pace_mode is not None else {}),
                },
            }
        ],
    }


def _three_driver_session_snapshot(*, retired_driver=None, dns_driver=None):
    results = []
    for number in range(1, 4):
        results.append(
            {
                "driver_number": number,
                "position": number,
                "number_of_laps": 50 if number == retired_driver else 49,
                "dnf": number == retired_driver,
                "dns": number == dns_driver,
                "dsq": False,
                "status": "Retired" if number == retired_driver else "Did Not Start" if number == dns_driver else None,
            }
        )
    return SessionSnapshot(
        session_key="three-driver-retirement-test",
        seq=81,
        generated_at="2026-07-12T00:00:00Z",
        source="test",
        drivers=[
            DriverState(
                driver_number=number,
                position=number,
                current_lap=50 if number == retired_driver else 49,
                last_lap_time=80.0 + number,
                best_lap_time=79.0 + number,
                current_compound="MEDIUM",
                tyre_age=10,
                stint_number=1,
                pit_status="OUT" if number == retired_driver else "TRACK",
                track_status="green",
            )
            for number in range(1, 4)
        ],
        session_results=results,
    )


def _three_driver_classification_contract_response(snapshot, *, retired_driver=None):
    distributions = {
        1: {"1": 0.7, "2": 0.2, "3": 0.1},
        2: {"1": 0.2, "2": 0.6, "3": 0.2},
        3: {"1": 0.1, "2": 0.2, "3": 0.7},
    }
    return {
        "predictionKind": "race",
        "predictions": [
            _remote_prediction_row(
                driver.driver_number,
                distributions[driver.driver_number],
                eligibility=(
                    "classification_eligible_retired"
                    if driver.driver_number == retired_driver
                    else "classification_eligible"
                ),
                participation=(
                    "retired_or_stopped"
                    if driver.driver_number == retired_driver
                    else "running_or_unknown"
                ),
                active_count=3,
                dnf_probability=1.0 if driver.driver_number == retired_driver else 0.0,
            )
            for driver in snapshot.drivers
        ],
    }


def _twelve_driver_session_snapshot():
    return SessionSnapshot(
        session_key="twelve-driver-retirement-test",
        seq=82,
        generated_at="2026-07-12T00:00:00Z",
        source="test",
        drivers=[DriverState(driver_number=number, position=number) for number in range(1, 13)],
    )


def _remote_prediction_row(
    driver_number,
    distribution,
    *,
    eligibility,
    active_count,
    participation="running_or_unknown",
    dnf_probability=0.0,
):
    win_probability = distribution["1"]
    podium_probability = sum(
        distribution[str(position)]
        for position in range(1, min(3, active_count) + 1)
    )
    points_probability = sum(
        distribution[str(position)]
        for position in range(1, min(10, active_count) + 1)
    )
    expected_position = sum(float(position) * probability for position, probability in distribution.items())
    return {
        "model_version": "prediction_service_classification_contract_v2",
        "features_version": "prediction_service_classification_features_v2",
        "driver_number": driver_number,
        "forecast_available": True,
        "unavailable_reason": None,
        "position_distribution": distribution,
        "expected_position": expected_position,
        "win_probability": win_probability,
        "podium_probability": podium_probability,
        "points_probability": points_probability,
        "dnf_probability": dnf_probability,
        "confidence": 0.5,
        "eligibility_status": eligibility,
        "participation_status": participation,
    }
