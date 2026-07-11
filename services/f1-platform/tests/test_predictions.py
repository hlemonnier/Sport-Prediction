import asyncio
import json

import pytest

from f1_platform.predictions import (
    HeuristicPredictionService,
    RemotePredictionConfig,
    RemotePredictionService,
    prediction_service_from_env,
)
from f1_platform.reducer import F1StateReducer
from f1_platform.sample_data import SAMPLE_SESSION_KEY, sample_events
from f1_platform.schemas import DriverState, SessionSnapshot


def test_remote_prediction_service_maps_model_response_to_snapshots():
    reducer = F1StateReducer(SAMPLE_SESSION_KEY)
    for event in sample_events(SAMPLE_SESSION_KEY):
        reducer.ingest(event)
    snapshot = reducer.snapshot()
    seen = []

    def transport(request, timeout):
        seen.append((request, timeout))
        body = json.loads(request.data.decode("utf-8"))
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
                }
            )
        return {"predictions": predictions}

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
    _assert_snapshot_joint_invariants(predictions)


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
    assert predictions[0].model_version == "platform_fallback_live_race_joint_baseline_v1"
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

    assert predictions[0].model_version == "platform_fallback_live_race_joint_baseline_v1"
    _assert_snapshot_joint_invariants(predictions)


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


def test_platform_fallback_twenty_driver_invariants_and_target_specificity():
    snapshot = _twenty_driver_session_snapshot()
    service = HeuristicPredictionService()

    race = asyncio.run(service.predict_race(snapshot))
    qualifying = asyncio.run(service.predict_qualifying(snapshot))
    next_lap = asyncio.run(service.predict_next_lap(snapshot))
    strategy = asyncio.run(service.predict_strategy(snapshot))

    assert len({race[0].model_version, qualifying[0].model_version, next_lap[0].model_version, strategy[0].model_version}) == 4
    assert race[0].driver_number == 1
    assert qualifying[0].driver_number == 20
    assert next_lap[0].driver_number == 20
    for predictions in (race, qualifying, next_lap, strategy):
        _assert_snapshot_joint_invariants(predictions, tolerance=1e-8)


def test_prediction_service_from_env_selects_remote(monkeypatch):
    monkeypatch.setenv("F1_PLATFORM_PREDICTION_URL", "http://prediction:8002")
    monkeypatch.setenv("F1_PLATFORM_PREDICTION_TIMEOUT_SECONDS", "2.25")

    service = prediction_service_from_env()

    assert isinstance(service, RemotePredictionService)
    assert service.config.base_url == "http://prediction:8002"
    assert service.config.timeout_seconds == 2.25


def _assert_snapshot_joint_invariants(predictions, *, tolerance=2e-6):
    total = len(predictions)
    assert total > 0
    for prediction in predictions:
        distribution = prediction.position_distribution
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
        assert sum(row.position_distribution[str(position)] for row in predictions) == pytest.approx(
            1.0,
            abs=tolerance,
        )
    assert sum(row.win_probability for row in predictions) == pytest.approx(1.0, abs=tolerance)
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
