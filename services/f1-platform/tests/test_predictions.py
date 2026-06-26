import asyncio
import json

from f1_platform.predictions import (
    RemotePredictionConfig,
    RemotePredictionService,
    prediction_service_from_env,
)
from f1_platform.reducer import F1StateReducer
from f1_platform.sample_data import SAMPLE_SESSION_KEY, sample_events


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
        return {
            "predictions": [
                {
                    "model_version": "remote_model_v1",
                    "prediction_time": "2026-06-25T00:00:00Z",
                    "source_event_sequence": body["snapshot"]["seq"],
                    "features_version": "features_v1",
                    "driver_number": 63,
                    "expected_position": 1.8,
                    "position_distribution": {"1": 0.34, "2": 0.40, "3": 0.26},
                    "win_probability": 0.34,
                    "podium_probability": 1.0,
                    "points_probability": 1.0,
                    "dnf_probability": 0.02,
                    "confidence": 0.71,
                }
            ]
        }

    service = RemotePredictionService(
        RemotePredictionConfig(base_url="http://prediction.local", timeout_seconds=3.5, fallback_on_error=False),
        transport=transport,
    )

    predictions = asyncio.run(service.predict_race(snapshot))

    assert len(predictions) == 1
    assert seen[0][0].full_url == "http://prediction.local/api/f1/predict/race"
    assert seen[0][1] == 3.5
    assert predictions[0].model_version == "remote_model_v1"
    assert predictions[0].driver_number == 63
    assert predictions[0].expected_position == 1.8
    assert predictions[0].position_distribution["2"] == 0.4
    assert predictions[0].position_p10 == 1.0
    assert predictions[0].position_p90 == 3.0


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
    assert predictions[0].model_version == "heuristic_live_race_v0"


def test_prediction_service_from_env_selects_remote(monkeypatch):
    monkeypatch.setenv("F1_PLATFORM_PREDICTION_URL", "http://prediction:8002")
    monkeypatch.setenv("F1_PLATFORM_PREDICTION_TIMEOUT_SECONDS", "2.25")

    service = prediction_service_from_env()

    assert isinstance(service, RemotePredictionService)
    assert service.config.base_url == "http://prediction:8002"
    assert service.config.timeout_seconds == 2.25
