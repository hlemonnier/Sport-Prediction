"""Actual reducer -> request -> model service -> mapped platform forecasts."""

import asyncio
import json
from pathlib import Path

import pytest

from f1_platform.predictions import (
    HeuristicPredictionService,
    RemotePredictionConfig,
    RemotePredictionService,
)
from f1_platform.reducer import F1StateReducer
from f1_platform.schemas import F1Event
from f1_prediction_service.model import predict_from_snapshot
from packages.f1.models.live_race.next_lap import MODEL_ID, TARGET

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    ROOT
    / "research/projects/F1/rising_qualification_prediction/Python/tests/fixtures/live_frontier_prefix.json"
)


def observed_snapshot():
    rows = json.loads(FIXTURE.read_text())["observations"]
    reducer = F1StateReducer("test-frontier", source="fastf1-test")
    for index, row in enumerate(rows):
        if row["Time"] > 600:
            continue
        reducer.ingest(
            F1Event(
                source="fastf1",
                topic="v1/laps",
                source_id=index + 1,
                source_key=f"{row['DriverNumber']}:{row['LapNumber']}",
                meeting_key=None,
                session_key="test-frontier",
                driver_number=int(row["DriverNumber"]),
                event_time=None,
                received_at="2026-09-07T00:00:00Z",
                payload={
                    "lap_number": row["LapNumber"],
                    "lap_duration": row["LapTime"],
                    "raw_fastf1": row,
                },
            )
        )
    snapshot = reducer.snapshot()
    snapshot.session_info = {"session_type": "Race"}
    return snapshot


def test_reducer_to_remote_service_and_fallback_share_the_frozen_model():
    state = observed_snapshot()
    assert state.lap_observations_as_of_time_seconds == 600
    assert len(state.lap_observations) == 22
    seen = []

    def transport(request, timeout):
        body = json.loads(request.data)
        seen.append(body)
        return predict_from_snapshot(
            body["snapshot"], prediction_kind=body["predictionKind"]
        )

    remote = RemotePredictionService(
        RemotePredictionConfig("http://model", fallback_on_error=False),
        transport=transport,
    )
    for kind in ("next-lap", "race"):
        results = asyncio.run(remote.predict(kind, state))
        local = asyncio.run(HeuristicPredictionService().predict(kind, state))
        by_driver = {p.driver_number: p.lap_time_forecast for p in local}
        assert len(results) == 4
        for row in results:
            forecast = row.lap_time_forecast
            assert forecast == by_driver[row.driver_number]
            assert (
                forecast["model_id"] == MODEL_ID and forecast["status"] == "available"
            )
            assert forecast["seconds"] > 0 and forecast["target"] == TARGET
            assert forecast["interval_seconds"] is None
            assert forecast["issued_at_session_seconds"] <= 600
    assert seen[0]["snapshot"]["lapObservations"] == state.lap_observations


def test_source_missing_fields_falls_back_and_retired_driver_gets_no_lap():
    snapshot = observed_snapshot().to_dict()
    for row in snapshot["lapObservations"]:
        row.pop("SpeedST")
    snapshot["drivers"][0]["retired"] = True
    result = predict_from_snapshot(snapshot, prediction_kind="next-lap")
    by_number = {
        row["driver_number"]: row["lap_time_forecast"] for row in result["predictions"]
    }
    retired = snapshot["drivers"][0]["driver_number"]
    assert by_number[retired]["seconds"] is None
    assert by_number[retired]["status"] == "unavailable"
    for number, payload in by_number.items():
        if number == retired:
            continue
        assert payload["status"] == "fallback"
        assert payload["model_id"] == "last_observed_lap_snapshot_baseline_v1"
        assert "missing observed columns" in payload["reason"]


def test_unknown_or_future_history_does_not_activate_the_model():
    snapshot = observed_snapshot().to_dict()
    snapshot["lapObservationsAsOfTimeSeconds"] = 300
    result = predict_from_snapshot(snapshot, prediction_kind="next-lap")
    assert all(
        row["lap_time_forecast"]["status"] == "fallback"
        for row in result["predictions"]
    )
    assert "future completion" in result["diagnostics"]["lapTimeForecast"]["reason"]


def test_qualifying_history_cannot_activate_the_race_model():
    snapshot = observed_snapshot().to_dict()
    snapshot["sessionInfo"]["session_type"] = "Qualifying"
    result = predict_from_snapshot(snapshot, prediction_kind="next-lap")
    assert not result["diagnostics"]["provenance"]["canonicalLapTimeModelUsed"]
    assert (
        result["diagnostics"]["lapTimeForecast"]["reason"]
        == "race_session_identity_required"
    )


def test_service_can_be_switched_back_to_the_baseline(monkeypatch):
    monkeypatch.setenv("F1_LIVE_NEXT_LAP_POINT_MODEL", "baseline")
    result = predict_from_snapshot(
        observed_snapshot().to_dict(), prediction_kind="next-lap"
    )
    assert result["diagnostics"]["lapTimeForecast"]["status"] == "disabled"
    assert all(
        row["lap_time_forecast"]["status"] == "fallback"
        for row in result["predictions"]
    )


def test_completed_lap_import_uses_endpoint_and_keeps_untimed_transitions():
    from f1_platform.fastf1_analysis import FastF1LoadedSession, fastf1_runtime_events

    loaded = FastF1LoadedSession(
        year=2026,
        event_name="Fixture",
        session_name="R",
        session_key="fixture",
        laps=[
            {
                "DriverNumber": "1",
                "LapNumber": 1,
                "LapStartDate": "2026-09-07T12:00:00Z",
                "TimeSeconds": 100.0,
                "LapStartTimeSeconds": 10.0,
                "LapTimeSeconds": 90.0,
            },
            {
                "DriverNumber": "1",
                "LapNumber": 2,
                "LapStartDate": "2026-09-07T12:01:30Z",
                "TimeSeconds": 210.0,
                "LapStartTimeSeconds": 100.0,
                "LapTimeSeconds": None,
                "PitOutTime": None,
                "Stint": 2,
            },
        ],
        weather=[],
        race_control=[],
        telemetry_laps=[],
    )
    laps = [
        event for event in fastf1_runtime_events(loaded) if event.topic == "v1/laps"
    ]
    assert len(laps) == 2
    assert laps[0].event_time == "2026-09-07T12:01:30Z"
    assert laps[1].event_time == "2026-09-07T12:03:20Z"


@pytest.mark.parametrize(
    "field,value",
    [
        ("seconds", -1.0),
        ("model_id", "unverified"),
        ("issued_at_session_seconds", 900.0),
    ],
)
def test_remote_boundary_rejects_corrupted_lap_time_payload(field, value):
    state = observed_snapshot()

    def transport(request, timeout):
        body = json.loads(request.data)
        result = predict_from_snapshot(
            body["snapshot"], prediction_kind=body["predictionKind"]
        )
        result["predictions"][0]["lap_time_forecast"][field] = value
        return result

    remote = RemotePredictionService(
        RemotePredictionConfig("http://model", fallback_on_error=False),
        transport=transport,
    )
    with pytest.raises(RuntimeError, match="lap-time"):
        asyncio.run(remote.predict_next_lap(state))
