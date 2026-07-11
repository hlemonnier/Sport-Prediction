import asyncio
import json

import pytest

from f1_platform.fastf1_analysis import (
    ArtifactRecord,
    FastF1ArtifactService,
    FastF1ArtifactStore,
    FastF1ImportRequest,
    FastF1ImportResult,
    FastF1LapTelemetry,
    FastF1LoadedSession,
    build_centerline,
    build_corner_metrics,
    build_telemetry_delta,
    fastf1_runtime_events,
    request_from_payload,
    resample_telemetry,
    _normalize_record,
)


def test_resample_centerline_and_delta_are_distance_aligned():
    telemetry_a = [
        {"Distance": 0, "Speed": 100, "Throttle": 10, "Brake": False, "nGear": 2, "DRS": 0, "X": 0, "Y": 0},
        {"Distance": 10, "Speed": 200, "Throttle": 100, "Brake": False, "nGear": 5, "DRS": 8, "X": 10, "Y": 0},
        {"Distance": 20, "Speed": 100, "Throttle": 50, "Brake": True, "nGear": 3, "DRS": 0, "X": 20, "Y": 10},
    ]
    telemetry_b = [
        {"Distance": 0, "Speed": 200, "Throttle": 20, "Brake": False, "nGear": 3, "DRS": 0, "X": 0, "Y": 0},
        {"Distance": 10, "Speed": 200, "Throttle": 80, "Brake": False, "nGear": 5, "DRS": 8, "X": 10, "Y": 0},
        {"Distance": 20, "Speed": 200, "Throttle": 80, "Brake": False, "nGear": 5, "DRS": 8, "X": 20, "Y": 10},
    ]

    aligned_a = resample_telemetry(telemetry_a, distance_step_meters=5)
    aligned_b = resample_telemetry(telemetry_b, distance_step_meters=5)
    centerline = build_centerline(aligned_a, smoothing_window=3)
    delta = build_telemetry_delta(aligned_a, aligned_b)

    assert [row["Distance"] for row in aligned_a] == [0.0, 5.0, 10.0, 15.0, 20.0]
    assert aligned_a[1]["Speed"] == 150.0
    assert aligned_a[1]["Throttle"] == 55.0
    assert aligned_a[-1]["Brake"] == 1.0
    assert centerline[-1]["Progress"] == 1.0
    assert centerline[1]["X"] == 5.0
    assert delta[-1]["DeltaSeconds"] > 0
    assert delta[-1]["SpeedA"] == 100.0
    assert delta[-1]["SpeedB"] == 200.0


def test_corner_metrics_detect_local_speed_minimum():
    telemetry = [
        {"Distance": 0, "Speed": 210, "Throttle": 0, "Brake": 0},
        {"Distance": 5, "Speed": 150, "Throttle": 20, "Brake": 1},
        {"Distance": 10, "Speed": 90, "Throttle": 30, "Brake": 1},
        {"Distance": 15, "Speed": 160, "Throttle": 80, "Brake": 0},
        {"Distance": 20, "Speed": 220, "Throttle": 100, "Brake": 0},
    ]

    metrics = build_corner_metrics(telemetry, driver="VER", lap_number=7)

    assert len(metrics) == 1
    [corner] = metrics
    assert corner["Driver"] == "VER"
    assert corner["LapNumber"] == 7
    assert corner["EntryDistance"] == 5
    assert corner["ApexDistance"] == 10
    assert corner["ExitDistance"] == 15
    assert corner["EntrySpeed"] == 150
    assert corner["MinimumSpeed"] == 90
    assert corner["ExitSpeed"] == 160
    assert corner["BrakeStartDistance"] == 5
    assert corner["ThrottleReapplicationDistance"] == 15
    assert corner["BrakingDurationSeconds"] > 0
    assert corner["CornerTimeSeconds"] > 0


def test_normalize_record_derives_seconds_from_fastf1_duration_strings():
    row = _normalize_record(
        {
            "LapTime": "P0DT0H1M6.054S",
            "Sector1Time": "P0DT0H0M16.579S",
            "PitOutTime": "NaT",
        }
    )

    assert row["LapTimeSeconds"] == 66.054
    assert row["Sector1TimeSeconds"] == 16.579
    assert "PitOutTimeSeconds" not in row


def test_fastf1_artifact_service_writes_session_engineering_artifacts(tmp_path):
    service = FastF1ArtifactService(
        FastF1ArtifactStore(tmp_path, allow_json_fallback=True),
        provider=FakeFastF1Provider(),
    )
    request = FastF1ImportRequest(
        year=2026,
        event="Austria",
        session_name="R",
        drivers=("VER", "RUS"),
        output_format="jsonl",
    )

    result = service.import_session(request)

    assert result.session_key == "fastf1:2026:austria:r"
    by_kind = {artifact.kind: artifact for artifact in result.artifacts}
    telemetry_artifacts = [
        artifact for artifact in result.artifacts if artifact.kind == "fastf1_distance_aligned_telemetry"
    ]
    corner_artifacts = [artifact for artifact in result.artifacts if artifact.kind == "fastf1_corner_metrics"]
    assert set(by_kind) == {
        "fastf1_laps",
        "fastf1_weather",
        "fastf1_race_control",
        "fastf1_distance_aligned_telemetry",
        "fastf1_corner_metrics",
        "fastf1_centerline",
        "fastf1_telemetry_delta",
    }
    assert by_kind["fastf1_laps"].row_count == 2
    assert by_kind["fastf1_centerline"].row_count == 5
    assert result.to_dict()["eventCount"] > 0
    lap_events = [event for event in result.runtime_events if event.topic == "v1/laps"]
    assert lap_events[0].driver_number == 1
    assert lap_events[0].payload["lap_duration"] == 65.1
    assert {event.topic for event in result.runtime_events} >= {"v1/sessions", "v1/drivers", "v1/position", "v1/laps"}
    assert len(telemetry_artifacts) == 2
    assert len(corner_artifacts) == 2
    telemetry_paths = {artifact.path for artifact in telemetry_artifacts}
    assert any("telemetry/year=2026/event=austria/session=r/driver=ver/lap=7/part-000.jsonl" in path for path in telemetry_paths)
    assert any("telemetry/year=2026/event=austria/session=r/driver=rus/lap=8/part-000.jsonl" in path for path in telemetry_paths)
    corner_paths = {artifact.path for artifact in corner_artifacts}
    assert any("corner_metrics/year=2026/event=austria/session=r/driver=ver/lap=7/part-000.jsonl" in path for path in corner_paths)
    for artifact in result.artifacts:
        assert artifact.format == "jsonl"
        assert artifact.path.startswith(str(tmp_path))
        assert artifact.metadata["sessionKey"] == "fastf1:2026:austria:r"

    delta_path = by_kind["fastf1_telemetry_delta"].path
    with open(delta_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows[-1]["DeltaSeconds"] != 0

    summary = service.engineering_summary(session_key="fastf1:2026:austria:r")
    assert summary["sessionKey"] == "fastf1:2026:austria:r"
    assert summary["artifactCounts"]["telemetryDelta"] == 1
    assert summary["artifactCounts"]["cornerMetrics"] == 2
    assert summary["telemetryDelta"]["driverA"] == "VER"
    assert summary["telemetryDelta"]["driverB"] == "RUS"
    assert summary["telemetryDelta"]["finalDeltaSeconds"] == rows[-1]["DeltaSeconds"]
    assert summary["telemetryDelta"]["series"][0]["distance"] == 0
    corner_by_driver = {item["driver"]: item for item in summary["cornerMetrics"]}
    assert corner_by_driver["VER"]["cornerCount"] == 1
    assert corner_by_driver["VER"]["corners"][0]["minimumSpeed"] == 180


def test_fastf1_runtime_events_do_not_mislabel_relative_durations_as_absolute_timestamps():
    loaded = FastF1LoadedSession(
        year=2025,
        event_name="Monaco Grand Prix",
        session_name="R",
        session_key="fastf1:2025:monaco-grand-prix:r",
        laps=[
            {
                "Driver": "NOR",
                "DriverNumber": "4",
                "LapNumber": 1,
                "LapTimeSeconds": 73.2,
                "Time": "P0DT0H57M36.09S",
            }
        ],
        weather=[{"Time": "P0DT0H1M18.584S", "TrackTemp": 42.1}],
        race_control=[{"Date": "2025-05-25T12:20:01", "Message": "Track clear"}],
        telemetry_laps=[],
    )

    events = fastf1_runtime_events(loaded)
    lap_event = next(event for event in events if event.topic == "v1/laps")
    weather_event = next(event for event in events if event.topic == "v1/weather")
    race_control_event = next(event for event in events if event.topic == "v1/race_control")

    assert lap_event.event_time is None
    assert weather_event.event_time is None
    assert race_control_event.event_time == "2025-05-25T12:20:01"


def test_artifact_store_lists_and_reads_bounded_rows(tmp_path):
    store = FastF1ArtifactStore(tmp_path, allow_json_fallback=True)
    record = store.write_table(
        [
            {"Distance": 0, "Speed": 180, "Throttle": 80},
            {"Distance": 5, "Speed": 185, "Throttle": 90},
        ],
        "telemetry/year=2026/event=austria/session=r/driver=ver/lap=7/part-000",
        preferred_format="jsonl",
        metadata={
            "kind": "fastf1_distance_aligned_telemetry",
            "sessionKey": "fastf1:2026:austria:r",
            "driver": "VER",
            "lapNumber": 7,
        },
    )

    listed = store.list_artifacts(session_key="fastf1:2026:austria:r")
    assert len(listed) == 1
    assert listed[0].artifact_id == record.artifact_id
    assert listed[0].relative_path.endswith("driver=ver/lap=7/part-000.jsonl")
    assert listed[0].row_count == 2

    preview = store.read_artifact_rows(str(record.artifact_id), limit=1)
    assert preview["artifact"]["artifactId"] == record.artifact_id
    assert preview["columns"] == ["Distance", "Speed", "Throttle"]
    assert preview["rows"] == [{"Distance": 0, "Speed": 180, "Throttle": 80}]
    assert preview["truncated"] is True


def test_fastf1_payload_parser_accepts_round_or_event_and_driver_list():
    parsed = request_from_payload(
        {
            "year": "2026",
            "round": "11",
            "session": "Race",
            "drivers": "VER, RUS",
            "distance_step_meters": "10",
            "telemetry_laps_per_driver": 2,
            "output_format": "jsonl",
        }
    )

    assert parsed.year == 2026
    assert parsed.event == 11
    assert parsed.session_name == "Race"
    assert parsed.drivers == ("VER", "RUS")
    assert parsed.distance_step_meters == 10
    assert parsed.telemetry_laps_per_driver == 2


def test_fastf1_payload_parser_rejects_missing_event():
    with pytest.raises(ValueError, match="event or round is required"):
        request_from_payload({"year": 2026})


def test_fastf1_import_endpoint_uses_state_service(tmp_path, monkeypatch):
    try:
        from f1_platform.app import create_app
    except ImportError as exc:
        pytest.skip(f"FastAPI unavailable: {exc}")

    monkeypatch.setenv("F1_PLATFORM_EVENT_STORE", str(tmp_path / "events"))
    monkeypatch.setenv("F1_PLATFORM_SQLITE_PROJECTION_STORE", str(tmp_path / "projection.sqlite"))
    monkeypatch.setenv("F1_PLATFORM_FASTF1_ARTIFACT_STORE", str(tmp_path / "artifacts"))
    monkeypatch.setenv("F1_PLATFORM_FASTF1_CACHE", str(tmp_path / "cache"))
    app = create_app()
    fake_service = FakeFastF1ArtifactService()
    app.state.fastf1_artifacts = fake_service
    route = next(route for route in app.routes if getattr(route, "path", "") == "/api/f1/fastf1/import")

    payload = asyncio.run(
        route.endpoint({"year": 2026, "event": "Austria", "session_name": "R", "output_format": "jsonl"})
    )

    assert payload["imported"] is True
    assert payload["sessionKey"] == "fake-session"
    assert payload["snapshot"]["source"] == "fastf1-history"
    assert payload["snapshot"]["drivers"][0]["acronym"] == "VER"
    assert payload["snapshot"]["lapChart"][0]["value"] == 65.1
    assert fake_service.seen.year == 2026
    assert fake_service.seen.event == "Austria"


def test_fastf1_artifact_endpoints_list_and_preview(tmp_path, monkeypatch):
    try:
        from f1_platform.app import create_app
    except ImportError as exc:
        pytest.skip(f"FastAPI unavailable: {exc}")

    monkeypatch.setenv("F1_PLATFORM_EVENT_STORE", str(tmp_path / "events"))
    monkeypatch.setenv("F1_PLATFORM_SQLITE_PROJECTION_STORE", str(tmp_path / "projection.sqlite"))
    monkeypatch.setenv("F1_PLATFORM_FASTF1_ARTIFACT_STORE", str(tmp_path / "artifacts"))
    monkeypatch.setenv("F1_PLATFORM_FASTF1_CACHE", str(tmp_path / "cache"))
    app = create_app()
    store = app.state.fastf1_artifacts.artifact_store
    record = store.write_table(
        [{"Distance": 0, "DeltaSeconds": 0.0}, {"Distance": 5, "DeltaSeconds": -0.02}],
        "telemetry_comparison/year=2026/event=austria/session=r/ver-lap-7__rus-lap-8",
        preferred_format="jsonl",
        metadata={
            "kind": "fastf1_telemetry_delta",
            "sessionKey": "fastf1:2026:austria:r",
            "driverA": "VER",
            "driverB": "RUS",
        },
    )
    store.write_table(
        [
            {
                "Driver": "VER",
                "LapNumber": 7,
                "CornerIndex": 1,
                "EntryDistance": 100,
                "ApexDistance": 110,
                "ExitDistance": 120,
                "EntrySpeed": 240,
                "MinimumSpeed": 150,
                "ExitSpeed": 220,
                "CornerTimeSeconds": 1.24,
            }
        ],
        "corner_metrics/year=2026/event=austria/session=r/driver=ver/lap=7/part-000",
        preferred_format="jsonl",
        metadata={
            "kind": "fastf1_corner_metrics",
            "sessionKey": "fastf1:2026:austria:r",
            "driver": "VER",
            "lapNumber": 7,
        },
    )

    list_route = next(route for route in app.routes if getattr(route, "path", "") == "/api/f1/fastf1/artifacts")
    listed = asyncio.run(
        list_route.endpoint(session_key="fastf1:2026:austria:r", kind="fastf1_telemetry_delta", limit=10)
    )
    assert listed["count"] == 1
    assert listed["artifacts"][0]["artifactId"] == record.artifact_id

    rows_route = next(
        route for route in app.routes if getattr(route, "path", "") == "/api/f1/fastf1/artifacts/{artifact_id}/rows"
    )
    preview = asyncio.run(rows_route.endpoint(str(record.artifact_id), limit=1))
    assert preview["artifact"]["kind"] == "fastf1_telemetry_delta"
    assert preview["rows"] == [{"Distance": 0, "DeltaSeconds": 0.0}]
    assert preview["truncated"] is True

    summary_route = next(
        route for route in app.routes if getattr(route, "path", "") == "/api/f1/fastf1/engineering-summary"
    )
    summary = asyncio.run(summary_route.endpoint(session_key="fastf1:2026:austria:r"))
    assert summary["telemetryDelta"]["finalDeltaSeconds"] == -0.02
    assert summary["cornerMetrics"][0]["driver"] == "VER"
    assert summary["cornerMetrics"][0]["corners"][0]["minimumSpeed"] == 150


class FakeFastF1Provider:
    def load_session(self, request: FastF1ImportRequest) -> FastF1LoadedSession:
        return FastF1LoadedSession(
            year=request.year,
            event_name=str(request.event),
            session_name=request.session_name,
            session_key="fastf1:2026:austria:r",
            laps=[
                {"Driver": "VER", "LapNumber": 7, "LapTimeSeconds": 65.1},
                {"Driver": "RUS", "LapNumber": 8, "LapTimeSeconds": 65.5},
            ],
            weather=[{"TimeSeconds": 1.0, "TrackTemp": 42.1}],
            race_control=[{"TimeSeconds": 2.0, "Message": "Track clear"}],
            telemetry_laps=[
                FastF1LapTelemetry("VER", 7, 65.1, _telemetry(speed=180)),
                FastF1LapTelemetry("RUS", 8, 65.5, _telemetry(speed=200)),
            ],
        )


class FakeFastF1ArtifactService:
    def __init__(self):
        self.seen = None

    def import_session(self, request: FastF1ImportRequest) -> FastF1ImportResult:
        self.seen = request
        loaded = FakeFastF1Provider().load_session(request)
        return FastF1ImportResult(
            session_key="fake-session",
            generated_at="2026-06-25T00:00:00Z",
            artifacts=[
                ArtifactRecord(
                    kind="fastf1_laps",
                    path="/tmp/fake/laps.jsonl",
                    format="jsonl",
                    row_count=1,
                    metadata={"sessionKey": "fake-session"},
                )
            ],
            notes=[],
            runtime_events=fastf1_runtime_events(loaded),
        )


def _telemetry(*, speed: float):
    speeds = {
        0: speed + 10,
        5: speed + 5,
        10: speed,
        15: speed + 5,
        20: speed + 10,
    }
    return [
        {
            "Distance": distance,
            "Speed": speeds[distance],
            "Throttle": 30 if distance == 10 else 80,
            "Brake": 1 if distance in {5, 10} else 0,
            "nGear": 6,
            "DRS": 8,
            "X": distance,
            "Y": distance / 2,
            "Z": 0,
        }
        for distance in (0, 5, 10, 15, 20)
    ]
