from f1_platform.replay import raw_event, run_replay
from f1_platform.reducer import F1StateReducer
from f1_platform.sample_data import SAMPLE_SESSION_KEY, sample_events
from f1_platform.track_geometry import TrackProjection


def test_reducer_keeps_highest_source_id_for_same_topic_key():
    reducer = F1StateReducer(session_key="s1")
    first = raw_event(
        10,
        "v1/laps",
        "63:lap:22",
        "s1",
        {"lap_number": 22, "lap_duration": 69.1},
        driver_number=63,
    )
    stale = raw_event(
        9,
        "v1/laps",
        "63:lap:22",
        "s1",
        {"lap_number": 22, "lap_duration": 67.0},
        driver_number=63,
    )
    corrected = raw_event(
        11,
        "v1/laps",
        "63:lap:22",
        "s1",
        {"lap_number": 22, "lap_duration": 68.4},
        driver_number=63,
    )

    assert reducer.ingest(first) is not None
    assert reducer.ingest(stale) is None
    assert reducer.ingest(corrected) is not None

    snapshot = reducer.snapshot()
    driver = snapshot.drivers[0]
    assert driver.driver_number == 63
    assert driver.last_lap_time == 68.4
    assert snapshot.seq == 2


def test_sector_only_lap_update_waits_for_completed_lap_duration():
    reducer = F1StateReducer(session_key="s1")

    partial = raw_event(
        10,
        "v1/laps",
        "63:lap:24",
        "s1",
        {"lap_number": 24, "duration_sector_1": 22.1},
        driver_number=63,
    )
    completed = raw_event(
        11,
        "v1/laps",
        "63:lap:24",
        "s1",
        {
            "lap_number": 24,
            "lap_duration": 68.9,
            "duration_sector_1": 22.1,
            "duration_sector_2": 29.4,
            "duration_sector_3": 17.4,
        },
        driver_number=63,
    )

    partial_update = reducer.ingest(partial)
    partial_snapshot = reducer.snapshot().to_dict()

    assert partial_update is not None
    assert partial_update.type == "lap.partial"
    assert partial_snapshot["drivers"][0]["current_lap"] == 24
    assert partial_snapshot["drivers"][0]["sector_times"]["sector_1"] == 22.1
    assert partial_snapshot["drivers"][0]["last_lap_time"] is None
    assert partial_snapshot["lapChart"] == []

    completed_update = reducer.ingest(completed)
    completed_snapshot = reducer.snapshot().to_dict()

    assert completed_update is not None
    assert completed_update.type == "lap.updated"
    assert completed_snapshot["drivers"][0]["last_lap_time"] == 68.9
    assert completed_snapshot["drivers"][0]["sector_times"]["sector_2"] == 29.4
    assert completed_snapshot["lapChart"] == [{"lap": 24, "driver_number": 63, "value": 68.9}]


def test_sample_replay_is_deterministic():
    events_a = sample_events(SAMPLE_SESSION_KEY)
    events_b = sample_events(SAMPLE_SESSION_KEY)

    reducer_a, updates_a = run_replay(events_a, session_key=SAMPLE_SESSION_KEY)
    reducer_b, updates_b = run_replay(events_b, session_key=SAMPLE_SESSION_KEY)

    snapshot_a = reducer_a.snapshot().to_dict()
    snapshot_b = reducer_b.snapshot().to_dict()

    snapshot_a.pop("generatedAt")
    snapshot_b.pop("generatedAt")

    assert [update.type for update in updates_a] == [update.type for update in updates_b]
    assert snapshot_a == snapshot_b


def test_location_uses_centerline_projector_when_available():
    reducer = F1StateReducer(session_key="s1", track_projector=StaticProjector())
    update = reducer.ingest(
        raw_event(
            1,
            "v1/location",
            "63:location",
            "s1",
            {"x": 12.0, "y": 3.0, "z": 0.0},
            driver_number=63,
        )
    )

    snapshot = reducer.snapshot()
    driver = snapshot.drivers[0]

    assert update is not None
    assert driver.track_progress == 0.42
    assert driver.last_location["projected_distance"] == 1234.5
    assert driver.last_location["projection_error"] == 2.5
    assert snapshot.replay["trackProjection"] == "centerline-test"


def test_location_crossings_build_custom_micro_sector_passages():
    reducer = F1StateReducer(session_key="s1")
    for source_id, (number, acronym, team, position) in enumerate(
        (
            (1, "VER", "Red Bull Racing", 1),
            (63, "RUS", "Mercedes", 2),
        ),
        start=1,
    ):
        reducer.ingest(
            raw_event(
                source_id,
                "v1/drivers",
                f"{number}:driver",
                "s1",
                {"name_acronym": acronym, "team_name": team},
                driver_number=number,
            )
        )
        reducer.ingest(
            raw_event(
                source_id + 10,
                "v1/position",
                f"{number}:position",
                "s1",
                {"position": position},
                driver_number=number,
            )
        )
        reducer.ingest(
            raw_event(
                source_id + 20,
                "v1/laps",
                f"{number}:lap:7",
                "s1",
                {"lap_number": 7, "lap_duration": 70.0 + position},
                driver_number=number,
            )
        )

    reducer.ingest(
        raw_event(
            100,
            "v1/location",
            "1:loc:0",
            "s1",
            {"x": 1000.0, "y": 0.0},
            driver_number=1,
            event_time="2026-06-28T13:32:18.000Z",
        )
    )
    reducer.ingest(
        raw_event(
            101,
            "v1/location",
            "1:loc:1",
            "s1",
            {"x": 1500.0, "y": 0.0},
            driver_number=1,
            event_time="2026-06-28T13:32:19.000Z",
        )
    )
    reducer.ingest(
        raw_event(
            102,
            "v1/location",
            "63:loc:0",
            "s1",
            {"x": 1000.0, "y": 0.0},
            driver_number=63,
            event_time="2026-06-28T13:32:18.000Z",
        )
    )
    reducer.ingest(
        raw_event(
            103,
            "v1/location",
            "63:loc:1",
            "s1",
            {"x": 1500.0, "y": 0.0},
            driver_number=63,
            event_time="2026-06-28T13:32:20.000Z",
        )
    )

    snapshot = reducer.snapshot().to_dict()
    passages = snapshot["customMicroSectors"]
    russell = [passage for passage in passages if passage["driver_number"] == 63][0]

    assert len(passages) == 2
    assert passages[0]["label"] == "custom micro-sector"
    assert passages[0]["sector_count"] == 25
    assert passages[0]["sector_index"] == 3
    assert passages[0]["source"] == "coordinate-fallback"
    assert passages[0]["passage_time"] == 1.0
    assert russell["passage_time"] == 2.0
    assert russell["car_ahead_delta"] == 1.0
    assert russell["session_best_delta"] == 1.0
    assert snapshot["replay"]["customMicroSectorCount"] == 25


def test_race_control_flags_update_track_status():
    reducer = F1StateReducer(session_key="s1")
    reducer.ingest(
        raw_event(
            1,
            "v1/drivers",
            "63:driver",
            "s1",
            {"name_acronym": "RUS"},
            driver_number=63,
        )
    )
    update = reducer.ingest(
        raw_event(
            2,
            "v1/race_control",
            "race-control:2",
            "s1",
            {"flag": "RED", "message": "Red flag - session suspended"},
        )
    )

    snapshot = reducer.snapshot()

    assert update is not None
    assert snapshot.drivers[0].track_status == "red_flag"
    assert snapshot.race_control[-1]["track_status"] == "red_flag"
    assert snapshot.replay["trackStatus"] == "red_flag"
    assert snapshot.replay["trackStatusSeq"] == snapshot.seq


def test_race_control_restart_messages_update_track_status():
    reducer = F1StateReducer(session_key="s1")
    reducer.ingest(
        raw_event(
            1,
            "v1/drivers",
            "63:driver",
            "s1",
            {"name_acronym": "RUS"},
            driver_number=63,
        )
    )
    update = reducer.ingest(
        raw_event(
            2,
            "v1/race_control",
            "race-control:2",
            "s1",
            {"message": "Race restarted after barrier repair"},
        )
    )

    snapshot = reducer.snapshot()

    assert update is not None
    assert snapshot.drivers[0].track_status == "restarted"
    assert snapshot.race_control[-1]["track_status"] == "restarted"
    assert snapshot.replay["trackStatus"] == "restarted"


def test_weather_events_keep_latest_and_sample_history():
    reducer = F1StateReducer(session_key="s1")
    first = raw_event(
        1,
        "v1/weather",
        "weather:2026-06-28T13:00:00Z",
        "s1",
        {
            "date": "2026-06-28T13:00:00Z",
            "air_temperature": 25.0,
            "track_temperature": 39.8,
            "rainfall": 0,
            "wind_speed": 2.1,
        },
        event_time="2026-06-28T13:00:00Z",
    )
    corrected = raw_event(
        3,
        "v1/weather",
        "weather:2026-06-28T13:00:00Z",
        "s1",
        {
            "date": "2026-06-28T13:00:00Z",
            "air_temperature": 25.2,
            "track_temperature": 40.2,
            "rainfall": 0,
            "wind_speed": 2.2,
        },
        event_time="2026-06-28T13:00:00Z",
    )
    later = raw_event(
        4,
        "v1/weather",
        "weather:2026-06-28T13:03:00Z",
        "s1",
        {
            "date": "2026-06-28T13:03:00Z",
            "air_temperature": 25.5,
            "track_temperature": 41.0,
            "rainfall": 0.1,
            "wind_speed": 2.8,
        },
        event_time="2026-06-28T13:03:00Z",
    )

    assert reducer.ingest(first) is not None
    assert reducer.ingest(corrected) is not None
    assert reducer.ingest(later) is not None
    snapshot = reducer.snapshot().to_dict()

    assert snapshot["weather"]["track_temperature"] == 41.0
    assert len(snapshot["weatherSamples"]) == 2
    assert snapshot["weatherSamples"][0]["track_temperature"] == 40.2
    assert snapshot["weatherSamples"][0]["source_id"] == 3
    assert snapshot["weatherSamples"][1]["rainfall"] == 0.1
    assert snapshot["replay"]["weatherSampleCount"] == 2


def test_session_events_are_exposed_in_snapshot_and_upserted():
    reducer = F1StateReducer(session_key="s1")
    first = raw_event(
        1,
        "v1/sessions",
        "session:s1",
        "s1",
        {
            "meeting_key": 2026001,
            "session_key": "s1",
            "session_name": "Sprint",
            "session_type": "Race",
            "date_start": "2026-06-25T10:00:00Z",
            "gmt_offset": "+02:00",
            "location": "Spielberg",
            "year": 2026,
            "is_cancelled": False,
        },
        event_time="2026-06-25T10:00:00Z",
    )
    corrected = raw_event(
        2,
        "v1/sessions",
        "session:s1",
        "s1",
        {
            "meeting_key": 2026001,
            "session_key": "s1",
            "session_name": "Race",
            "session_type": "Race",
            "date_start": "2026-06-25T13:00:00Z",
            "gmt_offset": "+02:00",
            "location": "Spielberg",
            "year": 2026,
            "is_cancelled": False,
        },
        event_time="2026-06-25T13:00:00Z",
    )

    assert reducer.ingest(first) is not None
    update = reducer.ingest(corrected)
    snapshot = reducer.snapshot().to_dict()

    assert update is not None
    assert update.type == "session.updated"
    assert snapshot["sessionInfo"]["session_name"] == "Race"
    assert snapshot["sessionInfo"]["session_type"] == "Race"
    assert snapshot["sessionInfo"]["location"] == "Spielberg"
    assert snapshot["sessionInfo"]["source_id"] == 2
    assert snapshot["replay"]["sessionName"] == "Race"


def test_overtake_events_are_exposed_in_snapshot_timeline():
    reducer = F1StateReducer(session_key="s1")
    update = reducer.ingest(
        raw_event(
            5,
            "v1/overtakes",
            "63:overtakes:44:12",
            "s1",
            {
                "overtaking_driver_number": 63,
                "overtaken_driver_number": 44,
                "lap_number": 12,
                "date": "2026-06-25T20:00:00Z",
            },
            driver_number=63,
            event_time="2026-06-25T20:00:00Z",
        )
    )

    snapshot = reducer.snapshot().to_dict()

    assert update is not None
    assert update.type == "overtake.updated"
    assert snapshot["overtakes"][-1]["overtaking_driver_number"] == 63
    assert snapshot["overtakes"][-1]["overtaken_driver_number"] == 44
    assert snapshot["overtakes"][-1]["lap_number"] == 12
    assert snapshot["overtakes"][-1]["source_id"] == 5


def test_pit_events_are_exposed_in_snapshot_timeline_and_upserted():
    reducer = F1StateReducer(session_key="s1")
    first = raw_event(
        7,
        "v1/pit",
        "63:pit:22",
        "s1",
        {"lap_number": 22, "pit_duration": 2.9, "date": "2026-06-25T20:01:00Z"},
        driver_number=63,
        event_time="2026-06-25T20:01:00Z",
    )
    corrected = raw_event(
        8,
        "v1/pit",
        "63:pit:22",
        "s1",
        {"lap_number": 22, "pit_duration": 2.7, "date": "2026-06-25T20:01:01Z"},
        driver_number=63,
        event_time="2026-06-25T20:01:01Z",
    )

    assert reducer.ingest(first) is not None
    assert reducer.ingest(corrected) is not None
    snapshot = reducer.snapshot().to_dict()

    assert snapshot["drivers"][0]["pit_status"] == "2.7s stop"
    assert len(snapshot["pitStops"]) == 1
    assert snapshot["pitStops"][0]["driver_number"] == 63
    assert snapshot["pitStops"][0]["lap_number"] == 22
    assert snapshot["pitStops"][0]["pit_duration"] == 2.7
    assert snapshot["pitStops"][0]["source_id"] == 8


def test_session_result_events_update_final_classification_and_upsert():
    reducer = F1StateReducer(session_key="s1")
    first = raw_event(
        9,
        "v1/session_result",
        "63:session_result",
        "s1",
        {
            "position": 3,
            "number_of_laps": 50,
            "duration": 5412.4,
            "gap_to_leader": 4.2,
            "dnf": False,
            "dns": False,
            "dsq": False,
        },
        driver_number=63,
        event_time="2026-06-25T20:12:00Z",
    )
    corrected = raw_event(
        10,
        "v1/session_result",
        "63:session_result",
        "s1",
        {
            "position": 2,
            "number_of_laps": 50,
            "duration": 5411.9,
            "gap_to_leader": 3.7,
            "dnf": False,
            "dns": False,
            "dsq": False,
        },
        driver_number=63,
        event_time="2026-06-25T20:12:30Z",
    )

    assert reducer.ingest(first) is not None
    update = reducer.ingest(corrected)
    snapshot = reducer.snapshot().to_dict()

    assert update is not None
    assert update.type == "session_result.updated"
    assert snapshot["drivers"][0]["position"] == 2
    assert len(snapshot["sessionResults"]) == 1
    assert snapshot["sessionResults"][0]["driver_number"] == 63
    assert snapshot["sessionResults"][0]["position"] == 2
    assert snapshot["sessionResults"][0]["duration"] == 5411.9
    assert snapshot["sessionResults"][0]["source_id"] == 10


def test_snapshot_contains_plan_surfaces():
    reducer, _ = run_replay(sample_events(), session_key=SAMPLE_SESSION_KEY)
    snapshot = reducer.snapshot().to_dict()

    assert snapshot["sessionKey"] == SAMPLE_SESSION_KEY
    assert len(snapshot["drivers"]) >= 6
    assert snapshot["drivers"][0]["position"] == 1
    assert snapshot["lapChart"]
    assert snapshot["strategyTimeline"]
    assert snapshot["raceControl"]
    assert snapshot["sessionInfo"]["session_name"] == "Race"
    assert snapshot["sessionResults"]
    assert snapshot["customMicroSectors"]
    assert snapshot["customMicroSectors"][0]["label"] == "custom micro-sector"
    assert snapshot["weather"]["track_temperature"] == 41.2
    assert len(snapshot["weatherSamples"]) == 3


class StaticProjector:
    def project(self, session_key, location):
        return TrackProjection(
            progress=0.42,
            distance=1234.5,
            x=10.0,
            y=2.0,
            z=0.0,
            error=2.5,
            source="centerline-test",
        )
