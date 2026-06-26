from f1_platform.analytics import (
    BATTLE_DASHBOARD_ANALYTIC_NAME,
    PACE_ANALYSIS_ANALYTIC_NAME,
    TYRE_DEGRADATION_ANALYTIC_NAME,
    WEATHER_EVOLUTION_ANALYTIC_NAME,
    build_battle_dashboard_analytics,
    build_pace_analysis_analytics,
    build_tyre_degradation_analytics,
    build_weather_evolution_analytics,
)
from f1_platform.reducer import F1StateReducer
from f1_platform.sample_data import SAMPLE_SESSION_KEY, sample_events
from f1_platform.schemas import LapPoint, SessionSnapshot, StintSegment


def test_tyre_degradation_analytics_joins_stints_and_adjusts_pace():
    reducer = F1StateReducer(SAMPLE_SESSION_KEY)
    for event in sample_events(SAMPLE_SESSION_KEY):
        reducer.ingest(event)

    payload = build_tyre_degradation_analytics(reducer.snapshot())

    assert payload["status"] == "ok"
    assert payload["sampleCount"] == 18
    assert payload["excludedCount"] == 0
    assert payload["method"] != "raw lap time by tyre age"
    assert "field_lap_median_fuel_track_adjustment" in payload["adjustments"]
    assert "driver_baseline_adjustment" in payload["adjustments"]

    by_compound = {compound["compound"]: compound for compound in payload["compounds"]}
    assert set(by_compound) == {"HARD", "MEDIUM"}
    assert {row["tyreAge"] for row in by_compound["MEDIUM"]["byTyreAge"]} == {9, 10, 11}
    assert by_compound["MEDIUM"]["slopeSecondsPerTyreLap"] is not None
    assert by_compound["HARD"]["slopeConfidenceInterval95"] is not None
    assert payload["cleanLapSample"][0]["adjustedPace"] is not None
    assert TYRE_DEGRADATION_ANALYTIC_NAME == "tyre_degradation_v1"


def test_tyre_degradation_analytics_filters_dirty_laps():
    snapshot = SessionSnapshot(
        session_key="dirty-filter",
        seq=1,
        generated_at="2026-06-25T00:00:00Z",
        source="test",
        drivers=[],
        lap_chart=[
            LapPoint(lap=1, driver_number=1, value=69.0),
            LapPoint(lap=1, driver_number=2, value=69.2),
            LapPoint(lap=1, driver_number=3, value=69.1),
            LapPoint(lap=2, driver_number=1, value=68.8),
            LapPoint(lap=2, driver_number=2, value=68.9),
            LapPoint(lap=2, driver_number=3, value=69.0),
            LapPoint(lap=3, driver_number=1, value=68.7),
            LapPoint(lap=3, driver_number=2, value=68.8),
            LapPoint(lap=3, driver_number=3, value=68.9),
            LapPoint(lap=4, driver_number=1, value=68.6),
            LapPoint(lap=4, driver_number=2, value=68.7),
            LapPoint(lap=4, driver_number=3, value=120.0),
        ],
        strategy_timeline=[
            StintSegment(driver_number=1, stint_number=1, compound="MEDIUM", start_lap=1, end_lap=None, tyre_age_start=0),
            StintSegment(driver_number=2, stint_number=1, compound="MEDIUM", start_lap=1, end_lap=None, tyre_age_start=0),
            StintSegment(driver_number=3, stint_number=1, compound="MEDIUM", start_lap=1, end_lap=None, tyre_age_start=0),
        ],
        race_control=[{"flag": "YELLOW", "lap": 3, "message": "Yellow flag sector 2"}],
    )

    payload = build_tyre_degradation_analytics(snapshot)

    assert payload["sampleCount"] == 5
    assert payload["filters"]["pit_boundary"] == 3
    assert payload["filters"]["race_status"] == 3
    assert payload["filters"]["field_anomaly"] == 1
    assert payload["compounds"][0]["cleanLapCount"] == 5


def test_weather_evolution_analytics_tracks_temperature_and_rainfall():
    reducer = F1StateReducer(SAMPLE_SESSION_KEY)
    for event in sample_events(SAMPLE_SESSION_KEY):
        reducer.ingest(event)

    payload = build_weather_evolution_analytics(reducer.snapshot())

    assert payload["status"] == "ok"
    assert payload["sampleCount"] == 3
    assert payload["latest"]["trackTemperature"] == 41.2
    assert payload["trackTemperatureDelta"] == 0.6
    assert payload["airTemperatureDelta"] == 0.3
    assert payload["windSpeedDelta"] == 0.3
    assert payload["rainfallDetected"] is False
    assert payload["series"][0]["eventTime"] == "2026-06-28T13:32:18.450Z"
    assert WEATHER_EVOLUTION_ANALYTIC_NAME == "weather_evolution_v1"


def test_pace_analysis_analytics_summarizes_reduced_lap_pace():
    reducer = F1StateReducer(SAMPLE_SESSION_KEY)
    for event in sample_events(SAMPLE_SESSION_KEY):
        reducer.ingest(event)

    payload = build_pace_analysis_analytics(reducer.snapshot())

    assert payload["status"] == "ok"
    assert payload["driverCount"] == 6
    assert len(payload["fieldSeries"]) == 3
    assert payload["fieldSeries"][-1]["lap"] == 23
    assert payload["drivers"][0]["driverNumber"] == 1
    assert payload["drivers"][0]["medianLapTime"] == 68.398
    assert payload["drivers"][0]["rollingMedianLast3"] == 68.398
    assert payload["drivers"][0]["trendLastVsFirst"] < 0
    assert payload["drivers"][0]["consistencyStdSeconds"] is not None
    assert PACE_ANALYSIS_ANALYTIC_NAME == "pace_analysis_v1"


def test_battle_dashboard_analytics_finds_drs_trains_and_windows():
    reducer = F1StateReducer(SAMPLE_SESSION_KEY)
    for event in sample_events(SAMPLE_SESSION_KEY):
        reducer.ingest(event)

    payload = build_battle_dashboard_analytics(reducer.snapshot())

    assert payload["status"] == "ok"
    assert payload["battleCount"] == 5
    assert payload["activeOvertakeWindows"] >= 2
    assert len(payload["drsTrains"]) == 2

    active_pairs = {
        (battle["ahead"]["driverNumber"], battle["chaser"]["driverNumber"])
        for battle in payload["battles"]
        if battle["windowState"] == "active"
    }
    assert (63, 16) in active_pairs
    assert (4, 81) in active_pairs
    assert payload["battles"][0]["overtakeWindowProbability"] >= payload["battles"][-1]["overtakeWindowProbability"]
    assert BATTLE_DASHBOARD_ANALYTIC_NAME == "battle_dashboard_v1"
