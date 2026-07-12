import asyncio
import json
import sqlite3

from f1_platform.projections import SqliteProjectionStore
from f1_platform.replay import raw_event
from f1_platform.runtime import F1PlatformRuntime
from f1_platform.sample_data import SAMPLE_SESSION_KEY, sample_events


def test_sqlite_projection_store_persists_snapshot_tables(tmp_path):
    async def run():
        db_path = tmp_path / "projection.sqlite"
        store = SqliteProjectionStore(db_path)
        store.initialize()
        runtime = F1PlatformRuntime(projection_store=store)

        snapshot = await runtime.reset_session(SAMPLE_SESSION_KEY, sample_events(SAMPLE_SESSION_KEY))
        counts = store.session_counts(SAMPLE_SESSION_KEY)

        assert counts["sessions"] == 1
        assert counts["sessionMetadata"] == 1
        assert counts["drivers"] == len(snapshot.drivers)
        assert counts["laps"] == len(snapshot.lap_chart)
        assert counts["stints"] == len(snapshot.strategy_timeline)
        assert counts["pitStops"] == len(snapshot.pit_stops)
        assert counts["raceControl"] == len(snapshot.race_control)
        assert counts["overtakes"] == len(snapshot.overtakes)
        assert counts["results"] == len(snapshot.session_results)
        assert counts["weatherSamples"] == len(snapshot.weather_samples)
        assert counts["customMicroSectors"] == len(snapshot.custom_micro_sectors)
        assert counts["predictions"] == len(snapshot.predictions)
        assert counts["derivedAnalytics"] == 5

        analytics = store.derived_analytics(SAMPLE_SESSION_KEY)
        tyre_degradation = analytics["analytics"]["tyre_degradation_v1"]
        weather_evolution = analytics["analytics"]["weather_evolution_v1"]
        pace_analysis = analytics["analytics"]["pace_analysis_v1"]
        battle_dashboard = analytics["analytics"]["battle_dashboard_v1"]
        assert tyre_degradation["status"] == "ok"
        assert tyre_degradation["sampleCount"] == len(snapshot.lap_chart)
        assert weather_evolution["sampleCount"] == len(snapshot.weather_samples)
        assert weather_evolution["latest"]["trackTemperature"] == 41.2
        assert pace_analysis["driverCount"] == len(snapshot.drivers)
        assert battle_dashboard["battleCount"] == len(snapshot.drivers) - 1
        assert "projection_summary" in analytics["analytics"]
        assert analytics["analytics"]["projection_summary"]["sessionMetadata"] == 1
        assert analytics["analytics"]["projection_summary"]["pitStops"] == len(snapshot.pit_stops)
        assert analytics["analytics"]["projection_summary"]["overtakes"] == len(snapshot.overtakes)
        assert analytics["analytics"]["projection_summary"]["sessionResults"] == len(snapshot.session_results)
        assert analytics["analytics"]["projection_summary"]["weatherSamples"] == len(snapshot.weather_samples)
        assert analytics["analytics"]["projection_summary"]["customMicroSectors"] == len(snapshot.custom_micro_sectors)

        persisted_prediction = snapshot.predictions[0]
        persisted_prediction.strategy = {
            "recommendedAction": "pit_next_lap",
            "policyVersion": "projection_test_v1",
        }
        persisted_prediction.forecast_available = False
        persisted_prediction.unavailable_reason = "did_not_start"
        persisted_prediction.eligibility_status = "not_eligible"
        persisted_prediction.participation_status = "dns"
        store.project_snapshot(snapshot)

        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT position_p10, position_p90, prediction_kind,
                       position_semantics, strategy_json, forecast_available,
                       unavailable_reason, eligibility_status, participation_status
                FROM f1_predictions
                WHERE session_key = ? AND driver_number = ?
                ORDER BY prediction_time DESC
                LIMIT 1
                """,
                (str(SAMPLE_SESSION_KEY), persisted_prediction.driver_number),
            ).fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[1] is not None
        assert row[0] <= row[1]
        assert row[2] == "race"
        assert row[3] == "race_finish_order"
        assert json.loads(row[4]) == {
            "recommendedAction": "pit_next_lap",
            "policyVersion": "projection_test_v1",
        }
        assert row[5:] == (0, "did_not_start", "not_eligible", "dns")

    asyncio.run(run())


def test_sqlite_projection_store_migrates_prediction_range_columns(tmp_path):
    db_path = tmp_path / "projection.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE f1_predictions (
                session_key TEXT NOT NULL,
                driver_number INTEGER NOT NULL,
                source_event_sequence INTEGER NOT NULL,
                prediction_time TEXT NOT NULL,
                model_version TEXT NOT NULL,
                features_version TEXT NOT NULL,
                expected_position REAL,
                win_probability REAL NOT NULL,
                podium_probability REAL NOT NULL,
                points_probability REAL NOT NULL,
                dnf_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                position_distribution_json TEXT NOT NULL,
                PRIMARY KEY (session_key, driver_number, source_event_sequence, model_version)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO f1_predictions (
                session_key, driver_number, source_event_sequence,
                prediction_time, model_version, features_version,
                expected_position, win_probability, podium_probability,
                points_probability, dnf_probability, confidence,
                position_distribution_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-session",
                63,
                8,
                "2026-07-01T00:00:00Z",
                "legacy_model",
                "legacy_features",
                2.0,
                0.2,
                0.4,
                0.8,
                0.1,
                0.5,
                '{"1":0.2,"2":0.8}',
            ),
        )

    store = SqliteProjectionStore(db_path)
    store.initialize()

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(f1_predictions)").fetchall()}
        legacy = conn.execute(
            """
            SELECT prediction_kind, position_semantics, strategy_json,
                   forecast_available, unavailable_reason,
                   eligibility_status, participation_status
            FROM f1_predictions
            WHERE session_key = 'legacy-session'
            """
        ).fetchone()
    assert "position_p10" in columns
    assert "position_p90" in columns
    assert "prediction_kind" in columns
    assert "position_semantics" in columns
    assert "strategy_json" in columns
    assert "forecast_available" in columns
    assert "unavailable_reason" in columns
    assert "eligibility_status" in columns
    assert "participation_status" in columns
    assert legacy == (
        "race",
        "race_finish_order",
        None,
        1,
        None,
        "classification_eligible",
        "running_or_unknown",
    )


def test_runtime_snapshot_reads_do_not_create_new_predictions(tmp_path):
    async def run():
        store = SqliteProjectionStore(tmp_path / "projection.sqlite")
        store.initialize()
        runtime = F1PlatformRuntime(projection_store=store)
        await runtime.reset_session(SAMPLE_SESSION_KEY, sample_events(SAMPLE_SESSION_KEY))

        first = await runtime.snapshot(SAMPLE_SESSION_KEY)
        second = await runtime.snapshot(SAMPLE_SESSION_KEY)
        counts = store.session_counts(SAMPLE_SESSION_KEY)

        assert len(first.predictions) == len(second.predictions)
        assert counts["predictions"] == len(first.predictions)

    asyncio.run(run())


def test_projection_updates_after_live_ingest(tmp_path):
    async def run():
        store = SqliteProjectionStore(tmp_path / "projection.sqlite")
        store.initialize()
        runtime = F1PlatformRuntime(projection_store=store)
        await runtime.reset_session(SAMPLE_SESSION_KEY, sample_events(SAMPLE_SESSION_KEY))
        before = store.session_counts(SAMPLE_SESSION_KEY)

        partial_event = raw_event(
            10_000,
            "v1/laps",
            "63:lap:99",
            SAMPLE_SESSION_KEY,
            {"lap_number": 99, "duration_sector_1": 22.431},
            driver_number=63,
        )
        partial_update = await runtime.ingest(partial_event)
        after_partial = store.session_counts(SAMPLE_SESSION_KEY)

        assert partial_update is not None
        assert partial_update.type == "lap.partial"
        assert after_partial["laps"] == before["laps"]
        assert after_partial["predictions"] == before["predictions"]

        event = raw_event(
            10_001,
            "v1/laps",
            "63:lap:99",
            SAMPLE_SESSION_KEY,
            {"lap_number": 99, "lap_duration": 67.432},
            driver_number=63,
        )
        update = await runtime.ingest(event)
        after = store.session_counts(SAMPLE_SESSION_KEY)

        assert update is not None
        assert after["laps"] == before["laps"] + 1
        assert after["predictions"] > before["predictions"]

    asyncio.run(run())


def test_projection_persists_overtakes_and_prediction_refresh(tmp_path):
    async def run():
        store = SqliteProjectionStore(tmp_path / "projection.sqlite")
        store.initialize()
        runtime = F1PlatformRuntime(projection_store=store)
        initial = await runtime.reset_session(SAMPLE_SESSION_KEY, sample_events(SAMPLE_SESSION_KEY))
        before = store.session_counts(SAMPLE_SESSION_KEY)

        event = raw_event(
            20_001,
            "v1/overtakes",
            "63:overtakes:44:12",
            SAMPLE_SESSION_KEY,
            {
                "overtaking_driver_number": 63,
                "overtaken_driver_number": 44,
                "lap_number": 12,
                "date": "2026-06-25T20:00:00Z",
            },
            driver_number=63,
            event_time="2026-06-25T20:00:00Z",
        )
        update = await runtime.ingest(event)
        after = store.session_counts(SAMPLE_SESSION_KEY)
        analytics = store.derived_analytics(SAMPLE_SESSION_KEY)

        assert update is not None
        assert update.type == "overtake.updated"
        assert after["overtakes"] == before["overtakes"] + 1
        assert after["predictions"] > len(initial.predictions)
        assert analytics["analytics"]["projection_summary"]["overtakes"] == 1

    asyncio.run(run())


def test_projection_persists_pit_stops_and_prediction_refresh(tmp_path):
    async def run():
        store = SqliteProjectionStore(tmp_path / "projection.sqlite")
        store.initialize()
        runtime = F1PlatformRuntime(projection_store=store)
        initial = await runtime.reset_session(SAMPLE_SESSION_KEY, sample_events(SAMPLE_SESSION_KEY))
        before = store.session_counts(SAMPLE_SESSION_KEY)

        event = raw_event(
            30_001,
            "v1/pit",
            "63:pit:22",
            SAMPLE_SESSION_KEY,
            {"lap_number": 22, "pit_duration": 2.8, "date": "2026-06-25T20:02:00Z"},
            driver_number=63,
            event_time="2026-06-25T20:02:00Z",
        )
        update = await runtime.ingest(event)
        after = store.session_counts(SAMPLE_SESSION_KEY)
        analytics = store.derived_analytics(SAMPLE_SESSION_KEY)

        assert update is not None
        assert update.type == "pit.updated"
        assert after["pitStops"] == before["pitStops"] + 1
        assert after["predictions"] > len(initial.predictions)
        assert analytics["analytics"]["projection_summary"]["pitStops"] == 1

    asyncio.run(run())


def test_projection_persists_session_results_and_prediction_refresh(tmp_path):
    async def run():
        store = SqliteProjectionStore(tmp_path / "projection.sqlite")
        store.initialize()
        runtime = F1PlatformRuntime(projection_store=store)
        initial = await runtime.reset_session(SAMPLE_SESSION_KEY, sample_events(SAMPLE_SESSION_KEY))
        before = store.session_counts(SAMPLE_SESSION_KEY)

        event = raw_event(
            40_001,
            "v1/session_result",
            "63:session_result",
            SAMPLE_SESSION_KEY,
            {
                "position": 2,
                "number_of_laps": 57,
                "duration": 5441.2,
                "gap_to_leader": 1.4,
                "dnf": False,
                "dns": False,
                "dsq": False,
            },
            driver_number=63,
            event_time="2026-06-25T20:05:00Z",
        )
        update = await runtime.ingest(event)
        after = store.session_counts(SAMPLE_SESSION_KEY)
        analytics = store.derived_analytics(SAMPLE_SESSION_KEY)

        assert update is not None
        assert update.type == "session_result.updated"
        assert after["results"] == before["results"]
        assert after["predictions"] > len(initial.predictions)
        assert analytics["analytics"]["projection_summary"]["sessionResults"] == len(initial.session_results)

    asyncio.run(run())
