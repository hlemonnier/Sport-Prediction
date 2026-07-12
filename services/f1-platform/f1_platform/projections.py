"""Durable SQL projections for F1 session state.

Raw events remain replayable in JSONL/Redis Streams. This module stores the
query-oriented current projection expected by the platform plan: sessions,
drivers, laps, stints, pit stops, race-control messages, overtakes, session
results, weather samples, custom micro-sectors, predictions, and derived
analytics. The SQL is intentionally conservative so the same contract can be
tested with SQLite and deployed against PostgreSQL.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Protocol

from .analytics import build_projection_analytics
from .schemas import JsonObject, SessionSnapshot
from .time import utc_now_iso


class ProjectionStore(Protocol):
    kind: str

    def initialize(self) -> None:
        ...

    def project_snapshot(self, snapshot: SessionSnapshot) -> None:
        ...

    def session_counts(self, session_key: int | str) -> JsonObject:
        ...

    def derived_analytics(self, session_key: int | str) -> JsonObject:
        ...


class NoopProjectionStore:
    kind = "disabled"

    def initialize(self) -> None:
        return None

    def project_snapshot(self, snapshot: SessionSnapshot) -> None:
        return None

    def session_counts(self, session_key: int | str) -> JsonObject:
        return {
            "sessionKey": str(session_key),
            "enabled": False,
            "sessions": 0,
            "sessionMetadata": 0,
            "drivers": 0,
            "laps": 0,
            "stints": 0,
            "pitStops": 0,
            "raceControl": 0,
            "overtakes": 0,
            "results": 0,
            "weatherSamples": 0,
            "customMicroSectors": 0,
            "predictions": 0,
            "derivedAnalytics": 0,
        }

    def derived_analytics(self, session_key: int | str) -> JsonObject:
        return {"sessionKey": str(session_key), "enabled": False, "analytics": {}}


class SqlProjectionStore:
    kind = "sql"

    def __init__(
        self,
        connect: Callable[[], Any],
        *,
        placeholder: str = "?",
        kind: str = "sql",
    ) -> None:
        self._connect = connect
        self._placeholder = placeholder
        self.kind = kind

    def initialize(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                for statement in _SCHEMA:
                    conn.execute(statement)
                self._ensure_prediction_contract_columns(conn)

    def project_snapshot(self, snapshot: SessionSnapshot) -> None:
        session_key = str(snapshot.session_key)
        now = utc_now_iso()
        with closing(self._connect()) as conn:
            with conn:
                self._execute(
                    conn,
                    """
                    INSERT INTO f1_sessions (
                        session_key, source, seq, generated_at, event_count,
                        replay_mode, weather_json, topic_watermarks_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_key) DO UPDATE SET
                        source = excluded.source,
                        seq = excluded.seq,
                        generated_at = excluded.generated_at,
                        event_count = excluded.event_count,
                        replay_mode = excluded.replay_mode,
                        weather_json = excluded.weather_json,
                        topic_watermarks_json = excluded.topic_watermarks_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        session_key,
                        snapshot.source,
                        snapshot.seq,
                        snapshot.generated_at,
                        _optional_int(snapshot.replay.get("eventCount")) or snapshot.seq,
                        _optional_str(snapshot.replay.get("mode")),
                        _json(snapshot.weather),
                        _json(snapshot.topic_watermarks),
                        now,
                    ),
                )

                self._replace_children(conn, "f1_session_metadata", session_key)
                if snapshot.session_info:
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_session_metadata (
                            session_key, meeting_key, session_name, session_type,
                            date_start, date_end, gmt_offset, location, year,
                            is_cancelled, source_id, event_time, session_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_key,
                            _optional_int(snapshot.session_info.get("meeting_key")),
                            _optional_str(snapshot.session_info.get("session_name")),
                            _optional_str(snapshot.session_info.get("session_type")),
                            _optional_str(snapshot.session_info.get("date_start")),
                            _optional_str(snapshot.session_info.get("date_end")),
                            _optional_str(snapshot.session_info.get("gmt_offset")),
                            _optional_str(snapshot.session_info.get("location")),
                            _optional_int(snapshot.session_info.get("year")),
                            _optional_bool_int(snapshot.session_info.get("is_cancelled")),
                            _optional_int(snapshot.session_info.get("source_id")),
                            _optional_str(snapshot.session_info.get("event_time")),
                            _json(snapshot.session_info),
                        ),
                    )

                self._replace_children(conn, "f1_weather_samples", session_key)
                for idx, sample in enumerate(snapshot.weather_samples):
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_weather_samples (
                            session_key, idx, event_time, air_temperature,
                            track_temperature, rainfall, wind_speed,
                            source_id, source_key, weather_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_key,
                            idx,
                            _optional_str(sample.get("event_time") or sample.get("date")),
                            _optional_float(sample.get("air_temperature")),
                            _optional_float(sample.get("track_temperature")),
                            _optional_float(sample.get("rainfall")),
                            _optional_float(sample.get("wind_speed")),
                            _optional_int(sample.get("source_id")),
                            _optional_str(sample.get("source_key")),
                            _json(sample),
                        ),
                    )

                self._replace_children(conn, "f1_drivers", session_key)
                for driver in snapshot.drivers:
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_drivers (
                            session_key, driver_number, position, acronym, full_name,
                            team_name, team_colour, current_lap, last_lap_time,
                            best_lap_time, gap_to_leader, interval, current_compound,
                            tyre_age, stint_number, pit_status, track_status,
                            last_speed, track_progress, last_location_json,
                            sector_times_json, last_update_seq
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_key,
                            driver.driver_number,
                            driver.position,
                            driver.acronym,
                            driver.full_name,
                            driver.team_name,
                            driver.team_colour,
                            driver.current_lap,
                            driver.last_lap_time,
                            driver.best_lap_time,
                            driver.gap_to_leader,
                            driver.interval,
                            driver.current_compound,
                            driver.tyre_age,
                            driver.stint_number,
                            driver.pit_status,
                            driver.track_status,
                            driver.last_speed,
                            driver.track_progress,
                            _json(driver.last_location),
                            _json(driver.sector_times),
                            driver.last_update_seq,
                        ),
                    )

                self._replace_children(conn, "f1_laps", session_key)
                for point in snapshot.lap_chart:
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_laps (session_key, driver_number, lap, lap_time)
                        VALUES (?, ?, ?, ?)
                        """,
                        (session_key, point.driver_number, point.lap, point.value),
                    )

                self._replace_children(conn, "f1_stints", session_key)
                for segment in snapshot.strategy_timeline:
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_stints (
                            session_key, driver_number, stint_number, compound,
                            start_lap, end_lap, tyre_age_start
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_key,
                            segment.driver_number,
                            segment.stint_number,
                            segment.compound,
                            segment.start_lap,
                            segment.end_lap,
                            segment.tyre_age_start,
                        ),
                    )

                self._replace_children(conn, "f1_pit_stops", session_key)
                for idx, pit_stop in enumerate(snapshot.pit_stops):
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_pit_stops (
                            session_key, idx, driver_number, lap, pit_duration,
                            event_time, pit_stop_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_key,
                            idx,
                            _optional_int(pit_stop.get("driver_number")),
                            _optional_int(pit_stop.get("lap_number")),
                            _optional_float(pit_stop.get("pit_duration")),
                            _optional_str(pit_stop.get("event_time") or pit_stop.get("date")),
                            _json(pit_stop),
                        ),
                    )

                self._replace_children(conn, "f1_race_control", session_key)
                for idx, message in enumerate(snapshot.race_control):
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_race_control (session_key, idx, message_json)
                        VALUES (?, ?, ?)
                        """,
                        (session_key, idx, _json(message)),
                    )

                self._replace_children(conn, "f1_overtakes", session_key)
                for idx, overtake in enumerate(snapshot.overtakes):
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_overtakes (
                            session_key, idx, lap, overtaking_driver_number,
                            overtaken_driver_number, event_time, overtake_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_key,
                            idx,
                            _optional_int(overtake.get("lap_number")),
                            _optional_int(overtake.get("overtaking_driver_number")),
                            _optional_int(overtake.get("overtaken_driver_number")),
                            _optional_str(overtake.get("event_time") or overtake.get("date")),
                            _json(overtake),
                        ),
                    )

                self._replace_children(conn, "f1_results", session_key)
                for result in snapshot.session_results:
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_results (
                            session_key, driver_number, position, number_of_laps,
                            duration, gap_to_leader, dnf, dns, dsq, event_time,
                            result_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_key,
                            _optional_int(result.get("driver_number")),
                            _optional_int(result.get("position")),
                            _optional_int(result.get("number_of_laps")),
                            _optional_float(result.get("duration")),
                            _optional_str(result.get("gap_to_leader")),
                            _optional_bool_int(result.get("dnf")),
                            _optional_bool_int(result.get("dns")),
                            _optional_bool_int(result.get("dsq")),
                            _optional_str(result.get("event_time") or result.get("date")),
                            _json(result),
                        ),
                    )

                self._replace_children(conn, "f1_custom_micro_sectors", session_key)
                for passage in snapshot.custom_micro_sectors:
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_custom_micro_sectors (
                            session_key, driver_number, lap, sector_index,
                            sector_count, progress_start, progress_end,
                            passage_time, personal_best_delta,
                            session_best_delta, car_ahead_delta, teammate_delta,
                            label, source, event_time, seq, passage_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_key,
                            passage.driver_number,
                            passage.lap if passage.lap is not None else 0,
                            passage.sector_index,
                            passage.sector_count,
                            passage.progress_start,
                            passage.progress_end,
                            passage.passage_time,
                            passage.personal_best_delta,
                            passage.session_best_delta,
                            passage.car_ahead_delta,
                            passage.teammate_delta,
                            passage.label,
                            passage.source,
                            passage.event_time,
                            passage.seq,
                            _json(passage.to_dict()),
                        ),
                    )

                for prediction in snapshot.predictions:
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_predictions (
                            session_key, driver_number, source_event_sequence,
                            prediction_time, model_version, features_version,
                            prediction_kind, position_semantics,
                            forecast_available, unavailable_reason,
                            eligibility_status, participation_status,
                            expected_position, position_p10, position_p90,
                            win_probability, podium_probability, points_probability,
                            dnf_probability, confidence,
                            position_distribution_json, strategy_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(
                            session_key, driver_number, source_event_sequence, model_version
                        ) DO UPDATE SET
                            prediction_time = excluded.prediction_time,
                            features_version = excluded.features_version,
                            prediction_kind = excluded.prediction_kind,
                            position_semantics = excluded.position_semantics,
                            forecast_available = excluded.forecast_available,
                            unavailable_reason = excluded.unavailable_reason,
                            eligibility_status = excluded.eligibility_status,
                            participation_status = excluded.participation_status,
                            expected_position = excluded.expected_position,
                            position_p10 = excluded.position_p10,
                            position_p90 = excluded.position_p90,
                            win_probability = excluded.win_probability,
                            podium_probability = excluded.podium_probability,
                            points_probability = excluded.points_probability,
                            dnf_probability = excluded.dnf_probability,
                            confidence = excluded.confidence,
                            position_distribution_json = excluded.position_distribution_json,
                            strategy_json = excluded.strategy_json
                        """,
                        (
                            session_key,
                            prediction.driver_number,
                            prediction.source_event_sequence,
                            prediction.prediction_time,
                            prediction.model_version,
                            prediction.features_version,
                            prediction.prediction_kind,
                            prediction.position_semantics,
                            int(bool(prediction.forecast_available)),
                            prediction.unavailable_reason,
                            prediction.eligibility_status,
                            prediction.participation_status,
                            prediction.expected_position,
                            prediction.position_p10,
                            prediction.position_p90,
                            prediction.win_probability,
                            prediction.podium_probability,
                            prediction.points_probability,
                            prediction.dnf_probability,
                            prediction.confidence,
                            _json(prediction.position_distribution),
                            _json(prediction.strategy) if prediction.strategy is not None else None,
                        ),
                    )

                for name, payload in build_projection_analytics(snapshot).items():
                    self._execute(
                        conn,
                        """
                        INSERT INTO f1_derived_analytics (session_key, name, payload_json, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(session_key, name) DO UPDATE SET
                            payload_json = excluded.payload_json,
                            updated_at = excluded.updated_at
                        """,
                        (session_key, name, _json(payload), now),
                    )

    def session_counts(self, session_key: int | str) -> JsonObject:
        session_id = str(session_key)
        with closing(self._connect()) as conn:
            return {
                "sessionKey": session_id,
                "enabled": True,
                "kind": self.kind,
                "sessions": self._count(conn, "f1_sessions", session_id),
                "sessionMetadata": self._count(conn, "f1_session_metadata", session_id),
                "drivers": self._count(conn, "f1_drivers", session_id),
                "laps": self._count(conn, "f1_laps", session_id),
                "stints": self._count(conn, "f1_stints", session_id),
                "pitStops": self._count(conn, "f1_pit_stops", session_id),
                "raceControl": self._count(conn, "f1_race_control", session_id),
                "overtakes": self._count(conn, "f1_overtakes", session_id),
                "results": self._count(conn, "f1_results", session_id),
                "weatherSamples": self._count(conn, "f1_weather_samples", session_id),
                "customMicroSectors": self._count(conn, "f1_custom_micro_sectors", session_id),
                "predictions": self._count(conn, "f1_predictions", session_id),
                "derivedAnalytics": self._count(conn, "f1_derived_analytics", session_id),
            }

    def derived_analytics(self, session_key: int | str) -> JsonObject:
        session_id = str(session_key)
        with closing(self._connect()) as conn:
            cursor = self._execute(
                conn,
                """
                SELECT name, payload_json, updated_at
                FROM f1_derived_analytics
                WHERE session_key = ?
                ORDER BY name
                """,
                (session_id,),
            )
            analytics: JsonObject = {}
            updated_at: dict[str, str] = {}
            for name, payload_json, row_updated_at in cursor.fetchall():
                analytics[str(name)] = json.loads(payload_json)
                updated_at[str(name)] = str(row_updated_at)
            return {
                "sessionKey": session_id,
                "enabled": True,
                "kind": self.kind,
                "analytics": analytics,
                "updatedAt": updated_at,
            }

    def _replace_children(self, conn: Any, table: str, session_key: str) -> None:
        self._execute(conn, f"DELETE FROM {table} WHERE session_key = ?", (session_key,))

    def _count(self, conn: Any, table: str, session_key: str) -> int:
        cursor = self._execute(conn, f"SELECT COUNT(*) FROM {table} WHERE session_key = ?", (session_key,))
        row = cursor.fetchone()
        return int(row[0] if row is not None else 0)

    def _execute(self, conn: Any, sql: str, params: tuple[Any, ...] = ()):
        return conn.execute(self._sql(sql), params)

    def _sql(self, sql: str) -> str:
        if self._placeholder == "?":
            return sql
        return sql.replace("?", self._placeholder)

    def _ensure_prediction_contract_columns(self, conn: Any) -> None:
        required = {
            "position_p10": "REAL",
            "position_p90": "REAL",
            "prediction_kind": "TEXT NOT NULL DEFAULT 'race'",
            "position_semantics": "TEXT NOT NULL DEFAULT 'race_finish_order'",
            "strategy_json": "TEXT",
            "forecast_available": "INTEGER NOT NULL DEFAULT 1",
            "unavailable_reason": "TEXT",
            "eligibility_status": "TEXT NOT NULL DEFAULT 'classification_eligible'",
            "participation_status": "TEXT NOT NULL DEFAULT 'running_or_unknown'",
        }
        if self._placeholder == "?":
            cursor = conn.execute("PRAGMA table_info(f1_predictions)")
            existing = {str(row[1]) for row in cursor.fetchall()}
        else:
            existing = set()
            for column in required:
                cursor = self._execute(
                    conn,
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = ? AND column_name = ?
                    """,
                    ("f1_predictions", column),
                )
                if cursor.fetchone() is not None:
                    existing.add(column)

        for column, column_type in required.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE f1_predictions ADD COLUMN {column} {column_type}")


class SqliteProjectionStore(SqlProjectionStore):
    kind = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(self._connect, kind="sqlite")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


class PostgresProjectionStore(SqlProjectionStore):
    kind = "postgres"

    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install psycopg[binary] to enable PostgreSQL projections") from exc
        self._psycopg = psycopg
        self.database_url = database_url
        super().__init__(self._connect, placeholder="%s", kind="postgres")

    def _connect(self):
        return self._psycopg.connect(self.database_url)


def projection_store_from_config(database_url: str | None, sqlite_path: str | Path | None) -> ProjectionStore:
    if database_url:
        store: ProjectionStore = PostgresProjectionStore(database_url)
    elif sqlite_path:
        store = SqliteProjectionStore(sqlite_path)
    else:
        store = NoopProjectionStore()
    store.initialize()
    return store


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_bool_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1
    if text in {"false", "0", "no", "n"}:
        return 0
    return None


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS f1_sessions (
        session_key TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        seq INTEGER NOT NULL,
        generated_at TEXT NOT NULL,
        event_count INTEGER NOT NULL,
        replay_mode TEXT,
        weather_json TEXT NOT NULL,
        topic_watermarks_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_session_metadata (
        session_key TEXT NOT NULL,
        meeting_key INTEGER,
        session_name TEXT,
        session_type TEXT,
        date_start TEXT,
        date_end TEXT,
        gmt_offset TEXT,
        location TEXT,
        year INTEGER,
        is_cancelled INTEGER,
        source_id INTEGER,
        event_time TEXT,
        session_json TEXT NOT NULL,
        PRIMARY KEY (session_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_drivers (
        session_key TEXT NOT NULL,
        driver_number INTEGER NOT NULL,
        position INTEGER,
        acronym TEXT,
        full_name TEXT,
        team_name TEXT,
        team_colour TEXT,
        current_lap INTEGER,
        last_lap_time REAL,
        best_lap_time REAL,
        gap_to_leader TEXT,
        interval TEXT,
        current_compound TEXT,
        tyre_age INTEGER,
        stint_number INTEGER,
        pit_status TEXT,
        track_status TEXT,
        last_speed REAL,
        track_progress REAL,
        last_location_json TEXT NOT NULL,
        sector_times_json TEXT NOT NULL,
        last_update_seq INTEGER NOT NULL,
        PRIMARY KEY (session_key, driver_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_weather_samples (
        session_key TEXT NOT NULL,
        idx INTEGER NOT NULL,
        event_time TEXT,
        air_temperature REAL,
        track_temperature REAL,
        rainfall REAL,
        wind_speed REAL,
        source_id INTEGER,
        source_key TEXT,
        weather_json TEXT NOT NULL,
        PRIMARY KEY (session_key, idx)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_laps (
        session_key TEXT NOT NULL,
        driver_number INTEGER NOT NULL,
        lap INTEGER NOT NULL,
        lap_time REAL NOT NULL,
        PRIMARY KEY (session_key, driver_number, lap)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_stints (
        session_key TEXT NOT NULL,
        driver_number INTEGER NOT NULL,
        stint_number INTEGER NOT NULL,
        compound TEXT NOT NULL,
        start_lap INTEGER NOT NULL,
        end_lap INTEGER,
        tyre_age_start INTEGER NOT NULL,
        PRIMARY KEY (session_key, driver_number, stint_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_pit_stops (
        session_key TEXT NOT NULL,
        idx INTEGER NOT NULL,
        driver_number INTEGER,
        lap INTEGER,
        pit_duration REAL,
        event_time TEXT,
        pit_stop_json TEXT NOT NULL,
        PRIMARY KEY (session_key, idx)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_race_control (
        session_key TEXT NOT NULL,
        idx INTEGER NOT NULL,
        message_json TEXT NOT NULL,
        PRIMARY KEY (session_key, idx)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_overtakes (
        session_key TEXT NOT NULL,
        idx INTEGER NOT NULL,
        lap INTEGER,
        overtaking_driver_number INTEGER,
        overtaken_driver_number INTEGER,
        event_time TEXT,
        overtake_json TEXT NOT NULL,
        PRIMARY KEY (session_key, idx)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_results (
        session_key TEXT NOT NULL,
        driver_number INTEGER NOT NULL,
        position INTEGER,
        number_of_laps INTEGER,
        duration REAL,
        gap_to_leader TEXT,
        dnf INTEGER,
        dns INTEGER,
        dsq INTEGER,
        event_time TEXT,
        result_json TEXT NOT NULL,
        PRIMARY KEY (session_key, driver_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_custom_micro_sectors (
        session_key TEXT NOT NULL,
        driver_number INTEGER NOT NULL,
        lap INTEGER NOT NULL,
        sector_index INTEGER NOT NULL,
        sector_count INTEGER NOT NULL,
        progress_start REAL NOT NULL,
        progress_end REAL NOT NULL,
        passage_time REAL NOT NULL,
        personal_best_delta REAL,
        session_best_delta REAL,
        car_ahead_delta REAL,
        teammate_delta REAL,
        label TEXT NOT NULL,
        source TEXT NOT NULL,
        event_time TEXT,
        seq INTEGER NOT NULL,
        passage_json TEXT NOT NULL,
        PRIMARY KEY (session_key, driver_number, lap, sector_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_predictions (
        session_key TEXT NOT NULL,
        driver_number INTEGER NOT NULL,
        source_event_sequence INTEGER NOT NULL,
        prediction_time TEXT NOT NULL,
        model_version TEXT NOT NULL,
        features_version TEXT NOT NULL,
        prediction_kind TEXT NOT NULL DEFAULT 'race',
        position_semantics TEXT NOT NULL DEFAULT 'race_finish_order',
        forecast_available INTEGER NOT NULL DEFAULT 1,
        unavailable_reason TEXT,
        eligibility_status TEXT NOT NULL DEFAULT 'classification_eligible',
        participation_status TEXT NOT NULL DEFAULT 'running_or_unknown',
        expected_position REAL,
        position_p10 REAL,
        position_p90 REAL,
        win_probability REAL NOT NULL,
        podium_probability REAL NOT NULL,
        points_probability REAL NOT NULL,
        dnf_probability REAL NOT NULL,
        confidence REAL NOT NULL,
        position_distribution_json TEXT NOT NULL,
        strategy_json TEXT,
        PRIMARY KEY (session_key, driver_number, source_event_sequence, model_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS f1_derived_analytics (
        session_key TEXT NOT NULL,
        name TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (session_key, name)
    )
    """,
)
