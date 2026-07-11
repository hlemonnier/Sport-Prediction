from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import run_weekend_data_download as downloader


class _SessionStub:
    def __init__(self, *, complete: bool) -> None:
        self.laps = pd.DataFrame(
            {
                "DriverNumber": [1, 1],
                "LapNumber": [1, 2],
                "LapTime": pd.to_timedelta([90.0, 89.5], unit="s"),
                "IsAccurate": [True, True],
                "Deleted": [False, False],
                "TrackStatus": ["1", "1"],
            }
        )
        self.results = (
            pd.DataFrame({"DriverNumber": [1], "Position": [1]})
            if complete
            else pd.DataFrame()
        )
        self.weather_data = pd.DataFrame({"Time": pd.to_timedelta([1.0], unit="s"), "AirTemp": [20.0]})
        self.race_control_messages = pd.DataFrame({"Time": pd.to_timedelta([2.0], unit="s"), "Message": ["CLEAR"]})
        self.load_kwargs: dict[str, object] = {}

    def load(self, **kwargs: object) -> None:
        self.load_kwargs = dict(kwargs)


class _FastF1Stub:
    __version__ = "test-version"

    def __init__(self) -> None:
        self.sessions = {
            "Practice 1": _SessionStub(complete=True),
            "Practice 2": _SessionStub(complete=False),
        }

    @staticmethod
    def get_event(_year: int, _round_number: int) -> pd.Series:
        return pd.Series(
            {
                "EventDate": pd.Timestamp("2026-07-05T00:00:00Z"),
                "Session1": "Practice 1",
                "Session1DateUtc": pd.Timestamp("2026-07-03T10:30:00Z"),
                "Session2": "Practice 2",
                "Session2DateUtc": pd.Timestamp("2026-07-03T14:00:00Z"),
                "Session3": pd.NA,
                "Session4": pd.NA,
                "Session5": pd.NA,
            }
        )

    def get_session(self, _year: int, _round_number: int, session_name: str) -> _SessionStub:
        return self.sessions[session_name]


def test_weekend_download_writes_hashed_point_in_time_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fastf1_stub = _FastF1Stub()
    monkeypatch.setattr(downloader, "fastf1", fastf1_stub)

    result = downloader.download_weekend(
        year=2026,
        round_number=9,
        event_name="British Grand Prix",
        event_format="sprint_2024_plus",
        output_root=tmp_path,
    )

    metadata_path = Path(result["metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == downloader.WEEKEND_METADATA_SCHEMA_VERSION
    assert metadata["source"] == "fastf1"
    assert metadata["source_version"] == "test-version"
    assert metadata["event_format"] == "sprint_2024_plus"
    assert metadata["snapshot_semantics"]["partial_sessions_rejected"] is True
    assert len(metadata["sessions"]) == 1
    session = metadata["sessions"][0]
    assert session["completion_status"] == "completed_provider_classification"
    assert session["completed"] is True
    assert session["available_at"] == session["captured_at"]
    assert metadata["snapshot_started_at"] <= session["captured_at"] <= metadata["generated_at"]
    assert metadata["snapshot_semantics"]["immutable_as_of"] == metadata["generated_at"]
    assert session["weather_rows"] == 1
    assert session["race_control_messages_rows"] == 1
    assert len(session["files"]["laps"]["sha256"]) == 64
    laps_path = Path(session["files"]["laps"]["path"])
    assert laps_path.is_absolute()
    assert hashlib.sha256(laps_path.read_bytes()).hexdigest() == session["files"]["laps"]["sha256"]
    assert any("partial snapshot rejected" in note for note in metadata["notes"])
    assert fastf1_stub.sessions["Practice 1"].load_kwargs == {
        "laps": True,
        "telemetry": False,
        "weather": True,
        "messages": True,
    }
