from __future__ import annotations

import importlib

import pandas as pd
import pytest

import packages.f1.data.providers.fastf1 as fastf1_module
from packages.f1.data.providers.fastf1 import FastF1Provider
from packages.f1.data.providers.openf1 import OpenF1Provider
from packages.f1.data.providers.practice_features import (
    FP_FEATURE_CONTRACT_VERSION,
    build_session_pace_features,
)
from packages.f1.data.utils import merge_fp_frames


def _lap_times() -> list[float]:
    return [90.0, 90.2, 91.0, 91.1, 91.2, 91.3]


def test_fastf1_and_openf1_shapes_share_identical_practice_semantics() -> None:
    fast_rows = []
    open_rows = []
    stint_rows = []
    for driver, offset, team in [(1, 0.0, "A"), (2, 0.4, "B")]:
        for lap_number, lap_time in enumerate(_lap_times(), start=1):
            fast_rows.append(
                {
                    "DriverNumber": driver,
                    "Driver": f"D{driver}",
                    "Team": team,
                    "LapNumber": lap_number,
                    "LapTime": lap_time + offset,
                    "Stint": 1,
                    "TyreLife": lap_number - 1,
                    "FreshTyre": lap_number == 1,
                    "Compound": "INTERMEDIATE",
                    "IsAccurate": True,
                    "Deleted": False,
                }
            )
            open_rows.append(
                {
                    "driver_number": driver,
                    "driver_name": f"D{driver}",
                    "team_name": team,
                    "lap_number": lap_number,
                    "lap_duration": lap_time + offset,
                    "is_pit_out_lap": False,
                }
            )
        stint_rows.append(
            {
                "driver_number": driver,
                "lap_start": 1,
                "lap_end": 6,
                "stint_number": 1,
                "tyre_age_at_start": 0,
                "compound": "INTERMEDIATE",
            }
        )

    fast = build_session_pace_features(pd.DataFrame(fast_rows), "FP1", provider="fastf1")
    opened = build_session_pace_features(
        pd.DataFrame(open_rows),
        "FP1",
        provider="openf1",
        stints=pd.DataFrame(stint_rows),
    )

    metric_columns = [
        "delta",
        "rank",
        "top3_delta",
        "median_delta",
        "lap_std",
        "lap_count",
        "slow_lap_ratio",
        "quali_sim_delta",
        "quali_sim_rank",
        "quali_sim_lap_count",
        "race_sim_delta",
        "race_sim_rank",
        "race_sim_lap_count",
        "wet_sim_delta",
        "wet_sim_rank",
        "wet_sim_lap_count",
        "quali_vs_race_gap",
    ]
    fast = fast.sort_values("driver_id").reset_index(drop=True)
    opened = opened.sort_values("driver_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(fast[metric_columns], opened[metric_columns], check_dtype=False)
    assert set(fast["feature_contract_version"]) == {FP_FEATURE_CONTRACT_VERSION}
    assert set(opened["feature_contract_version"]) == {FP_FEATURE_CONTRACT_VERSION}
    merged = merge_fp_frames([opened])
    assert merged["fp_wet_sim_delta"].notna().all()
    assert merged["fp_wet_sim_laps"].eq(6.0).all()


def test_openf1_prequal_session_selection_excludes_postqual_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenF1Provider(cache_dir=None)
    sessions = [
        {"session_key": 1, "session_name": "Practice 1", "date_start": "2026-01-01T10:00:00Z"},
        {"session_key": 2, "session_name": "Sprint Qualifying", "date_start": "2026-01-01T14:00:00Z"},
        {"session_key": 3, "session_name": "Sprint", "date_start": "2026-01-02T10:00:00Z"},
        {"session_key": 4, "session_name": "Qualifying", "date_start": "2026-01-02T14:00:00Z"},
        {"session_key": 5, "session_name": "Practice after qualifying", "date_start": "2026-01-03T10:00:00Z"},
    ]
    monkeypatch.setattr(provider, "_get_json", lambda endpoint, params: sessions)

    assert provider._pre_qualifying_sessions(99) == [(1, "FP1"), (2, "SQ"), (3, "Sprint")]


def test_fastf1_prequal_session_selection_uses_schedule_order(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("packages.f1.data.providers.fastf1")

    class FastF1Stub:
        @staticmethod
        def get_event_schedule(year: int) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "RoundNumber": 3,
                        "Session1": "Practice 1",
                        "Session2": "Sprint Qualifying",
                        "Session3": "Sprint",
                        "Session4": "Qualifying",
                        "Session5": "Race",
                    }
                ]
            )

    monkeypatch.setattr(module, "fastf1", FastF1Stub())
    provider = FastF1Provider.__new__(FastF1Provider)

    assert provider._pre_qualifying_sessions(2026, 3) == [
        ("Practice 1", "FP1"),
        ("Sprint Qualifying", "SQ"),
        ("Sprint", "Sprint"),
    ]


def test_fastf1_practice_and_result_surfaces_join_on_numeric_driver_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    laps = pd.DataFrame(
        {
            "DriverNumber": [1, 1, 63, 63],
            "Driver": ["VER", "VER", "RUS", "RUS"],
            "LapNumber": [1, 2, 1, 2],
            "LapTime": pd.to_timedelta([90.0, 89.0, 91.0, 90.0], unit="s"),
            "IsAccurate": [True, True, True, True],
        }
    )
    results = pd.DataFrame(
        {
            "DriverNumber": [1, 63],
            "Abbreviation": ["VER", "RUS"],
            "Position": [1.0, 2.0],
            "GridPosition": [1.0, 2.0],
        }
    )

    class Session:
        def __init__(self, *, laps_frame: pd.DataFrame | None = None) -> None:
            self.laps = laps_frame if laps_frame is not None else pd.DataFrame()
            self.results = results

        def load(self) -> None:
            return None

    class FastF1Stub:
        @staticmethod
        def get_session(_year: int, _round: int, name: str) -> Session:
            return Session(laps_frame=laps if name == "FP1" else None)

    monkeypatch.setattr(fastf1_module, "fastf1", FastF1Stub())
    provider = FastF1Provider.__new__(FastF1Provider)

    practice = provider._session_pace_features(2026, 1, "FP1")
    qualifying = provider.get_qualifying_results(2026, 1)
    race = provider.get_race_results(2026, 1)

    assert set(practice["driver_id"]) == {"1", "63"}
    assert set(qualifying["driver_id"]) == {"1", "63"}
    assert set(race["driver_id"]) == {"1", "63"}
    assert len(practice.merge(qualifying, on="driver_id", how="inner")) == 2
    assert dict(zip(qualifying["driver_id"], qualifying["driver_name"])) == {"1": "VER", "63": "RUS"}


def test_accuracy_column_fails_closed_when_no_lap_is_accurate() -> None:
    laps = pd.DataFrame(
        {
            "driver_number": [1, 1],
            "lap_number": [1, 2],
            "lap_duration": [90.0, 89.5],
            "is_accurate": [False, False],
        }
    )

    assert build_session_pace_features(laps, "FP1", provider="test").empty


def test_total_laps_excludes_overlapping_simulation_subsets() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["1"],
            "driver_name": ["VER"],
            "session": ["FP1"],
            "delta": [0.0],
            "rank": [1],
            "lap_count": [10],
            "quali_sim_lap_count": [2],
            "race_sim_lap_count": [5],
            "wet_sim_lap_count": [3],
        }
    )

    merged = merge_fp_frames([frame])

    assert float(merged.loc[0, "fp_total_laps"]) == 10.0
    assert float(merged.loc[0, "fp_quali_sim_laps"]) == 2.0
    assert float(merged.loc[0, "fp_race_sim_laps"]) == 5.0
    assert float(merged.loc[0, "fp_wet_sim_laps"]) == 3.0
