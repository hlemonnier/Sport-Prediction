from __future__ import annotations

import importlib

import pandas as pd
import pytest

import packages.f1.data.providers.fastf1 as fastf1_module
from packages.f1.data.providers.fastf1 import FastF1Provider
from packages.f1.data.providers.local_weekends import LocalWeekendProvider
from packages.f1.data.providers.openf1 import OpenF1Provider
from packages.f1.data.providers.base import _eligible_pace_session_indices
from packages.f1.data.providers.practice_features import (
    FP_FEATURE_CONTRACT_VERSION,
    build_session_pace_features,
)
from packages.f1.data.utils import complete_classification_positions, merge_fp_frames


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
    for frame in (fast, opened):
        intent_share = (
            pd.to_numeric(frame["quali_sim_evidence_share"], errors="coerce")
            + pd.to_numeric(frame["race_sim_evidence_share"], errors="coerce")
            + pd.to_numeric(frame["run_intent_unclassified_share"], errors="coerce")
        )
        assert intent_share.sub(1.0).abs().lt(1e-12).all()
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
        {"session_key": 5, "session_name": "Race", "date_start": "2026-01-03T10:00:00Z"},
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


def test_named_cutoffs_resolve_against_standard_and_current_sprint_formats() -> None:
    standard = ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"]
    sprint = ["Practice 1", "Sprint Qualifying", "Sprint", "Qualifying", "Race"]

    fp1, fp1_cutoff, standard_format = _eligible_pace_session_indices(
        year=2026,
        session_names=standard,
        prediction_target="qualifying",
        session_cutoff="post_fp1",
    )
    pre_q, pre_q_cutoff, sprint_format = _eligible_pace_session_indices(
        year=2026,
        session_names=sprint,
        prediction_target="qualifying",
        session_cutoff="pre_qualifying",
        event_format_hint="sprint",
    )

    assert fp1 == [0]
    assert fp1_cutoff == "after_FP1"
    assert standard_format == "standard"
    assert pre_q == [0, 1, 2]
    assert pre_q_cutoff == "before_Q"
    assert sprint_format == "sprint_2024_plus"
    with pytest.raises(ValueError, match="absent"):
        _eligible_pace_session_indices(
            year=2026,
            session_names=sprint,
            prediction_target="qualifying",
            session_cutoff="post_fp2",
            event_format_hint="sprint",
        )


def test_historical_sprint_eras_keep_target_safe_race_pace_sessions() -> None:
    original, original_cutoff, original_format = _eligible_pace_session_indices(
        year=2022,
        session_names=["FP1", "Qualifying", "FP2", "Sprint Qualifying", "Race"],
        prediction_target="race",
        session_cutoff="pre_race",
        event_format_hint="sprint",
    )
    standalone, standalone_cutoff, standalone_format = _eligible_pace_session_indices(
        year=2023,
        session_names=["FP1", "Qualifying", "Sprint Shootout", "Sprint", "Race"],
        prediction_target="race",
        session_cutoff="pre_race",
        event_format_hint="sprint",
    )

    assert original == [0, 2, 3]
    assert original_cutoff == "before_Race"
    assert original_format == "sprint_2021_2022"
    assert standalone == [0, 2, 3]
    assert standalone_cutoff == "before_Race"
    assert standalone_format == "sprint_2023"


def test_local_provider_rejects_partial_or_unprovable_as_of_sessions() -> None:
    complete = {
        "results_rows": 22,
        "results_path": "results.csv",
        "completion_status": "complete",
        "available_at": "2026-07-04T12:00:00Z",
    }
    partial = {**complete, "completion_status": "partial"}

    assert LocalWeekendProvider._session_is_complete(complete)
    assert not LocalWeekendProvider._session_is_complete(partial)
    assert LocalWeekendProvider._session_is_complete(
        complete,
        prediction_as_of="2026-07-04T12:00:01Z",
    )
    assert not LocalWeekendProvider._session_is_complete(
        complete,
        prediction_as_of="2026-07-04T11:59:59Z",
    )
    assert not LocalWeekendProvider._session_is_complete(
        {"results_rows": 22, "results_path": "legacy.csv"},
        prediction_as_of="2026-07-04T12:00:01Z",
    )


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


def test_accuracy_column_retains_roster_but_fails_closed_on_pace() -> None:
    laps = pd.DataFrame(
        {
            "driver_number": [1, 1],
            "lap_number": [1, 2],
            "lap_duration": [90.0, 89.5],
            "is_accurate": [False, False],
        }
    )

    result = build_session_pace_features(laps, "FP1", provider="test")

    assert len(result) == 1
    assert int(result.loc[0, "lap_count"]) == 0
    assert int(result.loc[0, "raw_timed_lap_count"]) == 2
    assert int(result.loc[0, "invalid_lap_count"]) == 2
    assert pd.isna(result.loc[0, "best_lap"])
    assert pd.isna(result.loc[0, "delta"])


def test_practice_quality_contract_excludes_flags_deleted_and_pit_laps() -> None:
    laps = pd.DataFrame(
        {
            "DriverNumber": [1] * 8,
            "Driver": ["VER"] * 8,
            "LapNumber": list(range(1, 9)),
            "LapTime": [90.0, 90.1, 90.2, 90.3, 90.4, 90.5, 90.6, 90.7],
            "Stint": [1] * 8,
            "TyreLife": list(range(8)),
            "IsAccurate": [True, True, True, True, False, True, True, True],
            "Deleted": [False, False, False, True, False, False, False, False],
            "TrackStatus": ["1", "2", "1", "1", "1", "1", "1", "1"],
            "PitOutTime": [None, None, None, None, None, "00:01", None, None],
            "PitInTime": [None, None, None, None, None, None, "00:02", None],
        }
    )

    result = build_session_pace_features(laps, "FP1", provider="test")

    assert len(result) == 1
    row = result.iloc[0]
    assert int(row["raw_timed_lap_count"]) == 8
    assert int(row["representative_lap_count"]) == 3
    assert int(row["neutralised_lap_count"]) == 1
    assert int(row["deleted_lap_count"]) == 1
    assert int(row["invalid_lap_count"]) == 1
    assert int(row["pit_out_lap_count"]) == 1
    assert int(row["pit_in_lap_count"]) == 1
    assert float(row["lap_quality_ratio"]) == pytest.approx(3.0 / 8.0)
    assert int(row["lap_count"]) == 3


def test_run_type_contract_does_not_invent_race_simulation_laps() -> None:
    laps = pd.DataFrame(
        {
            "driver_number": [1, 1, 1],
            "lap_number": [1, 2, 3],
            "lap_duration": [90.0, 90.1, 90.2],
            "stint_number": [1, 1, 1],
            "tyre_life": [0, 1, 2],
            "is_accurate": [True, True, True],
            "track_status": ["1", "1", "1"],
        }
    )

    result = build_session_pace_features(laps, "FP1", provider="test")

    assert int(result.loc[0, "race_sim_lap_count"]) == 0
    assert pd.isna(result.loc[0, "race_sim_lap"])
    assert pd.isna(result.loc[0, "race_sim_delta"])
    assert int(result.loc[0, "quali_sim_lap_count"]) > 0


def test_raw_long_run_degradation_reports_slope_and_uncertainty_without_fuel_claim() -> None:
    laps = pd.DataFrame(
        {
            "driver_number": [1] * 8,
            "lap_number": list(range(1, 9)),
            "lap_duration": [90.0 + (0.2 * lap) for lap in range(8)],
            "stint_number": [1] * 8,
            "tyre_life": list(range(8)),
            "is_accurate": [True] * 8,
            "track_status": ["1"] * 8,
        }
    )

    result = build_session_pace_features(laps, "FP2", provider="test")

    assert int(result.loc[0, "race_sim_lap_count"]) == 4
    assert float(result.loc[0, "race_sim_raw_degradation_sec_per_lap"]) == pytest.approx(0.2)
    assert float(result.loc[0, "race_sim_raw_degradation_mad"]) == pytest.approx(0.0)
    assert int(result.loc[0, "race_sim_degradation_stint_count"]) == 1
    assert bool(result.loc[0, "race_sim_degradation_is_fuel_corrected"]) is False
    assert bool(result.loc[0, "run_intent_labels_calibrated"]) is False


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


def test_latest_completed_session_owns_current_practice_roster() -> None:
    fp1 = pd.DataFrame(
        {
            "driver_id": ["1", "reserve", "3"],
            "driver_name": ["ONE", "RESERVE", "THREE"],
            "team_name": ["TEAM A", "TEAM B", "TEAM B"],
            "session": ["FP1", "FP1", "FP1"],
            "delta": [0.0, 0.5, 0.4],
            "rank": [1.0, 3.0, 2.0],
            "lap_count": [10, 8, 9],
        }
    )
    fp2 = pd.DataFrame(
        {
            "driver_id": ["1", "2", "3"],
            "driver_name": ["ONE", "TWO", "THREE"],
            "team_name": ["TEAM A", "TEAM B", "TEAM B"],
            "session": ["FP2", "FP2", "FP2"],
            "delta": [0.1, 0.3, 0.2],
            "rank": [1.0, 3.0, 2.0],
            "lap_count": [12, 11, 10],
        }
    )

    merged = merge_fp_frames([fp1, fp2]).set_index("driver_id")

    assert set(merged.index) == {"1", "2", "3"}
    assert pd.notna(merged.loc["1", "fp1_delta"])
    assert pd.isna(merged.loc["2", "fp1_delta"])
    assert merged["fp_roster_policy"].eq("latest_session_plus_team_seat_continuity").all()
    assert merged["fp_roster_latest_session_driver_count"].eq(3).all()
    assert merged["fp_roster_active_driver_count"].eq(3).all()
    assert merged["fp_roster_carried_forward_driver_count"].eq(0).all()
    assert merged["fp_roster_superseded_driver_count"].eq(1).all()


def test_roster_carries_latest_missing_car_but_not_superseded_reserve() -> None:
    fp1 = pd.DataFrame(
        {
            "driver_id": ["1", "2", "reserve"],
            "driver_name": ["ONE", "TWO", "RESERVE"],
            "team_name": ["TEAM A", "TEAM A", "TEAM B"],
            "session": ["FP1", "FP1", "FP1"],
            "delta": [0.0, 0.2, 0.4],
            "rank": [1.0, 2.0, 3.0],
            "lap_count": [10, 10, 8],
        }
    )
    fp2 = pd.DataFrame(
        {
            "driver_id": ["1", "3", "4"],
            "driver_name": ["ONE", "THREE", "FOUR"],
            "team_name": ["TEAM A", "TEAM B", "TEAM B"],
            "session": ["FP2", "FP2", "FP2"],
            "delta": [0.1, 0.3, 0.5],
            "rank": [1.0, 2.0, 3.0],
            "lap_count": [12, 11, 9],
        }
    )

    merged = merge_fp_frames([fp1, fp2])

    assert set(merged["driver_id"]) == {"1", "2", "3", "4"}
    assert "reserve" not in set(merged["driver_id"])
    assert merged["fp_roster_carried_forward_driver_count"].eq(1).all()
    assert merged["fp_roster_superseded_driver_count"].eq(1).all()


def test_unclassified_rows_receive_stable_tail_positions() -> None:
    raw = pd.DataFrame(
        {
            "driver_id": ["1", "2", "3", "4"],
            "position": [1.0, 2.0, None, None],
        }
    )

    completed = complete_classification_positions(raw)

    assert completed["position"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert completed["classification_position_imputed_tail"].tolist() == [False, False, True, True]
