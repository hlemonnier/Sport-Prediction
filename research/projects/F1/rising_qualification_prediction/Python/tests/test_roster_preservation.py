from __future__ import annotations

import pandas as pd

from packages.f1.data.providers.base import BaseProvider
from packages.f1.features.assembly import build_current_features, build_training_data


class _RosterProvider(BaseProvider):
    def list_rounds(self, year: int) -> list[dict[str, object]]:
        return [{"round_number": 1, "event_name": "Test Grand Prix"}]

    def get_fp_features(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "driver_id": ["1"],
                "driver_name": ["FP DRIVER"],
                "team_name": ["Team A"],
                "fp_mean_delta": [0.0],
                "fp_weighted_delta": [0.0],
                "fp_mean_rank": [1.0],
                "pace_sessions_available": [1.0],
                "fp_total_laps": [10.0],
            }
        )

    def get_qualifying_results(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "driver_id": ["1", "2"],
                "driver_name": ["QUAL DRIVER 1", "SUBSTITUTE"],
                "team_name": ["Team A", "Team B"],
                "position": [1.0, 2.0],
                "q3_time": [80.0, 80.5],
            }
        )

    def get_race_results(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "driver_id": ["1", "2"],
                "driver_name": ["RACE DRIVER 1", "SUBSTITUTE"],
                "team_name": ["Team A", "Team B"],
                "position": [2.0, 1.0],
                "grid_position": [1.0, 2.0],
                "grid_status": ["grid", "grid"],
            }
        )

    def get_starting_grid(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "driver_id": ["1", "2", "3"],
                "grid_position": [1.0, 2.0, 3.0],
                "grid_status": ["grid", "grid", "grid"],
                "grid_source": ["pre_race_official_grid"] * 3,
            }
        )


class _NoPracticeRosterProvider(_RosterProvider):
    def get_fp_features(self, year: int, round_number: int) -> pd.DataFrame:
        return pd.DataFrame()


def test_training_rosters_keep_target_drivers_without_clean_practice_laps() -> None:
    provider = _RosterProvider()

    qualifying, qualifying_notes = build_training_data(
        provider=provider,
        mode="qualifying",
        train_seasons=[2026],
        target_year=2026,
        target_round=2,
        include_standings=False,
    )
    race, race_notes = build_training_data(
        provider=provider,
        mode="race",
        train_seasons=[2026],
        target_year=2026,
        target_round=2,
        include_standings=False,
    )

    assert set(qualifying["driver_id"]) == {"1", "2"}
    assert set(race["driver_id"]) == {"1", "2"}
    assert pd.isna(qualifying.loc[qualifying["driver_id"].eq("2"), "fp_mean_delta"]).all()
    assert pd.isna(race.loc[race["driver_id"].eq("2"), "fp_mean_delta"]).all()
    assert qualifying_notes == []
    assert race_notes == []


def test_official_grid_can_add_driver_missing_from_practice_and_qualifying() -> None:
    features, notes = build_current_features(
        provider=_RosterProvider(),
        mode="race",
        year=2026,
        round_number=1,
        include_standings=False,
    )

    assert set(features["driver_id"]) == {"1", "2", "3"}
    grid_only = features.loc[features["driver_id"].eq("3")].iloc[0]
    assert float(grid_only["grid_position"]) == 3.0
    assert grid_only["grid_source"] == "pre_race_official_grid"
    assert grid_only["driver_name"] == "3"
    assert pd.isna(grid_only["fp_mean_delta"])
    assert notes == []


def test_postqual_race_roster_does_not_disappear_when_practice_is_empty() -> None:
    features, notes = build_current_features(
        provider=_NoPracticeRosterProvider(),
        mode="race",
        year=2026,
        round_number=1,
        include_standings=False,
    )

    assert set(features["driver_id"]) == {"1", "2", "3"}
    assert features.loc[features["driver_id"].eq("1"), "grid_position"].iloc[0] == 1.0
    assert features.loc[features["driver_id"].eq("3"), "grid_source"].iloc[0] == "pre_race_official_grid"
    assert features["pace_sessions_available"].fillna(0.0).eq(0.0).all()
    assert any("Aucune donnee FP" in note for note in notes)
