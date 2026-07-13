from __future__ import annotations

import math

import pandas as pd
import pytest

from packages.f1.features.qualifying_lap import (
    build_quality_aware_rehearsal_features,
    finite_lap_seconds,
)
from run_best_estimated_lap_2026_backtest import _quality_aware_rehearsal


def test_one_finite_lap_is_not_discarded_by_cleaner_or_aggregation() -> None:
    assert finite_lap_seconds(92.4).tolist() == [92.4]
    features = build_quality_aware_rehearsal_features(
        pd.DataFrame(
            {
                "Driver": ["ONE"],
                "Team": ["Solo"],
                "LapTime": [92.4],
                "Deleted": [False],
                "IsAccurate": [True],
                "Time": [100.0],
            }
        )
    )

    assert features.loc[0, "valid_clean_best_seconds"] == pytest.approx(92.4)
    assert features.loc[0, "quality_aware_anchor_seconds"] == pytest.approx(92.4)
    assert features.loc[0, "valid_clean_lap_count"] == 1
    assert math.isnan(features.loc[0, "best_two_spread_seconds"])


def test_completed_provider_timing_is_official_only_when_explicitly_declared() -> None:
    laps = pd.DataFrame(
        {
            "Driver": ["ONE"],
            "Team": ["Solo"],
            "LapTime": [92.4],
            "Deleted": [False],
            "IsAccurate": [True],
        }
    )

    arbitrary = build_quality_aware_rehearsal_features(laps)
    completed = build_quality_aware_rehearsal_features(
        laps,
        official_session_timing=True,
    )

    assert math.isnan(
        arbitrary.loc[0, "official_classified_rehearsal_best_seconds"]
    )
    assert (
        arbitrary.loc[0, "official_rehearsal_evidence_provenance"]
        == "not_declared_official"
    )
    assert arbitrary.loc[0, "anchor_source"] == "valid_clean_rehearsal"
    assert completed.loc[
        0, "official_classified_rehearsal_best_seconds"
    ] == pytest.approx(92.4)
    assert (
        completed.loc[0, "official_rehearsal_evidence_provenance"]
        == "completed_provider_session_timing"
    )
    assert completed.loc[0, "anchor_source"] == "official_classified_rehearsal"
    assert completed.loc[0, "feature_contract"] == "quality_aware_rehearsal_lap_v2"


def test_best_lap_production_adapter_declares_completed_provider_timing(
    tmp_path,
) -> None:
    laps_path = tmp_path / "event_practice_3_laps.csv"
    results_path = tmp_path / "event_practice_3_results.csv"
    pd.DataFrame(
        {
            "Driver": ["ONE"],
            "Team": ["Solo"],
            "LapTime": [92.4],
            "Deleted": [False],
            "IsAccurate": [True],
        }
    ).to_csv(laps_path, index=False)
    pd.DataFrame(
        {"Abbreviation": ["ONE"], "TeamName": ["Solo"]}
    ).to_csv(results_path, index=False)

    features = _quality_aware_rehearsal(
        laps_path,
        event_key=202601,
        source="practice_3",
        include_earlier_evidence=False,
    )

    assert features.loc[
        0, "official_classified_rehearsal_best_seconds"
    ] == pytest.approx(92.4)
    assert (
        features.loc[0, "official_rehearsal_evidence_provenance"]
        == "completed_provider_session_timing"
    )


def test_deleted_lap_stays_potential_but_can_repair_unrepresentative_valid_anchor() -> None:
    laps = pd.DataFrame(
        [
            {
                "Driver": "ALO",
                "Team": "Aston",
                "LapTime": 101.311,
                "Deleted": False,
                "IsAccurate": True,
                "TrackStatus": "12",
                "Sector1Time": 31.856,
                "Sector2Time": 34.856,
                "Sector3Time": 34.599,
                "Time": 1.0,
            },
            {
                "Driver": "ALO",
                "Team": "Aston",
                "LapTime": 92.490,
                "Deleted": True,
                "IsAccurate": True,
                "TrackStatus": "12",
                "Sector1Time": 31.762,
                "Sector2Time": 35.157,
                "Sector3Time": 25.571,
                "Time": 2.0,
            },
            {
                "Driver": "STR",
                "Team": "Aston",
                "LapTime": 129.082,
                "Deleted": True,
                "IsAccurate": True,
                "TrackStatus": "1",
                "Sector1Time": 40.0,
                "Sector2Time": 45.0,
                "Sector3Time": 44.082,
                "Time": 2.0,
            },
            {
                "Driver": "X",
                "Team": "Other",
                "LapTime": 92.0,
                "Deleted": False,
                "IsAccurate": True,
                "Sector1Time": 30.0,
                "Sector2Time": 35.0,
                "Sector3Time": 27.0,
                "Time": 1.0,
            },
            {
                "Driver": "Y",
                "Team": "Other",
                "LapTime": 93.0,
                "Deleted": False,
                "IsAccurate": True,
                "Sector1Time": 31.0,
                "Sector2Time": 35.0,
                "Sector3Time": 27.0,
                "Time": 1.0,
            },
        ]
    )

    features = build_quality_aware_rehearsal_features(laps).set_index("driver_id")

    assert features.loc["ALO", "valid_clean_best_seconds"] == pytest.approx(101.311)
    assert features.loc["ALO", "deleted_potential_best_seconds"] == pytest.approx(92.490)
    assert features.loc["ALO", "potential_is_credible"]
    assert features.loc["ALO", "quality_aware_anchor_seconds"] == pytest.approx(101.311)
    assert features.loc["ALO", "latent_potential_adjusted_anchor_seconds"] < 93.0
    assert features.loc["ALO", "latent_anchor_uses_potential"]

    assert not features.loc["STR", "potential_is_credible"]
    assert features.loc["STR", "potential_credibility_reason"] == "rejected_field_and_teammate_outlier"
    assert features.loc["STR", "anchor_source"] == "teammate_partial_pool"
    assert features.loc["STR", "latent_potential_adjusted_anchor_seconds"] == pytest.approx(
        features.loc["ALO", "latent_potential_adjusted_anchor_seconds"]
    )
    assert features.loc["STR", "latent_potential_adjusted_anchor_seconds"] < 95.0


def test_fallback_chain_preserves_missing_entrant_and_uses_earlier_then_team_prior() -> None:
    entrants = pd.DataFrame(
        {"driver_id": ["earlier", "prior"], "team_id": ["A", "B"]}
    )
    earlier = pd.DataFrame(
        {
            "driver_id": ["earlier"],
            "team_id": ["A"],
            "session": ["FP2"],
            "lap_time_seconds": [90.2],
            "is_deleted": [False],
            "is_accurate": [True],
        }
    )

    features = build_quality_aware_rehearsal_features(
        pd.DataFrame(),
        entrants=entrants,
        earlier_laps=earlier,
        team_priors={"B": 91.0},
    ).set_index("driver_id")

    assert features.loc["earlier", "quality_aware_anchor_seconds"] == pytest.approx(90.2)
    assert features.loc["earlier", "anchor_source"] == "earlier_session:practice_2"
    assert features.loc["prior", "quality_aware_anchor_seconds"] == pytest.approx(91.0)
    assert features.loc["prior", "anchor_source"] == "team_historical_prior"
    assert features.loc["prior", "anchor_is_imputed"]


def test_as_of_cutoff_excludes_later_faster_lap() -> None:
    features = build_quality_aware_rehearsal_features(
        pd.DataFrame(
            {
                "Driver": ["A", "A"],
                "Team": ["T", "T"],
                "LapTime": [93.0, 90.0],
                "Deleted": [False, False],
                "IsAccurate": [True, True],
                "Time": [100.0, 200.0],
            }
        ),
        as_of=150.0,
    )

    assert features.loc[0, "valid_clean_best_seconds"] == pytest.approx(93.0)
    assert features.loc[0, "feature_as_of"] == "150.0"
