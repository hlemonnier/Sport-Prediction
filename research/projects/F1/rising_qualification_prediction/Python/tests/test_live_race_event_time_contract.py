from __future__ import annotations

import pandas as pd
import pytest

from packages.f1.models.live_race.predict import (
    _blend_next_lap_point_forecast,
    _prior_observations_for_baseline,
)
from packages.f1.models.live_race.sources import _standardize_laps


def test_missing_clock_time_is_not_relabelled_as_global_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "Driver": ["AAA", "AAA"],
            "LapNumber": [1, 2],
            "LapTime": [90.0, 91.0],
        }
    )

    result = _standardize_laps(raw, event_key=202601, source_used="test", session_name="Race")

    assert result["timestamp"].isna().all()
    assert not result["timestamp_known"].any()
    assert result["timestamp_source"].eq("unavailable").all()


def test_cross_driver_baseline_uses_global_event_time_not_lap_number() -> None:
    observations = pd.DataFrame(
        {
            "driver_id": ["current", "late_lapped", "already_known"],
            "lap_number": [10, 9, 9],
            "timestamp": [100.0, 110.0, 90.0],
            "timestamp_known": [True, True, True],
            "lap_time_seconds": [90.0, 92.0, 91.0],
        }
    )

    prior, mode = _prior_observations_for_baseline(
        observations,
        lap_number=10,
        timestamp=100.0,
        timestamp_known=True,
    )

    assert mode == "global_event_time"
    assert prior["driver_id"].tolist() == ["already_known"]


def test_unknown_event_time_fallback_is_explicitly_non_global() -> None:
    observations = pd.DataFrame(
        {
            "driver_id": ["prior_lap", "same_lap"],
            "lap_number": [4, 5],
            "timestamp": [float("nan"), float("nan")],
            "timestamp_known": [False, False],
        }
    )

    prior, mode = _prior_observations_for_baseline(
        observations,
        lap_number=5,
        timestamp=float("nan"),
        timestamp_known=False,
    )

    assert mode == "lap_number_fallback_not_global_time"
    assert prior["driver_id"].tolist() == ["prior_lap"]


def test_next_lap_blend_is_causal_and_falls_back_when_history_is_missing() -> None:
    assert _blend_next_lap_point_forecast(90.0, 92.0, ssm_weight=0.45) == pytest.approx(91.1)
    assert _blend_next_lap_point_forecast(
        90.0,
        float("nan"),
        ssm_weight=0.45,
    ) == 90.0
