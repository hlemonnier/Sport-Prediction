from __future__ import annotations

import pandas as pd
import pytest

from packages.f1.models.ultimate_lap_time import (
    UltimateLapTimeConfig,
    UltimateLapTimeModel,
    predict_ultimate_lap_time,
    train_ultimate_lap_time,
)


def _sector_laps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": ["VER", "VER", "VER"],
            "event_key": [202601, 202601, 202601],
            "team_name": ["Red Bull", "Red Bull", "Red Bull"],
            "circuit_id": ["bahrain", "bahrain", "bahrain"],
            "lap_time_seconds": [90.5, 90.2, 89.8],
            "sector1_seconds": [30.0, 29.8, 30.2],
            "sector2_seconds": [30.5, 30.0, 29.9],
            "sector3_seconds": [30.0, 30.4, 29.7],
            "compound": ["SOFT", "SOFT", "SOFT"],
            "tyre_age": [0, 1, 2],
            "is_accurate": [True, True, True],
            "is_box_lap": [False, False, False],
            "track_status": ["1", "1", "1"],
        }
    )


def test_training_replaces_placeholder_with_sector_ideal_prediction() -> None:
    model = train_ultimate_lap_time(_sector_laps())

    assert isinstance(model, UltimateLapTimeModel)
    assert model.training_summary.global_anchor_seconds == pytest.approx(89.4)
    assert model.training_summary.sector_columns == (
        "sector1_seconds",
        "sector2_seconds",
        "sector3_seconds",
    )

    prediction = predict_ultimate_lap_time(
        model,
        {
            "driver_id": "VER",
            "event_key": 202601,
            "team_name": "Red Bull",
            "circuit_id": "bahrain",
            "compound": "SOFT",
            "tyre_age": 0,
        },
    )

    assert prediction == pytest.approx(89.4, abs=0.01)

    details = predict_ultimate_lap_time(
        model,
        {
            "driver_id": "VER",
            "event_key": 202601,
            "team_name": "Red Bull",
            "circuit_id": "bahrain",
        },
        return_details=True,
    )
    assert isinstance(details, pd.DataFrame)
    assert details.loc[0, "anchor_source"] == "driver_event"
    assert details.loc[0, "pace_floor_seconds"] < details.loc[0, "ultimate_lap_time_seconds"]
    assert details.loc[0, "pace_ceiling_seconds"] > details.loc[0, "ultimate_lap_time_seconds"]


def test_invalid_training_frame_requires_timing_columns() -> None:
    with pytest.raises(ValueError, match="lap time column or all three sector time columns"):
        train_ultimate_lap_time(pd.DataFrame({"driver_id": ["VER"], "event_key": [202601]}))


def test_pit_and_non_green_laps_do_not_define_ultimate_pace() -> None:
    laps = pd.concat(
        [
            _sector_laps(),
            pd.DataFrame(
                {
                    "driver_id": ["VER", "VER"],
                    "event_key": [202601, 202601],
                    "team_name": ["Red Bull", "Red Bull"],
                    "circuit_id": ["bahrain", "bahrain"],
                    "lap_time_seconds": [60.0, 65.0],
                    "sector1_seconds": [20.0, 21.0],
                    "sector2_seconds": [20.0, 22.0],
                    "sector3_seconds": [20.0, 22.0],
                    "compound": ["SOFT", "SOFT"],
                    "tyre_age": [0, 0],
                    "is_accurate": [True, True],
                    "is_box_lap": [True, False],
                    "track_status": ["1", "2"],
                }
            ),
        ],
        ignore_index=True,
    )

    model = train_ultimate_lap_time(laps)
    prediction = predict_ultimate_lap_time(
        model,
        {"driver_id": "VER", "event_key": 202601, "team_name": "Red Bull", "circuit_id": "bahrain"},
    )

    assert model.training_summary.clean_laps_used == 3
    assert prediction > 88.0


def test_lap_time_only_training_falls_back_to_fastest_clean_lap() -> None:
    laps = pd.DataFrame(
        {
            "driver_id": ["NOR", "NOR", "PIA", "PIA"],
            "event_key": [202601, 202601, 202601, 202601],
            "team_name": ["McLaren", "McLaren", "McLaren", "McLaren"],
            "circuit_id": ["bahrain", "bahrain", "bahrain", "bahrain"],
            "lap_duration": [91.2, 90.9, 91.0, 90.7],
            "is_accurate": [True, True, True, True],
            "is_box_lap": [False, False, False, False],
            "track_status": ["1", "1", "1", "1"],
        }
    )

    model = train_ultimate_lap_time(laps)
    details = predict_ultimate_lap_time(
        model,
        pd.DataFrame(
            [
                {"driver_id": "NOR", "event_key": 202601, "team_name": "McLaren", "circuit_id": "bahrain"},
                {"driver_id": "PIA", "event_key": 202601, "team_name": "McLaren", "circuit_id": "bahrain"},
            ]
        ),
        return_details=True,
    )

    assert model.training_summary.lap_time_column == "lap_duration"
    assert model.training_summary.sector_columns == (None, None, None)
    assert list(details["ultimate_lap_time_seconds"]) == pytest.approx([90.9, 90.7], abs=0.15)

    empty_prediction = predict_ultimate_lap_time(model, pd.DataFrame(columns=["driver_id", "event_key"]))
    assert isinstance(empty_prediction, pd.Series)
    assert empty_prediction.empty


def test_context_penalties_are_trained_and_deterministic() -> None:
    rows: list[dict[str, object]] = []
    for age in range(6):
        soft_lap = 90.0 + (0.04 * age)
        medium_lap = 90.7 + (0.06 * age)
        rows.extend(
            [
                {
                    "driver_id": "LEC",
                    "event_key": 202601,
                    "team_name": "Ferrari",
                    "circuit_id": "bahrain",
                    "lap_time_seconds": soft_lap,
                    "sector1_seconds": soft_lap / 3.0,
                    "sector2_seconds": soft_lap / 3.0,
                    "sector3_seconds": soft_lap / 3.0,
                    "compound": "SOFT",
                    "tyre_age": age,
                    "is_accurate": True,
                    "is_box_lap": False,
                    "track_status": "1",
                },
                {
                    "driver_id": "LEC",
                    "event_key": 202601,
                    "team_name": "Ferrari",
                    "circuit_id": "bahrain",
                    "lap_time_seconds": medium_lap,
                    "sector1_seconds": medium_lap / 3.0,
                    "sector2_seconds": medium_lap / 3.0,
                    "sector3_seconds": medium_lap / 3.0,
                    "compound": "MEDIUM",
                    "tyre_age": age,
                    "is_accurate": True,
                    "is_box_lap": False,
                    "track_status": "1",
                },
            ]
        )

    model = train_ultimate_lap_time(
        pd.DataFrame(rows),
        config=UltimateLapTimeConfig(min_context_observations=3),
    )
    contexts = pd.DataFrame(
        [
            {
                "driver_id": "LEC",
                "event_key": 202601,
                "team_name": "Ferrari",
                "circuit_id": "bahrain",
                "compound": "SOFT",
                "tyre_age": 0,
            },
            {
                "driver_id": "LEC",
                "event_key": 202601,
                "team_name": "Ferrari",
                "circuit_id": "bahrain",
                "compound": "MEDIUM",
                "tyre_age": 5,
            },
        ]
    )

    first = predict_ultimate_lap_time(model, contexts)
    second = predict_ultimate_lap_time(model, contexts)

    assert isinstance(first, pd.Series)
    assert first.equals(second)
    assert first.iloc[1] > first.iloc[0] + 0.4
    assert model.context_coefficients["tyre_age"] > 0.0


def test_entrypoints_no_longer_raise_notimplementederror() -> None:
    try:
        model = train_ultimate_lap_time(_sector_laps())
        predict_ultimate_lap_time(
            model,
            {"driver_id": "VER", "event_key": 202601, "team_name": "Red Bull", "circuit_id": "bahrain"},
        )
    except NotImplementedError as exc:  # pragma: no cover - regression assertion
        pytest.fail(f"ultimate lap-time entrypoint still raises NotImplementedError: {exc}")
