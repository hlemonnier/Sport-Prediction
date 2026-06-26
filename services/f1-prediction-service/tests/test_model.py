import pytest

from f1_prediction_service.model import FEATURES_VERSION, MODEL_VERSION, predict_from_snapshot


def test_predict_from_snapshot_returns_prediction_contract():
    result = predict_from_snapshot(_snapshot())

    assert result["modelVersion"] == MODEL_VERSION
    assert result["featuresVersion"] == FEATURES_VERSION
    assert result["diagnostics"]["driverCount"] == 3
    assert len(result["predictions"]) == 3

    first = result["predictions"][0]
    assert first["driver_number"] == 1
    assert first["model_version"] == MODEL_VERSION
    assert first["features_version"] == FEATURES_VERSION
    assert first["expected_position"] == 1.0
    assert 1 <= first["position_p10"] <= first["position_p90"] <= 3
    assert first["win_probability"] > 0.0
    assert first["podium_probability"] >= first["win_probability"]
    assert first["points_probability"] >= first["podium_probability"]
    assert first["strategy"]["recommendedAction"] in {"stay_out", "pit_next_lap", "pit_now"}
    assert abs(sum(first["position_distribution"].values()) - 1.0) < 1e-5


def test_strategy_pressure_reduces_overaged_driver_ranking():
    snapshot = _snapshot()
    snapshot["drivers"][0]["tyre_age"] = 36
    snapshot["drivers"][0]["current_compound"] = "SOFT"

    result = predict_from_snapshot(snapshot)
    ranks = {item["driver_number"]: item["expected_position"] for item in result["predictions"]}
    strategy = {item["driver_number"]: item["strategy"] for item in result["predictions"]}

    assert ranks[1] > 1.0
    assert strategy[1]["pitUrgency"] is None or strategy[1]["pitUrgency"] > 0.0


def test_package_strategy_adapter_path_when_pandas_available():
    pytest.importorskip("pandas")

    result = predict_from_snapshot(_snapshot())

    assert result["diagnostics"]["strategyPolicyEnabled"] is True
    assert result["diagnostics"]["strategyPolicyError"] is None
    assert result["predictions"][0]["strategy"]["policyVersion"] == "deterministic_baseline_v1"


def _snapshot():
    return {
        "sessionKey": "sample-race",
        "seq": 42,
        "raceControl": [{"flag": "GREEN", "message": "Track clear"}],
        "drivers": [
            {
                "driver_number": 1,
                "acronym": "VER",
                "full_name": "Max Verstappen",
                "team_name": "Red Bull Racing",
                "position": 1,
                "current_lap": 23,
                "last_lap_time": 68.322,
                "best_lap_time": 68.322,
                "current_compound": "MEDIUM",
                "tyre_age": 11,
                "gap_to_leader": "0.000",
                "last_speed": 309,
            },
            {
                "driver_number": 63,
                "acronym": "RUS",
                "full_name": "George Russell",
                "team_name": "Mercedes",
                "position": 2,
                "current_lap": 23,
                "last_lap_time": 68.431,
                "best_lap_time": 68.431,
                "current_compound": "MEDIUM",
                "tyre_age": 11,
                "gap_to_leader": "+1.842",
                "last_speed": 306,
            },
            {
                "driver_number": 16,
                "acronym": "LEC",
                "full_name": "Charles Leclerc",
                "team_name": "Ferrari",
                "position": 3,
                "current_lap": 23,
                "last_lap_time": 68.64,
                "best_lap_time": 68.64,
                "current_compound": "HARD",
                "tyre_age": 11,
                "gap_to_leader": "+2.753",
                "last_speed": 304,
            },
        ],
    }
