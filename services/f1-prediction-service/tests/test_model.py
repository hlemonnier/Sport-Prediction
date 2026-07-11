import pytest

import f1_prediction_service.model as model_module
from f1_prediction_service.model import TARGET_MODELS, predict_from_snapshot


def test_predict_from_snapshot_returns_prediction_contract():
    result = predict_from_snapshot(_snapshot())

    assert result["modelVersion"] == TARGET_MODELS["race"]["model_version"]
    assert result["featuresVersion"] == TARGET_MODELS["race"]["features_version"]
    assert result["diagnostics"]["driverCount"] == 3
    assert len(result["predictions"]) == 3

    first = result["predictions"][0]
    assert first["driver_number"] == 1
    assert first["model_version"] == TARGET_MODELS["race"]["model_version"]
    assert first["features_version"] == TARGET_MODELS["race"]["features_version"]
    assert 1.0 <= first["expected_position"] <= 3.0
    assert 1 <= first["position_p10"] <= first["position_p90"] <= 3
    assert first["win_probability"] > 0.0
    assert first["podium_probability"] >= first["win_probability"]
    assert first["points_probability"] >= first["podium_probability"]
    assert first["strategy"]["recommendedAction"] in {"stay_out", "pit_next_lap", "pit_now"}
    _assert_joint_invariants(result)


def test_twenty_driver_field_has_coherent_joint_marginals():
    result = predict_from_snapshot(_twenty_driver_snapshot())

    assert len(result["predictions"]) == 20
    _assert_joint_invariants(result, tolerance=1e-8)
    diagnostics = result["diagnostics"]["jointDistribution"]
    assert diagnostics["method"] == "sinkhorn_balanced_gaussian_rank_kernel"
    assert diagnostics["rowMaxAbsError"] < 1e-8
    assert diagnostics["columnMaxAbsError"] < 1e-8


def test_targets_use_distinct_models_and_inputs():
    snapshot = _snapshot()
    snapshot["drivers"][0]["best_lap_time"] = 70.0
    snapshot["drivers"][0]["last_lap_time"] = 70.2
    snapshot["drivers"][2]["best_lap_time"] = 67.5
    snapshot["drivers"][2]["last_lap_time"] = 67.7

    race = predict_from_snapshot(snapshot, prediction_kind="race")
    qualifying = predict_from_snapshot(snapshot, prediction_kind="qualifying")
    next_lap = predict_from_snapshot(snapshot, prediction_kind="next-lap")
    strategy = predict_from_snapshot(snapshot, prediction_kind="strategy")

    assert race["predictions"][0]["driver_number"] == 1
    assert qualifying["predictions"][0]["driver_number"] == 16
    assert next_lap["predictions"][0]["driver_number"] == 16
    assert len({race["modelVersion"], qualifying["modelVersion"], next_lap["modelVersion"], strategy["modelVersion"]}) == 4
    assert qualifying["predictions"][0]["strategy"] is None
    assert qualifying["diagnostics"]["strategyPolicyEnabled"] is False
    assert "packages_f1_live_strategy_v1" not in str((race, qualifying, next_lap, strategy))
    for result in (race, qualifying, next_lap, strategy):
        _assert_joint_invariants(result)


def test_provenance_is_honest_about_canonical_package_boundary():
    result = predict_from_snapshot(_snapshot(), prediction_kind="race")
    provenance = result["diagnostics"]["provenance"]

    assert provenance["modelType"] == "deterministic_untrained_snapshot_baseline"
    assert provenance["canonicalLiveRaceModelUsed"] is False
    assert "causal_lap_history_trace" in provenance["canonicalLiveRaceUnavailableReason"]
    if result["diagnostics"]["strategyPolicyEnabled"]:
        assert provenance["canonicalPackageComponents"] == [
            "packages.f1.models.live_race.strategy.BaselineStrategyPolicyAdapter"
        ]


def test_unsupported_prediction_kind_fails_closed():
    with pytest.raises(ValueError, match="unsupported prediction_kind"):
        predict_from_snapshot(_snapshot(), prediction_kind="one-model-for-everything")


def test_duplicate_driver_numbers_fail_closed():
    snapshot = _snapshot()
    snapshot["drivers"][1]["driver_number"] = snapshot["drivers"][0]["driver_number"]

    with pytest.raises(ValueError, match="duplicate driver_number"):
        predict_from_snapshot(snapshot)


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


def test_strategy_adapter_failure_is_reported_and_uses_honest_fallback(monkeypatch):
    def unavailable():
        raise ImportError("canonical package unavailable in isolated service runtime")

    monkeypatch.setattr(model_module, "_load_strategy_policy_adapter", unavailable)

    result = predict_from_snapshot(_snapshot())

    assert result["diagnostics"]["strategyPolicyEnabled"] is False
    assert "ImportError" in result["diagnostics"]["strategyPolicyError"]
    assert result["diagnostics"]["provenance"]["fallbackStrategyPolicyUsed"] is True
    assert result["diagnostics"]["provenance"]["canonicalPackageComponents"] == []
    assert result["predictions"][0]["strategy"]["policyVersion"] == "fallback_strategy_policy_v1"
    _assert_joint_invariants(result)


def _assert_joint_invariants(result, *, tolerance=2e-6):
    predictions = result["predictions"]
    total = len(predictions)
    assert total > 0
    for prediction in predictions:
        distribution = prediction["position_distribution"]
        assert set(distribution) == {str(position) for position in range(1, total + 1)}
        assert abs(sum(distribution.values()) - 1.0) < tolerance
        assert prediction["win_probability"] == pytest.approx(distribution["1"], abs=tolerance)
        assert prediction["podium_probability"] == pytest.approx(
            sum(distribution[str(position)] for position in range(1, min(3, total) + 1)),
            abs=tolerance,
        )
        assert prediction["points_probability"] == pytest.approx(
            sum(distribution[str(position)] for position in range(1, min(10, total) + 1)),
            abs=tolerance,
        )
    for position in range(1, total + 1):
        assert sum(row["position_distribution"][str(position)] for row in predictions) == pytest.approx(
            1.0,
            abs=tolerance,
        )
    assert sum(row["win_probability"] for row in predictions) == pytest.approx(1.0, abs=tolerance)
    assert sum(row["podium_probability"] for row in predictions) == pytest.approx(min(3, total), abs=tolerance)
    assert sum(row["points_probability"] for row in predictions) == pytest.approx(min(10, total), abs=tolerance)


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


def _twenty_driver_snapshot():
    return {
        "sessionKey": "twenty-car-race",
        "seq": 101,
        "raceControl": [{"flag": "GREEN", "message": "Track clear"}],
        "drivers": [
            {
                "driver_number": number,
                "acronym": f"D{number}",
                "position": number,
                "current_lap": 31,
                "last_lap_time": 80.0 + (number * 0.07),
                "best_lap_time": 79.5 + (number * 0.05),
                "current_compound": "MEDIUM" if number % 2 else "HARD",
                "tyre_age": 8 + number,
                "gap_to_leader": str((number - 1) * 1.2),
            }
            for number in range(1, 21)
        ],
    }
