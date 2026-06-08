from __future__ import annotations

from packages.f1.orchestration.model_promotion import (
    evaluate_model_promotion,
    live_strategy_promotion_config,
    ultimate_lap_time_promotion_config,
)


ULTIMATE_LAP_TIME_CANDIDATE_MODEL_ID = "ultimate_lap_time_distance_tcn_v1"
ULTIMATE_LAP_TIME_BASELINE_MODEL_ID = "ultimate_lap_time_deterministic_baseline_v1"
LIVE_STRATEGY_CANDIDATE_MODEL_ID = "live_strategy_conservative_offline_q_v1"
LIVE_STRATEGY_BASELINE_MODEL_ID = "deterministic_baseline_v1"


def _ultimate_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "p50_mae": 0.20,
        "p50_rmse": 0.25,
        "p05_pinball": 0.12,
        "p50_pinball": 0.10,
        "p90_pinball": 0.13,
        "interval_coverage": 0.82,
        "fastest_lap_winner_hit_rate": 0.50,
        "top3_fastest_lap_accuracy": 0.67,
    }
    metrics.update(overrides)
    return metrics


def test_ultimate_lap_promotion_requires_candidate_to_beat_deterministic_baseline() -> None:
    decision = evaluate_model_promotion(
        candidate_model_id=ULTIMATE_LAP_TIME_CANDIDATE_MODEL_ID,
        baseline_model_id=ULTIMATE_LAP_TIME_BASELINE_MODEL_ID,
        candidate_metrics=_ultimate_metrics(p05_pinball=0.09),
        baseline_metrics=_ultimate_metrics(p05_pinball=0.12),
        config=ultimate_lap_time_promotion_config(),
    )

    assert decision.promotion_gate_passed is True
    assert decision.promotion_status == "promoted"
    assert decision.reasons == ()
    assert decision.metric_comparisons[0].metric == "p05_pinball"
    assert decision.metric_comparisons[0].direction == "lower"


def test_promotion_fails_closed_without_baseline_or_required_metrics() -> None:
    metrics = _ultimate_metrics()
    metrics.pop("p50_rmse")

    decision = evaluate_model_promotion(
        candidate_model_id=ULTIMATE_LAP_TIME_CANDIDATE_MODEL_ID,
        baseline_model_id=ULTIMATE_LAP_TIME_BASELINE_MODEL_ID,
        candidate_metrics=metrics,
        baseline_metrics=None,
        config=ultimate_lap_time_promotion_config(),
    )

    assert decision.promotion_gate_passed is False
    assert "candidate_missing_required_metrics" in decision.reasons
    assert "baseline_metrics_missing" in decision.reasons
    assert "no_valid_baseline_metric_comparisons" in decision.reasons
    assert decision.missing_metrics == ("p50_rmse",)


def test_promotion_fails_when_candidate_does_not_beat_baseline() -> None:
    decision = evaluate_model_promotion(
        candidate_model_id=ULTIMATE_LAP_TIME_CANDIDATE_MODEL_ID,
        baseline_model_id=ULTIMATE_LAP_TIME_BASELINE_MODEL_ID,
        candidate_metrics=_ultimate_metrics(p05_pinball=0.14),
        baseline_metrics=_ultimate_metrics(p05_pinball=0.12),
        config=ultimate_lap_time_promotion_config(),
    )

    assert decision.promotion_gate_passed is False
    assert decision.reasons == ("candidate_does_not_beat_baseline:p05_pinball",)


def test_live_strategy_promotion_requires_locked_simulator_validation() -> None:
    config = live_strategy_promotion_config(require_simulator_validation=True)
    candidate = {"policy_value": 12.0, "illegal_action_rate": 0.0, "regret_vs_oracle": 1.0}
    baseline = {"policy_value": 10.0, "illegal_action_rate": 0.05, "regret_vs_oracle": 2.0}

    missing_sim = evaluate_model_promotion(
        candidate_model_id=LIVE_STRATEGY_CANDIDATE_MODEL_ID,
        baseline_model_id=LIVE_STRATEGY_BASELINE_MODEL_ID,
        candidate_metrics=candidate,
        baseline_metrics=baseline,
        config=config,
    )
    validated = evaluate_model_promotion(
        candidate_model_id=LIVE_STRATEGY_CANDIDATE_MODEL_ID,
        baseline_model_id=LIVE_STRATEGY_BASELINE_MODEL_ID,
        candidate_metrics=candidate,
        baseline_metrics=baseline,
        config=config,
        simulator_validation_passed=True,
    )

    assert missing_sim.promotion_gate_passed is False
    assert "simulator_validation_missing_or_failed" in missing_sim.reasons
    assert validated.promotion_gate_passed is True
