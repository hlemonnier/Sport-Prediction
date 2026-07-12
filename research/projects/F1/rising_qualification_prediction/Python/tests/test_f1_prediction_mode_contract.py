from __future__ import annotations

from packages.f1.orchestration.contracts import architecture_payload


EXPECTED_MODE_KEYS = {
    "qualifying_prediction",
    "race_final_position",
    "best_estimated_lap_time",
    "live_race_intelligence",
}


def _modes_by_key() -> dict[str, dict[str, object]]:
    payload = architecture_payload()
    return {str(mode["key"]): mode for mode in payload["modes"]}


def test_architecture_exposes_exactly_four_user_facing_modes() -> None:
    payload = architecture_payload()
    modes = payload["modes"]
    stage_keys = {stage["key"] for stage in payload["stages"]}

    assert len(modes) == 4
    assert {mode["key"] for mode in modes} == EXPECTED_MODE_KEYS
    assert stage_keys == EXPECTED_MODE_KEYS
    assert payload["contract_invariants"]["user_facing_mode_count"] == 4
    assert set(payload["active_flow"]) | set(payload["experimental_branches"]) == EXPECTED_MODE_KEYS


def test_every_mode_declares_target_unit_horizon_baseline_models_metrics_and_maturity() -> None:
    required_nonempty_fields = {
        "output",
        "target",
        "unit",
        "legal_information_horizons",
        "reference_baseline",
        "candidate_model_families",
        "candidate_model_justification",
        "evaluation_metrics",
        "maturity",
    }

    for mode in architecture_payload()["modes"]:
        assert required_nonempty_fields <= set(mode)
        for field in required_nonempty_fields:
            assert mode[field], f"{mode['key']} has an empty {field} contract"


def test_qualifying_classification_is_not_called_a_grid() -> None:
    qualifying = _modes_by_key()["qualifying_prediction"]

    assert "classification" in str(qualifying["output"]).lower()
    assert "classification" in str(qualifying["target"]).lower()
    assert "grid" not in str(qualifying["name"]).lower()
    assert "grid" not in str(qualifying["output"]).lower()
    assert "grid" not in str(qualifying["target"]).lower()


def test_best_lap_keeps_lower_bound_and_achievable_estimate_distinct() -> None:
    best_lap = _modes_by_key()["best_estimated_lap_time"]
    semantics = {item["key"]: item for item in best_lap["internal_semantics"]}

    assert set(semantics) == {
        "theoretical_sector_lower_bound",
        "achievable_session_end_estimate",
    }
    lower_bound = semantics["theoretical_sector_lower_bound"]
    achievable = semantics["achievable_session_end_estimate"]
    assert best_lap["output"] == "p05_p50_p90_achievable_best_qualifying_lap_seconds"
    assert "qualifying" in str(best_lap["target"]).lower()
    assert best_lap["reference_baseline"] == achievable["reference_baseline"]
    assert lower_bound["layer"] == "diagnostic"
    assert lower_bound["output"] == "compatible_sector_lower_bound_seconds"
    assert achievable["output"] == "p05_p50_p90_achievable_best_lap_seconds"
    assert lower_bound["target"] != achievable["target"]
    assert lower_bound["reference_baseline"] != achievable["reference_baseline"]
    assert "interval coverage" not in " ".join(lower_bound["evaluation_metrics"]).lower()
    assert "interval coverage" in " ".join(achievable["evaluation_metrics"]).lower()


def test_live_forecasts_and_constrained_decisions_are_separate_contracts() -> None:
    modes = _modes_by_key()
    live = modes["live_race_intelligence"]
    forecasts = {item["key"]: item for item in live["forecasting_subcontracts"]}
    decisions = {item["key"]: item for item in live["decision_subcontracts"]}

    assert set(forecasts) == {"next_lap", "degradation", "order", "final_status"}
    assert set(decisions) == {"pit", "compound", "pace"}
    assert all(item["layer"] == "forecasting" for item in forecasts.values())
    assert all(item["layer"] == "decision" for item in decisions.values())
    assert all(not item["reinforcement_learning_eligible"] for item in forecasts.values())
    assert all(item["reinforcement_learning_eligible"] for item in decisions.values())
    assert all(item["legal_action_mask_required"] for item in decisions.values())
    assert live["reinforcement_learning_scope"] == "decision_subcontracts_only"
    assert all(
        mode["reinforcement_learning_scope"] == "not_allowed"
        for key, mode in modes.items()
        if key != "live_race_intelligence"
    )

    forecasting_models = " ".join(
        model
        for item in forecasts.values()
        for model in item["candidate_model_families"]
    ).lower()
    assert "reinforcement learning" not in forecasting_models


def test_live_rl_promotion_requires_all_safety_and_evidence_gates() -> None:
    live = _modes_by_key()["live_race_intelligence"]
    requirements = set(live["promotion_requirements"])

    assert {
        "legal_action_masks_enforced_in_policy_and_environment",
        "calibrated_counterfactual_simulator_with_holdout_diagnostics",
        "beats_mpc_and_simple_policy_baselines_on_locked_replay",
        "causal_event_time_replay_with_first_seen_snapshots",
        "locked_off_policy_evaluation_with_support_and_uncertainty_checks",
        "locked_shadow_mode_evidence_before_any_live_recommendation",
    } <= requirements
