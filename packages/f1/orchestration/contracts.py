"""User-facing F1 prediction-mode and evidence contracts.

The four modes below are product contracts, not a claim that every candidate
model family is already trained or promotion-ready.  Each mode fixes one
target, output unit, legal information horizon, baseline ladder, evaluation
surface and honest maturity.  Nested semantics are used only where one product
mode genuinely contains distinct estimands or decision layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class F1ContractComponent:
    """One estimand or live subcontract within a user-facing mode."""

    key: str
    layer: str
    output: str
    target: str
    unit: str
    legal_information_horizons: tuple[str, ...]
    reference_baseline: str
    candidate_model_families: tuple[str, ...]
    candidate_model_justification: str
    evaluation_metrics: tuple[str, ...]
    maturity: str
    legal_action_mask_required: bool = False
    reinforcement_learning_eligible: bool = False


@dataclass(frozen=True)
class F1PredictionStage:
    """Stable contract for one of the four user-facing F1 prediction modes."""

    key: str
    name: str
    maturity: str
    output: str
    target: str
    unit: str
    legal_information_horizons: tuple[str, ...]
    reference_baseline: str
    candidate_model_families: tuple[str, ...]
    candidate_model_justification: str
    evaluation_metrics: tuple[str, ...]
    inputs: tuple[str, ...]
    internal_semantics: tuple[F1ContractComponent, ...] = ()
    forecasting_subcontracts: tuple[F1ContractComponent, ...] = ()
    decision_subcontracts: tuple[F1ContractComponent, ...] = ()
    promotion_requirements: tuple[str, ...] = ()
    reinforcement_learning_scope: str = "not_allowed"
    next_stage: str | None = None

    @property
    def status(self) -> str:
        return self.maturity

    @property
    def predicts(self) -> str:
        return self.output

    @property
    def output_contract(self) -> tuple[str, str]:
        return (self.output, self.unit)


QUALIFYING_HORIZONS: tuple[str, ...] = (
    "pre_fp_provisional",
    "post_fp1",
    "post_fp2",
    "post_fp3_pre_qualifying",
    "post_sprint_pre_qualifying",
)

RACE_HORIZONS: tuple[str, ...] = (
    "pre_fp_provisional",
    "post_fp_pre_qualifying",
    "post_sprint_pre_qualifying",
    "post_qualifying_pre_final_grid",
    "pre_race_final_grid_as_of",
)

BEST_LAP_HORIZONS: tuple[str, ...] = (
    "pre_fp_provisional",
    "post_fp1",
    "post_fp2",
    "post_fp3_pre_qualifying",
    "post_sprint_pre_qualifying",
)

LIVE_EVENT_TIME_HORIZONS: tuple[str, ...] = (
    "completed_lap_boundary_as_of_event_time",
    "race_control_update_as_first_seen",
    "pit_decision_opportunity_before_action",
)


BEST_LAP_INTERNAL_SEMANTICS: tuple[F1ContractComponent, ...] = (
    F1ContractComponent(
        key="theoretical_sector_lower_bound",
        layer="diagnostic",
        output="compatible_sector_lower_bound_seconds",
        target=(
            "sum of minimum valid representative sector times within one compatible "
            "session, compound and car-state group"
        ),
        unit="seconds",
        legal_information_horizons=BEST_LAP_HORIZONS,
        reference_baseline=(
            "same-season hierarchical sector prior before practice; deterministic "
            "compatible-sector minimum aggregator after a completed eligible session"
        ),
        candidate_model_families=(
            "deterministic compatible-sector minimum aggregator",
            "hierarchical sector-floor shrinkage model",
            "monotone gradient-boosted sector model",
        ),
        candidate_model_justification=(
            "The estimand is additive across sectors and sample-starved by event, so the "
            "baseline preserves physics while hierarchical and monotone models add shrinkage."
        ),
        evaluation_metrics=(
            "lower-bound MAE and RMSE in seconds",
            "lower-bound violation rate",
            "rank correlation within event and session",
        ),
        maturity="diagnostic_only_not_the_user_facing_target",
    ),
    F1ContractComponent(
        key="achievable_session_end_estimate",
        layer="estimand",
        output="p05_p50_p90_achievable_best_lap_seconds",
        target="best valid representative lap actually achieved by the named session end",
        unit="seconds",
        legal_information_horizons=BEST_LAP_HORIZONS,
        reference_baseline=(
            "target-aligned rehearsal lap plus the source-specific median rehearsal-to-"
            "qualifying shift learned from strictly earlier same-season events"
        ),
        candidate_model_families=(
            "quantile gradient boosting",
            "hierarchical Bayesian lap-time model",
            "distance-normalized telemetry TCN after grouped temporal validation",
        ),
        candidate_model_justification=(
            "Achievable best laps have nonlinear tabular effects and conditional uncertainty; "
            "a telemetry sequence model is justified only when grouped data can support it."
        ),
        evaluation_metrics=(
            "p50 MAE and RMSE in seconds",
            "p05 p50 p90 pinball loss",
            "interval coverage and mean interval width",
            "fastest-lap winner and top-three accuracy",
        ),
        maturity="experimental_research_not_promotion_certified",
    ),
)


LIVE_FORECASTING_SUBCONTRACTS: tuple[F1ContractComponent, ...] = (
    F1ContractComponent(
        key="next_lap",
        layer="forecasting",
        output="next_representative_lap_p05_p50_p90",
        target="next completed representative green-condition lap time",
        unit="seconds",
        legal_information_horizons=LIVE_EVENT_TIME_HORIZONS,
        reference_baseline="raw last eligible clean lap with no unvalidated adjustment",
        candidate_model_families=(
            "state-space pace filter",
            "quantile gradient boosting",
            "survival-aware sequence model",
        ),
        candidate_model_justification=(
            "Next-lap pace is a latent, time-varying state with regime-dependent uncertainty "
            "and censoring from pits, flags and incomplete laps."
        ),
        evaluation_metrics=(
            "MAE and RMSE in seconds",
            "pinball loss",
            "interval coverage and width",
            "calibration by track-status regime",
        ),
        maturity="experimental_causal_replay_only",
    ),
    F1ContractComponent(
        key="degradation",
        layer="forecasting",
        output="stint_pace_loss_distribution",
        target="clean-lap within-stint pace slope conditional on compound, tyre age and regime",
        unit="seconds per lap",
        legal_information_horizons=LIVE_EVENT_TIME_HORIZONS,
        reference_baseline="robust within-stint linear slope with partial pooling",
        candidate_model_families=(
            "hierarchical state-space degradation model",
            "mixed-effects robust regression",
            "quantile gradient boosting",
        ),
        candidate_model_justification=(
            "Degradation is a repeated-measures slope with driver, compound and track effects, "
            "so partial pooling and robust nonlinear alternatives are appropriate."
        ),
        evaluation_metrics=(
            "slope MAE in seconds per lap",
            "multi-lap forecast RMSE",
            "interval coverage and width",
            "error by compound and track-status regime",
        ),
        maturity="experimental_causal_replay_only",
    ),
    F1ContractComponent(
        key="order",
        layer="forecasting",
        output="joint_running_order_distribution_at_named_checkpoint",
        target="observed running order at the predeclared future checkpoint",
        unit="ordinal position and normalized probability per position",
        legal_information_horizons=LIVE_EVENT_TIME_HORIZONS,
        reference_baseline="current-order persistence with deterministic pit-loss adjustment",
        candidate_model_families=(
            "Plackett-Luce transition ranker",
            "calibrated event-driven Monte Carlo simulator",
            "survival-aware listwise ranker",
        ),
        candidate_model_justification=(
            "Running order is a joint ranking changed by discrete pit and incident events; "
            "listwise likelihoods and event-driven simulation match that structure."
        ),
        evaluation_metrics=(
            "position MAE",
            "Spearman and Kendall rank correlation",
            "checkpoint log loss and Brier score",
            "joint-order calibration diagnostics",
        ),
        maturity="experimental_causal_replay_only",
    ),
    F1ContractComponent(
        key="final_status",
        layer="forecasting",
        output="classified_finish_dnf_nc_dsq_probability",
        target="official terminal race status after classification is finalized",
        unit="categorical probability",
        legal_information_horizons=LIVE_EVENT_TIME_HORIZONS,
        reference_baseline="same-season reliability hazard conditioned on elapsed race distance",
        candidate_model_families=(
            "competing-risks survival model",
            "calibrated gradient-boosted hazard model",
            "Bayesian reliability model",
        ),
        candidate_model_justification=(
            "Terminal states are censored time-to-event outcomes with competing failure modes, "
            "which calls for hazards and explicit probability calibration."
        ),
        evaluation_metrics=(
            "multiclass log loss",
            "Brier score",
            "expected calibration error",
            "time-dependent concordance",
        ),
        maturity="experimental_causal_replay_only",
    ),
)


LIVE_DECISION_SUBCONTRACTS: tuple[F1ContractComponent, ...] = (
    F1ContractComponent(
        key="pit",
        layer="decision",
        output="pit_or_stay_out_action_at_next_legal_opportunity",
        target="counterfactual expected race utility under each legal pit action",
        unit="discrete constrained action",
        legal_information_horizons=LIVE_EVENT_TIME_HORIZONS,
        reference_baseline="simple pit-window rule and constrained model-predictive control",
        candidate_model_families=(
            "constrained model-predictive control",
            "dynamic programming policy",
            "offline reinforcement learning after all promotion requirements pass",
        ),
        candidate_model_justification=(
            "Pit timing is a sequential counterfactual choice with hard sporting constraints; "
            "planning baselines are mandatory before any offline policy learner."
        ),
        evaluation_metrics=(
            "expected final-position and points uplift",
            "regret versus clairvoyant replay oracle",
            "illegal-action rate",
            "off-policy value with uncertainty bounds",
        ),
        maturity="research_only_no_live_policy_promotion",
        legal_action_mask_required=True,
        reinforcement_learning_eligible=True,
    ),
    F1ContractComponent(
        key="compound",
        layer="decision",
        output="next_legal_tyre_compound_action",
        target="counterfactual expected race utility under each available legal compound",
        unit="categorical constrained action",
        legal_information_horizons=LIVE_EVENT_TIME_HORIZONS,
        reference_baseline="fastest legal deterministic compound rule and constrained MPC",
        candidate_model_families=(
            "constrained model-predictive control",
            "scenario-tree dynamic programming",
            "offline reinforcement learning after all promotion requirements pass",
        ),
        candidate_model_justification=(
            "Compound choice is a constrained scenario decision with uncertain future weather, "
            "traffic and degradation, making explicit planning the reference family."
        ),
        evaluation_metrics=(
            "expected final-position and points uplift",
            "compound-choice regret",
            "illegal-action rate",
            "off-policy value with uncertainty bounds",
        ),
        maturity="research_only_no_live_policy_promotion",
        legal_action_mask_required=True,
        reinforcement_learning_eligible=True,
    ),
    F1ContractComponent(
        key="pace",
        layer="decision",
        output="legal_pace_target_adjustment",
        target="counterfactual expected race utility under feasible pace and conservation bands",
        unit="seconds per lap delta within constrained action bands",
        legal_information_horizons=LIVE_EVENT_TIME_HORIZONS,
        reference_baseline="zero-adjustment policy and constrained MPC",
        candidate_model_families=(
            "constrained model-predictive control",
            "robust dynamic programming",
            "offline reinforcement learning after all promotion requirements pass",
        ),
        candidate_model_justification=(
            "Pace targets trade lap time against tyre and reliability state under bounded "
            "actions, so robust control is the minimum credible comparator."
        ),
        evaluation_metrics=(
            "expected final-position and points uplift",
            "pace-target regret",
            "constraint-violation rate",
            "off-policy value with uncertainty bounds",
        ),
        maturity="research_only_no_live_policy_promotion",
        legal_action_mask_required=True,
        reinforcement_learning_eligible=True,
    ),
)


LIVE_POLICY_PROMOTION_REQUIREMENTS: tuple[str, ...] = (
    "legal_action_masks_enforced_in_policy_and_environment",
    "calibrated_counterfactual_simulator_with_holdout_diagnostics",
    "beats_mpc_and_simple_policy_baselines_on_locked_replay",
    "causal_event_time_replay_with_first_seen_snapshots",
    "locked_off_policy_evaluation_with_support_and_uncertainty_checks",
    "locked_shadow_mode_evidence_before_any_live_recommendation",
)


F1_MODEL_ARCHITECTURE: tuple[F1PredictionStage, ...] = (
    F1PredictionStage(
        key="qualifying_prediction",
        name="Qualifying Prediction",
        maturity="active_research_evidence_limited",
        output="driver-level probability distribution over qualifying classification positions",
        target="official grand-prix qualifying session classification at session completion",
        unit="ordinal position and normalized probability per position",
        legal_information_horizons=QUALIFYING_HORIZONS,
        reference_baseline=(
            "first available causal pace rank in priority order: latest qualifying rehearsal, "
            "event pace, FP qualifying simulation, then FP mean; round-one pre-FP has no "
            "prior-season substitute"
        ),
        candidate_model_families=(
            "Plackett-Luce or Gumbel listwise ranker",
            "gradient-boosted listwise ranker with calibrated marginals",
            "hierarchical Bayesian driver-team pace model",
        ),
        candidate_model_justification=(
            "Qualifying is a full-field ordering problem with sparse same-season evidence, so "
            "listwise likelihoods and hierarchical shrinkage fit the target better than regression."
        ),
        evaluation_metrics=(
            "position MAE",
            "Spearman and Kendall rank correlation",
            "top-three and top-ten accuracy",
            "position log loss and Brier score",
            "expected calibration error",
        ),
        inputs=(
            "completed target-safe practice and Sprint evidence at the named cutoff",
            "same-season driver and team form",
            "quality-filtered qualifying-simulation pace with explicit missingness",
            "circuit and weather priors available at the cutoff",
        ),
        next_stage="race_final_position",
    ),
    F1PredictionStage(
        key="race_final_position",
        name="Race Final Position",
        maturity="active_research_evidence_limited",
        output="joint distribution over official final classified position and terminal status",
        target="official race classification and terminal status after final classification",
        unit="ordinal position, categorical status and normalized probabilities",
        legal_information_horizons=RACE_HORIZONS,
        reference_baseline=(
            "resolved starting-order persistence when available; otherwise qualifying context "
            "or the qualifying-prediction distribution, with no unvalidated mobility adjustment"
        ),
        candidate_model_families=(
            "survival-aware Plackett-Luce race ranker",
            "calibrated event-driven Monte Carlo race simulator",
            "gradient-boosted position and terminal-status ensemble",
        ),
        candidate_model_justification=(
            "Race outcomes couple a ranking with attrition and discrete incidents; survival-aware "
            "ranking and event simulation preserve those dependencies."
        ),
        evaluation_metrics=(
            "position MAE",
            "Spearman and Kendall rank correlation",
            "win top-three top-five and top-ten accuracy",
            "position and status log loss",
            "Brier score and expected calibration error",
            "DNF and non-classification calibration",
        ),
        inputs=(
            "resolved starting order as of the prediction timestamp when available",
            "qualifying-prediction distribution when the session is not yet complete",
            "quality-filtered race-pace and Sprint evidence",
            "same-season reliability, strategy, circuit and weather priors",
        ),
        next_stage="live_race_intelligence",
    ),
    F1PredictionStage(
        key="best_estimated_lap_time",
        name="Best Estimated Lap Time",
        maturity="conditional_research_point_estimate_retained_full_mode_blocked",
        output="p05_p50_p90_achievable_best_qualifying_lap_seconds",
        target="best valid representative lap achieved by the end of grand-prix qualifying",
        unit="seconds",
        legal_information_horizons=BEST_LAP_HORIZONS,
        reference_baseline=(
            "target-aligned rehearsal lap plus the source-specific median rehearsal-to-"
            "qualifying shift learned from strictly earlier same-season events"
        ),
        candidate_model_families=(
            "robust source-specific rehearsal-shift baseline",
            "hierarchical or quantile gradient-boosted lap-time model",
            "distance-normalized telemetry TCN after grouped temporal validation",
        ),
        candidate_model_justification=(
            "The product target is an achievable conditional distribution. The compatible-sector "
            "lower bound remains a separately tagged diagnostic and may not replace this target."
        ),
        evaluation_metrics=(
            "conditional p50 MAE and RMSE in seconds on matched causal rows",
            "end-to-end roster and target-observation coverage",
            "rank correlation and fastest-driver hit rate within event",
            "p05 p50 p90 pinball loss interval coverage and width",
        ),
        inputs=(
            "target-aligned FP3 or Sprint Qualifying rehearsal lap available at the cutoff",
            "strictly earlier same-season source-specific rehearsal-to-qualifying residuals",
            "telemetry only when distance-normalized complete and timestamped before qualifying",
        ),
        internal_semantics=BEST_LAP_INTERNAL_SEMANTICS,
        next_stage=None,
    ),
    F1PredictionStage(
        key="live_race_intelligence",
        name="Live Race Intelligence",
        maturity="experimental_research_shadow_only",
        output="one of seven named outputs, always tagged with subcontract key and layer",
        target="the target declared by the selected forecasting or decision subcontract",
        unit="the selected subcontract unit; cross-unit aggregation is forbidden",
        legal_information_horizons=LIVE_EVENT_TIME_HORIZONS,
        reference_baseline="forecast and decision baselines declared by each subcontract",
        candidate_model_families=(
            "forecasting models for state estimation and calibrated prediction",
            "constrained MPC or dynamic programming for decisions",
            "offline reinforcement learning for decision subcontracts only",
        ),
        candidate_model_justification=(
            "Forecasts estimate observable future state, whereas actions require counterfactual "
            "planning under legal constraints; the two layers cannot share one learning objective."
        ),
        evaluation_metrics=(
            "forecast accuracy and calibration by forecasting subcontract",
            "counterfactual value, regret and constraint safety by decision subcontract",
        ),
        inputs=(
            "first-seen event-time race state up to the declared cutoff",
            "completed laps, tyre state, pit events, weather and race-control status",
            "explicit legal-action mask and counterfactual simulator for decision outputs",
        ),
        forecasting_subcontracts=LIVE_FORECASTING_SUBCONTRACTS,
        decision_subcontracts=LIVE_DECISION_SUBCONTRACTS,
        promotion_requirements=LIVE_POLICY_PROMOTION_REQUIREMENTS,
        reinforcement_learning_scope="decision_subcontracts_only",
        next_stage=None,
    ),
)


def _mode_payload(mode: F1PredictionStage) -> dict[str, Any]:
    payload = asdict(mode)
    # Compatibility aliases for older consumers of ``stages``.  Their values
    # are derived from the canonical contract so they cannot drift.
    payload["status"] = mode.maturity
    payload["predicts"] = mode.output
    payload["output_contract"] = (mode.output, mode.unit)
    return payload


def architecture_payload() -> dict[str, Any]:
    modes = [_mode_payload(mode) for mode in F1_MODEL_ARCHITECTURE]
    return {
        "name": "F1 Prediction System",
        "version": "f1_prediction_architecture_v3_four_mode_contract",
        "modes": modes,
        "stages": modes,
        "active_flow": [
            "qualifying_prediction",
            "race_final_position",
        ],
        "experimental_branches": [
            "best_estimated_lap_time",
            "live_race_intelligence",
        ],
        "legacy_stage_aliases": {
            "pre_quali": "qualifying_prediction",
            "pre_race": "race_final_position",
            "ultimate_lap_time": "best_estimated_lap_time",
            "live_race": "live_race_intelligence",
        },
        "contract_invariants": {
            "user_facing_mode_count": 4,
            "qualifying_is_session_classification_not_starting_grid": True,
            "final_starting_grid_is_race_context_only": True,
            "best_lap_internal_semantic_tag_required": True,
            "live_forecasting_and_decision_layers_are_distinct": True,
            "reinforcement_learning_allowed_only_for_live_decisions": True,
        },
        "point_in_time_contract": {
            "implementation": "packages/f1/domain/weekend.py",
            "named_session_cutoff_required": True,
            "completed_session_classification_required": True,
            "partial_or_future_sessions_eligible": False,
            "format_inapplicable_session_cutoffs_rejected": True,
            "final_grid_is_distinct_from_qualifying_classification": True,
            "mutable_provider_as_of_replay_requires_first_seen_snapshot": True,
        },
        "evaluation_contract": {
            "complete_field_required": True,
            "chronological_walk_forward": True,
            "selection_calibration_final_audit_event_disjoint": True,
            "same_season_2026_primary_arm": True,
            "cross_regime_transfer_is_separate_ablation": True,
            "promotion_requires_reference_baseline_comparison": True,
        },
        "pre_quali_to_race_contract": {
            "enabled": True,
            "description": (
                "Before qualifying is complete, race_final_position integrates the "
                "qualifying_prediction classification distribution as uncertain starting-order "
                "context. It is never relabelled as the official final grid."
            ),
        },
        "weather_scenarios": {
            "enabled": True,
            "outputs": ["base_no_weather", "weather_integrated"],
            "description": (
                "Qualifying and race predictions expose a weather-neutral view and a "
                "weather-integrated view using only priors available at the cutoff."
            ),
        },
    }


__all__ = [
    "BEST_LAP_INTERNAL_SEMANTICS",
    "F1ContractComponent",
    "F1PredictionStage",
    "F1_MODEL_ARCHITECTURE",
    "LIVE_DECISION_SUBCONTRACTS",
    "LIVE_FORECASTING_SUBCONTRACTS",
    "LIVE_POLICY_PROMOTION_REQUIREMENTS",
    "architecture_payload",
]
