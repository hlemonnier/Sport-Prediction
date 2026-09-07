"""Evidence contract for learned strategy promotion, separate from simulation fit."""
from __future__ import annotations
import math
from typing import Mapping
from .evaluation_protocol import EvaluationProtocol, event_ordinal, utc_time


def strategy_evidence_issues(evidence: Mapping[str, object] | None, *, candidate_model_id: str,
                             protocol: EvaluationProtocol | None) -> tuple[str, ...]:
    try:
        return _strategy_evidence_issues(evidence, candidate_model_id=candidate_model_id, protocol=protocol)
    except (TypeError, ValueError, AttributeError, OverflowError):
        return ("strategy_evidence_payload_invalid",)


def _strategy_evidence_issues(evidence, *, candidate_model_id, protocol):
    if not isinstance(evidence, Mapping):
        return ("strategy_ope_and_shadow_evidence_missing",)
    issues: list[str] = []
    if evidence.get("schema_version") != "strategy_promotion_evidence_v1":
        issues.append("strategy_evidence_schema_invalid")
    if evidence.get("candidate_model_id") != candidate_model_id:
        issues.append("strategy_evidence_candidate_mismatch")
    if protocol is None or evidence.get("candidate_sha256") != protocol.evaluated_candidate_sha256:
        issues.append("strategy_evidence_candidate_hash_mismatch")
    ope, shadow = evidence.get("ope"), evidence.get("live_shadow")
    if not isinstance(ope, Mapping):
        issues.append("strategy_ope_missing")
    else:
        if ope.get("estimator") not in {"weighted_importance_sampling", "doubly_robust", "fitted_q_evaluation"}:
            issues.append("strategy_ope_estimator_unsupported")
        if ope.get("held_out") is not True or ope.get("behavior_support_verified") is not True:
            issues.append("strategy_ope_holdout_or_behavior_support_unverified")
        if ope.get("uncertainty_unit") != "event":
            issues.append("strategy_ope_uncertainty_must_use_event_blocks")
        try:
            low, high = (float(x) for x in ope["ci95_improvement_vs_baseline"])
            ess = float(ope["effective_sample_size"])
            trajectories = int(ope["trajectory_count"])
            if not all(math.isfinite(v) for v in (low, high, ess)) or not (0 < low <= high):
                issues.append("strategy_ope_improvement_not_supported")
            if not (30 <= ess <= trajectories):
                issues.append("strategy_ope_effective_sample_size_insufficient")
        except (KeyError, TypeError, ValueError, OverflowError):
            issues.append("strategy_ope_uncertainty_or_support_missing")
        if protocol is None or [event_ordinal(k) for k in ope.get("event_keys", ())] != [event_ordinal(k) for k in protocol.audit]:
            issues.append("strategy_ope_audit_population_mismatch")
    if not isinstance(shadow, Mapping):
        issues.append("strategy_live_shadow_missing")
    else:
        keys = [event_ordinal(k) for k in shadow.get("event_keys", ())]
        if any(k is None for k in keys) or len(set(keys)) < 3 or len(keys) != len(set(keys)):
            issues.append("strategy_live_shadow_event_support_insufficient")
        if shadow.get("prospective") is not True or shadow.get("used_for_training_or_selection") is not False:
            issues.append("strategy_live_shadow_not_prospective_held_out")
        try:
            if float(shadow["illegal_action_rate"]) != 0 or int(shadow["decision_count"]) <= 0:
                issues.append("strategy_live_shadow_invalid_actions_or_empty")
        except (KeyError, TypeError, ValueError, OverflowError):
            issues.append("strategy_live_shadow_invalid_actions_or_empty")
        started = utc_time(shadow.get("started_at_utc", ""))
        freeze = utc_time(protocol.frozen_at_utc) if protocol is not None else None
        if started is None or freeze is None or started <= freeze:
            issues.append("strategy_live_shadow_not_after_freeze")
        role_keys = [] if protocol is None else [event_ordinal(key) for key in (*protocol.development, *protocol.selection, *protocol.calibration, *protocol.audit)]
        if keys and (any(k is None for k in keys) or (role_keys and min(keys) <= max(role_keys))):
            issues.append("strategy_live_shadow_not_after_evaluation_partitions")
        times = [utc_time(shadow.get("event_prediction_times_utc", {}).get(key, "")) for key in shadow.get("event_keys", ())]
        if not times or any(time is None for time in times):
            issues.append("strategy_live_shadow_event_timing_missing")
        elif started is None or min(times) < started or times != sorted(times) or any(time.year != key // 100 for key, time in zip(keys, times)):
            issues.append("strategy_live_shadow_event_timing_invalid")
    return tuple(dict.fromkeys(issues))
