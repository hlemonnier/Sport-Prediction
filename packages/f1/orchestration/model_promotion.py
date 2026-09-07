"""Fail-closed promotion gates for F1 advanced models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from .evaluation_protocol import EvaluationProtocol
from .non_live_validation import EventError, paired_event_diagnostics
from .strategy_evidence import strategy_evidence_issues


LOWER_IS_BETTER_METRICS: tuple[str, ...] = (
    "p50_mae",
    "p50_rmse",
    "p05_pinball",
    "p50_pinball",
    "p90_pinball",
    "illegal_action_rate",
    "regret_vs_oracle",
    "pit_loss_mae",
    "one_step_lap_time_mae",
    "downside_cvar_seconds",
)
HIGHER_IS_BETTER_METRICS: tuple[str, ...] = (
    "interval_coverage",
    "fastest_lap_winner_hit_rate",
    "top3_fastest_lap_accuracy",
    "policy_value",
    "mean_return",
    "simulator_value",
)


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    candidate_value: float
    baseline_value: float
    direction: str
    delta: float
    passed: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "candidate_value": float(self.candidate_value),
            "baseline_value": float(self.baseline_value),
            "direction": self.direction,
            "delta_candidate_minus_baseline": float(self.delta),
            "passed": bool(self.passed),
        }


@dataclass(frozen=True)
class PromotionGateConfig:
    """Promotion controls shared by F1 advanced model families."""

    required_metrics: tuple[str, ...]
    baseline_comparison_metrics: tuple[str, ...]
    lower_is_better_metrics: tuple[str, ...] = LOWER_IS_BETTER_METRICS
    higher_is_better_metrics: tuple[str, ...] = HIGHER_IS_BETTER_METRICS
    require_baseline_comparison: bool = True
    fail_closed_on_missing_metrics: bool = True
    fail_closed_on_leakage_issues: bool = True
    require_simulator_validation: bool = False
    min_relative_improvement: float = 0.0
    paired_error_metric: str = "p50_mae"
    require_strategy_evidence: bool = False
    require_interval_evidence: bool = False


@dataclass(frozen=True)
class PromotionDecision:
    """Stable payload for model promotion audit reports."""

    candidate_model_id: str
    baseline_model_id: str
    promotion_gate_passed: bool
    promotion_status: str
    reasons: tuple[str, ...]
    missing_metrics: tuple[str, ...]
    metric_comparisons: tuple[MetricComparison, ...] = field(default_factory=tuple)
    deterministic_fallback_model_id: Optional[str] = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "candidate_model_id": self.candidate_model_id,
            "baseline_model_id": self.baseline_model_id,
            "promotion_gate_passed": bool(self.promotion_gate_passed),
            "promotion_status": self.promotion_status,
            "reasons": list(self.reasons),
            "missing_metrics": list(self.missing_metrics),
            "metric_comparisons": [item.to_payload() for item in self.metric_comparisons],
            "deterministic_fallback_model_id": self.deterministic_fallback_model_id,
            "diagnostics": self.diagnostics,
        }


def ultimate_lap_time_promotion_config() -> PromotionGateConfig:
    return PromotionGateConfig(
        required_metrics=(
            "p50_mae",
            "p50_rmse",
            "p05_pinball",
            "p50_pinball",
            "p90_pinball",
            "interval_coverage",
            "interval_width",
            "fastest_lap_winner_hit_rate",
            "top3_fastest_lap_accuracy",
        ),
        baseline_comparison_metrics=("p50_mae", "p50_rmse", "p50_pinball"),
        fail_closed_on_leakage_issues=True,
        min_relative_improvement=0.01,
        require_interval_evidence=True,
    )


def live_strategy_promotion_config(*, require_simulator_validation: bool = True) -> PromotionGateConfig:
    return PromotionGateConfig(
        required_metrics=(
            "policy_value",
            "illegal_action_rate",
            "regret_vs_oracle",
        ),
        baseline_comparison_metrics=("policy_value", "illegal_action_rate", "regret_vs_oracle"),
        require_simulator_validation=bool(require_simulator_validation),
        paired_error_metric="regret_vs_oracle",
        require_strategy_evidence=True,
    )


def evaluate_model_promotion(
    *,
    candidate_model_id: str,
    baseline_model_id: str,
    candidate_metrics: Mapping[str, object],
    baseline_metrics: Optional[Mapping[str, object]],
    config: PromotionGateConfig,
    leakage_issues: Sequence[str] = (),
    simulator_validation_passed: Optional[bool] = None,
    deterministic_fallback_model_id: Optional[str] = None,
    evaluation_protocol: EvaluationProtocol | None = None,
    paired_events: Sequence[EventError] = (),
    strategy_evidence: Mapping[str, object] | None = None,
) -> PromotionDecision:
    """Evaluate whether an F1 model can be promoted over its baseline.

    The gate is intentionally conservative: missing metrics, missing baseline
    comparison, leakage issues, or missing simulator validation all fail closed.
    """

    reasons: list[str] = []
    paired_payload: dict[str, object] = {}
    if evaluation_protocol is None:
        reasons.append("evaluation_protocol_missing")
    else:
        reasons.extend(evaluation_protocol.issues([event.event_key for event in paired_events], strata=[event.stratum for event in paired_events]))
    try:
        paired = paired_event_diagnostics(paired_events)
        paired_payload = paired.to_payload()
        if paired.ci95_delta[1] >= 0 or not paired.leave_one_event_out_all_improve or paired.largest_positive_gain_share > 0.5:
            reasons.append("paired_event_improvement_not_stable")
        if not all(paired.stratum_mean_deltas.get(label, 0) < 0 for label in ("standard", "sprint")):
            reasons.append("paired_event_weekend_stratum_does_not_improve")
        for label, measured, metrics in (("candidate", paired.candidate_mean, candidate_metrics), ("baseline", paired.baseline_mean, baseline_metrics or {})):
            supplied = _numeric_or_none(metrics.get(config.paired_error_metric))
            if supplied is None or not np.isclose(measured, supplied, rtol=1e-8, atol=1e-10):
                reasons.append(f"paired_event_{label}_metric_mismatch:{config.paired_error_metric}")
    except ValueError:
        reasons.append("paired_event_evidence_missing_or_invalid")
    if config.require_strategy_evidence:
        reasons.extend(strategy_evidence_issues(strategy_evidence, candidate_model_id=candidate_model_id, protocol=evaluation_protocol))
    if config.require_interval_evidence:
        coverage = _numeric_or_none(candidate_metrics.get("interval_coverage"))
        width = _numeric_or_none(candidate_metrics.get("interval_width"))
        baseline_width = _numeric_or_none((baseline_metrics or {}).get("interval_width"))
        if coverage is None or abs(coverage - 0.85) > 0.05:
            reasons.append("p05_p90_interval_coverage_outside_tolerance")
        if width is None or baseline_width is None or not (0 < width <= 1.10 * baseline_width):
            reasons.append("interval_width_missing_or_inflated")
        for metric in ("p05_pinball", "p90_pinball"):
            c, b = _numeric_or_none(candidate_metrics.get(metric)), _numeric_or_none((baseline_metrics or {}).get(metric))
            if c is None or b is None or c > b:
                reasons.append(f"candidate_tail_score_regression:{metric}")
    for metric, value in candidate_metrics.items():
        numeric = _numeric_or_none(value)
        if numeric is None:
            continue
        bounded = metric.endswith(("_rate", "_accuracy")) or metric == "interval_coverage"
        nonnegative = metric.endswith(("_mae", "_rmse", "_pinball")) or metric in {"regret_vs_oracle", "interval_width"}
        if (bounded and not 0 <= numeric <= 1) or (nonnegative and numeric < 0):
            reasons.append(f"candidate_metric_out_of_domain:{metric}")
    missing = _missing_required_metrics(candidate_metrics, config.required_metrics)
    if missing and config.fail_closed_on_missing_metrics:
        reasons.append("candidate_missing_required_metrics")

    if config.fail_closed_on_leakage_issues and leakage_issues:
        reasons.append("candidate_has_leakage_issues")

    baseline_payload = dict(baseline_metrics or {})
    if config.require_baseline_comparison and not baseline_payload:
        reasons.append("baseline_metrics_missing")

    if config.require_simulator_validation and simulator_validation_passed is not True:
        reasons.append("simulator_validation_missing_or_failed")

    comparisons: list[MetricComparison] = []
    if baseline_payload:
        for metric in config.baseline_comparison_metrics:
            comparison = _compare_metric(
                metric,
                candidate_metrics.get(metric),
                baseline_payload.get(metric),
                config=config,
            )
            if comparison is None:
                reasons.append(f"baseline_comparison_unavailable:{metric}")
                continue
            comparisons.append(comparison)
            if not comparison.passed:
                reasons.append(f"candidate_does_not_beat_baseline:{metric}")

    if config.require_baseline_comparison and not comparisons:
        reasons.append("no_valid_baseline_metric_comparisons")

    unique_reasons = tuple(dict.fromkeys(reasons))
    passed = bool(not unique_reasons)
    return PromotionDecision(
        candidate_model_id=str(candidate_model_id),
        baseline_model_id=str(baseline_model_id),
        promotion_gate_passed=passed,
        promotion_status="promoted" if passed else "rejected",
        reasons=unique_reasons,
        missing_metrics=missing,
        metric_comparisons=tuple(comparisons),
        deterministic_fallback_model_id=deterministic_fallback_model_id or baseline_model_id,
        diagnostics={
            "fail_closed_on_missing_metrics": bool(config.fail_closed_on_missing_metrics),
            "fail_closed_on_leakage_issues": bool(config.fail_closed_on_leakage_issues),
            "require_baseline_comparison": bool(config.require_baseline_comparison),
            "require_simulator_validation": bool(config.require_simulator_validation),
            "simulator_validation_passed": simulator_validation_passed,
            "leakage_issues": list(leakage_issues),
            "evaluation_protocol": evaluation_protocol.to_payload() if evaluation_protocol else None,
            "paired_event_evidence": paired_payload,
            "strategy_evidence_required": config.require_strategy_evidence,
        },
    )


def _missing_required_metrics(metrics: Mapping[str, object], required: Sequence[str]) -> tuple[str, ...]:
    missing: list[str] = []
    for metric in required:
        value = metrics.get(metric)
        if _numeric_or_none(value) is None:
            missing.append(str(metric))
    return tuple(missing)


def _compare_metric(
    metric: str,
    candidate_value: object,
    baseline_value: object,
    *,
    config: PromotionGateConfig,
) -> Optional[MetricComparison]:
    candidate = _numeric_or_none(candidate_value)
    baseline = _numeric_or_none(baseline_value)
    if candidate is None or baseline is None:
        return None

    direction = _metric_direction(metric, config)
    if direction is None:
        return None
    delta = float(candidate - baseline)
    tolerance = abs(float(baseline)) * float(max(0.0, config.min_relative_improvement))
    if metric == "illegal_action_rate":
        passed = bool(candidate == 0.0 and candidate <= baseline)
    elif direction == "lower":
        passed = bool(candidate < baseline - tolerance)
    else:
        passed = bool(candidate > baseline + tolerance)
    return MetricComparison(
        metric=str(metric),
        candidate_value=float(candidate),
        baseline_value=float(baseline),
        direction=direction,
        delta=float(delta),
        passed=passed,
    )


def _metric_direction(metric: str, config: PromotionGateConfig) -> Optional[str]:
    if metric in set(config.lower_is_better_metrics):
        return "lower"
    if metric in set(config.higher_is_better_metrics):
        return "higher"
    return None


def _numeric_or_none(value: object) -> Optional[float]:
    try:
        numeric = float(value)  # type: ignore[arg-type]
        if np.isfinite(numeric):
            return numeric
    except Exception:
        return None
    return None


__all__ = [
    "HIGHER_IS_BETTER_METRICS",
    "LOWER_IS_BETTER_METRICS",
    "MetricComparison",
    "PromotionDecision",
    "PromotionGateConfig",
    "evaluate_model_promotion",
    "live_strategy_promotion_config",
    "ultimate_lap_time_promotion_config",
]
