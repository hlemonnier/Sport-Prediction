"""Causal, event-block promotion gates for the three non-live F1 modes.

The generic model-promotion helper checks metric presence and direction.  This
module adds the statistical conditions that matter for small F1 event samples:
paired event resampling, leave-one-event-out stability, gain concentration,
strict population coverage, and mode-specific non-regression constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EventError:
    """Paired baseline/challenger error for one complete event block."""

    event_key: str
    baseline_error: float
    candidate_error: float
    stratum: str = "all"


@dataclass(frozen=True)
class PairedEventDiagnostics:
    event_count: int
    baseline_mean: float
    candidate_mean: float
    mean_delta_candidate_minus_baseline: float
    relative_improvement: float
    ci95_delta: tuple[float, float]
    probability_of_improvement: float
    leave_one_event_out_deltas: tuple[float, ...]
    leave_one_event_out_all_improve: bool
    largest_positive_gain_share: float
    largest_gain_event_key: str | None
    stratum_mean_deltas: dict[str, float] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "event_count": int(self.event_count),
            "baseline_mean": float(self.baseline_mean),
            "candidate_mean": float(self.candidate_mean),
            "mean_delta_candidate_minus_baseline": float(self.mean_delta_candidate_minus_baseline),
            "relative_improvement": float(self.relative_improvement),
            "ci95_delta": [float(value) for value in self.ci95_delta],
            "probability_of_improvement": float(self.probability_of_improvement),
            "leave_one_event_out_deltas": [float(value) for value in self.leave_one_event_out_deltas],
            "leave_one_event_out_all_improve": bool(self.leave_one_event_out_all_improve),
            "largest_positive_gain_share": float(self.largest_positive_gain_share),
            "largest_gain_event_key": self.largest_gain_event_key,
            "stratum_mean_deltas": dict(self.stratum_mean_deltas),
        }


@dataclass(frozen=True)
class NonLivePromotionDecision:
    mode: str
    promoted: bool
    reasons: tuple[str, ...]
    diagnostics: PairedEventDiagnostics
    metric_checks: dict[str, bool] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "promoted": bool(self.promoted),
            "status": "promoted" if self.promoted else "rejected",
            "reasons": list(self.reasons),
            "metric_checks": dict(self.metric_checks),
            "diagnostics": self.diagnostics.to_payload(),
        }


def validate_event_partitions(
    *,
    development: Sequence[str],
    selection: Sequence[str],
    calibration: Sequence[str],
    audit: Sequence[str],
) -> tuple[str, ...]:
    """Return fail-closed issues for disjoint, chronological event partitions."""

    named = {
        "development": tuple(str(value) for value in development),
        "selection": tuple(str(value) for value in selection),
        "calibration": tuple(str(value) for value in calibration),
        "audit": tuple(str(value) for value in audit),
    }
    issues: list[str] = []
    for name, values in named.items():
        if not values:
            issues.append(f"{name}_events_missing")
        if len(values) != len(set(values)):
            issues.append(f"{name}_events_duplicated")

    ordered_names = tuple(named)
    for left_index, left_name in enumerate(ordered_names[:-1]):
        left = set(named[left_name])
        for right_name in ordered_names[left_index + 1 :]:
            if left.intersection(named[right_name]):
                issues.append(f"event_partition_overlap:{left_name}:{right_name}")

    numeric = {name: [_event_ordinal(value) for value in values] for name, values in named.items()}
    if all(all(value is not None for value in values) for values in numeric.values()):
        prior_max: int | None = None
        for name in ordered_names:
            ordinals = [int(value) for value in numeric[name] if value is not None]
            if ordinals != sorted(ordinals):
                issues.append(f"{name}_events_not_chronological")
            if prior_max is not None and ordinals and min(ordinals) <= prior_max:
                issues.append(f"event_partition_order_invalid:{name}")
            if ordinals:
                prior_max = max(ordinals)
    return tuple(dict.fromkeys(issues))


def paired_event_diagnostics(
    events: Sequence[EventError],
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 20260713,
) -> PairedEventDiagnostics:
    """Measure a lower-is-better challenger on paired complete event blocks."""

    if len(events) < 2:
        raise ValueError("paired event diagnostics require at least two complete events")
    keys = [str(event.event_key) for event in events]
    if len(keys) != len(set(keys)):
        raise ValueError("event_key must be unique for paired event diagnostics")

    baseline = np.asarray([event.baseline_error for event in events], dtype=float)
    candidate = np.asarray([event.candidate_error for event in events], dtype=float)
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("paired event errors must all be finite")
    if np.any(baseline < 0.0) or np.any(candidate < 0.0):
        raise ValueError("paired event errors must be non-negative")
    if int(bootstrap_samples) < 1_000:
        raise ValueError("bootstrap_samples must be at least 1000")

    delta = candidate - baseline
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, len(delta), size=(int(bootstrap_samples), len(delta)))
    draws = delta[indexes].mean(axis=1)

    leave_one_out = tuple(
        float(np.delete(delta, index).mean())
        for index in range(len(delta))
    )
    positive_gain = np.maximum(baseline - candidate, 0.0)
    gross_gain = float(positive_gain.sum())
    if gross_gain > 0.0:
        largest_index = int(np.argmax(positive_gain))
        largest_share = float(positive_gain[largest_index] / gross_gain)
        largest_key: str | None = keys[largest_index]
    else:
        largest_share = 0.0
        largest_key = None

    strata: dict[str, list[float]] = {}
    for event, value in zip(events, delta):
        strata.setdefault(str(event.stratum), []).append(float(value))
    stratum_means = {key: float(np.mean(values)) for key, values in sorted(strata.items())}
    baseline_mean = float(baseline.mean())
    relative_improvement = (
        float((baseline_mean - float(candidate.mean())) / baseline_mean)
        if baseline_mean > 0.0
        else 0.0
    )
    return PairedEventDiagnostics(
        event_count=len(events),
        baseline_mean=baseline_mean,
        candidate_mean=float(candidate.mean()),
        mean_delta_candidate_minus_baseline=float(delta.mean()),
        relative_improvement=relative_improvement,
        ci95_delta=tuple(float(value) for value in np.quantile(draws, [0.025, 0.975])),
        probability_of_improvement=float(np.mean(draws < 0.0)),
        leave_one_event_out_deltas=leave_one_out,
        leave_one_event_out_all_improve=bool(all(value < 0.0 for value in leave_one_out)),
        largest_positive_gain_share=largest_share,
        largest_gain_event_key=largest_key,
        stratum_mean_deltas=stratum_means,
    )


def evaluate_qualifying_promotion(
    events: Sequence[EventError],
    *,
    baseline_kendall: float,
    candidate_kendall: float,
    pole_non_regression: bool,
    top3_non_regression: bool,
    top10_non_regression: bool,
    tail_excluded_delta: float,
    bootstrap_samples: int = 20_000,
    seed: int = 20260713,
) -> NonLivePromotionDecision:
    diagnostics = paired_event_diagnostics(events, bootstrap_samples=bootstrap_samples, seed=seed)
    checks = {
        "mae_absolute_improvement_at_least_0_15": diagnostics.mean_delta_candidate_minus_baseline <= -0.15,
        "kendall_improvement_at_least_0_02": float(candidate_kendall) - float(baseline_kendall) >= 0.02,
        "pole_non_regression": bool(pole_non_regression),
        "top3_non_regression": bool(top3_non_regression),
        "top10_non_regression": bool(top10_non_regression),
        "all_weekend_strata_improve": bool(diagnostics.stratum_mean_deltas)
        and all(delta < 0.0 for delta in diagnostics.stratum_mean_deltas.values()),
        "tail_excluded_population_improves": float(tail_excluded_delta) < 0.0,
        "bootstrap_upper_bound_below_zero": diagnostics.ci95_delta[1] < 0.0,
        "gain_not_concentrated": diagnostics.largest_positive_gain_share <= 0.50,
        "leave_one_event_out_stable": diagnostics.leave_one_event_out_all_improve,
    }
    return _decision("qualifying", diagnostics, checks)


def evaluate_race_promotion(
    events: Sequence[EventError],
    *,
    baseline_kendall: float,
    candidate_kendall: float,
    baseline_status_brier: float,
    candidate_status_brier: float,
    baseline_status_log_loss: float,
    candidate_status_log_loss: float,
    entrant_coverage: float,
    all_classifications_legal: bool,
    bootstrap_samples: int = 20_000,
    seed: int = 20260713,
) -> NonLivePromotionDecision:
    diagnostics = paired_event_diagnostics(events, bootstrap_samples=bootstrap_samples, seed=seed)
    checks = {
        "mae_relative_improvement_at_least_5pct": diagnostics.relative_improvement >= 0.05,
        "kendall_decline_at_most_0_02": float(candidate_kendall) >= float(baseline_kendall) - 0.02,
        "status_brier_improves": float(candidate_status_brier) < float(baseline_status_brier),
        "status_log_loss_improves": float(candidate_status_log_loss) < float(baseline_status_log_loss),
        "entrant_coverage_complete": math.isclose(float(entrant_coverage), 1.0, abs_tol=1e-12),
        "all_classifications_legal": bool(all_classifications_legal),
        "bootstrap_upper_bound_below_zero": diagnostics.ci95_delta[1] < 0.0,
        "probability_of_improvement_at_least_0_95": diagnostics.probability_of_improvement >= 0.95,
        "gain_not_concentrated": diagnostics.largest_positive_gain_share <= 0.50,
        "leave_one_event_out_stable": diagnostics.leave_one_event_out_all_improve,
    }
    return _decision("race_final_position", diagnostics, checks)


def evaluate_best_lap_promotion(
    events: Sequence[EventError],
    *,
    entrant_output_coverage: float,
    fastest_driver_non_regression: bool,
    top3_non_regression: bool,
    interval_coverage: float,
    nominal_interval_coverage: float,
    baseline_interval_width: float,
    candidate_interval_width: float,
    bootstrap_samples: int = 20_000,
    seed: int = 20260713,
) -> NonLivePromotionDecision:
    diagnostics = paired_event_diagnostics(events, bootstrap_samples=bootstrap_samples, seed=seed)
    checks = {
        "mae_relative_improvement_at_least_5pct": diagnostics.relative_improvement >= 0.05,
        "entrant_output_coverage_complete": math.isclose(float(entrant_output_coverage), 1.0, abs_tol=1e-12),
        "fastest_driver_non_regression": bool(fastest_driver_non_regression),
        "top3_non_regression": bool(top3_non_regression),
        "interval_coverage_within_5pct_points": abs(
            float(interval_coverage) - float(nominal_interval_coverage)
        ) <= 0.05,
        "interval_width_inflation_at_most_10pct": float(candidate_interval_width)
        <= 1.10 * float(baseline_interval_width),
        "bootstrap_upper_bound_below_zero": diagnostics.ci95_delta[1] < 0.0,
        "gain_not_concentrated": diagnostics.largest_positive_gain_share <= 0.50,
        "leave_one_event_out_stable": diagnostics.leave_one_event_out_all_improve,
    }
    return _decision("best_estimated_lap", diagnostics, checks)


def _decision(
    mode: str,
    diagnostics: PairedEventDiagnostics,
    checks: Mapping[str, bool],
) -> NonLivePromotionDecision:
    reasons = tuple(f"gate_failed:{name}" for name, passed in checks.items() if not bool(passed))
    return NonLivePromotionDecision(
        mode=str(mode),
        promoted=not reasons,
        reasons=reasons,
        diagnostics=diagnostics,
        metric_checks=dict(checks),
    )


def _event_ordinal(value: str) -> int | None:
    digits = "".join(character for character in str(value) if character.isdigit())
    if len(digits) < 5:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


__all__ = [
    "EventError",
    "NonLivePromotionDecision",
    "PairedEventDiagnostics",
    "evaluate_best_lap_promotion",
    "evaluate_qualifying_promotion",
    "evaluate_race_promotion",
    "paired_event_diagnostics",
    "validate_event_partitions",
]
