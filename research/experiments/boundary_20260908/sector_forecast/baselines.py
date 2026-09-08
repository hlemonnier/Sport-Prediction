"""Fixed same-checkpoint sector references and one Gaussian research prototype.

The caller supplies chronological, already available, raw-valid own-driver
completion records. This module never consults final eligibility, future labels,
canonical lap numbers, tyre-age guesses or other drivers. Its current-lap branch
is a heuristic for the next recorded eligible lap when future cleanliness is
unknown, not a certificate that the current lap will become that target.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real

import numpy as np

REFERENCE_NAMES = (
    "sector_last_template",
    "sector_joint_median5",
    "sector_weighted_median10",
    "sector_pace_scaled_median5",
)
PACE_REFERENCE = REFERENCE_NAMES[3]
GAUSSIAN_NAME = "sector_gaussian_conditional10"
MINIMUM_HISTORY = 3
RECENCY_ALPHA = .2
PACE_RATIO_BOUNDS = (.97, 1.03)
COHERENCE_TOLERANCE_SECONDS = .003 + 1e-9
GAUSSIAN_DIAGONAL_SHRINKAGE = .5
GAUSSIAN_DIAGONAL_FLOOR_SECONDS_SQUARED = .01


def _positive_vector(value, size, name):
    if not isinstance(value, (Sequence, np.ndarray)) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if len(value) != size:
        raise ValueError(f"{name} requires exactly {size} values")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in value):
        raise ValueError(f"{name} requires numeric seconds, without boolean coercion")
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all() or not (array > 0).all():
        raise ValueError(f"{name} must be finite and positive")
    return array.copy()


def _templates(history):
    history = list(history)
    if len(history) < MINIMUM_HISTORY:
        raise ValueError("At least three already available valid own-driver completions are required")
    sectors, full_laps = [], []
    for record in history:
        if not isinstance(record, Mapping):
            raise ValueError("History records must contain sectors_seconds and full_lap_seconds")
        vector = _positive_vector(record["sectors_seconds"], 3, "historical sectors")
        lap = _positive_vector([record["full_lap_seconds"]], 1, "historical full lap")[0]
        total = float(vector.sum())
        if not np.isfinite(total) or abs(total-lap) > COHERENCE_TOLERANCE_SECONDS:
            raise ValueError("Completed sectors and full lap fail the frozen millisecond coherence check")
        sectors.append(vector)
        full_laps.append(lap)
    return np.asarray(sectors), np.asarray(full_laps)


def weighted_median(values, weights):
    """Lower inverse-CDF weighted median; an absolute-loss minimizer.

    The median is taken over complete scalar remainders, never separately over
    sector coordinates. Weights may be zero but must have positive total mass.
    """
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if (values.ndim != 1 or not len(values) or values.shape != weights.shape
            or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights < 0).any() or not (weights > 0).any()):
        raise ValueError("Weighted median requires finite values and nonnegative positive-total weights")
    weights = weights/weights.max()  # Preserve scale invariance without overflow.
    indexes = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[indexes] / weights.sum())
    index = int(np.searchsorted(cumulative, .5, side="left"))
    return float(values[indexes[index]])


def _gaussian(history_sectors, history_laps, prefix, fallback):
    """Conditional remainder mean under an explicitly assumed joint Gaussian.

    Joint coordinates are observed S1[, S2] and R=full_lap-observed_prefix.
    Shrinking covariance of full_lap instead would also shrink the coefficient
    of the already known prefix; computing R first preserves that coefficient.
    """
    n, k = min(len(history_laps), 10), len(prefix)
    sectors, laps = history_sectors[-n:], history_laps[-n:]
    remainder = laps-sectors[:, :k].sum(1)
    samples = np.column_stack((sectors[:, :k], remainder))
    diagnostics = {"status": "available", "history_count": n,
                   "conditional_remainder_before_guard_seconds": None,
                   "statistic": "conditional_mean_equals_median_only_under_assumed_gaussian"}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            center = samples.mean(0)
            centered = samples-center
            covariance = centered.T@centered/(n-1)
            covariance = ((1-GAUSSIAN_DIAGONAL_SHRINKAGE)*covariance
                          + GAUSSIAN_DIAGONAL_SHRINKAGE*np.diag(np.diag(covariance))
                          + GAUSSIAN_DIAGONAL_FLOOR_SECONDS_SQUARED*np.eye(k+1))
            if not np.isfinite(covariance).all():
                raise FloatingPointError("Nonfinite conditional covariance")
            adjustment = covariance[-1, :k]@np.linalg.solve(covariance[:k, :k], prefix-center[:k])
            remaining = float(center[-1]+adjustment)
            point = float(prefix.sum()+remaining)
        diagnostics["conditional_remainder_before_guard_seconds"] = remaining if np.isfinite(remaining) else None
        if not np.isfinite(remaining) or not np.isfinite(point):
            diagnostics["status"] = "fallback_nonfinite_conditional_point"
        elif remaining <= 0:
            diagnostics["status"] = "fallback_nonpositive_conditional_remainder"
        else:
            return point, diagnostics
    except np.linalg.LinAlgError:
        diagnostics["status"] = "fallback_linear_solve_failed"
    except FloatingPointError:
        diagnostics["status"] = "fallback_nonfinite_moments"
    return fallback, diagnostics


def predict_baselines(history, prefix_seconds, *, known_asof_contamination):
    """Return all four fixed references and the frozen Gaussian candidate.

    ``history`` is oldest first, restricted by the caller to
    observed_valid_completed is True and available_ms < checkpoint_ms.
    At least three such own-driver records are required before any model issues.
    Each record supplies sectors_seconds=[S1,S2,S3] and full_lap_seconds.
    ``prefix_seconds`` contains only the current, unambiguous S1 or S1+S2.

    Known prior/current-epoch pit or neutralization contamination switches to
    corresponding whole-lap historical estimates. This boolean must be obtained
    from already received state, never from eventual CSV eligibility. Unknown
    future contamination is deliberately not inferred by these fixed references.
    """
    if type(known_asof_contamination) is not bool:
        raise ValueError("known_asof_contamination must be an explicit boolean")
    if not isinstance(prefix_seconds, (Sequence, np.ndarray)) or isinstance(prefix_seconds, (str, bytes)):
        raise ValueError("Observed prefix must contain one or two sectors")
    k = len(prefix_seconds)
    if k not in (1, 2):
        raise ValueError("Observed prefix must contain one or two sectors")
    prefix = _positive_vector(prefix_seconds, k, "observed prefix")
    prefix_sum = float(prefix.sum())
    if not np.isfinite(prefix_sum):
        raise ValueError("Observed prefix total must be finite")
    sectors, laps = _templates(history)
    remainders = laps-sectors[:, :k].sum(1)
    if not np.isfinite(remainders).all() or not (remainders > 0).all():
        raise ValueError("Historical remainder must be finite and positive")
    n_recent = min(len(laps), 10)
    age = np.arange(n_recent-1, -1, -1)
    weights = RECENCY_ALPHA*(1-RECENCY_ALPHA)**age
    if known_asof_contamination:
        points = {
            REFERENCE_NAMES[0]: float(laps[-1]),
            REFERENCE_NAMES[1]: float(np.median(laps[-5:])),
            REFERENCE_NAMES[2]: weighted_median(laps[-10:], weights),
            REFERENCE_NAMES[3]: float(np.median(laps[-5:])),
        }
        gaussian = {"status": "known_contamination_whole_lap_median5", "history_count": min(len(laps), 5),
                    "conditional_remainder_before_guard_seconds": None,
                    "statistic": "empirical_whole_lap_median"}
        points[GAUSSIAN_NAME] = points[PACE_REFERENCE]
    else:
        historical_prefix = sectors[-5:, :k].sum(1)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            ratios = np.clip(prefix_sum/historical_prefix, *PACE_RATIO_BOUNDS)
        points = {
            REFERENCE_NAMES[0]: prefix_sum+float(remainders[-1]),
            REFERENCE_NAMES[1]: prefix_sum+float(np.median(remainders[-5:])),
            REFERENCE_NAMES[2]: prefix_sum+weighted_median(remainders[-10:], weights),
            REFERENCE_NAMES[3]: prefix_sum+float(np.median(ratios*remainders[-5:])),
        }
        points[GAUSSIAN_NAME], gaussian = _gaussian(sectors, laps, prefix, points[PACE_REFERENCE])
    if any(not np.isfinite(value) or value <= 0 for value in points.values()):
        raise ValueError("Sector point forecasts must all be positive and finite")
    return {"points": points, "diagnostics": {
        "history_count": len(laps), "prefix_sector_count": k,
        "known_asof_contamination": known_asof_contamination,
        "reference_branch": "whole_lap" if known_asof_contamination else "observed_prefix_plus_remainder",
        "gaussian": gaussian,
    }}
