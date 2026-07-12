"""Causal baseline for the achievable session-end best lap.

This model answers the user-facing question "what best lap should this driver
actually achieve in qualifying?"  It deliberately does not reuse the
theoretical sector-floor target.  The baseline starts from the latest
target-aligned rehearsal (FP3 or Sprint Qualifying) and learns only a robust,
source-specific session-to-session shift from earlier events in the same
season.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    TARGET_CONTRACT_SEMANTICS,
)


REHEARSAL_LAP_COLUMN = "rehearsal_lap_time_seconds"
QUALITY_AWARE_ANCHOR_COLUMN = "quality_aware_anchor_seconds"
LATENT_POTENTIAL_ANCHOR_COLUMN = "latent_potential_adjusted_anchor_seconds"
ACTUAL_LAP_COLUMN = "achievable_session_end_lap_time_seconds"
REHEARSAL_SOURCE_COLUMN = "rehearsal_source"
EVENT_KEY_COLUMN = "event_key"
DRIVER_ID_COLUMN = "driver_id"
ALLOWED_REHEARSAL_SOURCES = frozenset({"practice_3", "sprint_qualifying"})
FORBIDDEN_INFERENCE_COLUMNS = frozenset(
    {
        ACTUAL_LAP_COLUMN,
        "qualifying_best_lap_time_seconds",
        "actual_lap_time",
        "lap_p05",
        "lap_p50",
        "lap_p90",
        "target",
    }
)

ROBUST_NUMERIC_FEATURE_ALLOWLIST: tuple[str, ...] = (
    "valid_minus_potential_seconds",
    "best_two_spread_seconds",
    "best_three_spread_seconds",
    "push_lap_count",
    "best_lap_recency_seconds",
    "best_lap_session_progress",
    "track_evolution_seconds_per_progress",
    "best_lap_tyre_age_laps",
    "best_lap_speed_trap",
    "teammate_relative_anchor_seconds",
    "field_relative_anchor_seconds",
    "evidence_coverage_rate",
    "anchor_uncertainty_seconds",
)
VALID_LAP_COLUMN = "has_valid_qualifying_lap"
REACHED_Q2_COLUMN = "reached_q2"
REACHED_Q3_COLUMN = "reached_q3"
HISTORY_WEIGHT_COLUMN = "history_weight"
WEAK_PRIOR_COLUMN = "weak_transfer_prior"


def _source_name(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fp3": "practice_3",
        "practice3": "practice_3",
        "sq": "sprint_qualifying",
        "sprint_shootout": "sprint_qualifying",
    }
    return aliases.get(normalized, normalized)


def _finite_seconds(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return every finite timing value, including a one-row sample."""

    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.where(np.isfinite(values) & values.between(40.0, 180.0))


def robust_huber_location(values: Sequence[float] | pd.Series, *, delta: float = 1.5) -> float:
    """Robust scalar location with a defined one-observation result.

    The previous experimental cleaner treated a zero-MAD sample as unusable.
    That discards the exact cold-start case this mode must support.  A single
    finite value is now returned unchanged; larger samples use Huber IRLS.
    """

    sample = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan")
    if sample.size == 1:
        return float(sample[0])
    location = float(np.median(sample))
    scale = float(np.median(np.abs(sample - location)) * 1.4826)
    if not np.isfinite(scale) or scale <= 1e-9:
        return location
    for _ in range(50):
        standardized = np.abs(sample - location) / scale
        weights = np.ones_like(standardized)
        tail = standardized > float(delta)
        weights[tail] = float(delta) / standardized[tail]
        updated = float(np.sum(weights * sample) / np.sum(weights))
        if abs(updated - location) <= 1e-9:
            break
        location = updated
    return location


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    threshold = float(np.clip(probability, 0.0, 1.0)) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, threshold, side="left")), len(ordered) - 1)
    if index + 1 < len(ordered) and np.isclose(cumulative[index], threshold):
        return float((ordered[index] + ordered[index + 1]) * 0.5)
    return float(ordered[index])


@dataclass(frozen=True)
class QualifyingStageProbabilities:
    """Hurdle probabilities for a classified valid lap and stage survival."""

    valid_lap: float
    q2_given_valid: float
    q3_given_q2: float
    status: str

    def __post_init__(self) -> None:
        for name in ("valid_lap", "q2_given_valid", "q3_given_q2"):
            value = float(getattr(self, name))
            if np.isfinite(value) and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability")

    def as_dict(self) -> dict[str, float | str]:
        valid = float(self.valid_lap)
        q2 = float(self.q2_given_valid)
        q3 = float(self.q3_given_q2)
        if not (np.isfinite(valid) and np.isfinite(q2) and np.isfinite(q3)):
            return {
                "valid_lap_probability": valid,
                "no_valid_lap_probability": 1.0 - valid if np.isfinite(valid) else float("nan"),
                "q2_given_valid_probability": q2,
                "q3_given_q2_probability": q3,
                "q1_only_probability": float("nan"),
                "q2_only_probability": float("nan"),
                "q3_probability": float("nan"),
                "stage_probability_status": self.status,
            }
        return {
            "valid_lap_probability": valid,
            "no_valid_lap_probability": 1.0 - valid,
            "q2_given_valid_probability": q2,
            "q3_given_q2_probability": q3,
            "q1_only_probability": valid * (1.0 - q2),
            "q2_only_probability": valid * q2 * (1.0 - q3),
            "q3_probability": valid * q2 * q3,
            "stage_probability_status": self.status,
        }


@dataclass(frozen=True)
class StageHurdleCalibration:
    valid_successes: float = 0.0
    valid_trials: float = 0.0
    q2_successes: float = 0.0
    q2_trials: float = 0.0
    q3_successes: float = 0.0
    q3_trials: float = 0.0
    event_keys: tuple[int, ...] = ()
    explicit_labels: bool = False
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def probabilities(self) -> QualifyingStageProbabilities:
        if not self.explicit_labels:
            return QualifyingStageProbabilities(
                valid_lap=1.0,
                q2_given_valid=float("nan"),
                q3_given_q2=float("nan"),
                status="unavailable_conditional_valid_population_only",
            )
        status = (
            "event_balanced_beta_binomial"
            if self.q2_trials > 0.0 and self.q3_trials > 0.0
            else "valid_event_balanced_stage_prior_only"
        )
        return QualifyingStageProbabilities(
            valid_lap=self._posterior(self.valid_successes, self.valid_trials),
            q2_given_valid=self._posterior(self.q2_successes, self.q2_trials),
            q3_given_q2=self._posterior(self.q3_successes, self.q3_trials),
            status=status,
        )

    def _posterior(self, successes: float, trials: float) -> float:
        return float(
            (float(successes) + float(self.prior_alpha))
            / (float(trials) + float(self.prior_alpha) + float(self.prior_beta))
        )


@dataclass(frozen=True)
class RobustHierarchicalResidualModel:
    """Small inspectable Huber/ridge residual with partial-pooled effects."""

    feature_names: tuple[str, ...] = ()
    centers: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    coefficients: tuple[float, ...] = ()
    intercept: float = 0.0
    team_effects: Mapping[str, float] = field(default_factory=dict)
    driver_effects: Mapping[str, float] = field(default_factory=dict)
    event_keys: tuple[int, ...] = ()
    fitted: bool = False

    def predict_adjustment(self, row: pd.Series) -> float:
        adjustment = float(self.intercept)
        for name, center, scale, coefficient in zip(
            self.feature_names, self.centers, self.scales, self.coefficients
        ):
            value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
            standardized = 0.0 if not np.isfinite(value) else (float(value) - center) / scale
            adjustment += float(coefficient) * standardized
        adjustment += float(self.team_effects.get(str(row.get("team_id", "")), 0.0))
        adjustment += float(self.driver_effects.get(str(row.get(DRIVER_ID_COLUMN, "")), 0.0))
        return float(np.clip(adjustment, -3.0, 3.0))


@dataclass(frozen=True)
class JointLapSamples:
    driver_ids: tuple[str, ...]
    lap_seconds: np.ndarray
    valid_mask: np.ndarray
    deepest_stage: np.ndarray


def decompose_event_fastest_and_driver_gap(
    frame: pd.DataFrame,
    *,
    anchor_column: str,
    target_column: str = ACTUAL_LAP_COLUMN,
) -> pd.DataFrame:
    """Split absolute lap time into event pace and driver gap components.

    This prevents a model from confusing a globally faster circuit/weekend
    with a driver's relative performance.  The decomposition is lossless on
    finite labelled rows and remains grouped by immutable event key.
    """

    required = {EVENT_KEY_COLUMN, anchor_column, target_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"event/gap decomposition is missing columns: {missing}")
    result = frame.copy()
    anchor = pd.to_numeric(result[anchor_column], errors="coerce")
    target = pd.to_numeric(result[target_column], errors="coerce")
    result["event_fastest_anchor_seconds"] = anchor.groupby(result[EVENT_KEY_COLUMN]).transform(
        "min"
    )
    result["event_fastest_target_seconds"] = target.groupby(result[EVENT_KEY_COLUMN]).transform(
        "min"
    )
    result["anchor_gap_to_event_fastest_seconds"] = (
        anchor - result["event_fastest_anchor_seconds"]
    )
    result["target_gap_to_event_fastest_seconds"] = (
        target - result["event_fastest_target_seconds"]
    )
    result["event_fastest_shift_seconds"] = (
        result["event_fastest_target_seconds"] - result["event_fastest_anchor_seconds"]
    )
    result["driver_gap_residual_seconds"] = (
        result["target_gap_to_event_fastest_seconds"]
        - result["anchor_gap_to_event_fastest_seconds"]
    )
    result["decomposed_total_residual_seconds"] = (
        result["event_fastest_shift_seconds"] + result["driver_gap_residual_seconds"]
    )
    return result


@dataclass(frozen=True)
class AchievableLapSourceCalibration:
    source: str
    event_keys: tuple[int, ...]
    event_shifts_seconds: tuple[float, ...]
    prequential_residuals_seconds: tuple[float, ...]
    event_prequential_residuals_seconds: tuple[tuple[float, ...], ...] = ()
    event_weights: tuple[float, ...] = ()
    conformal_event_keys: tuple[int, ...] = ()

    @property
    def shift_seconds(self) -> float:
        if not self.event_shifts_seconds:
            return 0.0
        values = np.asarray(self.event_shifts_seconds, dtype=float)
        weights = (
            np.asarray(self.event_weights, dtype=float)
            if len(self.event_weights) == len(values)
            else np.ones(len(values), dtype=float)
        )
        return _weighted_quantile(values, weights, 0.50)

    @property
    def event_count(self) -> int:
        return len(self.event_keys)

    @property
    def conformal_event_count(self) -> int:
        return len(self.conformal_event_keys)

    def centered_residual_quantiles(self, *, interval_mass: float = 0.85) -> tuple[float, float]:
        if not self.prequential_residuals_seconds:
            return float("nan"), float("nan")
        residuals = np.asarray(self.prequential_residuals_seconds, dtype=float)
        blocks = self.event_prequential_residuals_seconds
        if blocks and sum(len(block) for block in blocks) == len(residuals):
            block_weights = (
                self.event_weights
                if len(self.event_weights) == len(blocks)
                else tuple(1.0 for _ in blocks)
            )
            weights = np.concatenate(
                [
                    np.full(len(block), float(weight) / max(1, len(block)))
                    for block, weight in zip(blocks, block_weights)
                ]
            )
        else:
            weights = np.ones(len(residuals), dtype=float)
        alpha = 1.0 - float(interval_mass)
        # Equal block weight is the key protection against pseudo-replication.
        # We intentionally report small-sample status separately instead of
        # forcing a finite-sample max-residual interval that becomes unusably
        # wide with only a handful of Sprint blocks.
        lower_probability = alpha / 2.0
        upper_probability = 1.0 - alpha / 2.0
        q_low = _weighted_quantile(residuals, weights, lower_probability)
        q50 = _weighted_quantile(residuals, weights, 0.50)
        q_high = _weighted_quantile(residuals, weights, upper_probability)
        event_count = max(1, len(self.conformal_event_keys or self.event_keys))
        small_sample_inflation = float(np.sqrt((event_count + 1.0) / event_count))
        return (
            float((q_low - q50) * small_sample_inflation),
            float((q_high - q50) * small_sample_inflation),
        )


@dataclass(frozen=True)
class AchievableBestLapModel:
    target_event_key: int
    calibrations: Mapping[str, AchievableLapSourceCalibration]
    min_calibration_events: int = 2
    residual_model: RobustHierarchicalResidualModel = field(
        default_factory=RobustHierarchicalResidualModel
    )
    stage_calibration: StageHurdleCalibration = field(default_factory=StageHurdleCalibration)
    interval_mass: float = 0.85
    model_name: str = "achievable_best_lap_quality_aware_huber_v2"

    @property
    def training_event_keys(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    event_key
                    for calibration in self.calibrations.values()
                    for event_key in calibration.event_keys
                }
            )
        )

    def predict(self, inputs: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(inputs, pd.DataFrame):
            raise TypeError("achievable best-lap inference inputs must be a pandas DataFrame")
        leaked = sorted(FORBIDDEN_INFERENCE_COLUMNS.intersection(map(str, inputs.columns)))
        if leaked:
            raise ValueError(f"achievable best-lap inference contains target/outcome columns: {leaked}")
        required = {EVENT_KEY_COLUMN, DRIVER_ID_COLUMN, REHEARSAL_SOURCE_COLUMN}
        missing = sorted(required.difference(map(str, inputs.columns)))
        if missing:
            raise ValueError(f"achievable best-lap inference is missing columns: {missing}")
        if not any(
            column in inputs.columns
            for column in (
                LATENT_POTENTIAL_ANCHOR_COLUMN,
                QUALITY_AWARE_ANCHOR_COLUMN,
                REHEARSAL_LAP_COLUMN,
            )
        ):
            raise ValueError(
                "achievable best-lap inference requires quality_aware_anchor_seconds "
                "or rehearsal_lap_time_seconds"
            )
        if inputs.empty:
            return pd.DataFrame(index=inputs.index)

        event_keys = pd.to_numeric(inputs[EVENT_KEY_COLUMN], errors="coerce")
        if event_keys.isna().any() or set(event_keys.astype(int).tolist()) != {int(self.target_event_key)}:
            raise ValueError("inference rows must belong only to the model target_event_key")
        if self.training_event_keys and max(self.training_event_keys) >= int(self.target_event_key):
            raise ValueError("training history is not strictly earlier than the target event")

        anchor_column = next(
            column
            for column in (
                LATENT_POTENTIAL_ANCHOR_COLUMN,
                QUALITY_AWARE_ANCHOR_COLUMN,
                REHEARSAL_LAP_COLUMN,
            )
            if column in inputs.columns
        )
        rehearsal = _finite_seconds(inputs, anchor_column)
        if anchor_column == REHEARSAL_LAP_COLUMN and rehearsal.isna().any():
            raise ValueError("rehearsal lap inputs must be finite seconds between 40 and 180")

        rows: list[dict[str, object]] = []
        stage_payload = self.stage_calibration.probabilities().as_dict()
        for index, row in inputs.iterrows():
            source = _source_name(row[REHEARSAL_SOURCE_COLUMN])
            if source not in ALLOWED_REHEARSAL_SOURCES:
                raise ValueError(f"unsupported target-aligned rehearsal source: {source!r}")
            calibration = self.calibrations.get(
                source,
                AchievableLapSourceCalibration(source, (), (), ()),
            )
            anchor = float(rehearsal.loc[index]) if np.isfinite(rehearsal.loc[index]) else float("nan")
            raw_quality_anchor = pd.to_numeric(
                pd.Series([row.get(QUALITY_AWARE_ANCHOR_COLUMN)]), errors="coerce"
            ).iloc[0]
            raw_latent_anchor = pd.to_numeric(
                pd.Series([row.get(LATENT_POTENTIAL_ANCHOR_COLUMN)]), errors="coerce"
            ).iloc[0]
            residual_adjustment = self.residual_model.predict_adjustment(row)
            p50 = float(anchor + calibration.shift_seconds + residual_adjustment)
            lower_offset, upper_offset = calibration.centered_residual_quantiles(
                interval_mass=self.interval_mass
            )
            anchor_uncertainty = pd.to_numeric(
                pd.Series([row.get("anchor_uncertainty_seconds")]), errors="coerce"
            ).iloc[0]
            if np.isfinite(anchor_uncertainty):
                if not np.isfinite(lower_offset):
                    lower_offset = -float(anchor_uncertainty)
                else:
                    lower_offset = min(float(lower_offset), -float(anchor_uncertainty))
                if not np.isfinite(upper_offset):
                    upper_offset = float(anchor_uncertainty)
                else:
                    upper_offset = max(float(upper_offset), float(anchor_uncertainty))
            p05 = float(p50 + lower_offset) if np.isfinite(lower_offset) else float("nan")
            p90 = float(p50 + upper_offset) if np.isfinite(upper_offset) else float("nan")
            if np.isfinite(p05) and np.isfinite(p90):
                p05, p50, p90 = sorted((p05, p50, p90))
            interval_status = (
                "calibrated_minimum_event_count_met"
                if calibration.conformal_event_count >= int(self.min_calibration_events)
                else "diagnostic_underpowered"
                if calibration.conformal_event_count > 0
                else "fallback_anchor_uncertainty_no_current_regime_calibration"
                if np.isfinite(anchor_uncertainty)
                else "unavailable_no_same_source_history"
            )
            rows.append(
                {
                    EVENT_KEY_COLUMN: int(self.target_event_key),
                    DRIVER_ID_COLUMN: str(row[DRIVER_ID_COLUMN]),
                    REHEARSAL_SOURCE_COLUMN: source,
                    REHEARSAL_LAP_COLUMN: (
                        float(row[REHEARSAL_LAP_COLUMN])
                        if REHEARSAL_LAP_COLUMN in row.index
                        and np.isfinite(pd.to_numeric(pd.Series([row[REHEARSAL_LAP_COLUMN]]), errors="coerce").iloc[0])
                        else float("nan")
                    ),
                    QUALITY_AWARE_ANCHOR_COLUMN: (
                        float(raw_quality_anchor)
                        if np.isfinite(raw_quality_anchor)
                        else anchor
                    ),
                    LATENT_POTENTIAL_ANCHOR_COLUMN: (
                        float(raw_latent_anchor)
                        if np.isfinite(raw_latent_anchor)
                        else anchor
                    ),
                    "lap_p05": p05,
                    "lap_p50": p50,
                    "lap_p90": p90,
                    "session_shift_seconds": calibration.shift_seconds,
                    "robust_residual_adjustment_seconds": residual_adjustment,
                    "source_history_event_count": calibration.event_count,
                    "conformal_current_regime_event_count": calibration.conformal_event_count,
                    "interval_status": interval_status,
                    "interval_method": "rolling_equal_event_weight_conformal",
                    "interval_nominal_mass": float(self.interval_mass),
                    "anchor_source": row.get("anchor_source", "legacy_rehearsal_lap"),
                    "latent_anchor_source": row.get(
                        "latent_anchor_source", row.get("anchor_source", "legacy_rehearsal_lap")
                    ),
                    "latent_anchor_uses_potential": bool(
                        row.get("latent_anchor_uses_potential", False)
                    ),
                    "anchor_quality": row.get("anchor_quality", "legacy_valid_clean"),
                    "anchor_uncertainty_seconds": (
                        float(anchor_uncertainty)
                        if np.isfinite(anchor_uncertainty)
                        else float("nan")
                    ),
                    "anchor_available": bool(np.isfinite(anchor)),
                    "target_contract": ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
                    "target_semantics": TARGET_CONTRACT_SEMANTICS[
                        ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT
                    ],
                    "model": self.model_name,
                    "target_decomposition": "event_fastest_time_plus_driver_gap",
                    **stage_payload,
                }
            )
        return pd.DataFrame(rows, index=inputs.index)


def fit_achievable_best_lap_model(
    history: pd.DataFrame,
    *,
    target_event_key: int,
    min_calibration_events: int = 2,
    enable_robust_residual: bool = True,
    model_name: str | None = None,
) -> AchievableBestLapModel:
    """Fit source-specific shifts from strictly earlier labelled events."""

    if not isinstance(history, pd.DataFrame):
        raise TypeError("achievable best-lap history must be a pandas DataFrame")
    if int(min_calibration_events) < 1:
        raise ValueError("min_calibration_events must be positive")
    required = {
        EVENT_KEY_COLUMN,
        DRIVER_ID_COLUMN,
        REHEARSAL_SOURCE_COLUMN,
        ACTUAL_LAP_COLUMN,
    }
    if history.empty:
        return AchievableBestLapModel(
            target_event_key=int(target_event_key),
            calibrations={},
            min_calibration_events=int(min_calibration_events),
            model_name=model_name or (
                "achievable_best_lap_quality_aware_huber_v2"
                if enable_robust_residual
                else "achievable_best_lap_rehearsal_shift_v1"
            ),
        )
    missing = sorted(required.difference(map(str, history.columns)))
    if missing:
        raise ValueError(f"achievable best-lap history is missing columns: {missing}")

    if not any(
        column in history.columns
        for column in (
            LATENT_POTENTIAL_ANCHOR_COLUMN,
            QUALITY_AWARE_ANCHOR_COLUMN,
            REHEARSAL_LAP_COLUMN,
        )
    ):
        raise ValueError(
            "achievable best-lap history requires quality_aware_anchor_seconds "
            "or rehearsal_lap_time_seconds"
        )
    frame = history.copy()
    frame[EVENT_KEY_COLUMN] = pd.to_numeric(frame[EVENT_KEY_COLUMN], errors="coerce")
    anchor_column = next(
        column
        for column in (
            LATENT_POTENTIAL_ANCHOR_COLUMN,
            QUALITY_AWARE_ANCHOR_COLUMN,
            REHEARSAL_LAP_COLUMN,
        )
        if column in frame.columns
    )
    frame["_training_anchor_seconds"] = _finite_seconds(frame, anchor_column)
    frame[ACTUAL_LAP_COLUMN] = _finite_seconds(frame, ACTUAL_LAP_COLUMN)
    frame[REHEARSAL_SOURCE_COLUMN] = frame[REHEARSAL_SOURCE_COLUMN].map(_source_name)
    stage_calibration = _fit_stage_hurdle_calibration(frame)
    frame = frame.dropna(
        subset=[EVENT_KEY_COLUMN, "_training_anchor_seconds", ACTUAL_LAP_COLUMN]
    )
    if frame.empty:
        raise ValueError("achievable best-lap history has no valid labelled rows")
    frame[EVENT_KEY_COLUMN] = frame[EVENT_KEY_COLUMN].astype(int)
    if int(frame[EVENT_KEY_COLUMN].max()) >= int(target_event_key):
        raise ValueError("history must contain only events strictly earlier than target_event_key")
    unknown_sources = sorted(set(frame[REHEARSAL_SOURCE_COLUMN]) - ALLOWED_REHEARSAL_SOURCES)
    if unknown_sources:
        raise ValueError(f"unsupported target-aligned rehearsal sources: {unknown_sources}")

    calibrations: dict[str, AchievableLapSourceCalibration] = {}
    for source, source_rows in frame.groupby(REHEARSAL_SOURCE_COLUMN, sort=True):
        event_keys: list[int] = []
        event_shifts: list[float] = []
        event_weights: list[float] = []
        prequential_residuals: list[float] = []
        event_prequential_residuals: list[tuple[float, ...]] = []
        event_is_weak_prior: list[bool] = []
        for event_key, event_rows in source_rows.groupby(EVENT_KEY_COLUMN, sort=True):
            forecast_shift = (
                _weighted_quantile(
                    np.asarray(event_shifts, dtype=float),
                    np.asarray(event_weights, dtype=float),
                    0.50,
                )
                if event_shifts
                else 0.0
            )
            raw_shift = event_rows[ACTUAL_LAP_COLUMN] - event_rows["_training_anchor_seconds"]
            event_shift = robust_huber_location(raw_shift)
            event_errors = (
                event_rows[ACTUAL_LAP_COLUMN]
                - (event_rows["_training_anchor_seconds"] + forecast_shift)
            )
            event_error_values = tuple(float(value) for value in event_errors.dropna().tolist())
            prequential_residuals.extend(event_error_values)
            event_prequential_residuals.append(event_error_values)
            event_keys.append(int(event_key))
            event_shifts.append(event_shift)
            if HISTORY_WEIGHT_COLUMN in event_rows.columns:
                event_weight = pd.to_numeric(
                    event_rows[HISTORY_WEIGHT_COLUMN], errors="coerce"
                ).median(skipna=True)
            else:
                event_weight = 1.0
            event_weights.append(
                float(event_weight) if np.isfinite(event_weight) and event_weight > 0.0 else 1.0
            )
            event_is_weak_prior.append(
                bool(
                    _probability_label(event_rows[WEAK_PRIOR_COLUMN]).fillna(0.0).astype(bool).all()
                )
                if WEAK_PRIOR_COLUMN in event_rows.columns
                else False
            )
        strong_indices = [
            index for index, is_weak in enumerate(event_is_weak_prior) if not is_weak
        ][-12:]
        rolling_event_residuals = [event_prequential_residuals[index] for index in strong_indices]
        calibrations[str(source)] = AchievableLapSourceCalibration(
            source=str(source),
            event_keys=tuple(event_keys),
            event_shifts_seconds=tuple(event_shifts),
            prequential_residuals_seconds=tuple(
                value for block in rolling_event_residuals for value in block
            ),
            event_prequential_residuals_seconds=tuple(rolling_event_residuals),
            event_weights=tuple(event_weights),
            conformal_event_keys=tuple(event_keys[index] for index in strong_indices),
        )

    residual_model = (
        _fit_robust_hierarchical_residual(frame, calibrations)
        if enable_robust_residual
        else RobustHierarchicalResidualModel()
    )
    return AchievableBestLapModel(
        target_event_key=int(target_event_key),
        calibrations=calibrations,
        min_calibration_events=int(min_calibration_events),
        residual_model=residual_model,
        stage_calibration=stage_calibration,
        model_name=model_name or (
            "achievable_best_lap_quality_aware_huber_v2"
            if enable_robust_residual
            else "achievable_best_lap_rehearsal_shift_v1"
        ),
    )


def _fit_stage_hurdle_calibration(frame: pd.DataFrame) -> StageHurdleCalibration:
    explicit = VALID_LAP_COLUMN in frame.columns
    if not explicit:
        return StageHurdleCalibration()
    usable = frame.dropna(subset=[EVENT_KEY_COLUMN]).copy()
    if usable.empty:
        return StageHurdleCalibration(explicit_labels=True)
    usable["_valid_label"] = _probability_label(usable[VALID_LAP_COLUMN])
    if REACHED_Q2_COLUMN in usable.columns:
        usable["_q2_label"] = _probability_label(usable[REACHED_Q2_COLUMN])
    else:
        usable["_q2_label"] = np.nan
    if REACHED_Q3_COLUMN in usable.columns:
        usable["_q3_label"] = _probability_label(usable[REACHED_Q3_COLUMN])
    else:
        usable["_q3_label"] = np.nan

    # Each event contributes total weight one, so a 20-car weekend cannot
    # overpower a sparse weekend or masquerade as 20 independent events.
    event_size = usable.groupby(EVENT_KEY_COLUMN)[EVENT_KEY_COLUMN].transform("size")
    usable["_event_weight"] = 1.0 / event_size.clip(lower=1).astype(float)
    if HISTORY_WEIGHT_COLUMN in usable.columns:
        transfer_weight = pd.to_numeric(
            usable[HISTORY_WEIGHT_COLUMN], errors="coerce"
        ).fillna(1.0).clip(lower=0.0)
        usable["_event_weight"] *= transfer_weight
    valid_rows = usable.loc[usable["_valid_label"].notna()]
    q2_rows = usable.loc[(usable["_valid_label"] == 1.0) & usable["_q2_label"].notna()]
    q3_rows = usable.loc[(usable["_q2_label"] == 1.0) & usable["_q3_label"].notna()]
    return StageHurdleCalibration(
        valid_successes=float((valid_rows["_valid_label"] * valid_rows["_event_weight"]).sum()),
        valid_trials=float(valid_rows["_event_weight"].sum()),
        q2_successes=float((q2_rows["_q2_label"] * q2_rows["_event_weight"]).sum()),
        q2_trials=float(q2_rows["_event_weight"].sum()),
        q3_successes=float((q3_rows["_q3_label"] * q3_rows["_event_weight"]).sum()),
        q3_trials=float(q3_rows["_event_weight"].sum()),
        event_keys=tuple(sorted(usable[EVENT_KEY_COLUMN].astype(int).unique().tolist())),
        explicit_labels=True,
    )


def _probability_label(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(float)
    numeric = pd.to_numeric(values, errors="coerce")
    normalized = values.astype(str).str.strip().str.lower().map(
        {"true": 1.0, "yes": 1.0, "false": 0.0, "no": 0.0}
    )
    result = numeric.fillna(normalized)
    return result.where(result.isin([0.0, 1.0]))


def _fit_robust_hierarchical_residual(
    frame: pd.DataFrame,
    calibrations: Mapping[str, AchievableLapSourceCalibration],
) -> RobustHierarchicalResidualModel:
    if frame.empty:
        return RobustHierarchicalResidualModel()
    working = decompose_event_fastest_and_driver_gap(
        frame,
        anchor_column="_training_anchor_seconds",
        target_column=ACTUAL_LAP_COLUMN,
    )
    working["_base_shift"] = working[REHEARSAL_SOURCE_COLUMN].map(
        {source: calibration.shift_seconds for source, calibration in calibrations.items()}
    ).fillna(0.0)
    working["_residual"] = (
        working["event_fastest_shift_seconds"]
        - working["_base_shift"]
        + working["driver_gap_residual_seconds"]
    )
    features = tuple(
        name
        for name in ROBUST_NUMERIC_FEATURE_ALLOWLIST
        if name in working.columns
        and pd.to_numeric(working[name], errors="coerce").nunique(dropna=True) >= 2
    )
    event_counts = working.groupby(EVENT_KEY_COLUMN)[EVENT_KEY_COLUMN].transform("size")
    event_weights = 1.0 / event_counts.clip(lower=1).to_numpy(dtype=float)
    if HISTORY_WEIGHT_COLUMN in working.columns:
        event_weights *= pd.to_numeric(
            working[HISTORY_WEIGHT_COLUMN], errors="coerce"
        ).fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)
    y = pd.to_numeric(working["_residual"], errors="coerce").to_numpy(dtype=float)
    finite_y = np.isfinite(y)
    working = working.loc[finite_y].copy()
    y = y[finite_y]
    event_weights = event_weights[finite_y]
    if len(y) == 0:
        return RobustHierarchicalResidualModel()

    centers: list[float] = []
    scales: list[float] = []
    columns: list[np.ndarray] = []
    for feature in features:
        raw = pd.to_numeric(working[feature], errors="coerce").to_numpy(dtype=float)
        finite = raw[np.isfinite(raw)]
        center = float(np.median(finite)) if len(finite) else 0.0
        scale = float(np.median(np.abs(finite - center)) * 1.4826) if len(finite) else 1.0
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = float(np.std(finite)) if len(finite) else 1.0
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        centers.append(center)
        scales.append(scale)
        columns.append(np.where(np.isfinite(raw), (raw - center) / scale, 0.0))
    x = np.column_stack([np.ones(len(y)), *columns])
    coefficients = _huber_ridge(x, y, event_weights, ridge=2.0, delta=1.5)
    fitted = x @ coefficients
    remainder = y - fitted
    working["_remainder"] = remainder

    group_effect_rows = (
        working.loc[
            ~_probability_label(working[WEAK_PRIOR_COLUMN]).fillna(0.0).astype(bool)
        ].copy()
        if WEAK_PRIOR_COLUMN in working.columns
        else working.copy()
    )
    team_effects = _partial_pooled_effects(
        group_effect_rows,
        group_column="team_id",
        residual_column="_remainder",
        shrinkage=6.0,
    )
    team_adjustment = (
        working["team_id"].astype(str).map(team_effects).fillna(0.0).to_numpy(dtype=float)
        if "team_id" in working.columns
        else np.zeros(len(working), dtype=float)
    )
    working["_driver_remainder"] = remainder - team_adjustment
    group_effect_rows = working.loc[group_effect_rows.index]
    group_effect_rows["_driver_remainder"] = working.loc[
        group_effect_rows.index, "_driver_remainder"
    ]
    driver_effects = _partial_pooled_effects(
        group_effect_rows,
        group_column=DRIVER_ID_COLUMN,
        residual_column="_driver_remainder",
        shrinkage=10.0,
    )
    return RobustHierarchicalResidualModel(
        feature_names=features,
        centers=tuple(centers),
        scales=tuple(scales),
        coefficients=tuple(float(value) for value in coefficients[1:]),
        # The source-specific event shift is the intercept.  Keeping the
        # residual intercept at zero prevents a second, row-count-weighted
        # global shift from undoing event-balanced source calibration.
        intercept=0.0,
        team_effects=team_effects,
        driver_effects=driver_effects,
        event_keys=tuple(sorted(working[EVENT_KEY_COLUMN].astype(int).unique().tolist())),
        fitted=bool(len(y) >= 2),
    )


def _huber_ridge(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    ridge: float,
    delta: float,
) -> np.ndarray:
    if len(y) == 1:
        result = np.zeros(x.shape[1], dtype=float)
        result[0] = float(y[0])
        return result
    penalty = np.eye(x.shape[1], dtype=float) * float(ridge)
    penalty[0, 0] = 1e-9
    weights = np.asarray(sample_weight, dtype=float)
    coefficients = np.zeros(x.shape[1], dtype=float)
    coefficients[0] = robust_huber_location(y)
    for _ in range(75):
        residual = y - (x @ coefficients)
        scale = float(np.median(np.abs(residual - np.median(residual))) * 1.4826)
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = max(float(np.std(residual)), 1e-6)
        standardized = np.abs(residual) / scale
        huber_weight = np.ones_like(standardized)
        tail = standardized > float(delta)
        huber_weight[tail] = float(delta) / standardized[tail]
        combined = weights * huber_weight
        xtw = x.T * combined
        updated = np.linalg.solve(xtw @ x + penalty, xtw @ y)
        if np.max(np.abs(updated - coefficients)) <= 1e-8:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def _partial_pooled_effects(
    frame: pd.DataFrame,
    *,
    group_column: str,
    residual_column: str,
    shrinkage: float,
) -> dict[str, float]:
    if group_column not in frame.columns:
        return {}
    effects: dict[str, float] = {}
    for key, rows in frame.groupby(group_column, sort=False, dropna=False):
        values = pd.to_numeric(rows[residual_column], errors="coerce").dropna()
        if len(values) == 0:
            continue
        location = robust_huber_location(values)
        weight = float(len(values) / (len(values) + float(shrinkage)))
        effects[str(key)] = float(np.clip(location * weight, -1.5, 1.5))
    return effects


def sample_joint_qualifying_laps(
    predictions: pd.DataFrame,
    *,
    samples: int = 10_000,
    seed: int = 0,
    shared_session_fraction: float = 0.30,
) -> JointLapSamples:
    """Draw one coherent field of valid-lap/stage outcomes per simulation."""

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    required = {DRIVER_ID_COLUMN, "lap_p50", "valid_lap_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"joint lap sampling is missing columns: {missing}")
    if int(samples) < 1:
        raise ValueError("samples must be positive")
    rng = np.random.default_rng(int(seed))
    drivers = tuple(predictions[DRIVER_ID_COLUMN].astype(str).tolist())
    count = len(drivers)
    p50 = pd.to_numeric(predictions["lap_p50"], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(predictions.get("lap_p05", np.nan), errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(predictions.get("lap_p90", np.nan), errors="coerce").to_numpy(dtype=float)
    sigma = (upper - lower) / 2.9264
    fallback_sigma = pd.to_numeric(
        predictions.get("anchor_uncertainty_seconds", 0.75), errors="coerce"
    )
    if np.isscalar(fallback_sigma):
        fallback_sigma = np.full(count, float(fallback_sigma))
    else:
        fallback_sigma = fallback_sigma.to_numpy(dtype=float)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.02), sigma, fallback_sigma)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.02), sigma, 0.75)

    valid_probability = pd.to_numeric(
        predictions["valid_lap_probability"], errors="coerce"
    ).fillna(1.0).to_numpy(dtype=float)
    q2_probability = pd.to_numeric(
        predictions.get("q2_given_valid_probability", np.nan), errors="coerce"
    )
    q3_probability = pd.to_numeric(
        predictions.get("q3_given_q2_probability", np.nan), errors="coerce"
    )
    if np.isscalar(q2_probability):
        q2_probability = np.full(count, 0.75)
    else:
        q2_probability = q2_probability.fillna(0.75).to_numpy(dtype=float)
    if np.isscalar(q3_probability):
        q3_probability = np.full(count, 0.50)
    else:
        q3_probability = q3_probability.fillna(0.50).to_numpy(dtype=float)

    valid = rng.random((int(samples), count)) < np.clip(valid_probability, 0.0, 1.0)
    reached_q2 = valid & (rng.random((int(samples), count)) < np.clip(q2_probability, 0.0, 1.0))
    reached_q3 = reached_q2 & (rng.random((int(samples), count)) < np.clip(q3_probability, 0.0, 1.0))
    stages = np.where(reached_q3, 3, np.where(reached_q2, 2, np.where(valid, 1, 0)))
    stage_penalty = np.where(stages == 1, 0.40, np.where(stages == 2, 0.15, 0.0))
    shared = rng.normal(0.0, 1.0, size=(int(samples), 1))
    independent = rng.normal(0.0, 1.0, size=(int(samples), count))
    fraction = float(np.clip(shared_session_fraction, 0.0, 0.95))
    noise = sigma * (fraction * shared + np.sqrt(1.0 - fraction**2) * independent)
    lap_seconds = p50 + stage_penalty + noise
    lap_seconds[~valid] = np.nan
    lap_seconds[:, ~np.isfinite(p50)] = np.nan
    valid[:, ~np.isfinite(p50)] = False
    stages[:, ~np.isfinite(p50)] = 0
    return JointLapSamples(
        driver_ids=drivers,
        lap_seconds=lap_seconds,
        valid_mask=valid,
        deepest_stage=stages,
    )


def summarize_joint_lap_samples(samples: JointLapSamples) -> pd.DataFrame:
    """Summarize fastest and top-three probabilities from joint field draws."""

    lap_seconds = np.asarray(samples.lap_seconds, dtype=float)
    if lap_seconds.ndim != 2 or lap_seconds.shape[1] != len(samples.driver_ids):
        raise ValueError("joint lap samples have inconsistent shape")
    valid_fields = np.isfinite(lap_seconds).any(axis=1)
    fastest_count = np.zeros(len(samples.driver_ids), dtype=float)
    top3_count = np.zeros(len(samples.driver_ids), dtype=float)
    for row in lap_seconds[valid_fields]:
        finite = np.flatnonzero(np.isfinite(row))
        order = finite[np.argsort(row[finite], kind="mergesort")]
        if len(order):
            fastest_count[order[0]] += 1.0
            top3_count[order[: min(3, len(order))]] += 1.0
    denominator = max(1, int(valid_fields.sum()))
    return pd.DataFrame(
        {
            DRIVER_ID_COLUMN: samples.driver_ids,
            "fastest_driver_probability": fastest_count / denominator,
            "top3_probability": top3_count / denominator,
            "valid_lap_probability_sampled": np.mean(samples.valid_mask, axis=0),
        }
    )


__all__ = [
    "ACTUAL_LAP_COLUMN",
    "JointLapSamples",
    "LATENT_POTENTIAL_ANCHOR_COLUMN",
    "QUALITY_AWARE_ANCHOR_COLUMN",
    "QualifyingStageProbabilities",
    "ROBUST_NUMERIC_FEATURE_ALLOWLIST",
    "RobustHierarchicalResidualModel",
    "StageHurdleCalibration",
    "AchievableBestLapModel",
    "AchievableLapSourceCalibration",
    "fit_achievable_best_lap_model",
    "decompose_event_fastest_and_driver_gap",
    "robust_huber_location",
    "sample_joint_qualifying_laps",
    "summarize_joint_lap_samples",
]
