"""Causal baseline for the achievable session-end best lap.

This model answers the user-facing question "what best lap should this driver
actually achieve in qualifying?"  It deliberately does not reuse the
theoretical sector-floor target.  The baseline starts from the latest
target-aligned rehearsal (FP3 or Sprint Qualifying) and learns only a robust,
source-specific session-to-session shift from earlier events in the same
season.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
import hashlib
import json
import math
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.schemas import (
    ACHIEVABLE_SESSION_END_LAP_TARGET_CONTRACT,
    TARGET_CONTRACT_SEMANTICS,
)
from packages.f1.domain.weekend import qualifying_elimination_rule


REHEARSAL_LAP_COLUMN = "rehearsal_lap_time_seconds"
QUALITY_AWARE_ANCHOR_COLUMN = "quality_aware_anchor_seconds"
LATENT_POTENTIAL_ANCHOR_COLUMN = "latent_potential_adjusted_anchor_seconds"
ACTUAL_LAP_COLUMN = "achievable_session_end_lap_time_seconds"
Q1_LAP_COLUMN = "qualifying_q1_lap_time_seconds"
Q2_LAP_COLUMN = "qualifying_q2_lap_time_seconds"
Q3_LAP_COLUMN = "qualifying_q3_lap_time_seconds"
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
        "qualy_position",
        "qualifying_position",
        "has_valid_qualifying_lap",
        "reached_q2",
        "reached_q3",
        Q1_LAP_COLUMN,
        Q2_LAP_COLUMN,
        Q3_LAP_COLUMN,
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
SHARED_QUALIFYING_SAMPLE_COUNT = 5_000
SHARED_QUALIFYING_SAMPLE_SEED_BASE = 20_260_713
SHARED_QUALIFYING_ENABLE_ROBUST_RESIDUAL = False


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
    official_positions: np.ndarray | None = None
    stage_lap_seconds: np.ndarray | None = None
    stage_advancement_status: str = "fitted_hurdle_gumbel_top_k"
    stage_time_distribution_status: str = "learned_stage_residual_dispersion"


@dataclass(frozen=True)
class LearnedStageTimeEffects:
    """Best-lap shifts learned for the deepest official stage reached."""

    q1_only_seconds: float = 0.0
    q2_only_seconds: float = 0.0
    q3_seconds: float = 0.0
    q1_residual_sigma_seconds: float = float("nan")
    q2_residual_sigma_seconds: float = float("nan")
    q3_residual_sigma_seconds: float = float("nan")
    event_keys: tuple[int, ...] = ()
    fitted: bool = False
    status: str = "unavailable_no_explicit_nested_stage_labels"

    def for_stage(self, stage: int) -> float:
        return {
            1: float(self.q1_only_seconds),
            2: float(self.q2_only_seconds),
            3: float(self.q3_seconds),
        }.get(int(stage), 0.0)

    def sigma_for_stage(self, stage: int) -> float:
        return {
            1: float(self.q1_residual_sigma_seconds),
            2: float(self.q2_residual_sigma_seconds),
            3: float(self.q3_residual_sigma_seconds),
        }.get(int(stage), float("nan"))


@dataclass(frozen=True)
class SharedQualifyingForecast:
    """Best-lap marginals and official classification from one fitted model."""

    lap_predictions: pd.DataFrame
    point_order: pd.DataFrame
    position_marginals: pd.DataFrame
    samples: JointLapSamples
    probability_calibration_status: str = "uncalibrated_joint_latent_samples"


def shared_qualifying_forecast_artifact(
    forecast: SharedQualifyingForecast,
    *,
    model: AchievableBestLapModel,
    training_partition_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Return a deterministic manifest binding both modes to exact joint draws."""

    if not isinstance(forecast, SharedQualifyingForecast):
        raise TypeError("forecast must be a SharedQualifyingForecast")
    if not isinstance(model, AchievableBestLapModel):
        raise TypeError("model must be an AchievableBestLapModel")
    if not isinstance(training_partition_manifest, Mapping):
        raise TypeError("training_partition_manifest must be a mapping")
    sample_digest = hashlib.sha256()
    sample_digest.update(
        json.dumps(list(forecast.samples.driver_ids), separators=(",", ":")).encode(
            "utf-8"
        )
    )
    sample_digest.update(forecast.samples.stage_advancement_status.encode("utf-8"))
    sample_digest.update(forecast.samples.stage_time_distribution_status.encode("utf-8"))
    for name, values, dtype in (
        ("lap_seconds", forecast.samples.lap_seconds, "<f8"),
        ("valid_mask", forecast.samples.valid_mask, "u1"),
        ("deepest_stage", forecast.samples.deepest_stage, "i1"),
        ("official_positions", forecast.samples.official_positions, "<i2"),
        ("stage_lap_seconds", forecast.samples.stage_lap_seconds, "<f8"),
    ):
        sample_digest.update(name.encode("utf-8"))
        if values is None:
            sample_digest.update(b"<NONE>")
            continue
        array = np.ascontiguousarray(np.asarray(values).astype(dtype, copy=False))
        sample_digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        sample_digest.update(array.tobytes(order="C"))

    def frame_digest(frame: pd.DataFrame) -> str:
        # Pandas indices are execution-local bookkeeping, not forecast data.
        # Canonicalize row/column order so the Qualifying and Best-Lap runners
        # hash identical shared outputs even when they slice the same event from
        # differently indexed season frames.
        canonical = frame.copy()
        sort_columns = [
            column
            for column in (EVENT_KEY_COLUMN, DRIVER_ID_COLUMN)
            if column in canonical.columns
        ]
        if sort_columns:
            canonical = canonical.sort_values(sort_columns, kind="mergesort")
        canonical = canonical.reindex(sorted(canonical.columns), axis=1).reset_index(
            drop=True
        )
        payload = canonical.to_json(
            orient="split",
            date_format="iso",
            double_precision=15,
            index=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    event_keys = pd.to_numeric(
        forecast.lap_predictions.get(EVENT_KEY_COLUMN), errors="coerce"
    ).dropna()
    event_key = int(event_keys.iloc[0]) if len(event_keys) else None
    training_partition_payload = json.loads(
        json.dumps(dict(training_partition_manifest), sort_keys=True)
    )
    training_partition_sha256 = hashlib.sha256(
        json.dumps(
            training_partition_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    model_manifest = {
        "model_name": model.model_name,
        "point_predictor_sha256": shared_point_predictor_sha256(model),
        "point_training_event_keys": list(model.training_event_keys),
        "target_event_key": int(model.target_event_key),
        "interval_calibration_event_keys": list(model.calibration_event_keys),
        "interval_calibration_validated": bool(model.calibration_partition_validated),
        "training_partition_sha256": training_partition_sha256,
    }
    model_manifest_sha256 = hashlib.sha256(
        json.dumps(model_manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    payload: dict[str, object] = {
        "schema_version": "f1_shared_qualifying_forecast_artifact_v2",
        "event_key": event_key,
        "driver_ids": list(forecast.samples.driver_ids),
        "joint_sample_count": int(forecast.samples.lap_seconds.shape[0]),
        "joint_samples_sha256": sample_digest.hexdigest(),
        "stage_advancement_status": forecast.samples.stage_advancement_status,
        "stage_time_distribution_status": (
            forecast.samples.stage_time_distribution_status
        ),
        "best_lap_outputs_sha256": frame_digest(forecast.lap_predictions),
        "qualifying_point_outputs_sha256": frame_digest(forecast.point_order),
        "qualifying_position_marginals_sha256": frame_digest(
            forecast.position_marginals
        ),
        "model_sha256": model_manifest["point_predictor_sha256"],
        "model_manifest_sha256": model_manifest_sha256,
        "model_manifest": model_manifest,
        "training_partition_manifest": training_partition_payload,
        "shared_samples_drive_best_lap_and_qualifying": True,
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


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

    def centered_residual_quantiles(
        self,
        *,
        lower_probability: float = 0.05,
        upper_probability: float = 0.90,
    ) -> tuple[float, float]:
        if not self.prequential_residuals_seconds:
            return float("nan"), float("nan")
        if not 0.0 <= float(lower_probability) < 0.5:
            raise ValueError("lower_probability must be in [0, 0.5)")
        if not 0.5 < float(upper_probability) <= 1.0:
            raise ValueError("upper_probability must be in (0.5, 1]")
        if float(lower_probability) >= float(upper_probability):
            raise ValueError("residual quantile probabilities must be ordered")
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
        # Equal block weight is the key protection against pseudo-replication.
        # We intentionally report small-sample status separately instead of
        # forcing a finite-sample max-residual interval that becomes unusably
        # wide with only a handful of Sprint blocks.
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
    stage_probability_model: Any | None = None
    stage_time_effects: LearnedStageTimeEffects = field(default_factory=LearnedStageTimeEffects)
    interval_calibrations: Mapping[str, AchievableLapSourceCalibration] = field(
        default_factory=dict
    )
    calibration_event_keys: tuple[int, ...] = ()
    calibration_partition_validated: bool = False
    interval_mass: float = 0.85
    interval_lower_probability: float = 0.05
    interval_upper_probability: float = 0.90
    model_name: str = "shared_qualifying_latent_lap_v3"

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
        conditioned_stages: pd.DataFrame | None = None
        if self.stage_probability_model is not None:
            conditioned_stages = self.stage_probability_model.predict_event(inputs)
        for index, row in inputs.iterrows():
            source = _source_name(row[REHEARSAL_SOURCE_COLUMN])
            if source not in ALLOWED_REHEARSAL_SOURCES:
                raise ValueError(f"unsupported target-aligned rehearsal source: {source!r}")
            calibration = self.calibrations.get(
                source,
                AchievableLapSourceCalibration(source, (), (), ()),
            )
            interval_calibration = self.interval_calibrations.get(
                source,
                AchievableLapSourceCalibration(source, (), (), ()),
            )
            if interval_calibration.conformal_event_count < int(
                self.min_calibration_events
            ):
                pooled_interval = self.interval_calibrations.get("__all__")
                if pooled_interval is not None:
                    interval_calibration = pooled_interval
            diagnostic_or_validated_interval = (
                interval_calibration
                if interval_calibration.conformal_event_count > 0
                else calibration
            )
            anchor = float(rehearsal.loc[index]) if np.isfinite(rehearsal.loc[index]) else float("nan")
            raw_quality_anchor = pd.to_numeric(
                pd.Series([row.get(QUALITY_AWARE_ANCHOR_COLUMN)]), errors="coerce"
            ).iloc[0]
            raw_latent_anchor = pd.to_numeric(
                pd.Series([row.get(LATENT_POTENTIAL_ANCHOR_COLUMN)]), errors="coerce"
            ).iloc[0]
            residual_adjustment = self.residual_model.predict_adjustment(row)
            latent_location = float(anchor + calibration.shift_seconds + residual_adjustment)
            lower_offset, upper_offset = (
                diagnostic_or_validated_interval.centered_residual_quantiles(
                lower_probability=self.interval_lower_probability,
                upper_probability=self.interval_upper_probability,
                )
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
            stage_payload = self._stage_payload_for_row(
                index=index,
                conditioned=conditioned_stages,
            )
            sigma = (
                float((upper_offset - lower_offset) / 2.926405326)
                if np.isfinite(lower_offset)
                and np.isfinite(upper_offset)
                and upper_offset > lower_offset
                else float(anchor_uncertainty) / 1.644853627
                if np.isfinite(anchor_uncertainty) and anchor_uncertainty > 0.0
                else float("nan")
            )
            stage_weights = _conditional_stage_weights(stage_payload)
            stage_means = tuple(
                latent_location + self.stage_time_effects.for_stage(stage)
                for stage in (1, 2, 3)
            )
            if stage_weights is not None and np.isfinite(sigma) and sigma > 0.0:
                p05 = _normal_mixture_quantile(
                    stage_weights,
                    stage_means,
                    sigma,
                    self.interval_lower_probability,
                )
                p50 = _normal_mixture_quantile(stage_weights, stage_means, sigma, 0.50)
                p90 = _normal_mixture_quantile(
                    stage_weights,
                    stage_means,
                    sigma,
                    self.interval_upper_probability,
                )
            else:
                p50 = latent_location
                p05 = (
                    float(latent_location + lower_offset)
                    if np.isfinite(lower_offset)
                    else float("nan")
                )
                p90 = (
                    float(latent_location + upper_offset)
                    if np.isfinite(upper_offset)
                    else float("nan")
                )
            if np.isfinite(p05) and np.isfinite(p90):
                p05, p50, p90 = sorted((p05, p50, p90))
            interval_status = (
                "calibrated_disjoint_event_partition"
                if self.calibration_partition_validated
                and interval_calibration.conformal_event_count
                >= int(self.min_calibration_events)
                else "diagnostic_calibration_rows_reused_for_model_fit"
                if self.calibration_event_keys
                and not self.calibration_partition_validated
                and calibration.conformal_event_count > 0
                else "diagnostic_underpowered_disjoint_calibration"
                if self.calibration_partition_validated
                and interval_calibration.conformal_event_count > 0
                else "diagnostic_no_disjoint_calibration_partition"
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
                    "lap_p05_quantile_probability": float(self.interval_lower_probability),
                    "lap_p50_quantile_probability": 0.50,
                    "lap_p90_quantile_probability": float(self.interval_upper_probability),
                    "latent_lap_location_seconds": latent_location,
                    "latent_lap_sigma_seconds": sigma,
                    "stage_q1_time_effect_seconds": self.stage_time_effects.q1_only_seconds,
                    "stage_q2_time_effect_seconds": self.stage_time_effects.q2_only_seconds,
                    "stage_q3_time_effect_seconds": self.stage_time_effects.q3_seconds,
                    "stage_q1_residual_sigma_seconds": (
                        self.stage_time_effects.q1_residual_sigma_seconds
                    ),
                    "stage_q2_residual_sigma_seconds": (
                        self.stage_time_effects.q2_residual_sigma_seconds
                    ),
                    "stage_q3_residual_sigma_seconds": (
                        self.stage_time_effects.q3_residual_sigma_seconds
                    ),
                    "stage_time_effect_status": self.stage_time_effects.status,
                    "session_shift_seconds": calibration.shift_seconds,
                    "robust_residual_adjustment_seconds": residual_adjustment,
                    "source_history_event_count": calibration.event_count,
                    "conformal_current_regime_event_count": (
                        interval_calibration.conformal_event_count
                    ),
                    "interval_status": interval_status,
                    "interval_method": (
                        "stage_mixture_equal_event_weight_conformal"
                        if self.calibration_partition_validated
                        else (
                            "diagnostic_prequential_anchor_shift_residual_quantiles_"
                            "not_final_predictor_calibration"
                        )
                    ),
                    "interval_nominal_mass": float(self.interval_mass),
                    "interval_lower_probability": float(self.interval_lower_probability),
                    "interval_upper_probability": float(self.interval_upper_probability),
                    "calibration_partition_validated": bool(
                        self.calibration_partition_validated
                    ),
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

    def _stage_payload_for_row(
        self,
        *,
        index: object,
        conditioned: pd.DataFrame | None,
    ) -> dict[str, float | str]:
        if conditioned is None:
            return self.stage_calibration.probabilities().as_dict()
        row = conditioned.loc[index]
        valid = float(row["p_valid_qualifying_lap"])
        q2 = float(row["p_q2_given_valid"])
        q3 = float(row["p_q3_given_q2"])
        return {
            "valid_lap_probability": valid,
            "no_valid_lap_probability": 1.0 - valid,
            "q2_given_valid_probability": q2,
            "q3_given_q2_probability": q3,
            "q1_only_probability": valid * (1.0 - q2),
            "q2_only_probability": valid * q2 * (1.0 - q3),
            "q3_probability": valid * q2 * q3,
            "stage_probability_status": str(row["probability_calibration_status"]),
        }

    def predict_qualifying(
        self,
        inputs: pd.DataFrame,
        *,
        samples: int = 5_000,
        seed: int = 0,
        q2_slots: int | None = None,
        q3_slots: int | None = None,
        allow_diagnostic_stage_fallback: bool = False,
    ) -> SharedQualifyingForecast:
        """Predict Best Lap and the legal official classification jointly."""

        analytic_predictions = self.predict(inputs)
        joint = sample_joint_qualifying_laps(
            analytic_predictions,
            samples=samples,
            seed=seed,
            q2_slots=q2_slots,
            q3_slots=q3_slots,
            allow_diagnostic_pace_fallback=allow_diagnostic_stage_fallback,
        )
        lap_predictions = analytic_predictions.copy()
        for column in ("lap_p05", "lap_p50", "lap_p90"):
            lap_predictions[f"analytic_{column}"] = lap_predictions[column]
        for driver_index in range(len(joint.driver_ids)):
            conditional = joint.lap_seconds[:, driver_index]
            conditional = conditional[np.isfinite(conditional)]
            if conditional.size:
                lap_predictions.iloc[
                    driver_index, lap_predictions.columns.get_loc("lap_p05")
                ] = float(np.quantile(conditional, self.interval_lower_probability))
                lap_predictions.iloc[
                    driver_index, lap_predictions.columns.get_loc("lap_p50")
                ] = float(np.quantile(conditional, 0.50))
                lap_predictions.iloc[
                    driver_index, lap_predictions.columns.get_loc("lap_p90")
                ] = float(np.quantile(conditional, self.interval_upper_probability))
            else:
                for column in ("lap_p05", "lap_p50", "lap_p90"):
                    lap_predictions.iloc[
                        driver_index, lap_predictions.columns.get_loc(column)
                    ] = float("nan")
            analytic_row = analytic_predictions.iloc[driver_index]
            has_distribution_evidence = bool(
                np.isfinite(pd.to_numeric(analytic_row.get("analytic_lap_p05"), errors="coerce"))
                if "analytic_lap_p05" in analytic_predictions.columns
                else np.isfinite(pd.to_numeric(analytic_row.get("lap_p05"), errors="coerce"))
            )
            # The sampler needs a conservative fallback variance to rank a
            # cold-start field, but that fallback must not manufacture public
            # Best-Lap interval bounds without timing uncertainty evidence.
            if not has_distribution_evidence:
                lap_predictions.iloc[
                    driver_index, lap_predictions.columns.get_loc("lap_p05")
                ] = float("nan")
                lap_predictions.iloc[
                    driver_index, lap_predictions.columns.get_loc("lap_p90")
                ] = float("nan")
        lap_predictions["distribution_source"] = "shared_joint_qualifying_samples"
        lap_predictions["joint_sample_count"] = int(samples)
        lap_predictions["joint_sample_seed"] = int(seed)
        marginals = summarize_joint_lap_samples(joint)
        # The fitted hurdle heads are unconstrained row probabilities, while
        # an official session has a fixed number of Q2/Q3 places. Public stage
        # probabilities therefore come from the same legal joint draws used by
        # the classification. Preserve the raw heads for diagnostics.
        for column in (
            "valid_lap_probability",
            "q2_given_valid_probability",
            "q3_given_q2_probability",
            "stage_probability_status",
        ):
            if column in lap_predictions.columns:
                lap_predictions[f"unconstrained_hurdle_{column}"] = lap_predictions[
                    column
                ]
        sampled_valid = pd.to_numeric(
            marginals["valid_lap_probability_sampled"], errors="coerce"
        ).to_numpy(dtype=float)
        sampled_q2 = pd.to_numeric(
            marginals["reaches_q2_probability_sampled"], errors="coerce"
        ).to_numpy(dtype=float)
        sampled_q3 = pd.to_numeric(
            marginals["reaches_q3_probability_sampled"], errors="coerce"
        ).to_numpy(dtype=float)
        q2_given_valid = np.divide(
            sampled_q2,
            sampled_valid,
            out=np.zeros_like(sampled_q2),
            where=sampled_valid > 0.0,
        )
        q3_given_q2 = np.divide(
            sampled_q3,
            sampled_q2,
            out=np.zeros_like(sampled_q3),
            where=sampled_q2 > 0.0,
        )
        diagnostic_stage_fallback = bool(
            joint.stage_advancement_status == "diagnostic_pace_fallback_missing_hurdles"
            or joint.stage_time_distribution_status
            == "diagnostic_missing_learned_stage_residual_dispersion"
        )
        if diagnostic_stage_fallback:
            for column in (
                "valid_lap_probability",
                "no_valid_lap_probability",
                "q2_given_valid_probability",
                "q3_given_q2_probability",
                "q1_only_probability",
                "q2_only_probability",
                "q3_probability",
            ):
                lap_predictions[column] = np.nan
            stage_probability_status = "unavailable_diagnostic_joint_stage_model"
            probability_status = "unavailable_diagnostic_joint_stage_model"
        else:
            lap_predictions["valid_lap_probability"] = sampled_valid
            lap_predictions["no_valid_lap_probability"] = 1.0 - sampled_valid
            lap_predictions["q2_given_valid_probability"] = q2_given_valid
            lap_predictions["q3_given_q2_probability"] = q3_given_q2
            lap_predictions["q1_only_probability"] = sampled_valid - sampled_q2
            lap_predictions["q2_only_probability"] = sampled_q2 - sampled_q3
            lap_predictions["q3_probability"] = sampled_q3
            stage_probability_status = (
                "legal_field_constrained_joint_samples_uncalibrated"
            )
            probability_status = "uncalibrated_joint_latent_samples"
        lap_predictions["stage_probability_status"] = stage_probability_status
        marginals["probability_calibration_status"] = probability_status
        marginals["position_marginals_calibrated"] = False
        expected = marginals["expected_qualifying_position"].to_numpy(dtype=float)
        drivers = marginals[DRIVER_ID_COLUMN].astype(str).to_numpy()
        stable_order = np.lexsort((drivers, expected))
        point_positions = np.empty(len(drivers), dtype=int)
        point_positions[stable_order] = np.arange(1, len(drivers) + 1)
        point = lap_predictions.merge(
            marginals,
            on=DRIVER_ID_COLUMN,
            how="left",
            validate="one_to_one",
            suffixes=("", "_sampled"),
        )
        point["predicted_qualifying_position"] = point_positions
        point["qualifying_model"] = self.model_name
        point["probability_calibration_status"] = probability_status
        point["position_marginals_calibrated"] = False
        point = point.sort_values(
            ["predicted_qualifying_position", DRIVER_ID_COLUMN], kind="mergesort"
        )
        return SharedQualifyingForecast(
            lap_predictions=lap_predictions,
            point_order=point,
            position_marginals=marginals,
            samples=joint,
            probability_calibration_status=probability_status,
        )


def fit_achievable_best_lap_model(
    history: pd.DataFrame,
    *,
    target_event_key: int,
    min_calibration_events: int = 2,
    enable_robust_residual: bool = True,
    calibration_event_keys: Sequence[int] | None = None,
    model_name: str | None = None,
) -> AchievableBestLapModel:
    """Fit source-specific shifts from strictly earlier labelled events."""

    if not isinstance(history, pd.DataFrame):
        raise TypeError("achievable best-lap history must be a pandas DataFrame")
    if int(min_calibration_events) < 1:
        raise ValueError("min_calibration_events must be positive")
    requested_calibration_keys = tuple(
        sorted({int(value) for value in (calibration_event_keys or ())})
    )
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
            calibration_event_keys=requested_calibration_keys,
            calibration_partition_validated=False,
            model_name=model_name or (
                "shared_qualifying_latent_lap_huber_v3"
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
    stage_history = frame.copy()
    frame = frame.dropna(
        subset=[EVENT_KEY_COLUMN, "_training_anchor_seconds", ACTUAL_LAP_COLUMN]
    )
    if frame.empty:
        raise ValueError("achievable best-lap history has no valid labelled rows")
    frame[EVENT_KEY_COLUMN] = frame[EVENT_KEY_COLUMN].astype(int)
    if int(frame[EVENT_KEY_COLUMN].max()) >= int(target_event_key):
        raise ValueError("history must contain only events strictly earlier than target_event_key")
    history_event_keys = set(frame[EVENT_KEY_COLUMN].astype(int).unique().tolist())
    unknown_calibration_keys = sorted(set(requested_calibration_keys) - history_event_keys)
    if unknown_calibration_keys:
        raise ValueError(
            "calibration_event_keys must be a subset of historical events: "
            f"{unknown_calibration_keys}"
        )
    # These rows currently contribute to the fitted source/residual/stage
    # models. Merely naming them as a calibration partition does not make the
    # residual intervals split conformal. Keep interval publication fail-closed
    # until callers provide a truly held-out final-predictor calibration set.
    calibration_partition_validated = False
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
        if requested_calibration_keys:
            strong_indices = [
                index
                for index, (event_key, is_weak) in enumerate(
                    zip(event_keys, event_is_weak_prior)
                )
                if int(event_key) in set(requested_calibration_keys) and not is_weak
            ]
        else:
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
    stage_probability_model = _fit_driver_conditioned_stage_model(
        stage_history,
        target_event_key=int(target_event_key),
    )
    stage_time_effects = _fit_learned_stage_time_effects(
        frame,
        calibrations=calibrations,
        residual_model=residual_model,
    )
    return AchievableBestLapModel(
        target_event_key=int(target_event_key),
        calibrations=calibrations,
        min_calibration_events=int(min_calibration_events),
        residual_model=residual_model,
        stage_calibration=stage_calibration,
        stage_probability_model=stage_probability_model,
        stage_time_effects=stage_time_effects,
        calibration_event_keys=requested_calibration_keys,
        calibration_partition_validated=calibration_partition_validated,
        model_name=model_name or (
            "shared_qualifying_latent_lap_huber_v3"
            if enable_robust_residual
            else "achievable_best_lap_rehearsal_shift_v1"
        ),
    )


def calibrate_achievable_best_lap_model(
    model: AchievableBestLapModel,
    calibration_predictions: pd.DataFrame,
) -> AchievableBestLapModel:
    """Attach held-out final-predictor residuals without changing point fit.

    ``calibration_predictions`` must contain predictions emitted by the frozen
    model plus their later labels. Calibration event keys must be disjoint from
    and later than every point-training event. Only interval residuals change;
    source shifts, stage heads, residual coefficients, and point locations stay
    byte-identical under :func:`shared_point_predictor_sha256`.
    """

    if not isinstance(model, AchievableBestLapModel):
        raise TypeError("model must be an AchievableBestLapModel")
    if not isinstance(calibration_predictions, pd.DataFrame):
        raise TypeError("calibration_predictions must be a pandas DataFrame")
    required = {
        EVENT_KEY_COLUMN,
        DRIVER_ID_COLUMN,
        REHEARSAL_SOURCE_COLUMN,
        "lap_p50",
        ACTUAL_LAP_COLUMN,
    }
    missing = sorted(required.difference(calibration_predictions.columns))
    if missing:
        raise ValueError(f"held-out interval calibration is missing columns: {missing}")
    frame = calibration_predictions.copy()
    frame[EVENT_KEY_COLUMN] = pd.to_numeric(frame[EVENT_KEY_COLUMN], errors="coerce")
    frame["lap_p50"] = _finite_seconds(frame, "lap_p50")
    frame[ACTUAL_LAP_COLUMN] = _finite_seconds(frame, ACTUAL_LAP_COLUMN)
    frame = frame.dropna(subset=[EVENT_KEY_COLUMN, "lap_p50", ACTUAL_LAP_COLUMN])
    if frame.empty:
        raise ValueError("held-out interval calibration has no finite labelled rows")
    frame[EVENT_KEY_COLUMN] = frame[EVENT_KEY_COLUMN].astype(int)
    calibration_keys = tuple(sorted(frame[EVENT_KEY_COLUMN].unique().tolist()))
    overlap = sorted(set(calibration_keys).intersection(model.training_event_keys))
    if overlap:
        raise ValueError(
            f"interval calibration events overlap point-training events: {overlap}"
        )
    if model.training_event_keys and min(calibration_keys) <= max(model.training_event_keys):
        raise ValueError("interval calibration events must follow all point-training events")
    if max(calibration_keys) >= int(model.target_event_key):
        raise ValueError("interval calibration events must precede model target_event_key")
    frame[REHEARSAL_SOURCE_COLUMN] = frame[REHEARSAL_SOURCE_COLUMN].map(_source_name)
    frame["_final_predictor_residual"] = frame[ACTUAL_LAP_COLUMN] - frame["lap_p50"]
    calibrations: dict[str, AchievableLapSourceCalibration] = {}
    for source, source_rows in frame.groupby(REHEARSAL_SOURCE_COLUMN, sort=True):
        event_keys: list[int] = []
        blocks: list[tuple[float, ...]] = []
        for event_key, event_rows in source_rows.groupby(EVENT_KEY_COLUMN, sort=True):
            values = tuple(
                float(value)
                for value in event_rows["_final_predictor_residual"].dropna().tolist()
            )
            if values:
                event_keys.append(int(event_key))
                blocks.append(values)
        calibrations[str(source)] = AchievableLapSourceCalibration(
            source=str(source),
            event_keys=tuple(event_keys),
            event_shifts_seconds=(),
            prequential_residuals_seconds=tuple(
                value for block in blocks for value in block
            ),
            event_prequential_residuals_seconds=tuple(blocks),
            event_weights=tuple(1.0 for _ in blocks),
            conformal_event_keys=tuple(event_keys),
        )
    pooled_event_keys: list[int] = []
    pooled_blocks: list[tuple[float, ...]] = []
    for event_key, event_rows in frame.groupby(EVENT_KEY_COLUMN, sort=True):
        values = tuple(
            float(value)
            for value in event_rows["_final_predictor_residual"].dropna().tolist()
        )
        if values:
            pooled_event_keys.append(int(event_key))
            pooled_blocks.append(values)
    calibrations["__all__"] = AchievableLapSourceCalibration(
        source="__all__",
        event_keys=tuple(pooled_event_keys),
        event_shifts_seconds=(),
        prequential_residuals_seconds=tuple(
            value for block in pooled_blocks for value in block
        ),
        event_prequential_residuals_seconds=tuple(pooled_blocks),
        event_weights=tuple(1.0 for _ in pooled_blocks),
        conformal_event_keys=tuple(pooled_event_keys),
    )
    if not calibrations:
        raise ValueError("held-out interval calibration has no supported rehearsal source")
    return replace(
        model,
        interval_calibrations=calibrations,
        calibration_event_keys=calibration_keys,
        calibration_partition_validated=True,
    )


def shared_point_predictor_sha256(model: AchievableBestLapModel) -> str:
    """Fingerprint the complete point predictor while excluding target/interval state."""

    if not isinstance(model, AchievableBestLapModel):
        raise TypeError("model must be an AchievableBestLapModel")
    stage_model = model.stage_probability_model
    if stage_model is not None and is_dataclass(stage_model):
        stage_model = replace(stage_model, target_event_key=None)
    canonical = replace(
        model,
        target_event_key=0,
        stage_probability_model=stage_model,
        interval_calibrations={},
        calibration_event_keys=(),
        calibration_partition_validated=False,
    )
    return hashlib.sha256(pickle.dumps(canonical, protocol=5)).hexdigest()


def build_shared_qualifying_event_forecast(
    history: pd.DataFrame,
    inference: pd.DataFrame,
    *,
    target_event_key: int,
    interval_calibration_predictions: pd.DataFrame | None = None,
) -> tuple[AchievableBestLapModel, SharedQualifyingForecast, dict[str, object]]:
    """Common frozen model/sampler path for Qualifying and Best Lap runners."""

    model = fit_achievable_best_lap_model(
        history,
        target_event_key=int(target_event_key),
        calibration_event_keys=(),
        enable_robust_residual=SHARED_QUALIFYING_ENABLE_ROBUST_RESIDUAL,
        model_name="shared_qualifying_latent_lap_v3",
    )
    if interval_calibration_predictions is not None:
        model = calibrate_achievable_best_lap_model(
            model,
            interval_calibration_predictions,
        )
    forecast = model.predict_qualifying(
        inference,
        samples=SHARED_QUALIFYING_SAMPLE_COUNT,
        seed=SHARED_QUALIFYING_SAMPLE_SEED_BASE + int(target_event_key),
        allow_diagnostic_stage_fallback=True,
    )
    numeric_events = pd.to_numeric(history.get(EVENT_KEY_COLUMN), errors="coerce")
    if WEAK_PRIOR_COLUMN in history.columns:
        weak = _probability_label(history[WEAK_PRIOR_COLUMN]).fillna(0.0).astype(bool)
    else:
        weak = pd.Series(False, index=history.index)
    partition_manifest = {
        "weak_prior_event_keys": sorted(
            int(value) for value in numeric_events.loc[weak].dropna().unique()
        ),
        "strong_same_season_point_fit_event_keys": sorted(
            int(value) for value in numeric_events.loc[~weak].dropna().unique()
        ),
        "enable_robust_residual": SHARED_QUALIFYING_ENABLE_ROBUST_RESIDUAL,
        "joint_sample_count": SHARED_QUALIFYING_SAMPLE_COUNT,
        "joint_sample_seed": SHARED_QUALIFYING_SAMPLE_SEED_BASE
        + int(target_event_key),
    }
    artifact = shared_qualifying_forecast_artifact(
        forecast,
        model=model,
        training_partition_manifest=partition_manifest,
    )
    return model, forecast, artifact


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


def _fit_driver_conditioned_stage_model(
    history: pd.DataFrame,
    *,
    target_event_key: int,
) -> Any | None:
    """Fit the same row-conditioned nested hurdles consumed by the sampler."""

    required_labels = {VALID_LAP_COLUMN, REACHED_Q2_COLUMN, REACHED_Q3_COLUMN}
    if not required_labels.issubset(history.columns):
        return None
    usable = history.dropna(subset=[EVENT_KEY_COLUMN, DRIVER_ID_COLUMN]).copy()
    if WEAK_PRIOR_COLUMN in usable.columns:
        strong = usable.loc[
            ~_probability_label(usable[WEAK_PRIOR_COLUMN]).fillna(0.0).astype(bool)
        ].copy()
        # Old seasons may set an event-balanced reliability/format prior, but
        # cannot create driver-specific Q2/Q3 pace effects for the new regime.
        if strong.empty:
            return None
        usable = strong
    if usable.empty or usable[EVENT_KEY_COLUMN].nunique() < 2:
        return None
    anchor_features = (
        ("field_relative_anchor_seconds",)
        if "field_relative_anchor_seconds" in usable.columns
        else tuple(
            column
            for column in (LATENT_POTENTIAL_ANCHOR_COLUMN, QUALITY_AWARE_ANCHOR_COLUMN)
            if column in usable.columns
        )
        or ((REHEARSAL_LAP_COLUMN,) if REHEARSAL_LAP_COLUMN in usable.columns else ())
    )
    candidate_features = (
        *anchor_features,
        *ROBUST_NUMERIC_FEATURE_ALLOWLIST,
        "push_lap_count",
        "lap_evidence_count",
        "valid_clean_lap_count",
        "deleted_potential_lap_count",
        "latent_anchor_uses_potential",
        "anchor_is_imputed",
    )
    feature_columns = tuple(
        dict.fromkeys(column for column in candidate_features if column in usable.columns)
    )
    if not feature_columns:
        return None
    try:
        # Delayed import avoids a package initialization cycle while making the
        # standalone hurdle API and shared latent model use one implementation.
        from packages.f1.models.pre_quali.classification import (
            StageProbabilityConfig,
            fit_qualifying_stage_probability_model,
        )

        return fit_qualifying_stage_probability_model(
            usable,
            config=StageProbabilityConfig(
                feature_columns=feature_columns,
                rehearsal_source_column=REHEARSAL_SOURCE_COLUMN,
                minimum_training_events=2,
            ),
            target_event_key=int(target_event_key),
        )
    except ValueError as error:
        # Missing eligible Q2/Q3 populations is a legitimate cold-start. The
        # event-balanced beta prior remains explicit instead of fabricating a
        # fitted driver model.
        if "no eligible observed rows" in str(error):
            return None
        raise


def _fit_learned_stage_time_effects(
    frame: pd.DataFrame,
    *,
    calibrations: Mapping[str, AchievableLapSourceCalibration],
    residual_model: RobustHierarchicalResidualModel,
) -> LearnedStageTimeEffects:
    required = {VALID_LAP_COLUMN, REACHED_Q2_COLUMN, REACHED_Q3_COLUMN}
    if not required.issubset(frame.columns):
        return LearnedStageTimeEffects()
    working = frame.copy()
    valid = _probability_label(working[VALID_LAP_COLUMN])
    q2 = _probability_label(working[REACHED_Q2_COLUMN])
    q3 = _probability_label(working[REACHED_Q3_COLUMN])
    if ((q2 > valid) | (q3 > q2)).fillna(False).any():
        raise ValueError("Qualifying stage labels must be nested before fitting time effects")
    working["_deepest_stage"] = np.where(
        q3.eq(1.0), 3, np.where(q2.eq(1.0), 2, np.where(valid.eq(1.0), 1, 0))
    )
    working = working.loc[working["_deepest_stage"].between(1, 3)].copy()
    if WEAK_PRIOR_COLUMN in working.columns:
        strong = working.loc[
            ~_probability_label(working[WEAK_PRIOR_COLUMN]).fillna(0.0).astype(bool)
        ]
        if not strong.empty:
            working = strong
    if working.empty:
        return LearnedStageTimeEffects()
    working["_base_location"] = [
        float(row["_training_anchor_seconds"])
        + float(calibrations[str(row[REHEARSAL_SOURCE_COLUMN])].shift_seconds)
        + float(residual_model.predict_adjustment(row))
        for _, row in working.iterrows()
    ]
    stage_time_columns = {1: Q1_LAP_COLUMN, 2: Q2_LAP_COLUMN, 3: Q3_LAP_COLUMN}
    if set(stage_time_columns.values()).issubset(working.columns):
        direct: dict[int, float] = {}
        direct_rows: dict[int, pd.DataFrame] = {}
        observed_direct: list[int] = []
        for stage, column in stage_time_columns.items():
            stage_rows = working.loc[
                pd.to_numeric(working[column], errors="coerce").between(40.0, 180.0)
            ].copy()
            if stage_rows.empty:
                direct[stage] = 0.0
                direct_rows[stage] = stage_rows
                continue
            stage_rows["_period_residual"] = (
                pd.to_numeric(stage_rows[column], errors="coerce")
                - stage_rows["_base_location"]
            )
            event_residual = stage_rows.groupby(EVENT_KEY_COLUMN)["_period_residual"].apply(
                robust_huber_location
            )
            direct[stage] = float(
                np.clip(robust_huber_location(event_residual), -2.0, 2.0)
            )
            direct_rows[stage] = stage_rows
            observed_direct.append(stage)
        if len(observed_direct) >= 2:
            sigmas = _learned_stage_residual_sigmas(
                direct_rows,
                stage_locations=direct,
                residual_column="_period_residual",
            )
            return LearnedStageTimeEffects(
                q1_only_seconds=direct[1],
                q2_only_seconds=direct[2],
                q3_seconds=direct[3],
                q1_residual_sigma_seconds=sigmas[1],
                q2_residual_sigma_seconds=sigmas[2],
                q3_residual_sigma_seconds=sigmas[3],
                event_keys=tuple(
                    sorted(working[EVENT_KEY_COLUMN].astype(int).unique().tolist())
                ),
                fitted=True,
                status="learned_event_balanced_period_specific_time_effect",
            )
    working["_stage_residual"] = (
        pd.to_numeric(working[ACTUAL_LAP_COLUMN], errors="coerce")
        - working["_base_location"]
    )
    event_stage = (
        working.groupby([EVENT_KEY_COLUMN, "_deepest_stage"], sort=True)["_stage_residual"]
        .apply(robust_huber_location)
        .reset_index()
    )
    raw: dict[int, float] = {}
    for stage in (1, 2, 3):
        values = event_stage.loc[event_stage["_deepest_stage"].eq(stage), "_stage_residual"]
        raw[stage] = robust_huber_location(values) if len(values) else 0.0
    observed_stages = sorted(event_stage["_deepest_stage"].astype(int).unique().tolist())
    if len(observed_stages) < 2:
        return LearnedStageTimeEffects(
            event_keys=tuple(sorted(working[EVENT_KEY_COLUMN].astype(int).unique().tolist())),
            status="underidentified_single_observed_stage",
        )
    stage_counts = working["_deepest_stage"].value_counts().to_dict()
    center = float(
        sum(raw[stage] * float(stage_counts.get(stage, 0)) for stage in observed_stages)
        / max(1.0, sum(float(stage_counts.get(stage, 0)) for stage in observed_stages))
    )
    centered = {
        stage: float(np.clip(raw[stage] - center, -1.5, 1.5))
        if stage in observed_stages
        else 0.0
        for stage in (1, 2, 3)
    }
    deepest_rows = {
        stage: working.loc[working["_deepest_stage"].eq(stage)].copy()
        for stage in (1, 2, 3)
    }
    sigmas = _learned_stage_residual_sigmas(
        deepest_rows,
        stage_locations=centered,
        residual_column="_stage_residual",
    )
    return LearnedStageTimeEffects(
        q1_only_seconds=centered[1],
        q2_only_seconds=centered[2],
        q3_seconds=centered[3],
        q1_residual_sigma_seconds=sigmas[1],
        q2_residual_sigma_seconds=sigmas[2],
        q3_residual_sigma_seconds=sigmas[3],
        event_keys=tuple(sorted(working[EVENT_KEY_COLUMN].astype(int).unique().tolist())),
        fitted=True,
        status="learned_event_balanced_deepest_stage_effect",
    )


def _learned_stage_residual_sigmas(
    stage_rows: Mapping[int, pd.DataFrame],
    *,
    stage_locations: Mapping[int, float],
    residual_column: str,
    shrinkage_events: float = 3.0,
    measurement_floor_seconds: float = 0.02,
) -> dict[int, float]:
    """Estimate stage dispersion with event balance and pooled shrinkage."""

    centered_parts: list[pd.DataFrame] = []
    raw_sigma: dict[int, float] = {}
    event_counts: dict[int, int] = {}
    for stage in (1, 2, 3):
        rows = stage_rows.get(stage, pd.DataFrame()).copy()
        if rows.empty or residual_column not in rows.columns:
            raw_sigma[stage] = float("nan")
            event_counts[stage] = 0
            continue
        rows["_centered_stage_residual"] = (
            pd.to_numeric(rows[residual_column], errors="coerce")
            - float(stage_locations.get(stage, 0.0))
        )
        rows = rows.dropna(
            subset=[EVENT_KEY_COLUMN, "_centered_stage_residual"]
        )
        if rows.empty:
            raw_sigma[stage] = float("nan")
            event_counts[stage] = 0
            continue
        event_scale = rows.groupby(EVENT_KEY_COLUMN)["_centered_stage_residual"].apply(
            lambda values: float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))
        )
        raw_sigma[stage] = robust_huber_location(event_scale)
        event_counts[stage] = int(len(event_scale))
        centered_parts.append(
            rows[[EVENT_KEY_COLUMN, "_centered_stage_residual"]]
        )
    if not centered_parts:
        return {stage: float("nan") for stage in (1, 2, 3)}
    pooled = pd.concat(centered_parts, ignore_index=True)
    pooled_event_scale = pooled.groupby(EVENT_KEY_COLUMN)[
        "_centered_stage_residual"
    ].apply(
        lambda values: float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))
    )
    pooled_sigma = max(
        float(measurement_floor_seconds),
        float(robust_huber_location(pooled_event_scale)),
    )
    output: dict[int, float] = {}
    for stage in (1, 2, 3):
        count = float(event_counts[stage])
        weight = count / (count + float(shrinkage_events)) if count > 0.0 else 0.0
        observed = raw_sigma[stage] if np.isfinite(raw_sigma[stage]) else pooled_sigma
        output[stage] = max(
            float(measurement_floor_seconds),
            float(weight * observed + (1.0 - weight) * pooled_sigma),
        )
    return output


def _conditional_stage_weights(
    payload: Mapping[str, float | str],
) -> tuple[float, float, float] | None:
    q2 = float(payload.get("q2_given_valid_probability", float("nan")))
    q3 = float(payload.get("q3_given_q2_probability", float("nan")))
    if not (np.isfinite(q2) and np.isfinite(q3)):
        return None
    q2 = float(np.clip(q2, 0.0, 1.0))
    q3 = float(np.clip(q3, 0.0, 1.0))
    return (1.0 - q2, q2 * (1.0 - q3), q2 * q3)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _normal_mixture_quantile(
    weights: Sequence[float],
    means: Sequence[float],
    sigma: float,
    probability: float,
) -> float:
    if not np.isfinite(sigma) or sigma <= 0.0:
        return _weighted_quantile(
            np.asarray(means, dtype=float),
            np.asarray(weights, dtype=float),
            probability,
        )
    low = float(min(means) - 8.0 * sigma)
    high = float(max(means) + 8.0 * sigma)
    total = max(float(sum(weights)), 1e-12)
    for _ in range(80):
        midpoint = (low + high) * 0.5
        cdf = sum(
            float(weight) * _normal_cdf((midpoint - float(mean)) / sigma)
            for weight, mean in zip(weights, means)
        ) / total
        if cdf < float(probability):
            low = midpoint
        else:
            high = midpoint
    return float((low + high) * 0.5)


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
    q2_slots: int | None = None,
    q3_slots: int | None = None,
    allow_diagnostic_pace_fallback: bool = False,
) -> JointLapSamples:
    """Draw legal official Qualifying classifications from one latent engine.

    Every simulation first samples valid Q1 laps, then fills the legal Q2 and
    Q3 cut sizes. Conditional hurdle probabilities determine advancement with
    a Gumbel top-k draw; stage times order the official Q3/Q2/Q1 blocks. This
    avoids counting pace both in the hurdle and an arbitrary seconds offset.
    """

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    required = {DRIVER_ID_COLUMN, "lap_p50", "valid_lap_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"joint lap sampling is missing columns: {missing}")
    if int(samples) < 1:
        raise ValueError("samples must be positive")
    drivers = tuple(predictions[DRIVER_ID_COLUMN].astype(str).tolist())
    count = len(drivers)
    if (q2_slots is None) != (q3_slots is None):
        raise ValueError("q2_slots and q3_slots must be supplied together")
    if q2_slots is None and q3_slots is None:
        if count >= 12:
            regulation = qualifying_elimination_rule(count)
            q2_slots = int(regulation.period_2_cars)
            q3_slots = int(regulation.period_3_cars)
        else:
            # Tiny synthetic fields are useful for unit contracts but are not
            # eligible for an FIA three-period session.
            q2_slots = count
            q3_slots = count
    if int(q2_slots) < 1 or int(q3_slots) < 1 or int(q3_slots) > int(q2_slots):
        raise ValueError("Qualifying stage slots must satisfy 1 <= q3_slots <= q2_slots")
    rng = np.random.default_rng(int(seed))
    p50 = pd.to_numeric(
        predictions.get("latent_lap_location_seconds", predictions["lap_p50"]),
        errors="coerce",
    ).to_numpy(dtype=float)
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
    ).to_numpy(dtype=float)
    q2_probability = (
        pd.to_numeric(predictions["q2_given_valid_probability"], errors="coerce")
        .to_numpy(dtype=float)
        if "q2_given_valid_probability" in predictions.columns
        else np.full(count, np.nan, dtype=float)
    )
    q3_probability = (
        pd.to_numeric(predictions["q3_given_q2_probability"], errors="coerce")
        .to_numpy(dtype=float)
        if "q3_given_q2_probability" in predictions.columns
        else np.full(count, np.nan, dtype=float)
    )
    hurdle_probabilities_available = bool(
        np.isfinite(valid_probability).all()
        and np.isfinite(q2_probability).all()
        and np.isfinite(q3_probability).all()
    )
    if not hurdle_probabilities_available and not allow_diagnostic_pace_fallback:
        raise ValueError(
            "joint qualifying probabilities require finite valid/Q2/Q3 hurdles; "
            "set allow_diagnostic_pace_fallback=True only for non-promotable diagnostics"
        )
    if not hurdle_probabilities_available:
        valid_probability = np.where(np.isfinite(valid_probability), valid_probability, 1.0)

    valid = rng.random((int(samples), count)) < np.clip(valid_probability, 0.0, 1.0)
    shared = rng.normal(0.0, 1.0, size=(int(samples), 1))
    independent = rng.normal(0.0, 1.0, size=(int(samples), count))
    fraction = float(np.clip(shared_session_fraction, 0.0, 0.95))
    persistent_noise = sigma * (
        fraction * shared + np.sqrt(1.0 - fraction**2) * independent
    )
    stage_effects = np.column_stack(
        [
            pd.to_numeric(
                predictions.get(f"stage_q{stage}_time_effect_seconds", 0.0),
                errors="coerce",
            )
            if f"stage_q{stage}_time_effect_seconds" in predictions.columns
            else np.zeros(count, dtype=float)
            for stage in (1, 2, 3)
        ]
    ).astype(float)
    stage_effects = np.where(np.isfinite(stage_effects), stage_effects, 0.0)
    stage_sigmas = np.column_stack(
        [
            pd.to_numeric(
                predictions.get(f"stage_q{stage}_residual_sigma_seconds", np.nan),
                errors="coerce",
            )
            if f"stage_q{stage}_residual_sigma_seconds" in predictions.columns
            else np.full(count, np.nan, dtype=float)
            for stage in (1, 2, 3)
        ]
    ).astype(float)
    stage_dispersion_available = bool(
        np.isfinite(stage_sigmas).all() and np.all(stage_sigmas >= 0.0)
    )
    if not stage_dispersion_available and not allow_diagnostic_pace_fallback:
        raise ValueError(
            "joint qualifying probabilities require learned stage residual dispersion; "
            "set allow_diagnostic_pace_fallback=True only for non-promotable diagnostics"
        )
    stage_sigmas = np.where(np.isfinite(stage_sigmas), stage_sigmas, 0.0)
    stage_specific_noise = rng.normal(
        0.0,
        stage_sigmas[np.newaxis, :, :],
        size=(int(samples), count, 3),
    )
    stage_laps = (
        p50[np.newaxis, :, np.newaxis]
        + persistent_noise[:, :, np.newaxis]
        + stage_effects[np.newaxis, :, :]
        + stage_specific_noise
    )
    stages = np.zeros((int(samples), count), dtype=np.int8)
    official_positions = np.zeros((int(samples), count), dtype=np.int16)
    stable_driver_order = np.argsort(np.asarray(drivers, dtype=str), kind="stable")
    for sample_index in range(int(samples)):
        valid_indices = np.flatnonzero(valid[sample_index] & np.isfinite(p50))
        stages[sample_index, valid_indices] = 1
        q2_count = min(int(q2_slots), len(valid_indices))
        if q2_count:
            q2_indices = _sample_stage_advancers(
                valid_indices,
                conditional_probability=q2_probability,
                fallback_lap_seconds=stage_laps[sample_index, :, 0],
                slots=q2_count,
                rng=rng,
            )
        else:
            q2_indices = np.asarray([], dtype=int)
        stages[sample_index, q2_indices] = 2
        q3_count = min(int(q3_slots), len(q2_indices))
        if q3_count:
            q3_indices = _sample_stage_advancers(
                q2_indices,
                conditional_probability=q3_probability,
                fallback_lap_seconds=stage_laps[sample_index, :, 1],
                slots=q3_count,
                rng=rng,
            )
        else:
            q3_indices = np.asarray([], dtype=int)
        stages[sample_index, q3_indices] = 3

        q3_order = q3_indices[
            np.argsort(stage_laps[sample_index, q3_indices, 2], kind="stable")
        ]
        q2_only = q2_indices[stages[sample_index, q2_indices] == 2]
        q2_order = q2_only[
            np.argsort(stage_laps[sample_index, q2_only, 1], kind="stable")
        ]
        q1_only = valid_indices[stages[sample_index, valid_indices] == 1]
        q1_order = q1_only[
            np.argsort(stage_laps[sample_index, q1_only, 0], kind="stable")
        ]
        invalid = stable_driver_order[
            ~np.isin(stable_driver_order, valid_indices, assume_unique=False)
        ]
        official_order = np.concatenate([q3_order, q2_order, q1_order, invalid])
        official_positions[sample_index, official_order] = np.arange(1, count + 1)

    available_stage = np.arange(1, 4)[np.newaxis, np.newaxis, :] <= stages[:, :, np.newaxis]
    achieved = np.where(available_stage, stage_laps, np.inf)
    lap_seconds = np.min(achieved, axis=2)
    lap_seconds[~np.isfinite(lap_seconds)] = np.nan
    lap_seconds[stages == 0] = np.nan
    stage_laps[~available_stage] = np.nan
    lap_seconds[:, ~np.isfinite(p50)] = np.nan
    valid[:, ~np.isfinite(p50)] = False
    stages[:, ~np.isfinite(p50)] = 0
    return JointLapSamples(
        driver_ids=drivers,
        lap_seconds=lap_seconds,
        valid_mask=valid,
        deepest_stage=stages,
        official_positions=official_positions,
        stage_lap_seconds=stage_laps,
        stage_advancement_status=(
            "fitted_hurdle_gumbel_top_k"
            if hurdle_probabilities_available
            else "diagnostic_pace_fallback_missing_hurdles"
        ),
        stage_time_distribution_status=(
            "learned_stage_residual_dispersion"
            if stage_dispersion_available
            else "diagnostic_missing_learned_stage_residual_dispersion"
        ),
    )


def _sample_stage_advancers(
    eligible_indices: np.ndarray,
    *,
    conditional_probability: np.ndarray,
    fallback_lap_seconds: np.ndarray,
    slots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a legal top-k without inventing a missing hurdle prior."""

    eligible = np.asarray(eligible_indices, dtype=int)
    if int(slots) >= len(eligible):
        return eligible.copy()
    probabilities = np.asarray(conditional_probability, dtype=float)[eligible]
    if np.isfinite(probabilities).all():
        utility = _log_odds(np.clip(probabilities, 1e-6, 1.0 - 1e-6))
        utility += rng.gumbel(0.0, 1.0, size=len(eligible))
        order = np.argsort(-utility, kind="stable")
    else:
        # An unavailable hurdle is explicitly backed off to the sampled stage
        # timing, not a hidden 0.75/0.50 probability constant.
        times = np.asarray(fallback_lap_seconds, dtype=float)[eligible]
        order = np.argsort(times, kind="stable")
    return eligible[order[: int(slots)]]


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
    payload: dict[str, object] = {
        DRIVER_ID_COLUMN: samples.driver_ids,
        "fastest_driver_probability": fastest_count / denominator,
        "fastest_lap_top3_probability": top3_count / denominator,
        "valid_lap_probability_sampled": np.mean(samples.valid_mask, axis=0),
        "reaches_q2_probability_sampled": np.mean(samples.deepest_stage >= 2, axis=0),
        "reaches_q3_probability_sampled": np.mean(samples.deepest_stage >= 3, axis=0),
    }
    if samples.official_positions is not None:
        positions = np.asarray(samples.official_positions, dtype=int)
        if positions.shape != lap_seconds.shape:
            raise ValueError("official position samples have inconsistent shape")
        payload["expected_qualifying_position"] = np.mean(positions, axis=0)
        payload["pole_probability"] = np.mean(positions == 1, axis=0)
        payload["top3_probability"] = np.mean(positions <= 3, axis=0)
        for position in range(1, positions.shape[1] + 1):
            payload[f"p_position_{position}"] = np.mean(positions == position, axis=0)
    else:
        payload["expected_qualifying_position"] = np.nan
        payload["pole_probability"] = fastest_count / denominator
        payload["top3_probability"] = top3_count / denominator
    return pd.DataFrame(payload)


def _log_odds(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


__all__ = [
    "ACTUAL_LAP_COLUMN",
    "JointLapSamples",
    "LearnedStageTimeEffects",
    "LATENT_POTENTIAL_ANCHOR_COLUMN",
    "QUALITY_AWARE_ANCHOR_COLUMN",
    "Q1_LAP_COLUMN",
    "Q2_LAP_COLUMN",
    "Q3_LAP_COLUMN",
    "QualifyingStageProbabilities",
    "ROBUST_NUMERIC_FEATURE_ALLOWLIST",
    "RobustHierarchicalResidualModel",
    "StageHurdleCalibration",
    "SharedQualifyingForecast",
    "SHARED_QUALIFYING_ENABLE_ROBUST_RESIDUAL",
    "SHARED_QUALIFYING_SAMPLE_COUNT",
    "SHARED_QUALIFYING_SAMPLE_SEED_BASE",
    "AchievableBestLapModel",
    "AchievableLapSourceCalibration",
    "calibrate_achievable_best_lap_model",
    "build_shared_qualifying_event_forecast",
    "fit_achievable_best_lap_model",
    "decompose_event_fastest_and_driver_gap",
    "robust_huber_location",
    "sample_joint_qualifying_laps",
    "shared_qualifying_forecast_artifact",
    "shared_point_predictor_sha256",
    "summarize_joint_lap_samples",
]
