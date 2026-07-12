"""Separate valid-lap and Qualifying-stage probability contracts.

Advancement is conditional: Q2 is fitted only among entrants with a valid Q1
classification, and Q3 only among entrants that reached Q2.  This prevents a
single opaque classifier from conflating invalid laps with lack of pace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - dependency failure path
    LogisticRegression = None


STAGE_PROBABILITY_STATUS = "regularized_logistic_not_posthoc_calibrated"

_KNOWN_STAGE_OUTCOMES = frozenset(
    {
        "target",
        "actual",
        "actual_position",
        "qualy_position",
        "qualifying_position",
        "qualy_gap_to_best",
        "qualifying_gap_to_best",
        "q1_time",
        "q2_time",
        "q3_time",
        "best_qualifying_lap_seconds",
    }
)

QUALITY_AWARE_STAGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "latent_potential_adjusted_anchor_seconds",
    "anchor_uncertainty_seconds",
    "valid_minus_potential_seconds",
    "push_lap_count",
    "lap_evidence_count",
    "valid_clean_lap_count",
    "deleted_potential_lap_count",
    "best_lap_recency_seconds",
    "best_lap_session_progress",
    "best_lap_tyre_age_laps",
    "best_lap_speed_trap",
    "traffic_or_flag_evidence",
    "tyre_evidence_complete",
    "latent_anchor_uses_potential",
    "anchor_is_imputed",
    "teammate_relative_anchor_seconds",
    "field_relative_anchor_seconds",
    "evidence_item_count",
    "evidence_coverage_rate",
)


@dataclass(frozen=True)
class StageProbabilityConfig:
    """Explicit feature/label schema for the Qualifying hurdle models."""

    feature_columns: tuple[str, ...]
    event_column: str = "event_key"
    driver_column: str = "driver_id"
    rehearsal_source_column: str = "latest_qualifying_rehearsal_source"
    valid_label_column: str = "has_valid_qualifying_lap"
    q2_label_column: str = "reached_q2"
    q3_label_column: str = "reached_q3"
    regularization_c: float = 0.35
    prior_alpha: float = 1.0
    minimum_training_events: int = 4
    max_iterations: int = 1000
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must be a non-empty explicit allowlist")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must not contain duplicates")
        if self.regularization_c <= 0.0 or self.prior_alpha <= 0.0:
            raise ValueError("regularization_c and prior_alpha must be positive")
        if self.minimum_training_events < 2:
            raise ValueError("minimum_training_events must be at least two")

    @property
    def label_columns(self) -> tuple[str, str, str]:
        return (self.valid_label_column, self.q2_label_column, self.q3_label_column)

    @property
    def required_inference_columns(self) -> tuple[str, ...]:
        return (
            self.event_column,
            self.driver_column,
            self.rehearsal_source_column,
            *self.feature_columns,
        )

    @property
    def required_training_columns(self) -> tuple[str, ...]:
        return (*self.required_inference_columns, *self.label_columns)


@dataclass(frozen=True)
class _StageDesign:
    names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def transform(self, frame: pd.DataFrame, config: StageProbabilityConfig) -> np.ndarray:
        raw, names = _raw_stage_design(frame, config)
        if names != self.names:
            raise ValueError("stage probability feature schema changed after fitting")
        filled = np.where(np.isfinite(raw), raw, self.medians[np.newaxis, :])
        return (filled - self.means[np.newaxis, :]) / self.scales[np.newaxis, :]


@dataclass(frozen=True)
class _BinaryProbabilityModel:
    estimator: Any | None
    constant_probability: float
    observed_rows: int
    positive_rows: int
    observed_event_keys: tuple[int, ...]

    def predict(self, values: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            return np.full(values.shape[0], self.constant_probability, dtype=float)
        return np.asarray(self.estimator.predict_proba(values)[:, 1], dtype=float)


@dataclass
class QualifyingStageProbabilityModel:
    """Fitted valid/Q2/Q3 models with immutable historical event evidence."""

    config: StageProbabilityConfig
    design: _StageDesign
    valid_model: _BinaryProbabilityModel
    q2_given_valid_model: _BinaryProbabilityModel
    q3_given_q2_model: _BinaryProbabilityModel
    training_event_keys: tuple[int, ...]
    target_event_key: int | None = None

    @property
    def required_inference_columns(self) -> tuple[str, ...]:
        return self.config.required_inference_columns

    def predict_event(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Emit conditional and unconditional stage probabilities for one field."""

        event_key = _single_event(frame, self.config.event_column)
        if self.target_event_key is not None and event_key != self.target_event_key:
            raise ValueError(
                f"stage model was frozen for target_event_key={self.target_event_key}, got {event_key}"
            )
        if self.training_event_keys and max(self.training_event_keys) >= event_key:
            raise ValueError("stage inference event must be later than every training event")
        if self.config.driver_column not in frame.columns:
            raise ValueError(f"missing driver column: {self.config.driver_column}")
        drivers = frame[self.config.driver_column].fillna("").astype(str)
        if drivers.str.strip().eq("").any() or drivers.duplicated().any():
            raise ValueError("stage inference driver ids must be non-empty and unique")

        values = self.design.transform(frame, self.config)
        p_valid = np.clip(self.valid_model.predict(values), 0.0, 1.0)
        p_q2_conditional = np.clip(self.q2_given_valid_model.predict(values), 0.0, 1.0)
        p_q3_conditional = np.clip(self.q3_given_q2_model.predict(values), 0.0, 1.0)
        p_q2 = p_valid * p_q2_conditional
        p_q3 = p_q2 * p_q3_conditional
        return pd.DataFrame(
            {
                self.config.event_column: [event_key] * len(frame),
                self.config.driver_column: drivers.to_numpy(),
                "p_valid_qualifying_lap": p_valid,
                "p_q2_given_valid": p_q2_conditional,
                "p_q3_given_q2": p_q3_conditional,
                "p_reaches_q2": p_q2,
                "p_reaches_q3": p_q3,
                "p_no_valid_qualifying_lap": 1.0 - p_valid,
                "probability_calibration_status": [STAGE_PROBABILITY_STATUS] * len(frame),
            },
            index=frame.index,
        )


def _validate_features(frame: pd.DataFrame, config: StageProbabilityConfig) -> None:
    missing = [column for column in config.feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing stage feature columns: {missing}")
    if config.rehearsal_source_column not in frame.columns:
        raise ValueError(f"missing rehearsal source column: {config.rehearsal_source_column}")
    labels = set(config.label_columns)
    forbidden = {
        column
        for column in config.feature_columns
        if column in labels
        or column in _KNOWN_STAGE_OUTCOMES
        or column in {config.event_column, config.driver_column}
        or column.startswith("actual_")
        or column.endswith("_target")
    }
    if forbidden:
        raise ValueError(f"outcome/identity columns are forbidden as stage features: {sorted(forbidden)}")


def _event_keys(frame: pd.DataFrame, event_column: str) -> pd.Series:
    if event_column not in frame.columns:
        raise ValueError(f"missing event column: {event_column}")
    values = pd.to_numeric(frame[event_column], errors="coerce")
    raw = values.to_numpy(dtype=float)
    if values.isna().any() or not np.isfinite(raw).all() or not np.allclose(raw, np.rint(raw)):
        raise ValueError("event keys must be finite integers")
    return pd.Series(np.rint(raw).astype(np.int64), index=frame.index)


def _single_event(frame: pd.DataFrame, event_column: str) -> int:
    values = _event_keys(frame, event_column)
    unique = values.unique().tolist()
    if len(unique) != 1:
        raise ValueError("stage predict_event accepts exactly one complete event")
    return int(unique[0])


def _raw_stage_design(
    frame: pd.DataFrame, config: StageProbabilityConfig
) -> tuple[np.ndarray, tuple[str, ...]]:
    _validate_features(frame, config)
    if config.rehearsal_source_column in frame.columns:
        source = frame[config.rehearsal_source_column].fillna("").astype(str).str.lower()
        sprint = source.str.contains("sprint", regex=False).astype(float).to_numpy()
    else:
        sprint = np.zeros(len(frame), dtype=float)
    values: list[np.ndarray] = [sprint]
    names: list[str] = ["is_sprint_rehearsal"]
    for column in config.feature_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values.extend((numeric, numeric * sprint))
        names.extend((column, f"{column}__x_sprint_rehearsal"))
    return np.column_stack(values), tuple(names)


def _fit_stage_design(
    history: pd.DataFrame, config: StageProbabilityConfig
) -> tuple[_StageDesign, np.ndarray]:
    raw, names = _raw_stage_design(history, config)
    finite = np.where(np.isfinite(raw), raw, np.nan)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(raw), raw, medians[np.newaxis, :])
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    return (
        _StageDesign(names=names, medians=medians, means=means, scales=scales),
        (filled - means[np.newaxis, :]) / scales[np.newaxis, :],
    )


def _binary_labels(values: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    if not observed.isin([0, 1]).all():
        raise ValueError(f"{label} must contain only binary 0/1 labels")
    return numeric.astype(float)


def _fit_binary_probability(
    *,
    values: np.ndarray,
    labels: pd.Series,
    eligible: pd.Series,
    events: pd.Series,
    config: StageProbabilityConfig,
) -> _BinaryProbabilityModel:
    observed = eligible.fillna(False).astype(bool) & labels.notna()
    positions = np.flatnonzero(observed.to_numpy())
    if len(positions) == 0:
        raise ValueError("conditional stage model has no eligible observed rows")
    y = labels.iloc[positions].astype(int).to_numpy()
    observed_events = events.iloc[positions].astype(int)
    positives = int(y.sum())
    constant = (positives + config.prior_alpha) / (
        len(y) + (2.0 * config.prior_alpha)
    )
    unique_events = tuple(sorted(observed_events.unique().tolist()))
    if np.unique(y).size < 2:
        estimator = None
    else:
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn is required for stage logistic models")
        weights = np.zeros(len(y), dtype=float)
        for event_key, event_positions in observed_events.groupby(observed_events, sort=False).groups.items():
            local = np.flatnonzero(observed_events.to_numpy() == int(event_key))
            weights[local] = 1.0 / float(len(local))
        weights *= float(len(y)) / float(max(1, len(unique_events)))
        estimator = LogisticRegression(
            C=float(config.regularization_c),
            solver="lbfgs",
            max_iter=int(config.max_iterations),
            random_state=int(config.random_state),
        )
        estimator.fit(values[positions], y, sample_weight=weights)
    return _BinaryProbabilityModel(
        estimator=estimator,
        constant_probability=float(constant),
        observed_rows=int(len(y)),
        positive_rows=positives,
        observed_event_keys=unique_events,
    )


def fit_qualifying_stage_probability_model(
    history: pd.DataFrame,
    *,
    config: StageProbabilityConfig,
    target_event_key: int | None = None,
) -> QualifyingStageProbabilityModel:
    """Fit three event-weighted models on strictly earlier complete events."""

    _validate_features(history, config)
    missing_labels = [column for column in config.label_columns if column not in history.columns]
    if missing_labels:
        raise ValueError(f"missing stage labels: {missing_labels}")
    events = _event_keys(history, config.event_column).reset_index(drop=True)
    unique_events = tuple(sorted(int(value) for value in events.unique().tolist()))
    if len(unique_events) < config.minimum_training_events:
        raise ValueError(
            f"stage models require {config.minimum_training_events} training events; got {len(unique_events)}"
        )
    if target_event_key is not None and max(unique_events) >= int(target_event_key):
        raise ValueError("stage history must be strictly earlier than target_event_key")

    design, values = _fit_stage_design(history, config)
    valid = _binary_labels(history[config.valid_label_column], label=config.valid_label_column).reset_index(
        drop=True
    )
    q2 = _binary_labels(history[config.q2_label_column], label=config.q2_label_column).reset_index(drop=True)
    q3 = _binary_labels(history[config.q3_label_column], label=config.q3_label_column).reset_index(drop=True)
    q2_inconsistent = valid.notna() & q2.notna() & q2.gt(valid)
    q3_inconsistent = q2.notna() & q3.notna() & q3.gt(q2)
    if q2_inconsistent.any() or q3_inconsistent.any():
        raise ValueError("Qualifying stage labels must be nested: Q3 <= Q2 <= valid lap")
    all_rows = pd.Series(True, index=valid.index)
    valid_eligible = valid.eq(1.0)
    q2_eligible = q2.eq(1.0)
    return QualifyingStageProbabilityModel(
        config=config,
        design=design,
        valid_model=_fit_binary_probability(
            values=values,
            labels=valid,
            eligible=all_rows,
            events=events,
            config=config,
        ),
        q2_given_valid_model=_fit_binary_probability(
            values=values,
            labels=q2,
            eligible=valid_eligible,
            events=events,
            config=config,
        ),
        q3_given_q2_model=_fit_binary_probability(
            values=values,
            labels=q3,
            eligible=q2_eligible,
            events=events,
            config=config,
        ),
        training_event_keys=unique_events,
        target_event_key=None if target_event_key is None else int(target_event_key),
    )


def quality_aware_stage_probability_config(**overrides: object) -> StageProbabilityConfig:
    """Build the canonical stage config for quality-aware rehearsal rows."""

    values: dict[str, object] = {
        "feature_columns": QUALITY_AWARE_STAGE_FEATURE_COLUMNS,
        "rehearsal_source_column": "rehearsal_source",
    }
    values.update(overrides)
    return StageProbabilityConfig(**values)


__all__ = [
    "QUALITY_AWARE_STAGE_FEATURE_COLUMNS",
    "QualifyingStageProbabilityModel",
    "STAGE_PROBABILITY_STATUS",
    "StageProbabilityConfig",
    "fit_qualifying_stage_probability_model",
    "quality_aware_stage_probability_config",
]
