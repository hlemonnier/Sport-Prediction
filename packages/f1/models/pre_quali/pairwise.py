"""Causal pairwise challenger for Grand Prix Qualifying classification.

The generic F1 model historically treated finishing positions as independent
regression targets.  This module instead learns within-event comparisons and
always projects the result back to one legal field permutation.  The retained
rehearsal rank remains a strong prior: the challenger may move an entrant only
within a configurable band around that baseline.

Position marginals produced here are deliberately labelled *uncalibrated*.
They are useful shadow evidence, but must not be presented as calibrated odds
until an event-disjoint calibration audit passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:  # sklearn is the production fallback when optional LTR runtimes fail.
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - exercised only by dependency doctoring
    LogisticRegression = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - sklearn normally supplies scipy
    linear_sum_assignment = None


UNCALIBRATED_JOINT_SAMPLES = "uncalibrated_joint_samples"

# Direct adapter for ``packages.f1.features.qualifying_lap``.  Categorical
# provenance remains in the output/audit surface; only causal numeric evidence
# enters this first regularized challenger.
QUALITY_AWARE_PAIRWISE_FEATURE_COLUMNS: tuple[str, ...] = (
    "latent_potential_adjusted_anchor_seconds",
    "anchor_uncertainty_seconds",
    "valid_minus_potential_seconds",
    "best_two_spread_seconds",
    "best_three_spread_seconds",
    "push_lap_count",
    "lap_evidence_count",
    "valid_clean_lap_count",
    "deleted_potential_lap_count",
    "best_lap_recency_seconds",
    "best_lap_session_progress",
    "track_evolution_seconds_per_progress",
    "best_lap_tyre_age_laps",
    "best_lap_fresh_tyre",
    "best_lap_speed_trap",
    "best_lap_is_accurate",
    "traffic_or_flag_evidence",
    "tyre_evidence_complete",
    "latent_anchor_uses_potential",
    "anchor_is_imputed",
    "teammate_relative_anchor_seconds",
    "field_relative_anchor_seconds",
    "evidence_item_count",
    "evidence_coverage_rate",
)

_KNOWN_OUTCOME_COLUMNS = frozenset(
    {
        "target",
        "actual",
        "actual_position",
        "classification_position",
        "finish_position",
        "final_position",
        "qualy_position",
        "qualifying_position",
        "q1_position",
        "q2_position",
        "q3_position",
        "q1_time",
        "q2_time",
        "q3_time",
        "qualy_gap_to_best",
        "qualifying_gap_to_best",
        "best_qualifying_lap_seconds",
        "official_qualifying_lap_seconds",
    }
)


@dataclass(frozen=True)
class PairwiseRankerConfig:
    """Configuration for the inspectable Bradley-Terry-style challenger."""

    feature_columns: tuple[str, ...]
    event_column: str = "event_key"
    driver_column: str = "driver_id"
    target_column: str = "qualy_position"
    baseline_rank_column: str = "latest_qualifying_rehearsal_rank"
    rehearsal_source_column: str = "latest_qualifying_rehearsal_source"
    max_movement: int = 3
    regularization_c: float = 0.5
    minimum_training_events: int = 4
    max_iterations: int = 1000
    random_state: int = 42

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must be an explicit non-empty allowlist")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must not contain duplicates")
        if self.max_movement < 0:
            raise ValueError("max_movement must be non-negative")
        if self.regularization_c <= 0.0:
            raise ValueError("regularization_c must be positive")
        if self.minimum_training_events < 2:
            raise ValueError("minimum_training_events must be at least two")

    @property
    def required_inference_columns(self) -> tuple[str, ...]:
        return (
            self.event_column,
            self.driver_column,
            self.baseline_rank_column,
            self.rehearsal_source_column,
            *self.feature_columns,
        )

    @property
    def required_training_columns(self) -> tuple[str, ...]:
        return (*self.required_inference_columns, self.target_column)


@dataclass(frozen=True)
class PairwiseTrainingDataset:
    """Event-pure pair differences used by the logistic ranker."""

    values: np.ndarray
    labels: np.ndarray
    sample_weight: np.ndarray
    event_keys: tuple[int, ...]
    driver_pairs: tuple[tuple[str, str], ...]
    feature_names: tuple[str, ...]
    pair_counts_by_event: dict[int, int]


@dataclass(frozen=True)
class QualifyingRankingForecast:
    """One field forecast plus explicitly non-calibrated joint marginals."""

    point_order: pd.DataFrame
    position_marginals: pd.DataFrame
    pairwise_probabilities: pd.DataFrame
    probability_calibration_status: str = UNCALIBRATED_JOINT_SAMPLES
    position_marginals_calibrated: bool = False


@dataclass(frozen=True)
class _NumericDesign:
    feature_names: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def transform(self, frame: pd.DataFrame, config: PairwiseRankerConfig) -> np.ndarray:
        raw, names = _raw_design(frame, config)
        if names != self.feature_names:
            raise ValueError("inference feature schema differs from fitted schema")
        filled = np.where(np.isfinite(raw), raw, self.medians[np.newaxis, :])
        return (filled - self.means[np.newaxis, :]) / self.scales[np.newaxis, :]


@dataclass
class PairwiseQualifyingRanker:
    """Fitted pairwise model with an event-time causality boundary."""

    config: PairwiseRankerConfig
    design: _NumericDesign
    estimator: Any
    training_event_keys: tuple[int, ...]
    training_pair_counts: dict[int, int]
    target_event_key: int | None = None

    @property
    def model_name(self) -> str:
        return "qualifying_pairwise_logistic_residual_v1"

    @property
    def required_inference_columns(self) -> tuple[str, ...]:
        return self.config.required_inference_columns

    def predict_event(
        self,
        frame: pd.DataFrame,
        *,
        samples: int = 2000,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> QualifyingRankingForecast:
        """Predict exactly one later event and return a legal permutation."""

        event_key = _single_inference_event(frame, self.config)
        if self.target_event_key is not None and event_key != self.target_event_key:
            raise ValueError(
                f"model was frozen for target_event_key={self.target_event_key}, got {event_key}"
            )
        if self.training_event_keys and max(self.training_event_keys) >= event_key:
            raise ValueError("inference event must be strictly later than every training event")
        if samples < 1:
            raise ValueError("samples must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        _validate_inference_field(frame, self.config)
        transformed = self.design.transform(frame, self.config)
        probabilities = _pairwise_probability_matrix(self.estimator, transformed)
        expected_wins = probabilities.sum(axis=1) - 0.5
        drivers = frame[self.config.driver_column].astype(str).tolist()
        baseline = _stable_baseline_positions(frame, self.config)
        point_positions = _bounded_positions(
            utility=expected_wins,
            baseline_positions=baseline,
            max_movement=self.config.max_movement,
            driver_ids=drivers,
        )

        point = frame.copy()
        point["baseline_rank_prior"] = baseline.astype(int)
        point["pairwise_expected_wins"] = expected_wins.astype(float)
        point["predicted_qualifying_position"] = point_positions.astype(int)
        point["movement_from_baseline"] = point["baseline_rank_prior"] - point[
            "predicted_qualifying_position"
        ]
        point["ranking_model"] = self.model_name
        point = point.sort_values(
            ["predicted_qualifying_position", self.config.driver_column], kind="mergesort"
        )

        rng = np.random.default_rng(int(seed))
        counts = np.zeros((len(frame), len(frame)), dtype=np.int64)
        scale = float(temperature)
        for _ in range(int(samples)):
            sampled_utility = expected_wins + rng.gumbel(size=len(frame)) * scale
            sampled_positions = _bounded_positions(
                utility=sampled_utility,
                baseline_positions=baseline,
                max_movement=self.config.max_movement,
                driver_ids=drivers,
            )
            counts[np.arange(len(frame)), sampled_positions - 1] += 1

        marginals = pd.DataFrame(
            {
                self.config.event_column: [event_key] * len(frame),
                self.config.driver_column: drivers,
                "expected_position": (counts * np.arange(1, len(frame) + 1)).sum(axis=1)
                / float(samples),
                "p_pole": counts[:, 0] / float(samples),
                "p_top3": counts[:, : min(3, len(frame))].sum(axis=1) / float(samples),
                "p_top10": counts[:, : min(10, len(frame))].sum(axis=1) / float(samples),
                "probability_calibration_status": [UNCALIBRATED_JOINT_SAMPLES] * len(frame),
                "position_marginals_calibrated": [False] * len(frame),
            },
            index=frame.index,
        )
        for position in range(1, len(frame) + 1):
            marginals[f"p_position_{position}"] = counts[:, position - 1] / float(samples)

        pair_rows: list[dict[str, object]] = []
        for left in range(len(frame)):
            for right in range(left + 1, len(frame)):
                pair_rows.append(
                    {
                        self.config.event_column: event_key,
                        "driver_a": drivers[left],
                        "driver_b": drivers[right],
                        "p_driver_a_ahead": float(probabilities[left, right]),
                        "calibration_status": UNCALIBRATED_JOINT_SAMPLES,
                    }
                )
        pairwise = pd.DataFrame(pair_rows)
        return QualifyingRankingForecast(
            point_order=point,
            position_marginals=marginals,
            pairwise_probabilities=pairwise,
        )


def _validate_feature_allowlist(config: PairwiseRankerConfig, frame: pd.DataFrame) -> None:
    missing = [column for column in config.feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing pairwise feature columns: {missing}")
    if config.rehearsal_source_column not in frame.columns:
        raise ValueError(f"missing rehearsal source column: {config.rehearsal_source_column}")
    forbidden = {
        column
        for column in config.feature_columns
        if column in _KNOWN_OUTCOME_COLUMNS
        or column == config.target_column
        or column.startswith("actual_")
        or column.endswith("_target")
    }
    forbidden.update(
        column
        for column in config.feature_columns
        if column in {config.event_column, config.driver_column}
    )
    if forbidden:
        raise ValueError(f"outcome/identity columns are forbidden as pairwise features: {sorted(forbidden)}")


def _numeric_event_keys(frame: pd.DataFrame, event_column: str) -> pd.Series:
    if event_column not in frame.columns:
        raise ValueError(f"missing event column: {event_column}")
    values = pd.to_numeric(frame[event_column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("event keys must be finite numeric values")
    rounded = np.rint(values.to_numpy(dtype=float))
    if not np.allclose(values.to_numpy(dtype=float), rounded):
        raise ValueError("event keys must be integral")
    return pd.Series(rounded.astype(np.int64), index=frame.index)


def _sprint_indicator(frame: pd.DataFrame, source_column: str) -> np.ndarray:
    if source_column not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    source = frame[source_column].fillna("").astype(str).str.lower()
    return source.str.contains("sprint", regex=False).astype(float).to_numpy()


def _stable_baseline_positions(frame: pd.DataFrame, config: PairwiseRankerConfig) -> np.ndarray:
    if config.baseline_rank_column not in frame.columns:
        raise ValueError(f"missing baseline rank column: {config.baseline_rank_column}")
    numeric = pd.to_numeric(frame[config.baseline_rank_column], errors="coerce").to_numpy(dtype=float)
    numeric = np.where(np.isfinite(numeric), numeric, np.inf)
    drivers = frame[config.driver_column].astype(str).to_numpy()
    event_values = _numeric_event_keys(frame, config.event_column).reset_index(drop=True).to_numpy()
    positions = np.zeros(len(frame), dtype=int)
    for event_key in sorted(np.unique(event_values).tolist()):
        event_positions = np.flatnonzero(event_values == event_key)
        local_order = np.lexsort(
            (
                event_positions,
                drivers[event_positions],
                numeric[event_positions],
            )
        )
        ordered_positions = event_positions[local_order]
        positions[ordered_positions] = np.arange(1, len(event_positions) + 1)
    return positions


def _raw_design(
    frame: pd.DataFrame, config: PairwiseRankerConfig
) -> tuple[np.ndarray, tuple[str, ...]]:
    _validate_feature_allowlist(config, frame)
    baseline = _stable_baseline_positions(frame, config).astype(float)
    events = _numeric_event_keys(frame, config.event_column).reset_index(drop=True)
    field_sizes = events.map(events.value_counts()).to_numpy(dtype=float)
    denominator = np.maximum(1.0, field_sizes - 1.0)
    baseline_fraction = (baseline - 1.0) / denominator
    sprint = _sprint_indicator(frame, config.rehearsal_source_column)

    values: list[np.ndarray] = [baseline_fraction, sprint]
    names: list[str] = ["baseline_rank_fraction", "is_sprint_rehearsal"]
    for column in config.feature_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values.append(numeric)
        names.append(column)
        values.append(numeric * sprint)
        names.append(f"{column}__x_sprint_rehearsal")
    return np.column_stack(values), tuple(names)


def _fit_design(frame: pd.DataFrame, config: PairwiseRankerConfig) -> tuple[_NumericDesign, np.ndarray]:
    raw, names = _raw_design(frame, config)
    finite = np.where(np.isfinite(raw), raw, np.nan)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(raw), raw, medians[np.newaxis, :])
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    design = _NumericDesign(feature_names=names, medians=medians, means=means, scales=scales)
    return design, (filled - means[np.newaxis, :]) / scales[np.newaxis, :]


def build_event_pairwise_dataset(
    frame: pd.DataFrame,
    *,
    config: PairwiseRankerConfig,
    transformed_rows: np.ndarray | None = None,
) -> PairwiseTrainingDataset:
    """Create pair differences strictly within each complete event block.

    Both orientations are emitted so the classification sample is balanced.
    Every event receives the same total sample weight regardless of field size.
    """

    _validate_feature_allowlist(config, frame)
    _validate_training_rows(frame, config)
    if config.target_column not in frame.columns:
        raise ValueError(f"missing qualifying target column: {config.target_column}")
    events = _numeric_event_keys(frame, config.event_column)
    target = pd.to_numeric(frame[config.target_column], errors="coerce")
    if transformed_rows is None:
        _, transformed_rows = _fit_design(frame, config)
    transformed_rows = np.asarray(transformed_rows, dtype=float)
    if transformed_rows.shape[0] != len(frame):
        raise ValueError("transformed_rows must have one row per input entrant")

    pair_values: list[np.ndarray] = []
    labels: list[int] = []
    pair_events: list[int] = []
    driver_pairs: list[tuple[str, str]] = []
    counts: dict[int, int] = {}
    positional_events = events.reset_index(drop=True)
    positional_target = target.reset_index(drop=True)
    positional_drivers = frame[config.driver_column].astype(str).reset_index(drop=True)

    for event_key in sorted(positional_events.unique().tolist()):
        event_positions = np.flatnonzero(positional_events.to_numpy() == event_key)
        event_positions = np.asarray(
            sorted(event_positions.tolist(), key=lambda pos: (positional_drivers.iloc[pos], pos)),
            dtype=int,
        )
        before = len(labels)
        for offset, left in enumerate(event_positions):
            left_target = positional_target.iloc[left]
            if not np.isfinite(left_target):
                continue
            for right in event_positions[offset + 1 :]:
                right_target = positional_target.iloc[right]
                if not np.isfinite(right_target) or float(left_target) == float(right_target):
                    continue
                label = int(float(left_target) < float(right_target))
                difference = transformed_rows[left] - transformed_rows[right]
                pair_values.extend((difference, -difference))
                labels.extend((label, 1 - label))
                pair_events.extend((int(event_key), int(event_key)))
                left_driver = positional_drivers.iloc[left]
                right_driver = positional_drivers.iloc[right]
                driver_pairs.extend(((left_driver, right_driver), (right_driver, left_driver)))
        counts[int(event_key)] = len(labels) - before

    if not pair_values:
        raise ValueError("no non-tied within-event qualifying pairs are available")
    event_count = len([count for count in counts.values() if count > 0])
    if event_count < config.minimum_training_events:
        raise ValueError(
            f"pairwise ranker requires {config.minimum_training_events} training events; got {event_count}"
        )

    event_array = np.asarray(pair_events, dtype=np.int64)
    weights = np.zeros(len(pair_events), dtype=float)
    for event_key, count in counts.items():
        if count > 0:
            weights[event_array == event_key] = 1.0 / float(count)
    weights *= float(len(weights)) / float(event_count)
    feature_names = _raw_design(frame, config)[1]
    return PairwiseTrainingDataset(
        values=np.vstack(pair_values),
        labels=np.asarray(labels, dtype=np.int8),
        sample_weight=weights,
        event_keys=tuple(int(value) for value in pair_events),
        driver_pairs=tuple(driver_pairs),
        feature_names=feature_names,
        pair_counts_by_event=counts,
    )


def fit_pairwise_qualifying_ranker(
    history: pd.DataFrame,
    *,
    config: PairwiseRankerConfig,
    target_event_key: int | None = None,
) -> PairwiseQualifyingRanker:
    """Fit on chronological history without silently reading the target event."""

    if LogisticRegression is None:
        raise RuntimeError("scikit-learn is required for the pairwise logistic fallback")
    _validate_training_rows(history, config)
    events = _numeric_event_keys(history, config.event_column)
    unique_events = tuple(sorted(int(value) for value in events.unique().tolist()))
    if target_event_key is not None and unique_events and max(unique_events) >= int(target_event_key):
        raise ValueError("history must contain only events strictly earlier than target_event_key")

    design, transformed = _fit_design(history, config)
    dataset = build_event_pairwise_dataset(history, config=config, transformed_rows=transformed)
    estimator = LogisticRegression(
        C=float(config.regularization_c),
        fit_intercept=False,
        solver="lbfgs",
        max_iter=int(config.max_iterations),
        random_state=int(config.random_state),
    )
    estimator.fit(dataset.values, dataset.labels, sample_weight=dataset.sample_weight)
    return PairwiseQualifyingRanker(
        config=config,
        design=design,
        estimator=estimator,
        training_event_keys=unique_events,
        training_pair_counts=dataset.pair_counts_by_event,
        target_event_key=None if target_event_key is None else int(target_event_key),
    )


def quality_aware_pairwise_config(**overrides: object) -> PairwiseRankerConfig:
    """Build the canonical config for quality-aware rehearsal feature rows."""

    values: dict[str, object] = {
        "feature_columns": QUALITY_AWARE_PAIRWISE_FEATURE_COLUMNS,
        "baseline_rank_column": "quality_aware_anchor_seconds",
        "rehearsal_source_column": "rehearsal_source",
    }
    values.update(overrides)
    return PairwiseRankerConfig(**values)


def _single_inference_event(frame: pd.DataFrame, config: PairwiseRankerConfig) -> int:
    events = _numeric_event_keys(frame, config.event_column)
    unique = events.unique().tolist()
    if len(unique) != 1:
        raise ValueError("predict_event accepts exactly one complete event field")
    return int(unique[0])


def _validate_inference_field(frame: pd.DataFrame, config: PairwiseRankerConfig) -> None:
    if frame.empty:
        raise ValueError("inference field must not be empty")
    if config.driver_column not in frame.columns:
        raise ValueError(f"missing driver column: {config.driver_column}")
    drivers = frame[config.driver_column].astype(str)
    if drivers.str.strip().eq("").any() or drivers.duplicated().any():
        raise ValueError("driver ids must be non-empty and unique within an event")
    _validate_feature_allowlist(config, frame)
    _stable_baseline_positions(frame, config)


def _validate_training_rows(frame: pd.DataFrame, config: PairwiseRankerConfig) -> None:
    if frame.empty:
        raise ValueError("pairwise training history must not be empty")
    if config.driver_column not in frame.columns:
        raise ValueError(f"missing driver column: {config.driver_column}")
    drivers = frame[config.driver_column].fillna("").astype(str)
    if drivers.str.strip().eq("").any():
        raise ValueError("training driver ids must be non-empty")
    events = _numeric_event_keys(frame, config.event_column)
    identities = pd.DataFrame({"event": events.to_numpy(), "driver": drivers.to_numpy()})
    if identities.duplicated(["event", "driver"]).any():
        raise ValueError("pairwise training requires one row per driver and event")


def _pairwise_probability_matrix(estimator: Any, transformed: np.ndarray) -> np.ndarray:
    count = transformed.shape[0]
    output = np.full((count, count), 0.5, dtype=float)
    for left in range(count):
        for right in range(left + 1, count):
            difference = (transformed[left] - transformed[right]).reshape(1, -1)
            probability = float(estimator.predict_proba(difference)[0, 1])
            probability = float(np.clip(probability, 0.0, 1.0))
            output[left, right] = probability
            output[right, left] = 1.0 - probability
    return output


def _bounded_positions(
    *,
    utility: Sequence[float],
    baseline_positions: Sequence[int],
    max_movement: int,
    driver_ids: Sequence[str],
) -> np.ndarray:
    """Solve the band-constrained assignment and return positions per row."""

    utility_array = np.asarray(utility, dtype=float)
    baseline = np.asarray(baseline_positions, dtype=int)
    count = len(baseline)
    if utility_array.shape != (count,) or len(driver_ids) != count:
        raise ValueError("utility, baseline_positions, and driver_ids must have equal length")
    if sorted(baseline.tolist()) != list(range(1, count + 1)):
        raise ValueError("baseline_positions must be a complete permutation")

    target_order = sorted(
        range(count),
        key=lambda idx: (-float(utility_array[idx]), int(baseline[idx]), str(driver_ids[idx])),
    )
    target_rank = np.empty(count, dtype=int)
    target_rank[np.asarray(target_order, dtype=int)] = np.arange(1, count + 1)
    positions = np.arange(1, count + 1, dtype=int)
    cost = (target_rank[:, np.newaxis] - positions[np.newaxis, :]).astype(float) ** 2
    allowed = np.abs(baseline[:, np.newaxis] - positions[np.newaxis, :]) <= int(max_movement)
    cost = np.where(allowed, cost, 1e9)
    # Stable perturbations make scipy and the fallback agree on exact ties.
    cost += baseline[:, np.newaxis] * 1e-8 + positions[np.newaxis, :] * 1e-10

    if linear_sum_assignment is not None:
        row_ind, column_ind = linear_sum_assignment(cost)
        assigned = np.empty(count, dtype=int)
        assigned[row_ind] = column_ind + 1
    else:  # pragma: no cover - scipy is an sklearn dependency in supported envs
        assigned = _greedy_bounded_positions(utility_array, baseline, int(max_movement), driver_ids)
    if sorted(assigned.tolist()) != list(range(1, count + 1)):
        raise RuntimeError("bounded ranking projection failed to produce a permutation")
    if np.any(np.abs(assigned - baseline) > int(max_movement)):
        raise RuntimeError("bounded ranking projection violated the movement cap")
    return assigned


def _greedy_bounded_positions(
    utility: np.ndarray,
    baseline: np.ndarray,
    max_movement: int,
    driver_ids: Sequence[str],
) -> np.ndarray:
    """Dependency-light feasible scheduler used only if scipy is unavailable."""

    count = len(baseline)
    remaining = set(range(count))
    assigned = np.zeros(count, dtype=int)
    for position in range(1, count + 1):
        eligible = [
            idx for idx in remaining if max(1, baseline[idx] - max_movement) <= position
        ]
        overdue = [
            idx for idx in eligible if min(count, baseline[idx] + max_movement) <= position
        ]
        pool = overdue or eligible
        if not pool:
            raise RuntimeError("movement band has no feasible entrant for position")
        chosen = sorted(
            pool,
            key=lambda idx: (
                min(count, baseline[idx] + max_movement) if overdue else 0,
                -float(utility[idx]),
                int(baseline[idx]),
                str(driver_ids[idx]),
            ),
        )[0]
        assigned[chosen] = position
        remaining.remove(chosen)
    return assigned


__all__ = [
    "PairwiseQualifyingRanker",
    "PairwiseRankerConfig",
    "PairwiseTrainingDataset",
    "QualifyingRankingForecast",
    "QUALITY_AWARE_PAIRWISE_FEATURE_COLUMNS",
    "UNCALIBRATED_JOINT_SAMPLES",
    "build_event_pairwise_dataset",
    "fit_pairwise_qualifying_ranker",
    "quality_aware_pairwise_config",
]
