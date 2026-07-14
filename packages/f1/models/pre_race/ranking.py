"""Regularized Bradley-Terry conditional running-order model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from packages.f1.features.race import (
    RACE_ORDER_FEATURE_COLUMNS,
    engineer_survival_aware_race_features,
)
from packages.f1.models.pre_race.status import TerminalStatus, reason_code_terminal_status


@dataclass(frozen=True)
class ConditionalOrderConfig:
    """Controls the fixed grid prior and learned residual movement."""

    regularization_c: float = 0.5
    grid_prior_weight: float = 2.0
    residual_weight: float = 0.45
    cold_start_event_k: float = 8.0
    coefficient_bound: float = 8.0
    max_iter: int = 800
    random_state: int = 17
    include_missing_indicators: bool = True

    def __post_init__(self) -> None:
        if self.regularization_c <= 0.0:
            raise ValueError("regularization_c must be positive")
        if self.grid_prior_weight <= 0.0:
            raise ValueError("grid_prior_weight must be positive")
        if self.residual_weight < 0.0:
            raise ValueError("residual_weight cannot be negative")
        if not np.isfinite(float(self.cold_start_event_k)) or float(
            self.cold_start_event_k
        ) <= 0.0:
            raise ValueError("cold_start_event_k must be finite and positive")
        if not np.isfinite(float(self.coefficient_bound)) or float(
            self.coefficient_bound
        ) <= 0.0:
            raise ValueError("coefficient_bound must be finite and positive")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be positive")


class _BoundedOffsetLogistic:
    """Deterministic projected-gradient logistic model with a fixed offset."""

    def __init__(self, *, c: float, max_iter: int, coefficient_bound: float) -> None:
        self.c = c
        self.max_iter = max_iter
        self.coefficient_bound = coefficient_bound
        self.coef_: np.ndarray | None = None
        self.n_iter_ = 0
        self.converged_ = False

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        offset: np.ndarray,
        sample_weight: np.ndarray,
    ) -> "_BoundedOffsetLogistic":
        design = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float)
        fixed_offset = np.asarray(offset, dtype=float)
        weights = np.asarray(sample_weight, dtype=float)
        if design.ndim != 2 or design.shape[0] == 0:
            raise ValueError("pairwise design must be a non-empty matrix")
        if any(
            values.shape != (design.shape[0],)
            for values in (target, fixed_offset, weights)
        ):
            raise ValueError("targets, offsets, and weights must align with pairs")
        if (
            not np.isfinite(design).all()
            or not np.isfinite(target).all()
            or not np.isfinite(fixed_offset).all()
            or not np.isfinite(weights).all()
            or (weights <= 0.0).any()
            or ((target != 0.0) & (target != 1.0)).any()
        ):
            raise ValueError("pairwise optimizer inputs must be finite and valid")
        coefficient = np.zeros(x.shape[1], dtype=float)
        weighted_design = design * np.sqrt(weights)[:, None]
        spectral = float(np.linalg.norm(weighted_design, ord=2) ** 2)
        lipschitz = max(1.0, 0.25 * spectral + (1.0 / self.c))
        step = 1.0 / lipschitz
        for iteration in range(self.max_iter):
            logits = np.clip(fixed_offset + design @ coefficient, -35.0, 35.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            gradient = design.T @ ((probability - target) * weights)
            gradient += coefficient / self.c
            updated = np.clip(
                coefficient - (step * gradient),
                -float(self.coefficient_bound),
                float(self.coefficient_bound),
            )
            self.n_iter_ = iteration + 1
            if np.max(np.abs(updated - coefficient)) < 1e-8:
                coefficient = updated
                self.converged_ = True
                break
            coefficient = updated
        self.coef_ = coefficient.reshape(1, -1)
        return self


class BradleyTerryOrderRanker:
    """Pairwise residual ranker with an explicit strong signed grid prior.

    Training pairs are generated inside complete events only; there is no
    random driver-row split.  Positive scores mean a driver is expected to
    finish ahead.  Sampling these utilities with Gumbel noise produces the
    Plackett-Luce conditional order used by the joint simulator.
    """

    def __init__(
        self,
        config: ConditionalOrderConfig | None = None,
        *,
        feature_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.config = config or ConditionalOrderConfig()
        default_residuals = tuple(
            column
            for column in RACE_ORDER_FEATURE_COLUMNS
            if column != "race_grid_prior_score"
        )
        self.feature_columns = feature_columns or default_residuals
        self._medians = pd.Series(dtype=float)
        self._scales = pd.Series(dtype=float)
        self._model: object | None = None
        self._design_columns: tuple[str, ...] = ()
        self.backend = "unfitted"
        self.training_events = 0
        self.training_pairs = 0
        self.training_pair_weight_sum = 0.0
        self.training_event_pair_counts: dict[str, int] = {}
        self.training_event_pair_weight_sums: dict[str, float] = {}
        self.training_grid_offset_mean_abs = 0.0
        self.training_max_as_of: str | None = None

    def _matrix(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        values = pd.DataFrame(index=frame.index)
        for column in self.feature_columns:
            source = (
                frame[column]
                if column in frame.columns
                else pd.Series(np.nan, index=frame.index, dtype=float)
            )
            numeric = pd.to_numeric(source, errors="coerce")
            values[column] = numeric
            if self.config.include_missing_indicators:
                values[f"{column}__missing"] = numeric.isna().astype(float)
        if fit:
            self._design_columns = tuple(values.columns)
            self._medians = values.median(axis=0, skipna=True).fillna(0.0)
            filled = values.fillna(self._medians)
            scales = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
            self._scales = scales
        filled = values.fillna(self._medians).fillna(0.0)
        return ((filled - self._medians) / self._scales).to_numpy(dtype=float)

    def fit_grid_prior_only(self, frame: pd.DataFrame) -> "BradleyTerryOrderRanker":
        """Initialize a zero-residual model when no same-regime event exists."""

        if frame.empty:
            raise ValueError("grid-prior initialization requires a non-empty roster")
        engineered = engineer_survival_aware_race_features(frame.reset_index(drop=True))
        matrix = self._matrix(engineered, fit=True)
        model = _BoundedOffsetLogistic(
            c=self.config.regularization_c,
            max_iter=1,
            coefficient_bound=self.config.coefficient_bound,
        )
        model.coef_ = np.zeros((1, matrix.shape[1]), dtype=float)
        self._model = model
        self.backend = "fixed_grid_prior_no_same_regime_pairs"
        self.training_events = 0
        self.training_pairs = 0
        self.training_pair_weight_sum = 0.0
        self.training_event_pair_counts = {}
        self.training_event_pair_weight_sums = {}
        self.training_grid_offset_mean_abs = 0.0
        self.training_max_as_of = None
        return self

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        event_col: str = "event_key",
        target_col: str = "finish_position",
        terminal_status_col: str = "terminal_status",
        event_as_of_col: str = "event_as_of",
        cutoff: object | None = None,
    ) -> "BradleyTerryOrderRanker":
        if frame.empty:
            raise ValueError("conditional-order ranker requires historical rows")
        if event_col not in frame.columns or target_col not in frame.columns:
            raise ValueError(f"ranker requires {event_col!r} and {target_col!r}")
        rows = frame.copy().reset_index(drop=True)
        if cutoff is not None:
            if event_as_of_col not in rows.columns:
                raise ValueError("cutoff requires event_as_of for causal filtering")
            event_times = pd.to_datetime(rows[event_as_of_col], errors="coerce", utc=True)
            cutoff_time = pd.to_datetime(cutoff, errors="coerce", utc=True)
            if pd.isna(cutoff_time) or event_times.isna().any():
                raise ValueError("ranker timestamps must be valid and timezone-aware")
            rows = rows.loc[event_times < cutoff_time].copy()
            event_times = event_times.loc[rows.index]
            if rows.empty:
                raise ValueError("no conditional-order rows precede cutoff")
            self.training_max_as_of = event_times.max().isoformat().replace("+00:00", "Z")
        elif event_as_of_col in rows.columns:
            event_times = pd.to_datetime(rows[event_as_of_col], errors="coerce", utc=True)
            if event_times.isna().any():
                raise ValueError("event_as_of contains invalid timestamps")
            self.training_max_as_of = event_times.max().isoformat().replace("+00:00", "Z")

        # Preserve the complete event field before removing terminal cars.  The
        # fixed grid offset must use the same denominator as inference on the
        # complete roster, not the smaller classified-finisher subset.
        field_size_column = "__conditional_order_full_event_field_size"
        rows[field_size_column] = rows.groupby(
            event_col, sort=False, dropna=False
        )[event_col].transform("size")

        if terminal_status_col in rows.columns:
            status = rows[terminal_status_col].map(reason_code_terminal_status)
            # Conditional running order learns from classified finishers only;
            # status and retirement timing are owned by the hazard/simulator.
            rows = rows.loc[status.eq(TerminalStatus.CLASSIFIED_FINISH)].copy()
        rows[target_col] = pd.to_numeric(rows[target_col], errors="coerce")
        rows = rows.loc[rows[target_col].notna()].copy()
        if rows.empty:
            raise ValueError("no classified finishers available for conditional-order fit")

        engineered = engineer_survival_aware_race_features(rows)
        full_field_size = pd.to_numeric(
            engineered[field_size_column], errors="coerce"
        ).clip(lower=1.0)
        raw_grid = pd.to_numeric(
            engineered.get(
                "grid_position",
                pd.Series(np.nan, index=engineered.index, dtype=float),
            ),
            errors="coerce",
        )
        grid_status = engineered.get(
            "grid_status", pd.Series("", index=engineered.index)
        ).astype(str).str.lower()
        raw_pit_lane = engineered.get("grid_pit_lane_start")
        if raw_pit_lane is None:
            pit_lane = grid_status.isin({"pit_lane", "started_pit_lane"})
        else:
            pit_lane = pd.Series(raw_pit_lane, index=engineered.index).fillna(
                False
            ).astype(bool)
        grid_for_score = raw_grid.where(
            raw_grid.notna(),
            np.where(pit_lane, full_field_size + 1.0, np.nan),
        )
        engineered["race_grid_prior_score"] = -(
            grid_for_score - 1.0
        ) / full_field_size
        mobility = pd.to_numeric(
            engineered["race_circuit_mobility"], errors="coerce"
        ).clip(lower=0.0, upper=1.0)
        engineered["race_grid_mobility_score"] = engineered[
            "race_grid_prior_score"
        ] * (1.0 - mobility)
        matrix = self._matrix(engineered, fit=True)
        grid_prior = pd.to_numeric(
            engineered["race_grid_prior_score"], errors="coerce"
        ).fillna(-1.0).to_numpy(dtype=float)
        pairs: list[np.ndarray] = []
        labels: list[float] = []
        offsets: list[float] = []
        pair_weights: list[float] = []
        event_pair_counts: dict[str, int] = {}
        event_pair_weight_sums: dict[str, float] = {}
        for event_key, event_rows in engineered.groupby(
            event_col, sort=True, dropna=False
        ):
            local = engineered.index.get_indexer(event_rows.index)
            target = pd.to_numeric(event_rows[target_col], errors="coerce").to_numpy(dtype=float)
            event_pairs: list[tuple[np.ndarray, float, float]] = []
            for left in range(len(local)):
                for right in range(left + 1, len(local)):
                    if target[left] == target[right]:
                        continue
                    delta = matrix[local[left]] - matrix[local[right]]
                    label = float(target[left] < target[right])
                    grid_offset = float(self.config.grid_prior_weight) * (
                        grid_prior[local[left]] - grid_prior[local[right]]
                    )
                    event_pairs.append((delta, label, grid_offset))
            if not event_pairs:
                continue
            event_key_text = str(event_key)
            event_pair_counts[event_key_text] = len(event_pairs)
            event_pair_weight_sums[event_key_text] = 0.0
            event_weight = 1.0 / float(len(event_pairs))
            for delta, label, grid_offset in event_pairs:
                pairs.append(delta)
                labels.append(label)
                offsets.append(grid_offset)
                pair_weights.append(event_weight)
                event_pair_weight_sums[event_key_text] += event_weight
        if not pairs:
            raise ValueError("conditional-order history produced no within-event pairs")
        x_pair = np.vstack(pairs)
        y_pair = np.asarray(labels, dtype=float)
        model = _BoundedOffsetLogistic(
            c=self.config.regularization_c,
            max_iter=self.config.max_iter,
            coefficient_bound=self.config.coefficient_bound,
        ).fit(
            x_pair,
            y_pair,
            offset=np.asarray(offsets, dtype=float),
            sample_weight=np.asarray(pair_weights, dtype=float),
        )
        self.backend = "deterministic_bounded_grid_offset_logistic"
        self._model = model
        self.training_events = len(event_pair_counts)
        self.training_pairs = len(y_pair)
        self.training_pair_weight_sum = float(np.sum(pair_weights))
        self.training_event_pair_counts = event_pair_counts
        self.training_event_pair_weight_sums = event_pair_weight_sums
        self.training_grid_offset_mean_abs = float(
            np.mean(np.abs(np.asarray(offsets, dtype=float)))
        )
        return self

    @property
    def coefficients(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("conditional-order ranker must be fitted")
        coefficients = np.asarray(getattr(self._model, "coef_"), dtype=float).reshape(-1)
        if len(self._design_columns) != len(coefficients):
            raise RuntimeError("conditional-order coefficient schema is inconsistent")
        return dict(zip(self._design_columns, coefficients.tolist()))

    @property
    def effective_residual_weight(self) -> float:
        """Apply an explicit empirical-Bayes-style same-season cold-start shrinkage."""

        event_count = float(self.training_events)
        shrinkage = event_count / (event_count + float(self.config.cold_start_event_k))
        return float(self.config.residual_weight) * shrinkage

    def score(
        self,
        frame: pd.DataFrame,
        *,
        prediction_as_of: object | None = None,
        feature_as_of_col: str = "feature_as_of",
    ) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("conditional-order ranker must be fitted")
        if frame.empty:
            return pd.DataFrame(index=frame.index)
        if prediction_as_of is not None:
            cutoff = pd.to_datetime(prediction_as_of, errors="coerce", utc=True)
            if pd.isna(cutoff):
                raise ValueError("prediction_as_of must be a valid timestamp")
            if self.training_max_as_of is not None:
                training_max = pd.to_datetime(
                    self.training_max_as_of, errors="coerce", utc=True
                )
                if pd.isna(training_max) or training_max >= cutoff:
                    raise ValueError("order model training evidence is not strictly pre-cutoff")
            if feature_as_of_col in frame.columns:
                feature_times = pd.to_datetime(
                    frame[feature_as_of_col], errors="coerce", utc=True
                )
                if feature_times.isna().any() or (feature_times > cutoff).any():
                    raise ValueError("order features contain invalid or post-cutoff evidence")
        engineered = engineer_survival_aware_race_features(frame)
        matrix = self._matrix(engineered, fit=False)
        coefficients = np.asarray(getattr(self._model, "coef_"), dtype=float).reshape(-1)
        residual = matrix @ coefficients
        effective_residual_weight = self.effective_residual_weight
        grid_prior = pd.to_numeric(
            engineered["race_grid_prior_score"], errors="coerce"
        ).fillna(-1.0).to_numpy(dtype=float)
        score = (
            self.config.grid_prior_weight * grid_prior
            + effective_residual_weight * residual
        )
        result = pd.DataFrame(index=frame.index)
        result["driver_id"] = engineered.get(
            "driver_id", pd.Series(engineered.index.astype(str), index=engineered.index)
        ).astype(str)
        result["conditional_order_score"] = score
        result["conditional_order_grid_component"] = self.config.grid_prior_weight * grid_prior
        result["conditional_order_residual_raw"] = residual
        result["conditional_order_residual_component"] = (
            effective_residual_weight * residual
        )
        result["conditional_order_configured_residual_weight"] = float(
            self.config.residual_weight
        )
        result["conditional_order_effective_residual_weight"] = (
            effective_residual_weight
        )
        result["conditional_order_cold_start_event_k"] = float(
            self.config.cold_start_event_k
        )
        result["conditional_order_backend"] = self.backend
        result["conditional_order_training_events"] = self.training_events
        result["conditional_order_training_pairs"] = self.training_pairs
        result["conditional_order_training_max_as_of"] = self.training_max_as_of
        raw_grid = (
            engineered["grid_position"]
            if "grid_position" in engineered.columns
            else pd.Series(np.nan, index=engineered.index, dtype=float)
        )
        order = result.assign(
            _grid=pd.to_numeric(raw_grid, errors="coerce").fillna(np.inf)
        ).sort_values(
            ["conditional_order_score", "_grid", "driver_id"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        rank_by_index = pd.Series(np.arange(1, len(order) + 1), index=order.index)
        result["conditional_order_rank"] = rank_by_index.reindex(result.index).astype(int)
        return result

    def pairwise_probability(self, left: pd.Series, right: pd.Series) -> float:
        pair = pd.DataFrame([left, right])
        scored = self.score(pair)["conditional_order_score"].to_numpy(dtype=float)
        return float(1.0 / (1.0 + np.exp(-np.clip(scored[0] - scored[1], -35.0, 35.0))))


__all__ = ["BradleyTerryOrderRanker", "ConditionalOrderConfig"]
