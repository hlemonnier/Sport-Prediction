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
    max_iter: int = 800
    random_state: int = 17

    def __post_init__(self) -> None:
        if self.regularization_c <= 0.0:
            raise ValueError("regularization_c must be positive")
        if self.grid_prior_weight <= 0.0:
            raise ValueError("grid_prior_weight must be positive")
        if self.residual_weight < 0.0:
            raise ValueError("residual_weight cannot be negative")


class _NumpyLogistic:
    """Small deterministic L2-logistic fallback when sklearn is unavailable."""

    def __init__(self, *, c: float, max_iter: int) -> None:
        self.c = c
        self.max_iter = max_iter
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "_NumpyLogistic":
        weights = np.ones(len(y), dtype=float) if sample_weight is None else sample_weight.astype(float)
        coefficient = np.zeros(x.shape[1], dtype=float)
        spectral = float(np.linalg.norm(x, ord=2) ** 2) if x.size else 1.0
        lipschitz = max(1.0, 0.25 * spectral * float(weights.max()) + (1.0 / self.c))
        step = 1.0 / lipschitz
        for _ in range(self.max_iter):
            logits = np.clip(x @ coefficient, -35.0, 35.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            gradient = x.T @ ((probability - y) * weights)
            gradient += coefficient / self.c
            updated = coefficient - (step * gradient)
            if np.max(np.abs(updated - coefficient)) < 1e-8:
                coefficient = updated
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
        self.backend = "unfitted"
        self.training_events = 0
        self.training_pairs = 0
        self.training_max_as_of: str | None = None

    def _matrix(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        values = pd.DataFrame(index=frame.index)
        for column in self.feature_columns:
            values[column] = pd.to_numeric(frame.get(column), errors="coerce")
        if fit:
            self._medians = values.median(axis=0, skipna=True).fillna(0.0)
            filled = values.fillna(self._medians)
            scales = filled.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
            self._scales = scales
        filled = values.fillna(self._medians).fillna(0.0)
        return ((filled - self._medians) / self._scales).to_numpy(dtype=float)

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
        matrix = self._matrix(engineered, fit=True)
        pairs: list[np.ndarray] = []
        labels: list[float] = []
        for _, event_rows in engineered.groupby(event_col, sort=True, dropna=False):
            positions = event_rows.index.to_numpy()
            local = [engineered.index.get_loc(index) for index in positions]
            target = pd.to_numeric(event_rows[target_col], errors="coerce").to_numpy(dtype=float)
            for left in range(len(local)):
                for right in range(left + 1, len(local)):
                    if target[left] == target[right]:
                        continue
                    delta = matrix[local[left]] - matrix[local[right]]
                    label = float(target[left] < target[right])
                    pairs.extend((delta, -delta))
                    labels.extend((label, 1.0 - label))
        if not pairs:
            raise ValueError("conditional-order history produced no within-event pairs")
        x_pair = np.vstack(pairs)
        y_pair = np.asarray(labels, dtype=float)

        try:
            from sklearn.linear_model import LogisticRegression

            model: object = LogisticRegression(
                C=self.config.regularization_c,
                fit_intercept=False,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
                solver="lbfgs",
            )
            model.fit(x_pair, y_pair)
            self.backend = "sklearn_logistic_regression"
        except (ImportError, OSError):
            model = _NumpyLogistic(
                c=self.config.regularization_c,
                max_iter=self.config.max_iter,
            ).fit(x_pair, y_pair)
            self.backend = "numpy_l2_logistic_fallback"
        self._model = model
        self.training_events = int(engineered[event_col].nunique(dropna=False))
        self.training_pairs = len(y_pair)
        return self

    @property
    def coefficients(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("conditional-order ranker must be fitted")
        coefficients = np.asarray(getattr(self._model, "coef_"), dtype=float).reshape(-1)
        return dict(zip(self.feature_columns, coefficients.tolist(), strict=True))

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
        grid_prior = pd.to_numeric(
            engineered["race_grid_prior_score"], errors="coerce"
        ).fillna(-1.0).to_numpy(dtype=float)
        score = (
            self.config.grid_prior_weight * grid_prior
            + self.config.residual_weight * residual
        )
        result = pd.DataFrame(index=frame.index)
        result["driver_id"] = engineered.get(
            "driver_id", pd.Series(engineered.index.astype(str), index=engineered.index)
        ).astype(str)
        result["conditional_order_score"] = score
        result["conditional_order_grid_component"] = self.config.grid_prior_weight * grid_prior
        result["conditional_order_residual_component"] = self.config.residual_weight * residual
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
