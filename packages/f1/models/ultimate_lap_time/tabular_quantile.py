"""Tabular quantile challenger for Ultimate Lap-Time."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.datasets import TARGET_AND_PREDICTION_COLUMNS
from packages.f1.models.ultimate_lap_time.schemas import UltimateLapTelemetryExample


QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.90)
PREDICTION_COLUMNS: tuple[str, ...] = ("lap_p05", "lap_p50", "lap_p90")
DEFAULT_EXCLUDED_FEATURE_COLUMNS: frozenset[str] = frozenset(
    set(TARGET_AND_PREDICTION_COLUMNS)
    | {
        "split_key",
        "split_name",
        "fold",
        "lap_number",
        "source",
        "channels",
        "distance_bins",
    }
)


def _as_dataframe(data: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[UltimateLapTelemetryExample]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, UltimateLapTelemetryExample):
            rows.append(item.as_flat_record())
        else:
            rows.append(dict(item))
    return pd.DataFrame(rows)


def _find_target_column(frame: pd.DataFrame, target_column: str) -> str:
    candidates = (target_column, "lap_time_seconds", "lap_duration", "LapTime", "lap_time", "duration")
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"training data is missing target column from {candidates}")


def _finite_target(frame: pd.DataFrame, target_column: str) -> pd.Series:
    column = _find_target_column(frame, target_column)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values


@dataclass(frozen=True)
class TabularQuantileConfig:
    """Training controls for the tabular quantile challenger."""

    backend: str = "auto"
    quantiles: tuple[float, ...] = QUANTILES
    feature_columns: tuple[str, ...] | None = None
    target_column: str = "lap_time_seconds"
    random_state: int = 42
    min_rows_for_boosting: int = 8
    n_estimators: int = 160
    learning_rate: float = 0.04
    max_depth: int = 3
    max_categories_per_feature: int = 64


@dataclass
class TabularFeatureEncoder:
    """Deterministic dense encoder for mixed tabular F1 context features."""

    numeric_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    numeric_medians: dict[str, float] | None = None
    categories: dict[str, tuple[str, ...]] | None = None
    feature_names_out: tuple[str, ...] = ()

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str] | None,
        max_categories_per_feature: int,
    ) -> "TabularFeatureEncoder":
        if feature_columns is None:
            columns = [
                str(column)
                for column in frame.columns
                if str(column) not in DEFAULT_EXCLUDED_FEATURE_COLUMNS and not str(column).startswith("predicted_")
            ]
        else:
            columns = [str(column) for column in feature_columns]
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ValueError(f"training data is missing feature columns: {missing}")

        numeric: list[str] = []
        categorical: list[str] = []
        medians: dict[str, float] = {}
        categories: dict[str, tuple[str, ...]] = {}
        for column in columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                values = pd.to_numeric(series, errors="coerce")
                finite = values[np.isfinite(values)]
                medians[column] = float(finite.median()) if len(finite) else 0.0
                numeric.append(column)
            else:
                normalized = series.astype("string").fillna("__missing__")
                counts = normalized.value_counts(dropna=False)
                kept = tuple(str(value) for value in counts.head(max(1, int(max_categories_per_feature))).index)
                categories[column] = kept
                categorical.append(column)

        names: list[str] = []
        names.extend(numeric)
        for column in categorical:
            names.extend([f"{column}={category}" for category in categories[column]])
        if not names:
            names.append("__intercept__")
        self.numeric_columns = tuple(numeric)
        self.categorical_columns = tuple(categorical)
        self.numeric_medians = medians
        self.categories = categories
        self.feature_names_out = tuple(names)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.numeric_medians is None or self.categories is None:
            raise RuntimeError("TabularFeatureEncoder is not fitted")
        parts: list[np.ndarray] = []
        for column in self.numeric_columns:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            else:
                values = np.full(len(frame), np.nan, dtype=float)
            values = np.where(np.isfinite(values), values, self.numeric_medians[column])
            parts.append(values.reshape(-1, 1))

        for column in self.categorical_columns:
            if column in frame.columns:
                values = frame[column].astype("string").fillna("__missing__").astype(str).to_numpy()
            else:
                values = np.full(len(frame), "__missing__", dtype=object)
            for category in self.categories[column]:
                parts.append((values == category).astype(float).reshape(-1, 1))

        if not parts:
            return np.ones((len(frame), 1), dtype=float)
        matrix = np.hstack(parts).astype(np.float32, copy=False)
        if matrix.ndim != 2:
            raise ValueError("encoded feature matrix must be two-dimensional")
        return matrix


@dataclass
class _EmpiricalQuantileRegressor:
    quantile: float
    value: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_EmpiricalQuantileRegressor":
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            raise ValueError("cannot fit empirical quantile regressor with no finite targets")
        self.value = float(np.quantile(finite, self.quantile))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.value is None:
            raise RuntimeError("empirical quantile regressor is not fitted")
        return np.full(X.shape[0], self.value, dtype=float)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _backend_candidates(config: TabularQuantileConfig, row_count: int) -> tuple[str, ...]:
    requested = config.backend.strip().lower()
    if row_count < int(config.min_rows_for_boosting):
        return ("empirical",)
    if requested != "auto":
        return (requested, "empirical")
    return ("lightgbm", "xgboost", "sklearn_hist", "sklearn_gbr", "empirical")


def _make_estimator(backend: str, quantile: float, config: TabularQuantileConfig) -> Any:
    if backend == "lightgbm":
        if not _module_available("lightgbm"):
            raise ImportError("LightGBM is not installed")
        from lightgbm import LGBMRegressor  # type: ignore

        return LGBMRegressor(
            objective="quantile",
            alpha=float(quantile),
            n_estimators=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth),
            random_state=int(config.random_state),
            verbosity=-1,
        )

    if backend == "xgboost":
        if not _module_available("xgboost"):
            raise ImportError("XGBoost is not installed")
        from xgboost import XGBRegressor  # type: ignore

        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=float(quantile),
            n_estimators=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth),
            random_state=int(config.random_state),
            tree_method="hist",
            verbosity=0,
        )

    if backend == "sklearn_hist":
        if not _module_available("sklearn"):
            raise ImportError("scikit-learn is not installed")
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(quantile),
            max_iter=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_leaf_nodes=31,
            l2_regularization=0.0,
            random_state=int(config.random_state),
        )

    if backend == "sklearn_gbr":
        if not _module_available("sklearn"):
            raise ImportError("scikit-learn is not installed")
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            loss="quantile",
            alpha=float(quantile),
            n_estimators=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth),
            random_state=int(config.random_state),
        )

    if backend == "empirical":
        return _EmpiricalQuantileRegressor(float(quantile))

    raise ValueError(f"unknown tabular quantile backend: {backend}")


@dataclass
class TabularQuantileLapTimeModel:
    """Fitted tabular quantile model with graceful optional backends."""

    config: TabularQuantileConfig
    backend_name: str
    encoder: TabularFeatureEncoder
    models: dict[float, Any]
    target_column: str
    training_summary: dict[str, Any]

    def predict(self, records: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[UltimateLapTelemetryExample]) -> pd.DataFrame:
        frame = _as_dataframe(records)
        if frame.empty:
            return pd.DataFrame(columns=[*PREDICTION_COLUMNS, "model"])
        X = self.encoder.transform(frame)
        predictions = []
        for quantile in self.config.quantiles:
            model = self.models[float(quantile)]
            predictions.append(np.asarray(model.predict(X), dtype=float))
        matrix = np.vstack(predictions).T
        matrix = np.maximum.accumulate(matrix, axis=1)
        output = pd.DataFrame(matrix, columns=PREDICTION_COLUMNS[: len(self.config.quantiles)], index=frame.index)
        for column in PREDICTION_COLUMNS:
            if column not in output.columns:
                output[column] = np.nan
        output = output.loc[:, list(PREDICTION_COLUMNS)]
        output["model"] = f"ultimate_lap_time_tabular_quantile_{self.backend_name}"
        return output


def fit_tabular_quantile_model(
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[UltimateLapTelemetryExample],
    *,
    config: TabularQuantileConfig | None = None,
) -> TabularQuantileLapTimeModel:
    """Fit a p05/p50/p90 tabular quantile challenger."""

    cfg = config or TabularQuantileConfig()
    if tuple(cfg.quantiles) != QUANTILES:
        raise ValueError("the current evaluation contract expects quantiles (0.05, 0.50, 0.90)")

    frame = _as_dataframe(records)
    if frame.empty:
        raise ValueError("records must contain at least one training row")
    y = _finite_target(frame, cfg.target_column).to_numpy(dtype=float)
    finite_mask = np.isfinite(y)
    if not finite_mask.any():
        raise ValueError("records contain no finite lap-time targets")
    frame = frame.loc[finite_mask].reset_index(drop=True)
    y = y[finite_mask]

    encoder = TabularFeatureEncoder().fit(
        frame,
        feature_columns=cfg.feature_columns,
        max_categories_per_feature=int(cfg.max_categories_per_feature),
    )
    X = encoder.transform(frame)

    notes: list[str] = []
    last_error: Exception | None = None
    for backend in _backend_candidates(cfg, len(frame)):
        models: dict[float, Any] = {}
        try:
            for quantile in cfg.quantiles:
                estimator = _make_estimator(backend, float(quantile), cfg)
                estimator.fit(X, y)
                models[float(quantile)] = estimator
            return TabularQuantileLapTimeModel(
                config=cfg,
                backend_name=backend,
                encoder=encoder,
                models=models,
                target_column=_find_target_column(frame, cfg.target_column),
                training_summary={
                    "rows_seen": int(len(_as_dataframe(records))),
                    "rows_used": int(len(frame)),
                    "backend": backend,
                    "feature_count": int(X.shape[1]),
                    "feature_names": list(encoder.feature_names_out),
                    "notes": notes,
                },
            )
        except Exception as exc:  # pragma: no cover - depends on optional backend versions
            last_error = exc
            notes.append(f"{backend} unavailable or failed: {exc}")
            continue

    raise RuntimeError(f"could not fit any tabular quantile backend: {last_error}")


def predict_tabular_quantiles(
    model: TabularQuantileLapTimeModel,
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[UltimateLapTelemetryExample],
) -> pd.DataFrame:
    """Predict p05/p50/p90 lap-time quantiles from a fitted tabular model."""

    if not isinstance(model, TabularQuantileLapTimeModel):
        raise TypeError("model must be a TabularQuantileLapTimeModel")
    return model.predict(records)


__all__ = [
    "PREDICTION_COLUMNS",
    "QUANTILES",
    "TabularFeatureEncoder",
    "TabularQuantileConfig",
    "TabularQuantileLapTimeModel",
    "fit_tabular_quantile_model",
    "predict_tabular_quantiles",
]
