"""Tabular quantile challenger for Ultimate Lap-Time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.ultimate_lap_time.achievable import FORBIDDEN_INFERENCE_COLUMNS
from packages.f1.models.ultimate_lap_time.datasets import TARGET_AND_PREDICTION_COLUMNS
from packages.f1.models.ultimate_lap_time.schemas import UltimateLapTelemetryExample
from packages.f1.orchestration.model_runtime import inspect_optional_model_runtime


QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.90)
PREDICTION_COLUMNS: tuple[str, ...] = ("lap_p05", "lap_p50", "lap_p90")
PREDICTION_COLUMN_BY_QUANTILE: dict[float, str] = {
    0.05: "lap_p05",
    0.50: "lap_p50",
    0.90: "lap_p90",
}
TABULAR_QUANTILE_BACKENDS: tuple[str, ...] = (
    "auto",
    "lightgbm",
    "xgboost",
    "sklearn_hist",
    "sklearn_gbr",
    "empirical",
)
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
CAUSAL_FEATURE_FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    set(TARGET_AND_PREDICTION_COLUMNS)
    | set(FORBIDDEN_INFERENCE_COLUMNS)
    | {
        "event_key",
        "event_id",
        "meeting_key",
        "weekend_key",
        "driver_id",
        "driver_number",
        "lap_id",
        "row_id",
        "season",
        "event_as_of",
        "feature_as_of",
        "target_as_of",
        "qualy_position",
        "actual_qualifying_position",
        "finish_position",
        "race_position",
        "has_valid_qualifying_lap",
        "reached_q2",
        "reached_q3",
        "lap_residual_seconds",
        "split_key",
        "split_name",
        "fold",
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
    season_column: str = "season"
    target_season: int | None = None
    same_season_only: bool = True
    event_time_column: str = "event_as_of"
    fit_before: str | None = None
    random_state: int = 42
    min_rows_for_boosting: int = 8
    n_estimators: int = 160
    learning_rate: float = 0.04
    max_depth: int = 3
    max_categories_per_feature: int = 64
    allow_requested_backend_fallback: bool = False

    def __post_init__(self) -> None:
        backend = str(self.backend).strip().lower()
        if backend not in TABULAR_QUANTILE_BACKENDS:
            raise ValueError(f"backend must be one of {TABULAR_QUANTILE_BACKENDS}")
        normalized_quantiles = tuple(float(value) for value in self.quantiles)
        if len(normalized_quantiles) != len(set(normalized_quantiles)):
            raise ValueError("quantiles must be unique")
        if set(normalized_quantiles) != set(QUANTILES):
            raise ValueError("the lap-time contract requires p05, p50 and p90")
        if any(not 0.0 < value < 1.0 for value in normalized_quantiles):
            raise ValueError("quantiles must be between zero and one")
        if not self.feature_columns:
            raise ValueError("feature_columns must be an explicit non-empty causal allowlist")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must not contain duplicates")
        forbidden = set(CAUSAL_FEATURE_FORBIDDEN_COLUMNS) | {
            str(self.target_column),
            str(self.season_column),
            str(self.event_time_column),
        }
        overlap = sorted(
            column
            for column in self.feature_columns
            if column in forbidden or str(column).startswith("predicted_")
        )
        if overlap:
            raise ValueError(
                "feature_columns contain forbidden outcome or identifier fields: "
                f"{overlap}"
            )
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        if self.min_rows_for_boosting < 1 or self.n_estimators < 1:
            raise ValueError("row and estimator counts must be positive")
        if self.learning_rate <= 0.0 or self.max_depth < 1:
            raise ValueError("learning_rate and max_depth must be positive")
        if self.max_categories_per_feature < 1:
            raise ValueError("max_categories_per_feature must be positive")
        if self.target_season is not None and int(self.target_season) < 1950:
            raise ValueError("target_season must be a plausible four-digit season")
        if self.fit_before is not None:
            cutoff = pd.to_datetime(self.fit_before, errors="coerce", utc=True)
            if pd.isna(cutoff):
                raise ValueError("fit_before must be a valid timezone-aware timestamp")

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend"] = str(self.backend).strip().lower()
        payload["quantiles"] = [float(value) for value in self.quantiles]
        if self.feature_columns is not None:
            payload["feature_columns"] = list(self.feature_columns)
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class TabularQuantileBackendUnavailable(RuntimeError):
    """Raised when the explicitly requested challenger cannot be fitted."""

    status = "unavailable"

    def __init__(self, requested_backend: str, attempts: Sequence[Mapping[str, Any]]) -> None:
        self.requested_backend = str(requested_backend)
        self.attempts = tuple(dict(attempt) for attempt in attempts)
        super().__init__(f"tabular quantile backend {requested_backend!r} is unavailable")

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "requested_backend": self.requested_backend,
            "backend_attempts": [dict(attempt) for attempt in self.attempts],
        }


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
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _package_version(name: str) -> str | None:
    try:
        return str(importlib.metadata.version(name))
    except Exception:
        return None


def inspect_tabular_quantile_backend(backend: str) -> dict[str, Any]:
    """Return an explicit serializable availability state for one backend."""

    normalized = str(backend).strip().lower()
    if normalized not in TABULAR_QUANTILE_BACKENDS or normalized == "auto":
        raise ValueError("backend inspection requires one concrete tabular quantile backend")
    if normalized in {"lightgbm", "xgboost"}:
        runtime = inspect_optional_model_runtime(normalized)
        return {
            "backend": normalized,
            "status": "available" if runtime.available else "unavailable",
            "available": runtime.available,
            "version": runtime.version,
            "runtime": runtime.to_payload(),
            "reason": runtime.issue,
        }
    if normalized.startswith("sklearn"):
        available = _module_available("sklearn")
        return {
            "backend": normalized,
            "status": "available" if available else "unavailable",
            "available": available,
            "version": _package_version("scikit-learn"),
            "reason": None if available else "package_not_installed",
        }
    return {
        "backend": "empirical",
        "status": "available",
        "available": True,
        "version": "builtin",
        "reason": None,
    }


def _backend_candidates(config: TabularQuantileConfig, row_count: int) -> tuple[str, ...]:
    requested = config.backend.strip().lower()
    if requested == "auto" and row_count < int(config.min_rows_for_boosting):
        return ("empirical",)
    if requested != "auto":
        if config.allow_requested_backend_fallback and requested != "empirical":
            return (requested, "sklearn_gbr", "empirical")
        return (requested,)
    return ("lightgbm", "xgboost", "sklearn_hist", "sklearn_gbr", "empirical")


def _make_estimator(backend: str, quantile: float, config: TabularQuantileConfig) -> Any:
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor  # type: ignore

        return LGBMRegressor(
            objective="quantile",
            alpha=float(quantile),
            n_estimators=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth),
            min_child_samples=5,
            subsample=1.0,
            colsample_bytree=1.0,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            random_state=int(config.random_state),
            verbosity=-1,
        )

    if backend == "xgboost":
        from xgboost import XGBRegressor  # type: ignore

        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=float(quantile),
            n_estimators=int(config.n_estimators),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth),
            random_state=int(config.random_state),
            tree_method="hist",
            n_jobs=1,
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
        ordered_quantiles = tuple(sorted(float(value) for value in self.config.quantiles))
        def predict_one(quantile: float) -> np.ndarray:
            estimator = self.models[quantile]
            if self.backend_name == "lightgbm" and hasattr(estimator, "booster_"):
                values = estimator.booster_.predict(X)
            else:
                values = estimator.predict(X)
            return np.asarray(values, dtype=float)

        raw = np.vstack(
            [predict_one(quantile) for quantile in ordered_quantiles]
        ).T
        if raw.shape != (len(frame), len(ordered_quantiles)) or not np.isfinite(raw).all():
            raise ValueError("quantile backend returned non-finite or malformed predictions")

        # Independent quantile models can cross.  Row-wise monotone rearrangement
        # preserves the three predicted values while restoring their alpha order;
        # unlike cumulative clipping it does not manufacture a duplicated upper
        # quantile.  Column labels are bound to alpha, never tuple position.
        monotone = np.sort(raw, axis=1)
        by_column = {
            PREDICTION_COLUMN_BY_QUANTILE[quantile]: monotone[:, index]
            for index, quantile in enumerate(ordered_quantiles)
        }
        output = pd.DataFrame(by_column, index=frame.index).loc[:, list(PREDICTION_COLUMNS)]
        output["model"] = f"ultimate_lap_time_tabular_quantile_{self.backend_name}"
        return output

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the frozen training manifest."""

        return json.loads(json.dumps(self.training_summary, sort_keys=True, default=str))


def fit_tabular_quantile_model(
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[UltimateLapTelemetryExample],
    *,
    config: TabularQuantileConfig | None = None,
) -> TabularQuantileLapTimeModel:
    """Fit a p05/p50/p90 tabular quantile challenger."""

    cfg = config or TabularQuantileConfig()

    raw_frame = _as_dataframe(records)
    frame = raw_frame.copy()
    if frame.empty:
        raise ValueError("records must contain at least one training row")
    excluded_at_or_after_cutoff = 0
    training_max_time: str | None = None
    if cfg.fit_before is not None:
        if cfg.event_time_column not in frame.columns:
            raise ValueError(
                "chronological quantile training requires the configured event-time column"
            )
        event_times = pd.to_datetime(
            frame[cfg.event_time_column], errors="coerce", utc=True
        )
        if event_times.isna().any():
            raise ValueError("quantile event timestamps must all be valid")
        cutoff = pd.to_datetime(cfg.fit_before, errors="raise", utc=True)
        causal = event_times.lt(cutoff)
        excluded_at_or_after_cutoff = int((~causal).sum())
        frame = frame.loc[causal].copy()
        event_times = event_times.loc[causal]
        if frame.empty:
            raise ValueError("no quantile training rows strictly precede fit_before")
        training_max_time = pd.Timestamp(event_times.max()).isoformat().replace(
            "+00:00", "Z"
        )
    resolved_season: int | None = None
    excluded_other_season_rows = 0
    if cfg.same_season_only:
        if cfg.season_column not in frame.columns:
            raise ValueError(
                "same-season quantile training requires an explicit season column"
            )
        seasons = pd.to_numeric(frame[cfg.season_column], errors="coerce")
        if seasons.isna().any() or not np.equal(seasons, np.floor(seasons)).all():
            raise ValueError("season identifiers must be finite integers")
        resolved_season = (
            int(seasons.max()) if cfg.target_season is None else int(cfg.target_season)
        )
        same_season_mask = seasons.eq(float(resolved_season))
        excluded_other_season_rows = int((~same_season_mask).sum())
        frame = frame.loc[same_season_mask].copy()
        if frame.empty:
            raise ValueError(
                f"no tabular quantile rows are available for target season {resolved_season}"
            )
    y = _finite_target(frame, cfg.target_column).to_numpy(dtype=float)
    finite_mask = np.isfinite(y)
    if not finite_mask.any():
        raise ValueError("records contain no finite lap-time targets")
    frame = frame.loc[finite_mask].reset_index(drop=True)
    y = y[finite_mask]

    selected_features = tuple(cfg.feature_columns or ())

    encoder = TabularFeatureEncoder().fit(
        frame,
        feature_columns=selected_features,
        max_categories_per_feature=int(cfg.max_categories_per_feature),
    )
    X = encoder.transform(frame)

    data_hash = hashlib.sha256()
    data_hash.update(np.asarray(X.shape, dtype=np.int64).tobytes())
    data_hash.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    data_hash.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())

    notes: list[str] = []
    attempts: list[dict[str, Any]] = []
    for backend in _backend_candidates(cfg, len(frame)):
        models: dict[float, Any] = {}
        try:
            availability = inspect_tabular_quantile_backend(backend)
            if not bool(availability["available"]):
                attempts.append(availability)
                notes.append(
                    f"{backend} unavailable: {availability.get('reason') or 'unknown reason'}"
                )
                continue
            for quantile in sorted(float(value) for value in cfg.quantiles):
                estimator = _make_estimator(backend, float(quantile), cfg)
                estimator.fit(X, y)
                models[float(quantile)] = estimator
            requested_backend = str(cfg.backend).strip().lower()
            fallback_used = (
                (requested_backend != "auto" and backend != requested_backend)
                or (
                    requested_backend == "auto"
                    and (backend != "lightgbm" or len(frame) < int(cfg.min_rows_for_boosting))
                )
            )
            selected_attempt = {
                **availability,
                "status": "selected",
                "quantile_objectives": {
                    PREDICTION_COLUMN_BY_QUANTILE[quantile]: {
                        "alpha": quantile,
                        "objective": "quantile",
                    }
                    for quantile in sorted(float(value) for value in cfg.quantiles)
                },
            }
            attempts.append(selected_attempt)
            event_count = (
                int(frame["event_key"].nunique(dropna=False))
                if "event_key" in frame.columns
                else None
            )
            return TabularQuantileLapTimeModel(
                config=cfg,
                backend_name=backend,
                encoder=encoder,
                models=models,
                target_column=_find_target_column(frame, cfg.target_column),
                training_summary={
                    "schema_version": "f1_tabular_quantile_manifest_v1",
                    "status": "available_fallback" if fallback_used else "available",
                    "requested_backend": str(cfg.backend).strip().lower(),
                    "selected_backend": backend,
                    "fallback_used": bool(fallback_used),
                    "config": cfg.to_payload(),
                    "config_sha256": cfg.fingerprint,
                    "training_data_sha256": data_hash.hexdigest(),
                    "rows_seen": int(len(raw_frame)),
                    "rows_used": int(len(frame)),
                    "event_count": event_count,
                    "training_season": resolved_season,
                    "season_transfer_policy": (
                        "same_season_absolute_lap_time_only"
                        if cfg.same_season_only
                        else "explicit_multi_season_training_enabled"
                    ),
                    "other_season_rows_excluded_from_fit": excluded_other_season_rows,
                    "rows_excluded_at_or_after_fit_before": excluded_at_or_after_cutoff,
                    "fit_before": cfg.fit_before,
                    "training_max_event_time": training_max_time,
                    "chronological_cutoff_enforced": cfg.fit_before is not None,
                    "backend": backend,
                    "feature_count": int(X.shape[1]),
                    "feature_names": list(encoder.feature_names_out),
                    "target_column": _find_target_column(frame, cfg.target_column),
                    "quantile_semantics": {
                        "lap_p05": {
                            "alpha": 0.05,
                            "meaning": "fifth percentile; faster/lower lap-time tail",
                        },
                        "lap_p50": {"alpha": 0.50, "meaning": "median lap time"},
                        "lap_p90": {
                            "alpha": 0.90,
                            "meaning": "ninetieth percentile; slower/upper lap-time tail",
                        },
                        "p05_to_p90_nominal_coverage": 0.85,
                        "monotonicity_repair": "rowwise_monotone_rearrangement",
                    },
                    "backend_attempts": attempts,
                    "notes": notes,
                },
            )
        except Exception as exc:  # depends on optional backend versions and native libraries
            attempt = {
                "backend": backend,
                "status": "unavailable",
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}"[:3000],
            }
            attempts.append(attempt)
            notes.append(f"{backend} unavailable or failed: {exc}")
            continue

    raise TabularQuantileBackendUnavailable(str(cfg.backend).strip().lower(), attempts)


def predict_tabular_quantiles(
    model: TabularQuantileLapTimeModel,
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[UltimateLapTelemetryExample],
) -> pd.DataFrame:
    """Predict p05/p50/p90 lap-time quantiles from a fitted tabular model."""

    if not isinstance(model, TabularQuantileLapTimeModel):
        raise TypeError("model must be a TabularQuantileLapTimeModel")
    return model.predict(records)


__all__ = [
    "CAUSAL_FEATURE_FORBIDDEN_COLUMNS",
    "PREDICTION_COLUMNS",
    "PREDICTION_COLUMN_BY_QUANTILE",
    "QUANTILES",
    "TABULAR_QUANTILE_BACKENDS",
    "TabularFeatureEncoder",
    "TabularQuantileBackendUnavailable",
    "TabularQuantileConfig",
    "TabularQuantileLapTimeModel",
    "fit_tabular_quantile_model",
    "inspect_tabular_quantile_backend",
    "predict_tabular_quantiles",
]
