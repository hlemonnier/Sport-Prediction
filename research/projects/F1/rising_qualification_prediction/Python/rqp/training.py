"""Model training utilities with walk-forward model selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - optional dependency
    HistGradientBoostingRegressor = None
    IsotonicRegression = None
    LogisticRegression = None
    Ridge = None
    SimpleImputer = None
    StandardScaler = None

try:
    from xgboost import XGBRanker, XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRanker = None
    XGBRegressor = None

try:
    from lightgbm import LGBMRanker
except Exception:  # pragma: no cover - optional dependency
    LGBMRanker = None

from .dl_models import (
    TorchTabularConfig,
    TorchTabularRegressor,
    resolve_device as resolve_dl_device,
    torch_available,
)
from .utils import team_column


@dataclass
class TrainingResult:
    model: Optional[object]
    model_name: str
    model_family: str
    device_used: Optional[str]
    dl_available: bool
    candidate_leaderboard: List[dict[str, Any]]
    notes: List[str]


@dataclass
class CandidateSpec:
    name: str
    build_model: Callable[[], object]
    task: str  # regression | ranking | eb
    family: str  # ml | dl
    scale_features: bool = False
    device_hint: Optional[str] = None


@dataclass
class CandidateScore:
    name: str
    family: str
    mae: float
    spearman: float
    ndcg10: float
    hit10: float
    composite: float
    device_used: Optional[str] = None


class FeaturePipeline:
    def __init__(self, feature_cols: List[str], scale: bool) -> None:
        self.feature_cols = list(feature_cols)
        self.scale = scale
        self.imputer: Optional[object] = None
        self.scaler: Optional[object] = None
        self.fill_values = pd.Series(dtype=float)

    def _base_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame.reindex(columns=self.feature_cols).copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        if not X.empty:
            all_missing_cols = [col for col in X.columns if X[col].notna().sum() == 0]
            if all_missing_cols:
                X[all_missing_cols] = 0.0
        return X

    def fit(self, frame: pd.DataFrame) -> None:
        X = self._base_frame(frame)
        if X.empty:
            self.fill_values = pd.Series(0.0, index=self.feature_cols, dtype=float)
            return
        if SimpleImputer is not None:
            self.imputer = SimpleImputer(strategy="median")
            transformed = self.imputer.fit_transform(X)
            stats = pd.Series(self.imputer.statistics_, index=self.feature_cols, dtype=float)
            self.fill_values = stats.fillna(0.0)
        else:
            self.fill_values = X.median(numeric_only=True).reindex(self.feature_cols).fillna(0.0)
            transformed = X.fillna(self.fill_values).to_numpy(dtype=float)
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scale and StandardScaler is not None:
            self.scaler = StandardScaler()
            self.scaler.fit(transformed)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        X = self._base_frame(frame)
        if X.empty:
            return np.empty((0, len(self.feature_cols)))
        if self.imputer is not None:
            transformed = self.imputer.transform(X)
        else:
            fill_values = self.fill_values.reindex(self.feature_cols).fillna(0.0)
            transformed = X.fillna(fill_values).to_numpy(dtype=float)
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scale and self.scaler is not None:
            transformed = self.scaler.transform(transformed)
        return transformed

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _safe_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class ProbabilityCalibrator:
    method: str
    model: Optional[object] = None
    center: float = 0.0
    scale: float = 1.0
    shift: float = 0.0

    @classmethod
    def heuristic(cls, scores: pd.Series, labels: pd.Series) -> "ProbabilityCalibrator":
        x = (-pd.to_numeric(scores, errors="coerce")).dropna()
        y = pd.to_numeric(labels, errors="coerce").dropna()
        if x.empty:
            return cls(method="heuristic", center=0.0, scale=1.0, shift=0.0)
        center = float(x.median())
        scale = float(x.std())
        if not np.isfinite(scale) or scale < 1e-6:
            iqr = float(x.quantile(0.75) - x.quantile(0.25))
            scale = iqr / 1.349 if iqr > 1e-6 else 1.0
        raw = _sigmoid(((x.to_numpy(dtype=float) - center) / scale))
        raw_mean = float(np.clip(raw.mean(), 1e-6, 1.0 - 1e-6))
        base_rate = float(np.clip(y.mean() if not y.empty else 0.5, 1e-6, 1.0 - 1e-6))
        shift = math.log(base_rate / (1.0 - base_rate)) - math.log(raw_mean / (1.0 - raw_mean))
        return cls(method="heuristic", center=center, scale=scale, shift=shift)

    def predict(self, scores: pd.Series) -> pd.Series:
        values = pd.to_numeric(scores, errors="coerce")
        output = pd.Series(0.0, index=values.index, dtype=float)
        valid = values.notna()
        if not valid.any():
            return output
        x = (-values.loc[valid]).to_numpy(dtype=float)
        if self.method == "isotonic" and self.model is not None:
            proba = self.model.predict(x)
        elif self.method == "platt" and self.model is not None:
            proba = self.model.predict_proba(x.reshape(-1, 1))[:, 1]
        else:
            raw = _sigmoid((x - self.center) / max(self.scale, 1e-6))
            proba = _sigmoid(_safe_logit(raw) + self.shift)
        output.loc[valid] = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
        return output


@dataclass
class FittedModel:
    estimator: object
    preprocessor: FeaturePipeline
    model_name: str
    family: str
    task: str
    device_used: Optional[str] = None
    calibrators: dict[str, ProbabilityCalibrator] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        X = self.preprocessor.transform(frame)
        preds = self.estimator.predict(X)
        if self.task == "ranking":
            # Normalize ranking model output to the shared convention:
            # lower score means better expected rank.
            preds = -np.asarray(preds, dtype=float)
        return pd.Series(preds, index=frame.index, dtype=float)

    def predict_probabilities(self, scores: pd.Series) -> dict[str, pd.Series]:
        output: dict[str, pd.Series] = {}
        for label, calibrator in self.calibrators.items():
            output[label] = calibrator.predict(scores)
        return output


class QualifyingPositionBaseline:
    """Race baseline that predicts finish order from qualifying context."""

    def __init__(
        self,
        fill_value: float = 10.0,
        primary_column: str = "qualy_position",
        fallback_column: str = "qualy_position",
    ) -> None:
        self.fill_value = float(fill_value)
        self.primary_column = primary_column
        self.fallback_column = fallback_column

    @staticmethod
    def _column_values(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.asarray([], dtype=float)
        values = self._column_values(frame, self.primary_column)
        if values.empty or values.notna().sum() == 0:
            values = self._column_values(frame, self.fallback_column)
        if values.empty or values.notna().sum() == 0:
            return np.full(shape=(len(frame),), fill_value=self.fill_value, dtype=float)
        fill = float(values.median(skipna=True))
        if not np.isfinite(fill):
            fill = self.fill_value
        return values.fillna(fill).to_numpy(dtype=float)


class ColumnBaselineModel:
    """Generic baseline that predicts from a single numeric column."""

    def __init__(self, column: str, fill_value: float = 0.0) -> None:
        self.column = column
        self.fill_value = float(fill_value)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.asarray([], dtype=float)
        if self.column not in frame.columns:
            return np.full(shape=(len(frame),), fill_value=self.fill_value, dtype=float)
        values = pd.to_numeric(frame[self.column], errors="coerce")
        if values.notna().sum() == 0:
            return np.full(shape=(len(frame),), fill_value=self.fill_value, dtype=float)
        fill = float(values.median(skipna=True))
        if not np.isfinite(fill):
            fill = self.fill_value
        return values.fillna(fill).to_numpy(dtype=float)


class EBRankModel:
    """Empirical-Bayes additive rank model using driver/team/track IDs."""

    def __init__(
        self,
        *,
        decay: float = 0.015,
        driver_prior: float = 8.0,
        team_prior: float = 10.0,
        track_prior: float = 12.0,
        iterations: int = 4,
    ) -> None:
        self.decay = float(max(decay, 0.0))
        self.driver_prior = float(max(driver_prior, 1e-6))
        self.team_prior = float(max(team_prior, 1e-6))
        self.track_prior = float(max(track_prior, 1e-6))
        self.iterations = int(max(iterations, 1))
        self.global_mean: float = 10.0
        self.driver_effects: dict[str, float] = {}
        self.team_effects: dict[str, float] = {}
        self.track_effects: dict[str, float] = {}
        self.driver_col = "driver_id"
        self.team_col: Optional[str] = None
        self.track_col: Optional[str] = None
        self.calibrators: dict[str, ProbabilityCalibrator] = {}

    @staticmethod
    def _clean_key(values: pd.Series, fallback: str) -> pd.Series:
        clean = (
            values.astype(str)
            .str.strip()
            .str.lower()
            .replace({"": fallback, "nan": fallback, "none": fallback, "<na>": fallback})
        )
        return clean.fillna(fallback)

    @staticmethod
    def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
        denom = float(weights.sum())
        if denom <= 0.0:
            return float(np.nanmean(values)) if values.size else 0.0
        return float(np.dot(values, weights) / denom)

    @staticmethod
    def _shrink_effect(
        keys: pd.Series,
        residual: np.ndarray,
        weights: np.ndarray,
        prior: float,
    ) -> dict[str, float]:
        sum_wr: dict[str, float] = {}
        sum_w: dict[str, float] = {}
        for key, res, weight in zip(keys.tolist(), residual.tolist(), weights.tolist()):
            key_str = str(key)
            sum_wr[key_str] = sum_wr.get(key_str, 0.0) + (float(res) * float(weight))
            sum_w[key_str] = sum_w.get(key_str, 0.0) + float(weight)
        effects: dict[str, float] = {}
        for key, total_weight in sum_w.items():
            effects[key] = float(sum_wr[key] / (total_weight + float(prior)))
        return effects

    @staticmethod
    def _sort_for_time_decay(frame: pd.DataFrame) -> pd.DataFrame:
        sort_cols = [col for col in ["event_year", "event_round", "event_key"] if col in frame.columns]
        if "driver_id" in frame.columns:
            sort_cols.append("driver_id")
        if not sort_cols:
            return frame.copy()
        return frame.sort_values(sort_cols, kind="mergesort").copy()

    def _resolve_track_col(self, frame: pd.DataFrame) -> Optional[str]:
        for candidate in ("event_name_norm", "track_id", "event_name"):
            if candidate in frame.columns:
                return candidate
        return None

    def fit(self, frame: pd.DataFrame) -> "EBRankModel":
        if frame.empty or "target" not in frame.columns:
            return self

        rows = self._sort_for_time_decay(frame)
        y = pd.to_numeric(rows["target"], errors="coerce")
        valid = y.notna()
        if valid.sum() == 0:
            return self
        rows = rows.loc[valid].copy()
        y = y.loc[valid].astype(float)

        self.team_col = team_column(rows)
        self.track_col = self._resolve_track_col(rows)

        driver_key = self._clean_key(rows.get("driver_id", pd.Series(index=rows.index, data="unknown_driver")), "unknown_driver")
        if self.team_col:
            team_key = self._clean_key(rows[self.team_col], "unknown_team")
        else:
            team_key = pd.Series("unknown_team", index=rows.index, dtype=object)
        if self.track_col:
            track_key = self._clean_key(rows[self.track_col], "unknown_track")
        else:
            track_key = pd.Series("unknown_track", index=rows.index, dtype=object)

        n = len(rows)
        age = np.arange(n, dtype=float)
        if n > 1 and self.decay > 0.0:
            weights = np.exp(-self.decay * (float(n - 1) - age))
        else:
            weights = np.ones(n, dtype=float)
        weights = np.clip(weights, 1e-6, None)

        y_np = y.to_numpy(dtype=float)
        self.global_mean = self._weighted_mean(y_np, weights)
        self.driver_effects = {}
        self.team_effects = {}
        self.track_effects = {}

        for _ in range(self.iterations):
            driver_pred = np.asarray([self.driver_effects.get(str(v), 0.0) for v in driver_key.tolist()], dtype=float)
            team_pred = np.asarray([self.team_effects.get(str(v), 0.0) for v in team_key.tolist()], dtype=float)
            track_pred = np.asarray([self.track_effects.get(str(v), 0.0) for v in track_key.tolist()], dtype=float)

            residual_driver = y_np - (self.global_mean + team_pred + track_pred)
            self.driver_effects = self._shrink_effect(
                keys=driver_key,
                residual=residual_driver,
                weights=weights,
                prior=self.driver_prior,
            )

            driver_pred = np.asarray([self.driver_effects.get(str(v), 0.0) for v in driver_key.tolist()], dtype=float)
            residual_team = y_np - (self.global_mean + driver_pred + track_pred)
            self.team_effects = self._shrink_effect(
                keys=team_key,
                residual=residual_team,
                weights=weights,
                prior=self.team_prior,
            )

            team_pred = np.asarray([self.team_effects.get(str(v), 0.0) for v in team_key.tolist()], dtype=float)
            residual_track = y_np - (self.global_mean + driver_pred + team_pred)
            self.track_effects = self._shrink_effect(
                keys=track_key,
                residual=residual_track,
                weights=weights,
                prior=self.track_prior,
            )

        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.asarray([], dtype=float)

        driver_key = self._clean_key(frame.get("driver_id", pd.Series(index=frame.index, data="unknown_driver")), "unknown_driver")
        if self.team_col and self.team_col in frame.columns:
            team_key = self._clean_key(frame[self.team_col], "unknown_team")
        else:
            team_key = pd.Series("unknown_team", index=frame.index, dtype=object)
        if self.track_col and self.track_col in frame.columns:
            track_key = self._clean_key(frame[self.track_col], "unknown_track")
        else:
            track_key = pd.Series("unknown_track", index=frame.index, dtype=object)

        pred = np.full(len(frame), self.global_mean, dtype=float)
        pred += np.asarray([self.driver_effects.get(str(v), 0.0) for v in driver_key.tolist()], dtype=float)
        pred += np.asarray([self.team_effects.get(str(v), 0.0) for v in team_key.tolist()], dtype=float)
        pred += np.asarray([self.track_effects.get(str(v), 0.0) for v in track_key.tolist()], dtype=float)
        return pred

    def predict_probabilities(self, scores: pd.Series) -> dict[str, pd.Series]:
        output: dict[str, pd.Series] = {}
        for label, calibrator in self.calibrators.items():
            output[label] = calibrator.predict(scores)
        return output


@dataclass
class WeightedBlendModel:
    primary_model: object
    baseline_column: str
    model_weight: float
    baseline_fill: float = 0.0
    calibrators: dict[str, ProbabilityCalibrator] = field(default_factory=dict)

    def _baseline(self, frame: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(frame.get(self.baseline_column), errors="coerce")
        if values is None or len(values) == 0:
            return pd.Series(self.baseline_fill, index=frame.index, dtype=float)
        fill = float(values.median(skipna=True))
        if not np.isfinite(fill):
            fill = self.baseline_fill
        return pd.Series(values.fillna(fill), index=frame.index, dtype=float)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=float)
        primary_raw = self.primary_model.predict(frame)
        primary = pd.Series(primary_raw, index=frame.index, dtype=float)
        baseline = self._baseline(frame)
        weight = float(np.clip(self.model_weight, 0.0, 1.0))
        pred = (weight * primary) + ((1.0 - weight) * baseline)
        return pd.Series(pred, index=frame.index, dtype=float)

    def predict_probabilities(self, scores: pd.Series) -> dict[str, pd.Series]:
        output: dict[str, pd.Series] = {}
        for label, calibrator in self.calibrators.items():
            output[label] = calibrator.predict(scores)
        return output


def _candidate_models(
    *,
    enable_dl_candidates: bool,
    compare_families: list[str],
    dl_arch: str,
    dl_hyperparams: dict[str, Any],
    dl_seed: int,
    dl_device: str,
    requested_model: str,
    notes: list[str],
) -> tuple[list[CandidateSpec], bool]:
    candidates: list[CandidateSpec] = []
    requested = str(requested_model or "auto").strip().lower()
    families = {str(item).strip().lower() for item in (compare_families or ["ml"])}
    allow_ml = "ml" in families or "baseline" in families
    allow_dl = enable_dl_candidates and ("dl" in families) and requested == "auto"
    dl_available = torch_available()
    include_default_ml = requested == "auto"
    include_xgb_rank = requested in {"auto", "xgb_rank"}
    include_eb_rank = requested == "eb_rank"
    include_lgbm_rank = requested == "lgbm_rank"

    if include_xgb_rank and allow_ml:
        if XGBRanker is None:
            notes.append("XGBoost ranker indisponible: xgboost_pairwise ignore.")
        else:
            candidates.append(
                CandidateSpec(
                    name="xgboost_pairwise",
                    task="ranking",
                    family="ml",
                    build_model=lambda: XGBRanker(
                        objective="rank:pairwise",
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        n_jobs=1,
                        verbosity=0,
                    ),
                ),
            )

    if include_default_ml and allow_ml and XGBRegressor is not None:
        candidates.append(
            CandidateSpec(
                name="xgboost",
                task="regression",
                family="ml",
                build_model=lambda: XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        )

    if include_default_ml and allow_ml and HistGradientBoostingRegressor is not None:
        candidates.append(
            CandidateSpec(
                name="hist_gradient_boosting",
                task="regression",
                family="ml",
                build_model=lambda: HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=5,
                    max_iter=600,
                    random_state=42,
                ),
            ),
        )

    if include_default_ml and allow_ml and Ridge is not None:
        candidates.append(
            CandidateSpec(
                name="ridge",
                task="regression",
                family="ml",
                scale_features=True,
                build_model=lambda: Ridge(alpha=1.0),
            ),
        )

    if include_eb_rank:
        candidates.append(
            CandidateSpec(
                name="eb_rank",
                task="eb",
                family="baseline",
                build_model=lambda: EBRankModel(),
            ),
        )

    if include_lgbm_rank:
        if LGBMRanker is None:
            notes.append("LightGBM indisponible: lgbm_rank ignore.")
        else:
            candidates.append(
                CandidateSpec(
                    name="lgbm_rank",
                    task="ranking",
                    family="ml",
                    build_model=lambda: LGBMRanker(
                        objective="lambdarank",
                        n_estimators=500,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                    ),
                ),
            )
    elif requested in {"auto", "baseline", "xgb_rank", "eb_rank"} and LGBMRanker is None:
        notes.append("LightGBM candidate skipped: package lightgbm indisponible.")

    if allow_dl:
        if not dl_available:
            notes.append("DL candidate skipped: PyTorch indisponible.")
        elif dl_arch != "mlp_tabular_v1":
            notes.append(f"DL candidate skipped: architecture non supportee ({dl_arch}).")
        else:
            hidden_dims_raw = dl_hyperparams.get("hidden_dims", [128, 64])
            if not isinstance(hidden_dims_raw, (list, tuple)) or not hidden_dims_raw:
                hidden_dims_raw = [128, 64]
            hidden_dims = tuple(int(max(8, int(v))) for v in hidden_dims_raw)
            cfg = TorchTabularConfig(
                hidden_dims=hidden_dims,
                dropout=float(dl_hyperparams.get("dropout", 0.15)),
                lr=float(dl_hyperparams.get("lr", 1e-3)),
                weight_decay=float(dl_hyperparams.get("weight_decay", 1e-4)),
                epochs=int(dl_hyperparams.get("epochs", 400)),
                batch_size=int(dl_hyperparams.get("batch_size", 64)),
                early_stopping_patience=int(dl_hyperparams.get("early_stopping_patience", 30)),
                seed=int(dl_seed),
                device=str(dl_device or "auto"),
            )
            device_hint = resolve_dl_device(cfg.device)
            notes.append(f"DL candidate active: torch_mlp_tabular_v1 (device={device_hint}).")
            candidates.append(
                CandidateSpec(
                    name="torch_mlp_tabular_v1",
                    task="regression",
                    family="dl",
                    scale_features=True,
                    device_hint=device_hint,
                    build_model=lambda cfg=cfg: TorchTabularRegressor(config=cfg),
                ),
            )

    return candidates, dl_available


def _prepare_training_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if frame.empty or "target" not in frame.columns:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
    y = pd.to_numeric(frame["target"], errors="coerce")
    mask = y.notna()
    filtered = frame.loc[mask].copy()
    y = y.loc[mask]
    if filtered.empty:
        return filtered, y, pd.Series(dtype=float)
    if "event_key" in filtered.columns:
        event_key = pd.to_numeric(filtered["event_key"], errors="coerce")
    else:
        event_key = pd.Series(0, index=filtered.index, dtype=float)
    return filtered, y, event_key


def _sanitize_event_key(event_key: pd.Series) -> Optional[pd.Series]:
    if event_key.empty:
        return None
    cleaned = pd.to_numeric(event_key, errors="coerce").copy()
    if cleaned.notna().sum() == 0:
        return None
    if cleaned.isna().any():
        start = int(cleaned.dropna().max()) + 1
        missing_idx = list(cleaned[cleaned.isna()].index)
        cleaned.loc[missing_idx] = list(range(start, start + len(missing_idx)))
    return cleaned.astype(int)


def _group_sizes_from_sorted_event(event_key: pd.Series) -> list[int]:
    groups: list[int] = []
    prev: Optional[int] = None
    count = 0
    for value in event_key.astype(int).tolist():
        if prev is None or value == prev:
            count += 1
        else:
            groups.append(count)
            count = 1
        prev = value
    if count > 0:
        groups.append(count)
    return groups


def _ranking_relevance_labels(y: pd.Series, event_key: pd.Series) -> pd.Series:
    labels = pd.Series(index=y.index, dtype=float)
    for _, idx in event_key.groupby(event_key, sort=False).groups.items():
        values = y.loc[idx]
        rank = values.rank(method="first", ascending=True)
        labels.loc[idx] = (len(values) + 1 - rank).astype(float)
    return labels


def _fit_candidate(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    candidate: CandidateSpec,
) -> Optional[object]:
    rows, y, event_key = _prepare_training_rows(train_df)
    if rows.empty:
        return None
    model = candidate.build_model()
    try:
        if candidate.task == "eb":
            model.fit(rows)
            return model

        preprocessor = FeaturePipeline(feature_cols=feature_cols, scale=candidate.scale_features)
        if candidate.task == "ranking":
            clean_event = _sanitize_event_key(event_key)
            if clean_event is None:
                return None
            order = clean_event.sort_values(kind="mergesort").index
            rows_sorted = rows.loc[order]
            y_sorted = y.loc[order]
            event_sorted = clean_event.loc[order]
            X_train = preprocessor.fit_transform(rows_sorted)
            y_rank = _ranking_relevance_labels(y_sorted, event_sorted)
            group = _group_sizes_from_sorted_event(event_sorted)
            model.fit(X_train, y_rank.to_numpy(dtype=float), group=group)
        else:
            X_train = preprocessor.fit_transform(rows)
            model.fit(X_train, y.to_numpy(dtype=float))
    except Exception:
        return None
    return FittedModel(
        estimator=model,
        preprocessor=preprocessor,
        model_name=candidate.name,
        family=candidate.family,
        task=candidate.task,
        device_used=getattr(model, "device_used", None),
    )


def _mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true - y_pred).abs().mean())


def _safe_spearman(y_true: pd.Series, y_pred: pd.Series) -> float:
    if y_true.nunique(dropna=True) < 2 or y_pred.nunique(dropna=True) < 2:
        return 0.0
    value = y_true.corr(y_pred, method="spearman")
    if pd.isna(value):
        return 0.0
    return float(max(-1.0, min(1.0, value)))


def _event_groups(event_key: pd.Series, index: pd.Index) -> list[pd.Index]:
    clean_event = _sanitize_event_key(event_key)
    if clean_event is None:
        return [index]
    return [pd.Index(idx) for idx in clean_event.groupby(clean_event, sort=False).groups.values()]


def _ndcg_at_k(actual_rank: pd.Series, pred_score: pd.Series, k: int) -> float:
    relevance = (k + 1.0 - actual_rank).clip(lower=0.0).to_numpy(dtype=float)
    if relevance.size == 0:
        return 0.0
    order = pred_score.sort_values(ascending=True).index
    ranked_relevance = actual_rank.loc[order]
    ranked_relevance = (k + 1.0 - ranked_relevance).clip(lower=0.0).to_numpy(dtype=float)
    top_rel = ranked_relevance[:k]
    if top_rel.size == 0:
        return 0.0
    gains = np.power(2.0, top_rel) - 1.0
    discounts = np.log2(np.arange(2, 2 + len(top_rel)))
    dcg = float((gains / discounts).sum())
    ideal_rel = np.sort(relevance)[::-1][:k]
    ideal_gains = np.power(2.0, ideal_rel) - 1.0
    ideal_discounts = np.log2(np.arange(2, 2 + len(ideal_rel)))
    idcg = float((ideal_gains / ideal_discounts).sum())
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def _topk_hit_rate(actual_rank: pd.Series, pred_score: pd.Series, k: int) -> float:
    actual_top = set(actual_rank[actual_rank <= k].index)
    if not actual_top:
        return 0.0
    pred_top = set(pred_score.sort_values(ascending=True).head(k).index)
    denom = float(min(k, len(actual_top)))
    if denom <= 0:
        return 0.0
    return float(len(actual_top.intersection(pred_top)) / denom)


def _fold_metrics(y_true: pd.Series, pred: pd.Series, event_key: pd.Series) -> dict[str, float]:
    mae_values: list[float] = []
    spearman_values: list[float] = []
    ndcg_values: list[float] = []
    hit_values: list[float] = []
    for idx in _event_groups(event_key, y_true.index):
        y_event = y_true.loc[idx]
        p_event = pred.loc[idx]
        if y_event.empty:
            continue
        actual_rank = y_event.rank(method="first", ascending=True)
        pred_rank = p_event.rank(method="first", ascending=True)
        mae_values.append(_mean_absolute_error(actual_rank, pred_rank))
        spearman_values.append(_safe_spearman(actual_rank, pred_rank))
        ndcg_values.append(_ndcg_at_k(actual_rank, p_event, k=10))
        hit_values.append(_topk_hit_rate(actual_rank, p_event, k=10))
    mae = float(sum(mae_values) / len(mae_values)) if mae_values else float("inf")
    spearman = float(sum(spearman_values) / len(spearman_values)) if spearman_values else 0.0
    ndcg10 = float(sum(ndcg_values) / len(ndcg_values)) if ndcg_values else 0.0
    hit10 = float(sum(hit_values) / len(hit_values)) if hit_values else 0.0
    return {
        "mae": mae,
        "spearman": spearman,
        "ndcg10": ndcg10,
        "hit10": hit10,
    }


def _composite_score(mae: float, spearman: float, ndcg10: float, hit10: float) -> float:
    mae_score = 1.0 / (1.0 + max(mae, 0.0))
    spearman_norm = (max(-1.0, min(1.0, spearman)) + 1.0) / 2.0
    return float((0.35 * mae_score) + (0.25 * spearman_norm) + (0.25 * ndcg10) + (0.15 * hit10))


def _walk_forward_folds(train: pd.DataFrame) -> list[tuple[set[int], int]]:
    if "event_key" not in train.columns:
        return []
    keys = pd.to_numeric(train["event_key"], errors="coerce").dropna().astype(int).unique()
    ordered_keys = sorted(int(k) for k in keys)
    if len(ordered_keys) < 4:
        return []
    min_train_events = max(3, len(ordered_keys) // 3)
    folds: list[tuple[set[int], int]] = []
    for idx in range(min_train_events, len(ordered_keys)):
        folds.append((set(ordered_keys[:idx]), ordered_keys[idx]))
    return folds


def _evaluate_candidate(
    train: pd.DataFrame,
    feature_cols: List[str],
    candidate: CandidateSpec,
    folds: list[tuple[set[int], int]],
) -> Optional[CandidateScore]:
    event_key = pd.to_numeric(train["event_key"], errors="coerce") if "event_key" in train.columns else None
    fold_metrics: list[dict[str, float]] = []
    for train_keys, val_key in folds:
        if event_key is None:
            break
        train_df = train.loc[event_key.isin(train_keys)]
        val_df = train.loc[event_key == val_key]
        if train_df.empty or val_df.empty:
            continue
        fitted = _fit_candidate(train_df, feature_cols, candidate)
        if fitted is None:
            return None
        val_rows, y_val, event_val = _prepare_training_rows(val_df)
        if val_rows.empty:
            continue
        preds = fitted.predict(val_rows)
        fold_metrics.append(_fold_metrics(y_val, preds, event_val))
    if not fold_metrics:
        return None
    mae = float(sum(m["mae"] for m in fold_metrics) / len(fold_metrics))
    spearman = float(sum(m["spearman"] for m in fold_metrics) / len(fold_metrics))
    ndcg10 = float(sum(m["ndcg10"] for m in fold_metrics) / len(fold_metrics))
    hit10 = float(sum(m["hit10"] for m in fold_metrics) / len(fold_metrics))
    composite = _composite_score(mae=mae, spearman=spearman, ndcg10=ndcg10, hit10=hit10)
    return CandidateScore(
        name=candidate.name,
        family=candidate.family,
        mae=mae,
        spearman=spearman,
        ndcg10=ndcg10,
        hit10=hit10,
        composite=composite,
        device_used=candidate.device_hint if candidate.family == "dl" else None,
    )


def _evaluate_column_baseline(
    train: pd.DataFrame,
    folds: list[tuple[set[int], int]],
    column: str,
    name: str,
    default_fill: float = 0.0,
) -> Optional[CandidateScore]:
    if column not in train.columns:
        return None
    event_key = pd.to_numeric(train["event_key"], errors="coerce") if "event_key" in train.columns else None
    if event_key is None:
        return None
    fold_metrics: list[dict[str, float]] = []
    for train_keys, val_key in folds:
        train_df = train.loc[event_key.isin(train_keys)]
        val_df = train.loc[event_key == val_key]
        if train_df.empty or val_df.empty:
            continue
        train_col = pd.to_numeric(train_df.get(column), errors="coerce")
        fill_value = (
            float(train_col.median(skipna=True))
            if train_col.notna().sum() > 0
            else float(default_fill)
        )
        val_rows, y_val, event_val = _prepare_training_rows(val_df)
        if val_rows.empty:
            continue
        pred = pd.to_numeric(val_rows.get(column), errors="coerce")
        pred = pred.fillna(fill_value)
        pred = pd.Series(pred.to_numpy(dtype=float), index=val_rows.index, dtype=float)
        fold_metrics.append(_fold_metrics(y_val, pred, event_val))
    if not fold_metrics:
        return None
    mae = float(sum(m["mae"] for m in fold_metrics) / len(fold_metrics))
    spearman = float(sum(m["spearman"] for m in fold_metrics) / len(fold_metrics))
    ndcg10 = float(sum(m["ndcg10"] for m in fold_metrics) / len(fold_metrics))
    hit10 = float(sum(m["hit10"] for m in fold_metrics) / len(fold_metrics))
    composite = _composite_score(mae=mae, spearman=spearman, ndcg10=ndcg10, hit10=hit10)
    return CandidateScore(
        name=name,
        family="baseline",
        mae=mae,
        spearman=spearman,
        ndcg10=ndcg10,
        hit10=hit10,
        composite=composite,
    )


def _evaluate_blend_candidates(
    train: pd.DataFrame,
    feature_cols: List[str],
    candidate: CandidateSpec,
    folds: list[tuple[set[int], int]],
    baseline_col: str,
    model_weights: list[float],
    default_baseline_fill: float = 0.0,
) -> list[CandidateScore]:
    if baseline_col not in train.columns:
        return []
    event_key = pd.to_numeric(train["event_key"], errors="coerce") if "event_key" in train.columns else None
    if event_key is None:
        return []
    weights = sorted({float(np.clip(w, 0.0, 1.0)) for w in model_weights})
    if not weights:
        return []
    fold_metrics_by_weight: dict[float, list[dict[str, float]]] = {w: [] for w in weights}
    for train_keys, val_key in folds:
        train_df = train.loc[event_key.isin(train_keys)]
        val_df = train.loc[event_key == val_key]
        if train_df.empty or val_df.empty:
            continue
        fitted = _fit_candidate(train_df, feature_cols, candidate)
        if fitted is None:
            return []
        val_rows, y_val, event_val = _prepare_training_rows(val_df)
        if val_rows.empty:
            continue
        model_pred = pd.Series(fitted.predict(val_rows), index=val_rows.index, dtype=float)
        train_base = pd.to_numeric(train_df.get(baseline_col), errors="coerce")
        base_fill = (
            float(train_base.median(skipna=True))
            if train_base.notna().sum() > 0
            else float(default_baseline_fill)
        )
        base_pred = pd.Series(
            pd.to_numeric(val_rows.get(baseline_col), errors="coerce").fillna(base_fill),
            index=val_rows.index,
            dtype=float,
        )
        for model_weight in weights:
            blend = (model_weight * model_pred) + ((1.0 - model_weight) * base_pred)
            fold_metrics_by_weight[model_weight].append(_fold_metrics(y_val, blend, event_val))

    output: list[CandidateScore] = []
    for model_weight in weights:
        fold_metrics = fold_metrics_by_weight[model_weight]
        if not fold_metrics:
            continue
        mae = float(sum(m["mae"] for m in fold_metrics) / len(fold_metrics))
        spearman = float(sum(m["spearman"] for m in fold_metrics) / len(fold_metrics))
        ndcg10 = float(sum(m["ndcg10"] for m in fold_metrics) / len(fold_metrics))
        hit10 = float(sum(m["hit10"] for m in fold_metrics) / len(fold_metrics))
        composite = _composite_score(mae=mae, spearman=spearman, ndcg10=ndcg10, hit10=hit10)
        output.append(
            CandidateScore(
                name=f"pace_blend::{candidate.name}::{baseline_col}::{model_weight:.2f}",
                family=candidate.family,
                mae=mae,
                spearman=spearman,
                ndcg10=ndcg10,
                hit10=hit10,
                composite=composite,
                device_used=candidate.device_hint if candidate.family == "dl" else None,
            )
        )
    return output


def _topk_labels_from_target(y: pd.Series, event_key: pd.Series, k: int) -> pd.Series:
    labels = pd.Series(0.0, index=y.index, dtype=float)
    for idx in _event_groups(event_key, y.index):
        y_event = y.loc[idx]
        if y_event.empty:
            continue
        rank = y_event.rank(method="first", ascending=True)
        labels.loc[idx] = (rank <= k).astype(float)
    return labels


def _fit_probability_calibrator(scores: pd.Series, labels: pd.Series) -> ProbabilityCalibrator:
    valid = scores.notna() & labels.notna()
    x = scores.loc[valid]
    y = labels.loc[valid]
    if x.empty:
        return ProbabilityCalibrator.heuristic(scores, labels)
    if y.nunique(dropna=True) < 2:
        return ProbabilityCalibrator.heuristic(x, y)
    x_neg = (-x).to_numpy(dtype=float)
    y_int = y.astype(int).to_numpy()
    if IsotonicRegression is not None and x.nunique(dropna=True) >= 2:
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x_neg, y_int)
            return ProbabilityCalibrator(method="isotonic", model=iso)
        except Exception:
            pass
    if LogisticRegression is not None and x.nunique(dropna=True) >= 2:
        try:
            platt = LogisticRegression(solver="lbfgs", max_iter=400)
            platt.fit(x_neg.reshape(-1, 1), y_int)
            return ProbabilityCalibrator(method="platt", model=platt)
        except Exception:
            pass
    return ProbabilityCalibrator.heuristic(x, y)


def _normalize_requested_model(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    allowed = {"auto", "baseline", "xgb_rank", "eb_rank", "lgbm_rank"}
    if normalized in allowed:
        return normalized
    return "auto"


def train_model(
    train: pd.DataFrame,
    feature_cols: List[str],
    *,
    enable_dl_candidates: bool = False,
    compare_families: Optional[List[str]] = None,
    dl_device: str = "auto",
    dl_arch: str = "mlp_tabular_v1",
    dl_hyperparams: Optional[dict[str, Any]] = None,
    dl_seed: int = 42,
    f1_model: str = "auto",
) -> TrainingResult:
    notes: List[str] = []
    leaderboard_data: List[dict[str, Any]] = []
    compare = compare_families or ["ml"]
    dl_hparams = dict(dl_hyperparams or {})
    requested_model = _normalize_requested_model(f1_model)
    dl_available = torch_available()
    dl_requested = enable_dl_candidates and ("dl" in {str(f).strip().lower() for f in compare})

    if requested_model == "lgbm_rank" and LGBMRanker is None:
        raise SystemExit(
            "f1_model=lgbm_rank demande mais LightGBM n'est pas installe. "
            "Installez `lightgbm` ou choisissez un autre --f1_model."
        )

    if train.empty:
        if dl_requested and not dl_available:
            notes.append("DL candidate skipped: PyTorch indisponible.")
        notes.append("Pas assez de data historique: fallback heuristique.")
        return TrainingResult(
            model=None,
            model_name="heuristic",
            model_family="heuristic",
            device_used=None,
            dl_available=dl_available,
            candidate_leaderboard=leaderboard_data,
            notes=notes,
        )

    event_count = 0
    if "event_key" in train.columns:
        event_series = pd.to_numeric(train["event_key"], errors="coerce").dropna()
        if not event_series.empty:
            event_count = int(event_series.astype(int).nunique())

    candidates, dl_available = _candidate_models(
        enable_dl_candidates=enable_dl_candidates,
        compare_families=compare,
        dl_arch=dl_arch,
        dl_hyperparams=dl_hparams,
        dl_seed=dl_seed,
        dl_device=dl_device,
        requested_model=requested_model,
        notes=notes,
    )
    race_baseline_col = "qualy_context_position" if "qualy_context_position" in feature_cols else "qualy_position"
    race_baseline_supported = race_baseline_col in feature_cols
    race_pace_baseline_cols = [
        col for col in ["fp_race_sim_rank", "fp_race_sim_delta", "event_pace_index"] if col in feature_cols
    ]
    qualifying_baseline_cols = [
        col
        for col in [
            "fp_quali_sim_rank",
            "fp_mean_rank",
            "fp_quali_sim_delta",
            "event_pace_index",
            "fp_weighted_delta",
        ]
        if col in feature_cols
    ]
    qualifying_baseline_supported = bool(qualifying_baseline_cols) and not race_baseline_supported
    small_history_qualifying = (
        qualifying_baseline_supported
        and event_count > 0
        and event_count < 8
    )
    if (
        small_history_qualifying
        and "fp_weighted_delta" in qualifying_baseline_cols
        and "fp_mean_rank" in qualifying_baseline_cols
    ):
        qualifying_baseline_cols = [col for col in qualifying_baseline_cols if col != "fp_weighted_delta"]
        notes.append(
            "Historique court (<8 events): baseline fp_weighted_delta retiree pour limiter l'instabilite.",
        )
    if small_history_qualifying:
        notes.append("Historique court (<8 events): selection restreinte aux baselines qualif.")

    if requested_model != "auto":
        notes.append(f"Selection forcee via f1_model={requested_model}.")

    if not candidates and not race_baseline_supported and not qualifying_baseline_supported:
        notes.append("Aucun modele disponible: fallback heuristique.")
        return TrainingResult(
            model=None,
            model_name="heuristic",
            model_family="heuristic",
            device_used=None,
            dl_available=dl_available,
            candidate_leaderboard=leaderboard_data,
            notes=notes,
        )

    notes.append(
        "Score composite model selection: 0.35*MAE_score + 0.25*Spearman_norm + "
        "0.25*NDCG@10 + 0.15*Top10Hit (MAE_score=1/(1+MAE), Spearman_norm=(rho+1)/2).",
    )

    score_lookup: dict[str, CandidateScore] = {}
    candidate_lookup: dict[str, CandidateSpec] = {c.name: c for c in candidates}
    selected_from_cv: Optional[str] = None

    folds = _walk_forward_folds(train)
    if folds:
        if not small_history_qualifying:
            for candidate in candidates:
                score = _evaluate_candidate(train, feature_cols, candidate, folds)
                if score is None:
                    continue
                score_lookup[candidate.name] = score
        if race_baseline_supported:
            baseline_score = _evaluate_column_baseline(
                train=train,
                folds=folds,
                column=race_baseline_col,
                name="qualifying_baseline",
                default_fill=10.0,
            )
            if baseline_score is not None:
                score_lookup[baseline_score.name] = baseline_score
            for col in race_pace_baseline_cols:
                pace_score = _evaluate_column_baseline(
                    train=train,
                    folds=folds,
                    column=col,
                    name=f"pace_baseline::{col}",
                    default_fill=0.0,
                )
                if pace_score is not None:
                    score_lookup[pace_score.name] = pace_score
        if qualifying_baseline_supported:
            for col in qualifying_baseline_cols:
                baseline_score = _evaluate_column_baseline(
                    train=train,
                    folds=folds,
                    column=col,
                    name=f"pace_baseline::{col}",
                    default_fill=0.0,
                )
                if baseline_score is not None:
                    score_lookup[baseline_score.name] = baseline_score

            if not small_history_qualifying:
                blend_weights = [0.20, 0.35, 0.50, 0.65, 0.80]
                for candidate in candidates:
                    for blend_col in qualifying_baseline_cols:
                        blend_scores = _evaluate_blend_candidates(
                            train=train,
                            feature_cols=feature_cols,
                            candidate=candidate,
                            folds=folds,
                            baseline_col=blend_col,
                            model_weights=blend_weights,
                            default_baseline_fill=0.0,
                        )
                        for blend_score in blend_scores:
                            score_lookup[blend_score.name] = blend_score

        if score_lookup:
            ranking = sorted(
                score_lookup.values(),
                key=lambda s: (-s.composite, s.mae),
            )
            leaderboard_data = [
                {
                    "name": s.name,
                    "family": s.family,
                    "device_used": s.device_used,
                    "composite": float(s.composite),
                    "mae": float(s.mae),
                    "spearman": float(s.spearman),
                    "ndcg10": float(s.ndcg10),
                    "hit10": float(s.hit10),
                }
                for s in ranking
            ]
            leaderboard = ", ".join(
                (
                    f"{s.name}[{s.family}](C={s.composite:.3f}, MAE={s.mae:.3f}, "
                    f"rho={s.spearman:.3f}, NDCG10={s.ndcg10:.3f}, Hit10={s.hit10:.3f})"
                )
                for s in ranking
            )
            notes.append(f"Model selection walk-forward: {leaderboard}.")
            selected_from_cv = ranking[0].name
            notes.append(
                f"Modele retenu: {selected_from_cv} (score composite={ranking[0].composite:.3f}).",
            )
        else:
            notes.append("Aucun score walk-forward exploitable; selection par priorite.")
    else:
        notes.append("Historique insuffisant pour validation walk-forward, selection par priorite.")
        if race_baseline_supported:
            selected_from_cv = "qualifying_baseline"
            notes.append("Mode conservateur: baseline qualif prioritaire sans folds.")
        elif qualifying_baseline_supported:
            default_col = (
                "event_pace_index"
                if "event_pace_index" in qualifying_baseline_cols
                else qualifying_baseline_cols[0]
            )
            selected_from_cv = f"pace_baseline::{default_col}"
            notes.append("Mode conservateur: baseline pace prioritaire sans folds.")

    if requested_model == "baseline":
        if race_baseline_supported:
            selected_from_cv = "qualifying_baseline"
        elif qualifying_baseline_supported:
            default_col = (
                "event_pace_index"
                if "event_pace_index" in qualifying_baseline_cols
                else qualifying_baseline_cols[0]
            )
            selected_from_cv = f"pace_baseline::{default_col}"
        else:
            notes.append("f1_model=baseline demande mais aucune baseline exploitable n'est disponible.")
    elif requested_model == "xgb_rank":
        selected_from_cv = "xgboost_pairwise"
    elif requested_model == "eb_rank":
        selected_from_cv = "eb_rank"
    elif requested_model == "lgbm_rank":
        selected_from_cv = "lgbm_rank"

    if requested_model != "auto" and selected_from_cv:
        notes.append(f"Selection finale forcee: {selected_from_cv}.")

    selected_model: Optional[object] = None
    selected_name: Optional[str] = None
    selected_family = "heuristic"
    selected_device: Optional[str] = None

    def _median_fill(column: str, default_fill: float) -> float:
        values = pd.to_numeric(train.get(column), errors="coerce")
        if values.notna().sum() == 0:
            return float(default_fill)
        fill = float(values.median(skipna=True))
        if not np.isfinite(fill):
            return float(default_fill)
        return fill

    if selected_from_cv == "qualifying_baseline":
        fill_value = _median_fill(race_baseline_col, 10.0)
        selected_model = QualifyingPositionBaseline(
            fill_value=fill_value,
            primary_column=race_baseline_col,
            fallback_column="qualy_position",
        )
        selected_name = "qualifying_baseline"
        selected_family = "baseline"
    elif selected_from_cv and selected_from_cv.startswith("pace_baseline::"):
        _, _, baseline_col = selected_from_cv.partition("::")
        baseline_col = baseline_col.strip()
        if baseline_col:
            fill_value = _median_fill(baseline_col, 0.0)
            selected_model = ColumnBaselineModel(column=baseline_col, fill_value=fill_value)
            selected_name = selected_from_cv
            selected_family = "baseline"
    elif selected_from_cv and selected_from_cv.startswith("pace_blend::"):
        parts = selected_from_cv.split("::")
        if len(parts) == 4:
            _, candidate_name, baseline_col, weight_text = parts
            candidate = candidate_lookup.get(candidate_name)
            try:
                model_weight = float(weight_text)
            except ValueError:
                model_weight = 0.30
            if candidate is not None:
                fitted = _fit_candidate(train, feature_cols, candidate)
                if fitted is not None:
                    selected_model = WeightedBlendModel(
                        primary_model=fitted,
                        baseline_column=baseline_col,
                        model_weight=model_weight,
                        baseline_fill=_median_fill(baseline_col, 0.0),
                    )
                    selected_name = selected_from_cv
                    selected_family = candidate.family
                    selected_device = getattr(fitted, "device_used", None)

    else:
        candidate_order: list[CandidateSpec] = []
        if selected_from_cv and selected_from_cv in candidate_lookup:
            candidate_order.append(candidate_lookup[selected_from_cv])
        candidate_order.extend([c for c in candidates if c.name != (selected_from_cv or "")])
        for candidate in candidate_order:
            fitted = _fit_candidate(train, feature_cols, candidate)
            if fitted is None:
                continue
            selected_model = fitted
            selected_name = candidate.name
            selected_family = candidate.family
            selected_device = getattr(fitted, "device_used", None)
            break

    if selected_model is None and race_baseline_supported:
        fill_value = _median_fill(race_baseline_col, 10.0)
        selected_model = QualifyingPositionBaseline(
            fill_value=fill_value,
            primary_column=race_baseline_col,
            fallback_column="qualy_position",
        )
        selected_name = "qualifying_baseline"
        selected_family = "baseline"
        notes.append("Fallback prioritaire active: baseline qualif.")
    if selected_model is None and qualifying_baseline_supported:
        default_col = (
            "event_pace_index"
            if "event_pace_index" in qualifying_baseline_cols
            else qualifying_baseline_cols[0]
        )
        selected_model = ColumnBaselineModel(
            column=default_col,
            fill_value=_median_fill(default_col, 0.0),
        )
        selected_name = f"pace_baseline::{default_col}"
        selected_family = "baseline"
        notes.append("Fallback prioritaire active: baseline pace.")

    if selected_model is None:
        notes.append("Echec entrainement de tous les candidats: fallback heuristique.")
        return TrainingResult(
            model=None,
            model_name="heuristic",
            model_family="heuristic",
            device_used=None,
            dl_available=dl_available,
            candidate_leaderboard=leaderboard_data,
            notes=notes,
        )

    if selected_from_cv is None and selected_name is not None:
        notes.append(f"Modele retenu par defaut: {selected_name}.")

    rows, y_train, event_train = _prepare_training_rows(train)
    if (
        not rows.empty
        and hasattr(selected_model, "calibrators")
        and isinstance(getattr(selected_model, "calibrators"), dict)
    ):
        try:
            in_sample_scores = pd.Series(selected_model.predict(rows), index=rows.index, dtype=float)
            top10_labels = _topk_labels_from_target(y_train, event_train, k=10)
            top3_labels = _topk_labels_from_target(y_train, event_train, k=3)
            calibrator_top10 = _fit_probability_calibrator(in_sample_scores, top10_labels)
            calibrator_top3 = _fit_probability_calibrator(in_sample_scores, top3_labels)
            selected_model.calibrators["top10"] = calibrator_top10
            selected_model.calibrators["top3"] = calibrator_top3
            notes.append(
                f"Calibration probabiliste: top10={calibrator_top10.method}, top3={calibrator_top3.method}.",
            )
        except Exception:
            pass

    if selected_name == "qualifying_baseline":
        notes.append("Prediction race: baseline qualif contextuelle (qualif + signal predit).")
    elif selected_name and selected_name.startswith("pace_baseline::"):
        notes.append("Prediction qualif: baseline pace local (FP/sprint).")
    elif selected_name and selected_name.startswith("pace_blend::"):
        notes.append("Prediction qualif: blend modele + baseline pace.")
    elif selected_name and selected_name.startswith("torch_"):
        notes.append("Prediction DL active: MLP tabulaire PyTorch.")

    if SimpleImputer is None:
        notes.append("SimpleImputer indisponible: imputation mediane pandas utilisee.")
    if Ridge is not None and StandardScaler is None:
        notes.append("StandardScaler indisponible: scaling desactive pour Ridge.")

    return TrainingResult(
        model=selected_model,
        model_name=selected_name or "unknown",
        model_family=selected_family,
        device_used=selected_device or getattr(selected_model, "device_used", None),
        dl_available=dl_available,
        candidate_leaderboard=leaderboard_data,
        notes=notes,
    )
