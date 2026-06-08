"""Model training utilities with walk-forward model selection."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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

from packages.f1.models.deep_learning import (
    TorchTabularConfig,
    TorchTabularRegressor,
    resolve_device as resolve_dl_device,
    torch_available,
)
from packages.f1.models.probability import pl_gumbel_probabilities
from packages.f1.data.utils import team_column

SAMPLE_WEIGHT_COL = "_sample_weight"
DEFAULT_PL_SAMPLES = 2000
DEFAULT_PL_SEED = 42
DEFAULT_PROBABILITY_AUDIT_MIN_EVENTS = 5
DEFAULT_PROBABILITY_AUDIT_BOOTSTRAP_SAMPLES = 200
PROBABILITY_AUDIT_SCHEMA_VERSION = "pl_gumbel_probability_audit_v2"


@dataclass
class TrainingResult:
    model: Optional[object]
    model_name: str
    model_family: str
    device_used: Optional[str]
    dl_available: bool
    candidate_leaderboard: List[dict[str, Any]]
    notes: List[str]
    listwise_temperature: Optional[float] = None
    probability_audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSpec:
    name: str
    build_model: Callable[[], object]
    task: str  # regression | ranking | eb | strategic_race
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


@dataclass(frozen=True)
class TargetSpec:
    train_col: str = "target"
    actual_col: str = "target"
    base_col: Optional[str] = None
    base_fill: float = 0.0
    label: str = "finish_position"
    constraint_mode: str = "constrained"

    @property
    def uses_offset(self) -> bool:
        return bool(self.base_col)


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


TEAM_CAR_ARCHETYPE_PROFILES: dict[str, dict[str, float]] = {
    "red bull": {"power": 0.72, "downforce": 0.88, "low_speed": 0.78, "high_speed": 0.86, "traction": 0.80, "braking": 0.78, "tyre": 0.78},
    "ferrari": {"power": 0.80, "downforce": 0.86, "low_speed": 0.88, "high_speed": 0.78, "traction": 0.84, "braking": 0.76, "tyre": 0.72},
    "mclaren": {"power": 0.82, "downforce": 0.88, "low_speed": 0.76, "high_speed": 0.90, "traction": 0.78, "braking": 0.78, "tyre": 0.82},
    "mercedes": {"power": 0.82, "downforce": 0.78, "low_speed": 0.72, "high_speed": 0.82, "traction": 0.72, "braking": 0.80, "tyre": 0.78},
    "williams": {"power": 0.82, "downforce": 0.58, "low_speed": 0.52, "high_speed": 0.72, "traction": 0.58, "braking": 0.62, "tyre": 0.58},
    "aston martin": {"power": 0.76, "downforce": 0.74, "low_speed": 0.72, "high_speed": 0.68, "traction": 0.70, "braking": 0.72, "tyre": 0.66},
    "alpine": {"power": 0.72, "downforce": 0.62, "low_speed": 0.62, "high_speed": 0.58, "traction": 0.66, "braking": 0.66, "tyre": 0.62},
    "haas": {"power": 0.78, "downforce": 0.58, "low_speed": 0.56, "high_speed": 0.60, "traction": 0.62, "braking": 0.64, "tyre": 0.56},
    "racing bulls": {"power": 0.72, "downforce": 0.66, "low_speed": 0.68, "high_speed": 0.64, "traction": 0.70, "braking": 0.66, "tyre": 0.64},
    "rb": {"power": 0.72, "downforce": 0.66, "low_speed": 0.68, "high_speed": 0.64, "traction": 0.70, "braking": 0.66, "tyre": 0.64},
    "sauber": {"power": 0.76, "downforce": 0.56, "low_speed": 0.54, "high_speed": 0.58, "traction": 0.58, "braking": 0.60, "tyre": 0.56},
    "kick sauber": {"power": 0.76, "downforce": 0.56, "low_speed": 0.54, "high_speed": 0.58, "traction": 0.58, "braking": 0.60, "tyre": 0.56},
}


class StrategicRaceDeltaModel:
    """Single race scorer combining grid, history, FP, and circuit features."""

    def __init__(
        self,
        *,
        driver_prior: float = 2.0,
        team_prior: float = 3.0,
        circuit_prior: float = 4.0,
        cap: float = 2.5,
    ) -> None:
        self.driver_prior = float(max(driver_prior, 1e-6))
        self.team_prior = float(max(team_prior, 1e-6))
        self.circuit_prior = float(max(circuit_prior, 1e-6))
        self.cap = float(max(cap, 0.25))
        self.global_delta: float = 0.0
        self.driver_delta: dict[str, tuple[float, float]] = {}
        self.team_delta: dict[str, tuple[float, float]] = {}
        self.circuit_delta: dict[str, tuple[float, float]] = {}
        self.archetype_delta: dict[str, tuple[float, float]] = {}
        self.team_col: Optional[str] = None
        self.circuit_col: Optional[str] = None
        self.archetype_col: Optional[str] = None
        self.calibrators: dict[str, ProbabilityCalibrator] = {}

    @staticmethod
    def _clean_key(value: object, fallback: str) -> str:
        if value is None or pd.isna(value):
            return fallback
        text = str(value).strip().lower()
        return text if text and text not in {"nan", "none", "<na>"} else fallback

    @staticmethod
    def _numeric(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(float(default), index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").fillna(float(default))

    @staticmethod
    def _rank_component(values: pd.Series, fallback: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().sum() == 0:
            return fallback.copy()
        ranked = numeric.rank(method="average", ascending=True)
        return ranked.fillna(fallback)

    @staticmethod
    def _team_profile(value: object) -> dict[str, float]:
        team = StrategicRaceDeltaModel._clean_key(value, "")
        for key, profile in TEAM_CAR_ARCHETYPE_PROFILES.items():
            if key in team or team in key:
                return profile
        return {"power": 0.65, "downforce": 0.65, "low_speed": 0.65, "high_speed": 0.65, "traction": 0.65, "braking": 0.65, "tyre": 0.65}

    @staticmethod
    def _shrunken_group_delta(
        rows: pd.DataFrame,
        key_col: str,
        *,
        prior: float,
        weights: Optional[np.ndarray],
    ) -> dict[str, tuple[float, float]]:
        if key_col not in rows.columns or "race_delta_target" not in rows.columns:
            return {}
        keys = rows[key_col].map(lambda value: StrategicRaceDeltaModel._clean_key(value, "unknown"))
        deltas = pd.to_numeric(rows["race_delta_target"], errors="coerce")
        valid = deltas.notna()
        if not valid.any():
            return {}
        if weights is None:
            weight_series = pd.Series(1.0, index=rows.index, dtype=float)
        else:
            weight_series = pd.Series(weights, index=rows.index, dtype=float)
        output: dict[str, tuple[float, float]] = {}
        for key, idx in keys.loc[valid].groupby(keys.loc[valid], sort=False).groups.items():
            w = weight_series.loc[idx].astype(float).clip(lower=1e-6)
            y = deltas.loc[idx].astype(float)
            denom = float(w.sum())
            if denom <= 0.0:
                continue
            mean = float((y * w).sum() / denom)
            output[str(key)] = (mean, denom)
        return output

    def _resolve_circuit_col(self, frame: pd.DataFrame) -> Optional[str]:
        for candidate in ("circuit_card_id", "event_name_norm", "event_name"):
            if candidate in frame.columns:
                return candidate
        return None

    def _resolve_archetype_col(self, frame: pd.DataFrame) -> Optional[str]:
        return "circuit_archetype" if "circuit_archetype" in frame.columns else None

    def fit(self, frame: pd.DataFrame) -> "StrategicRaceDeltaModel":
        if frame.empty or "race_delta_target" not in frame.columns:
            return self
        rows = frame.copy()
        y = pd.to_numeric(rows["race_delta_target"], errors="coerce")
        valid = y.notna()
        if not valid.any():
            return self
        rows = rows.loc[valid].copy()
        y = y.loc[valid].astype(float)
        weights = _sample_weight_array(rows)
        if weights is None:
            self.global_delta = float(y.mean())
        else:
            self.global_delta = float(np.average(y.to_numpy(dtype=float), weights=weights))
        if not np.isfinite(self.global_delta):
            self.global_delta = 0.0

        self.team_col = team_column(rows)
        self.circuit_col = self._resolve_circuit_col(rows)
        self.archetype_col = self._resolve_archetype_col(rows)
        self.driver_delta = self._shrunken_group_delta(rows, "driver_id", prior=self.driver_prior, weights=weights)
        self.team_delta = self._shrunken_group_delta(rows, self.team_col, prior=self.team_prior, weights=weights) if self.team_col else {}
        self.circuit_delta = (
            self._shrunken_group_delta(rows, self.circuit_col, prior=self.circuit_prior, weights=weights)
            if self.circuit_col
            else {}
        )
        self.archetype_delta = (
            self._shrunken_group_delta(rows, self.archetype_col, prior=self.circuit_prior, weights=weights)
            if self.archetype_col
            else {}
        )
        return self

    def _group_prior(self, frame: pd.DataFrame, column: Optional[str], store: dict[str, tuple[float, float]], prior: float) -> pd.Series:
        if not column or column not in frame.columns or not store:
            return pd.Series(self.global_delta, index=frame.index, dtype=float)
        values: list[float] = []
        for value in frame[column].tolist():
            key = self._clean_key(value, "unknown")
            mean, weight = store.get(key, (self.global_delta, 0.0))
            shrink = float(weight / (weight + prior)) if weight > 0.0 else 0.0
            values.append((shrink * mean) + ((1.0 - shrink) * self.global_delta))
        return pd.Series(values, index=frame.index, dtype=float)

    def _mobility(self, frame: pd.DataFrame) -> pd.Series:
        mobility = (
            self._numeric(frame, "track_finish_order_mobility", 0.35)
            if "track_finish_order_mobility" in frame.columns
            else self._numeric(frame, "track_overtake_propensity", 0.35)
        ).clip(0.0, 1.0)
        drs = self._numeric(frame, "circuit_drs_effectiveness", 0.45).clip(0.0, 1.0)
        difficulty = self._numeric(frame, "circuit_overtaking_difficulty", 0.50).clip(0.0, 1.0)
        safety = self._numeric(frame, "track_safety_car_prior", 0.25).clip(0.0, 1.0)
        chaos = self._numeric(frame, "track_chaos_index", 0.20).clip(0.0, 1.0)
        strategy = self._numeric(frame, "track_strategy_variance_prior", 0.35).clip(0.0, 1.0)
        value = (
            0.06
            + (0.48 * mobility)
            + (0.16 * drs)
            - (0.35 * difficulty)
            + (0.12 * safety)
            + (0.16 * chaos)
            + (0.08 * strategy)
        )
        return value.clip(lower=0.04, upper=0.85)

    def _team_car_fit_rank(self, frame: pd.DataFrame, fallback: pd.Series) -> pd.Series:
        if not self.team_col or self.team_col not in frame.columns:
            return fallback.copy()
        demands = {
            "power": self._numeric(frame, "circuit_power_sensitivity", 0.55).clip(0.0, 1.0),
            "downforce": self._numeric(frame, "circuit_downforce_demand", 0.55).clip(0.0, 1.0),
            "low_speed": self._numeric(frame, "circuit_low_speed_corner_demand", 0.55).clip(0.0, 1.0),
            "high_speed": self._numeric(frame, "circuit_high_speed_corner_demand", 0.55).clip(0.0, 1.0),
            "traction": self._numeric(frame, "circuit_traction_demand", 0.55).clip(0.0, 1.0),
            "braking": self._numeric(frame, "circuit_braking_demand", 0.55).clip(0.0, 1.0),
            "tyre": self._numeric(frame, "circuit_tyre_degradation", 0.55).clip(0.0, 1.0),
        }
        fit_values: list[float] = []
        for idx, team_value in frame[self.team_col].items():
            profile = self._team_profile(team_value)
            numerator = 0.0
            denominator = 0.0
            for key, demand_series in demands.items():
                demand = float(demand_series.loc[idx])
                if not np.isfinite(demand):
                    continue
                numerator += demand * float(profile.get(key, 0.65))
                denominator += demand
            fit_values.append(numerator / denominator if denominator > 0.0 else 0.65)
        fit = pd.Series(fit_values, index=frame.index, dtype=float)
        return (-fit).rank(method="average", ascending=True).fillna(fallback)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.asarray([], dtype=float)
        grid = self._numeric(frame, "grid_position", 10.0)
        if grid.notna().sum() == 0:
            grid = self._numeric(frame, "qualy_position", 10.0)
        grid = grid.fillna(float(grid.median(skipna=True) if grid.notna().any() else 10.0))

        driver_prior = self._group_prior(frame, "driver_id", self.driver_delta, self.driver_prior)
        team_prior = self._group_prior(frame, self.team_col, self.team_delta, self.team_prior)
        circuit_prior = self._group_prior(frame, self.circuit_col, self.circuit_delta, self.circuit_prior)
        archetype_prior = self._group_prior(frame, self.archetype_col, self.archetype_delta, self.circuit_prior)

        fp_race_rank = self._rank_component(frame.get("fp_race_sim_rank", pd.Series(index=frame.index, dtype=float)), grid)
        fp_mean_rank = self._rank_component(frame.get("fp_mean_rank", pd.Series(index=frame.index, dtype=float)), grid)
        fp_quali_rank = self._rank_component(frame.get("fp_quali_sim_rank", pd.Series(index=frame.index, dtype=float)), grid)
        circuit_fit = self._rank_component(frame.get("circuit_fit_index", pd.Series(index=frame.index, dtype=float)), grid)
        team_car_fit = self._team_car_fit_rank(frame, grid)
        driver_archetype = self._rank_component(
            frame.get("driver_archetype_form_3_fp_weighted_delta", pd.Series(index=frame.index, dtype=float)),
            grid,
        )
        team_archetype_perf = self._rank_component(
            frame.get("team_archetype_form_3_fp_weighted_delta", pd.Series(index=frame.index, dtype=float)),
            grid,
        )
        driver_circuit = self._rank_component(
            frame.get("driver_circuit_hist_fp_weighted_delta", pd.Series(index=frame.index, dtype=float)),
            grid,
        )
        team_circuit = self._rank_component(
            frame.get("team_circuit_hist_fp_weighted_delta", pd.Series(index=frame.index, dtype=float)),
            grid,
        )

        fp_race_move = fp_race_rank - grid
        fp_mean_move = fp_mean_rank - grid
        fp_quali_move = fp_quali_rank - grid
        circuit_fit_move = circuit_fit - grid
        team_car_fit_move = team_car_fit - grid
        driver_archetype_move = driver_archetype - grid
        team_archetype_move = team_archetype_perf - grid
        driver_circuit_move = driver_circuit - grid
        team_circuit_move = team_circuit - grid

        dnf_penalty = (
            self._numeric(frame, "track_dnf_prior", 0.08).clip(0.0, 0.60)
            * (1.0 + self._numeric(frame, "fp_slow_lap_ratio", 0.0).clip(0.0, 1.0))
            * 0.65
        )
        strategic_delta = (
            (0.34 * driver_prior)
            + (0.18 * team_prior)
            + (0.10 * circuit_prior)
            + (0.08 * archetype_prior)
            + (0.12 * fp_race_move)
            + (0.06 * fp_mean_move)
            + (0.04 * fp_quali_move)
            + (0.04 * circuit_fit_move)
            + (0.06 * team_car_fit_move)
            + (0.03 * driver_archetype_move)
            + (0.03 * team_archetype_move)
            + (0.02 * driver_circuit_move)
            + (0.02 * team_circuit_move)
            + dnf_penalty
        )
        mobility = self._mobility(frame)
        max_delta = np.maximum(0.60, self.cap * (0.35 + mobility.to_numpy(dtype=float)))
        delta = strategic_delta.to_numpy(dtype=float) * (0.30 + (0.70 * mobility.to_numpy(dtype=float)))
        delta = np.clip(delta, -max_delta, max_delta)
        score = grid.to_numpy(dtype=float) + delta
        return np.asarray(score, dtype=float)

    def predict_probabilities(self, scores: pd.Series) -> dict[str, pd.Series]:
        output: dict[str, pd.Series] = {}
        for label, calibrator in self.calibrators.items():
            output[label] = calibrator.predict(scores)
        return output


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
        sample_weights = _sample_weight_array(rows)
        if sample_weights is not None:
            weights = weights * sample_weights
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


@dataclass
class TargetOffsetModel:
    """Model wrapper for race deltas: final score = start/grid base + predicted delta."""

    base_model: object
    base_column: str
    base_fill: float = 0.0
    constraint_mode: str = "constrained"
    calibrators: dict[str, ProbabilityCalibrator] = field(default_factory=dict)

    def _base(self, frame: pd.DataFrame) -> pd.Series:
        if self.base_column not in frame.columns:
            return pd.Series(float(self.base_fill), index=frame.index, dtype=float)
        values = pd.to_numeric(frame[self.base_column], errors="coerce")
        fill = float(self.base_fill)
        if values.notna().sum() > 0:
            candidate = float(values.median(skipna=True))
            if np.isfinite(candidate):
                fill = candidate
        return pd.Series(values.fillna(fill), index=frame.index, dtype=float)

    @staticmethod
    def _series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(float(default), index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").fillna(float(default))

    def _circuit_mobility(self, frame: pd.DataFrame) -> pd.Series:
        circuit_cols = {
            "track_finish_order_mobility",
            "track_overtake_propensity",
            "circuit_drs_effectiveness",
            "circuit_overtaking_difficulty",
            "track_chaos_index",
            "race_generation_variance_prior",
            "track_strategy_variance_prior",
        }
        if not any(col in frame.columns for col in circuit_cols):
            return pd.Series(1.0, index=frame.index, dtype=float)
        mobility = (
            self._series(frame, "track_finish_order_mobility", 0.35)
            if "track_finish_order_mobility" in frame.columns
            else self._series(frame, "track_overtake_propensity", 0.35)
        ).clip(0.0, 1.0)
        drs = self._series(frame, "circuit_drs_effectiveness", 0.45).clip(0.0, 1.0)
        difficulty = self._series(frame, "circuit_overtaking_difficulty", 0.50).clip(0.0, 1.0)
        variance = self._series(frame, "race_generation_variance_prior", 0.20).clip(0.0, 1.0)
        strategy = self._series(frame, "track_strategy_variance_prior", 0.35).clip(0.0, 1.0)
        mobility = (
            0.08
            + (0.58 * mobility)
            + (0.18 * drs)
            - (0.40 * difficulty)
            + (0.20 * variance)
            + (0.10 * strategy)
        )
        return mobility.clip(lower=0.04, upper=0.85)

    def _constrain_delta(self, frame: pd.DataFrame, delta: pd.Series) -> pd.Series:
        if str(self.constraint_mode).strip().lower() == "unconstrained":
            return delta
        mobility = self._circuit_mobility(frame).reindex(delta.index).fillna(1.0)
        if (mobility >= 0.999).all():
            return delta
        field_size = max(1.0, float(len(frame)))
        max_delta = np.maximum(1.0, mobility.to_numpy(dtype=float) * max(1.0, field_size - 1.0))
        shrink = (0.20 + (0.80 * mobility)).clip(lower=0.20, upper=1.0)
        values = delta.to_numpy(dtype=float) * shrink.to_numpy(dtype=float)
        values = np.clip(values, -max_delta, max_delta)
        return pd.Series(values, index=delta.index, dtype=float)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=float)
        raw_delta = self.base_model.predict(frame)
        delta = pd.Series(raw_delta, index=frame.index, dtype=float)
        return self._base(frame) + self._constrain_delta(frame, delta)

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


def _prepare_training_rows(
    frame: pd.DataFrame,
    *,
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if frame.empty or target_col not in frame.columns:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
    y = pd.to_numeric(frame[target_col], errors="coerce")
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


def _target_base_fill(frame: pd.DataFrame, base_col: Optional[str], default: float = 0.0) -> float:
    if not base_col or frame.empty or base_col not in frame.columns:
        return float(default)
    values = pd.to_numeric(frame[base_col], errors="coerce")
    if values.notna().sum() == 0:
        return float(default)
    fill = float(values.median(skipna=True))
    return fill if np.isfinite(fill) else float(default)


def _infer_target_spec(train: pd.DataFrame) -> TargetSpec:
    if train.empty:
        return TargetSpec()
    if {"target", "race_delta_target", "grid_position"}.issubset(train.columns):
        delta = pd.to_numeric(train["race_delta_target"], errors="coerce")
        actual = pd.to_numeric(train["target"], errors="coerce")
        grid = pd.to_numeric(train["grid_position"], errors="coerce")
        valid = delta.notna() & actual.notna() & grid.notna()
        if valid.sum() > 0:
            return TargetSpec(
                train_col="race_delta_target",
                actual_col="target",
                base_col="grid_position",
                base_fill=_target_base_fill(train.loc[valid], "grid_position", default=10.0),
                label="race_grid_delta",
            )
    return TargetSpec()


def _wrap_target_offset(model: object, target_spec: TargetSpec, frame: pd.DataFrame) -> object:
    if not target_spec.uses_offset or not target_spec.base_col:
        return model
    return TargetOffsetModel(
        base_model=model,
        base_column=target_spec.base_col,
        base_fill=_target_base_fill(frame, target_spec.base_col, target_spec.base_fill),
        constraint_mode=target_spec.constraint_mode,
    )


def _sample_weight_array(frame: pd.DataFrame) -> Optional[np.ndarray]:
    if frame.empty or SAMPLE_WEIGHT_COL not in frame.columns:
        return None
    weights = pd.to_numeric(frame[SAMPLE_WEIGHT_COL], errors="coerce")
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1e-6)
    arr = weights.to_numpy(dtype=float)
    if arr.size == 0 or np.allclose(arr, 1.0):
        return None
    return arr


def _event_group_weights(event_sorted: pd.Series, row_weights: np.ndarray) -> list[float]:
    output: list[float] = []
    start = 0
    for size in _group_sizes_from_sorted_event(event_sorted):
        stop = start + size
        group_weight = float(np.nanmean(row_weights[start:stop])) if stop > start else 1.0
        output.append(group_weight if np.isfinite(group_weight) and group_weight > 0.0 else 1.0)
        start = stop
    return output


def _fit_with_optional_weights(model: object, *args: object, sample_weight: Optional[np.ndarray] = None, **kwargs: object) -> None:
    if sample_weight is None:
        model.fit(*args, **kwargs)
        return
    try:
        model.fit(*args, sample_weight=sample_weight, **kwargs)
    except TypeError:
        model.fit(*args, **kwargs)
    except ValueError:
        model.fit(*args, **kwargs)


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
    target_spec: Optional[TargetSpec] = None,
) -> Optional[object]:
    spec = target_spec or TargetSpec()
    if spec.uses_offset and candidate.task == "ranking":
        return None
    rows, y, event_key = _prepare_training_rows(train_df, target_col=spec.train_col)
    if rows.empty:
        return None
    model = candidate.build_model()
    try:
        if candidate.task == "strategic_race":
            if not spec.uses_offset:
                return None
            strategic_rows = rows.copy()
            if "race_delta_target" not in strategic_rows.columns:
                strategic_rows["race_delta_target"] = y
            model.fit(strategic_rows)
            return model

        if candidate.task == "eb":
            eb_rows = rows.copy()
            eb_rows["target"] = y
            model.fit(eb_rows)
            return _wrap_target_offset(model, spec, rows)

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
            row_weights = _sample_weight_array(rows_sorted)
            group_weights = _event_group_weights(event_sorted, row_weights) if row_weights is not None else None
            _fit_with_optional_weights(
                model,
                X_train,
                y_rank.to_numpy(dtype=float),
                sample_weight=np.asarray(group_weights, dtype=float) if group_weights is not None else None,
                group=group,
            )
        else:
            X_train = preprocessor.fit_transform(rows)
            _fit_with_optional_weights(
                model,
                X_train,
                y.to_numpy(dtype=float),
                sample_weight=_sample_weight_array(rows),
            )
    except Exception:
        return None
    fitted = FittedModel(
        estimator=model,
        preprocessor=preprocessor,
        model_name=candidate.name,
        family=candidate.family,
        task=candidate.task,
        device_used=getattr(model, "device_used", None),
    )
    return _wrap_target_offset(fitted, spec, rows)


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


def _frame_mean_weight(frame: pd.DataFrame) -> float:
    weights = _sample_weight_array(frame)
    if weights is None:
        return 1.0
    value = float(np.nanmean(weights))
    return value if np.isfinite(value) and value > 0.0 else 1.0


def _weighted_metric_mean(items: list[tuple[dict[str, float], float]], key: str, default: float) -> float:
    values: list[float] = []
    weights: list[float] = []
    for metrics, weight in items:
        value = float(metrics.get(key, default))
        if not np.isfinite(value):
            continue
        values.append(value)
        weights.append(float(weight) if np.isfinite(weight) and weight > 0.0 else 1.0)
    if not values:
        return float(default)
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


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
    target_spec: Optional[TargetSpec] = None,
) -> Optional[CandidateScore]:
    spec = target_spec or TargetSpec()
    event_key = pd.to_numeric(train["event_key"], errors="coerce") if "event_key" in train.columns else None
    fold_metrics: list[tuple[dict[str, float], float]] = []
    for train_keys, val_key in folds:
        if event_key is None:
            break
        train_df = train.loc[event_key.isin(train_keys)]
        val_df = train.loc[event_key == val_key]
        if train_df.empty or val_df.empty:
            continue
        fitted = _fit_candidate(train_df, feature_cols, candidate, target_spec=spec)
        if fitted is None:
            return None
        val_rows, y_val, event_val = _prepare_training_rows(val_df, target_col=spec.actual_col)
        if val_rows.empty:
            continue
        preds = pd.Series(fitted.predict(val_rows), index=val_rows.index, dtype=float)
        fold_metrics.append((_fold_metrics(y_val, preds, event_val), _frame_mean_weight(val_df)))
    if not fold_metrics:
        return None
    mae = _weighted_metric_mean(fold_metrics, "mae", float("inf"))
    spearman = _weighted_metric_mean(fold_metrics, "spearman", 0.0)
    ndcg10 = _weighted_metric_mean(fold_metrics, "ndcg10", 0.0)
    hit10 = _weighted_metric_mean(fold_metrics, "hit10", 0.0)
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
    target_spec: Optional[TargetSpec] = None,
) -> Optional[CandidateScore]:
    spec = target_spec or TargetSpec()
    if column not in train.columns:
        return None
    event_key = pd.to_numeric(train["event_key"], errors="coerce") if "event_key" in train.columns else None
    if event_key is None:
        return None
    fold_metrics: list[tuple[dict[str, float], float]] = []
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
        val_rows, y_val, event_val = _prepare_training_rows(val_df, target_col=spec.actual_col)
        if val_rows.empty:
            continue
        pred = pd.to_numeric(val_rows.get(column), errors="coerce")
        pred = pred.fillna(fill_value)
        pred = pd.Series(pred.to_numpy(dtype=float), index=val_rows.index, dtype=float)
        fold_metrics.append((_fold_metrics(y_val, pred, event_val), _frame_mean_weight(val_df)))
    if not fold_metrics:
        return None
    mae = _weighted_metric_mean(fold_metrics, "mae", float("inf"))
    spearman = _weighted_metric_mean(fold_metrics, "spearman", 0.0)
    ndcg10 = _weighted_metric_mean(fold_metrics, "ndcg10", 0.0)
    hit10 = _weighted_metric_mean(fold_metrics, "hit10", 0.0)
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
    target_spec: Optional[TargetSpec] = None,
) -> list[CandidateScore]:
    spec = target_spec or TargetSpec()
    if baseline_col not in train.columns:
        return []
    event_key = pd.to_numeric(train["event_key"], errors="coerce") if "event_key" in train.columns else None
    if event_key is None:
        return []
    weights = sorted({float(np.clip(w, 0.0, 1.0)) for w in model_weights})
    if not weights:
        return []
    fold_metrics_by_weight: dict[float, list[tuple[dict[str, float], float]]] = {w: [] for w in weights}
    for train_keys, val_key in folds:
        train_df = train.loc[event_key.isin(train_keys)]
        val_df = train.loc[event_key == val_key]
        if train_df.empty or val_df.empty:
            continue
        fitted = _fit_candidate(train_df, feature_cols, candidate, target_spec=spec)
        if fitted is None:
            return []
        val_rows, y_val, event_val = _prepare_training_rows(val_df, target_col=spec.actual_col)
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
            fold_metrics_by_weight[model_weight].append(
                (_fold_metrics(y_val, blend, event_val), _frame_mean_weight(val_df)),
            )

    output: list[CandidateScore] = []
    for model_weight in weights:
        fold_metrics = fold_metrics_by_weight[model_weight]
        if not fold_metrics:
            continue
        mae = _weighted_metric_mean(fold_metrics, "mae", float("inf"))
        spearman = _weighted_metric_mean(fold_metrics, "spearman", 0.0)
        ndcg10 = _weighted_metric_mean(fold_metrics, "ndcg10", 0.0)
        hit10 = _weighted_metric_mean(fold_metrics, "hit10", 0.0)
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


def _logsumexp(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    max_value = float(np.max(values))
    if not np.isfinite(max_value):
        return 0.0
    return max_value + float(np.log(np.exp(values - max_value).sum()))


def _pl_event_nll(scores: pd.Series, actual: pd.Series, temperature: float) -> float:
    valid = scores.notna() & actual.notna()
    if valid.sum() <= 1:
        return 0.0
    score_values = pd.to_numeric(scores.loc[valid], errors="coerce")
    actual_values = pd.to_numeric(actual.loc[valid], errors="coerce")
    valid = score_values.notna() & actual_values.notna()
    if valid.sum() <= 1:
        return 0.0
    score_values = score_values.loc[valid]
    actual_values = actual_values.loc[valid]
    order = actual_values.sort_values(ascending=True, kind="mergesort").index
    utility = (-score_values.loc[order].to_numpy(dtype=float)) / float(max(temperature, 1e-6))
    remaining = list(range(len(utility)))
    nll = 0.0
    for pos in range(len(utility)):
        chosen = remaining[0]
        rem_values = utility[remaining]
        nll += _logsumexp(rem_values) - float(utility[chosen])
        remaining.pop(0)
        if not remaining:
            break
    return float(nll / max(1, len(utility)))


def _pl_oof_nll(
    scores: pd.Series,
    actual: pd.Series,
    event_key: pd.Series,
    temperature: float,
) -> float:
    values: list[float] = []
    for idx in _event_groups(event_key, scores.index):
        event_nll = _pl_event_nll(scores.loc[idx], actual.loc[idx], temperature)
        if np.isfinite(event_nll):
            values.append(float(event_nll))
    return float(np.mean(values)) if values else float("inf")


def _fit_pl_temperature_from_oof(
    scores: pd.Series,
    actual: pd.Series,
    event_key: pd.Series,
) -> tuple[Optional[float], dict[str, Any]]:
    valid = scores.notna() & actual.notna() & event_key.notna()
    scores = pd.to_numeric(scores.loc[valid], errors="coerce")
    actual = pd.to_numeric(actual.loc[valid], errors="coerce")
    event_key = pd.to_numeric(event_key.loc[valid], errors="coerce")
    valid = scores.notna() & actual.notna() & event_key.notna()
    scores = scores.loc[valid]
    actual = actual.loc[valid]
    event_key = event_key.loc[valid]
    event_count = int(event_key.nunique(dropna=True))
    if len(scores) < 8 or event_count < 2:
        return None, {
            "available": False,
            "reason": "insufficient_oof_events",
            "row_count": int(len(scores)),
            "event_count": event_count,
        }

    coarse = np.asarray([0.20, 0.30, 0.45, 0.65, 0.90, 1.20, 1.60, 2.20, 3.20, 4.80, 7.20], dtype=float)
    scored = [(float(temp), _pl_oof_nll(scores, actual, event_key, float(temp))) for temp in coarse]
    scored = [(temp, nll) for temp, nll in scored if np.isfinite(nll)]
    if not scored:
        return None, {
            "available": False,
            "reason": "nll_unavailable",
            "row_count": int(len(scores)),
            "event_count": event_count,
        }
    best_temp, _ = min(scored, key=lambda item: item[1])
    low = max(0.10, best_temp / 1.8)
    high = min(10.0, best_temp * 1.8)
    fine = np.linspace(low, high, num=25)
    fine_scored = [(float(temp), _pl_oof_nll(scores, actual, event_key, float(temp))) for temp in fine]
    fine_scored = [(temp, nll) for temp, nll in fine_scored if np.isfinite(nll)]
    if fine_scored:
        best_temp, best_nll = min(fine_scored, key=lambda item: item[1])
    else:
        best_temp, best_nll = min(scored, key=lambda item: item[1])
    return float(best_temp), {
        "available": True,
        "source": "walk_forward_oof",
        "objective": "plackett_luce_negative_log_likelihood",
        "temperature": float(best_temp),
        "nll": float(best_nll),
        "row_count": int(len(scores)),
        "event_count": event_count,
    }


def _pl_probabilities_for_oof_audit(
    scores: pd.Series,
    event_key: pd.Series,
    temperature: float,
    *,
    samples: int = DEFAULT_PL_SAMPLES,
    seed: int = DEFAULT_PL_SEED,
) -> pd.DataFrame:
    sampled = pl_gumbel_probabilities(
        scores=scores,
        event_key=event_key,
        samples=int(max(1, samples)),
        temperature=float(temperature),
        seed=int(seed),
    )
    return pd.DataFrame(
        {
            "win": sampled["p_win"].reindex(scores.index).fillna(0.0),
            "top3": sampled["p_top3"].reindex(scores.index).fillna(0.0),
            "top10": sampled["p_top10"].reindex(scores.index).fillna(0.0),
        },
        index=scores.index,
    ).clip(lower=0.0, upper=1.0)


def _probability_event_total_audit(probabilities: pd.DataFrame, event_key: pd.Series) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    passed = True
    for label, expected_k in [("win", 1), ("top3", 3), ("top10", 10)]:
        deviations: list[float] = []
        event_count = 0
        for idx in _event_groups(event_key, probabilities.index):
            p = pd.to_numeric(probabilities[label].reindex(idx), errors="coerce").dropna()
            if p.empty:
                continue
            event_count += 1
            expected = min(float(expected_k), float(len(p)))
            deviations.append(abs(float(p.sum()) - expected))
        max_abs_error = float(max(deviations)) if deviations else float("inf")
        label_passed = bool(np.isfinite(max_abs_error) and max_abs_error <= 0.08)
        passed = passed and label_passed
        checks[label] = {
            "passed": label_passed,
            "event_count": int(event_count),
            "max_abs_error": max_abs_error,
            "expected_total": float(expected_k),
        }
    return {"passed": bool(passed), "markets": checks}


def _binary_log_loss(labels: pd.Series, probabilities: pd.Series) -> float:
    valid = labels.notna() & probabilities.notna()
    if valid.sum() == 0:
        return float("nan")
    y = labels.loc[valid].to_numpy(dtype=float)
    p = np.clip(probabilities.loc[valid].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _calibration_slope(labels: pd.Series, probabilities: pd.Series) -> Optional[float]:
    valid = labels.notna() & probabilities.notna()
    if valid.sum() < 8:
        return None
    y = labels.loc[valid].astype(float)
    if y.nunique(dropna=True) < 2:
        return None
    p = probabilities.loc[valid].astype(float).clip(1e-6, 1.0 - 1e-6)
    x = _safe_logit(p.to_numpy(dtype=float))
    if np.nanstd(x) < 1e-9:
        return None
    if LogisticRegression is not None:
        try:
            model = LogisticRegression(solver="lbfgs", max_iter=400)
            model.fit(x.reshape(-1, 1), y.astype(int).to_numpy())
            return float(model.coef_[0][0])
        except Exception:
            pass
    return float(np.cov(x, y.to_numpy(dtype=float), ddof=0)[0, 1] / np.var(x))


def _bootstrap_probability_delta_ci(
    labels: pd.Series,
    probabilities: pd.Series,
    baseline_probabilities: pd.Series,
    event_key: pd.Series,
    *,
    samples: int = DEFAULT_PROBABILITY_AUDIT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_PL_SEED,
) -> dict[str, Any]:
    valid = labels.notna() & probabilities.notna() & baseline_probabilities.notna() & event_key.notna()
    if valid.sum() == 0:
        return {"available": False, "reason": "empty"}
    y = labels.loc[valid].astype(float)
    p = probabilities.loc[valid].astype(float).clip(1e-6, 1.0 - 1e-6)
    base = baseline_probabilities.loc[valid].astype(float).clip(1e-6, 1.0 - 1e-6)
    events = event_key.loc[valid]
    unique_events = pd.unique(events)
    if len(unique_events) < 2:
        return {"available": False, "reason": "insufficient_events", "event_count": int(len(unique_events))}

    rng = np.random.default_rng(int(seed))
    brier_deltas: list[float] = []
    log_loss_deltas: list[float] = []
    for _ in range(int(max(1, samples))):
        sampled_events = rng.choice(unique_events, size=len(unique_events), replace=True)
        sampled_index_parts = [events[events == event].index for event in sampled_events]
        if not sampled_index_parts:
            continue
        sampled_index = sampled_index_parts[0].append(sampled_index_parts[1:]) if len(sampled_index_parts) > 1 else sampled_index_parts[0]
        y_s = y.loc[sampled_index]
        p_s = p.loc[sampled_index]
        base_s = base.loc[sampled_index]
        brier_deltas.append(float(((p_s - y_s) ** 2).mean() - ((base_s - y_s) ** 2).mean()))
        model_loss = -(y_s * np.log(p_s) + (1.0 - y_s) * np.log(1.0 - p_s)).mean()
        baseline_loss = -(y_s * np.log(base_s) + (1.0 - y_s) * np.log(1.0 - base_s)).mean()
        log_loss_deltas.append(float(model_loss - baseline_loss))
    if not brier_deltas or not log_loss_deltas:
        return {"available": False, "reason": "bootstrap_empty", "event_count": int(len(unique_events))}
    brier_arr = np.asarray(brier_deltas, dtype=float)
    log_loss_arr = np.asarray(log_loss_deltas, dtype=float)
    return {
        "available": True,
        "event_count": int(len(unique_events)),
        "samples": int(max(1, samples)),
        "brier_delta_ci95": [float(np.nanpercentile(brier_arr, 2.5)), float(np.nanpercentile(brier_arr, 97.5))],
        "log_loss_delta_ci95": [
            float(np.nanpercentile(log_loss_arr, 2.5)),
            float(np.nanpercentile(log_loss_arr, 97.5)),
        ],
    }


def _probability_audit_from_oof(
    scores: pd.Series,
    actual: pd.Series,
    event_key: pd.Series,
    temperature: Optional[float],
    temperature_audit: dict[str, Any],
    *,
    samples: int = DEFAULT_PL_SAMPLES,
    seed: int = DEFAULT_PL_SEED,
) -> dict[str, Any]:
    if temperature is None:
        return {
            "available": False,
            "passed": False,
            "reason": temperature_audit.get("reason", "temperature_unavailable"),
            "temperature_fit": dict(temperature_audit),
        }
    valid = scores.notna() & actual.notna() & event_key.notna()
    scores = scores.loc[valid]
    actual = actual.loc[valid]
    event_key = event_key.loc[valid]
    if scores.empty:
        return {
            "available": False,
            "passed": False,
            "reason": "oof_scores_empty",
            "temperature_fit": dict(temperature_audit),
        }
    probabilities = _pl_probabilities_for_oof_audit(
        scores,
        event_key,
        float(temperature),
        samples=int(samples),
        seed=int(seed),
    )
    total_audit = _probability_event_total_audit(probabilities, event_key)
    thresholds = {
        "max_brier": 0.30,
        "max_log_loss": 1.25,
        "baseline_tolerance": 0.005,
        "max_brier_delta_ci95_upper": 0.005,
        "max_log_loss_delta_ci95_upper": 0.005,
        "min_calibration_slope": 0.35,
        "max_calibration_slope": 2.75,
        "min_oof_events": DEFAULT_PROBABILITY_AUDIT_MIN_EVENTS,
        "bootstrap_samples": DEFAULT_PROBABILITY_AUDIT_BOOTSTRAP_SAMPLES,
    }
    metrics: dict[str, dict[str, Any]] = {}
    passed = bool(total_audit.get("passed", False))
    reasons: list[str] = [] if passed else ["event_total_failed"]
    audit_event_count = int(pd.to_numeric(event_key, errors="coerce").nunique(dropna=True))
    if audit_event_count < int(thresholds["min_oof_events"]):
        passed = False
        reasons.append("insufficient_probability_audit_events")
    for label, k in [("win", 1), ("top3", 3), ("top10", 10)]:
        y = _topk_labels_from_target(actual, event_key, k=k)
        p = probabilities[label].reindex(y.index)
        valid_label = y.notna() & p.notna()
        if valid_label.sum() == 0:
            metrics[label] = {"available": False, "reason": "empty_label"}
            passed = False
            reasons.append(f"{label}_empty")
            continue
        brier = float(((p.loc[valid_label] - y.loc[valid_label]) ** 2).mean())
        log_loss = _binary_log_loss(y.loc[valid_label], p.loc[valid_label])
        slope = _calibration_slope(y.loc[valid_label], p.loc[valid_label])
        baseline_p = pd.Series(index=y.index, dtype=float)
        for idx in _event_groups(event_key.reindex(y.index), y.index):
            n_event = max(1, len(idx))
            baseline_p.loc[idx] = min(float(k), float(n_event)) / float(n_event)
        baseline_p = baseline_p.reindex(y.index).fillna(float(y.loc[valid_label].mean()))
        baseline_brier = float(((baseline_p.loc[valid_label] - y.loc[valid_label]) ** 2).mean())
        baseline_log_loss = _binary_log_loss(y.loc[valid_label], baseline_p.loc[valid_label])
        brier_delta = float(brier - baseline_brier)
        log_loss_delta = float(log_loss - baseline_log_loss)
        bootstrap = _bootstrap_probability_delta_ci(
            y.loc[valid_label],
            p.loc[valid_label],
            baseline_p.loc[valid_label],
            event_key.reindex(valid_label[valid_label].index),
            samples=int(thresholds["bootstrap_samples"]),
            seed=int(seed) + int(k),
        )
        metric = {
            "available": True,
            "brier": brier,
            "log_loss": log_loss,
            "baseline_brier": baseline_brier,
            "baseline_log_loss": baseline_log_loss,
            "brier_delta_vs_baseline": brier_delta,
            "log_loss_delta_vs_baseline": log_loss_delta,
            "bootstrap": bootstrap,
            "calibration_slope": slope,
            "base_rate": float(y.loc[valid_label].mean()),
            "row_count": int(valid_label.sum()),
        }
        ci_gate_passed = False
        if bool(bootstrap.get("available", False)):
            brier_ci = bootstrap.get("brier_delta_ci95")
            log_loss_ci = bootstrap.get("log_loss_delta_ci95")
            if isinstance(brier_ci, list) and len(brier_ci) == 2 and isinstance(log_loss_ci, list) and len(log_loss_ci) == 2:
                brier_upper = pd.to_numeric(pd.Series([brier_ci[1]]), errors="coerce").iloc[0]
                log_loss_upper = pd.to_numeric(pd.Series([log_loss_ci[1]]), errors="coerce").iloc[0]
                ci_gate_passed = bool(
                    pd.notna(brier_upper)
                    and pd.notna(log_loss_upper)
                    and float(brier_upper) <= float(thresholds["max_brier_delta_ci95_upper"])
                    and float(log_loss_upper) <= float(thresholds["max_log_loss_delta_ci95_upper"])
                )
        metric["ci_gate_passed"] = bool(ci_gate_passed)
        metric_passed = (
            brier <= thresholds["max_brier"]
            and log_loss <= thresholds["max_log_loss"]
            and brier <= baseline_brier + thresholds["baseline_tolerance"]
            and log_loss <= baseline_log_loss + thresholds["baseline_tolerance"]
            and slope is not None
            and thresholds["min_calibration_slope"] <= slope <= thresholds["max_calibration_slope"]
            and bool(bootstrap.get("available", False))
            and ci_gate_passed
        )
        metric["passed"] = bool(metric_passed)
        if not metric_passed:
            passed = False
            reasons.append(f"{label}_calibration_failed")
        metrics[label] = metric

    return {
        "available": True,
        "passed": bool(passed),
        "reason": "passed" if passed else ",".join(reasons),
        "schema_version": PROBABILITY_AUDIT_SCHEMA_VERSION,
        "source": "walk_forward_oof",
        "probability_layer": "pl_gumbel",
        "same_probability_layer_as_production": True,
        "samples": int(max(1, samples)),
        "seed": int(seed),
        "temperature": float(temperature),
        "temperature_fit": dict(temperature_audit),
        "event_total_audit": total_audit,
        "thresholds": thresholds,
        "metrics": metrics,
        "row_count": int(len(scores)),
        "event_count": int(pd.to_numeric(event_key, errors="coerce").nunique(dropna=True)),
    }


def _oof_scores_for_selection(
    *,
    train: pd.DataFrame,
    feature_cols: List[str],
    selected_name: str,
    candidate_lookup: dict[str, CandidateSpec],
    folds: list[tuple[set[int], int]],
    race_baseline_col: str,
    target_spec: Optional[TargetSpec] = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    spec = target_spec or TargetSpec()
    event_key = pd.to_numeric(train["event_key"], errors="coerce") if "event_key" in train.columns else None
    if event_key is None or not folds:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    score_parts: list[pd.Series] = []
    target_parts: list[pd.Series] = []
    event_parts: list[pd.Series] = []

    def _median_fill(frame: pd.DataFrame, column: str, default_fill: float) -> float:
        values = pd.to_numeric(frame.get(column), errors="coerce")
        if values.notna().sum() == 0:
            return float(default_fill)
        fill = float(values.median(skipna=True))
        if not np.isfinite(fill):
            return float(default_fill)
        return fill

    for train_keys, val_key in folds:
        train_df = train.loc[event_key.isin(train_keys)]
        val_df = train.loc[event_key == val_key]
        if train_df.empty or val_df.empty:
            continue

        val_rows, y_val, event_val = _prepare_training_rows(val_df, target_col=spec.actual_col)
        if val_rows.empty:
            continue

        model: Optional[object] = None
        if selected_name == "qualifying_baseline":
            model = QualifyingPositionBaseline(
                fill_value=_median_fill(train_df, race_baseline_col, 10.0),
                primary_column=race_baseline_col,
                fallback_column="qualy_position",
            )
        elif selected_name.startswith("pace_baseline::"):
            _, _, baseline_col = selected_name.partition("::")
            baseline_col = baseline_col.strip()
            if baseline_col:
                model = ColumnBaselineModel(
                    column=baseline_col,
                    fill_value=_median_fill(train_df, baseline_col, 0.0),
                )
        elif selected_name.startswith("pace_blend::"):
            parts = selected_name.split("::")
            if len(parts) == 4:
                _, candidate_name, baseline_col, weight_text = parts
                candidate = candidate_lookup.get(candidate_name)
                try:
                    model_weight = float(weight_text)
                except ValueError:
                    model_weight = 0.30
                if candidate is not None and candidate.family != "dl":
                    fitted = _fit_candidate(train_df, feature_cols, candidate, target_spec=spec)
                    if fitted is not None:
                        model = WeightedBlendModel(
                            primary_model=fitted,
                            baseline_column=baseline_col,
                            model_weight=model_weight,
                            baseline_fill=_median_fill(train_df, baseline_col, 0.0),
                        )
        else:
            candidate = candidate_lookup.get(selected_name)
            if candidate is not None:
                model = _fit_candidate(train_df, feature_cols, candidate, target_spec=spec)

        if model is None:
            continue

        try:
            scores = pd.Series(model.predict(val_rows), index=val_rows.index, dtype=float)
        except Exception:
            continue
        score_parts.append(scores)
        target_parts.append(pd.Series(y_val, index=val_rows.index, dtype=float))
        event_parts.append(pd.Series(event_val, index=val_rows.index, dtype=float))

    if not score_parts:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    return (
        pd.concat(score_parts).sort_index(),
        pd.concat(target_parts).sort_index(),
        pd.concat(event_parts).sort_index(),
    )


def _normalize_requested_model(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    allowed = {"auto", "baseline", "xgb_rank", "eb_rank", "lgbm_rank"}
    if normalized in allowed:
        return normalized
    return "auto"


def _normalize_constraint_mode(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"unconstrained", "none", "off"}:
        return "unconstrained"
    return "constrained"


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
    f1_pl_samples: int = DEFAULT_PL_SAMPLES,
    f1_listwise_seed: int = DEFAULT_PL_SEED,
    race_delta_constraint_mode: str = "constrained",
) -> TrainingResult:
    notes: List[str] = []
    leaderboard_data: List[dict[str, Any]] = []
    compare = compare_families or ["ml"]
    dl_hparams = dict(dl_hyperparams or {})
    requested_model = _normalize_requested_model(f1_model)
    dl_available = torch_available()
    dl_requested = enable_dl_candidates and ("dl" in {str(f).strip().lower() for f in compare})
    if dl_requested:
        notes.append("DL candidates are shadow-only for F1 model selection; they are evaluated but cannot be selected by default.")

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
    target_spec = _infer_target_spec(train)
    target_spec = replace(target_spec, constraint_mode=_normalize_constraint_mode(race_delta_constraint_mode))
    strategic_race_supported = bool(target_spec.uses_offset and "grid_position" in train.columns)
    if target_spec.uses_offset:
        notes.append(
            "Race target transform active: models train on race_delta_target and "
            "validation/calibration score reconstructed finish_position=grid_position+predicted_delta.",
        )
        if strategic_race_supported:
            candidates = [
                CandidateSpec(
                    name="strategic_race_delta",
                    task="strategic_race",
                    family="baseline",
                    build_model=lambda: StrategicRaceDeltaModel(),
                ),
            ]
            notes.append(
                "Race model policy: unified strategic_race_delta active "
                "(grid anchor + FP race pace + driver/team/circuit deltas + circuit-card/car-fit features).",
            )
        if any(candidate.task == "ranking" for candidate in candidates):
            notes.append(
                "Ranking candidates skipped for race_delta_target: pairwise relevance on raw deltas would optimize movers, not finish order.",
            )
    race_baseline_col = (
        "grid_position"
        if target_spec.uses_offset and "grid_position" in feature_cols
        else ("qualy_context_position" if "qualy_context_position" in feature_cols else "qualy_position")
    )
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
                score = _evaluate_candidate(train, feature_cols, candidate, folds, target_spec=target_spec)
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
                target_spec=target_spec,
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
                    target_spec=target_spec,
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
                    target_spec=target_spec,
                )
                if baseline_score is not None:
                    score_lookup[baseline_score.name] = baseline_score

            if not small_history_qualifying:
                blend_weights = [0.20, 0.35, 0.50, 0.65, 0.80]
                for candidate in candidates:
                    if candidate.family == "dl":
                        continue
                    for blend_col in qualifying_baseline_cols:
                        blend_scores = _evaluate_blend_candidates(
                            train=train,
                            feature_cols=feature_cols,
                            candidate=candidate,
                            folds=folds,
                            baseline_col=blend_col,
                            model_weights=blend_weights,
                            default_baseline_fill=0.0,
                            target_spec=target_spec,
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
            eligible_ranking = [score for score in ranking if score.family != "dl"]
            if strategic_race_supported and "strategic_race_delta" in candidate_lookup:
                selected_from_cv = "strategic_race_delta"
                if "strategic_race_delta" in score_lookup:
                    notes.append(
                        "Race policy kept the unified strategic race model as production model; "
                        "grid/pace baselines remain audit comparators only.",
                    )
            elif eligible_ranking:
                selected_from_cv = eligible_ranking[0].name
                if ranking[0].family == "dl":
                    notes.append(
                        "Best composite candidate is DL, but DL is shadow-only; selecting best non-DL candidate instead.",
                    )
            else:
                notes.append("Only DL candidates were ranked; DL shadow-only policy blocks selection.")
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
                    selected_from_cv = None
            if selected_from_cv and selected_from_cv in score_lookup:
                notes.append(
                    f"Modele retenu: {selected_from_cv} "
                    f"(score composite={score_lookup[selected_from_cv].composite:.3f}).",
                )
            elif selected_from_cv:
                notes.append(f"Modele retenu par politique conservatrice: {selected_from_cv}.")
        else:
            notes.append("Aucun score walk-forward exploitable; selection par priorite.")
    else:
        notes.append("Historique insuffisant pour validation walk-forward, selection par priorite.")
        if strategic_race_supported and "strategic_race_delta" in candidate_lookup:
            selected_from_cv = "strategic_race_delta"
            notes.append("Mode race: modele strategique unifie prioritaire sans folds.")
        elif race_baseline_supported:
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
        if strategic_race_supported and "strategic_race_delta" in candidate_lookup:
            selected_from_cv = "strategic_race_delta"
        elif race_baseline_supported:
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
            if candidate is not None and candidate.family != "dl":
                fitted = _fit_candidate(train, feature_cols, candidate, target_spec=target_spec)
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
        candidate_order = [candidate for candidate in candidate_order if candidate.family != "dl"]
        if not candidate_order and dl_requested:
            notes.append("Only DL candidates were available; DL shadow-only policy forces heuristic/baseline fallback.")
        for candidate in candidate_order:
            fitted = _fit_candidate(train, feature_cols, candidate, target_spec=target_spec)
            if fitted is None:
                continue
            selected_model = fitted
            selected_name = candidate.name
            selected_family = candidate.family
            selected_device = getattr(fitted, "device_used", None)
            break

    if selected_model is None and race_baseline_supported:
        if strategic_race_supported and "strategic_race_delta" in candidate_lookup:
            fitted = _fit_candidate(
                train,
                feature_cols,
                candidate_lookup["strategic_race_delta"],
                target_spec=target_spec,
            )
            if fitted is not None:
                selected_model = fitted
                selected_name = "strategic_race_delta"
                selected_family = "baseline"
                notes.append("Fallback race: modele strategique unifie ajuste sur tout l'historique.")
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

    rows, y_train, event_train = _prepare_training_rows(train, target_col=target_spec.actual_col)
    calibration_scores = pd.Series(dtype=float)
    calibration_y = pd.Series(dtype=float)
    calibration_event = pd.Series(dtype=float)
    calibration_source = "unavailable"
    if selected_name is not None:
        calibration_scores, calibration_y, calibration_event = _oof_scores_for_selection(
            train=train,
            feature_cols=feature_cols,
            selected_name=selected_name or "",
            candidate_lookup=candidate_lookup,
            folds=folds,
            race_baseline_col=race_baseline_col,
            target_spec=target_spec,
        )
        calibration_source = "walk-forward_oof"
    if calibration_scores.empty or calibration_event.nunique(dropna=True) < 2:
        if not rows.empty:
            try:
                calibration_scores = pd.Series(selected_model.predict(rows), index=rows.index, dtype=float)
                calibration_y = y_train
                calibration_event = event_train
                calibration_source = "in_sample"
            except Exception:
                calibration_scores = pd.Series(dtype=float)
                calibration_y = pd.Series(dtype=float)
                calibration_event = pd.Series(dtype=float)
                calibration_source = "unavailable"

    listwise_temperature: Optional[float] = None
    probability_audit: dict[str, Any] = {}
    if calibration_source == "walk-forward_oof":
        listwise_temperature, temperature_audit = _fit_pl_temperature_from_oof(
            calibration_scores,
            calibration_y,
            calibration_event,
        )
        probability_audit = _probability_audit_from_oof(
            calibration_scores,
            calibration_y,
            calibration_event,
            listwise_temperature,
            temperature_audit,
            samples=int(f1_pl_samples),
            seed=int(f1_listwise_seed),
        )
        if listwise_temperature is not None:
            nll_value = temperature_audit.get("nll")
            nll_text = f"{float(nll_value):.4f}" if isinstance(nll_value, (int, float)) else "nan"
            notes.append(
                "Listwise PL temperature fitted on walk-forward OOF likelihood: "
                f"temperature={listwise_temperature:.3f}, nll={nll_text}.",
            )
        else:
            notes.append(
                "Listwise PL temperature fitting skipped: "
                f"{temperature_audit.get('reason', 'unavailable')}.",
            )
    else:
        probability_audit = {
            "available": False,
            "passed": False,
            "reason": f"{calibration_source}_probability_audit_not_oof",
            "source": calibration_source,
        }

    if (
        not rows.empty
        and hasattr(selected_model, "calibrators")
        and isinstance(getattr(selected_model, "calibrators"), dict)
    ):
        try:
            win_labels = _topk_labels_from_target(calibration_y, calibration_event, k=1)
            top3_labels = _topk_labels_from_target(calibration_y, calibration_event, k=3)
            top10_labels = _topk_labels_from_target(calibration_y, calibration_event, k=10)
            calibrator_win = _fit_probability_calibrator(calibration_scores, win_labels)
            calibrator_top3 = _fit_probability_calibrator(calibration_scores, top3_labels)
            calibrator_top10 = _fit_probability_calibrator(calibration_scores, top10_labels)
            selected_model.calibrators["win"] = calibrator_win
            selected_model.calibrators["top10"] = calibrator_top10
            selected_model.calibrators["top3"] = calibrator_top3
            notes.append(
                "Calibration probabiliste "
                f"({calibration_source}): win={calibrator_win.method}, "
                f"top3={calibrator_top3.method}, top10={calibrator_top10.method}.",
            )
        except Exception:
            pass

    if selected_name == "qualifying_baseline":
        notes.append("Prediction race: baseline qualif contextuelle (qualif + signal predit).")
    elif selected_name == "strategic_race_delta":
        notes.append(
            "Prediction race: modele strategique unifie (grid anchor, FP race pace, "
            "historique pilote/team/circuit, circuit cards, safety-car/DNF/mobility, car-fit).",
        )
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
        listwise_temperature=listwise_temperature,
        probability_audit=probability_audit,
    )
