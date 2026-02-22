"""Model training utilities with walk-forward model selection."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, List, Optional

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


@dataclass
class TrainingResult:
    model: Optional[object]
    model_name: str
    notes: List[str]


@dataclass
class CandidateSpec:
    name: str
    build_model: Callable[[], object]
    task: str  # regression | ranking
    scale_features: bool = False


@dataclass
class CandidateScore:
    name: str
    mae: float
    spearman: float
    ndcg10: float
    hit10: float
    composite: float


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
    task: str
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
    """Race baseline that predicts finish order from qualifying order."""

    def __init__(self, fill_value: float = 10.0) -> None:
        self.fill_value = float(fill_value)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.asarray([], dtype=float)
        if "qualy_position" not in frame.columns:
            return np.full(shape=(len(frame),), fill_value=self.fill_value, dtype=float)
        values = pd.to_numeric(frame["qualy_position"], errors="coerce")
        if values.notna().sum() == 0:
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


def _candidate_models() -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    if XGBRanker is not None:
        candidates.append(
            CandidateSpec(
                name="xgboost_pairwise",
                task="ranking",
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
    if XGBRegressor is not None:
        candidates.append(
            CandidateSpec(
                name="xgboost",
                task="regression",
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
    if HistGradientBoostingRegressor is not None:
        candidates.append(
            CandidateSpec(
                name="hist_gradient_boosting",
                task="regression",
                build_model=lambda: HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=5,
                    max_iter=600,
                    random_state=42,
                ),
            ),
        )
    if Ridge is not None:
        candidates.append(
            CandidateSpec(
                name="ridge",
                task="regression",
                scale_features=True,
                build_model=lambda: Ridge(alpha=1.0),
            ),
        )
    return candidates


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
) -> Optional[FittedModel]:
    rows, y, event_key = _prepare_training_rows(train_df)
    if rows.empty:
        return None
    model = candidate.build_model()
    preprocessor = FeaturePipeline(feature_cols=feature_cols, scale=candidate.scale_features)
    try:
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
        task=candidate.task,
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
        mae=mae,
        spearman=spearman,
        ndcg10=ndcg10,
        hit10=hit10,
        composite=composite,
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
                mae=mae,
                spearman=spearman,
                ndcg10=ndcg10,
                hit10=hit10,
                composite=composite,
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


def train_model(train: pd.DataFrame, feature_cols: List[str]) -> TrainingResult:
    notes: List[str] = []
    if train.empty:
        notes.append("Pas assez de data historique: fallback heuristique.")
        return TrainingResult(model=None, model_name="heuristic", notes=notes)

    event_count = 0
    if "event_key" in train.columns:
        event_series = pd.to_numeric(train["event_key"], errors="coerce").dropna()
        if not event_series.empty:
            event_count = int(event_series.astype(int).nunique())

    candidates = _candidate_models()
    race_baseline_supported = "qualy_position" in feature_cols
    qualifying_baseline_cols = [
        col
        for col in ["event_pace_index", "fp_mean_rank", "fp_weighted_delta"]
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

    if not candidates and not race_baseline_supported and not qualifying_baseline_supported:
        notes.append("Aucun modele ML disponible (installer scikit-learn ou xgboost).")
        return TrainingResult(model=None, model_name="heuristic", notes=notes)

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
                column="qualy_position",
                name="qualifying_baseline",
                default_fill=10.0,
            )
            if baseline_score is not None:
                score_lookup[baseline_score.name] = baseline_score
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
            leaderboard = ", ".join(
                (
                    f"{s.name}(C={s.composite:.3f}, MAE={s.mae:.3f}, "
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

    selected_model: Optional[object] = None
    selected_name: Optional[str] = None

    def _median_fill(column: str, default_fill: float) -> float:
        values = pd.to_numeric(train.get(column), errors="coerce")
        if values.notna().sum() == 0:
            return float(default_fill)
        fill = float(values.median(skipna=True))
        if not np.isfinite(fill):
            return float(default_fill)
        return fill

    if selected_from_cv == "qualifying_baseline":
        fill_value = _median_fill("qualy_position", 10.0)
        selected_model = QualifyingPositionBaseline(fill_value=fill_value)
        selected_name = "qualifying_baseline"
    elif selected_from_cv and selected_from_cv.startswith("pace_baseline::"):
        _, _, baseline_col = selected_from_cv.partition("::")
        baseline_col = baseline_col.strip()
        if baseline_col:
            fill_value = _median_fill(baseline_col, 0.0)
            selected_model = ColumnBaselineModel(column=baseline_col, fill_value=fill_value)
            selected_name = selected_from_cv
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
            break

    if selected_model is None and race_baseline_supported:
        fill_value = _median_fill("qualy_position", 10.0)
        selected_model = QualifyingPositionBaseline(fill_value=fill_value)
        selected_name = "qualifying_baseline"
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
        notes.append("Fallback prioritaire active: baseline pace.")

    if selected_model is None:
        notes.append("Echec entrainement de tous les candidats: fallback heuristique.")
        return TrainingResult(model=None, model_name="heuristic", notes=notes)

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
        notes.append("Prediction race: baseline qualifying position (guardrail anti-surapprentissage).")
    elif selected_name and selected_name.startswith("pace_baseline::"):
        notes.append("Prediction qualif: baseline pace local (FP/sprint).")
    elif selected_name and selected_name.startswith("pace_blend::"):
        notes.append("Prediction qualif: blend modele ML + baseline pace.")

    if SimpleImputer is None:
        notes.append("SimpleImputer indisponible: imputation mediane pandas utilisee.")
    if Ridge is not None and StandardScaler is None:
        notes.append("StandardScaler indisponible: scaling desactive pour Ridge.")

    return TrainingResult(model=selected_model, model_name=selected_name or "unknown", notes=notes)
