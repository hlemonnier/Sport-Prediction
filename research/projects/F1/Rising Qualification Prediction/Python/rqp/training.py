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
    mae = _mean_absolute_error(y_true, pred)
    spearman = _safe_spearman(y_true, pred)
    ndcg_values: list[float] = []
    hit_values: list[float] = []
    for idx in _event_groups(event_key, y_true.index):
        y_event = y_true.loc[idx]
        p_event = pred.loc[idx]
        if y_event.empty:
            continue
        actual_rank = y_event.rank(method="first", ascending=True)
        ndcg_values.append(_ndcg_at_k(actual_rank, p_event, k=10))
        hit_values.append(_topk_hit_rate(actual_rank, p_event, k=10))
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

    candidates = _candidate_models()
    if not candidates:
        notes.append("Aucun modele ML disponible (installer scikit-learn ou xgboost).")
        return TrainingResult(model=None, model_name="heuristic", notes=notes)

    notes.append(
        "Score composite model selection: 0.35*MAE_score + 0.25*Spearman_norm + "
        "0.25*NDCG@10 + 0.15*Top10Hit (MAE_score=1/(1+MAE), Spearman_norm=(rho+1)/2).",
    )

    best_candidate: Optional[CandidateSpec] = None
    folds = _walk_forward_folds(train)
    if folds:
        score_lookup: dict[str, CandidateScore] = {}
        candidate_lookup: dict[str, CandidateSpec] = {c.name: c for c in candidates}
        for candidate in candidates:
            score = _evaluate_candidate(train, feature_cols, candidate, folds)
            if score is None:
                continue
            score_lookup[candidate.name] = score
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
            best_name = ranking[0].name
            best_candidate = candidate_lookup.get(best_name)
            notes.append(f"Modele retenu: {best_name} (score composite={ranking[0].composite:.3f}).")
    else:
        notes.append("Historique insuffisant pour validation walk-forward, selection par priorite.")

    candidate_order: list[CandidateSpec] = []
    if best_candidate is not None:
        candidate_order.append(best_candidate)
    candidate_order.extend([c for c in candidates if c.name != (best_candidate.name if best_candidate else "")])

    fitted_model: Optional[FittedModel] = None
    selected_name: Optional[str] = None
    for candidate in candidate_order:
        fitted = _fit_candidate(train, feature_cols, candidate)
        if fitted is None:
            continue
        fitted_model = fitted
        selected_name = candidate.name
        break

    if fitted_model is None:
        notes.append("Echec entrainement de tous les candidats: fallback heuristique.")
        return TrainingResult(model=None, model_name="heuristic", notes=notes)

    if best_candidate is None and selected_name is not None:
        notes.append(f"Modele retenu par defaut: {selected_name}.")

    rows, y_train, event_train = _prepare_training_rows(train)
    if not rows.empty:
        in_sample_scores = fitted_model.predict(rows)
        top10_labels = _topk_labels_from_target(y_train, event_train, k=10)
        top3_labels = _topk_labels_from_target(y_train, event_train, k=3)
        calibrator_top10 = _fit_probability_calibrator(in_sample_scores, top10_labels)
        calibrator_top3 = _fit_probability_calibrator(in_sample_scores, top3_labels)
        fitted_model.calibrators["top10"] = calibrator_top10
        fitted_model.calibrators["top3"] = calibrator_top3
        notes.append(
            f"Calibration probabiliste: top10={calibrator_top10.method}, top3={calibrator_top3.method}.",
        )

    if SimpleImputer is None:
        notes.append("SimpleImputer indisponible: imputation mediane pandas utilisee.")
    if Ridge is not None and StandardScaler is None:
        notes.append("StandardScaler indisponible: scaling desactive pour Ridge.")

    return TrainingResult(model=fitted_model, model_name=selected_name or "unknown", notes=notes)
