"""Model training utilities with walk-forward model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
except Exception:  # pragma: no cover - optional dependency
    HistGradientBoostingRegressor = None
    Ridge = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


@dataclass
class TrainingResult:
    model: Optional[object]
    model_name: str
    notes: List[str]


def _candidate_models() -> list[tuple[str, Callable[[], object]]]:
    candidates: list[tuple[str, Callable[[], object]]] = []
    if XGBRegressor is not None:
        candidates.append((
            "xgboost",
            lambda: XGBRegressor(
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
        ))
    if HistGradientBoostingRegressor is not None:
        candidates.append((
            "hist_gradient_boosting",
            lambda: HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=5,
                max_iter=600,
                random_state=42,
            ),
        ))
    if Ridge is not None:
        candidates.append(("ridge", lambda: Ridge(alpha=1.0)))
    return candidates


def _prepare_features(frame: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = frame.reindex(columns=feature_cols).copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)
    return X


def _prepare_training_xy(
    frame: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[pd.DataFrame, pd.Series]:
    X = _prepare_features(frame, feature_cols)
    y = pd.to_numeric(frame["target"], errors="coerce")
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def _mean_absolute_error(y_true: pd.Series, y_pred: object) -> float:
    pred = pd.Series(y_pred, index=y_true.index)
    return float((y_true - pred).abs().mean())


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
    build_model: Callable[[], object],
    folds: list[tuple[set[int], int]],
) -> Optional[float]:
    event_key = pd.to_numeric(train["event_key"], errors="coerce")
    scores: list[float] = []
    for train_keys, val_key in folds:
        train_df = train.loc[event_key.isin(train_keys)]
        val_df = train.loc[event_key == val_key]
        if train_df.empty or val_df.empty:
            continue
        X_train, y_train = _prepare_training_xy(train_df, feature_cols)
        X_val, y_val = _prepare_training_xy(val_df, feature_cols)
        if X_train.empty or X_val.empty:
            continue
        model = build_model()
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
        except Exception:
            return None
        scores.append(_mean_absolute_error(y_val, preds))
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def train_model(train: pd.DataFrame, feature_cols: List[str]) -> TrainingResult:
    notes: List[str] = []
    if train.empty:
        notes.append("Pas assez de data historique: fallback heuristique.")
        return TrainingResult(model=None, model_name="heuristic", notes=notes)

    candidates = _candidate_models()
    if not candidates:
        notes.append("Aucun modele ML disponible (installer scikit-learn ou xgboost).")
        return TrainingResult(model=None, model_name="heuristic", notes=notes)

    best_name: Optional[str] = None
    best_builder: Optional[Callable[[], object]] = None
    folds = _walk_forward_folds(train)
    if folds:
        scores: list[tuple[float, str, Callable[[], object]]] = []
        for name, builder in candidates:
            score = _evaluate_candidate(train, feature_cols, builder, folds)
            if score is None:
                continue
            scores.append((score, name, builder))
        if scores:
            scores.sort(key=lambda x: x[0])
            leaderboard = ", ".join(f"{name}={score:.3f}" for score, name, _ in scores)
            notes.append(f"Model selection (MAE walk-forward): {leaderboard}.")
            best_score, best_name, best_builder = scores[0]
            notes.append(f"Modele retenu: {best_name} (MAE={best_score:.3f}).")
    else:
        notes.append("Historique insuffisant pour validation walk-forward, selection par priorite.")

    if best_builder is None:
        best_name, best_builder = candidates[0]
        notes.append(f"Modele retenu par defaut: {best_name}.")

    X_train, y_train = _prepare_training_xy(train, feature_cols)
    if X_train.empty:
        notes.append("Features d'entrainement vides: fallback heuristique.")
        return TrainingResult(model=None, model_name="heuristic", notes=notes)

    model = best_builder()
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        notes.append(f"Echec entrainement {best_name}: {exc}. Fallback heuristique.")
        return TrainingResult(model=None, model_name="heuristic", notes=notes)

    return TrainingResult(model=model, model_name=best_name, notes=notes)
