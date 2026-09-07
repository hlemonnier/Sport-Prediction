"""Model training routines for football match result prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .constants import (
    AWAY_WIN_CLASS,
    BASELINE_PSEUDOCOUNT,
    BASELINE_SEASON_WINDOW,
    BASELINE_TEAM_WEIGHT,
    BENCHMARK_MIN_SAMPLES,
    BENCHMARK_MIN_VALIDATION,
    BENCHMARK_VALIDATION_FRACTION,
    CALIBRATION_MIN_CLASS_SAMPLES,
    CALIBRATION_MIN_SAMPLES,
    DIAGNOSTIC_ECE_BINS,
    DRAW_CLASS,
    HOME_WIN_CLASS,
    OUTCOME_CLASSES,
    HYBRID_WEIGHT_GRID,
    RECENT_FORM_WINDOW,
)
from .data import FixtureRecord, MatchRecord, match_available_at
from .joint import DixonColesModel, fit_dixon_coles, default_dixon_coles_model as _default_dixon_coles_model, dixon_coles_tau as _dixon_coles_tau
from .score_distribution import build_score_distribution
from .utils import outcome_class, record_sort_key, parse_datetime

try:
    import numpy as np  # type: ignore
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import log_loss  # type: ignore

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore
    IsotonicRegression = None  # type: ignore
    LogisticRegression = None  # type: ignore
    log_loss = None  # type: ignore
    SKLEARN_AVAILABLE = False


@dataclass
class ProbabilityCalibrator:
    method: str
    per_class_functions: list[Callable[[float], float]]

    def apply(self, probabilities: tuple[float, float, float]) -> tuple[float, float, float]:
        probabilities = normalize_probabilities(probabilities)
        if len(self.per_class_functions) != 3:
            raise ValueError("Calibration requires exactly three class functions.")
        calibrated = [float(fn(p)) for fn, p in zip(self.per_class_functions, probabilities)]
        if any(not math.isfinite(p) or p < 0 for p in calibrated):
            raise ValueError("Calibrator returned invalid probabilities.")
        # Avoid learned exact impossibilities from finite calibration samples.
        if self.method != "identity":
            calibrated = [max(1e-12, p) for p in calibrated]
        return normalize_probabilities(tuple(calibrated))



@dataclass
class BenchmarkModel:
    model: Any
    classes: list[int]
    log_loss_value: float | None
    brier_value: float | None
    feature_names: list[str]
    validation_labels: list[int] = field(default_factory=list)
    validation_probabilities: list[tuple[float, float, float]] = field(default_factory=list)
    validation_pairs: list[tuple[str, str]] = field(default_factory=list)

    def predict(self, features: list[float]) -> tuple[float, float, float]:
        raw = self.model.predict_proba([features])
        aligned = _align_class_probabilities(raw, self.classes)[0]
        return (
            aligned[HOME_WIN_CLASS],
            aligned[DRAW_CLASS],
            aligned[AWAY_WIN_CLASS],
        )


@dataclass
class FrequencyBaselineModel:
    outcome_counts: tuple[float, float, float]
    season_outcome_counts: dict[int, tuple[float, float, float]]
    home_team_outcome_counts: dict[str, tuple[float, float, float]]
    away_team_outcome_counts: dict[str, tuple[float, float, float]]
    pseudo_count: float = BASELINE_PSEUDOCOUNT
    team_weight: float = BASELINE_TEAM_WEIGHT
    season_window: int = BASELINE_SEASON_WINDOW

    def predict(self, record: FixtureRecord | MatchRecord) -> tuple[float, float, float]:
        weighted = [self.pseudo_count, self.pseudo_count, self.pseudo_count]
        season = record.season if isinstance(record.season, int) else None

        season_weight_total = 0.0
        if season is not None:
            for known_season, counts in self.season_outcome_counts.items():
                distance = abs(known_season - season)
                if distance > self.season_window:
                    continue
                weight = float(self.season_window + 1 - distance)
                season_weight_total += weight
                for cls in OUTCOME_CLASSES:
                    weighted[cls] += counts[cls] * weight

        if season_weight_total <= 0:
            for cls in OUTCOME_CLASSES:
                weighted[cls] += self.outcome_counts[cls]

        home_counts = self.home_team_outcome_counts.get(record.home_team_id)
        if home_counts is not None:
            for cls in OUTCOME_CLASSES:
                weighted[cls] += home_counts[cls] * self.team_weight

        away_counts = self.away_team_outcome_counts.get(record.away_team_id)
        if away_counts is not None:
            for cls in OUTCOME_CLASSES:
                weighted[cls] += away_counts[cls] * self.team_weight

        return normalize_probabilities(
            (
                weighted[HOME_WIN_CLASS],
                weighted[DRAW_CLASS],
                weighted[AWAY_WIN_CLASS],
            )
        )


def _empty_outcome_counts() -> list[float]:
    return [0.0, 0.0, 0.0]


def _freeze_counts(counts: list[float]) -> tuple[float, float, float]:
    return (
        float(counts[HOME_WIN_CLASS]),
        float(counts[DRAW_CLASS]),
        float(counts[AWAY_WIN_CLASS]),
    )


def _freeze_nested_counts(
    counts: dict[Any, list[float]],
) -> dict[Any, tuple[float, float, float]]:
    return {key: _freeze_counts(value) for key, value in counts.items()}


def fit_frequency_baseline(matches: list[MatchRecord], notes: list[str]) -> FrequencyBaselineModel:
    outcome_counts = _empty_outcome_counts()
    season_counts: dict[int, list[float]] = {}
    home_team_counts: dict[str, list[float]] = {}
    away_team_counts: dict[str, list[float]] = {}
    used_matches = 0

    for match in matches:
        if match.home_goals is None or match.away_goals is None:
            continue
        used_matches += 1
        label = outcome_class(match.home_goals, match.away_goals)
        outcome_counts[label] += 1.0

        if isinstance(match.season, int):
            season_bucket = season_counts.setdefault(match.season, _empty_outcome_counts())
            season_bucket[label] += 1.0

        home_bucket = home_team_counts.setdefault(match.home_team_id, _empty_outcome_counts())
        away_bucket = away_team_counts.setdefault(match.away_team_id, _empty_outcome_counts())
        home_bucket[label] += 1.0
        away_bucket[label] += 1.0

    if used_matches == 0:
        notes.append("Baseline frequence: historique vide, fallback uniforme.")
    else:
        notes.append(
            "Baseline frequence entraine "
            f"(n={used_matches}, seasons={len(season_counts)}, teams_home={len(home_team_counts)}, teams_away={len(away_team_counts)})."
        )

    return FrequencyBaselineModel(
        outcome_counts=_freeze_counts(outcome_counts),
        season_outcome_counts=_freeze_nested_counts(season_counts),
        home_team_outcome_counts=_freeze_nested_counts(home_team_counts),
        away_team_outcome_counts=_freeze_nested_counts(away_team_counts),
    )


def normalize_probabilities(probabilities: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(probabilities) != 3:
        raise ValueError("Expected exactly three outcome probabilities.")
    values = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Outcome probabilities must be finite and nonnegative.")
    total = math.fsum(values)
    if total <= 0:
        raise ValueError("Outcome probabilities must contain positive total mass.")
    return tuple(value / total for value in values)


def _validate_scoring_rows(labels: list[int], probabilities: list[list[float]]) -> None:
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal lengths.")
    if any(label not in OUTCOME_CLASSES for label in labels):
        raise ValueError("Invalid outcome label.")
    for row in probabilities:
        normalized = normalize_probabilities(tuple(row))
        if abs(math.fsum(row) - 1.0) > 1e-8:
            raise ValueError("Scoring probabilities must sum to one.")


def rank_outcome_classes(probabilities: tuple[float, float, float]) -> list[int]:
    normalized = normalize_probabilities(probabilities)
    return sorted(OUTCOME_CLASSES, key=lambda cls: (-normalized[cls], cls))


def _multiclass_log_loss(y_true: list[int], probabilities: list[list[float]]) -> float:
    _validate_scoring_rows(y_true, probabilities)
    if not y_true:
        return float("nan")
    eps = 1e-12
    total = 0.0
    for outcome, probs in zip(y_true, probabilities):
        probability = min(max(probs[outcome], eps), 1.0)
        total += -math.log(probability)
    return total / len(y_true)


def _ranking_metrics(y_true: list[int], probabilities: list[list[float]]) -> tuple[float, float, float, float]:
    if not y_true:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    top1_hits = 0.0
    top2_hits = 0.0
    reciprocal_rank_sum = 0.0
    confidence_sum = 0.0

    for outcome, probs in zip(y_true, probabilities):
        ranked = rank_outcome_classes((probs[0], probs[1], probs[2]))
        if ranked[0] == outcome:
            top1_hits += 1.0
        if outcome in ranked[:2]:
            top2_hits += 1.0
        reciprocal_rank_sum += 1.0 / float(ranked.index(outcome) + 1)
        confidence_sum += probs[ranked[0]]

    sample_size = float(len(y_true))
    return (
        top1_hits / sample_size,
        top2_hits / sample_size,
        reciprocal_rank_sum / sample_size,
        confidence_sum / sample_size,
    )


def _expected_calibration_error(
    y_true: list[int], probabilities: list[list[float]], bins: int
) -> tuple[float, list[dict[str, float | int | None]]]:
    _validate_scoring_rows(y_true, probabilities)
    if not y_true:
        return float("nan"), []

    n_bins = max(2, bins)
    bin_counts = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins
    bin_accuracy_sum = [0.0] * n_bins

    for outcome, probs in zip(y_true, probabilities):
        ranked = rank_outcome_classes((probs[0], probs[1], probs[2]))
        predicted = ranked[0]
        confidence = probs[predicted]
        index = min(int(confidence * n_bins), n_bins - 1)
        bin_counts[index] += 1
        bin_conf_sum[index] += confidence
        if predicted == outcome:
            bin_accuracy_sum[index] += 1.0

    sample_size = float(len(y_true))
    ece = 0.0
    calibration_bins: list[dict[str, float | int | None]] = []
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        count = bin_counts[index]
        if count == 0:
            calibration_bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "avg_confidence": None,
                    "accuracy": None,
                }
            )
            continue
        avg_conf = bin_conf_sum[index] / float(count)
        accuracy = bin_accuracy_sum[index] / float(count)
        ece += (float(count) / sample_size) * abs(accuracy - avg_conf)
        calibration_bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": accuracy,
            }
        )

    return ece, calibration_bins


def evaluate_match_probabilities(
    matches: list[MatchRecord],
    predictor: Callable[[MatchRecord], tuple[float, float, float]],
    bins: int = DIAGNOSTIC_ECE_BINS,
) -> dict[str, Any]:
    y_true: list[int] = []
    probabilities: list[list[float]] = []

    for match in sorted(matches, key=record_sort_key):
        if match.home_goals is None or match.away_goals is None:
            continue
        predicted = normalize_probabilities(predictor(match))
        y_true.append(outcome_class(match.home_goals, match.away_goals))
        probabilities.append([predicted[0], predicted[1], predicted[2]])

    if not y_true:
        return {
            "sample_size": 0,
            "calibration": {
                "log_loss": None,
                "brier": None,
                "ece": None,
                "bins": [],
            },
            "ranking": {
                "top1_accuracy": None,
                "top2_accuracy": None,
                "mean_reciprocal_rank": None,
                "avg_confidence": None,
            },
        }

    log_loss_value = _multiclass_log_loss(y_true, probabilities)
    brier_value = _multiclass_brier_score(y_true, probabilities)
    ece_value, calibration_bins = _expected_calibration_error(y_true, probabilities, bins=bins)
    top1_accuracy, top2_accuracy, mean_reciprocal_rank, avg_confidence = _ranking_metrics(
        y_true, probabilities
    )

    return {
        "sample_size": len(y_true),
        "calibration": {
            "log_loss": log_loss_value,
            "brier": brier_value,
            "ece": ece_value,
            "bins": calibration_bins,
        },
        "ranking": {
            "top1_accuracy": top1_accuracy,
            "top2_accuracy": top2_accuracy,
            "mean_reciprocal_rank": mean_reciprocal_rank,
            "avg_confidence": avg_confidence,
        },
    }


def _identity(value: float) -> float:
    return float(value)


def score_probability_matrix(lambda_home: float, lambda_away: float, rho: float, max_goals: int = 1) -> list[list[float]]:
    """Compatibility API: max_goals is a minimum support, never a truncation cap."""
    return [list(row) for row in build_score_distribution(lambda_home, lambda_away, rho, min_max_goals=max_goals).matrix]


def outcome_probabilities(lambda_home: float, lambda_away: float, rho: float) -> tuple[float, float, float]:
    return build_score_distribution(lambda_home, lambda_away, rho).outcome_probabilities


def most_likely_scoreline(lambda_home: float, lambda_away: float, rho: float) -> tuple[int, int, float]:
    return build_score_distribution(lambda_home, lambda_away, rho).most_likely_scoreline


def fit_probability_calibrator(matches: list[MatchRecord], model: DixonColesModel, notes: list[str]) -> ProbabilityCalibrator:
    return fit_probability_calibrator_with_policy(matches, model, notes, policy="auto")


def fit_probability_calibrator_from_rows(
    probabilities: list[tuple[float, float, float]],
    labels: list[int],
    notes: list[str],
    *,
    policy: str = "auto",
    context: str = "model",
) -> ProbabilityCalibrator:
    policy_name = str(policy or "auto").strip().lower()
    if policy_name not in {"off", "auto", "platt", "isotonic"}:
        raise ValueError("Unknown calibration policy.")
    _validate_scoring_rows(labels, [list(row) for row in probabilities])
    if policy_name == "off":
        notes.append(f"Calibration {context}: forcee off, identity.")
        return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])

    if not SKLEARN_AVAILABLE:
        notes.append(f"Calibration {context}: scikit-learn indisponible, identity.")
        return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])

    if len(labels) < CALIBRATION_MIN_SAMPLES or len(probabilities) < CALIBRATION_MIN_SAMPLES:
        notes.append(
            f"Calibration {context}: echantillon insuffisant "
            f"({min(len(labels), len(probabilities))}<{CALIBRATION_MIN_SAMPLES}), identity."
        )
        return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        notes.append(f"Calibration {context}: format proba invalide, identity.")
        return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])

    def _fit_isotonic() -> tuple[ProbabilityCalibrator | None, int]:
        isotonic_functions: list[Callable[[float], float]] = []
        isotonic_count = 0
        for cls in OUTCOME_CLASSES:
            x_cls = p[:, cls]
            y_cls = (y == cls).astype(int)
            positives = int(y_cls.sum())
            negatives = int(len(y_cls) - positives)
            if positives < CALIBRATION_MIN_CLASS_SAMPLES or negatives < CALIBRATION_MIN_CLASS_SAMPLES:
                isotonic_functions.append(_identity)
                continue
            if len(np.unique(x_cls)) < 4:
                isotonic_functions.append(_identity)
                continue
            model_iso = IsotonicRegression(out_of_bounds="clip")
            model_iso.fit(x_cls, y_cls)

            def _iso_fn(value: float, transformer: IsotonicRegression = model_iso) -> float:
                return float(transformer.predict([value])[0])

            isotonic_functions.append(_iso_fn)
            isotonic_count += 1
        if isotonic_count < 1:
            return None, 0
        return ProbabilityCalibrator(method="isotonic", per_class_functions=isotonic_functions), isotonic_count

    def _fit_platt() -> tuple[ProbabilityCalibrator | None, int]:
        platt_functions: list[Callable[[float], float]] = []
        platt_count = 0
        for cls in OUTCOME_CLASSES:
            bounded = np.clip(p[:, cls], 1e-12, 1.0 - 1e-12)
            x_cls = np.log(bounded / (1.0 - bounded)).reshape(-1, 1)
            y_cls = (y == cls).astype(int)
            positives = int(y_cls.sum())
            negatives = int(len(y_cls) - positives)
            if positives < CALIBRATION_MIN_CLASS_SAMPLES or negatives < CALIBRATION_MIN_CLASS_SAMPLES:
                platt_functions.append(_identity)
                continue
            model_lr = LogisticRegression(max_iter=400)
            model_lr.fit(x_cls, y_cls)

            def _platt_fn(value: float, transformer: LogisticRegression = model_lr) -> float:
                bounded = min(1.0 - 1e-12, max(1e-12, value))
                return float(transformer.predict_proba([[math.log(bounded / (1.0 - bounded))]])[0][1])

            platt_functions.append(_platt_fn)
            platt_count += 1
        if platt_count < 1:
            return None, 0
        return ProbabilityCalibrator(method="platt", per_class_functions=platt_functions), platt_count

    if policy_name == "isotonic" or (policy_name == "auto" and len(labels) >= 200):
        calibrator, count = _fit_isotonic()
        if calibrator is not None:
            notes.append(f"Calibration {context}: isotonic active sur {count}/3 classes.")
            return calibrator
        if policy_name == "isotonic":
            notes.append(f"Calibration {context}: isotonic impossible, fallback identity.")
            return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])

    if policy_name in {"auto", "platt"}:
        calibrator, count = _fit_platt()
        if calibrator is not None:
            notes.append(f"Calibration {context}: Platt active sur {count}/3 classes.")
            return calibrator
        if policy_name == "platt":
            notes.append(f"Calibration {context}: Platt impossible, fallback identity.")
            return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])

    notes.append(f"Calibration {context}: fallback identity.")
    return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])


def fit_probability_calibrator_with_policy(
    matches: list[MatchRecord],
    model: DixonColesModel,
    notes: list[str],
    *,
    policy: str = "auto",
) -> ProbabilityCalibrator:
    policy_name = str(policy or "auto").strip().lower()
    if policy_name == "off":
        return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])
    fit_ids = set(model.diagnostics.get("fit_match_ids", []))
    if not fit_ids or any(match.match_id in fit_ids for match in matches):
        raise ValueError("Calibration requires documented, disjoint held-out match identities.")
    fit_available = parse_datetime(model.diagnostics.get("latest_fit_result_available_at"))
    if fit_available is None or any(match.date is None or match.date < fit_available for match in matches):
        raise ValueError("Calibration matches must follow availability of every fit result.")
    probabilities: list[tuple[float, float, float]] = []
    labels: list[int] = []
    for match in matches:
        if match.home_goals is None or match.away_goals is None:
            continue
        labels.append(outcome_class(match.home_goals, match.away_goals))
        lambda_home, lambda_away = model.expected_goals(match.home_team_id, match.away_team_id)
        probabilities.append(outcome_probabilities(lambda_home, lambda_away, model.rho))
    return fit_probability_calibrator_from_rows(
        probabilities=probabilities,
        labels=labels,
        notes=notes,
        policy=policy_name,
        context="dixon",
    )


def select_hybrid_weight(
    labels: list[int],
    dc_probabilities: list[tuple[float, float, float]],
    gbdt_probabilities: list[tuple[float, float, float]],
    notes: list[str],
) -> float:
    if not labels or not dc_probabilities or not gbdt_probabilities:
        notes.append("Hybrid weight: validation indisponible, w=0.0.")
        return 0.0
    if not (len(labels) == len(dc_probabilities) == len(gbdt_probabilities)):
        notes.append("Hybrid weight: tailles validation incoherentes, w=0.0.")
        return 0.0

    best_weight = 0.0
    best_loss = float("inf")
    for weight in HYBRID_WEIGHT_GRID:
        blended_rows: list[list[float]] = []
        for dc, gbdt in zip(dc_probabilities, gbdt_probabilities):
            blended = normalize_probabilities(
                (
                    (weight * gbdt[HOME_WIN_CLASS]) + ((1.0 - weight) * dc[HOME_WIN_CLASS]),
                    (weight * gbdt[DRAW_CLASS]) + ((1.0 - weight) * dc[DRAW_CLASS]),
                    (weight * gbdt[AWAY_WIN_CLASS]) + ((1.0 - weight) * dc[AWAY_WIN_CLASS]),
                )
            )
            blended_rows.append([blended[HOME_WIN_CLASS], blended[DRAW_CLASS], blended[AWAY_WIN_CLASS]])
        loss_value = _multiclass_log_loss(labels, blended_rows)
        if not math.isfinite(loss_value):
            continue
        if loss_value < best_loss:
            best_loss = loss_value
            best_weight = float(weight)
    notes.append(f"Hybrid weight selection: w={best_weight:.1f} (log-loss={best_loss:.4f}).")
    return float(best_weight)


def _empty_history() -> list[dict[str, float]]:
    return []


def _points_for_result(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _append_history(
    history: dict[str, list[dict[str, float]]],
    team_id: str,
    points: int,
    goal_diff: int,
    xg_diff: float,
) -> None:
    team_history = history.setdefault(team_id, _empty_history())
    team_history.append(
        {
            "points": float(points),
            "goal_diff": float(goal_diff),
            "xg_diff": float(xg_diff),
        }
    )


def _recent_stats(team_history: list[dict[str, float]], window: int) -> tuple[float, float, float, float]:
    if not team_history:
        return 0.0, 0.0, 0.0, 0.0
    recent = team_history[-window:]
    count = float(len(recent))
    return (
        sum(item["points"] for item in recent) / count,
        sum(item["goal_diff"] for item in recent) / count,
        sum(item["xg_diff"] for item in recent) / count,
        count,
    )


def _build_feature_vector(
    history: dict[str, list[dict[str, float]]], home_team_id: str, away_team_id: str, window: int
) -> list[float]:
    home_points, home_goal_diff, home_xg_diff, home_games = _recent_stats(
        history.get(home_team_id, []), window
    )
    away_points, away_goal_diff, away_xg_diff, away_games = _recent_stats(
        history.get(away_team_id, []), window
    )
    return [
        home_points,
        away_points,
        home_points - away_points,
        home_goal_diff,
        away_goal_diff,
        home_goal_diff - away_goal_diff,
        home_xg_diff,
        away_xg_diff,
        home_xg_diff - away_xg_diff,
        home_games,
        away_games,
    ]


def build_history_from_matches(matches: list[MatchRecord], *, as_of: Any = None) -> dict[str, list[dict[str, float]]]:
    history: dict[str, list[dict[str, float]]] = {}
    for match in sorted(matches, key=record_sort_key):
        if match.home_goals is None or match.away_goals is None:
            continue
        if as_of is not None:
            available = match_available_at(match)
            if available is None or available > as_of or match.date >= as_of:
                continue
        xg_known = match.xg_available_at is not None and (as_of is None or match.xg_available_at <= as_of)
        home_xg = match.home_xg if xg_known and match.home_xg is not None else float(match.home_goals)
        away_xg = match.away_xg if xg_known and match.away_xg is not None else float(match.away_goals)
        xg_delta = home_xg - away_xg
        _append_history(
            history=history,
            team_id=match.home_team_id,
            points=_points_for_result(match.home_goals, match.away_goals),
            goal_diff=match.home_goals - match.away_goals,
            xg_diff=xg_delta,
        )
        _append_history(
            history=history,
            team_id=match.away_team_id,
            points=_points_for_result(match.away_goals, match.home_goals),
            goal_diff=match.away_goals - match.home_goals,
            xg_diff=-xg_delta,
        )
    return history


def fixture_feature_vector(
    fixture: FixtureRecord, history: dict[str, list[dict[str, float]]]
) -> list[float]:
    return _build_feature_vector(
        history=history,
        home_team_id=fixture.home_team_id,
        away_team_id=fixture.away_team_id,
        window=RECENT_FORM_WINDOW,
    )


def _align_class_probabilities(raw: Any, classes: list[int]) -> list[list[float]]:
    aligned: list[list[float]] = []
    for row in raw:
        output = [0.0, 0.0, 0.0]
        for class_index, class_value in enumerate(classes):
            if class_value in OUTCOME_CLASSES:
                output[class_value] = float(row[class_index])
        total = sum(output)
        if total <= 0:
            aligned.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        else:
            aligned.append([value / total for value in output])
    return aligned


def _multiclass_brier_score(y_true: list[int], probabilities: list[list[float]]) -> float:
    _validate_scoring_rows(y_true, probabilities)
    if not y_true:
        return float("nan")
    total = 0.0
    for outcome, probs in zip(y_true, probabilities):
        for cls in OUTCOME_CLASSES:
            expected = 1.0 if outcome == cls else 0.0
            total += (probs[cls] - expected) ** 2
    return total / len(y_true)


FEATURE_NAMES = ["home_form_points", "away_form_points", "form_points_diff",
                 "home_goal_diff", "away_goal_diff", "goal_diff_delta",
                 "home_xg_diff", "away_xg_diff", "xg_diff_delta",
                 "home_recent_games", "away_recent_games"]


def train_gradient_boosting_model(matches: list[MatchRecord], notes: list[str]) -> BenchmarkModel | None:
    """Fit on exactly the supplied population; no hidden validation split."""
    if not SKLEARN_AVAILABLE:
        notes.append("GBDT unavailable: optional scikit-learn dependency is missing.")
        return None
    valid = [m for m in sorted(matches, key=record_sort_key)
             if m.home_goals is not None and m.away_goals is not None and m.date is not None]
    if len(valid) < BENCHMARK_MIN_SAMPLES:
        notes.append(f"GBDT fit unavailable: {len(valid)} < {BENCHMARK_MIN_SAMPLES} dated rows.")
        return None
    labels = [outcome_class(m.home_goals, m.away_goals) for m in valid]
    if len(set(labels)) < 3:
        notes.append("GBDT unavailable: all three outcome classes are required to avoid assigning an unseen class zero probability.")
        return None
    features = [fixture_feature_vector(m, build_history_from_matches(valid, as_of=m.date)) for m in valid]
    model = GradientBoostingClassifier(random_state=42)
    model.fit(np.asarray(features, dtype=float), np.asarray(labels, dtype=int))
    notes.append(f"GBDT fitted on {len(valid)} causal rows; evaluation is external to fitting.")
    return BenchmarkModel(model=model, classes=model.classes_.tolist(), log_loss_value=None,
                          brier_value=None, feature_names=list(FEATURE_NAMES))


def train_gradient_boosting_benchmark(matches: list[MatchRecord], notes: list[str]) -> BenchmarkModel | None:
    """Compatibility benchmark with a whole-day chronological holdout >=30."""
    from .protocol import split_chronological_tail
    train, valid = split_chronological_tail(matches,
        max(BENCHMARK_MIN_VALIDATION, math.ceil(len(matches)*BENCHMARK_VALIDATION_FRACTION)))
    if len(train) < BENCHMARK_MIN_SAMPLES or len(valid) < BENCHMARK_MIN_VALIDATION:
        notes.append("GBDT benchmark requires at least 45 fit and 30 held-out dated rows.")
        return None
    model = train_gradient_boosting_model(train, notes)
    if model is None:
        return None
    probabilities = [model.predict(fixture_feature_vector(m, build_history_from_matches(matches, as_of=m.date))) for m in valid]
    labels = [outcome_class(m.home_goals, m.away_goals) for m in valid]
    model.log_loss_value = _multiclass_log_loss(labels, [list(row) for row in probabilities])
    model.brier_value = _multiclass_brier_score(labels, [list(row) for row in probabilities])
    model.validation_labels = labels
    model.validation_probabilities = probabilities
    model.validation_pairs = [(m.home_team_id, m.away_team_id) for m in valid]
    return model
