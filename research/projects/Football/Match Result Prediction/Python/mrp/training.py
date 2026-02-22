"""Model training routines for football match result prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .constants import (
    AWAY_WIN_CLASS,
    BENCHMARK_MIN_SAMPLES,
    BENCHMARK_MIN_VALIDATION,
    BENCHMARK_VALIDATION_FRACTION,
    CALIBRATION_MIN_CLASS_SAMPLES,
    CALIBRATION_MIN_SAMPLES,
    DEFAULT_AWAY_GOALS_PRIOR,
    DEFAULT_HOME_GOALS_PRIOR,
    DIXON_COLES_LEARNING_RATE,
    DIXON_COLES_MAX_ITER,
    DIXON_COLES_RATE_CLAMP_MAX,
    DIXON_COLES_RATE_CLAMP_MIN,
    DIXON_COLES_REGULARIZATION,
    DIXON_COLES_RHO_MAX,
    DIXON_COLES_RHO_MIN,
    DIXON_COLES_RHO_STEP,
    DIXON_COLES_TOLERANCE,
    DRAW_CLASS,
    HOME_WIN_CLASS,
    OUTCOME_CLASSES,
    OUTCOME_GOAL_GRID_MAX,
    RECENT_FORM_WINDOW,
    SCORELINE_GOAL_GRID_MAX,
)
from .data import FixtureRecord, MatchRecord
from .utils import clamp, outcome_class, record_sort_key, safe_mean

try:
    import numpy as np  # type: ignore
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import log_loss  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore
    IsotonicRegression = None  # type: ignore
    LogisticRegression = None  # type: ignore
    log_loss = None  # type: ignore
    SKLEARN_AVAILABLE = False


@dataclass
class DixonColesModel:
    attack: dict[str, float]
    defense: dict[str, float]
    home_intercept: float
    away_intercept: float
    rho: float

    def expected_goals(self, home_team_id: str, away_team_id: str) -> tuple[float, float]:
        attack_home = self.attack.get(home_team_id, 0.0)
        attack_away = self.attack.get(away_team_id, 0.0)
        defense_home = self.defense.get(home_team_id, 0.0)
        defense_away = self.defense.get(away_team_id, 0.0)
        lambda_home = math.exp(self.home_intercept + attack_home + defense_away)
        lambda_away = math.exp(self.away_intercept + attack_away + defense_home)
        return (
            clamp(lambda_home, DIXON_COLES_RATE_CLAMP_MIN, DIXON_COLES_RATE_CLAMP_MAX),
            clamp(lambda_away, DIXON_COLES_RATE_CLAMP_MIN, DIXON_COLES_RATE_CLAMP_MAX),
        )


@dataclass
class ProbabilityCalibrator:
    method: str
    per_class_functions: list[Callable[[float], float]]

    def apply(self, probabilities: tuple[float, float, float]) -> tuple[float, float, float]:
        calibrated = [
            max(0.0, fn(probability))
            for fn, probability in zip(self.per_class_functions, probabilities)
        ]
        total = sum(calibrated)
        if total <= 0:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        return (
            calibrated[HOME_WIN_CLASS] / total,
            calibrated[DRAW_CLASS] / total,
            calibrated[AWAY_WIN_CLASS] / total,
        )


@dataclass
class BenchmarkModel:
    model: Any
    classes: list[int]
    log_loss_value: float
    brier_value: float
    feature_names: list[str]

    def predict(self, features: list[float]) -> tuple[float, float, float]:
        raw = self.model.predict_proba([features])
        aligned = _align_class_probabilities(raw, self.classes)[0]
        return (
            aligned[HOME_WIN_CLASS],
            aligned[DRAW_CLASS],
            aligned[AWAY_WIN_CLASS],
        )


def _identity(value: float) -> float:
    return float(value)


def _default_dixon_coles_model() -> DixonColesModel:
    return DixonColesModel(
        attack={},
        defense={},
        home_intercept=math.log(DEFAULT_HOME_GOALS_PRIOR),
        away_intercept=math.log(DEFAULT_AWAY_GOALS_PRIOR),
        rho=0.0,
    )


def fit_dixon_coles(matches: list[MatchRecord], notes: list[str]) -> DixonColesModel:
    if not matches:
        notes.append(
            "Dixon-Coles: aucun historique entrainable, fallback sur prior neutre (intensites moyennes)."
        )
        return _default_dixon_coles_model()

    teams = sorted(
        {
            match.home_team_id
            for match in matches
            if match.home_goals is not None and match.away_goals is not None
        }
        | {
            match.away_team_id
            for match in matches
            if match.home_goals is not None and match.away_goals is not None
        }
    )
    if not teams:
        notes.append(
            "Dixon-Coles: historique sans equipes exploitables, fallback sur prior neutre."
        )
        return _default_dixon_coles_model()

    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    goals_home = [float(match.home_goals or 0) for match in matches]
    goals_away = [float(match.away_goals or 0) for match in matches]
    n_teams = len(teams)

    attack = [0.0] * n_teams
    defense = [0.0] * n_teams
    home_intercept = math.log(max(safe_mean(goals_home, DEFAULT_HOME_GOALS_PRIOR), 0.2))
    away_intercept = math.log(max(safe_mean(goals_away, DEFAULT_AWAY_GOALS_PRIOR), 0.2))

    learning_rate = DIXON_COLES_LEARNING_RATE
    previous_log_likelihood = -float("inf")

    for _ in range(DIXON_COLES_MAX_ITER):
        grad_attack = [0.0] * n_teams
        grad_defense = [0.0] * n_teams
        grad_home_intercept = 0.0
        grad_away_intercept = 0.0
        log_likelihood = 0.0

        for match in matches:
            if match.home_goals is None or match.away_goals is None:
                continue
            idx_home = team_to_idx[match.home_team_id]
            idx_away = team_to_idx[match.away_team_id]
            goals_home_i = float(match.home_goals)
            goals_away_i = float(match.away_goals)

            eta_home = home_intercept + attack[idx_home] + defense[idx_away]
            eta_away = away_intercept + attack[idx_away] + defense[idx_home]
            lambda_home = math.exp(eta_home)
            lambda_away = math.exp(eta_away)

            error_home = goals_home_i - lambda_home
            error_away = goals_away_i - lambda_away

            grad_home_intercept += error_home
            grad_away_intercept += error_away
            grad_attack[idx_home] += error_home
            grad_defense[idx_away] += error_home
            grad_attack[idx_away] += error_away
            grad_defense[idx_home] += error_away

            log_likelihood += (
                goals_home_i * eta_home
                - lambda_home
                - math.lgamma(goals_home_i + 1.0)
                + goals_away_i * eta_away
                - lambda_away
                - math.lgamma(goals_away_i + 1.0)
            )

        reg = DIXON_COLES_REGULARIZATION
        for idx in range(n_teams):
            grad_attack[idx] -= reg * attack[idx]
            grad_defense[idx] -= reg * defense[idx]
            log_likelihood -= 0.5 * reg * (attack[idx] ** 2 + defense[idx] ** 2)
        grad_home_intercept -= reg * home_intercept
        grad_away_intercept -= reg * away_intercept

        scale = 1.0 / max(len(matches), 1)
        for idx in range(n_teams):
            attack[idx] += learning_rate * grad_attack[idx] * scale
            defense[idx] += learning_rate * grad_defense[idx] * scale
            attack[idx] = clamp(attack[idx], -3.0, 3.0)
            defense[idx] = clamp(defense[idx], -3.0, 3.0)
        home_intercept += learning_rate * grad_home_intercept * scale
        away_intercept += learning_rate * grad_away_intercept * scale
        home_intercept = clamp(home_intercept, -1.5, 2.0)
        away_intercept = clamp(away_intercept, -1.5, 2.0)

        mean_attack = sum(attack) / n_teams
        mean_defense = sum(defense) / n_teams
        attack = [value - mean_attack for value in attack]
        defense = [value - mean_defense for value in defense]

        if log_likelihood < previous_log_likelihood:
            learning_rate = max(learning_rate * 0.5, 0.003)
        if abs(log_likelihood - previous_log_likelihood) < DIXON_COLES_TOLERANCE:
            break
        previous_log_likelihood = log_likelihood

    attack_map = {team: attack[team_to_idx[team]] for team in teams}
    defense_map = {team: defense[team_to_idx[team]] for team in teams}
    model = DixonColesModel(
        attack=attack_map,
        defense=defense_map,
        home_intercept=home_intercept,
        away_intercept=away_intercept,
        rho=0.0,
    )
    model.rho = _estimate_rho(model, matches)
    notes.append(
        "Dixon-Coles entraine "
        f"({len(matches)} matchs, {len(teams)} equipes, rho={model.rho:.3f})."
    )
    return model


def _dixon_coles_tau(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - (lambda_home * lambda_away * rho)
    if home_goals == 0 and away_goals == 1:
        return 1.0 + (lambda_home * rho)
    if home_goals == 1 and away_goals == 0:
        return 1.0 + (lambda_away * rho)
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _estimate_rho(model: DixonColesModel, matches: list[MatchRecord]) -> float:
    if not matches:
        return 0.0
    best_rho = 0.0
    best_log_likelihood = -float("inf")
    step_count = int(round((DIXON_COLES_RHO_MAX - DIXON_COLES_RHO_MIN) / DIXON_COLES_RHO_STEP))

    for step in range(step_count + 1):
        rho = DIXON_COLES_RHO_MIN + (step * DIXON_COLES_RHO_STEP)
        valid = True
        ll = 0.0
        for match in matches:
            if match.home_goals is None or match.away_goals is None:
                continue
            lambda_home, lambda_away = model.expected_goals(match.home_team_id, match.away_team_id)
            tau = _dixon_coles_tau(match.home_goals, match.away_goals, lambda_home, lambda_away, rho)
            if tau <= 0:
                valid = False
                break
            if match.home_goals <= 1 and match.away_goals <= 1:
                ll += math.log(tau)
        if valid and ll > best_log_likelihood:
            best_log_likelihood = ll
            best_rho = rho
    return best_rho


def _poisson_vector(rate: float, max_goals: int) -> list[float]:
    probabilities = [0.0] * (max_goals + 1)
    probabilities[0] = math.exp(-rate)
    for goals in range(1, max_goals + 1):
        probabilities[goals] = probabilities[goals - 1] * rate / goals
    return probabilities


def score_probability_matrix(
    lambda_home: float, lambda_away: float, rho: float, max_goals: int
) -> list[list[float]]:
    home_probs = _poisson_vector(lambda_home, max_goals)
    away_probs = _poisson_vector(lambda_away, max_goals)
    matrix = [
        [home_probs[home] * away_probs[away] for away in range(max_goals + 1)]
        for home in range(max_goals + 1)
    ]

    for home_goals in (0, 1):
        for away_goals in (0, 1):
            tau = _dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho)
            matrix[home_goals][away_goals] = max(matrix[home_goals][away_goals] * tau, 0.0)

    total = sum(sum(row) for row in matrix)
    if total <= 0:
        uniform = 1.0 / ((max_goals + 1) ** 2)
        return [[uniform for _ in range(max_goals + 1)] for _ in range(max_goals + 1)]
    return [[value / total for value in row] for row in matrix]


def outcome_probabilities(lambda_home: float, lambda_away: float, rho: float) -> tuple[float, float, float]:
    matrix = score_probability_matrix(lambda_home, lambda_away, rho, OUTCOME_GOAL_GRID_MAX)
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            if home_goals > away_goals:
                home_win += probability
            elif home_goals < away_goals:
                away_win += probability
            else:
                draw += probability
    total = home_win + draw + away_win
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (home_win / total, draw / total, away_win / total)


def most_likely_scoreline(
    lambda_home: float, lambda_away: float, rho: float
) -> tuple[int, int, float]:
    matrix = score_probability_matrix(lambda_home, lambda_away, rho, SCORELINE_GOAL_GRID_MAX)
    best_home = 0
    best_away = 0
    best_probability = -1.0
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            if probability > best_probability:
                best_home = home_goals
                best_away = away_goals
                best_probability = probability
    return best_home, best_away, max(best_probability, 0.0)


def fit_probability_calibrator(
    matches: list[MatchRecord], model: DixonColesModel, notes: list[str]
) -> ProbabilityCalibrator:
    if not SKLEARN_AVAILABLE:
        notes.append("Calibration: scikit-learn indisponible, fallback identity.")
        return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])

    training_rows: list[tuple[tuple[float, float, float], int]] = []
    for match in matches:
        if match.home_goals is None or match.away_goals is None:
            continue
        label = outcome_class(match.home_goals, match.away_goals)
        lambda_home, lambda_away = model.expected_goals(match.home_team_id, match.away_team_id)
        probabilities = outcome_probabilities(lambda_home, lambda_away, model.rho)
        training_rows.append((probabilities, label))

    if len(training_rows) < CALIBRATION_MIN_SAMPLES:
        notes.append(
            f"Calibration: echantillon insuffisant ({len(training_rows)}<{CALIBRATION_MIN_SAMPLES}), identity."
        )
        return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])

    y = np.asarray([label for _, label in training_rows], dtype=int)
    p = np.asarray([probs for probs, _ in training_rows], dtype=float)

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

    if isotonic_count >= 1:
        notes.append(f"Calibration: isotonic active sur {isotonic_count}/3 classes.")
        return ProbabilityCalibrator(method="isotonic", per_class_functions=isotonic_functions)

    platt_functions: list[Callable[[float], float]] = []
    platt_count = 0
    for cls in OUTCOME_CLASSES:
        x_cls = p[:, cls].reshape(-1, 1)
        y_cls = (y == cls).astype(int)
        positives = int(y_cls.sum())
        negatives = int(len(y_cls) - positives)
        if positives < CALIBRATION_MIN_CLASS_SAMPLES or negatives < CALIBRATION_MIN_CLASS_SAMPLES:
            platt_functions.append(_identity)
            continue
        model_lr = LogisticRegression(max_iter=400)
        model_lr.fit(x_cls, y_cls)

        def _platt_fn(value: float, transformer: LogisticRegression = model_lr) -> float:
            return float(transformer.predict_proba([[value]])[0][1])

        platt_functions.append(_platt_fn)
        platt_count += 1

    if platt_count >= 1:
        notes.append(f"Calibration: Platt active sur {platt_count}/3 classes.")
        return ProbabilityCalibrator(method="platt", per_class_functions=platt_functions)

    notes.append("Calibration: classes insuffisantes, fallback identity.")
    return ProbabilityCalibrator(method="identity", per_class_functions=[_identity, _identity, _identity])


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


def build_history_from_matches(matches: list[MatchRecord]) -> dict[str, list[dict[str, float]]]:
    history: dict[str, list[dict[str, float]]] = {}
    for match in sorted(matches, key=record_sort_key):
        if match.home_goals is None or match.away_goals is None:
            continue
        home_xg = match.home_xg if match.home_xg is not None else float(match.home_goals)
        away_xg = match.away_xg if match.away_xg is not None else float(match.away_goals)
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
    if not y_true:
        return float("nan")
    total = 0.0
    for outcome, probs in zip(y_true, probabilities):
        for cls in OUTCOME_CLASSES:
            expected = 1.0 if outcome == cls else 0.0
            total += (probs[cls] - expected) ** 2
    return total / len(y_true)


def train_gradient_boosting_benchmark(
    matches: list[MatchRecord], notes: list[str]
) -> BenchmarkModel | None:
    if not SKLEARN_AVAILABLE:
        notes.append("Benchmark GBM: scikit-learn indisponible.")
        return None

    history: dict[str, list[dict[str, float]]] = {}
    features: list[list[float]] = []
    labels: list[int] = []
    for match in sorted(matches, key=record_sort_key):
        if match.home_goals is None or match.away_goals is None:
            continue
        feature_vector = _build_feature_vector(
            history=history,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            window=RECENT_FORM_WINDOW,
        )
        features.append(feature_vector)
        labels.append(outcome_class(match.home_goals, match.away_goals))

        home_xg = match.home_xg if match.home_xg is not None else float(match.home_goals)
        away_xg = match.away_xg if match.away_xg is not None else float(match.away_goals)
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

    if len(features) < BENCHMARK_MIN_SAMPLES:
        notes.append(
            f"Benchmark GBM: echantillon insuffisant ({len(features)}<{BENCHMARK_MIN_SAMPLES})."
        )
        return None

    split_index = int(round(len(features) * (1.0 - BENCHMARK_VALIDATION_FRACTION)))
    split_index = max(split_index, len(features) - BENCHMARK_MIN_VALIDATION)
    split_index = min(split_index, len(features) - 1)
    if split_index <= 0 or (len(features) - split_index) < BENCHMARK_MIN_VALIDATION:
        notes.append("Benchmark GBM: validation set insuffisant.")
        return None

    x_train = np.asarray(features[:split_index], dtype=float)
    y_train = np.asarray(labels[:split_index], dtype=int)
    x_valid = np.asarray(features[split_index:], dtype=float)
    y_valid = np.asarray(labels[split_index:], dtype=int)

    if len(set(y_train.tolist())) < 2:
        notes.append("Benchmark GBM: classes insuffisantes dans le train split.")
        return None

    model = GradientBoostingClassifier(random_state=42)
    model.fit(x_train, y_train)

    raw_probabilities = model.predict_proba(x_valid)
    aligned_probabilities = _align_class_probabilities(raw_probabilities, model.classes_.tolist())
    y_valid_list = y_valid.tolist()
    brier_value = _multiclass_brier_score(y_valid_list, aligned_probabilities)
    log_loss_value = float(log_loss(y_valid, np.asarray(aligned_probabilities), labels=list(OUTCOME_CLASSES)))
    notes.append(
        "Benchmark GBM entrainable "
        f"(log-loss={log_loss_value:.4f}, brier={brier_value:.4f}, n_valid={len(y_valid_list)})."
    )

    return BenchmarkModel(
        model=model,
        classes=model.classes_.tolist(),
        log_loss_value=log_loss_value,
        brier_value=brier_value,
        feature_names=[
            "home_form_points",
            "away_form_points",
            "form_points_diff",
            "home_goal_diff",
            "away_goal_diff",
            "goal_diff_delta",
            "home_xg_diff",
            "away_xg_diff",
            "xg_diff_delta",
            "home_recent_games",
            "away_recent_games",
        ],
    )
