"""Causal shot-count strengths and identity-shrunk multiclass corrections."""
from __future__ import annotations
from dataclasses import dataclass
import os
import warnings
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from packages.football.mrp.joint import fit_dixon_coles
from packages.football.mrp.protocol import chronological_populations
from packages.football.mrp.score_distribution import build_score_distribution
from packages.football.mrp.training import fit_probability_calibrator_with_policy


def production_default_fit(matches):
    """The actual default DC+auto policy, on a caller-declared history window.

    GBDT and shadow evaluation do not affect the default selected DC prediction;
    the research comparator omits only those unused computations.
    """
    populations = chronological_populations(matches)
    model = fit_dixon_coles(populations.fit, [])
    calibrator = fit_probability_calibrator_with_policy(populations.calibration, model, [], policy="auto")
    return model, calibrator, populations


def production_default_predict(model, calibrator, home, away):
    p = build_score_distribution(*model.expected_goals(home, away), model.rho).outcome_probabilities
    return calibrator.apply(p)


def count_design(teams, fixtures):
    lookup = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    design = np.zeros((2*len(fixtures), 2*n+1))
    for i, (home, away) in enumerate(fixtures):
        for row, attack, defend, home_flag in ((2*i, home, away, 1), (2*i+1, away, home, 0)):
            if attack in lookup:
                design[row, lookup[attack]] = 1
            if defend in lookup:
                design[row, n+lookup[defend]] = 1
            design[row, -1] = home_flag
    return design


@dataclass
class ShotStrength:
    teams: list[str]
    shot_model: PoissonRegressor
    target_model: PoissonRegressor
    diagnostics: dict

    def predict(self, fixtures):
        x = count_design(self.teams, fixtures)
        shots = self.shot_model.predict(x).reshape(-1, 2)
        targets = self.target_model.predict(x).reshape(-1, 2)
        means = np.column_stack((shots, targets))
        if np.any(~np.isfinite(means)) or np.any(means <= 0):
            raise ValueError("Invalid shot-count means")
        return np.log(means)


def fit_shot_strength(matches, statistics, cutoff, half_life, options):
    teams = sorted({m.home_team_id for m in matches} | {m.away_team_id for m in matches})
    fixtures = [(m.home_team_id, m.away_team_id) for m in matches]
    x = count_design(teams, fixtures)
    raw = np.asarray([statistics[m.match_id] for m in matches], dtype=float)
    if raw.shape != (len(matches), 4) or np.any(~np.isfinite(raw)) or np.any(raw < 0) or np.any(raw != np.floor(raw)):
        raise ValueError("Shot counts must be finite nonnegative integers")
    ages = np.asarray([(cutoff-m.date).total_seconds()/86400 for m in matches])
    if np.any(ages <= 0):
        raise ValueError("Shot model requires strictly previous matches")
    weights = np.repeat(np.exp2(-ages/half_life), 2)
    fitted = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for columns in ((0, 1), (2, 3)):
            model = PoissonRegressor(alpha=options["alpha"], fit_intercept=True,
                solver=options["solver"], max_iter=options["max_iter"], tol=options["tol"])
            model.fit(x, raw[:, columns].reshape(-1), sample_weight=weights)
            fitted.append(model)
    return ShotStrength(teams, *fitted, {
        "half_life_days": half_life, "fit_n": len(matches), "teams": teams,
        "alpha": options["alpha"], "objective": "weighted mean Poisson deviance/2 + alpha/2 * squared coefficient norm; intercept unpenalized",
        "effective_match_n": float(weights.sum()**2/(weights@weights)/2),
        "models": [{"coef": m.coef_.tolist(), "intercept": float(m.intercept_), "iterations": int(m.n_iter_)} for m in fitted]})


def feature_vector(row, mechanism):
    p = row["probabilities"]
    mix, dc, elo = (np.asarray(p[k]) for k in ("dc365_elo50", "dc_365", "elo_component"))
    features = [np.log(mix[0]/mix[2]), np.log(mix[1]/mix[2]),
                np.log(dc[0]/dc[2])-np.log(elo[0]/elo[2]),
                np.log(dc[1]/dc[2])-np.log(elo[1]/elo[2]),
                float(row["league"] == "SP1"), float(row["league"] == "I1")]
    if mechanism != "log_probability_residual_only":
        key = "90" if mechanism == "shot_strength_90d" else "365"
        features.extend(row["log_shot_means"][key])
    return features


def offset_objective(flat, x, log_prior, labels, penalty):
    w = flat.reshape(x.shape[1], 3)
    logits = log_prior + x@w
    objective = np.mean(logsumexp(logits, axis=1)-logits[np.arange(len(labels)), labels])
    objective += penalty/2*float(np.sum(w*w))
    residual = softmax(logits, axis=1)
    residual[np.arange(len(labels)), labels] -= 1
    gradient = x.T@residual/len(labels) + penalty*w
    return float(objective), gradient.ravel()


@dataclass
class OffsetModel:
    mechanism: str
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    diagnostics: dict

    def predict(self, rows):
        features = np.asarray([feature_vector(r, self.mechanism) for r in rows])
        x = np.column_stack((np.ones(len(rows)), (features-self.mean)/self.scale))
        prior = np.asarray([r["probabilities"]["dc365_elo50"] for r in rows])
        return softmax(np.log(prior)+x@self.weights, axis=1)


def fit_offset(rows, mechanism, penalty):
    features = np.asarray([feature_vector(r, mechanism) for r in rows])
    mean, scale = features.mean(0), features.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    x = np.column_stack((np.ones(len(rows)), (features-mean)/scale))
    prior = np.asarray([r["probabilities"]["dc365_elo50"] for r in rows])
    labels = np.asarray([r["label"] for r in rows], dtype=int)
    initial = np.zeros(x.shape[1]*3)
    args = (x, np.log(prior), labels, penalty)
    result = minimize(offset_objective, initial, args=args, jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7})
    gradient = float(np.max(np.abs(offset_objective(result.x, *args)[1])))
    if not result.success or not np.isfinite(result.fun) or gradient > 1e-6:
        raise RuntimeError(f"Offset model not converged: {result.message}, gradient={gradient}")
    return OffsetModel(mechanism, mean, scale, result.x.reshape(-1, 3), {
        "fit_n": len(rows), "penalty": penalty, "converged": True, "iterations": int(result.nit),
        "objective": float(result.fun), "identity_objective": offset_objective(initial, *args)[0],
        "gradient_inf_norm": gradient, "mean": mean.tolist(), "scale": scale.tolist(),
        "weights": result.x.reshape(-1, 3).tolist()})
