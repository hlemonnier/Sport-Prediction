"""Continuous xG quasi-likelihood strengths and prior-relative probability models."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from research.experiments.boundary_20260908.football.models import count_design, offset_objective, feature_vector as prior_features

ZONES = {"E0": "Europe/London", "SP1": "Europe/Madrid", "I1": "Europe/Rome"}


def available_at(row, delay_days):
    if delay_days < 0 or int(delay_days) != delay_days:
        raise ValueError("delay must be nonnegative whole days")
    local = datetime.fromisoformat(row["canonical_date"]).replace(tzinfo=ZoneInfo(ZONES[row["league"]]))
    return (local + timedelta(days=delay_days + 1)).astimezone(timezone.utc).replace(tzinfo=None)


def eligible_history(rows, league, cutoff, delay_days, history_days=1095):
    lower = cutoff - timedelta(days=history_days)
    # Filter using identity/date/availability before touching measured xG values.
    return [r for r in rows if r["league"] == league and r["accepted_exact_join"] is True
            and lower <= datetime.fromisoformat(r["canonical_date"]) < cutoff
            and available_at(r, delay_days) <= cutoff]


@dataclass
class XGStrength:
    teams: list
    coef: np.ndarray
    intercept: float
    diagnostics: dict

    def predict(self, fixtures):
        x = count_design(self.teams, fixtures)
        log_mean = (self.intercept + x @ self.coef).reshape(-1, 2)
        if not np.isfinite(log_mean).all() or not np.isfinite(np.exp(log_mean)).all():
            raise ValueError("invalid expected xG")
        return log_mean


def fit_strength(rows, cutoff, half_life, options):
    if not rows:
        raise ValueError("no available historical xG")
    teams = sorted({r["home"] for r in rows} | {r["away"] for r in rows})
    design = count_design(teams, [(r["home"], r["away"]) for r in rows])
    values = np.array([[r["home_xg"], r["away_xg"]] for r in rows], float).ravel()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("xG must be finite and nonnegative")
    ages = np.array([(cutoff - datetime.fromisoformat(r["canonical_date"])).total_seconds() / 86400 for r in rows])
    if (ages <= 0).any():
        raise ValueError("xG fit includes current/future matches")
    weights = np.repeat(np.exp2(-ages / half_life), 2)
    model = PoissonRegressor(alpha=options["alpha"], fit_intercept=True, solver=options["solver"],
                             max_iter=options["max_iter"], tol=options["tol"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(design, values, sample_weight=weights)
    prediction = model.predict(design)
    residual = weights * (prediction - values) / weights.sum()
    gradient = np.r_[design.T @ residual + options["alpha"] * model.coef_, residual.sum()]
    error = float(np.max(np.abs(gradient)))
    if error > 1e-6:
        raise ValueError(f"quasi-score gradient not converged: {error}")
    diagnostic = {"teams": teams, "coef": model.coef_.tolist(), "intercept": float(model.intercept_),
                  "half_life": half_life, "fit_n": len(rows), "iterations": int(model.n_iter_),
                  "gradient_max": error, "effective_match_count": float(weights.sum() ** 2 / (weights @ weights) / 2),
                  "training_ids": [r["canonical_match_id"] for r in rows],
                  "latest_completion_date": max(r["canonical_date"] for r in rows),
                  "objective": "weighted mean(exp(eta)-continuous_xg*eta) + alpha/2*coef_norm_squared; intercept unpenalized"}
    return XGStrength(teams, model.coef_.copy(), float(model.intercept_), diagnostic)


def feature_vector(row, config):
    old = "shot_strength_90d" if config["mode"] == "add" else "log_probability_residual_only"
    return [*prior_features(row, old), *row["log_xg_means"][str(config["half_life"])]]


@dataclass
class Offset:
    config: dict
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    diagnostics: dict

    def predict(self, rows):
        features = np.array([feature_vector(r, self.config) for r in rows])
        x = np.column_stack([np.ones(len(rows)), (features - self.mean) / self.scale])
        p = np.array([r["probabilities"]["dc365_elo50"] for r in rows])
        return softmax(np.log(p) + x @ self.weights, axis=1)


def fit_offset(rows, config, penalty=.1):
    features = np.array([feature_vector(r, config) for r in rows])
    mean, scale = features.mean(0), features.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    x = np.column_stack([np.ones(len(rows)), (features - mean) / scale])
    p = np.array([r["probabilities"]["dc365_elo50"] for r in rows])
    y = np.array([r["label"] for r in rows], int)
    args = (x, np.log(p), y, penalty)
    fit = minimize(offset_objective, np.zeros(x.shape[1] * 3), args=args, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7})
    error = float(np.max(np.abs(offset_objective(fit.x, *args)[1])))
    if not fit.success or not np.isfinite(fit.fun) or error > 1e-6:
        raise ValueError(f"offset fit failed: {fit.message}; gradient {error}")
    weights = fit.x.reshape(-1, 3)
    diagnostic = {"config": config, "penalty": penalty, "fit_n": len(rows), "iterations": int(fit.nit),
                  "gradient_max": error, "objective": float(fit.fun), "mean": mean.tolist(),
                  "scale": scale.tolist(), "weights": weights.tolist()}
    return Offset(config, mean, scale, weights, diagnostic)
