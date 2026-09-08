"""Validated execution of the frozen continuous-xG and probability objectives.

Date-only historical identity uses UTC midnight for the inherited history window.
Availability is a different clock: next league-local midnight after completion,
then 168/336 elapsed UTC hours. Naive caller cutoffs mean UTC, never local time.
These are retrospective availability assumptions, not provider publication times.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from numbers import Integral, Real
from zoneinfo import ZoneInfo
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from research.experiments.boundary_20260908.football.models import (
    count_design,
    feature_vector as prior_features,
    offset_objective,
)
from research.experiments.boundary_20260908.football_xg_data.availability import (
    ZONES,
    proxy_available_at,
)


def _utc(value):
    if not isinstance(value, datetime):
        raise ValueError("cutoff must be a datetime; naive values explicitly mean UTC")
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _positive(value, name, *, integer=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    if not np.isfinite(value) or value <= 0 or (integer and not isinstance(value, Integral)):
        raise ValueError(f"{name} must be {'an integer' if integer else 'finite'} and positive")
    return int(value) if integer else float(value)


def _delay(value):
    value = _positive(value, "delay_days", integer=True)
    if value not in (7, 14):
        raise ValueError("only frozen 7/14-day xG availability delays are supported")
    return value


def _accepted(row):
    value = row["accepted_exact_join"]
    return value is True or (isinstance(value, str) and value == "True")


def _day(row):
    value = row["canonical_date"]
    if not isinstance(value, str):
        raise ValueError("canonical_date must be YYYY-MM-DD")
    day = datetime.strptime(value, "%Y-%m-%d").date()
    if day.isoformat() != value:
        raise ValueError("canonical_date must be YYYY-MM-DD")
    return day


def _metadata_clock(row, delay_days):
    """Compute eligibility without reading xG, scores or provider measurements."""
    league = row["league"]
    if league not in ZONES:
        raise ValueError(f"unsupported league: {league}")
    midnight = datetime.combine(_day(row) + timedelta(days=1), time.min, ZoneInfo(ZONES[league]))
    return midnight.astimezone(timezone.utc) + timedelta(days=delay_days)


def available_at(row, delay_days):
    """Frozen validated availability, returned as UTC-naive for runner parity.

    For history selection use eligible_history: it rejects unavailable metadata
    before this full join validation is permitted to access observed final goals.
    Exact Python True and CSV 'True' are the only accepted join flags.
    """
    delay_days = _delay(delay_days)
    if not _accepted(row):
        return None
    expected = _metadata_clock(row, delay_days)
    actual = proxy_available_at(row, delay_days)
    if actual != expected:
        raise ValueError("frozen xG availability contract mismatch")
    return actual.astimezone(timezone.utc).replace(tzinfo=None)


def eligible_history(rows, league, cutoff, delay_days, history_days=1095):
    """Return original records with past identity and xG available <= UTC cutoff.

    Availability equality follows the published minimum-information contract.
    Neither future/unavailable goals nor any xG value is read during filtering.
    The inherited date window is [cutoff-history_days, cutoff), in UTC.
    """
    cutoff = _utc(cutoff)
    delay_days = _delay(delay_days)
    history_days = _positive(history_days, "history_days", integer=True)
    if league not in ZONES:
        raise ValueError(f"unsupported league: {league}")
    lower = cutoff - timedelta(days=history_days)
    selected = []
    for row in rows:
        if row["league"] != league or not _accepted(row):
            continue
        identity_day = datetime.combine(_day(row), time.min, timezone.utc)
        if not lower <= identity_day < cutoff:
            continue
        if _metadata_clock(row, delay_days) > cutoff:
            continue
        # Only now may the frozen contract validate final dates and goals.
        clock = available_at(row, delay_days)
        if clock is None or _utc(clock) > cutoff:
            raise ValueError("validated xG availability changed after metadata eligibility")
        selected.append(row)
    return selected


def _fixtures(fixtures):
    fixtures = list(fixtures)
    for fixture in fixtures:
        if not isinstance(fixture, (tuple, list)) or len(fixture) != 2:
            raise ValueError("each fixture must be a home/away pair")
        if any(not isinstance(team, str) or not team.strip() for team in fixture) or fixture[0] == fixture[1]:
            raise ValueError("fixture requires distinct nonempty home/away team identities")
    return fixtures


def _finite_array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape {shape} and finite entries")
    return value.copy()


@dataclass
class XGStrength:
    teams: list
    coef: np.ndarray
    intercept: float
    diagnostics: dict

    def __post_init__(self):
        self.teams = list(self.teams)
        if (not self.teams or any(not isinstance(t, str) or not t.strip() for t in self.teams)
                or len(set(self.teams)) != len(self.teams)):
            raise ValueError("teams must be distinct nonempty string identities")
        self.coef = _finite_array(self.coef, (2 * len(self.teams) + 1,), "strength coefficients")
        if isinstance(self.intercept, (bool, np.bool_)) or not np.isfinite(float(self.intercept)):
            raise ValueError("strength intercept must be finite")
        self.intercept = float(self.intercept)
        self.diagnostics = deepcopy(self.diagnostics)

    def predict(self, fixtures):
        """Return home/away log expected xG, including declared unseen-team fallback."""
        fixtures = _fixtures(fixtures)
        x = count_design(self.teams, fixtures)
        log_mean = (self.intercept + x @ self.coef).reshape(-1, 2)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            means = np.exp(log_mean)
        if not np.isfinite(log_mean).all() or not np.isfinite(means).all() or (means <= 0).any():
            raise ValueError("expected xG must be positive and finite")
        return log_mean

    def to_dict(self):
        return {"schema_version": 1, "model_type": "continuous_xg_strength",
                "teams": list(self.teams), "coef": self.coef.tolist(), "intercept": self.intercept,
                "diagnostics": deepcopy(self.diagnostics)}

    @classmethod
    def from_dict(cls, value):
        _serialized_type(value, "continuous_xg_strength")
        return cls(value["teams"], value["coef"], value["intercept"], value["diagnostics"])


def _serialized_type(value, name):
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1 or value.get("model_type") != name:
        raise ValueError("unsupported serialized xG model schema/type")


def quasi_objective(flat, design, values, weights, alpha):
    """Weighted mean quasi-Poisson objective and score; intercept is last and free.

    Continuous xG has the same mean estimating equation as Poisson regression.
    No Poisson count sampling distribution, variance or interval is asserted.
    """
    eta = design @ flat[:-1] + flat[-1]
    mean = np.exp(eta)
    normalized = weights / weights.sum()
    loss = normalized @ (mean - values * eta) + alpha / 2 * (flat[:-1] @ flat[:-1])
    residual = normalized * (mean - values)
    gradient = np.r_[design.T @ residual + alpha * flat[:-1], residual.sum()]
    return float(loss), gradient


def fit_strength(rows, cutoff, half_life, options):
    rows = list(rows)
    cutoff = _utc(cutoff)
    half_life = _positive(half_life, "half_life")
    alpha = _positive(options["alpha"], "alpha")
    max_iter = _positive(options["max_iter"], "max_iter", integer=True)
    tol = _positive(options["tol"], "tol")
    delay_days = _delay(options.get("delay_days", 7))
    history_days = _positive(options.get("history_days", 1095), "history_days", integer=True)
    if options["solver"] not in ("newton-cholesky", "lbfgs"):
        raise ValueError("unsupported Poisson mean solver")
    if options.get("intercept_penalized", False) is not False:
        raise ValueError("frozen quasi-Poisson intercept must remain unpenalized")
    if not rows:
        raise ValueError("no available historical xG")
    leagues = {r["league"] for r in rows}
    if len(leagues) != 1:
        raise ValueError("strength fit must be within one league")
    eligible = eligible_history(rows, next(iter(leagues)), cutoff, delay_days, history_days)
    if len(eligible) != len(rows):
        raise ValueError("strength fit includes rejected, unavailable or out-of-window xG")
    ids = [r["canonical_match_id"] for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("strength fit requires distinct canonical match IDs")
    fixtures = _fixtures([(r["home"], r["away"]) for r in rows])
    teams = sorted({t for fixture in fixtures for t in fixture})
    design = count_design(teams, fixtures)
    measured = [r[key] for r in rows for key in ("home_xg", "away_xg")]
    if any(isinstance(v, (bool, np.bool_)) for v in measured):
        raise ValueError("xG cannot be boolean")
    values = np.asarray(measured, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
        raise ValueError("xG must be finite and nonnegative, with positive total support")
    ages = np.asarray([(cutoff - datetime.combine(_day(r), time.min, timezone.utc)).total_seconds() / 86400 for r in rows])
    # Common scaling cancels in sklearn's weighted mean and in the score/Kish N.
    # Subtracting the youngest age avoids all weights underflowing together.
    weights = np.repeat(np.exp2(-(ages - ages.min()) / half_life), 2)
    if not np.isfinite(weights).all() or weights.sum() <= 0 or weights @ values <= 0:
        raise ValueError("invalid decayed xG support")
    model = PoissonRegressor(alpha=alpha, fit_intercept=True, solver=options["solver"],
                             max_iter=max_iter, tol=tol)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(design, values, sample_weight=weights)
    flat = np.r_[model.coef_, model.intercept_]
    objective, gradient = quasi_objective(flat, design, values, weights, alpha)
    error = float(np.max(np.abs(gradient)))
    if not np.isfinite(objective) or not np.isfinite(gradient).all() or error > 1e-6:
        raise ValueError(f"quasi-score gradient not converged: {error}")
    clocks = [available_at(r, delay_days) for r in rows]
    diagnostic = {"teams": teams, "coef": model.coef_.tolist(), "intercept": float(model.intercept_),
                  "half_life": half_life, "alpha": alpha, "delay_days": delay_days,
                  "history_days": history_days, "fit_n": len(rows), "iterations": int(model.n_iter_),
                  "gradient_max": error, "objective_value": objective,
                  "effective_match_count": float(weights.sum() ** 2 / (weights @ weights) / 2),
                  "training_ids": ids, "latest_completion_date": max(r["canonical_date"] for r in rows),
                  "latest_available_at_utc": max(clocks).isoformat() + "Z",
                  "cutoff_utc": cutoff.isoformat(),
                  "objective": "weighted mean(exp(eta)-continuous_xg*eta) + alpha/2*coef_norm_squared; intercept unpenalized"}
    fitted = XGStrength(teams, model.coef_, float(model.intercept_), diagnostic)
    fitted.predict(fixtures)
    return fitted


def _config(config):
    if config.get("mode") not in ("add", "replace"):
        raise ValueError("xG mode must be one of frozen add/replace variants")
    half_life = _positive(config["half_life"], "config half_life", integer=True)
    if half_life not in (90, 365):
        raise ValueError("xG configuration half_life must be 90 or 365 days")
    return deepcopy(config)


def _probabilities(value, name):
    p = _finite_array(value, (3,), name)
    if (p <= 0).any() or not np.isclose(p.sum(), 1., rtol=0., atol=1e-10):
        raise ValueError(f"{name} must be positive probabilities summing to one")
    return p


def feature_vector(row, config):
    config = _config(config)
    if row["league"] not in ZONES:
        raise ValueError("unsupported probability-feature league")
    for key in ("dc365_elo50", "dc_365", "elo_component"):
        _probabilities(row["probabilities"][key], key)
    if config["mode"] == "add":
        _finite_array(row["log_shot_means"]["90"], (4,), "prior shot log means")
    log_xg = _finite_array(row["log_xg_means"][str(config["half_life"])], (2,), "predicted xG log means")
    old = "shot_strength_90d" if config["mode"] == "add" else "log_probability_residual_only"
    result = np.asarray([*prior_features(row, old), *log_xg], dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite probability correction features")
    return result.tolist()


@dataclass
class Offset:
    config: dict
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        self.config = _config(self.config)
        n = 12 if self.config["mode"] == "add" else 8
        self.mean = _finite_array(self.mean, (n,), "offset feature mean")
        self.scale = _finite_array(self.scale, (n,), "offset feature scale")
        self.weights = _finite_array(self.weights, (n + 1, 3), "offset weights")
        if (self.scale <= 0).any():
            raise ValueError("offset feature scales must be positive")
        self.diagnostics = deepcopy(self.diagnostics)

    def predict(self, rows):
        rows = list(rows)
        if not rows:
            return np.empty((0, 3), dtype=float)
        features = np.asarray([feature_vector(r, self.config) for r in rows])
        x = np.column_stack([np.ones(len(rows)), (features - self.mean) / self.scale])
        p = np.asarray([r["probabilities"]["dc365_elo50"] for r in rows], dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            logits = np.log(p) + x @ self.weights
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite probability correction logits")
        prediction = softmax(logits, axis=1)
        for row in prediction:
            _probabilities(row, "predicted 1X2")
        return prediction

    def to_dict(self):
        return {"schema_version": 1, "model_type": "continuous_xg_offset", "config": deepcopy(self.config),
                "mean": self.mean.tolist(), "scale": self.scale.tolist(), "weights": self.weights.tolist(),
                "diagnostics": deepcopy(self.diagnostics)}

    @classmethod
    def from_dict(cls, value):
        _serialized_type(value, "continuous_xg_offset")
        return cls(value["config"], value["mean"], value["scale"], value["weights"], value["diagnostics"])


def fit_offset(rows, config, penalty=.1):
    rows = list(rows)
    config = _config(config)
    penalty = _positive(penalty, "penalty")
    if not rows:
        raise ValueError("offset fit requires nonempty historical features")
    features = np.asarray([feature_vector(r, config) for r in rows])
    mean, scale = features.mean(0), features.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("invalid offset training standardization")
    x = np.column_stack([np.ones(len(rows)), (features - mean) / scale])
    p = np.asarray([r["probabilities"]["dc365_elo50"] for r in rows], dtype=float)
    labels = [r["label"] for r in rows]
    if any(isinstance(y, (bool, np.bool_)) or not isinstance(y, Real) or y not in (0, 1, 2) for y in labels):
        raise ValueError("1X2 labels must be exact 0/1/2, without boolean or fractional coercion")
    y = np.asarray(labels, dtype=int)
    args = (x, np.log(p), y, penalty)
    initial = np.zeros(x.shape[1] * 3)
    fit = minimize(offset_objective, initial, args=args, jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-7})
    _, gradient = offset_objective(fit.x, *args)
    error = float(np.max(np.abs(gradient)))
    if not fit.success or not np.isfinite(fit.fun) or not np.isfinite(gradient).all() or error > 1e-6:
        raise ValueError(f"offset fit failed: {fit.message}; gradient {error}")
    weights = fit.x.reshape(-1, 3)
    diagnostic = {"config": deepcopy(config), "penalty": penalty, "fit_n": len(rows), "iterations": int(fit.nit),
                  "gradient_max": error, "objective": float(fit.fun),
                  "identity_objective": offset_objective(initial, *args)[0],
                  "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist()}
    fitted = Offset(config, mean, scale, weights, diagnostic)
    fitted.predict(rows)
    return fitted
