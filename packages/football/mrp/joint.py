"""Constrained joint Dixon-Coles likelihood and reproducible fit diagnostics.

Rates and all four low-score cells are feasible for every fitted/neutral team
pair, including combinations absent from the training schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Any

from .constants import (
    DEFAULT_HOME_GOALS_PRIOR, DEFAULT_AWAY_GOALS_PRIOR,
    DIXON_COLES_MAX_ITER, DIXON_COLES_REGULARIZATION, DIXON_COLES_TOLERANCE,
    DIXON_COLES_RATE_CLAMP_MIN, DIXON_COLES_RATE_CLAMP_MAX,
    DIXON_COLES_RHO_MIN, DIXON_COLES_RHO_MAX,
)
from .data import MatchRecord, match_available_at, _normalize_timestamp

SUPPORT_EPSILON = 1e-9


@dataclass
class DixonColesModel:
    attack: dict[str, float]
    defense: dict[str, float]
    home_intercept: float
    away_intercept: float
    rho: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def expected_goals(self, home_team_id: str, away_team_id: str) -> tuple[float, float]:
        home = math.exp(self.home_intercept + self.attack.get(home_team_id, 0.0)
                        + self.defense.get(away_team_id, 0.0))
        away = math.exp(self.away_intercept + self.attack.get(away_team_id, 0.0)
                        + self.defense.get(home_team_id, 0.0))
        # Do not clip after fitting: that would change the optimized likelihood.
        if not (math.isfinite(home) and math.isfinite(away) and home > 0 and away > 0):
            raise ValueError("Dixon-Coles produced non-finite or non-positive rates.")
        validate_rho(home, away, self.rho)
        return home, away


def rho_bounds(lambda_home: float, lambda_away: float) -> tuple[float, float]:
    if not all(math.isfinite(x) and x > 0 for x in (lambda_home, lambda_away)):
        raise ValueError("Goal rates must be finite and strictly positive.")
    return max(-1.0 / lambda_home, -1.0 / lambda_away), min(1.0, 1.0 / (lambda_home * lambda_away))


def validate_rho(lambda_home: float, lambda_away: float, rho: float) -> None:
    lower, upper = rho_bounds(lambda_home, lambda_away)
    if not math.isfinite(rho) or rho < lower or rho > upper:
        raise ValueError(f"rho={rho} violates full score support [{lower}, {upper}].")


def dixon_coles_tau(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float, rho: float) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def default_dixon_coles_model() -> DixonColesModel:
    return DixonColesModel({}, {}, math.log(DEFAULT_HOME_GOALS_PRIOR),
                           math.log(DEFAULT_AWAY_GOALS_PRIOR), 0.0,
                           {"status": "prior_only", "converged": None, "fit_sample_size": 0})


def fit_dixon_coles(matches: list[MatchRecord], notes: list[str], *,
                    half_life_days: float | None = None,
                    reference_time: datetime | None = None) -> DixonColesModel:
    """Maximize the joint, optionally time-weighted, penalized likelihood.

    No decay is applied by default. A specified half-life is a modelling
    assumption that must be selected outside the final evaluation sample.
    """
    valid = [m for m in matches if m.home_goals is not None and m.away_goals is not None]
    if not valid:
        notes.append("Dixon-Coles: no scored fit rows; explicit neutral prior only.")
        return default_dixon_coles_model()
    for m in valid:
        if m.home_team_id == m.away_team_id:
            raise ValueError("A football match must contain distinct teams.")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (m.home_goals, m.away_goals)):
            raise ValueError("Final football scores must be nonnegative integers.")
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize, lsq_linear
    except ImportError as exc:
        raise RuntimeError("Joint Dixon-Coles fitting requires numpy and scipy; install packages/football/requirements.txt.") from exc

    teams = sorted({m.home_team_id for m in valid} | {m.away_team_id for m in valid})
    index = {team: i for i, team in enumerate(teams)}
    t, n = len(teams), len(valid)
    size = 2 * t + 3
    home_ids = np.array([index[m.home_team_id] for m in valid])
    away_ids = np.array([index[m.away_team_id] for m in valid])
    goals_home = np.asarray([m.home_goals for m in valid], dtype=float)
    goals_away = np.asarray([m.away_goals for m in valid], dtype=float)
    dh = np.zeros((n, size)); da = np.zeros((n, size))
    rows = np.arange(n)
    dh[rows, home_ids] = 1; dh[rows, t + away_ids] = 1; dh[:, 2*t] = 1
    da[rows, away_ids] = 1; da[rows, t + home_ids] = 1; da[:, 2*t+1] = 1

    weights = np.ones(n)
    if half_life_days is not None:
        if not math.isfinite(half_life_days) or half_life_days <= 0:
            raise ValueError("half_life_days must be finite and positive.")
        if any(m.date is None for m in valid):
            raise ValueError("Temporal weighting requires dates on every fit match.")
        reference_time = _normalize_timestamp(reference_time) or max(m.date for m in valid if m.date is not None)
        age = np.asarray([(reference_time - m.date).total_seconds() / 86400.0 for m in valid])
        if np.any(age < 0):
            raise ValueError("Decay reference time precedes a fit match.")
        weights = np.exp2(-age / half_life_days)
        if float(weights.sum()) <= 0:
            raise ValueError("Temporal weights underflowed; choose a supported half-life.")
    normalizer = float(weights.sum())

    # Include the zero-strength neutral team so unseen-team forecasts satisfy
    # the same constraints. Cartesian pairs also cover unseen combinations.
    pairs = [(h, a) for h in range(t + 1) for a in range(t + 1)]
    ph = np.zeros((len(pairs), size)); pa = np.zeros_like(ph)
    for k, (h, a) in enumerate(pairs):
        if h < t: ph[k, h] = 1; pa[k, t+h] = 1
        if a < t: ph[k, t+a] = 1; pa[k, a] = 1
    ph[:, 2*t] = 1; pa[:, 2*t+1] = 1
    rate_design = np.vstack((ph, pa))
    sums = np.zeros((2, size)); sums[0, :t] = 1; sums[1, t:2*t] = 1
    mask00 = (goals_home == 0) & (goals_away == 0)
    mask01 = (goals_home == 0) & (goals_away == 1)
    mask10 = (goals_home == 1) & (goals_away == 0)
    mask11 = (goals_home == 1) & (goals_away == 1)
    factorials = np.asarray([math.lgamma(h+1) + math.lgamma(a+1) for h, a in zip(goals_home, goals_away)])
    ridge = DIXON_COLES_REGULARIZATION

    def objective(x: Any) -> tuple[float, Any]:
        eh, ea = dh @ x, da @ x
        lh, la, rho = np.exp(eh), np.exp(ea), x[-1]
        tau = np.ones(n); th = np.zeros(n); ta = np.zeros(n); tr = np.zeros(n)
        product = lh * la
        tau[mask00] = 1 - product[mask00] * rho
        th[mask00] = ta[mask00] = -product[mask00] * rho
        tr[mask00] = -product[mask00]
        tau[mask01] = 1 + lh[mask01] * rho; th[mask01] = lh[mask01] * rho; tr[mask01] = lh[mask01]
        tau[mask10] = 1 + la[mask10] * rho; ta[mask10] = la[mask10] * rho; tr[mask10] = la[mask10]
        tau[mask11] = 1 - rho; tr[mask11] = -1
        # SLSQP can probe infeasible points. A smooth continuation of -log(tau)
        # below epsilon guides it back without evaluating log of negative mass.
        safe = np.maximum(tau, SUPPORT_EPSILON)
        log_tau = np.log(safe) + np.minimum(tau - SUPPORT_EPSILON, 0) / SUPPORT_EPSILON
        loss = float(np.sum(weights * (lh - goals_home*eh + la - goals_away*ea + factorials - log_tau)))
        loss += .5 * ridge * float(x[:2*t] @ x[:2*t])
        gh = weights * (lh - goals_home - th / safe)
        ga = weights * (la - goals_away - ta / safe)
        gradient = dh.T @ gh + da.T @ ga
        gradient[:2*t] += ridge * x[:2*t]
        gradient[-1] -= float(np.sum(weights * tr / safe))
        return loss / normalizer, gradient / normalizer

    def support(x: Any) -> Any:
        lh, la, rho = np.exp(ph @ x), np.exp(pa @ x), x[-1]
        return np.r_[1-lh*la*rho, 1+lh*rho, 1+la*rho]

    def support_jac(x: Any) -> Any:
        lh, la, rho = np.exp(ph @ x), np.exp(pa @ x), x[-1]
        product = lh * la
        j0 = -(product*rho)[:, None] * (ph+pa); j0[:, -1] = -product
        j1 = (lh*rho)[:, None] * ph; j1[:, -1] = lh
        j2 = (la*rho)[:, None] * pa; j2[:, -1] = la
        return np.vstack((j0, j1, j2))

    lo, hi = math.log(DIXON_COLES_RATE_CLAMP_MIN), math.log(DIXON_COLES_RATE_CLAMP_MAX)
    start = np.zeros(size)
    start[2*t] = math.log(min(DIXON_COLES_RATE_CLAMP_MAX, max(DIXON_COLES_RATE_CLAMP_MIN, float(np.average(goals_home, weights=weights)))))
    start[2*t+1] = math.log(min(DIXON_COLES_RATE_CLAMP_MAX, max(DIXON_COLES_RATE_CLAMP_MIN, float(np.average(goals_away, weights=weights)))))
    bounds = Bounds(np.r_[np.full(2*t, -3.), lo, lo, DIXON_COLES_RHO_MIN],
                    np.r_[np.full(2*t, 3.), hi, hi, DIXON_COLES_RHO_MAX])
    constraints = [LinearConstraint(sums, 0, 0), LinearConstraint(rate_design, lo, hi),
                   NonlinearConstraint(support, SUPPORT_EPSILON, np.inf, jac=support_jac)]
    result = minimize(objective, start, jac=True, method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": DIXON_COLES_MAX_ITER, "ftol": DIXON_COLES_TOLERANCE})
    x = result.x
    feasibility = min(float(np.min(support(x))) - SUPPORT_EPSILON,
                      float(np.min(rate_design @ x)) - lo, hi - float(np.max(rate_design @ x)))
    equality_error = float(np.max(np.abs(sums @ x)))
    if not result.success or not np.isfinite(result.fun) or feasibility < -1e-7 or equality_error > 1e-7:
        raise RuntimeError(f"Dixon-Coles fit did not converge to a feasible model: {result.message}; feasibility={feasibility:.3g}.")
    # Check first-order constrained stationarity independently of the solver's
    # success flag. Multipliers of active inequalities must be nonnegative.
    active_normals = []
    rate_values = rate_design @ x
    for row,value in zip(rate_design,rate_values):
        if value-lo <= 1e-6: active_normals.append(row)
        if hi-value <= 1e-6: active_normals.append(-row)
    for row,value in zip(support_jac(x),support(x)):
        if value-SUPPORT_EPSILON <= 1e-6: active_normals.append(row)
    for k in range(size):
        basis = np.zeros(size); basis[k] = 1
        if x[k]-bounds.lb[k] <= 1e-6: active_normals.append(basis)
        if bounds.ub[k]-x[k] <= 1e-6: active_normals.append(-basis)
    columns = [sums[0],sums[1],*[-row for row in active_normals]]
    system = np.column_stack(columns)
    target = -objective(x)[1]
    multipliers = lsq_linear(system,target,
        bounds=(np.r_[[-np.inf,-np.inf],np.zeros(len(active_normals))],np.full(len(columns),np.inf)),
        tol=1e-12,max_iter=1000)
    stationarity = float(np.max(np.abs(system @ multipliers.x - target)))
    if stationarity > 1e-5:
        raise RuntimeError(f"Dixon-Coles solver stopped without first-order convergence: KKT residual={stationarity:.3g}.")
    diagnostics = {
        "status": "fitted", "objective": "joint_dixon_coles_penalized_negative_log_likelihood",
        "solver": "SLSQP", "converged": True, "optimizer_message": str(result.message),
        "iterations": int(result.nit), "fit_sample_size": n,
        "fit_match_ids": [m.match_id for m in valid],
        "latest_fit_result_available_at": max((match_available_at(m).isoformat() for m in valid if match_available_at(m) is not None), default=None),
        "negative_log_likelihood_per_weight": float(result.fun),
        "initial_negative_log_likelihood_per_weight": float(objective(start)[0]),
        "kkt_stationarity_inf_norm": stationarity,
        "kkt_tolerance": 1e-5,
        "gradient_inf_norm": float(np.max(np.abs(objective(x)[1]))),
        "gradient_note": "Unconstrained gradient; may be nonzero at active support/rate constraints.",
        "minimum_tau_all_team_pairs": min(float(np.min(support(x))),1.0-float(x[-1])),
        "constraint_violation_max": max(0.0, -feasibility, equality_error),
        "half_life_days": half_life_days, "weight_sum": normalizer,
        "effective_sample_size": float(normalizer**2 / np.sum(weights**2)),
        "regularization_sum_objective": ridge,
    }
    model = DixonColesModel(dict(zip(teams, map(float, x[:t]))), dict(zip(teams, map(float, x[t:2*t]))),
                           float(x[2*t]), float(x[2*t+1]), float(x[-1]), diagnostics)
    # Independently validate the returned prediction API on every supported pair.
    neutral = "__football_unseen_neutral__"
    for h in [*teams, neutral]:
        for a in [*teams, neutral]:
            model.expected_goals(h, a)
    notes.append(f"Joint Dixon-Coles converged: n={n}, teams={t}, rho={model.rho:.5f}, iterations={result.nit}.")
    return model
