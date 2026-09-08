"""Event-weighted residual and pairwise ranking with complete permutations."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor


def rank(scores, ids):
    scores = np.asarray(scores, float)
    if not np.isfinite(scores).all() or len(set(ids)) != len(ids):
        raise ValueError("nonfinite scores or duplicate identities")
    order = np.lexsort((np.asarray(ids, str), scores))
    out = np.empty(len(order), dtype=int)
    out[order] = np.arange(1, len(order) + 1)
    return out


def pair_data(panels, targets):
    xx, yy, ww = [], [], []
    for x, y in zip(panels, targets):
        a, b = np.triu_indices(len(x), 1)
        xx.append(x[b] - x[a])
        yy.append((y[a] < y[b]).astype(float))
        ww.append(np.full(len(a), 1. / len(a) / len(panels)))
    return np.vstack(xx), np.concatenate(yy), np.concatenate(ww)


def objective(beta, x, y, w, penalty):
    z = x @ beta
    loss = np.dot(w, np.logaddexp(0., z) - y * z) + .5 * np.dot(penalty * beta, beta)
    grad = x.T @ (w * (expit(z) - y)) + penalty * beta
    return float(loss), grad


def fit_predict(panels, events, current, ids, config):
    raw = np.vstack(panels)
    weight = np.concatenate([np.full(len(p), 1. / len(p)) for p in panels])
    scaler = StandardScaler().fit(raw, sample_weight=weight)
    x = scaler.transform(raw)
    z = scaler.transform(current)
    if config["family"] == "pairwise":
        splits = np.cumsum([len(p) for p in panels])[:-1]
        pairs, target, w = pair_data(np.split(x, splits), [e.target for e in events])
        penalty = np.full(x.shape[1], config["penalty"])
        penalty[0] = config["q_penalty"]
        initial = np.zeros(x.shape[1]); initial[0] = 1.
        fit = minimize(objective, initial, args=(pairs, target, w, penalty), jac=True, method="L-BFGS-B",
                       bounds=[(0, None), *[(None, None)] * (x.shape[1] - 1)],
                       options={"maxiter": config["max_iterations"], "gtol": config["gradient_tolerance"] / 10, "ftol": 1e-14})
        value, gradient = objective(fit.x, pairs, target, w, penalty)
        if fit.x[0] <= 1e-12:
            gradient[0] = min(gradient[0], 0.)
        error = float(np.max(np.abs(gradient)))
        if error > config["gradient_tolerance"]:
            raise ValueError(f"pairwise projected gradient {error} exceeds tolerance: {fit.message}")
        score = z @ fit.x
        detail = {"objective": value, "projected_gradient_max": error,
                  "solver_success": bool(fit.success), "iterations": int(fit.nit), "pairs": len(pairs),
                  "coefficients": fit.x.tolist()}
    else:
        model = HistGradientBoostingRegressor(loss="absolute_error", max_leaf_nodes=config["leaves"],
                    max_iter=config["iterations"], min_samples_leaf=config["min_samples_leaf"],
                    learning_rate=config["learning_rate"], l2_regularization=config["l2"],
                    early_stopping=False, random_state=config["random_state"])
        y = np.concatenate([(e.target - e.qualifying) / (len(e.ids) - 1) for e in events])
        model.fit(x, y, sample_weight=weight / weight.mean())
        score = current[:, 0] + model.predict(z)
        detail = {"iterations": int(model.n_iter_)}
    return rank(score, ids), detail
