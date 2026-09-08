"""Stagewise multiclass offset boosting with auditable full-Hessian leaf fits."""
import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.tree import DecisionTreeRegressor
from research.experiments.boundary_20260908.football import models as parent


def features(rows):
    result = []
    for row in rows:
        p = np.asarray(row["probabilities"]["dc365_elo50"])
        dc = np.asarray(row["probabilities"]["dc_180"])
        f = parent.feature_vector(row, "shot_strength_90d")
        f.extend([float(p.max()), np.log(dc[0]/dc[2])-np.log(p[0]/p[2]),
                  np.log(dc[1]/dc[2])-np.log(p[1]/p[2])])
        result.append(f)
    # sklearn's tree routing uses float32 inputs; use the same representation
    # for serialized-tree replay, including threshold-adjacent values.
    x = np.asarray(result, dtype=np.float32)
    if x.shape != (len(rows), 13) or not np.isfinite(x).all():
        raise ValueError("Invalid frozen feature matrix")
    return x


def leaf_objective(w, logits, labels, ridge):
    z = logits+w
    p = softmax(z, axis=1)
    loss = float(np.sum(logsumexp(z, axis=1)-z[np.arange(len(labels)), labels]) + ridge/2*(w@w))
    grad = p.sum(0)-np.bincount(labels, minlength=3)+ridge*w
    hessian = np.diag(p.sum(0))-p.T@p+ridge*np.eye(3)
    return loss, grad, hessian


def fit_leaf(logits, labels, ridge, numerics):
    w = np.zeros(3)
    for iteration in range(numerics["leaf_max_iterations"]):
        value, gradient, hessian = leaf_objective(w, logits, labels, ridge)
        residual = float(np.max(np.abs(gradient)))
        if residual <= numerics["leaf_gradient_inf_tolerance"]:
            return w, {"iterations": iteration, "gradient_inf": residual, "objective": value}
        step = np.linalg.solve(hessian, gradient)
        scale = 1.
        for _ in range(numerics["maximum_backtracks"]):
            new = w-scale*step
            new_value = leaf_objective(new, logits, labels, ridge)[0]
            if new_value <= value-numerics["armijo_fraction"]*scale*(gradient@step)+1e-12:
                w = new
                break
            scale *= .5
        else:
            raise RuntimeError("Leaf Newton line search failed")
    raise RuntimeError("Leaf Newton fit failed convergence gate")


def tree_predict(tree, x):
    x = np.asarray(x, dtype=np.float32)
    left, right = np.asarray(tree["children_left"]), np.asarray(tree["children_right"])
    feat, thresh = np.asarray(tree["feature"]), np.asarray(tree["threshold"])
    nodes = np.zeros(len(x), dtype=int)
    active = left[nodes] >= 0
    while active.any():
        ids = np.flatnonzero(active)
        old = nodes[ids]
        nodes[ids] = np.where(x[ids, feat[old]] <= thresh[old], left[old], right[old])
        active = left[nodes] >= 0
    return np.asarray(tree["leaf_vectors"])[nodes]


def predict(rows, fitted):
    x = features(rows)
    logits = np.log(np.asarray([r["probabilities"]["dc365_elo50"] for r in rows]))
    for tree in fitted["trees"]:
        logits += tree_predict(tree, x)
    p = softmax(logits, axis=1)
    if not np.isfinite(p).all() or np.any(p <= 0) or not np.allclose(p.sum(1), 1., atol=1e-12, rtol=0):
        raise ValueError("Offset tree output is not a finite positive simplex")
    return p


def fit(rows, candidate, spec):
    options, numerical = spec["tree"], spec["numerics"]
    x = features(rows)
    y = np.asarray([r["label"] for r in rows], dtype=int)
    prior = np.asarray([r["probabilities"]["dc365_elo50"] for r in rows])
    if not np.isfinite(prior).all() or np.any(prior <= 0):
        raise ValueError("Invalid prior")
    logits = np.log(prior)
    true = np.eye(3)[y]
    initial = float(np.mean(logsumexp(logits, axis=1)-logits[np.arange(len(y)), y]))
    trees, history, max_gradient, penalty = [], [initial], 0., 0.
    for stage in range(candidate["trees"]):
        gradient = true-softmax(logits, axis=1)
        splitter = DecisionTreeRegressor(max_depth=candidate["depth"],
            min_samples_leaf=options["min_samples_leaf"], random_state=options["random_state"]+stage)
        splitter.fit(x, gradient)
        leaf_ids = splitter.apply(x)
        vectors = np.zeros((splitter.tree_.node_count, 3))
        leaf_diagnostics = {}
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            optimum, diagnostic = fit_leaf(logits[mask], y[mask], options["leaf_l2_sum_objective"], numerical)
            vectors[leaf_id] = options["learning_rate"]*optimum
            max_gradient = max(max_gradient, diagnostic["gradient_inf"])
            leaf_diagnostics[str(leaf_id)] = {**diagnostic, "n": int(mask.sum())}
        tree = {"children_left": splitter.tree_.children_left.tolist(), "children_right": splitter.tree_.children_right.tolist(),
                "feature": splitter.tree_.feature.tolist(), "threshold": splitter.tree_.threshold.tolist(),
                "leaf_vectors": vectors.tolist(), "leaves": leaf_diagnostics}
        delta = tree_predict(tree, x)
        # sklearn routes using float32 feature values. Preserve its training
        # partition when generating/serializing forecasts through this module.
        np.testing.assert_allclose(delta, vectors[leaf_ids], atol=0, rtol=0)
        logits += delta
        penalty += options["leaf_l2_sum_objective"]/2*float(np.sum(vectors*vectors))
        objective = float(np.mean(logsumexp(logits, axis=1)-logits[np.arange(len(y)), y]) + penalty/len(y))
        if objective > history[-1]+1e-10 or not np.isfinite(objective):
            raise RuntimeError("Stage objective failed monotonic descent check")
        history.append(objective)
        trees.append(tree)
    return {"trees": trees, "fit_n": len(rows), "configuration": candidate, "initial_objective": initial,
            "objective_history": history, "max_leaf_gradient_inf": max_gradient,
            "optimizer_status": "all_leaf_objectives_converged; greedy tree ensemble, no global convergence claim"}
