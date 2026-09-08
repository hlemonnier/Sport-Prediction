"""Math/representation tests; all generated rows are synthetic."""
import copy
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from sklearn.tree import DecisionTreeRegressor
from research.experiments.boundary_20260908.football import test_math
from research.experiments.boundary_20260908.football.cycle2 import model, run


def rows(n=360):
    result = test_math.rows(n)
    for row in result:
        row["probabilities"]["dc_180"] = row["probabilities"]["dc_365"]
    return result


def test_leaf_full_hessian_and_gradient():
    rng = np.random.default_rng(124)
    z = rng.normal(size=(110, 3)); labels = rng.integers(0,3,110);w=rng.normal(size=3)
    value, gradient, hessian = model.leaf_objective(w, z, labels, 25.)
    ng = approx_derivative(lambda v: model.leaf_objective(v,z,labels,25.)[0],w).ravel()
    nh = approx_derivative(lambda v: model.leaf_objective(v,z,labels,25.)[1],w)
    np.testing.assert_allclose(gradient, ng, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(hessian, nh, atol=1e-7, rtol=1e-7)
    assert np.linalg.eigvalsh(hessian).min() >= 25.-1e-10
    optimum, diagnostic = model.fit_leaf(z,labels,25.,run.specification()["numerics"])
    assert diagnostic["gradient_inf"] <= 1e-6
    assert model.leaf_objective(optimum,z,labels,25.)[0] <= model.leaf_objective(np.zeros(3),z,labels,25.)[0]
    np.testing.assert_allclose(optimum.sum(), 0., atol=1e-12)


def test_serialized_threshold_nextafter_routing_matches_sklearn():
    tree = DecisionTreeRegressor(max_depth=1).fit([[.1],[.2],[.3],[.4]], [0.,1.,1.,1.])
    t = tree.tree_;threshold=t.threshold[0]
    probes = np.array([[np.nextafter(threshold,-np.inf)],[threshold],[np.nextafter(threshold,np.inf)]],dtype=np.float64)
    # These straddle the float64 threshold but all round to the same float32.
    assert probes[0,0] < threshold < probes[2,0]
    vectors=np.repeat(np.arange(t.node_count)[:,None],3,axis=1)
    serialized={"children_left":t.children_left.tolist(),"children_right":t.children_right.tolist(),
        "feature":t.feature.tolist(),"threshold":t.threshold.tolist(),"leaf_vectors":vectors.tolist()}
    np.testing.assert_array_equal(model.tree_predict(serialized,probes),vectors[tree.apply(probes)])


def test_identity_stage_objective_and_future_input_immutability():
    training=rows();future=rows(11)
    np.testing.assert_allclose(model.predict(future,{"trees":[]}),[r["probabilities"]["dc365_elo50"] for r in future],atol=2e-16)
    spec=copy.deepcopy(run.specification());spec["tree"]["min_samples_leaf"]=20
    fitted=model.fit(training,{"depth":2,"trees":5},spec)
    assert np.all(np.diff(fitted["objective_history"]) <= 1e-10)
    assert fitted["max_leaf_gradient_inf"] <= 1e-6
    saved=copy.deepcopy(fitted)
    p=model.predict(future,fitted)
    assert fitted==saved
    assert np.isfinite(p).all() and (p>0).all()
    np.testing.assert_allclose(p.sum(1),1.,atol=1e-12)


def test_parent_inputs_remain_frozen_and_grid_is_three():
    spec=run.specification();assert len(spec["candidates"])==3
    features,scoring,hashes=run.inputs("selection",spec)
    assert len(features)==5700 and len(scoring)==760 and len(hashes)==3
    assert {r["season"] for r in scoring}=={2022,2023}
    assert {r["league"] for r in scoring}=={"E0"}
    assert set(spec["required_research_references"])=={"dc365_elo50","dc_180","shot_strength_90d_ridge0.1"}
