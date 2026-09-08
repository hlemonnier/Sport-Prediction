from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.race_practice import features, models, run


def raw_laps(drivers="ABCDE"):
    rows = []
    for i, driver in enumerate(drivers):
        for age in range(1, 7):
            rows.append({"Driver": driver, "Time": 100 + 100 * age + i,
                         "LapTime": 90 + .15 * age + .4 * i, "Stint": 1,
                         "Compound": "MEDIUM", "TyreLife": age, "IsAccurate": True,
                         "TrackStatus": 1, "PitInTime": np.nan, "PitOutTime": np.nan})
    return pd.DataFrame(rows)


def test_independent_peers_and_quality_filters():
    raw = raw_laps()
    measured, audit = features.session_measurements(raw)
    assert audit["matched_laps"] == 30
    assert np.median(measured["A"]["delta"]) < 0 < np.median(measured["E"]["delta"])
    # Many laps of a single other driver are not independent support.
    _, two = features.session_measurements(raw.loc[raw.Driver.isin(["A", "B"])])
    assert two["matched_laps"] == 0
    for column, value in [("IsAccurate", False), ("TrackStatus", "14"), ("PitInTime", 999), ("TyreLife", np.nan)]:
        bad = raw.copy(); bad[column] = value
        _, result = features.session_measurements(bad)
        assert result["clean_laps"] == 0


def test_measurements_ignore_race_labels_and_are_row_equivariant():
    raw = raw_laps(); expected, audit = features.session_measurements(raw)
    raw["RacePosition"] = np.arange(len(raw)) * 10000
    result, other = features.session_measurements(raw.sample(frac=1, random_state=9))
    assert expected == result and audit == other


def test_unsupported_fallback_and_feature_units():
    ids = list("ABCDE"); teams = list("XXYYZ"); q = np.arange(1, 6)
    blank, support = features.aggregate(ids, teams, q, [])
    assert support["supported_drivers"] == 0
    assert (blank[:, [0, 1, 2, 4, 5, 6, 7, 9]] == 0).all()
    assert (blank[:, [3, 8]] == 1).all()
    raw = raw_laps(); first, _ = features.session_measurements(raw)
    raw.LapTime *= 1.5
    second, _ = features.session_measurements(raw)
    a, _ = features.aggregate(ids, teams, q, [first]); b, _ = features.aggregate(ids, teams, q, [second])
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_only_completed_prequalifying_practice_is_read(tmp_path):
    raw_laps().to_csv(tmp_path / "fp.csv", index=False)
    event = SimpleNamespace(ids=list("ABCDE"), teams=list("XXYYZ"), qualifying=np.arange(1, 6))
    # A temporary root permits an isolated actual-file phase-boundary test.
    original_root = run.ROOT
    try:
        run.ROOT = tmp_path
        allowed = {"session_type": "free_practice", "session_order": 1, "results_rows": 5, "laps_path": "fp.csv"}
        meta = {"sessions": [allowed,
                 {**allowed, "session_type": "race", "session_order": 5, "laps_path": "must_not_open_race.csv"},
                 {**allowed, "session_order": 6, "laps_path": "must_not_open_future_fp.csv"},
                 {**allowed, "completed": False, "session_order": 2, "laps_path": "must_not_open_partial.csv"}]}
        hashes = {}; result, support = run.practice(event, meta, tmp_path, {"session_order": 4}, hashes)
        assert support["supported_drivers"] == 5 and set(hashes) == {"fp.csv"}
    finally:
        run.ROOT = original_root


def test_original_features_do_not_read_current_target():
    event = run.prior.Event("2023:01", 2023, 1, "test", "standard", list("ABCD"), list("XXYY"),
                            np.arange(1, 5), np.arange(4), np.ones(4, dtype=bool))
    x = run.prior.features_for(event, [], 0)
    event.target[:] = 1e9; event.classified_finish[:] = False
    np.testing.assert_array_equal(x, run.prior.features_for(event, [], 0))


def test_pairwise_objective_gradient_weights_and_orientation():
    panels = [np.arange(8).reshape(4, 2) / 8, np.arange(6).reshape(3, 2) / 6]
    targets = [np.arange(1, 5), np.arange(1, 4)]
    x, y, w = models.pair_data(panels, targets)
    assert w[:6].sum() == pytest.approx(.5) and w[6:].sum() == pytest.approx(.5)
    beta = np.array([.5, -.1]); penalty = np.array([.0005, .05])
    loss, gradient = models.objective(beta, x, y, w, penalty)
    for j in range(2):
        change = np.zeros(2); change[j] = 1e-6
        numeric = (models.objective(beta + change, x, y, w, penalty)[0]
                   - models.objective(beta - change, x, y, w, penalty)[0]) / 2e-6
        assert gradient[j] == pytest.approx(numeric, abs=1e-9)
    assert loss < np.log(2)  # Larger scores mean worse rank, so positive coefficient is correct.


@pytest.mark.parametrize("family", ["pairwise_practice", "hgb_practice"])
def test_fitted_models_preserve_permutation_and_current_roster_equivariance(family):
    rng = np.random.default_rng(11); panels, events = [], []
    for _ in range(10):
        q = np.arange(1, 6); x = np.column_stack([(q - 1) / 4, rng.normal(size=5)])
        target = models.rank(x[:, 0] - .4 * x[:, 1], list("ABCDE"))
        panels.append(x); events.append(SimpleNamespace(ids=list("ABCDE"), qualifying=q, target=target))
    config = json.loads((run.HERE / "specification.json").read_text())["candidates"][family]
    current = panels[-1]; ids = list("ABCDE")
    prediction, detail = models.fit_predict(panels, events, current, ids, config)
    perm = [4, 2, 0, 3, 1]
    changed, _ = models.fit_predict(panels, events, current[perm], [ids[i] for i in perm], config)
    assert sorted(prediction.tolist()) == list(range(1, 6))
    assert dict(zip(ids, prediction)) == dict(zip([ids[i] for i in perm], changed))
    if config["family"] == "pairwise":
        assert detail["projected_gradient_max"] < config["gradient_tolerance"]


def test_constant_improvement_has_exact_bootstrap_and_loo():
    base = [{"event": str(i), "year": 2024 + i // 4, "mae": 3.} for i in range(8)]
    candidate = [{**r, "mae": 2.7} for r in base]
    report = run.paired(candidate, base)
    assert report["relative_gain"] == pytest.approx(.1)
    np.testing.assert_allclose(report["block3_ci95"], [-.3, -.3])
    assert report["loo_max"] == pytest.approx(-.3)
