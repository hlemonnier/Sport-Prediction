"""Synthetic regressions are mathematical checks, not performance evidence."""
from datetime import datetime, timedelta
from unittest.mock import patch
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from packages.football.mrp.config import PredictionConfig
from packages.football.mrp.data import FixtureRecord, LocalFootballData, MatchRecord
from packages.football.mrp import prediction
from research.experiments.boundary_20260908.football import models, run


def matches(n=240):
    rng = np.random.default_rng(178)
    return [MatchRecord(str(i), datetime(2020, 1, 1)+timedelta(days=i), 2020, "synthetic", i,
        f"team{i%8}", f"team{(i+1+i//8)%8}" if (1+i//8)%8 else f"team{(i+1)%8}",
        int(rng.poisson(1.5)), int(rng.poisson(1.1)), None, None) for i in range(n)]


def rows(n=150):
    rng = np.random.default_rng(13)
    result = []
    for i in range(n):
        dc, elo = rng.dirichlet([4., 2., 3.], size=2)
        mix = (dc+elo)/2
        result.append({"match_id": str(i), "league": ["E0", "SP1", "I1"][i%3],
            "probabilities": {"dc_365": dc, "elo_component": elo, "dc365_elo50": mix},
            "label": int(rng.choice(3, p=mix)),
            "log_shot_means": {"90": rng.normal(1.5, .2, 4), "365": rng.normal(1.5, .2, 4)}})
    return result


def test_offset_gradient_and_identity():
    rng = np.random.default_rng(15)
    x = np.column_stack((np.ones(14), rng.normal(size=(14, 4))))
    prior = rng.dirichlet([2, 2, 2], size=14)
    labels = rng.integers(0, 3, 14)
    w = rng.normal(size=15)
    args = x, np.log(prior), labels, .1
    _, gradient = models.offset_objective(w, *args)
    numerical = approx_derivative(lambda z: models.offset_objective(z, *args)[0], w).ravel()
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=1e-6)
    identity_loss, _ = models.offset_objective(np.zeros(15), *args)
    assert identity_loss == pytest.approx(-np.log(prior[np.arange(14), labels]).mean())


@pytest.mark.parametrize("mechanism", ["log_probability_residual_only", "shot_strength_90d", "shot_strength_365d"])
def test_offset_predict_identity_and_fit_prefix(mechanism):
    train, future = rows(), rows(12)
    f = np.asarray([models.feature_vector(r, mechanism) for r in train])
    zero = models.OffsetModel(mechanism, f.mean(0), np.ones(f.shape[1]), np.zeros((f.shape[1]+1, 3)), {})
    np.testing.assert_allclose(zero.predict(future), [r["probabilities"]["dc365_elo50"] for r in future], atol=2e-16)
    model = models.fit_offset(train, mechanism, .01)
    assert model.diagnostics["objective"] <= model.diagnostics["identity_objective"]
    assert model.diagnostics["gradient_inf_norm"] < 1e-6
    np.testing.assert_allclose(model.mean, f.mean(0))
    before = model.weights.copy()
    model.predict(future)
    np.testing.assert_array_equal(before, model.weights)
    np.testing.assert_allclose(model.predict(future).sum(1), 1.)


def test_shot_design_opponents_home_order_and_unknown_team():
    x = models.count_design(["a", "b"], [("a", "b"), ("unknown", "a")])
    np.testing.assert_array_equal(x, [[1,0,0,1,1], [0,1,1,0,0], [0,0,1,0,1], [1,0,0,0,0]])


def test_shot_fit_never_reads_forecast_statistics():
    history = matches(80)
    spec = run.specification()["shot_strength_model"]
    stats = {m.match_id: [10+i%7, 8+i%5, 3+i%4, 2+i%3] for i,m in enumerate(history)}
    cutoff = history[-1].date+timedelta(days=2)
    model = models.fit_shot_strength(history, stats, cutoff, 90, spec)
    stats["future_fixture"] = [float("nan")]*4
    poisoned = models.fit_shot_strength(history, stats, cutoff, 90, spec)
    np.testing.assert_array_equal(model.predict([("team0", "team1")]), poisoned.predict([("team0", "team1")]))
    assert model.predict([("promoted", "team1")]).shape == (1, 4)
    with pytest.raises(ValueError, match="strictly previous"):
        models.fit_shot_strength(history, stats, history[-1].date, 90, spec)
    stats[history[0].match_id][0] = float("nan")
    with pytest.raises(ValueError, match="finite nonnegative integers"):
        models.fit_shot_strength(history, stats, cutoff, 90, spec)


def test_meta_training_uses_availability_and_never_same_day():
    cutoff = datetime(2024, 9, 1)
    candidates = [
        {"match_id":"prior", "forecast_cutoff_utc":"2024-08-31T00:00:00", "result_available_at":"2024-09-01T00:00:00"},
        {"match_id":"late", "forecast_cutoff_utc":"2024-08-20T00:00:00", "result_available_at":"2024-09-01T00:00:01"},
        {"match_id":"today", "forecast_cutoff_utc":"2024-09-01T00:00:00", "result_available_at":"2024-09-02T00:00:00"},
        {"match_id":"expired", "forecast_cutoff_utc":"2017-08-01T00:00:00", "result_available_at":"2017-08-02T00:00:00"}]
    assert [r["match_id"] for r in run.prior_training_rows(candidates, cutoff)] == ["prior"]


def test_comparator_matches_frozen_assessment_prediction_api():
    history = matches()
    target = FixtureRecord("target", history[-1].date+timedelta(days=4), 2020, "synthetic", 250, "team0", "team1")
    data = LocalFootballData(None, {}, history, [target])
    config = PredictionConfig("synthetic", 2020, 250, "1x2", shadow_eval=False)
    model, cal, groups = models.production_default_fit(history)
    expected = models.production_default_predict(model, cal, "team0", "team1")
    # This comparator remains the historical prefix policy. The named assessment
    # caller preserves it after the separately verified production-default update.
    # Only IO/population selectors are replaced; fitting and reconciliation run.
    with patch.object(prediction, "load_local_football_data", return_value=(data, [])), \
         patch.object(prediction, "select_target_fixtures", return_value=([target], [])), \
         patch.object(prediction, "select_training_matches", return_value=(history, [])):
        actual = prediction.run_assessment_prediction(config)
    row = actual.rows[0]
    np.testing.assert_allclose(expected, [float(row[k]) for k in ["home_win_prob", "draw_prob", "away_win_prob"]], atol=6e-13, rtol=0)
    for key, value in groups.metadata().items():
        assert actual.diagnostics["protocol"][key] == value
    assert actual.diagnostics["model_used"] == "dixon"


def test_frozen_six_candidates_and_manifest_inputs():
    spec = run.specification()
    assert len(run.configurations(spec)) == 6
    required = [run.b.DATA / (f"E0_{year}_{year+1}.csv" if league == "E0" else f"transfer_{league}_{year}_{year+1}.csv")
        for league in spec["leagues"] for year in range(2017, 2026)]
    if not all(path.is_file() for path in required):
        pytest.skip("Optional raw-provider audit requires the 27 closed local season CSVs")
    for league, zone in spec["leagues"].items():
        with run.league_context(zone):
            records, stats, _, hashes = run.load_league(league, list(range(2017, 2026)), spec)
        assert len(records) == 3420 and len(hashes) == 9
        a = np.asarray(list(stats.values()))
        assert np.isfinite(a).all() and (a >= 0).all() and (a == np.floor(a)).all()
        # One existing provider row violates target <= total. The frozen model
        # uses separate count GLMs, so retain rather than invent a replacement.
        violations = [mid for mid, v in stats.items() if v[2] > v[0] or v[3] > v[1]]
        assert violations == (["E0:2021:Newcastle:West Ham"] if league == "E0" else [])
