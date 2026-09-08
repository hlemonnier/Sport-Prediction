"""Synthetic objective, availability and exact-reference regression tests."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from scipy.special import softmax

from research.experiments.boundary_20260908.football import models as prior
from research.experiments.boundary_20260908.football_xg.execution import models as m
from research.experiments.boundary_20260908.football_xg_data.availability import proxy_available_at


OPTIONS = {"alpha": .05, "solver": "newton-cholesky", "max_iter": 200, "tol": 1e-8,
           "history_days": 1095, "delay_days": 7, "intercept_penalized": False}
CONFIGS = [{"half_life": h, "mode": mode} for h in (90, 365) for mode in ("add", "replace")]


def row(day="2025-03-29", **updates):
    value = {"league": "E0", "canonical_date": day, "provider_date": day,
             "accepted_exact_join": "True", "canonical_match_id": "E0:2024:A:B",
             "home": "A", "away": "B", "home_goals": "1", "away_goals": "0",
             "canonical_home_goals": "1", "canonical_away_goals": "0",
             "home_xg": "1.37", "away_xg": ".81"}
    value.update(updates)
    return value


class UnreadableMeasurements(dict):
    """Future raw providers may be poisoned without touching earlier forecasts."""
    def __getitem__(self, key):
        if key in {"home_xg", "away_xg", "home_goals", "away_goals", "canonical_home_goals",
                   "canonical_away_goals", "provider_date", "label"}:
            raise AssertionError(f"unavailable measurement read: {key}")
        return super().__getitem__(key)


def historical_rows():
    rng = np.random.default_rng(7)
    teams = ["A", "B", "C", "D"]
    data = []
    for i in range(120):
        home, away = rng.choice(teams, 2, replace=False)
        date = (datetime(2022, 1, 1) + timedelta(days=i)).date().isoformat()
        h = float(np.exp(.15 + .1 * teams.index(home) - .1 * teams.index(away)) + rng.uniform(-.2, .2))
        a = float(np.exp(.1 * teams.index(away) - .1 * teams.index(home)) + rng.uniform(-.2, .2))
        data.append(row(date, home=str(home), away=str(away), home_xg=str(h), away_xg=str(a),
                        canonical_match_id=f"synthetic:{i}"))
    return data


def prediction_rows(n=120):
    rng = np.random.default_rng(91)
    data = []
    for i in range(n):
        xg = rng.normal(0, .3, 2)
        data.append({"league": ("E0", "SP1", "I1")[i % 3],
                     "probabilities": {key: softmax(rng.normal(0, .4, 3)).tolist()
                                       for key in ("dc365_elo50", "dc_365", "elo_component")},
                     "log_shot_means": {"90": rng.normal(1.5, .3, 4).tolist()},
                     "log_xg_means": {"90": xg.tolist(), "365": (xg * .7).tolist()},
                     "label": int(rng.integers(0, 3))})
    return data


def numerical_gradient(fn, values, step=1e-6):
    out = np.empty_like(values)
    for i in range(len(values)):
        hi, lo = values.copy(), values.copy()
        hi[i] += step
        lo[i] -= step
        out[i] = (fn(hi) - fn(lo)) / (2 * step)
    return out


@pytest.mark.parametrize("league,day,expected", [
    ("E0", "2025-03-29", "2025-04-06T00:00:00"),
    ("E0", "2025-10-25", "2025-11-01T23:00:00"),
    ("SP1", "2025-03-29", "2025-04-05T23:00:00"),
    ("I1", "2025-10-25", "2025-11-01T22:00:00"),
    ("I1", "2025-02-06", "2025-02-13T23:00:00"),
])
@pytest.mark.parametrize("delay", [7, 14])
def test_availability_uses_elapsed_utc_hours_across_dst_and_completion_day(league, day, expected, delay):
    r = row(day, league=league)
    expected = datetime.fromisoformat(expected) + timedelta(days=delay - 7)
    assert m.available_at(r, delay) == expected
    assert m.available_at(r, delay).tzinfo is None
    assert m.available_at(r, delay) == proxy_available_at(r, delay).replace(tzinfo=None)


@pytest.mark.parametrize("accepted", [True, "True"])
def test_only_exact_python_or_csv_true_is_accepted(accepted):
    r = row(accepted_exact_join=accepted)
    assert m.eligible_history([r], "E0", datetime(2025, 4, 7), 7) == [r]


@pytest.mark.parametrize("accepted", [False, "False", "false", "true", "TRUE", "1", 1, 0, None, np.bool_(True)])
def test_nonaccepted_flags_are_never_truthiness_cast(accepted):
    r = UnreadableMeasurements(row(accepted_exact_join=accepted))
    assert m.available_at(r, 7) is None
    assert m.eligible_history([r], "E0", datetime(2025, 4, 7), 7) == []


@pytest.mark.parametrize("delay", [True, False, 0, -1, 6, 8, 7., "7", np.nan, np.inf])
def test_delay_parameters_cannot_change_frozen_contract(delay):
    with pytest.raises(ValueError):
        m.available_at(row(), delay)


def test_cutoff_is_explicit_utc_and_availability_equality_is_allowed():
    r = row()
    exact = datetime(2025, 4, 6)
    before = exact - timedelta(microseconds=1)
    assert m.eligible_history([r], "E0", before, 7) == []
    for cutoff in (exact, exact.replace(tzinfo=timezone.utc),
                   exact.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York"))):
        chosen = m.eligible_history([r], "E0", cutoff, 7)
        assert len(chosen) == 1 and chosen[0] is r
    with pytest.raises(ValueError, match="datetime"):
        m.eligible_history([r], "E0", "2025-04-06", 7)


def test_history_window_and_league_filter_preserve_original_records():
    cutoff = datetime(2022, 7, 1)
    lower = row("2022-06-01")
    outside = UnreadableMeasurements(row("2022-05-31"))
    wrong = UnreadableMeasurements(row("2022-06-01", league="SP1"))
    assert m.eligible_history([outside, wrong, lower], "E0", cutoff, 7, history_days=30) == [lower]
    assert m.eligible_history([lower], "E0", cutoff, 7, history_days=30)[0] is lower
    for invalid in (0, -10, True, 1.5):
        with pytest.raises(ValueError):
            m.eligible_history([lower], "E0", cutoff, 7, history_days=invalid)


def test_future_recent_and_unaccepted_poisoned_outcomes_are_unread():
    past = row("2022-05-01")
    current = UnreadableMeasurements(row("2022-07-01"))
    future = UnreadableMeasurements(row("2022-08-01"))
    waiting = UnreadableMeasurements(row("2022-06-28"))
    cutoff = datetime(2022, 7, 1)
    assert m.eligible_history([current, past, future, waiting], "E0", cutoff, 7) == [past]
    # Direct public fitting independently rejects bad timing before touching xG/goals.
    for bad in (current, future, waiting):
        with pytest.raises(ValueError, match="unavailable"):
            m.fit_strength([past, bad], cutoff, 90, OPTIONS)


def test_14_day_history_is_rebuilt_and_never_reuses_7_day_eligibility():
    r = row("2022-06-20")
    cutoff = datetime(2022, 7, 1)
    assert m.eligible_history([r], "E0", cutoff, 7) == [r]
    assert m.eligible_history([UnreadableMeasurements(r)], "E0", cutoff, 14) == []
    with pytest.raises(ValueError, match="unavailable"):
        m.fit_strength([UnreadableMeasurements(r)], cutoff, 90, {**OPTIONS, "delay_days": 14})


@pytest.mark.parametrize("update", [{"provider_date": "2025-03-28"}, {"canonical_home_goals": "2"}])
def test_available_rows_revalidate_frozen_exact_join(update):
    with pytest.raises(ValueError, match="accepted join"):
        m.eligible_history([row(**update)], "E0", datetime(2025, 4, 8), 7)


def test_poisson_quasi_gradient_matches_finite_difference_for_continuous_xg():
    rng = np.random.default_rng(19)
    x = rng.normal(0, .3, (14, 5))
    y, weights = rng.uniform(0, 3, 14), rng.uniform(.1, 2, 14)
    parameters = rng.normal(0, .2, 6)
    value, gradient = m.quasi_objective(parameters, x, y, weights, .05)
    numeric = numerical_gradient(lambda a: m.quasi_objective(a, x, y, weights, .05)[0], parameters)
    np.testing.assert_allclose(gradient, numeric, atol=4e-10, rtol=1e-7)
    unpenalized = m.quasi_objective(parameters, x, y, weights, 0)[1]
    np.testing.assert_allclose(gradient[:-1] - unpenalized[:-1], .05 * parameters[:-1], atol=1e-15)
    assert gradient[-1] == unpenalized[-1]
    repeated = m.quasi_objective(parameters, np.tile(x, (2, 1)), np.tile(y, 2), np.tile(weights, 2), .05)
    assert value == pytest.approx(repeated[0], abs=1e-15)
    np.testing.assert_allclose(gradient, repeated[1], atol=1e-15)


def test_strength_fit_fractional_targets_score_and_json_replay():
    data = historical_rows()
    model = m.fit_strength(data, datetime(2022, 7, 1), 90, OPTIONS)
    assert model.diagnostics["gradient_max"] <= 1e-6
    assert model.diagnostics["fit_n"] == 120
    assert model.diagnostics["training_ids"] == [r["canonical_match_id"] for r in data]
    assert 1 < model.diagnostics["effective_match_count"] <= 120
    x = prior.count_design(model.teams, [(r["home"], r["away"]) for r in data])
    y = np.array([[r["home_xg"], r["away_xg"]] for r in data], float).ravel()
    ages = np.array([(datetime(2022, 7, 1) - datetime.fromisoformat(r["canonical_date"])).days for r in data])
    weights = np.repeat(np.exp2(-ages / 90), 2)
    value, score = m.quasi_objective(np.r_[model.coef, model.intercept], x, y, weights, .05)
    assert np.max(np.abs(score)) <= 1e-6
    assert value == pytest.approx(model.diagnostics["objective_value"], abs=1e-14)
    clone = m.XGStrength.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
    fixtures = [("A", "B"), ("B", "A"), ("new home", "new away")]
    np.testing.assert_array_equal(clone.predict(fixtures), model.predict(fixtures))
    expected_unseen = [model.intercept + model.coef[-1], model.intercept]
    np.testing.assert_allclose(model.predict(fixtures)[-1], expected_unseen, atol=0, rtol=0)
    assert model.predict([]).shape == (0, 2)


def test_future_poison_cannot_change_prior_fitted_strength_or_forecast():
    data = historical_rows()
    cutoff = datetime(2022, 7, 1)
    future = [UnreadableMeasurements(row("2022-07-01")), UnreadableMeasurements(row("2022-08-01")),
              UnreadableMeasurements(row("2022-06-30"))]
    a = m.fit_strength(m.eligible_history(data, "E0", cutoff, 7), cutoff, 90, OPTIONS)
    b = m.fit_strength(m.eligible_history(data + future, "E0", cutoff, 7), cutoff, 90, OPTIONS)
    np.testing.assert_array_equal(a.coef, b.coef)
    assert a.intercept == b.intercept
    np.testing.assert_array_equal(a.predict([("A", "D")]), b.predict([("A", "D")]))


@pytest.mark.parametrize("key,value", [
    ("alpha", 0), ("alpha", -1), ("alpha", np.nan), ("alpha", True),
    ("tol", 0), ("max_iter", 1.5), ("max_iter", True), ("solver", "random"),
    ("intercept_penalized", True), ("intercept_penalized", "false"),
])
def test_strength_rejects_invalid_hyperparameters_before_fitting(key, value):
    with pytest.raises(ValueError):
        m.fit_strength(historical_rows(), datetime(2022, 7, 1), 90, {**OPTIONS, key: value})


@pytest.mark.parametrize("half_life", [0, -1, np.nan, np.inf, True])
def test_strength_rejects_bad_decay(half_life):
    with pytest.raises(ValueError):
        m.fit_strength(historical_rows(), datetime(2022, 7, 1), half_life, OPTIONS)


@pytest.mark.parametrize("bad_xg", [-.1, "nan", "inf", True])
def test_strength_rejects_invalid_available_measurements(bad_xg):
    data = historical_rows()
    data[0]["home_xg"] = bad_xg
    with pytest.raises(ValueError, match="xG"):
        m.fit_strength(data, datetime(2022, 7, 1), 90, OPTIONS)


def test_strength_rejects_zero_total_support_and_duplicate_observations():
    data = historical_rows()
    with pytest.raises(ValueError, match="distinct canonical"):
        m.fit_strength(data + [data[0]], datetime(2022, 7, 1), 90, OPTIONS)
    for r in data:
        r["home_xg"] = r["away_xg"] = "0"
    with pytest.raises(ValueError, match="positive total"):
        m.fit_strength(data, datetime(2022, 7, 1), 90, OPTIONS)


def test_strength_orientation_home_effect_unknown_teams_and_numeric_failures():
    model = m.XGStrength(["A", "B"], np.array([.1, .2, .3, .4, .5]), .6, {})
    np.testing.assert_allclose(model.predict([("A", "B"), ("B", "A")]), [[1.6, 1.1], [1.6, 1.1]])
    np.testing.assert_allclose(model.predict([("new", "A")]), [[1.4, .7]])
    for intercept in (1000., -1000.):
        invalid = m.XGStrength(["A", "B"], np.zeros(5), intercept, {})
        with pytest.raises(ValueError, match="positive and finite"):
            invalid.predict([("A", "B")])
    for fixture in (("A", "A"), ("A", ""), ("A", "B", "C")):
        with pytest.raises(ValueError, match="fixture"):
            model.predict([fixture])


def test_offset_objective_analytic_gradient_and_identity_prior():
    rng = np.random.default_rng(55)
    x = np.column_stack([np.ones(18), rng.normal(size=(18, 4))])
    p = softmax(rng.normal(size=(18, 3)), axis=1)
    y = rng.integers(0, 3, 18)
    w = rng.normal(0, .1, x.shape[1] * 3)
    value, gradient = m.offset_objective(w, x, np.log(p), y, .1)
    numeric = numerical_gradient(lambda a: m.offset_objective(a, x, np.log(p), y, .1)[0], w)
    np.testing.assert_allclose(gradient, numeric, atol=4e-10, rtol=1e-7)
    identity, _ = m.offset_objective(np.zeros_like(w), x, np.log(p), y, .1)
    assert identity == pytest.approx(-np.log(p[np.arange(len(y)), y]).mean(), abs=1e-15)
    assert np.isfinite(value)


@pytest.mark.parametrize("config", CONFIGS)
def test_all_four_offsets_fit_with_coherent_probabilities_and_exact_json_replay(config):
    rows = prediction_rows()
    original = deepcopy(rows)
    model = m.fit_offset(rows, config)
    assert rows == original
    assert model.diagnostics["gradient_max"] <= 1e-6
    assert model.diagnostics["objective"] <= model.diagnostics["identity_objective"] + 1e-10
    predictions = model.predict(rows)
    np.testing.assert_allclose(predictions.sum(1), 1., atol=2e-16)
    assert (predictions > 0).all()
    assert model.predict([]).shape == (0, 3)
    clone = m.Offset.from_dict(json.loads(json.dumps(model.to_dict(), allow_nan=False)))
    np.testing.assert_array_equal(clone.predict(rows), predictions)
    # Predict reads covariates only, never current outcomes or raw xG.
    poisoned = [UnreadableMeasurements(r) for r in rows]
    np.testing.assert_array_equal(model.predict(poisoned), predictions)
    reference = model.predict(rows[:1])
    distractor = deepcopy(rows[-1])
    distractor["log_xg_means"][str(config["half_life"])] = [50, -50]
    before = (model.mean.copy(), model.scale.copy())
    np.testing.assert_array_equal(model.predict(rows[:1] + [distractor])[:1], reference)
    np.testing.assert_array_equal(model.mean, before[0])
    np.testing.assert_array_equal(model.scale, before[1])


def test_prior_selected_shot_identity_ablation_retains_exact_features_and_predictions():
    rows = prediction_rows()
    config = {"mode": "add", "half_life": 90}
    base = prior.fit_offset(rows, "shot_strength_90d", .1)
    padded = m.Offset(config, np.r_[base.mean, 0., 0.], np.r_[base.scale, 1., 1.],
                      np.vstack([base.weights, np.zeros((2, 3))]), {})
    for r in rows:
        np.testing.assert_array_equal(m.feature_vector(r, config)[:-2], prior.feature_vector(r, "shot_strength_90d"))
    np.testing.assert_allclose(padded.predict(rows), base.predict(rows), atol=2e-16, rtol=0)
    # Refitting with constant xG adds exactly zero standardized columns, so it
    # also reproduces the old optimum instead of silently changing the prior.
    constant = deepcopy(rows)
    for r in constant:
        r["log_xg_means"]["90"] = [0., 0.]
    refit = m.fit_offset(constant, config)
    np.testing.assert_allclose(refit.predict(constant), base.predict(rows), atol=5e-9, rtol=0)
    assert refit.diagnostics["objective"] == pytest.approx(base.diagnostics["objective"], abs=1e-13)
    np.testing.assert_array_equal(refit.weights[-2:], np.zeros((2, 3)))


@pytest.mark.parametrize("key", ["dc365_elo50", "dc_365", "elo_component"])
@pytest.mark.parametrize("value", [[.4, .3, .4], [0., .5, .5], [-.1, .5, .6], [np.nan, .5, .5], [.5, .5]])
def test_incoherent_prior_probabilities_are_rejected(key, value):
    r = prediction_rows(1)[0]
    r["probabilities"][key] = value
    with pytest.raises(ValueError):
        m.feature_vector(r, CONFIGS[0])


@pytest.mark.parametrize("config", [
    {"mode": "unknown", "half_life": 90}, {"mode": "add", "half_life": 180},
    {"mode": "replace", "half_life": True}, {"mode": "add", "half_life": 90.},
])
def test_unfrozen_candidate_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        m.feature_vector(prediction_rows(1)[0], config)


@pytest.mark.parametrize("label", [True, "1", -1, 3, 1.5, np.nan])
def test_offset_labels_cannot_be_silently_cast(label):
    rows = prediction_rows(3)
    rows[0]["label"] = label
    with pytest.raises(ValueError, match="labels"):
        m.fit_offset(rows, CONFIGS[0])


@pytest.mark.parametrize("penalty", [0, -1, True, np.inf, np.nan])
def test_offset_requires_finite_positive_identity_penalty(penalty):
    with pytest.raises(ValueError):
        m.fit_offset(prediction_rows(3), CONFIGS[0], penalty)


def test_replace_features_do_not_read_shot_means_or_target():
    r = prediction_rows(1)[0]
    del r["log_shot_means"]
    del r["label"]
    assert len(m.feature_vector(r, {"mode": "replace", "half_life": 365})) == 8


def test_serialized_model_schema_shapes_and_arrays_are_validated():
    strength = m.XGStrength(["A", "B"], np.zeros(5), 0., {})
    for update in ({"schema_version": True}, {"model_type": "other"}, {"coef": [0]}, {"teams": ["A", "A"]}):
        with pytest.raises(ValueError):
            m.XGStrength.from_dict({**strength.to_dict(), **update})
    config = {"mode": "replace", "half_life": 90}
    offset = m.Offset(config, np.zeros(8), np.ones(8), np.zeros((9, 3)), {})
    for update in ({"scale": [0.] * 8}, {"mean": [np.nan] * 8}, {"weights": [[0., 0., 0.]]}):
        with pytest.raises(ValueError):
            m.Offset.from_dict({**offset.to_dict(), **update})
    # Serialization returns independent structures; mutating a payload does not
    # alter the live model or its diagnostic provenance.
    payload = offset.to_dict()
    payload["weights"][0][0] = 123.
    payload["config"]["mode"] = "add"
    assert offset.weights[0, 0] == 0.
    assert offset.config["mode"] == "replace"
