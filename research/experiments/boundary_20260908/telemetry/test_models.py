"""Synthetic invariants only: no historical inputs, locks, fits or scores."""
import copy

import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.telemetry import models as m


def frame(n=8):
    rng = np.random.default_rng(123)
    out = pd.DataFrame(rng.normal(size=(n, 80)), columns=m.BASE_FEATURES)
    out['event_key'] = [202201 if i < max(1, n // 4) else 202202 for i in range(n)]
    out['driver_id'] = '44'
    out['issued_after_lap_number'] = np.arange(n) + 1
    out['issued_at_timestamp'] = 100 + np.arange(n) * 90.0
    out['issuance_id'] = [f'synthetic-{i}' for i in range(n)]
    out['forecast_naive_seconds'] = 90.
    out['issued_at_ns'] = (out.issued_at_timestamp * 1_000_000_000).astype('int64')
    out['telemetry_cutoff_ns'] = out.issued_at_ns - 2_000_000_000
    out['telemetry_supported'] = [bool(i % 2) for i in range(n)]
    out['telemetry_values'] = [row.tolist() for row in rng.normal(size=(n, 90))]
    out['year'] = 2022
    out['outcome_status'] = 'matched'
    out['lap_time_seconds'] = 90 + np.linspace(-10, 10, n)
    return out


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    class Recorder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.number = len(calls)
            self.predict_rows = []
            calls.append(self)

        def fit(self, x, y, sample_weight):
            self.x = x.copy()
            self.y = np.asarray(y).copy()
            self.weights = np.asarray(sample_weight).copy()

        def predict(self, x):
            self.predict_rows.append(len(x))
            # Deliberately exceed the frozen correction bound.
            return np.full(len(x), [-20., 20., 2.][self.number])

    monkeypatch.setattr(m.original, 'HistGradientBoostingRegressor', Recorder)
    return calls


def test_full_population_weights_and_exact_hgb_objective(recorder):
    f = frame()
    bundle = m.fit_models(f)
    assert len(recorder) == 3
    assert [x.x.shape for x in recorder] == [(8, 80), (8, 170), (8, 170)]
    expected_weights = np.array([2., 2., 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 2 / 3])
    for fitted in recorder:
        np.testing.assert_array_equal(fitted.y, np.clip(f.lap_time_seconds - 90, -5, 5))
        np.testing.assert_array_equal(fitted.weights, expected_weights)
        assert fitted.kwargs == dict(loss='absolute_error', learning_rate=.06,
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=80,
            l2_regularization=10, early_stopping=False, random_state=20260907)
    assert bundle['fit_summary']['unsupported_training_rows_retained'] == 4
    assert bundle['fit_summary']['rows_per_fit'] == 8


def test_control_zeros_only_declared_content_and_preserves_quality(recorder):
    f = frame()
    f.at[0, 'telemetry_values'][0] = np.nan
    f.at[0, 'telemetry_values'][12] = np.nan
    m.fit_models(f)
    _, content, quality = m.layout()
    a, c = recorder[1].x.to_numpy(), recorder[2].x.to_numpy()
    np.testing.assert_array_equal(c[:, :80], a[:, :80])
    np.testing.assert_array_equal(c[:, 80 + np.array(content)], 0.)
    np.testing.assert_array_equal(c[:, 80 + np.array(quality)], a[:, 80 + np.array(quality)])
    assert recorder[1].x.columns.tolist() == recorder[2].x.columns.tolist()


def test_predictions_clip_at_anchor_and_fallback_is_bitwise_exact(recorder):
    f = frame()
    bundle = m.fit_models(f)
    issued = f.drop(columns=['lap_time_seconds', 'year', 'outcome_status'])
    p = m.predict_models(bundle, issued, latency_seconds=2)
    np.testing.assert_array_equal(p['base_hgb'], 87.)
    mask = f.telemetry_supported.to_numpy()
    np.testing.assert_array_equal(p['telemetry_hgb'][mask], 93.)
    np.testing.assert_array_equal(p['quality_hgb'][mask], 92.)
    for name in ['telemetry_hgb', 'quality_hgb']:
        np.testing.assert_array_equal(p[name][~mask].view(np.uint64), p['base_hgb'][~mask].view(np.uint64))
    assert [x.predict_rows for x in recorder] == [[8], [4], [4]]


def test_sensitivity_reuses_same_fits_and_never_reads_targets(recorder):
    f = frame()
    bundle = m.fit_models(f)
    p2 = m.predict_models(bundle, f, latency_seconds=2)
    zero = f.copy(deep=True)
    zero['telemetry_cutoff_ns'] = zero.issued_at_ns
    zero['lap_time_seconds'] = object()
    zero['outcome_status'] = 'unmatched'
    zero['target_timestamp'] = 'must not be read'
    p0 = m.predict_models(bundle, zero, latency_seconds=0)
    assert len(recorder) == 3
    for name in m.MODEL_NAMES:
        np.testing.assert_array_equal(p2[name], p0[name])


def test_all_unsupported_still_fits_all_rows_but_only_predicts_base(recorder):
    f = frame()
    f['telemetry_supported'] = False
    bundle = m.fit_models(f)
    p = m.predict_models(bundle, f, latency_seconds=2)
    assert len(recorder) == 3
    assert [x.predict_rows for x in recorder] == [[8], [], []]
    for name in m.MODEL_NAMES:
        np.testing.assert_array_equal(p[name], p['base_hgb'])


@pytest.mark.parametrize('column,value', [
    ('year', 2023), ('event_key', 202301), ('outcome_status', 'unmatched'),
    ('lap_time_seconds', np.nan), ('telemetry_supported', 'False'),
    ('telemetry_cutoff_ns', 1), ('issued_at_ns', 1.0),
])
def test_training_rejects_wrong_year_status_or_clock_before_any_fit(recorder, column, value):
    f = frame()
    f[column] = value
    with pytest.raises((ValueError, TypeError)):
        m.fit_models(f)
    assert not recorder


def test_duplicate_rows_cannot_change_event_weights(recorder):
    f = frame()
    f = pd.concat([f, f.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match='duplicate'):
        m.fit_models(f)
    assert not recorder


@pytest.mark.parametrize('latency', [-1, 1, 3, True])
def test_no_other_latency_or_sensitivity_fit(latency):
    with pytest.raises(ValueError, match='frozen'):
        m.matrices(frame(), latency_seconds=latency)


def test_feature_layout_and_invalid_infinity():
    names, content, quality = m.layout()
    assert len(names) == 90 and len(m.BASE_FEATURES) == 80
    assert content == tuple(j for start in [0, 30, 60] for j in range(start, start + 9))
    f = frame()
    f.at[0, 'telemetry_values'][0] = np.inf
    with pytest.raises(ValueError, match='finite-or-NaN'):
        m.matrices(f, latency_seconds=2)


def test_real_synthetic_base_fit_matches_exact_frontier_and_predicts_unmatched():
    from threadpoolctl import threadpool_limits
    f = frame(180)
    with threadpool_limits(limits=1):
        direct = m.original.fit_model(f, dict(m.CONFIG), list(m.BASE_FEATURES))
        bundle = m.fit_models(f)
        expected = m.original.predict_model(f, direct)
    # Prediction neither depends on nor requires target availability/status.
    issued = f.drop(columns=['lap_time_seconds', 'outcome_status'])
    p = m.predict_models(bundle, issued, latency_seconds=2)
    np.testing.assert_array_equal(p['base_hgb'].view(np.uint64), expected.view(np.uint64))
    assert all(np.isfinite(x).all() for x in p.values())
    assert all(np.max(np.abs(x - 90.)) <= 3 for x in p.values())


def test_bundle_rejects_shipped_later_fit(recorder):
    f = frame()
    bundle = m.fit_models(f)
    wrong = copy.deepcopy(bundle)
    wrong['fit_summary']['fit_year'] = [2022, 2023]
    with pytest.raises(ValueError, match='2022 primary'):
        m.predict_models(wrong, f, latency_seconds=2)
