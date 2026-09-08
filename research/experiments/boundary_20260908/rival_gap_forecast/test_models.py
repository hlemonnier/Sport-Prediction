"""Synthetic contracts for the two-fit gap increment; no historical inputs."""
from dataclasses import replace
import copy

import numpy as np
import pandas as pd
import pytest

from . import models as m


def frame(n=8, *, year=2022):
    index = np.arange(n)
    event = year*100+1+index%22
    if n > 100:
        # Unequal event sizes make a row-weight/event-weight error observable.
        event[:n//5] = year*100+1
    values = np.zeros((n, 45));values[:, :12] = np.arange(12)
    supported = index%3 != 0
    values[:, 39] = supported
    issued = (100+90*index)*1_000_000_000
    result = pd.DataFrame(np.zeros((n, 80)), columns=m.BASE_FEATURES)
    return result.assign(event_key=event, driver_id=np.where(index%2, '44', '1'),
        issued_after_lap_number=index+1, issued_at_timestamp=issued/1e9,
        issuance_id=[f'{year}-synthetic-{i}' for i in index], year=year,
        issued_at_ns=issued, gap_cutoff_ns=issued-2_000_000_000,
        gap_supported=supported, gap_values=list(values), forecast_naive_seconds=90.,
        outcome_status='matched', lap_time_seconds=90+np.linspace(-10, 10, n),
        target_lap_number=index+2, target_timestamp=issued/1e9+90, target_same_stint=True)


def target_free(value):
    return value.drop(columns=[c for c in value if c.startswith('target_') or c in ('lap_time_seconds', 'outcome_status')])


class SavedEstimator:
    n_features_in_ = 80

    def __init__(self):
        self.predict_calls = 0

    def fit(self, *args, **kwargs):
        pytest.fail('The saved base must never be fitted')

    def get_params(self):
        return dict(m.BASE_PARAMETERS)

    def predict(self, values):
        self.predict_calls += 1
        assert values.shape[1] == 80
        return np.full(len(values), .123)


def saved_model():
    return dict(kind='hgb', features=list(m.BASE_FEATURES), model=SavedEstimator())


def references(value, points=None):
    if points is None:
        points = np.full(len(value), np.nextafter(90., 91.), dtype=np.float64)
    return m.SavedBaseForecasts(tuple(value.issuance_id), tuple(value.gap_cutoff_ns), points.copy(), 'a'*64)


@pytest.fixture(scope='module')
def full_training():
    return frame(m.TRAIN_ROWS)


@pytest.fixture
def recording(monkeypatch):
    calls = []

    class Estimator:
        def __init__(self, **config):
            self.config = config;self.number = len(calls);self.predict_calls = 0;calls.append(self)

        def fit(self, x, y, sample_weight):
            self.shape = x.shape;self.columns = list(x.columns)
            self.first = x.iloc[0].to_numpy().copy()
            self.y = y.copy();self.weights = sample_weight.copy()

        def predict(self, x):
            self.predict_calls += 1;self.predicted_rows = len(x)
            return np.full(len(x), [-20., 20.][self.number])

    monkeypatch.setattr(m.old_models.original, 'HistGradientBoostingRegressor', Estimator)
    return calls


def fitted(full_training):
    saved = saved_model()
    return m.fit_models(full_training, expected_matched=full_training.copy(),
                        saved_base_model=saved, saved_base_model_sha256='b'*64), saved


def test_two125_column_fits_reuse_base_and_keep_all18363_rows_with_exact_event_weights(full_training, recording):
    value = full_training.copy(deep=True)
    first = value.gap_values.iloc[0].copy();first[:12] = np.nan
    first[12:] = np.arange(33);first[39] = 0
    value.at[0, 'gap_values'] = first
    before = copy.deepcopy(first)
    bundle, saved = fitted(value)
    assert bundle['models']['base_hgb'] is saved and saved['model'].predict_calls == 0
    assert len(recording) == 2 and all(c.shape == (18363, 125) for c in recording)
    assert bundle['fit_summary']['fits'] == 2 and bundle['fit_summary']['base_refits'] == 0
    assert bundle['fit_summary']['unsupported_training_rows_retained'] > 0
    counts = value.groupby('event_key').event_key.transform('size').to_numpy()
    for call in recording:
        assert call.config == m.BASE_PARAMETERS
        assert call.columns == [*m.BASE_FEATURES, *('gap__'+n for n in m.layout()[0])]
        np.testing.assert_allclose(call.weights, len(value)/(22*counts), rtol=1e-15, atol=0)
        np.testing.assert_array_equal(call.y, np.clip(value.lap_time_seconds-90, -5, 5))
    assert np.isnan(recording[0].first[80:92]).all()
    np.testing.assert_array_equal(recording[1].first[80:92], np.zeros(12))
    np.testing.assert_array_equal(recording[0].first[92:], recording[1].first[92:])
    np.testing.assert_array_equal(value.gap_values.iloc[0], before)


def test_closed_base_bits_both_lags_clip_and_unsupported_fallback_without_base_replay(full_training, recording):
    bundle, saved = fitted(full_training)
    issued = target_free(frame(year=2023));reference = references(issued)
    snapshot = reference.predictions.copy();supported = issued.gap_supported.to_numpy()
    result = m.predict_models(bundle, issued, latency_seconds=2, expected_issuances=issued.copy(), saved_base_forecasts=reference)
    assert set(result) == set(m.MODEL_NAMES) and saved['model'].predict_calls == 0
    np.testing.assert_array_equal(result['base_hgb'].view(np.uint64), snapshot.view(np.uint64))
    for name in m.NEW_MODEL_NAMES:
        np.testing.assert_array_equal(result[name][~supported].view(np.uint64), snapshot[~supported].view(np.uint64))
    np.testing.assert_array_equal(result['gap_hgb'][supported], 87.)
    np.testing.assert_array_equal(result['quality_hgb'][supported], 93.)
    assert all(c.predicted_rows == supported.sum() for c in recording)
    zero = issued.copy();zero['gap_cutoff_ns'] = zero.issued_at_ns
    again = m.predict_models(bundle, zero, latency_seconds=0, expected_issuances=zero.copy(),
                             saved_base_forecasts=references(zero, snapshot))
    for name in m.MODEL_NAMES:
        np.testing.assert_array_equal(again[name].view(np.uint64), result[name].view(np.uint64))
    np.testing.assert_array_equal(reference.predictions, snapshot)
    assert len(recording) == 2


def test_all_unsupported_inference_skips_both_new_estimators_and_still_keeps_every_row(full_training, recording):
    bundle, _ = fitted(full_training)
    issued = target_free(frame(year=2023));issued['gap_supported'] = False
    issued['gap_values'] = [np.zeros(45) for _ in range(len(issued))]
    reference = references(issued)
    result = m.predict_models(bundle, issued, latency_seconds=2, expected_issuances=issued.copy(), saved_base_forecasts=reference)
    assert all(c.predict_calls == 0 for c in recording)
    for name in m.MODEL_NAMES:
        np.testing.assert_array_equal(result[name].view(np.uint64), reference.predictions.view(np.uint64))


def test_saved_base_estimator_inference_only_for2022(full_training, recording):
    bundle, saved = fitted(full_training)
    issued = target_free(frame())
    result = m.predict_models(bundle, issued, latency_seconds=2, expected_issuances=issued.copy())
    assert saved['model'].predict_calls == 1
    np.testing.assert_array_equal(result['base_hgb'], 90.123)
    issued = target_free(frame(year=2023))
    with pytest.raises(ValueError, match='2023 inference requires'):
        m.predict_models(bundle, issued, latency_seconds=2, expected_issuances=issued.copy())


@pytest.mark.parametrize('column', ['lap_time_seconds', 'outcome_status', 'target_id', 'target_unseen_future', 'LapTime'])
def test_prediction_rejects_target_columns_before_any_model_call(column):
    issued = target_free(frame(year=2023));issued[column] = np.nan
    with pytest.raises(ValueError, match='target-free'):
        m.predict_models({}, issued, latency_seconds=2, expected_issuances=issued.copy())


@pytest.mark.parametrize('fault', ['rows', 'event', 'year', 'unmatched', 'target', 'order', 'feature', 'id', 'clock'])
def test_training_original_population_cannot_be_narrowed_or_changed(full_training, monkeypatch, fault):
    monkeypatch.setattr(m.old_models.original, 'fit_model', lambda *a, **k: pytest.fail('Invalid cohort reached fitting'))
    value = full_training.copy()
    if fault == 'rows': value = value.iloc[:-1]
    elif fault == 'event': value.loc[value.event_key.eq(202222), 'event_key'] = 202221
    elif fault == 'year': value['year'] = 2023
    elif fault == 'unmatched': value.loc[0, 'outcome_status'] = 'unmatched'
    elif fault == 'target': value.loc[0, 'lap_time_seconds'] += 1
    elif fault == 'order': value = value.iloc[::-1]
    elif fault == 'feature': value.loc[0, m.BASE_FEATURES[0]] += 1
    elif fault == 'id': value.loc[0, 'issuance_id'] = 'altered'
    else: value.loc[0, 'issued_at_ns'] += 1
    with pytest.raises(ValueError):
        m.fit_models(value, expected_matched=full_training, saved_base_model=saved_model(), saved_base_model_sha256='b'*64)


@pytest.mark.parametrize('fault', ['width', 'content_inf', 'quality_nan', 'quality_inf', 'support_string', 'support_mismatch', 'cutoff_float', 'wrong_cutoff'])
def test_gap_matrix_rejects_invalid_layout_clocks_and_support(fault):
    value = frame();v = value.gap_values.iloc[0].copy()
    if fault == 'width': value['gap_values'] = [np.zeros(44)]*len(value)
    elif fault == 'content_inf': v[0] = np.inf;value.at[0, 'gap_values'] = v
    elif fault == 'quality_nan': v[12] = np.nan;value.at[0, 'gap_values'] = v
    elif fault == 'quality_inf': v[12] = np.inf;value.at[0, 'gap_values'] = v
    elif fault == 'support_string': value['gap_supported'] = 'False'
    elif fault == 'support_mismatch': v[39] = 1;value.at[0, 'gap_values'] = v
    elif fault == 'cutoff_float': value['gap_cutoff_ns'] = value.gap_cutoff_ns.astype(float)
    else: value['gap_cutoff_ns'] += 1
    with pytest.raises(ValueError): m.matrices(value, latency_seconds=2)


@pytest.mark.parametrize('lag', [True, False, 2., .5, 1, 3, '2', None])
def test_only_exact_frozen_lag_integers_accepted(lag):
    with pytest.raises(ValueError): m.matrices(frame(), latency_seconds=lag)


@pytest.mark.parametrize('fault', ['ids', 'cutoffs', 'cutoff_float', 'dtype', 'shape', 'nan', 'zero', 'hash'])
def test_closed_base_reference_binding_cannot_be_relaxed(fault):
    value = frame();reference = references(value)
    if fault == 'ids': reference = replace(reference, issuance_ids=reference.issuance_ids[::-1])
    elif fault == 'cutoffs': reference = replace(reference, cutoff_ns=tuple(v+1 for v in reference.cutoff_ns))
    elif fault == 'cutoff_float': reference = replace(reference, cutoff_ns=tuple(float(v) for v in reference.cutoff_ns))
    elif fault == 'dtype': reference = replace(reference, predictions=reference.predictions.astype(np.float32))
    elif fault == 'shape': reference = replace(reference, predictions=reference.predictions[:, None])
    elif fault == 'nan': reference.predictions[0] = np.nan
    elif fault == 'zero': reference.predictions[0] = 0
    else: reference = replace(reference, source_sha256='not a hash')
    with pytest.raises(ValueError): m.reference_values(value, reference)


@pytest.mark.parametrize('fault', ['base_features', 'config', 'base_hash', 'saved_width'])
def test_wrong_saved_base_is_rejected_without_fitting(full_training, monkeypatch, fault):
    saved = saved_model();source = 'b'*64
    if fault == 'base_features': saved['features'] = saved['features'][::-1]
    elif fault == 'config': saved['model'].get_params = lambda: dict(m.BASE_PARAMETERS, max_iter=300)
    elif fault == 'base_hash': source = 'wrong'
    else: saved['model'].n_features_in_ = 81
    monkeypatch.setattr(m.old_models.original, 'fit_model', lambda *a, **k: pytest.fail('Wrong base reached fit'))
    with pytest.raises(ValueError):
        m.fit_models(full_training, expected_matched=full_training, saved_base_model=saved, saved_base_model_sha256=source)


@pytest.mark.parametrize('fault', ['fit_count', 'new_identity', 'feature_order'])
def test_prediction_bundle_cannot_change_candidate_identity(full_training, recording, fault):
    bundle, _ = fitted(full_training)
    if fault == 'fit_count': bundle['fit_summary']['fits'] = 3
    elif fault == 'new_identity': bundle['models']['other'] = bundle['models'].pop('gap_hgb')
    else: bundle['models']['gap_hgb']['features'] = list(reversed(bundle['models']['gap_hgb']['features']))
    issued = target_free(frame(year=2023))
    with pytest.raises(ValueError):
        m.predict_models(bundle, issued, latency_seconds=2, expected_issuances=issued.copy(), saved_base_forecasts=references(issued))
