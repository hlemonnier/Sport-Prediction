"""Synthetic JSON/oracle checks; no historical data or fitted-asset reads."""
from copy import deepcopy
import importlib.util
import json

import numpy as np
import pytest

from research.experiments.boundary_20260908.catboost_frontier import independent as m


def model():
    return {
        'features_info': {'float_features': [
            {'feature_index': 0, 'flat_feature_index': 0, 'feature_id': 'a',
             'borders': [0.5], 'has_nans': True, 'nan_value_treatment': 'AsFalse'},
            {'feature_index': 1, 'flat_feature_index': 1, 'feature_id': 'b',
             'borders': [1.5], 'has_nans': False, 'nan_value_treatment': 'AsIs'},
        ]},
        'model_info': {'params': {'loss_function': {'type': 'MAE'},
            'tree_learner_options': {'grow_policy': 'SymmetricTree'},
            'data_processing_options': {'float_features_binarization': {'nan_mode': 'Min'}},
            'flat_params': {'loss_function': 'MAE', 'nan_mode': 'Min', 'grow_policy': 'SymmetricTree'}}},
        'oblivious_trees': [{'splits': [
            {'border': 0.5, 'float_feature_index': 0, 'split_index': 0, 'split_type': 'FloatFeature'},
            {'border': 1.5, 'float_feature_index': 1, 'split_index': 1, 'split_type': 'FloatFeature'},
        ], 'leaf_values': [10., 20., 30., 40.], 'leaf_weights': [1., 1., 1., 1.]}],
        'scale_and_bias': [1., [0.]],
    }


def test_leaf_bit_order_and_strict_threshold_and_nan():
    x = [[0., 0.], [1., 0.], [0., 2.], [1., 2.], [.5, 1.5], [np.nan, 2.], [1., np.nan]]
    np.testing.assert_array_equal(m.json_predict(model(), x), [10, 20, 30, 40, 10, 30, 20])


def test_split_order_is_leaf_bit_order_not_global_border_index():
    payload = model()
    payload['oblivious_trees'][0]['splits'].reverse()
    np.testing.assert_array_equal(m.json_predict(payload, [[1., 0.], [0., 2.]]), [30, 20])


def test_float32_conversion_precedes_strict_comparison():
    above = np.nextafter(np.float64(.5), np.inf)
    assert above > .5 and np.float32(above) == np.float32(.5)
    np.testing.assert_array_equal(m.json_predict(model(), [[above, 0.]]), [10])


def test_flat_input_index_mapping():
    payload = model()
    features = payload['features_info']['float_features']
    features[0]['flat_feature_index'], features[1]['flat_feature_index'] = 1, 0
    np.testing.assert_array_equal(m.json_predict(payload, [[0., 1.], [2., 0.]]), [20, 30])


def test_metadata_feature_list_order_does_not_define_input_order():
    payload = model()
    payload['features_info']['float_features'].reverse()
    np.testing.assert_array_equal(m.json_predict(payload, [[1., 0.], [0., 2.]]), [20, 30])


def test_multiple_trees_then_scale_and_bias():
    payload = model()
    second = deepcopy(payload['oblivious_trees'][0])
    second['leaf_values'] = [-1., -2., -3., -4.]
    payload['oblivious_trees'].append(second)
    payload['scale_and_bias'] = [-2., [3.]]
    np.testing.assert_array_equal(m.json_predict(payload, [[1., 0.], [0., 2.]]), [-33, -51])


def test_constant_tree_and_empty_prediction_batch():
    payload = model()
    payload['oblivious_trees'] = [{'splits': None, 'leaf_values': [4.], 'leaf_weights': [10.]}]
    np.testing.assert_array_equal(m.json_predict(payload, [[0., 0.], [1., 1.]]), [4, 4])
    assert m.json_predict(payload, np.empty((0, 2))).shape == (0,)


def test_tree_metadata_and_inputs_unchanged():
    payload = model()
    before = json.dumps(payload, sort_keys=True)
    x = np.array([[np.nan, 0.], [1., 2.]])
    xb = x.view(np.uint64).copy()
    m.json_predict(payload, x)
    assert json.dumps(payload, sort_keys=True) == before
    np.testing.assert_array_equal(x.view(np.uint64), xb)


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(trees=[]),
    lambda p: p['features_info'].update(categorical_features=[]),
    lambda p: p['features_info'].update(ctrs=[]),
    lambda p: p['features_info']['float_features'][0].update(nan_value_treatment='AsTrue'),
    lambda p: p['features_info']['float_features'][0].update(nan_value_treatment='AsIs'),
    lambda p: p['features_info']['float_features'][0].update(has_nans='true'),
    lambda p: p['features_info']['float_features'][0].update(borders=[.5, .5]),
    lambda p: p['features_info']['float_features'][0].update(borders=[np.inf]),
    lambda p: p['features_info']['float_features'][0].update(feature_index=True),
    lambda p: p['features_info']['float_features'][0].update(feature_index=1),
    lambda p: p['features_info']['float_features'][0].update(flat_feature_index=1),
    lambda p: p['features_info']['float_features'][0].update(flat_feature_index=3),
    lambda p: p['model_info']['params']['loss_function'].update(type='Logloss'),
    lambda p: p['model_info']['params']['tree_learner_options'].update(grow_policy='Lossguide'),
    lambda p: p['model_info']['params']['data_processing_options']['float_features_binarization'].update(nan_mode='Max'),
    lambda p: p['model_info']['params']['flat_params'].update(loss_function='MultiRMSE'),
    lambda p: p['model_info'].update(params='not-an-object'),
    lambda p: p['oblivious_trees'][0].update(leaf_values=[1., 2.]),
    lambda p: p['oblivious_trees'][0].update(leaf_values=[[1.], [2.], [3.], [4.]]),
    lambda p: p['oblivious_trees'][0].update(leaf_values=[1., 2., 3., np.nan]),
    lambda p: p['oblivious_trees'][0].update(leaf_weights=[1., 2., 3., -4.]),
    lambda p: p['oblivious_trees'][0].update(leaf_weights=[1.]),
    lambda p: p['oblivious_trees'][0].update(left={}),
    lambda p: p['oblivious_trees'][0]['splits'][0].update(split_type='OnlineCtr'),
    lambda p: p['oblivious_trees'][0]['splits'][0].update(split_type='OneHotFeature'),
    lambda p: p['oblivious_trees'][0]['splits'][0].update(border=.6),
    lambda p: p['oblivious_trees'][0]['splits'][0].update(float_feature_index=99),
    lambda p: p['oblivious_trees'][0]['splits'][0].update(split_index=1),
    lambda p: p['oblivious_trees'][0]['splits'][0].update(extra='ignored'),
    lambda p: p.update(scale_and_bias=[1., [0., 1.]]),
    lambda p: p.update(scale_and_bias=[True, [0.]]),
    lambda p: p.update(scale_and_bias=[1., [np.inf]]),
    lambda p: p.update(oblivious_trees=[]),
])
def test_unsupported_or_inconsistent_model_fails(mutation):
    payload = model()
    mutation(payload)
    with pytest.raises(ValueError):
        m.json_predict(payload, [[0., 0.]])


@pytest.mark.parametrize('x', [np.zeros(2), np.zeros((2, 3)), [[np.inf, 0.]], [['0', '1']],
    np.array([[True, False]]), [[1e100, 0.]], np.array([[1j, 0]]), np.array([[object(), 0]])])
def test_invalid_input_shape_or_type_fails(x):
    with pytest.raises(ValueError):
        m.json_predict(model(), x)


def test_nonfinite_tree_sum_or_scaled_prediction_fails():
    payload = model()
    payload['oblivious_trees'][0]['leaf_values'] = [1e308] * 4
    payload['scale_and_bias'] = [2., [0.]]
    with pytest.raises(ValueError, match='Nonfinite scaled'):
        m.json_predict(payload, [[0., 0.]])
    payload['scale_and_bias'] = [1., [0.]]
    payload['oblivious_trees'] *= 2
    with pytest.raises(ValueError, match='Nonfinite raw'):
        m.json_predict(payload, [[0., 0.]])


def test_event_balancing_differs_from_row_balancing_and_repeats_exactly():
    event = ['202301']*3 + ['202302']
    y, candidate, reference = np.zeros(4), np.array([1., 1., 1., 4.]), np.array([3., 3., 3., 2.])
    a = m.comparison(event, y, candidate, reference)
    b = m.comparison(event, y, candidate, reference)
    assert a == b
    assert a['candidate_mae'] == 2.5 and a['baseline_mae'] == 2.5
    assert a['candidate_row_mae'] == 1.75 and a['baseline_row_mae'] == 2.75
    assert a['delta'] == 0 and a['loo_max_delta'] == 2.
    assert a['event_ci95'] == [-2., 2.]
    assert a['per_event'][0]['event_key'] == 202301


def test_perfect_candidate_and_zero_reference_are_not_invalid_arithmetic():
    a = m.comparison([202301, 202302], [0., 0.], [0., 0.], [0., 0.])
    assert a['relative_reduction'] is None
    assert a['event_ci95'] == [0., 0.] and a['loo_max_delta'] == 0.


@pytest.mark.parametrize('events', [[202301], [], [202301, 202301], [202301., 202302.],
    ['202301', '2023-02'], ['202300', '202302'], ['0202301', '202302'], [True, False],
    [['202301'], ['202302']]])
def test_invalid_event_input(events):
    with pytest.raises(ValueError):
        m.comparison(events, np.zeros(len(events)), np.zeros(len(events)), np.zeros(len(events)))


@pytest.mark.parametrize('vector', [[0.], [0., np.nan], [0., np.inf], ['0', '1'], [[0.], [0.]], [True, False]])
def test_invalid_prediction_input(vector):
    with pytest.raises(ValueError):
        m.comparison([202301, 202302], [0., 0.], vector, [0., 0.])


def test_statistics_source_pin_is_enforced(tmp_path, monkeypatch):
    path = tmp_path / 'untrusted.py'
    path.write_text('raise RuntimeError("must not execute")')
    monkeypatch.setattr(m, 'STATISTICS_PATH', path)
    with pytest.raises(ValueError, match='source hash changed'):
        m.comparison([202301, 202302], [0., 0.], [0., 0.], [0., 0.])


@pytest.mark.skipif(importlib.util.find_spec('catboost') is None, reason='optional pinned CatBoost backend unavailable')
@pytest.mark.parametrize('boosting_type', ['Plain', 'Ordered'])
def test_tiny_actual_export_numeric_nan_and_scale_parity(tmp_path, boosting_type):
    # Five trees on synthetic data only, never a production/research fitted asset.
    import catboost
    assert catboost.__version__ == '1.2.10'
    rng = np.random.default_rng(181)
    x = rng.normal(size=(64, 4))
    x[::4, 1] = np.nan
    y = .3*x[:, 0] + np.nan_to_num(x[:, 1])*.2 - .1*x[:, 3]
    estimator = catboost.CatBoostRegressor(iterations=5, depth=3, loss_function='MAE',
        boosting_type=boosting_type, task_type='CPU', thread_count=1,
        nan_mode='Min', bootstrap_type='No', random_seed=12,
        allow_writing_files=False, verbose=False)
    estimator.fit(x, y, sample_weight=np.linspace(.5, 1.5, len(x)))
    estimator.set_scale_and_bias(1.2, -.37)
    path = tmp_path / 'tiny.json'
    estimator.save_model(str(path), format='json')
    payload = json.loads(path.read_text())
    test = np.vstack((x, [[np.nan]*4], np.zeros((1, 4))))
    for feature in payload['features_info']['float_features']:
        for border in feature['borders'][:2]:
            point = np.zeros((3, 4))
            point[:, feature['flat_feature_index']] = [border, np.nextafter(border, np.inf), np.nextafter(border, -np.inf)]
            test = np.vstack((test, point))
    np.testing.assert_allclose(m.json_predict(payload, test), estimator.predict(test), rtol=0, atol=2e-15)
