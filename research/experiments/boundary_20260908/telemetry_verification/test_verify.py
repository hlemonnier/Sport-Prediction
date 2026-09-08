"""Synthetic independent-verifier tests; no historical artifact execution."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.telemetry_verification import verify as v


def test_closed_selection_required_before_any_input_or_model_read(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, 'read_csv', lambda *a, **k: pytest.fail('historical read before selection guard'))
    with pytest.raises(FileNotFoundError, match='Selection must be closed'):
        v.verify(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_event_weights_and_independent_circular_block_geometry():
    events = np.array([202301]+[202302]*9)
    y = np.zeros(10)
    reference, candidate = np.array([10.]+[1.]*9), np.array([8.]+[1.]*9)
    got = v.independent_comparison(events, y, candidate, reference)
    assert got['baseline_mae'] == 5.5 and got['candidate_mae'] == 4.5
    assert got['delta'] == -1 and got['loo_max_delta'] == 0
    assert got['event_ci95'] == [-2., 0.]
    assert got['block3_ci95'] == [-1., -1.]
    assert got['events_improved'] == 1
    assert got['baseline_mae'] != np.abs(reference).mean()


def test_uniform_improvement_bootstrap_and_loo_have_exact_known_values():
    events = np.repeat(np.arange(22)+202301, 3)
    result = v.independent_comparison(events, np.zeros(66), np.full(66, .75), np.ones(66))
    assert result['relative_reduction'] == .25
    assert result['event_ci95'] == result['block3_ci95'] == [-.25, -.25]
    assert result['loo_max_delta'] == -.25 and result['events_improved'] == 22


def test_bootstrap_is_deterministic_and_invariant_to_row_order():
    rng = np.random.default_rng(92)
    events = np.repeat(np.arange(7)+202301, np.arange(1, 8))
    y, a, b = rng.normal(size=(3, len(events)))
    first = v.independent_comparison(events, y, a, b)
    order = np.arange(len(events))[::-1]
    again = v.independent_comparison(events[order], y[order], a[order], b[order])
    v.equal(again, first, name='reordered')


def test_perfect_reference_is_serializable_and_cannot_advance():
    r = v.independent_comparison(np.arange(22), np.ones(22), np.ones(22), np.ones(22))
    assert r['relative_reduction'] is None
    json.dumps(r, allow_nan=False)
    comparisons = {lag: {name: copy.deepcopy(r) for name in ('base_hgb', 'quality_hgb')} for lag in ('2', '0')}
    assert v.gates(comparisons)[1] is False


def good_comparisons():
    result = dict(relative_reduction=.02, event_ci95=[-.2, -.1], block3_ci95=[-.2, -.1], loo_max_delta=-.1)
    return {lag: {name: copy.deepcopy(result) for name in ('base_hgb', 'quality_hgb')} for lag in ('2', '0')}


@pytest.mark.parametrize('reference', ['base_hgb', 'quality_hgb'])
@pytest.mark.parametrize('field,bad', [('relative_reduction', .0099), ('event_ci95', [-.2, 0.]),
    ('block3_ci95', [-.2, 0.]), ('loo_max_delta', 0.)])
def test_primary_cannot_bypass_either_reference_or_any_gate(reference, field, bad):
    comparisons = good_comparisons()
    assert v.gates(comparisons)[1] is True
    comparisons['2'][reference][field] = bad
    assert v.gates(comparisons)[1] is False


@pytest.mark.parametrize('reference', ['base_hgb', 'quality_hgb'])
def test_zero_lag_sensitivity_must_be_positive_against_both(reference):
    comparisons = good_comparisons()
    comparisons['0'][reference]['relative_reduction'] = 0.
    assert v.gates(comparisons)[1] is False


def test_binding_scanner_checks_records_maps_and_raw_receipt_paths(tmp_path):
    p = tmp_path/'data'/'source';p.parent.mkdir();p.write_bytes(b'raw')
    h = v.sha(p);bindings = v.Bindings(tmp_path)
    bindings.scan({'record': {'path': 'data/source', 'sha256': h},
        'sources': {'data/source': h}, 'raw': {'wire_body_path': 'data/source',
            'wire_body_sha256': h, 'wire_body_bytes': 3}})
    assert len(bindings.checked) == 1
    with pytest.raises(AssertionError): bindings.check(p, h, 4)
    with pytest.raises(AssertionError): bindings.check(p, '0'*64)
    with pytest.raises(ValueError): bindings.check('../outside', h)


@pytest.mark.parametrize('text,expected', [('0.999999999', 999999999), ('1.000000001', 1000000001), ('1e-9', 1)])
def test_original_decimal_clock_is_exact(text, expected):
    assert v.exact_ns(text) == expected


def test_independent_original_identity_and_feature_check_rejects_missing_or_changed_rows():
    original = [dict(event_key=202301, driver_id='1', issued_after_lap_number=2,
                     issued_at_timestamp=20., forecast_naive_seconds=90.)]
    actual = [{**original[0], 'issued_at_ns': 20000000000, 'issuance_id': '202301/1/2/20000000000'}]
    v.assert_original_rows(original, actual, {('1', 2): 20000000000}, 202301)
    with pytest.raises(AssertionError): v.assert_original_rows(original, [], {('1', 2): 20000000000}, 202301)
    actual[0]['forecast_naive_seconds'] += .01
    with pytest.raises(AssertionError): v.assert_original_rows(original, actual, {('1', 2): 20000000000}, 202301)


def test_decimal_oracle_packet_weighting_and_quality_have_manual_values():
    first = {'0': 1000, '2': 0, '3': 1, '4': 0, '5': 0, '45': 0}
    second = {'0': 9000, '2': 100, '3': 8, '4': 100, '5': 100, '45': 12}
    result = v.oracle_window([(0, [first]), (10*10**9, [second]*9)], 15*10**9, 30)
    np.testing.assert_array_equal(result[:9], [50., 50., 5000., 50., .5, .5, .5, 4.5, .5])
    assert result[9:14] == [2, 10., 5., 15., 5.]
    assert result[14:26] == [1.]*12 and result[-1] == 1.


def test_oracle_joint_validity_is_not_product_of_marginal_validity():
    result = v.oracle_window([(0, [{'4': 0, '5': 104}, {'4': 104, '5': 0}])], 1, 30)
    assert result[6] == 0 and result[21] == .5 and result[23] == .5
    assert result[-1] == 0


@pytest.mark.parametrize('value', [float(np.finfo(float).max), float(np.nextafter(0., 1.))])
def test_decimal_oracle_extremes_do_not_depend_on_float_intermediate_overflow(value):
    result = v.oracle_window([(0, [{'2': value, '0': value}]*3)], 1, 30)
    assert result[0] == result[2] == value and result[1] == 0.
    mixed = v.oracle_window([(0, [{'2': value}]), (1, [{'2': 0}])], 2, 30)
    assert mixed[0] == value/2 and mixed[1] == value/2


def packet(ms, channels):
    raw = json.dumps({'Entries': [{'Cars': {'1': {'Channels': channels}}}]}).encode()
    codec = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    envelope = json.dumps(base64.b64encode(codec.compress(raw)+codec.flush()).decode()).encode()
    seconds, millis = divmod(ms, 1000)
    return f'00:00:{seconds:02}.{millis:03}'.encode()+envelope+b'\n'


def test_raw_sample_requires_strict_cutoff_and_exact_support(tmp_path):
    values = {'0': 5000, '2': 100, '3': 4, '4': 50, '5': 0, '45': 0}
    times = [1000, 6000, 11000, 16000, 21000, 26000]
    path = tmp_path/'source';path.write_bytes(b''.join(packet(t, values) for t in times)+packet(31000, {'2': 999}))
    expected = [x for window in [30, 90, 180] for x in v.oracle_window([(t*1000000, [values]) for t in times], 31*10**9, window)]
    row = {'driver_id': '1', 'telemetry_cutoff_ns': 31*10**9,
           'telemetry_supported': True, 'telemetry_values': expected}
    assert v.raw_sample(path, [row])['coordinates'] == 90
    row['telemetry_values'][0] += 1
    with pytest.raises(AssertionError): v.raw_sample(path, [row])


class Estimator:
    def __init__(self, value=None): self.value = value
    def predict(self, x):
        return np.full(len(x), self.value) if self.value is not None else x.iloc[:, 80].to_numpy()


def test_direct_replay_uses_same_model_and_exact_unsupported_base():
    names = [f'b{i}' for i in range(80)];extra = [f't{i}' for i in range(90)]
    frame = pd.DataFrame(np.zeros((2, 80)), columns=names)
    frame['forecast_naive_seconds'] = [90., 91.]
    frame['telemetry_values'] = [[9.]*90, [3.]*90]
    bundle = {'models': {name: {'model': Estimator(.25 if name == 'base_hgb' else None)} for name in v.NAMES}}
    result = v.replay(bundle, frame, np.array([True, False]), names, extra)
    np.testing.assert_array_equal(result['base_hgb'], [90.25, 91.25])
    np.testing.assert_array_equal(result['telemetry_hgb'], [93., 91.25])
    np.testing.assert_array_equal(result['quality_hgb'], [90., 91.25])


def test_verifier_does_not_import_experiment_metric_prediction_or_fit_wrappers():
    source = Path(v.__file__).read_text()
    for forbidden in ['evaluate.comparison(', 'original.diagnostics(', 'models.predict_models(', 'models.fit_models(', '.fit(']:
        assert forbidden not in source
