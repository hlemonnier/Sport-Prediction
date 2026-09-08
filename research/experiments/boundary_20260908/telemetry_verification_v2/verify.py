"""Independent post-selection evidence replay; no fitting or candidate selection.

This directory is outside the experiment's frozen source inventory. Run only
after selection_lock.json exists. Suggested commit: research(f1-live): independently
verify telemetry forecasts, complete populations and frozen selection gates.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import pickle

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT/'artifacts/research/boundary_20260908/telemetry/execution'
KEYS = ('event_key', 'driver_id', 'issued_after_lap_number', 'issued_at_timestamp')
NAMES = ('base_hgb', 'telemetry_hgb', 'quality_hgb')
CONTENT = tuple(start+i for start in (0, 30, 60) for i in range(9))
CHANNELS = ('0', '2', '3', '4', '5', '45')
SAMPLE_EVENTS = (202201, 202218, 202301, 202313)
SEED = 20260907
BASE_MAE = 0.512080723333514


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    def invalid(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def normalized(value):
    if isinstance(value, dict): return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)): return [normalized(x) for x in value]
    if isinstance(value, np.generic): return normalized(value.item())
    if isinstance(value, float) and math.isnan(value): return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Bindings:
    def __init__(self, root=ROOT):
        self.root, self.checked = Path(root), {}

    def check(self, path, expected, size=None):
        path = (self.root/Path(path)).resolve()
        path.relative_to(self.root.resolve())
        if path not in self.checked:
            self.checked[path] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        observed = self.checked[path]
        if observed['sha256'] != expected or size is not None and observed['bytes'] != size:
            raise AssertionError('Hash/size mismatch: '+str(path))
        return path

    def record(self, item):
        return self.check(item['path'], item['sha256'], item.get('bytes'))

    def scan(self, value):
        if isinstance(value, list):
            for item in value: self.scan(item)
        elif isinstance(value, dict):
            if isinstance(value.get('path'), str) and 'sha256' in value:
                self.record(value)
            for key, item in value.items():
                if key.endswith('_path') and item is not None and key[:-5]+'_sha256' in value:
                    self.check(item, value[key[:-5]+'_sha256'], value.get(key[:-5]+'_bytes'))
                if isinstance(item, str) and len(item) == 64 and '/' in key:
                    self.check(key, item)
                elif isinstance(item, dict) and '/' in key and 'sha256' in item and 'path' not in item:
                    self.check(key, item['sha256'], item.get('bytes'))
                self.scan(item)

    def load(self, path):
        value = read(path)
        self.scan(value)
        return value


def equal(actual, expected, *, name, tolerance=1e-12):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), name
        for key, value in expected.items():
            assert key in actual, f'{name}.{key} missing'
            equal(actual[key], value, name=f'{name}.{key}', tolerance=tolerance)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), name
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, name=f'{name}[{i}]', tolerance=tolerance)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected and (not isinstance(expected, bool) or type(actual) is bool), name
    elif isinstance(expected, (int, np.integer)):
        assert actual == expected, name
    else:
        assert math.isfinite(float(actual)) and abs(float(actual)-float(expected)) <= tolerance, name


def independent_comparison(events, y, candidate, reference):
    events, y = np.asarray(events), np.asarray(y, float)
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    assert y.shape == candidate.shape == reference.shape == events.shape
    assert np.isfinite(y).all() and np.isfinite(candidate).all() and np.isfinite(reference).all()
    keys = np.unique(events)
    base = np.array([np.mean(np.abs(reference[events == key]-y[events == key])) for key in keys])
    loss = np.array([np.mean(np.abs(candidate[events == key]-y[events == key])) for key in keys])
    differences = loss-base
    rng = np.random.default_rng(SEED)
    event_indices = rng.integers(0, len(keys), size=(20000, len(keys)))
    event_means = np.mean(differences[event_indices], axis=1)
    starts = rng.integers(0, len(keys), size=(20000, math.ceil(len(keys)/3)))
    indices = np.concatenate([((starts+step) % len(keys))[:, :, None] for step in range(3)], axis=2)
    block_means = differences[indices.reshape(20000, -1)[:, :len(keys)]].mean(axis=1)
    loo = [np.delete(differences, i).mean() for i in range(len(keys))] if len(keys) > 1 else []
    return {
        'events': len(keys), 'rows': len(y), 'baseline_mae': float(base.mean()),
        'candidate_mae': float(loss.mean()),
        'relative_reduction': float(1-loss.mean()/base.mean()) if base.mean() else None,
        'delta': float(differences.mean()),
        'event_ci95': np.percentile(event_means, [2.5, 97.5]).tolist(),
        'block3_ci95': np.percentile(block_means, [2.5, 97.5]).tolist(),
        'events_improved': int((differences < 0).sum()),
        'loo_max_delta': float(max(loo)) if loo else None,
        'per_event': [{'event_key': int(key), 'baseline_mae': float(a), 'candidate_mae': float(b)}
                      for key, a, b in zip(keys, base, loss)],
    }


def gates(comparisons):
    primary, sensitivity = {}, {}
    for name in ('base_hgb', 'quality_hgb'):
        v, z = comparisons['2'][name], comparisons['0'][name]
        primary[name] = {
            'minimum_relative_gain': v['relative_reduction'] is not None and v['relative_reduction'] >= .01,
            'event_ci_upper_negative': v['event_ci95'][1] < 0,
            'block3_ci_upper_negative': v['block3_ci95'][1] < 0,
            'all_leave_one_event_out_negative': v['loo_max_delta'] is not None and v['loo_max_delta'] < 0,
        }
        sensitivity[name] = {'positive_relative_gain': z['relative_reduction'] is not None and z['relative_reduction'] > 0}
    checks = {'2': primary, '0': sensitivity}
    return checks, all(flag for lag in checks.values() for reference in lag.values() for flag in reference.values())


def exact_ns(text):
    with localcontext() as ctx:
        ctx.prec = max(60, len(text)+12)
        value = Decimal(text)*Decimal(1_000_000_000)
        assert value.is_finite() and value >= 0 and value == value.to_integral_value()
        return int(value)


def numeric_driver(text):
    value = Decimal(str(text))
    assert value.is_finite() and value == value.to_integral_value()
    return str(int(value))


def row_key(row):
    return tuple(row[name] for name in KEYS)


def assert_original_rows(original, actual, clock_index, event):
    assert len(original) == len(actual)
    for source, saved in zip(original, actual):
        assert all(normalized(source[key]) == saved[key] for key in source), 'original issuance value changed'
        ns = clock_index[(str(source['driver_id']), int(source['issued_after_lap_number']))]
        identity = f"{event}/{source['driver_id']}/{int(source['issued_after_lap_number'])}/{ns}"
        assert saved['issued_at_ns'] == ns and saved['issuance_id'] == identity


def replay(bundle, frame, support, base_features, telemetry_features):
    """Direct estimator calls, with no experiment prediction wrapper."""
    base = frame[base_features].copy()
    anchor = frame.forecast_naive_seconds.to_numpy(float)
    extra = np.array(frame.telemetry_values.tolist(), float)
    columns = ['telemetry__'+name for name in telemetry_features]
    augmented = pd.concat([base, pd.DataFrame(extra, index=frame.index, columns=columns)], axis=1)
    control = augmented.copy()
    control.iloc[:, [80+i for i in CONTENT]] = 0.
    output = {}
    with threadpool_limits(limits=1):
        output['base_hgb'] = anchor + np.clip(bundle['models']['base_hgb']['model'].predict(base), -3., 3.)
        for name, x in [('telemetry_hgb', augmented), ('quality_hgb', control)]:
            prediction = output['base_hgb'].copy()
            if support.any():
                prediction[support] = anchor[support] + np.clip(bundle['models'][name]['model'].predict(x.loc[support]), -3., 3.)
            output[name] = prediction
    return output


def valid(channel, value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False
    try: finite = math.isfinite(value)
    except OverflowError: return False
    if not finite or value < 0: return False
    if channel == '4': return value <= 100
    if channel == '5': return value in (0, 100)
    if channel == '3': return value == int(value) and value <= 8
    if channel == '45': return value == int(value)
    return True


def exact_mean(values):
    if not values: return 0.
    return float(sum(Fraction.from_float(float(x)) for x in values)/len(values))


def oracle_window(packets, cutoff, seconds):
    """Independent direct packet weighting, with exact rational means."""
    kept = [(time, group) for time, group in packets if cutoff-seconds*10**9 <= time < cutoff]
    if not kept: return [0.]*9+[0., 0., float(seconds), float(seconds), 0.]+[0.]*16
    content = [[] for _ in range(9)]; quality = []; states = []
    for _, entries in kept:
        values = {c: [row[c] for row in entries if c in row and valid(c, row[c])] for c in CHANNELS}
        pairs = [(row['4'], row['5']) for row in entries if valid('4', row.get('4')) and valid('5', row.get('5'))]
        observations = [values['2'], [], values['0'], values['4'], [float(v >= 95) for v in values['4']],
            [float(v == 100) for v in values['5']], [float(a <= 5 and b == 0) for a, b in pairs],
            values['3'], [float(v in (10, 12, 14)) for v in values['45']]]
        for i, value in enumerate(observations):
            if value: content[i].append(exact_mean(value))
        q = []
        for c in CHANNELS:
            q += [sum(c in row for row in entries)/len(entries), len(values[c])/len(entries)]
        q += [sum(row.get(c) == 104 for row in entries)/len(entries) for c in ('4', '5')]
        quality.append(q+[len(pairs)/len(entries)])
        states.append(tuple((c in entries[-1], json.dumps(entries[-1].get(c), sort_keys=True)) for c in CHANNELS))
    averages = [exact_mean(value) for value in content]
    if content[0]:
        speed = [Fraction.from_float(x) for x in content[0]]
        scale = max(speed)
        if scale:
            unit = [x/scale for x in speed]
            center = sum(unit)/len(unit)
            variance = sum((x-center)**2 for x in unit)/len(unit)
            averages[1] = float(scale)*math.sqrt(float(variance))
    times = [time for time, _ in kept]
    edges = [cutoff-seconds*10**9, *times, cutoff]
    qmean = [exact_mean([value[i] for value in quality]) for i in range(15)]
    final = [len(kept), (times[-1]-times[0])/1e9, (cutoff-times[-1])/1e9,
             max(b-a for a, b in zip(edges, edges[1:]))/1e9, exact_mean([len(group) for _, group in kept])]
    final += qmean[:14]+[exact_mean([float(a == b) for a, b in zip(states, states[1:])])]+qmean[14:]
    return averages+final


def raw_sample(path, sample_rows):
    from research.experiments.boundary_20260908.telemetry_pilot import packets as parser
    requests = sorted(sample_rows, key=lambda item: (item['telemetry_cutoff_ns'], item['driver_id']))
    drivers = {row['driver_id'] for row in requests}
    history = {driver: deque() for driver in drivers}
    iterator = iter(parser.iter_packets(path, before_ms=max(0, (max(r['telemetry_cutoff_ns'] for r in requests)+999999)//1000000)))
    pending = next(iterator, None)
    max_error = 0.
    for row in requests:
        cutoff = row['telemetry_cutoff_ns']
        while pending is not None and pending['available_ms']*1000000 < cutoff:
            instant = pending['available_ms']*1000000
            groups = {driver: [] for driver in drivers}
            for source in pending['payload']['Entries']:
                for driver, car in (source.get('Cars') or {}).items():
                    if driver in groups: groups[driver].append(car.get('Channels') or {})
            for driver in drivers:
                if groups[driver]: history[driver].append((instant, groups[driver]))
                while history[driver] and history[driver][0][0] < instant-180*10**9: history[driver].popleft()
            pending = next(iterator, None)
        own = list(history[row['driver_id']])
        values = np.array([value for window in (30, 90, 180) for value in oracle_window(own, cutoff, window)])
        saved = np.array(row['telemetry_values'], float)
        np.testing.assert_allclose(saved, values, rtol=1e-12, atol=1e-10)
        recent = [t for t, _ in own if cutoff-30*10**9 <= t < cutoff]
        supported = len(recent) >= 6 and recent[-1]-recent[0] >= 24*10**9 and cutoff-recent[-1] <= 5*10**9
        assert row['telemetry_supported'] is supported
        max_error = max(max_error, float(np.max(np.abs(saved-values))))
    if hasattr(iterator, 'close'): iterator.close()
    return {'snapshots': len(requests), 'coordinates': 90*len(requests), 'max_absolute_difference': max_error}


def coverage(frame, supported):
    matched = frame.outcome_status.eq('matched').to_numpy()
    result = {'original_full_issuance_inventory_verified': True, 'original_matched_inventory_verified': True,
        'rows_all': len(frame), 'rows_matched': int(matched.sum()), 'rows_unmatched_retained': int((~matched).sum()),
        'supported_rows_all': int(supported.sum()), 'fallback_rows_all': int((~supported).sum()),
        'supported_rows_matched': int((supported & matched).sum()), 'fallback_rows_matched': int((~supported & matched).sum()),
        'per_event': []}
    for event in sorted(frame.event_key.unique()):
        mask = frame.event_key.eq(event).to_numpy()
        result['per_event'].append({'event_key': int(event), 'rows_all': int(mask.sum()),
            'rows_matched': int((mask & matched).sum()), 'rows_unmatched_retained': int((mask & ~matched).sum()),
            'supported_rows_all': int((mask & supported).sum()), 'fallback_rows_all': int((mask & ~supported).sum()),
            'supported_rows_matched': int((mask & supported & matched).sum())})
    return result


def verify(execution_dir):
    out = Path(execution_dir).resolve()
    # This guard precedes ANY historical data, model or score read.
    if not (out/'selection_lock.json').is_file():
        raise FileNotFoundError('Selection must be closed before independent verification')
    from research.experiments.frontier_20260907 import live_frontier as original
    checked = Bindings()
    selection_lock = checked.load(out/'selection_lock.json')
    selection = checked.load(checked.record(selection_lock['selection']))
    design = checked.load(checked.record(selection['design_lock']))
    data_lock = checked.load(checked.record(selection['data_lock']))
    fit_lock = checked.load(checked.record(selection['fit_lock']))
    issuance = checked.load(checked.record(selection['issuance_lock']))
    spec = checked.load(checked.record(design['specification']))
    review = checked.load(checked.record(design['independent_review']))
    pretests = checked.load(checked.record(design['pre_fit_tests']))
    acquired = checked.load(checked.record(design['acquisition_manifest']))
    assert selection_lock['design_lock'] == data_lock['design_lock'] == selection['design_lock']
    assert fit_lock['data_lock'] == issuance['data_lock'] == selection['data_lock']
    assert issuance['fit_lock'] == selection['fit_lock']
    assert spec['selection_gate']['candidate'] == 'telemetry_hgb'
    assert spec['selection_gate']['references'] == ['base_hgb', 'quality_hgb']
    assert spec['selection_gate']['minimum_relative_event_mae_reduction_each_reference'] == .01
    for flag in ('all_leave_one_event_out_negative_each', 'event_and_block3_upper_ci_negative_each', 'zero_second_sensitivity_positive_each'):
        assert spec['selection_gate'][flag] is True
    assert spec['uncertainty']['seed'] == SEED and spec['uncertainty']['resamples_each'] == 20000
    assert spec['uncertainty']['circular_event_block_length'] == 3
    assert spec['features']['primary_lag_seconds'] == 2 and spec['features']['sensitivity_lag_seconds'] == 0
    assert spec['features']['sensitivity_refit'] is False
    assert review['approved_for_execution_lock'] is True and review['source_files'] == design['sources']
    assert pretests['exit_code'] == 0 and pretests['source_files'] == design['sources']
    assert design['candidate_fits_before_lock'] == 0 and data_lock['external_labels_attached'] is False
    assert data_lock['model_fits'] == 0 and fit_lock['external_selection_labels_attached'] is False
    assert issuance['selection_labels_attached'] is False and issuance['selection_matched_cache_read'] is False
    clocks = [design['closed_at_utc'], data_lock['closed_at_utc'], fit_lock['closed_at_utc'], issuance['closed_at_utc']]
    assert clocks == sorted(clocks)
    closures, validation, labels, references = {}, {}, {}, {}
    for year in (2022, 2023):
        info = data_lock['years'][str(year)]
        closures[year] = checked.load(checked.record(info['feature_closure']))
        validation[year] = checked.load(checked.record(info['input_validation']))
        assert validation[year]['status'] == 'PASS'
        assert validation[year]['feature_closure_sha256'] == info['feature_closure']['sha256']
        references.update({row['event_key']: rows(checked.record(row)) for row in info['original_references']})
        label_record = fit_lock['training_labels'] if year == 2022 else selection['selection_labels']
        closed_labels = checked.load(checked.record(label_record))
        assert closed_labels['feature_closure'] == info['feature_closure']
        if year == 2023: assert issuance['closed_at_utc'] <= closed_labels['closed_at_utc']
        labels.update({row['event_key']: rows(checked.record(row)) for row in closed_labels['events']})
    raw_streams = {row['event_key']: row for row in acquired['streams']}
    feature_entries = {row['event_key']: row for value in closures.values() for row in value['events']}
    expected_events = list(range(202201, 202223))+list(range(202301, 202323))
    assert list(feature_entries) == expected_events
    assert set(labels) == set(references) == set(raw_streams) == set(expected_events)
    base_features, telemetry_features = design['base_features'], design['telemetry_features']
    assert len(base_features) == 80 and len(telemetry_features) == 90
    all_frames = {2022: {0: [], 2: []}, 2023: {0: [], 2: []}}
    original_matched, sampled = [], []
    total_issued = total_matched = 0
    for source in design['original_input_manifest']:
        event = source['event_key']; year = event//100
        raw_path = checked.record(source)
        raw = pd.read_csv(raw_path)
        with raw_path.open(encoding='utf-8-sig', newline='') as stream:
            lexical = list(csv.DictReader(stream))
        exact = {(numeric_driver(row['DriverNumber']), int(Decimal(row['LapNumber']))): exact_ns(row['Time']) for row in lexical}
        assert len(exact) == len(raw) == source['raw_rows']
        original_issued, matched = original.enrich(raw, event)
        assert list(original.base.FEATURES)+[c for c in original_issued if c.startswith('x_')] == base_features
        canonical = original_issued.to_dict('records')
        assert len(canonical) == source['issuances'] and len(matched) == source['matched_rows']
        assert_original_rows(canonical, references[event], exact, event)
        targets = {row_key(row): row for row in matched.to_dict('records')}
        label_rows = labels[event]
        assert len(label_rows) == len(canonical)
        for row, label, saved in zip(canonical, label_rows, references[event]):
            assert row_key(row) == row_key(label) and label['issuance_id'] == saved['issuance_id']
            assert label['issued_at_ns'] == saved['issued_at_ns']
            target = targets.get(row_key(row))
            if target is None:
                assert label['outcome_status'] == 'unmatched'
                assert all(label[k] is None for k in ('target_id', 'target_at_ns', 'target_lap_number', 'target_timestamp', 'lap_time_seconds', 'target_same_stint'))
            else:
                assert label['outcome_status'] == 'matched'
                for key in ('target_lap_number', 'target_timestamp', 'lap_time_seconds', 'target_same_stint'):
                    assert label[key] == target[key], 'Original target value changed'
                ns = exact[(row['driver_id'], int(target['target_lap_number']))]
                assert label['target_at_ns'] == ns > label['issued_at_ns']
                assert label['target_id'] == f"{event}/{row['driver_id']}/{int(target['target_lap_number'])}"
        event_samples = []
        for lag in (0, 2):
            feature_rows = rows(checked.record(feature_entries[event]['ledgers'][str(lag)]))
            assert_original_rows(canonical, feature_rows, exact, event)
            for row in feature_rows:
                assert 'outcome_status' not in row and 'lap_time_seconds' not in row and 'target_id' not in row
                assert type(row['telemetry_supported']) is bool
                assert row['telemetry_cutoff_ns'] == row['issued_at_ns']-lag*10**9
                assert row['telemetry_lag_seconds'] == lag
                assert row['year'] == year
            if event in SAMPLE_EVENTS:
                event_samples.extend(feature_rows[i] for i in (0, len(feature_rows)//2, len(feature_rows)-1))
            all_frames[year][lag].extend(feature_rows)
        if event in SAMPLE_EVENTS:
            stream = raw_streams[event]
            assert stream['status'] != 'unavailable', 'Fixed raw-sample event is unavailable; explicit verifier limit required'
            value = raw_sample(checked.check(stream['decoded_body_path'], stream['decoded_body_sha256']), event_samples)
            sampled.append({'event_key': event, **value,
                            'issuance_ids': [r['issuance_id'] for r in event_samples],
                            'cutoffs_ns': [r['telemetry_cutoff_ns'] for r in event_samples]})
        matched['year'] = year
        original_matched.append(matched)
        total_issued += len(canonical); total_matched += len(matched)
        print(json.dumps({'stage': 'independent_original_event', 'event_key': event,
                          'issuances': len(canonical), 'matched': len(matched)}), flush=True)
    assert total_issued == spec['discovery']['expected_original_issuances'] == 39220
    assert total_matched == spec['discovery']['matched_rows'] == 38370
    canonical_matched = pd.concat(original_matched, ignore_index=True)
    cache_path = next(ROOT/r['path'] for r in spec['original_frontier_bindings'] if r['path'].endswith('/discovery_data.pkl'))
    cached = pd.read_pickle(cache_path)
    for col in [*KEYS, *base_features, 'forecast_naive_seconds', 'target_lap_number', 'target_timestamp', 'lap_time_seconds', 'target_same_stint']:
        assert normalized(cached[col].tolist()) == normalized(canonical_matched[col].tolist()), 'Original cache parity: '+col
    with checked.record(fit_lock['models']).open('rb') as stream:
        bundle = pickle.load(stream)
    assert set(bundle['models']) == set(NAMES) and bundle['base_features'] == base_features
    assert bundle['telemetry_features'] == telemetry_features
    training = canonical_matched.loc[canonical_matched.year.eq(2022)]
    train_keys = {row_key(row) for row in training.to_dict('records')}
    training_ids = [r['issuance_id'] for r in all_frames[2022][2] if row_key(r) in train_keys]
    assert len(training_ids) == 18363
    fit_summary = bundle['fit_summary']
    assert fit_summary['fit_year'] == 2022 and fit_summary['fit_latency_seconds'] == 2
    assert fit_summary['rows_per_fit'] == 18363 and fit_summary['fits'] == 3 and fit_summary['augmented_fits'] == 2
    assert fit_summary['issuance_ids_sha256'] == hashlib.sha256(json.dumps(training_ids, separators=(',', ':')).encode()).hexdigest()
    assert fit_summary == fit_lock['fit_summary']
    counts = training.event_key.value_counts()
    weights = len(training)/(len(counts)*training.event_key.map(counts).to_numpy())
    weights /= weights.sum()
    residual = np.clip(training.lap_time_seconds.to_numpy()-training.forecast_naive_seconds.to_numpy(), -5, 5)
    for name in NAMES:
        saved = bundle['models'][name]; estimator = saved['model']
        names = base_features if name == 'base_hgb' else base_features+['telemetry__'+x for x in telemetry_features]
        assert saved['features'] == names and estimator.n_features_in_ == len(names) and estimator.n_iter_ == 150
        expected = dict(loss='absolute_error', learning_rate=.06, max_iter=150, max_leaf_nodes=15,
            min_samples_leaf=80, l2_regularization=10, early_stopping=False, random_state=20260907)
        equal(estimator.get_params(), expected, name=name+'.parameters')
        initial = float(estimator._baseline_prediction.ravel()[0])
        assert weights[residual < initial].sum() <= .5+1e-12
        assert weights[residual <= initial].sum() >= .5-1e-12
    by_label = {row['issuance_id']: row for group in labels.values() for row in group}
    replay_errors, predictions, combined, metric, comparisons, cov = {}, {}, {}, {}, {}, {}
    for lag in (2, 0):
        raw_rows = all_frames[2023][lag]
        forecast = rows(checked.record(issuance['forecasts'][str(lag)]))
        assert len(raw_rows) == len(forecast)
        for row, predicted in zip(raw_rows, forecast):
            assert predicted['feature_row_sha256'] == digest(row)
            assert predicted['issuance_id'] == row['issuance_id'] and predicted['lag_seconds'] == lag
            assert predicted['telemetry_supported'] is row['telemetry_supported']
            assert not any(k in predicted for k in ('outcome_status', 'target_id', 'lap_time_seconds'))
        frame = pd.DataFrame(raw_rows)
        supported = np.array([r['telemetry_supported'] for r in raw_rows], bool)
        actual = replay(bundle, frame, supported, base_features, telemetry_features)
        recorded = {name: np.array([row['predictions'][name] for row in forecast]) for name in NAMES}
        replay_errors[str(lag)] = {}
        for name in NAMES:
            assert np.isfinite(recorded[name]).all() and (recorded[name] > 0).all()
            np.testing.assert_array_equal(actual[name].view(np.uint64), recorded[name].view(np.uint64))
            replay_errors[str(lag)][name] = float(np.max(np.abs(actual[name]-recorded[name])))
            if name != 'base_hgb':
                np.testing.assert_array_equal(recorded[name][~supported].view(np.uint64), recorded['base_hgb'][~supported].view(np.uint64))
        frame['outcome_status'] = [by_label[r['issuance_id']]['outcome_status'] for r in raw_rows]
        frame['lap_time_seconds'] = [by_label[r['issuance_id']]['lap_time_seconds'] for r in raw_rows]
        matched_mask = frame.outcome_status.eq('matched').to_numpy()
        assert matched_mask.sum() == 20007
        events, y = frame.loc[matched_mask, 'event_key'].to_numpy(), frame.loc[matched_mask, 'lap_time_seconds'].to_numpy(float)
        assert np.unique(events).tolist() == list(range(202301, 202323))
        metric[str(lag)] = {}
        for name in NAMES:
            error = np.abs(recorded[name][matched_mask]-y)
            event_loss = [error[events == key].mean() for key in np.unique(events)]
            metric[str(lag)][name] = {'rows': len(y), 'events': 22,
                'event_mae_seconds': float(np.mean(event_loss)), 'row_mae_seconds': float(error.mean())}
        comparisons[str(lag)] = {name: independent_comparison(events, y, recorded['telemetry_hgb'][matched_mask], recorded[name][matched_mask])
                                for name in ('base_hgb', 'quality_hgb')}
        predictions[lag], combined[lag], cov[str(lag)] = recorded, frame, coverage(frame, supported)
    assert combined[0].issuance_id.tolist() == combined[2].issuance_id.tolist()
    np.testing.assert_array_equal(predictions[0]['base_hgb'].view(np.uint64), predictions[2]['base_hgb'].view(np.uint64))
    assert abs(metric['2']['base_hgb']['event_mae_seconds']-BASE_MAE) <= 1e-12
    checks, advances = gates(comparisons)
    summary = selection['summary']
    for key, computed in [('metrics', metric), ('comparisons', comparisons), ('coverage', cov), ('gate_checks', checks)]:
        equal(summary[key], computed, name=key)
    assert summary['advances_to_later_evaluation'] is advances
    assert selection_lock['advances_to_later_evaluation'] is advances and selection_lock['promotion'] is False
    assert selection_lock['selected_candidate'] == 'telemetry_hgb' and summary['sensitivity_refitted'] is False
    for path, value in checked.checked.items():
        assert sha(path) == value['sha256'], 'Bound source changed during verification'
    return {'status': 'passed', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'execution_directory': str(out), 'selection_sha256': sha(out/'selection.json'),
        'bindings': {str(path.relative_to(ROOT)): value for path, value in sorted(checked.checked.items())},
        'original_issuances_per_lag': total_issued, 'original_matched_targets': total_matched,
        'unmatched_issuances_retained': total_issued-total_matched,
        'selection_issuances_per_lag': len(combined[2]), 'selection_matched_targets': 20007,
        'replay_max_absolute_errors': replay_errors, 'metrics': metric, 'comparisons': comparisons,
        'coverage': cov, 'gate_checks': checks, 'advances_to_later_evaluation': advances,
        'raw_sample_protocol': {'event_keys': list(SAMPLE_EVENTS), 'positions': ['first', 'floor(n/2)', 'last'],
            'lags_seconds': [0, 2], 'selection_basis': 'Original issuance position, independent of outcomes or support.',
            'rtol': 1e-12, 'atol': 1e-10, 'results': sampled},
        'historical_fits_performed': 0,
        'independence': 'Direct saved-estimator replay; separate event loss/bootstrap/LOO/gate code; frozen original encoder for canonical population, frozen raw decoder plus separate rational packet-weighted measurement oracle.',
        'limits': ['Archive timestamps are availability proxies, not certified live receipt.',
            'Full44-stream integrity is checked through recorded validation closures and current hashes; independent raw measurement reconstruction uses the declared24-snapshot sample.',
            'This verifier does not establish prospective performance or correct historical research selection.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not (args.execution/'selection_lock.json').exists():
        raise FileNotFoundError('Selection must be closed before independent verification')
    output = args.output or args.execution/'verification'/'result.json'
    if output.exists(): raise FileExistsError('Verification evidence is immutable')
    sources = {str(path.relative_to(ROOT)): sha(path) for path in Path(__file__).parent.glob('*.py')}
    try:
        value = verify(args.execution)
    except Exception as exc:
        value = {'status': 'failed', 'failure': type(exc).__name__+': '+str(exc),
                 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'historical_fits_performed': 0}
        raise
    finally:
        if 'value' in locals():
            value['verifier_sources'] = sources
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x') as stream: json.dump(normalized(value), stream, indent=2, sort_keys=True, allow_nan=False);stream.write('\n')
    print(json.dumps({'verification': str(output), 'sha256': sha(output), 'status': value['status']}), flush=True)


if __name__ == '__main__': main()
