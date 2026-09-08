"""Independent closed-selection CPC replay; never fits or chooses a checkpoint.

Suggested commit: research(f1-live): independently verify controlled CPC evidence
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.experiments.boundary_20260908.telemetry_verification_v2 import verify as old
from .numerics import representation

ROOT = old.ROOT
HERE = Path(__file__).resolve().parent
PROTOCOL_SHA = 'eb9fb0c7ffe8fcc1f543763fbe18ea6e1dc46bfff745d5b7887b4376adb0b813'
CONTROLS = ('ordered', 'permuted', 'random')
OLD_NAMES = ('base_hgb', 'telemetry_hgb', 'quality_hgb')
NEW_NAMES = tuple(name+'_hgb' for name in CONTROLS)
NAMES = OLD_NAMES + NEW_NAMES
REFERENCES = OLD_NAMES + ('permuted_hgb', 'random_hgb')
TRAIN_EVENTS = tuple(range(202201, 202223))
TEST_EVENTS = tuple(range(202301, 202323))
TARGETS = ('outcome_status', 'target_id', 'target_at_ns', 'target_lap_number',
           'target_timestamp', 'lap_time_seconds', 'target_same_stint')
NPZ_FIELDS = {'issuance_ids', 'cutoff_ns', 'empty_context', 'support',
              'original_feature_row_sha256', 'context_sha256', *CONTROLS}


def bits(actual, expected, name):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), name


def state_digest(parameters):
    result = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        value = np.ascontiguousarray(value)
        result.update(name.encode()); result.update(str(value.shape).encode())
        result.update(value.dtype.str.encode()); result.update(value.tobytes())
    return result.hexdigest()


def binary(item, bindings, dtype, shape=None):
    path = bindings.record(item)
    assert item['dtype'] == np.dtype(dtype).str
    shape = tuple(item['shape']) if shape is None else tuple(shape)
    assert tuple(item['shape']) == shape and np.prod(shape)*np.dtype(dtype).itemsize == item['bytes']
    if not np.prod(shape): return np.empty(shape, dtype=dtype)
    return np.memmap(path, mode='r', dtype=dtype, shape=shape)


def eligibility(times):
    """Independent timestamp-only enumeration; physical endpoint is inclusive."""
    times = np.asarray(times)
    assert times.dtype == np.dtype('<i8') and times.ndim == 1
    assert np.all(times >= 0) and np.all(times[1:] >= times[:-1])
    assert not len(times) or times[-1] <= np.iinfo(np.int64).max-180_000_000_001
    ids = np.arange(max(0, len(times)-32))
    t = times[ids]
    lo = np.searchsorted(times, t-29_999_999_999, side='left')
    support = (ids-lo+1 >= 6) & (t-times[lo] >= 24_000_000_000)
    neg = (np.searchsorted(times, t-180_000_000_000, side='right')
           +len(times)-np.searchsorted(times, t+180_000_000_000, side='left'))
    for k in (4, 16, 32): neg -= times[ids+k]-t >= 180_000_000_000
    return ids[support & (neg >= 31)]


def negative_pool(times, endpoint):
    # Boolean geometry intentionally differs from the producer's concatenated
    # searchsorted ranges. It preserves sorted physical IDs for PCG64 replay.
    ids = np.flatnonzero(np.abs(times-times[endpoint]) >= 180_000_000_000)
    return ids[~np.isin(ids, endpoint+np.array([4, 16, 32]))]


def verify_schedule(corpus, schedule, bindings):
    assert corpus['status'] == schedule['status'] == 'closed'
    assert corpus['years'] == [2022] and corpus['lap_labels_read'] is False and corpus['model_fits'] == 0
    assert tuple(e['event_key'] for e in corpus['events']) == TRAIN_EVENTS
    assert schedule['seed'] == 20260908 and schedule['rng'] == 'PCG64'
    assert schedule['steps'] == 800 and schedule['batch_size'] == 32
    assert schedule['horizons'] == [4, 16, 32] and schedule['negatives'] == 31
    assert schedule['model_fits_before_schedule_close'] == 0
    arrays = {name: binary(schedule['arrays'][name], bindings, dtype, shape) for name, dtype, shape in (
        ('stream_index', '<i4', (800, 32)), ('endpoint', '<i8', (800, 32)),
        ('positives', '<i8', (800, 32, 3)), ('negatives', '<i8', (800, 32, 31)))}
    streams, event_drivers, total_endpoints = [], [], 0
    for event in corpus['events']:
        assert event['raw_validation']['complete'] is True and event['raw_validation']['stream_input_valid'] is True
        assert [d['driver_id'] for d in event['drivers']] == sorted(d['driver_id'] for d in event['drivers'])
        good = []
        for driver in event['drivers']:
            fields = driver['arrays']; n = driver['packets']
            # Retain only compact integer geometry, not hundreds of open maps
            # or any full token tensor. Temporary maps close after copying.
            times = np.array(binary(fields['availability_ns'], bindings, '<i8', (n,)))
            seq = np.array(binary(fields['packet_sequence'], bindings, '<i8', (n,)))
            previous = binary(fields['previous_availability_ns'], bindings, '<i8', (n,))
            assert not n or previous[0] == -1
            assert np.array_equal(previous[1:], times[:-1]) and np.all(seq[1:] > seq[:-1])
            values = fields['values']; bindings.record(values)
            assert values['dtype'] == '<f4' and values['shape'] == [n, 34] and values['bytes'] == n*34*4
            valid = np.array(binary(fields['eligible_endpoints'], bindings, '<i8', (driver['eligible_endpoints'],)))
            assert np.array_equal(valid, eligibility(times)), 'Endpoint eligibility differs'
            total_endpoints += len(valid)
            if len(valid): good.append(len(streams))
            streams.append((event['event_key'], driver['driver_id'], times, valid, seq))
        assert good, 'A fixed event has no eligible driver'
        event_drivers.append(good)
    rng = np.random.Generator(np.random.PCG64(20260908)); frequencies = Counter()
    for step in range(800):
        for row in range(32):
            drivers = event_drivers[int(rng.integers(22))]
            stream = drivers[int(rng.integers(len(drivers)))]; _, _, times, valid, seq = streams[stream]
            endpoint = int(valid[int(rng.integers(len(valid)))])
            pool = negative_pool(times, endpoint)
            assert len(pool) >= 31 and endpoint+32 < len(times)
            neg = rng.choice(pool, 31, replace=False)
            assert arrays['stream_index'][step, row] == stream and arrays['endpoint'][step, row] == endpoint
            assert np.array_equal(arrays['positives'][step, row], endpoint+np.array([4, 16, 32]))
            assert np.array_equal(arrays['negatives'][step, row], neg), 'Frozen negative draw differs'
            assert len(np.unique(neg)) == 31 and not np.isin(neg, endpoint+np.array([4, 16, 32])).any()
            # Last128 with a strict180s window and a physical prefix. Equal-clock
            # later packets can never occur in this inclusive endpoint slice.
            first = max(0, endpoint-127, int(np.searchsorted(times, int(times[endpoint])+1-180_000_000_000)))
            assert 1 <= endpoint-first+1 <= 128 and np.all(seq[first:endpoint+1] <= seq[endpoint])
            frequencies[stream] += 1
    draws = [{'event_key': event, 'driver_id': driver, 'draws': frequencies[i]}
             for i, (event, driver, *_rest) in enumerate(streams)]
    old.equal(schedule['driver_draw_counts'], draws, name='driver draws')
    old.equal(schedule['event_draw_counts'], [{'event_key': event, 'draws': sum(d['draws'] for d in draws if d['event_key'] == event)}
                                             for event in TRAIN_EVENTS], name='event draws')
    return {'scheduled_rows': 25600, 'positive_identities': 76800, 'negative_identities': 793600,
            'events': 22, 'drivers': len(streams), 'eligible_endpoints': total_endpoints,
            'exact_PCG64_schedule': True, 'physical_prefix_inclusive_endpoint': True}


def verify_encoders(pretrained, corpus_lock, bindings):
    import torch
    from research.experiments.boundary_20260908.telemetry_sequence import encoder
    torch.set_num_threads(1)
    initial = {k: v.detach().numpy() for k, v in encoder.new_model(frozen=True).state_dict().items()}
    initial_hash = state_digest(initial)
    assert pretrained['initial_state_sha256'] == initial_hash and pretrained['fits'] == 2
    assert pretrained['historical_lap_labels_read'] is False and pretrained['checkpoint_policy'] == 'final_fixed_step_only'
    parameters = {}; elapsed = 0.
    for control in CONTROLS:
        item = pretrained['encoders'][control]
        state = torch.load(bindings.record(item), map_location='cpu', weights_only=True)
        value = {k: v.detach().cpu().numpy() for k, v in state.items()}
        digest = state_digest(value)
        assert digest == item['state_sha256'] and sum(v.size for v in value.values()) == 15048
        if control == 'random':
            assert digest == initial_hash
            for key in initial: assert np.array_equal(value[key], initial[key])
        else:
            trace = bindings.load(bindings.record(item['training']))
            old.equal(trace, {'control': control, 'steps': 800, 'batch_size': 32, 'seed': 20260908,
                'initial_state_sha256': initial_hash, 'final_state_sha256': digest, 'parameters': 15048,
                'learning_rate': .001, 'gradient_limit': 1., 'checkpoint_policy': 'final_fixed_step_only'}, name=control+' trace')
            assert trace['corpus']['sha256'] == corpus_lock['corpus']['sha256']
            assert trace['schedule']['sha256'] == corpus_lock['schedule']['sha256']
            assert np.isfinite(trace['elapsed_seconds']) and 0 < trace['elapsed_seconds'] <= 3600
            elapsed += trace['elapsed_seconds']
            for name in ('losses', 'gradient_norm_before_clip'):
                a = np.asarray(trace[name], float)
                assert a.shape == (800,) and np.isfinite(a).all() and np.all(a >= 0)
        parameters[control] = value
    assert elapsed <= 3600 and pretrained['shared_deadline_seconds'] == 3600
    return parameters, {'same_initial_state_sha256': initial_hash, 'trained_encoders': 2,
                        'recorded_steps_per_encoder': 800, 'random_is_exact_initial_state': True,
                        'limit': 'State and800-entry trace bindings verified; optimizer steps are not retrained.'}


def embedding_arrays(item, records, bindings):
    with np.load(bindings.record(item), allow_pickle=False) as f:
        assert set(f.files) == NPZ_FIELDS, 'Saved embedding schema differs'
        data = {name: f[name].copy() for name in f.files}
    n = len(records)
    assert n == item['rows']
    assert data['issuance_ids'].tolist() == [r['issuance_id'] for r in records]
    assert data['cutoff_ns'].dtype == np.int64 and data['cutoff_ns'].tolist() == [r['telemetry_cutoff_ns'] for r in records]
    assert data['support'].dtype == data['empty_context'].dtype == np.dtype(bool)
    assert data['support'].tolist() == [r['telemetry_supported'] for r in records]
    assert all(type(r['telemetry_supported']) is bool for r in records)
    assert data['original_feature_row_sha256'].tolist() == [old.digest(r) for r in records]
    for name in ('empty_context', 'support', 'context_sha256', 'original_feature_row_sha256'):
        assert data[name].shape == (n,)
    for control in CONTROLS:
        a = data[control]; assert a.dtype == np.float32 and a.shape == (n, 16) and np.isfinite(a).all()
        assert np.all(a[data['empty_context']] == 0)
        assert np.allclose(np.linalg.norm(a[~data['empty_context']], axis=1), 1., atol=2e-6, rtol=2e-6)
    return data


def replay_neural(entry, records, data, parameters):
    from research.experiments.boundary_20260908.telemetry_sequence import tokens
    positions = sorted({0, len(records)//2, len(records)-1})
    assert len(positions) == 3
    source = entry['raw_source']; cursor = None; output = []
    if source['status'] != 'unavailable': cursor = tokens.TokenCursor(source['decoded_body_path'])
    try:
        for position in positions:
            row = records[position]; cutoff = int(row['issued_at_ns'])-entry['lag_seconds']*1_000_000_000
            snapshot = (cursor.query(str(row['driver_id']), cutoff_ns=cutoff) if cursor else
                        tokens.empty_snapshot(str(row['driver_id']), cutoff_ns=cutoff))
            assert snapshot.supported == bool(data['support'][position]) and cutoff == data['cutoff_ns'][position]
            assert old.digest(snapshot.values) == data['context_sha256'][position]
            assert bool((snapshot.source_sequences == -1).all()) == bool(data['empty_context'][position])
            key = tokens.permutation_key(row['event_key'], row['driver_id'], snapshot.provenance['last_admitted_driver_packet_sequence'])
            errors = {}
            for control in CONTROLS:
                context = tokens.permute_context(snapshot.values, key=key) if control == 'permuted' else snapshot.values
                value = representation(parameters[control], context)
                assert np.allclose(value, data[control][position], atol=1e-5, rtol=1e-4), 'Independent neural forward differs'
                errors[control] = float(np.max(np.abs(value-data[control][position])))
            output.append({'event_key': entry['event_key'], 'lag_seconds': entry['lag_seconds'],
                'row_position': position, 'issuance_id': row['issuance_id'], 'cutoff_ns': cutoff,
                'context_sha256': str(data['context_sha256'][position]), 'supported': snapshot.supported,
                'reconstructed_admitted_sequences': snapshot.source_sequences[snapshot.source_sequences >= 0].tolist(),
                'maximum_absolute_forward_error': errors})
    finally:
        if cursor: cursor.close()
    return output


def attach_labels(records, labels):
    assert len(records) == len(labels)
    by_id = {r['issuance_id']: r for r in labels}
    assert len(by_id) == len(labels) and set(by_id) == {r['issuance_id'] for r in records}
    result = []
    for row in records:
        assert not set(TARGETS).intersection(row), 'Target entered feature ledger'
        target = by_id[row['issuance_id']]
        for key in (*old.KEYS, 'issued_at_ns', 'year'): assert row[key] == target[key]
        status = target['outcome_status']; assert status in ('matched', 'unmatched')
        if status == 'matched':
            assert np.isfinite(target['lap_time_seconds']) and target['lap_time_seconds'] > 0
            assert target['target_at_ns'] > row['issued_at_ns'] and target['target_timestamp'] > row['issued_at_timestamp']
        else:
            assert all(target[key] is None for key in TARGETS if key != 'outcome_status')
        result.append({**row, **{key: target[key] for key in TARGETS}})
    return pd.DataFrame(result)


def forecast_arrays(rows, feature_rows, lag, names):
    assert len(rows) == len(feature_rows)
    for point, row in zip(rows, feature_rows):
        assert not set(TARGETS).intersection(point)
        for key in (*old.KEYS, 'issuance_id', 'issued_at_ns', 'telemetry_supported'): assert point[key] == row[key]
        assert point['lag_seconds'] == lag and point['feature_row_sha256'] == old.digest(row)
        assert set(point['predictions']) == set(names)
    result = {name: np.array([row['predictions'][name] for row in rows], dtype=np.float64) for name in names}
    assert all(np.isfinite(v).all() and np.all(v > 0) for v in result.values())
    return result


def direct_replay(new_bundle, old_bundle, frame, tables):
    support = frame.telemetry_supported.to_numpy(bool)
    base_names, extra_names = new_bundle['base_features'], new_bundle['telemetry_features']
    assert len(base_names) == 80 and len(extra_names) == 90
    assert base_names == old_bundle['base_features'] and extra_names == old_bundle['telemetry_features']
    results = old.replay(old_bundle, frame, support, base_names, extra_names)
    base = frame[base_names].copy()
    extra = pd.DataFrame(np.array(frame.telemetry_values.tolist()), index=frame.index,
                         columns=['telemetry__'+x for x in extra_names])
    embed_names = [f'sequence__embedding_{i:02d}' for i in range(16)]
    assert new_bundle['embedding_features'] == embed_names
    anchor = frame.forecast_naive_seconds.to_numpy(float)
    for control in CONTROLS:
        x = pd.concat([base, extra, pd.DataFrame(tables[control], index=frame.index, columns=embed_names)], axis=1)
        saved = new_bundle['models'][control+'_hgb']
        assert saved['features'] == x.columns.tolist() and saved['kind'] == 'hgb'
        value = results['base_hgb'].copy()
        if support.any(): value[support] = anchor[support]+np.clip(saved['model'].predict(x.loc[support]), -3., 3.)
        results[control+'_hgb'] = value
    return results


def gates(comparisons):
    checks = {}
    for lag in ('2', '0'):
        assert set(comparisons[lag]) == set(REFERENCES)
        checks[lag] = {}
        for name, value in comparisons[lag].items():
            gain = value['relative_reduction']
            checks[lag][name] = ({'minimum_relative_gain': gain is not None and gain >= .01,
                'event_ci_upper_negative': value['event_ci95'][1] < 0,
                'block3_ci_upper_negative': value['block3_ci95'][1] < 0,
                'all_leave_one_event_out_negative': value['loo_max_delta'] is not None and value['loo_max_delta'] < 0}
                if lag == '2' else {'positive_relative_gain': gain is not None and gain > 0})
    return checks, all(value for names in checks.values() for flags in names.values() for value in flags.values())


def independently_score(frames, predictions):
    metrics, comparisons, checks, coverage = {}, {}, {}, {}
    for lag in ('2', '0'):
        frame = frames[lag]; matched = frame.outcome_status.eq('matched').to_numpy()
        y = frame.loc[matched, 'lap_time_seconds'].to_numpy(float)
        events = frame.loc[matched, 'event_key'].to_numpy()
        assert tuple(sorted(set(events))) == TEST_EVENTS and len(y) == 20007
        coverage[lag] = old.coverage(frame, frame.telemetry_supported.to_numpy(bool))
        points = {name: predictions[lag][name][matched] for name in NAMES}
        metrics[lag] = {}
        for name, value in points.items():
            errors = np.abs(value-y)
            metrics[lag][name] = {'rows': len(y), 'events': 22, 'row_mae_seconds': float(errors.mean()),
                'event_mae_seconds': float(np.mean([errors[events == e].mean() for e in TEST_EVENTS]))}
        comparisons[lag] = {name: old.independent_comparison(events, y, points['ordered_hgb'], points[name]) for name in REFERENCES}
    checks, advances = gates(comparisons)
    return {'metrics': metrics, 'comparisons': comparisons, 'gate_checks': checks, 'coverage': coverage,
        'advances_to_later_evaluation': advances,
        'decision': 'advance_to_frozen_later_evaluation' if advances else 'stop_no_new_transfer_or_promotion'}


def verify(execution, *, design_sha256, selection_sha256):
    execution = Path(execution).resolve()
    # This must be the first artifact access. No corpus, label, pickle, NPZ or
    # source graph may be opened while the actual selection is still pending.
    selected_path = execution/'selection_lock.json'
    if not selected_path.is_file(): raise ValueError('Closed selection_lock.json required before historical replay')
    if any(execution.glob('*_failure.json')):
        raise ValueError('Failed execution is preserved and cannot be verified as a completed selection')
    selected = old.read(selected_path)
    assert selected['selection']['sha256'] == selection_sha256
    assert selected['design_lock']['sha256'] == design_sha256
    bindings = old.Bindings()
    protocol = bindings.load(bindings.check(HERE/'protocol.json', PROTOCOL_SHA))
    for path in HERE.glob('*.py'): bindings.check(path, old.sha(path))
    selection = bindings.load(bindings.record(selected['selection']))
    design = bindings.load(bindings.record(selected['design_lock']))
    assert Path(selected['selection']['path']).resolve() == execution/'selection.json'
    assert Path(selected['design_lock']['path']).resolve() == execution/'design_lock.json'
    assert design['specification']['sha256'] == protocol['specification']['sha256']
    assert design['historical_sequence_builds_before_lock'] == design['new_fits_before_lock'] == 0
    review = bindings.load(bindings.record(design['independent_review']))
    tests = bindings.load(bindings.record(design['pre_fit_tests']))
    assert review['approved_for_execution_lock'] is True and review['source_files'] == design['sources']
    assert tests['exit_code'] == 0 and tests['source_files'] == design['sources']
    parent = design['parent']; authority = old.read(bindings.record(parent['verification']))
    assert authority['status'] == 'passed'
    # Check only the already-closed consumed parent graph, not its455-file
    # recursive raw replay. The current design independently hashes these bytes.
    for path, value in design['inputs'].items():
        rel = str(Path(path).resolve().relative_to(ROOT))
        if rel in authority['bindings']: assert authority['bindings'][rel]['sha256'] == value
    locks = {name: bindings.load(execution/(name+'.json')) for name in
             ('corpus_lock', 'pretrain_lock', 'embedding_lock', 'fit_lock', 'selection_issuance_lock')}
    corpus_lock, trained, embedded, fitted, issued = [locks[n] for n in locks]
    def same(item, filename): assert bindings.record(item) == execution/filename
    for item in (corpus_lock, trained, embedded): same(item['design_lock'], 'design_lock.json')
    same(trained['corpus_lock'], 'corpus_lock.json'); same(embedded['pretrain_lock'], 'pretrain_lock.json')
    same(fitted['embedding_lock'], 'embedding_lock.json'); same(issued['embedding_lock'], 'embedding_lock.json')
    same(issued['fit_lock'], 'fit_lock.json'); same(selection['issuance_lock'], 'selection_issuance_lock.json')
    for key, filename in (('design_lock', 'design_lock.json'), ('embedding_lock', 'embedding_lock.json'), ('fit_lock', 'fit_lock.json')):
        same(selection[key], filename)
    clocks = [design['closed_at_utc'], *[item['closed_at_utc'] for item in locks.values()],
              selection['completed_at_utc'], selected['closed_at_utc']]
    timestamps = [datetime.fromisoformat(value) for value in clocks]
    assert all(t.utcoffset() is not None for t in timestamps) and timestamps == sorted(timestamps)
    assert corpus_lock['source_years'] == [2022] and corpus_lock['encoder_fits'] == 0
    assert embedded['external_labels_read'] is False and embedded['supervised_fits'] == 0
    assert fitted['external_selection_labels_read'] is False and issued['external_selection_labels_read'] is False
    assert embedded['encoders'] == trained['encoders']
    assert issued['old_reference_forecasts'] == parent['reference_forecasts']
    assert fitted['training_labels'] == parent['years']['2022']['labels']
    assert selection['selection_labels'] == parent['years']['2023']['labels']
    corpus = bindings.load(bindings.record(corpus_lock['corpus']))
    schedule = bindings.load(bindings.record(corpus_lock['schedule']))
    assert schedule['corpus']['sha256'] == corpus_lock['corpus']['sha256']
    assert corpus['design_lock']['sha256'] == design_sha256
    assert (timestamps[0] <= datetime.fromisoformat(corpus['started_at_utc'])
            <= datetime.fromisoformat(corpus['completed_at_utc'])
            <= datetime.fromisoformat(schedule['closed_at_utc']) <= timestamps[1])
    assert corpus['source_files'] == schedule['source_files']
    assert all(design['sources'][key] == value for key, value in corpus['source_files'].items())
    schedule_checks = verify_schedule(corpus, schedule, bindings)
    parameters, encoder_checks = verify_encoders(trained, corpus_lock, bindings)
    with bindings.record(fitted['models']).open('rb') as f: bundle = pickle.load(f)
    old_fit = bindings.load(bindings.record(parent['fit_lock']))
    with bindings.record(old_fit['models']).open('rb') as f: old_bundle = pickle.load(f)
    assert bundle['fit_summary'] == fitted['fit_summary']
    old.equal(bundle['fit_summary'], {'fit_year': 2022, 'fit_latency_seconds': 2, 'fits': 3,
        'representation_fits': 0, 'old_reference_refits': 0, 'rows_per_fit': 18363,
        'events': list(TRAIN_EVENTS)}, name='supervised fitting')
    assert bundle['encoder_sha256'] == {c: trained['encoders'][c]['sha256'] for c in CONTROLS}
    entries = embedded['entries']
    assert [(e['event_key'], e['lag_seconds']) for e in entries] == [(e, lag) for e in TRAIN_EVENTS+TEST_EVENTS for lag in (2, 0)]
    assert embedded['issuances_per_lag'] == {'0': 39220, '2': 39220}
    records = {'0': [], '2': []}; vectors = {lag: {c: [] for c in CONTROLS} for lag in records}
    samples = []; train_ids = []; all_counts = Counter()
    for entry in entries:
        event, lag = entry['event_key'], str(entry['lag_seconds'])
        rows = old.rows(bindings.record(entry['original_features']))
        assert entry['raw_source'] == next(s for s in parent['streams'] if s['event_key'] == event)
        assert all(r['event_key'] == event and r['year'] == event//100 for r in rows)
        assert all(r['issued_at_ns']-int(lag)*1_000_000_000 == r['telemetry_cutoff_ns'] for r in rows)
        assert all(not set(TARGETS).intersection(r) for r in rows)
        original = next(r for r in parent['years'][str(event//100)]['original_references'] if r['event_key'] == event)
        original_rows = old.rows(bindings.record(original))
        assert len(original_rows) == len(rows) and len({r['issuance_id'] for r in rows}) == len(rows)
        for source, row in zip(original_rows, rows):
            assert all(old.normalized(row[k]) == old.normalized(v) for k, v in source.items()), 'Original issuance changed'
        arrays = embedding_arrays(entry, rows, bindings); all_counts[lag] += len(rows)
        if event in protocol['neural_replay_sample']['event_keys']:
            samples.extend(replay_neural(entry, rows, arrays, parameters))
        if event//100 == 2023:
            records[lag].extend(rows)
            for control in CONTROLS: vectors[lag][control].append(arrays[control])
        elif lag == '2': train_ids.extend(r['issuance_id'] for r in rows)
    assert dict(all_counts) == {'0': 39220, '2': 39220} and len(samples) == 24
    def label_rows(year):
        closure = bindings.load(bindings.record(parent['years'][str(year)]['labels']))
        assert tuple(e['event_key'] for e in closure['events']) == (TRAIN_EVENTS if year == 2022 else TEST_EVENTS)
        output = []
        for entry in closure['events']:
            rel = str(bindings.record(entry).relative_to(ROOT))
            assert authority['bindings'][rel]['sha256'] == entry['sha256']
            output.extend(old.rows(bindings.record(entry)))
        return output
    training_labels = label_rows(2022); labels = label_rows(2023)
    assert [r['issuance_id'] for r in training_labels] == train_ids
    matched_ids = [r['issuance_id'] for r in training_labels if r['outcome_status'] == 'matched']
    assert len(matched_ids) == 18363
    assert hashlib.sha256(json.dumps(matched_ids, separators=(',', ':')).encode()).hexdigest() == bundle['fit_summary']['issuance_ids_sha256']
    frames, predictions = {}, {}
    for lag in ('2', '0'):
        frame = pd.DataFrame(records[lag]); support = frame.telemetry_supported.to_numpy(bool)
        table = {c: np.concatenate(vectors[lag][c]) for c in CONTROLS}
        actual = forecast_arrays(old.rows(bindings.record(issued['forecasts'][lag])), records[lag], int(lag), NAMES)
        references = forecast_arrays(old.rows(bindings.record(parent['reference_forecasts'][lag])), records[lag], int(lag), OLD_NAMES)
        replayed = direct_replay(bundle, old_bundle, frame, table)
        for name in NAMES:
            assert np.allclose(actual[name], replayed[name], atol=1e-12, rtol=0), name+' direct estimator replay'
        for name in OLD_NAMES: bits(actual[name], references[name], name+' closed reference')
        for name in NEW_NAMES: bits(actual[name][~support], actual['base_hgb'][~support], name+' exact fallback')
        frames[lag] = attach_labels(records[lag], labels); predictions[lag] = actual
    assert [r['issuance_id'] for r in records['2']] == [r['issuance_id'] for r in records['0']]
    for name in (*old.KEYS, *TARGETS, 'issued_at_ns', 'forecast_naive_seconds', *bundle['base_features']):
        assert old.normalized(frames['2'][name].tolist()) == old.normalized(frames['0'][name].tolist()), 'Lag identity/label parity'
    bits(predictions['2']['base_hgb'], predictions['0']['base_hgb'], 'Lag baseline')
    recomputed = independently_score(frames, predictions)
    old.equal(selection['summary'], recomputed, name='selection', tolerance=1e-10)
    old.equal(selection['summary'], {'model_names': list(NAMES), 'references': list(REFERENCES),
        'candidate_eligible_for_advancement': 'ordered_hgb', 'primary_latency_seconds': 2,
        'sensitivity_latency_seconds': 0, 'sensitivity_refitted': False,
        'old_references_preserved_bitwise': True, 'promotion': False}, name='selection rules')
    assert abs(recomputed['metrics']['2']['base_hgb']['event_mae_seconds']-old.BASE_MAE) <= 1e-12
    assert selected['selected_candidate'] == 'ordered_hgb' and selected['promotion'] is False
    assert type(selected['advances_to_later_evaluation']) is bool
    assert selected['advances_to_later_evaluation'] == recomputed['advances_to_later_evaluation']
    # Recheck publication-critical closure bytes after all computations.
    assert not any(execution.glob('*_failure.json')), 'Execution failed during verification'
    assert old.sha(execution/'selection.json') == selection_sha256 and old.sha(execution/'design_lock.json') == design_sha256
    return {'status': 'passed', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_sha256': selection_sha256, 'design_sha256': design_sha256, 'protocol_sha256': PROTOCOL_SHA,
        'schedule': schedule_checks, 'encoders': encoder_checks, 'neural_samples': samples,
        'direct_estimator_predictions_checked': len(records['2'])*12,
        'all_issuances_per_lag': len(records['2']), 'matched_per_lag': 20007,
        'unmatched_per_lag': len(records['2'])-20007, 'recomputed': recomputed,
        'limits': ['No refits; original populations and labels reuse the hash-bound successful parent verification.',
            'Neural raw-prefix numerical replay covers24 predeclared snapshots; other embedding rows are hash-bound.',
            'Admitted sequence IDs are reconstructed and reported, not compared to absent NPZ sequence metadata.',
            'Schedule and recorded step/state bindings do not independently reproduce optimizer training.',
            'This verifier author also authored tokens/corpus; numerical forward and metric routines have separate authorship.',
            'Exposed historical seasons and archive availability clocks do not prove prospective performance.'],
        'bindings': {str(p.relative_to(ROOT)): v for p, v in sorted(bindings.checked.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--design-sha256', required=True); parser.add_argument('--selection-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError('Verification outputs are immutable')
    with threadpool_limits(limits=1):
        result = verify(args.execution, design_sha256=args.design_sha256, selection_sha256=args.selection_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f: json.dump(result, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'path': str(args.out), 'sha256': old.sha(args.out)}, allow_nan=False), flush=True)


if __name__ == '__main__': main()
