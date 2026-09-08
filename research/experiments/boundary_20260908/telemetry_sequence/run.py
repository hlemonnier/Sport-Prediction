"""Closed-input CPC orchestration; historical execution requires a reviewed lock.

Suggested commit: research(f1-live): execute locked causal telemetry sequence controls
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from . import encoder, tokens
from research.experiments.boundary_20260908.telemetry import data as old_data
from research.experiments.boundary_20260908.telemetry import run as old_run

ROOT = tokens.ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/research/boundary_20260908/telemetry_sequence'
PARENT = ROOT/'artifacts/research/boundary_20260908/telemetry/execution'
ACQUISITION = PARENT.parent/'acquisition.json'
YEARS = (2022, 2023)
LAGS = (2, 0)
CONTROLS = ('ordered', 'permuted', 'random')
EVENTS = tuple(range(202201, 202223))+tuple(range(202301, 202323))
TARGET_COLUMNS = old_run.TARGET_COLUMNS
SHARED_PRETRAIN_SECONDS = 3600.
EMBED_BATCH = 128


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return old_data._read_json(path)
def save(path, value): return old_data._write_json(path, value)
def record(path): return {'path': str(Path(path).resolve()), 'sha256': old_data.sha(path)}
def progress(**value): print(json.dumps(value, allow_nan=False), flush=True)


def check(value):
    path = Path(value['path'])
    if not path.is_absolute(): path = ROOT/path
    if old_data.sha(path) != value['sha256']: raise ValueError('Hash binding changed: '+str(path))
    return path


def load_record(value): return read(check(value))


def sources():
    paths = {p for pattern in ('*.py', '*.json', '*.md') for p in HERE.glob(pattern)}
    dependencies = tokens.dependency_bindings()
    # The new runner imports the already reviewed helpers, so their entire
    # recorded source graph is part of this execution's source lock as well.
    dependencies.update(read(PARENT/'design_lock.json')['sources'])
    for name, expected in dependencies.items():
        path = ROOT/name
        if old_data.sha(path) != expected: raise ValueError('Inherited source changed: '+name)
        paths.add(path)
    return {str(p.relative_to(ROOT)): old_data.sha(p) for p in sorted(paths)}


def parent_graph():
    """Read closed metadata and hash actual inputs; do not parse target rows."""
    paths = [PARENT/name for name in ('design_lock.json', 'data_lock.json', 'fit_lock.json',
        'selection_issuance_lock.json', 'selection_lock.json', 'verification/result_v2.json')]
    graph = {str(p): old_data.sha(p) for p in paths}
    design, data, fit, issued, selected, verified = map(read, paths)
    if (verified['status'] != 'passed' or verified['selection_sha256'] != selected['selection']['sha256']
            or data['external_labels_attached'] is not False or data['model_fits'] != 0):
        raise ValueError('A verified target-free parent data closure is required')
    if check(data['design_lock']) != paths[0] or check(fit['data_lock']) != paths[1]:
        raise ValueError('Parent closure path mismatch')
    if check(issued['fit_lock']) != paths[2] or check(issued['data_lock']) != paths[1]:
        raise ValueError('Parent forecast closure mismatch')
    if selected['design_lock'] != data['design_lock'] or issued['selection_labels_attached'] is not False:
        raise ValueError('Parent source or forecast chronology mismatch')
    # Result values were exposed previously. We bind the result but do not load
    # it here; later stage chronology concerns newly generated predictions.
    check(selected['selection']);graph[str(check(selected['selection']))] = selected['selection']['sha256']
    def bind(item):
        p = check(item);graph[str(p)] = item['sha256'];return p
    acquired = read(bind(record(ACQUISITION)))
    if tuple(r['event_key'] for r in acquired['streams']) != EVENTS: raise ValueError('Full 44-event acquisition required')
    streams = {r['event_key']: r for r in acquired['streams']}
    years = {}
    for year in YEARS:
        info = data['years'][str(year)]
        closure = read(bind(info['feature_closure']));validation = read(bind(info['input_validation']))
        if validation['status'] != 'PASS' or validation['feature_closure_sha256'] != info['feature_closure']['sha256']:
            raise ValueError('Parent full-stream validation is not closed')
        if closure['external_labels_attached'] is not False: raise ValueError('Parent features contain external labels')
        if [r['event_key'] for r in closure['events']] != list(range(year*100+1, year*100+23)):
            raise ValueError('Parent year event population changed')
        for event in closure['events']:
            for lag in LAGS: bind(event['ledgers'][str(lag)])
        for ref in info['original_references']: bind(ref)
        label_path = PARENT/f'labels_{year}'/'label_closure.json'
        label_record = record(label_path)
        if year == 2022 and label_record != fit['training_labels']: raise ValueError('Training label closure mismatch')
        labels = read(bind(label_record))
        if labels['feature_closure'] != info['feature_closure']: raise ValueError('Labels are not attached to these original issues')
        # Match the successful verifier's original hash; never infer label
        # legitimacy merely from a freshly computed replacement file hash.
        for item in [label_record, *labels['events']]:
            p = bind(item);relative = str(p.relative_to(ROOT))
            if verified['bindings'].get(relative, {}).get('sha256') != item['sha256']:
                raise ValueError('Label file is outside the verified parent graph')
        years[str(year)] = {**info, 'labels': label_record}
    for entry in issued['forecasts'].values(): bind(entry)
    for stream in streams.values():
        if stream['status'] == 'unavailable': continue
        item = {'path': stream['decoded_body_path'], 'sha256': stream['decoded_body_sha256']}
        p = bind(item)
        if p.stat().st_size != stream['decoded_body_bytes']: raise ValueError('Raw stream byte count changed')
    for path, digest in graph.items():
        relative = str(Path(path).relative_to(ROOT))
        # The verification file and selection lock naturally close after the
        # earlier verifier's input inventory; other files must be in its graph.
        if Path(path) not in (paths[4], paths[5]):
            if verified['bindings'].get(relative, {}).get('sha256') != digest:
                raise ValueError('Consumed input is not bound by the successful parent replay: '+relative)
    return graph, {'data_lock': record(paths[1]), 'fit_lock': record(paths[2]),
        'selection_lock': record(paths[4]), 'verification': record(paths[5]),
        'years': years, 'reference_forecasts': issued['forecasts'], 'streams': list(streams.values())}


def validate_spec(spec):
    expected = {
        'discovery': {'event_keys':list(EVENTS),'train_years':[2022],'selection_years':[2023],
            'original_issuances':39220,'train_matched_rows':18363,'selection_matched_rows':20007,'unmatched_retained':850},
        'tokens': {'coordinates':34,'max_packets':128,'window_seconds':180,'primary_lag_seconds':2,'sensitivity_lag_seconds':0},
        'encoder': {'embedding_dimension':16,'width':24,'parameters':15048,'fit_controls':['ordered','permuted'],
            'dilations':list(encoder.DILATIONS)},
        'ssl': {'steps':800,'batch_size':32,'seed':20260908,'temperature':.1,'negative_count':31,
            'positive_packet_offsets':[4,16,32],'selection_year_tokens_used_for_training':False},
        'resources': {'cpu_threads':1,'fits_serial':True,'pretraining_shared_seconds':3600,
            'max_resident_bytes':3*1024**3,'provider_driver_lru_limit':8},
        'supervised': {'fits':3,'features':186,'fit_years':[2022],'fit_latency_seconds':2,
            'model_names':['ordered_hgb','permuted_hgb','random_hgb'],'loss':'absolute_error','learning_rate':.06,
            'max_iter':150,'max_leaf_nodes':15,'min_samples_leaf':80,'l2_regularization':10,'early_stopping':False,
            'random_state':20260907,'training_residual_clip_seconds':5,'prediction_correction_clip_seconds':3,
            'representation_fine_tuning':False,'zero_lag_refit':False},
        'selection_gate': {'candidate':'ordered_hgb','references':['base_hgb','telemetry_hgb','quality_hgb','permuted_hgb','random_hgb'],
            'minimum_relative_event_mae_reduction_each_reference':.01,'required_events':22,
            'all_leave_one_event_out_negative_each':True,'event_and_block3_upper_ci_negative_each':True,
            'zero_second_sensitivity_positive_each':True,'reuse_old_reference_forecasts_bitwise':True},
        'uncertainty': {'seed':20260907,'resamples_each':20000,'circular_event_block_length':3}}
    for section, fields in expected.items():
        for key,value in fields.items():
            actual=spec[section][key]
            if actual!=value or (type(value) is bool and type(actual) is not bool):
                raise ValueError('Frozen specification mismatch: '+section+'.'+key)
    if ROOT/spec['telemetry90_execution_directory'] != PARENT:raise ValueError('Different parent execution directory')
    for item in [*spec['parent_bindings'].values(),*spec['frozen_components'].values()]:check(item)
    if (encoder.TRAIN_STEPS,encoder.BATCH_SIZE,encoder.EMBEDDING_DIM,tuple(encoder.CONTROLS),len(tokens.TOKEN_NAMES))!=(800,32,16,CONTROLS,34):
        raise ValueError('Frozen component layout changed')


def resource_check():
    from . import corpus
    return corpus._rss()


def freeze(out, review_path):
    out = Path(out)
    if (out/'design_lock.json').exists(): raise FileExistsError('Existing design is immutable')
    spec = read(HERE/'specification.json');validate_spec(spec)
    before = sources();review = read(review_path)
    if review.get('approved_for_execution_lock') is not True or review.get('source_files') != before:
        raise ValueError('Independent review must approve exactly these source bytes')
    bindings, parent = parent_graph()
    command = [sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib', '-p', 'no:cacheprovider', str(HERE)]
    tested = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    save(out/'pre_fit_tests.json', {'command': command, 'exit_code': tested.returncode, 'stdout': tested.stdout,
        'stderr': tested.stderr, 'source_files': before, 'completed_at_utc': now()})
    if tested.returncode: raise ValueError('Pre-fit test suite failed')
    if sources() != before: raise ValueError('Sources changed during tests')
    for path, expected in bindings.items():
        if old_data.sha(path) != expected: raise ValueError('Input changed during tests')
    save(out/'design_lock.json', {'closed_at_utc': now(), 'sources': before, 'inputs': bindings,
        'specification': record(HERE/'specification.json'), 'independent_review': record(review_path),
        'pre_fit_tests': record(out/'pre_fit_tests.json'), 'parent': parent,
        'historical_sequence_builds_before_lock': 0, 'new_fits_before_lock': 0,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__, 'threads': 1},
        'exposure': 'Parent 2023 labels and scores were exposed previously; this is retrospective research, not a fresh holdout.'})
    progress(stage='frozen', design_lock=record(out/'design_lock.json'))


def verify_design(out):
    out=Path(out)
    lock = read(out/'design_lock.json')  # Must precede any historical stage I/O.
    if any(out.glob('*_failure.json')):
        raise ValueError('A failed attempt is preserved; this execution cannot advance')
    if sources() != lock['sources']: raise ValueError('Current source differs from reviewed design')
    for path, expected in lock['inputs'].items():
        if old_data.sha(path) != expected: raise ValueError('Frozen input changed: '+path)
    for key in ('specification', 'independent_review', 'pre_fit_tests'): check(lock[key])
    spec = load_record(lock['specification']);validate_spec(spec)
    return lock, spec


@contextmanager
def attempt(out, stage):
    out = Path(out);save(out/(stage+'_attempt.json'), {'started_at_utc': now(), 'stage': stage})
    try:
        resource_check()
        yield
        resource_check()
    except Exception as exc:
        save(out/(stage+'_failure.json'), {'failed_at_utc': now(), 'stage': stage,
            'type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


def build_corpus(out):
    from . import corpus
    out = Path(out);lock, _ = verify_design(out)
    with attempt(out, 'corpus'):
        events = [row for row in lock['parent']['streams'] if row['event_key']//100 == 2022]
        path = corpus.build_corpus(events, out/'corpus', design_lock_path=out/'design_lock.json',
                                   design_sha256=old_data.sha(out/'design_lock.json'))
        schedule = corpus.build_schedule(path, out/'schedule')
        verify_design(out)
        save(out/'corpus_lock.json', {'closed_at_utc': now(), 'design_lock': record(out/'design_lock.json'),
            'corpus': record(path), 'schedule': record(schedule), 'source_years': [2022], 'encoder_fits': 0})
        progress(stage='corpus_closed', corpus_lock=record(out/'corpus_lock.json'))


def pretrain(out):
    from . import corpus
    out = Path(out);verify_design(out)
    with attempt(out, 'pretrain'):
        closed = read(out/'corpus_lock.json')
        if closed['design_lock'] != record(out/'design_lock.json') or closed['source_years'] != [2022]:
            raise ValueError('Corpus must close under this 2022-only design')
        torch.set_num_threads(1)
        provider = corpus.BatchProvider(check(closed['corpus']), check(closed['schedule']))
        initial = encoder.new_model(frozen=True);initial_digest = encoder.state_digest(initial)
        models_dir = out/'encoders';models_dir.mkdir(exist_ok=False)
        random_path = models_dir/'random.pt'
        with random_path.open('xb') as stream: torch.save(initial.state_dict(), stream)
        entries = {'random': {**record(random_path), 'state_sha256': initial_digest}}
        deadline = time.monotonic()+SHARED_PRETRAIN_SECONDS
        with provider, threadpool_limits(limits=1):
            for control in ('ordered', 'permuted'):
                progress(stage='encoder_started', control=control)
                model, trace = encoder.fit_cpc(provider, control=control, deadline_monotonic=deadline)
                if trace['initial_state_sha256'] != initial_digest: raise ValueError('Encoder initialization differs')
                if trace['steps'] != 800 or trace['batch_size'] != 32: raise ValueError('Incomplete fixed training schedule')
                trace={**trace,'corpus':closed['corpus'],'schedule':closed['schedule']}
                path = models_dir/(control+'.pt')
                with path.open('xb') as stream: torch.save(model.state_dict(), stream)
                trace_path = models_dir/(control+'_training.json');save(trace_path, trace)
                entries[control] = {**record(path), 'state_sha256': encoder.state_digest(model), 'training': record(trace_path)}
                progress(stage='encoder_completed', control=control, elapsed_seconds=trace['elapsed_seconds'])
        verify_design(out);check(closed['corpus']);check(closed['schedule'])
        save(out/'pretrain_lock.json', {'closed_at_utc': now(), 'design_lock': record(out/'design_lock.json'),
            'corpus_lock': record(out/'corpus_lock.json'), 'encoders': entries, 'fits': 2,
            'shared_deadline_seconds': SHARED_PRETRAIN_SECONDS, 'initial_state_sha256': initial_digest,
            'historical_lap_labels_read': False, 'checkpoint_policy': 'final_fixed_step_only'})
        progress(stage='pretraining_closed', pretrain_lock=record(out/'pretrain_lock.json'))


def load_encoders(closed):
    result = {}
    for control in CONTROLS:
        item = closed['encoders'][control]
        model = encoder.new_model(frozen=True)
        model.load_state_dict(torch.load(check(item), map_location='cpu', weights_only=True), strict=True)
        if encoder.state_digest(model) != item['state_sha256']: raise ValueError('Encoder semantic state hash changed')
        result[control] = model
    return result


def embed_event(feature_rows, stream, models, lag):
    """No target reads; every checkpoint is tied to the existing feature row."""
    if any(set(row) & set(TARGET_COLUMNS) for row in feature_rows): raise ValueError('Targets entered embedding stage')
    if not feature_rows or len({row['issuance_id'] for row in feature_rows}) != len(feature_rows):
        raise ValueError('Complete unique original issuance rows required')
    cursor = None if stream['status'] == 'unavailable' else tokens.TokenCursor(check(
        {'path': stream['decoded_body_path'], 'sha256': stream['decoded_body_sha256']}))
    snapshots, keys, batches = [], [], {control: [] for control in CONTROLS}
    ids, cutoffs, empty, support, hashes, token_hashes = [], [], [], [], [], []
    def flush():
        if not snapshots: return
        resource_check()
        numeric = np.stack([s.values for s in snapshots])
        for control in CONTROLS:
            batches[control].append(encoder.embeddings(models[control], numeric, context_keys=tuple(keys), control=control))
        snapshots.clear();keys.clear()
        resource_check()
    try:
        for row in feature_rows:
            cutoff = row['telemetry_cutoff_ns']
            if type(cutoff) is not int or cutoff != row['issued_at_ns']-lag*10**9 or row['telemetry_lag_seconds'] != lag:
                raise ValueError('Old feature cutoff/lag changed')
            if row['event_key'] != stream['event_key'] or type(row['telemetry_supported']) is not bool:
                raise ValueError('Original event/support metadata invalid')
            snapshot = (tokens.empty_snapshot(row['driver_id'], cutoff_ns=cutoff) if cursor is None
                        else cursor.query(row['driver_id'], cutoff_ns=cutoff))
            if snapshot.supported is not row['telemetry_supported']:
                raise ValueError('Token support differs from the closed telemetry90 support gate')
            snapshots.append(snapshot);keys.append(tokens.permutation_key(row['event_key'], row['driver_id'],
                snapshot.provenance['last_admitted_driver_packet_sequence']))
            ids.append(row['issuance_id']);cutoffs.append(cutoff);empty.append(bool((snapshot.source_sequences == -1).all()))
            support.append(snapshot.supported);hashes.append(old_data.digest(row));token_hashes.append(old_data.digest(snapshot.values))
            if len(snapshots) == EMBED_BATCH: flush()
        flush()
    finally:
        if cursor is not None: cursor.close()
    return {'issuance_ids': np.asarray(ids), 'cutoff_ns': np.asarray(cutoffs, dtype=np.int64),
        'empty_context': np.asarray(empty, dtype=bool), 'support': np.asarray(support, dtype=bool),
        'original_feature_row_sha256': np.asarray(hashes), 'context_sha256': np.asarray(token_hashes),
        **{control: np.concatenate(batches[control]) for control in CONTROLS}}


def embed(out):
    out = Path(out);lock, _ = verify_design(out)
    with attempt(out, 'embed'):
        pretrained = read(out/'pretrain_lock.json')
        if pretrained['design_lock'] != record(out/'design_lock.json') or pretrained['fits'] != 2:
            raise ValueError('Both encoders must close under this design')
        check(pretrained['corpus_lock']);torch.set_num_threads(1);models = load_encoders(pretrained)
        streams = {r['event_key']: r for r in lock['parent']['streams']};entries=[];counts={str(lag):0 for lag in LAGS}
        directory=out/'embeddings';directory.mkdir(exist_ok=False)
        with threadpool_limits(limits=1):
            for year in YEARS:
                closed = load_record(lock['parent']['years'][str(year)]['feature_closure'])
                for event in closed['events']:
                    for lag in LAGS:
                        item = event['ledgers'][str(lag)]
                        rows = old_data._read_rows(check(item))
                        value = embed_event(rows, streams[event['event_key']], models, lag)
                        path=directory/f"{event['event_key']}_lag{lag}.npz"
                        with path.open('xb') as f: np.savez(f, **value)
                        entries.append({'event_key':event['event_key'],'lag_seconds':lag,'rows':len(rows),
                            **record(path),'original_features':item,'raw_source':streams[event['event_key']]})
                        counts[str(lag)]+=len(rows)
                    progress(stage='event_embedded', event_key=event['event_key'], rows_per_lag=len(rows))
        if counts != {'0':39220,'2':39220}: raise ValueError('Embedding population differs from the full original inventory')
        verify_design(out)
        save(out/'embedding_lock.json', {'closed_at_utc':now(),'design_lock':record(out/'design_lock.json'),
            'pretrain_lock':record(out/'pretrain_lock.json'),'entries':entries,'issuances_per_lag':counts,
            'external_labels_read':False,'supervised_fits':0,'encoders':pretrained['encoders']})


def load_embedding_year(lock, embedded, year, lag):
    from . import supervised
    frame = old_run.load_year({'years': lock['parent']['years']},year,lag)
    entries=[r for r in embedded['entries'] if r['event_key']//100==year and r['lag_seconds']==lag]
    if [r['event_key'] for r in entries] != list(range(year*100+1,year*100+23)): raise ValueError('Embedding event order changed')
    arrays={key:[] for key in ('issuance_ids','cutoff_ns','empty_context','support','original_feature_row_sha256',*CONTROLS)}
    for entry in entries:
        with np.load(check(entry),allow_pickle=False) as value:
            if len(value['issuance_ids'])!=entry['rows']:raise ValueError('Embedding row count changed')
            for key in arrays: arrays[key].append(value[key].copy())
    joined={key:np.concatenate(value) for key,value in arrays.items()}
    if joined['issuance_ids'].tolist()!=frame.issuance_id.tolist():raise ValueError('Embedding issuance identity/order mismatch')
    if not np.array_equal(joined['cutoff_ns'],frame.telemetry_cutoff_ns):raise ValueError('Embedding cutoff mismatch')
    if not np.array_equal(joined['support'],frame.telemetry_supported):raise ValueError('Embedding support mismatch')
    if joined['original_feature_row_sha256'].tolist()!=[old_data.digest(r) for r in frame.to_dict('records')]:
        raise ValueError('Embedding/old feature row hash mismatch')
    tables={control:supervised.EmbeddingTable(tuple(joined['issuance_ids'].tolist()),tuple(joined['cutoff_ns'].tolist()),
        joined[control],joined['empty_context'],embedded['encoders'][control]['sha256'],control) for control in CONTROLS}
    return frame,tables


def subset_tables(tables, mask):
    from . import supervised
    mask=np.asarray(mask,dtype=bool)
    return {key:supervised.EmbeddingTable(tuple(np.asarray(t.issuance_ids)[mask].tolist()),
        tuple(np.asarray(t.cutoff_ns)[mask].tolist()),t.values[mask].copy(),t.empty_context[mask].copy(),
        t.encoder_sha256,t.control) for key,t in tables.items()}


def references(frame, item, lag):
    from . import supervised
    predictions=old_run.load_forecasts(frame,item,lag)
    return supervised.ReferenceForecasts(tuple(frame.issuance_id),tuple(frame.telemetry_cutoff_ns),predictions,item['sha256'])


def forecast_rows(frame, predictions, lag):
    from . import supervised
    names=(*supervised.OLD_MODEL_NAMES,*supervised.NEW_MODEL_NAMES)
    if set(predictions)!=set(names):raise ValueError('Every old and new forecast is required')
    for value in predictions.values():
        if np.asarray(value).shape!=(len(frame),) or not np.isfinite(value).all():raise ValueError('Invalid forecast vector')
    if any(name in frame for name in TARGET_COLUMNS):raise ValueError('Selection labels precede forecast closure')
    return [{**{key:row[key] for key in (*old_data.KEYS,'issuance_id','issued_at_ns','telemetry_supported')},
        'lag_seconds':lag,'feature_row_sha256':old_data.digest(row),
        'predictions':{name:float(predictions[name][i]) for name in names}}
        for i,row in enumerate(frame.to_dict('records'))]


def select(out):
    from . import supervised
    out=Path(out);lock,_=verify_design(out)
    with attempt(out,'selection'):
        embedded=read(out/'embedding_lock.json')
        if (embedded['design_lock']!=record(out/'design_lock.json') or embedded['external_labels_read'] is not False
                or embedded['supervised_fits']!=0):raise ValueError('Target-free embeddings must close before supervision')
        check(embedded['pretrain_lock'])
        frame,tables=load_embedding_year(lock,embedded,2022,2)
        train_all=old_run.join_labels(frame,lock['parent']['years']['2022']['labels'])
        mask=train_all.outcome_status.eq('matched').to_numpy();training=train_all.loc[mask].copy()
        expected=old_run.join_labels(old_run.load_year({'years':lock['parent']['years']},2022,2),
            lock['parent']['years']['2022']['labels']);expected=expected.loc[expected.outcome_status.eq('matched')].copy()
        with threadpool_limits(limits=1):bundle=supervised.fit_models(training,subset_tables(tables,mask),expected_matched=expected)
        resource_check()
        path=out/'supervised_models.pkl'
        with path.open('xb') as f:pickle.dump(bundle,f,protocol=pickle.HIGHEST_PROTOCOL)
        save(out/'fit_lock.json',{'closed_at_utc':now(),'embedding_lock':record(out/'embedding_lock.json'),
            'models':record(path),'training_labels':lock['parent']['years']['2022']['labels'],
            'fit_summary':bundle['fit_summary'],'external_selection_labels_read':False})
        with path.open('rb') as f:bundle=pickle.load(f)
        frames={};predictions={};refs={};forecasts={}
        for lag in LAGS:
            frame,tables=load_embedding_year(lock,embedded,2023,lag)
            refs[lag]=references(frame,lock['parent']['reference_forecasts'][str(lag)],lag)
            before=old_data.digest(frame.to_dict('records'))
            predictions[lag]=supervised.predict_models(bundle,frame,tables,latency_seconds=lag,saved_references=refs[lag])
            if old_data.digest(frame.to_dict('records'))!=before:raise ValueError('Prediction mutated target-free features')
            path=out/f'selection_lag{lag}_forecasts.jsonl'
            old_data._write_rows(path,forecast_rows(frame,predictions[lag],lag));forecasts[str(lag)]={**record(path),'rows':len(frame)}
            frames[lag]=frame;progress(stage='selection_forecasts_closed',lag_seconds=lag,rows=len(frame))
        verify_design(out)
        save(out/'selection_issuance_lock.json',{'closed_at_utc':now(),'fit_lock':record(out/'fit_lock.json'),
            'embedding_lock':record(out/'embedding_lock.json'),'forecasts':forecasts,
            'external_selection_labels_read':False,'old_reference_forecasts':lock['parent']['reference_forecasts']})
        # This is the first read of old 2023 external label rows by this runner.
        labeled={lag:old_run.join_labels(frame,lock['parent']['years']['2023']['labels']) for lag,frame in frames.items()}
        expected_issued=old_run.reference_year({'years':lock['parent']['years']},2023)
        expected_labeled=old_run.join_labels(expected_issued,lock['parent']['years']['2023']['labels'])
        expected_matched=expected_labeled.loc[expected_labeled.outcome_status.eq('matched')].copy()
        summary=supervised.evaluate_selection(labeled[2],predictions[2],labeled[0],predictions[0],
            expected_matched=expected_matched,expected_issuances=expected_issued,
            primary_references=refs[2],sensitivity_references=refs[0])
        verify_design(out)
        save(out/'selection.json',{'completed_at_utc':now(),'status':'retrospective_sequence_research_no_promotion',
            'summary':summary,'design_lock':record(out/'design_lock.json'),'embedding_lock':record(out/'embedding_lock.json'),
            'fit_lock':record(out/'fit_lock.json'),'issuance_lock':record(out/'selection_issuance_lock.json'),
            'selection_labels':lock['parent']['years']['2023']['labels']})
        save(out/'selection_lock.json',{'closed_at_utc':now(),'selection':record(out/'selection.json'),
            'design_lock':record(out/'design_lock.json'),'selected_candidate':'ordered_hgb',
            'advances_to_later_evaluation':summary['advances_to_later_evaluation'],'promotion':False})
        progress(stage='selection_complete',summary=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','corpus','pretrain','embed','select'))
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--review',type=Path)
    args=parser.parse_args()
    if args.stage=='freeze':
        if args.review is None:parser.error('--review must approve the exact sources')
        freeze(args.out,args.review)
    else:{'corpus':build_corpus,'pretrain':pretrain,'embed':embed,'select':select}[args.stage](args.out)


if __name__=='__main__':main()
