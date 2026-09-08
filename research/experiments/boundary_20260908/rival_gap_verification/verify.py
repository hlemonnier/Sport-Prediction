"""Independent saved-forecast and sampled raw-prefix replay. Never fits models.

Suggested commit: research(f1-live): independently replay rival-gap prediction evidence
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.experiments.boundary_20260908.telemetry_verification_v2.verify import (
    Bindings, digest, equal, gates, independent_comparison, read, rows, sha)
from .prefix import PrefixReplay

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
NAMES = ('base_hgb', 'gap_hgb', 'quality_hgb')
LAGS = (2,0)
ADDED = {'gap_values','gap_supported','gap_cutoff_ns','gap_lag_seconds',
         'gap_provenance','original_row_sha256'}
TARGETS = {'outcome_status','target_id','target_at_ns','target_lap_number',
           'target_timestamp','lap_time_seconds','target_same_stint'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def bits(a,b):
    a,b = np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64),b.view(np.uint64))


def labels_for(bindings, reference):
    closure = bindings.load(bindings.record(reference))
    result = []
    for item in closure['events']:
        result.extend(rows(bindings.record(item)))
    assert len(result) == closure['all_issuances']
    return result


def verify(execution, *, design_sha256, selection_sha256):
    execution = (ROOT/Path(execution)).resolve()
    assert not list(execution.glob('*_failure.json')), 'Failed execution cannot be certified'
    assert (execution/'selection_lock.json').is_file(), 'Selection must close first'
    b = Bindings(ROOT)
    design_path = b.check(execution/'design_lock.json',design_sha256)
    selection_path = b.check(execution/'selection.json',selection_sha256)
    design = b.load(design_path)
    selection = b.load(selection_path)
    selected = b.load(execution/'selection_lock.json')
    data = b.load(execution/'data_lock.json')
    fit = b.load(execution/'fit_lock.json')
    issued = b.load(execution/'selection_issuance_lock.json')
    assert selected['selection']['sha256'] == selection_sha256
    assert selected['design_lock']['sha256'] == design_sha256
    assert selected['selected_candidate'] == 'gap_hgb' and selected['promotion'] is False
    assert fit['external_selection_labels_read'] is False and issued['external_selection_labels_read'] is False
    assert data['external_labels_read'] is False and data['model_fits'] == 0
    assert data['original_issuances_per_lag'] == 39220
    for name,value in design['sources'].items():
        b.check(name,value)
    for name,value in design['inputs'].items():
        b.check(name,value)
    spec = b.load(b.record(design['specification']))
    protocol = b.load(b.record(design['verification_protocol']))
    assert b.record(design['verification_protocol']) == HERE/'protocol.json'
    statistical_source = b.check(protocol['statistics']['source_path'],protocol['statistics']['source_sha256'])
    assert design['sources'][protocol['statistics']['source_path']] == protocol['statistics']['source_sha256']
    assert protocol['raw_feature_sample']['snapshots'] == 24
    assert protocol['statistics']['resamples_each'] == 20000
    clock_items = [design,data,fit,issued,selected]
    clocks = [datetime.fromisoformat(v['closed_at_utc']) for v in clock_items]
    assert all(t.utcoffset() is not None for t in clocks) and clocks == sorted(clocks)
    with b.record(fit['models']).open('rb') as stream:
        bundle = pickle.load(stream)
    with b.record(design['parent']['models']).open('rb') as stream:
        old_bundle = pickle.load(stream)
    assert fit['fit_summary'] == bundle['fit_summary']
    summary = bundle['fit_summary']
    assert summary['fits'] == 2 and summary['base_refits'] == 0
    assert summary['rows_per_fit'] == 18363 and summary['fit_year'] == 2022
    base_columns = list(bundle['base_features'])
    gap_columns = ['gap__'+name for name in spec['features']['feature_order']]
    assert len(base_columns) == 80 and len(gap_columns) == 45
    assert bundle['gap_features'] == spec['features']['feature_order']
    expected_parameters = {k:spec['models'][k] for k in ('loss','max_iter','max_leaf_nodes',
        'learning_rate','min_samples_leaf','l2_regularization','early_stopping','random_state')}
    for name,stored in bundle['models'].items():
        assert name in NAMES and stored['kind'] == 'hgb'
        assert stored['features'] == (base_columns if name == 'base_hgb' else base_columns+gap_columns)
        equal(stored['model'].get_params(),expected_parameters,name='model_parameters')
        assert stored['model'].n_iter_ == 150
    training_labels = labels_for(b,design['parent']['years']['2022']['labels'])
    training_ids = [r['issuance_id'] for r in training_labels if r['outcome_status'] == 'matched']
    assert len(training_ids) == 18363
    import hashlib
    assert hashlib.sha256(json.dumps(training_ids,separators=(',',':')).encode()).hexdigest() == summary['issuance_ids_sha256']
    label_rows = labels_for(b,design['parent']['years']['2023']['labels'])
    by_id = {r['issuance_id']:r for r in label_rows}
    assert len(by_id) == len(label_rows) == 20432
    original = []
    for item in design['parent']['years']['2023']['original_references']:
        original.extend(rows(b.record(item)))
    assert [r['issuance_id'] for r in original] == [r['issuance_id'] for r in label_rows]
    year_closures = {year:b.load(b.record(data['years'][str(year)]['feature_closure'])) for year in (2022,2023)}
    metrics,comparisons,populations = {},{},{}
    total_forecasts = 0
    full_frames = {}
    for lag in LAGS:
        feature_rows = []
        for event in year_closures[2023]['events']:
            feature_rows.extend(rows(b.record(event['ledgers'][str(lag)])))
        assert len(feature_rows) == len(original)
        for row,source in zip(feature_rows,original):
            assert not (set(row)&TARGETS)
            assert {k:v for k,v in row.items() if k not in ADDED} == source
            assert row['original_row_sha256'] == digest(source)
            assert row['gap_lag_seconds'] == lag and row['gap_cutoff_ns'] == row['issued_at_ns']-lag*10**9
        frame = pd.DataFrame(feature_rows)
        full_frames[lag] = feature_rows
        saved = rows(b.record(issued['forecasts'][str(lag)]))
        previous = rows(b.record(design['parent']['reference_forecasts'][str(lag)]))
        assert len(saved) == len(previous) == len(frame) == 20432
        for row,prior,source in zip(saved,previous,feature_rows):
            assert row['issuance_id'] == prior['issuance_id'] == source['issuance_id']
            assert row['feature_row_sha256'] == digest(source)
            assert row['lag_seconds'] == prior['lag_seconds'] == lag
            assert row['gap_supported'] is source['gap_supported']
        predicted = {name:np.array([r['predictions'][name] for r in saved],dtype=np.float64) for name in NAMES}
        baseline = np.array([r['predictions']['base_hgb'] for r in previous],dtype=np.float64)
        assert bits(predicted['base_hgb'],baseline)
        support = np.array([r['gap_supported'] for r in feature_rows],dtype=bool)
        values = np.array([r['gap_values'] for r in feature_rows],dtype=float)
        assert values.shape == (20432,45) and not np.isinf(values).any() and np.isfinite(values[:,12:]).all()
        assert np.array_equal(values[:,39],support.astype(float))
        matrix = pd.concat([frame[base_columns],pd.DataFrame(values,columns=gap_columns,index=frame.index)],axis=1)
        anchor = frame.forecast_naive_seconds.to_numpy(float)
        with threadpool_limits(limits=1):
            for name in NAMES:
                x = frame[base_columns] if name == 'base_hgb' else matrix.copy()
                if name == 'quality_hgb':
                    x.loc[:,gap_columns[:12]] = 0.0
                replay = anchor+np.clip(bundle['models'][name]['model'].predict(x),-3,3)
                if name != 'base_hgb':
                    replay[~support] = baseline[~support]
                assert bits(replay,predicted[name]), 'Saved HGB replay mismatch: '+name
                assert np.isfinite(replay).all() and (replay > 0).all()
                total_forecasts += len(replay)
            old_base = anchor+np.clip(old_bundle['models']['base_hgb']['model'].predict(frame[base_columns]),-3,3)
            assert bits(old_base,baseline), 'Original baseline estimator changed'
        matched = np.array([by_id[r['issuance_id']]['outcome_status'] == 'matched' for r in feature_rows])
        assert int(matched.sum()) == 20007
        for row in label_rows:
            if row['outcome_status'] == 'unmatched':
                assert all(row[k] is None for k in TARGETS-{'outcome_status'} if k in row)
        events = frame.event_key.to_numpy()[matched]
        y = np.array([by_id[r['issuance_id']]['lap_time_seconds'] for r in feature_rows],dtype=float)[matched]
        assert np.isfinite(y).all() and len(np.unique(events)) == 22
        metrics[str(lag)] = {}
        for name in NAMES:
            errors = np.abs(predicted[name][matched]-y)
            per_event = [{'event_key':int(e),'rows':int((events == e).sum()),'mae_seconds':float(errors[events == e].mean())} for e in np.unique(events)]
            metrics[str(lag)][name] = {'rows':20007,'events':22,'event_mae_seconds':float(np.mean([r['mae_seconds'] for r in per_event])),
                                     'row_mae_seconds':float(errors.mean()),'per_event':per_event}
        comparisons[str(lag)] = {name:independent_comparison(events,y,predicted['gap_hgb'][matched],predicted[name][matched]) for name in ('base_hgb','quality_hgb')}
        populations[str(lag)] = {'rows_all':20432,'rows_matched':20007,'rows_unmatched_retained':425,
            'supported_rows_all':int(support.sum()),'fallback_rows_all':int((~support).sum()),
            'supported_rows_matched':int((support&matched).sum()),'fallback_rows_matched':int((~support&matched).sum())}
    checks,advances = gates(comparisons)
    reported = selection['summary']
    for name,expected in (('metrics',metrics),('comparisons',comparisons),('gate_checks',checks),('coverage',populations)):
        equal(reported[name],expected,name=name,tolerance=1e-10)
    assert reported['advances_to_later_evaluation'] is advances and selected['advances_to_later_evaluation'] is advances
    samples = []
    max_error = 0.0
    for event_key in protocol['raw_feature_sample']['event_keys']:
        event = next(e for e in year_closures[event_key//100]['events'] if e['event_key'] == event_key)
        for lag in LAGS:
            feature_rows = rows(b.record(event['ledgers'][str(lag)]))
            indices = [0,len(feature_rows)//2,len(feature_rows)-1]
            replay = PrefixReplay(b.record(event['source']['stream']))
            try:
                for index in indices:
                    saved = feature_rows[index]
                    actual = replay.reconstruct(saved['driver_id'],cutoff_ns=saved['gap_cutoff_ns'])
                    a,c = np.array(actual['values']),np.array(saved['gap_values'],dtype=float)
                    np.testing.assert_allclose(a,c,rtol=1e-9,atol=1e-9,equal_nan=True)
                    assert actual['supported'] is saved['gap_supported']
                    provenance = saved['gap_provenance']
                    assert actual['snapshot'] == provenance['pilot_snapshot']
                    recorded_history = [(r['sequence'],r['available_ns']//1_000_000,r['seconds']) for r in provenance['retained_interval_observations']]
                    assert actual['history'] == recorded_history
                    finite = np.isfinite(a)
                    error = float(np.max(np.abs(a[finite]-c[finite])))
                    max_error = max(max_error,error)
                    samples.append({'event_key':event_key,'lag_seconds':lag,'issuance_id':saved['issuance_id'],
                                    'support':saved['gap_supported'],'history_rows':len(recorded_history),'max_absolute_error':error})
            finally:
                replay.close()
    assert len(samples) == 24 and total_forecasts == 20432*3*2
    for path,value in list(b.checked.items()):
        assert sha(path) == value['sha256'], 'Input changed during independent verification'
    assert not list(execution.glob('*_failure.json')), 'Late failed execution cannot be certified'
    return {'status':'passed','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'design_sha256':design_sha256,'selection_sha256':selection_sha256,'protocol_sha256':sha(HERE/'protocol.json'),
        'model_fits':0,'saved_hgb_forecast_values_replayed':total_forecasts,
        'recomputed':{'metrics':metrics,'comparisons':comparisons,'gate_checks':checks,'coverage':populations,
                      'advances_to_later_evaluation':advances},
        'raw_prefix_snapshots':samples,'max_feature_absolute_error':max_error,
        'bindings':{str(path.relative_to(ROOT)):value for path,value in b.checked.items()},
        'source_files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(HERE.iterdir())|{statistical_source}) if p.suffix in ('.py','.json')},
        'limits':protocol['limits']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution',type=Path,required=True)
    parser.add_argument('--design-sha256',required=True)
    parser.add_argument('--selection-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Verification output is exclusive')
    try:
        with threadpool_limits(limits=1):
            result = verify(args.execution,design_sha256=args.design_sha256,selection_sha256=args.selection_sha256)
    except Exception as exc:
        write(args.out,{'status':'failed','error':str(exc),'traceback':traceback.format_exc()})
        raise
    write(args.out,result)
    print(json.dumps({'status':result['status'],'result_sha256':sha(args.out),
        'forecasts_replayed':result['saved_hgb_forecast_values_replayed'],'snapshots':len(result['raw_prefix_snapshots'])}),flush=True)


if __name__ == '__main__':
    main()
