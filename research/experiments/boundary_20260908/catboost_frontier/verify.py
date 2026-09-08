"""Replay the closed CatBoost screen without importing or fitting CatBoost.

Suggested commit: research(f1-live): verify exported symmetric-tree predictions
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.experiments.boundary_20260908.telemetry_verification_v2.verify import Bindings, digest, equal, read, rows, sha
from .independent import json_predict, comparison

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
NAMES = ('plain','ordered')


def array_hash(value): return hashlib.sha256(np.ascontiguousarray(value,dtype='<f8').tobytes()).hexdigest()
def bits(a,b): return np.array_equal(np.asarray(a,dtype=np.float64).view(np.uint64),np.asarray(b,dtype=np.float64).view(np.uint64))


def original(bindings, lock, year):
    result = []
    for item in lock['parent']['years'][str(year)]['original_references']:
        result.extend(rows(bindings.record(item)))
    assert len(result) == lock['parent']['years'][str(year)]['rows']
    return result


def outcomes(bindings, lock, year, originals):
    closed = bindings.load(bindings.record(lock['parent']['years'][str(year)]['labels']))
    result = []
    for item in closed['events']: result.extend(rows(bindings.record(item)))
    assert len(result) == len(originals) == closed['all_issuances']
    for label, source in zip(result,originals):
        for key in ('event_key','driver_id','issued_after_lap_number','issued_at_timestamp','issuance_id','issued_at_ns','year'):
            assert label[key] == source[key]
        assert label['outcome_status'] in ('matched','unmatched')
        if label['outcome_status'] == 'matched':
            assert label['target_at_ns'] > source['issued_at_ns']
            assert label['target_lap_number'] > source['issued_after_lap_number']
            assert np.isfinite(label['lap_time_seconds']) and label['lap_time_seconds'] > 0
        else:
            assert all(v is None for k,v in label.items() if k.startswith('target_') or k == 'lap_time_seconds')
    assert sum(r['outcome_status'] == 'matched' for r in result) == closed['matched']
    return result


def verify(execution,design_sha256,selection_sha256):
    assert not any(n == 'catboost' or n.startswith('catboost.') for n in sys.modules)
    execution = (ROOT/Path(execution)).resolve()
    assert (execution/'selection_lock.json').is_file() and not list(execution.glob('*_failure.json'))
    b = Bindings(ROOT)
    design = b.load(b.check(execution/'design_lock.json',design_sha256))
    selection = b.load(b.check(execution/'selection.json',selection_sha256))
    selected = b.load(execution/'selection_lock.json')
    fit = b.load(execution/'fit_lock.json');forecast = b.load(execution/'forecast_lock.json')
    spec = b.load(b.record(design['specification']))
    for path,value in {**design['sources'],**design['inputs']}.items(): b.check(path,value)
    assert selected['selection']['sha256'] == selection_sha256
    assert fit['fits'] == 2 and fit['base_refits'] == 0 and fit['rows_per_fit'] == 18363
    assert fit['external_selection_labels_read'] is False and forecast['external_selection_labels_read'] is False
    assert forecast['rows'] == 20432 and forecast['fit_lock']['sha256'] == sha(execution/'fit_lock.json')
    clocks = [datetime.fromisoformat(d['closed_at_utc']) for d in (design,fit,forecast,selected)]
    assert all(t.utcoffset() is not None for t in clocks) and clocks == sorted(clocks)
    exports = {name:b.load(b.record(fit['models'][name]['json'])) for name in NAMES}
    columns = [r['feature_id'] for r in sorted(exports['plain']['features_info']['float_features'],key=lambda r:r['flat_feature_index'])]
    assert len(columns) == 80 and len(set(columns)) == 80
    for name,payload in exports.items():
        assert [r['feature_id'] for r in sorted(payload['features_info']['float_features'],key=lambda r:r['flat_feature_index'])] == columns
        assert len(payload['oblivious_trees']) == fit['models'][name]['trees'] == 1000
        assert all(len(t['splits'] or []) <= 6 for t in payload['oblivious_trees'])
        expected = {**spec['models']['parameters'],**spec['models']['variant_parameters'][name]}
        flat = payload['model_info']['params']['flat_params']
        # Pinned CatBoost1.2.10 exports constructor verbose=False as integer0.
        assert type(flat['verbose']) is int and flat['verbose'] == 0
        expected['verbose'] = 0
        equal(flat,expected,name='exported_parameters',tolerance=1e-7)
    train = original(b,design,2022);train_y = outcomes(b,design,2022,train)
    training = [(r,y) for r,y in zip(train,train_y) if y['outcome_status'] == 'matched']
    assert len(training) == 18363
    assert digest([r['issuance_id'] for r,y in training]) == fit['training_ids_sha256']
    matrix = np.array([[r[c] for c in columns] for r,y in training],float)
    target = np.clip([y['lap_time_seconds']-r['forecast_naive_seconds'] for r,y in training],-5,5)
    events = np.array([r['event_key'] for r,y in training])
    unique,counts = np.unique(events,return_counts=True);assert len(unique) == 22
    weights = np.array([len(training)/(22*counts[np.flatnonzero(unique == e)[0]]) for e in events])
    assert array_hash(matrix) == fit['training_matrix_sha256'] and array_hash(target) == fit['target_sha256']
    assert array_hash(weights) == fit['weights_sha256']
    sources = original(b,design,2023);labels = outcomes(b,design,2023,sources)
    saved = rows(b.record(forecast['forecasts']));base = rows(b.record(design['parent']['reference_forecasts']))
    assert len(saved) == len(base) == len(sources) == 20432
    for row,prior,source in zip(saved,base,sources):
        assert row['issuance_id'] == prior['issuance_id'] == source['issuance_id']
        assert row['original_row_sha256'] == digest(source)
        for key in ('event_key','driver_id','issued_after_lap_number','issued_at_timestamp','issued_at_ns'):
            assert row[key] == prior[key] == source[key]
    x = np.array([[r[c] for c in columns] for r in sources],float)
    anchor = np.array([r['forecast_naive_seconds'] for r in sources])
    predictions = {name:np.array([r['predictions'][name] for r in saved]) for name in (*NAMES,'base_hgb')}
    prior = np.array([r['predictions']['base_hgb'] for r in base])
    assert bits(prior,predictions['base_hgb'])
    errors = {}
    for name in NAMES:
        actual = anchor+np.clip(json_predict(exports[name],x),-3,3)
        np.testing.assert_allclose(actual,predictions[name],rtol=1e-12,atol=1e-10)
        errors[name] = float(np.max(np.abs(actual-predictions[name])))
    with b.record(design['parent']['base_models']).open('rb') as f: base_model = pickle.load(f)['models']['base_hgb']
    assert base_model['features'] == columns
    with threadpool_limits(limits=1): replay_base = anchor+np.clip(base_model['model'].predict(pd.DataFrame(x,columns=columns)),-3,3)
    assert bits(replay_base,prior)
    matched = np.array([y['outcome_status'] == 'matched' for y in labels])
    assert matched.sum() == 20007
    y = np.array([r['lap_time_seconds'] for r in labels],float)[matched]
    events = np.array([r['event_key'] for r in sources])[matched]
    assert tuple(np.unique(events)) == tuple(range(202301,202323))
    metrics = {}
    for name,p in predictions.items():
        assert np.isfinite(p).all() and (p > 0).all()
        loss = np.abs(p[matched]-y)
        metrics[name] = {'rows':20007,'events':22,'row_mae_seconds':float(loss.mean()),
            'event_mae_seconds':float(np.mean([loss[events == e].mean() for e in np.unique(events)]))}
    comparisons = {name:comparison(events,y,predictions[name][matched],prior[matched]) for name in NAMES}
    checks = {name:{'minimum_relative_gain':c['relative_reduction'] is not None and c['relative_reduction'] >= .01,
        'event_ci_upper_negative':c['event_ci95'][1] < 0,'block3_ci_upper_negative':c['block3_ci95'][1] < 0,
        'all_leave_one_event_out_negative':c['loo_max_delta'] is not None and c['loo_max_delta'] < 0} for name,c in comparisons.items()}
    winner = min(NAMES,key=lambda n:metrics[n]['event_mae_seconds']);advance = all(checks[winner].values())
    summary = selection['summary'];equal(summary['metrics'],metrics,name='metrics',tolerance=1e-10)
    for name,c in comparisons.items():
        equal(summary['comparisons'][name],{k:v for k,v in c.items() if k not in ('candidate_row_mae','baseline_row_mae')},name='comparison',tolerance=1e-10)
    equal(summary['checks'],checks,name='checks')
    assert summary['winner'] == selected['winner'] == winner
    assert summary['advances_to_later_evaluation'] is selected['advances_to_later_evaluation'] is advance
    assert summary['promotion'] is selected['promotion'] is False
    for path,item in b.checked.items(): assert sha(path) == item['sha256'], 'Input changed during verification'
    assert not list(execution.glob('*_failure.json')), 'Late execution failure'
    assert not any(n == 'catboost' or n.startswith('catboost.') for n in sys.modules)
    return {'status':'passed','completed_at_utc':datetime.now(timezone.utc).isoformat(),'design_sha256':design_sha256,
        'selection_sha256':selection_sha256,'model_fits':0,'catboost_imported':False,
        'candidate_json_predictions_replayed':40864,'saved_baseline_predictions_replayed':20432,
        'maximum_absolute_prediction_error':errors,'recomputed':{'metrics':metrics,'comparisons':comparisons,'checks':checks,
        'winner':winner,'advances_to_later_evaluation':advance},
        'bindings':{str(path.relative_to(ROOT)):item for path,item in b.checked.items()},
        'limits':['No optimizer refit; native saved forecasts are checked against independent JSON arithmetic.',
                  'The original 80 features and labels are unchanged hash-bound inputs; raw feature extraction is not repeated.',
                  'Previously exposed seasons do not establish prospective performance.']}


def write(path,value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True)
    p.add_argument('--design-sha256',required=True);p.add_argument('--selection-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    if args.out.exists(): raise FileExistsError('Verification output is immutable')
    try:
        with threadpool_limits(limits=1): result = verify(args.execution,args.design_sha256,args.selection_sha256)
    except Exception as exc:
        write(args.out,{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});raise
    write(args.out,result);print(json.dumps({'status':result['status'],'sha256':sha(args.out),'winner':result['recomputed']['winner']}),flush=True)


if __name__ == '__main__': main()
