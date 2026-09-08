"""Frozen distributional experiment on unchanged original HGB issuances."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle
import platform
import numpy as np
import pandas as pd
import scipy, sklearn
from threadpoolctl import threadpool_limits
from research.experiments.boundary_20260908.online_residual import run as mechanics
from research.experiments.boundary_20260908.distributional.execution import model

ROOT=mechanics.ROOT
HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'artifacts/research/boundary_20260908/distributional'
OUT=ARCHIVE/'execution'
OLD=mechanics.OLD
SPEC=HERE.parent/'specification.json'
REFS=('global_empirical','stratified_empirical')
sha,write=mechanics.sha,mechanics.write


def now():return datetime.now(timezone.utc).isoformat()
def digest(x):
    import hashlib
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sources():
    paths=[*sorted(HERE.glob('*.py')),SPEC,Path(mechanics.__file__),Path(mechanics.original.__file__),mechanics.original.BASE_PATH]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def specification():
    spec=json.loads(SPEC.read_text());lock=json.loads((ARCHIVE/'design_lock.json').read_text())
    assert lock['spec_sha256']==sha(SPEC)
    for path,value in lock['base_input_artifacts'].items():assert sha(ROOT/path)==value
    execution=OUT/'execution_lock.json'
    if execution.exists():
        frozen=json.loads(execution.read_text())
        assert frozen['source_sha256']==digest(sources())
        for path,value in frozen['archived_files'].items():assert sha(ROOT/path)==value
    return spec


def validate_manifest(items):
    if not isinstance(items,list) or not items:raise ValueError('Empty input manifest')
    seen=set()
    for item in items:
        if not isinstance(item,dict) or not {'event_key','path','sha256'}<=item.keys():
            raise ValueError('Expected flat event_key/path/sha256 input manifest')
        if item['event_key'] in seen:raise ValueError('Duplicate manifest event')
        seen.add(item['event_key'])
        if sha(ROOT/item['path'])!=item['sha256']:raise ValueError('Input hash mismatch: '+item['path'])
    return len(items)


def validate_population(frame,features,years):
    if set(frame.year.unique())!=set(years):raise ValueError('Unexpected population years')
    if not np.all(frame.year.to_numpy()==frame.event_key.to_numpy()//100):raise ValueError('Event/year mismatch')
    if not (frame.target_timestamp>frame.issued_at_timestamp).all():raise ValueError('Target is not strictly later')
    if len(features)!=80 or any(c.startswith('target_') or c=='lap_time_seconds' for c in features):raise ValueError('Invalid causal feature set')
    if not np.isfinite(frame[features].to_numpy()).all():raise ValueError('Nonfinite features')
    keys=['event_key','driver_id','issued_after_lap_number','issued_at_timestamp']
    if frame.duplicated(keys).any():raise ValueError('Duplicate issuance')
    if not np.isfinite(frame.lap_time_seconds).all():raise ValueError('Nonfinite target')


def save_pickle(name,obj):
    with (OUT/name).open('wb') as f:pickle.dump(obj,f)


def coverage_pass(metric,spec):
    return all(metric['coverage'][str(level)]>=level-.05 for level in [.8,.9,.95])


def select_candidate(metrics,spec):
    candidate=min(spec['candidates'],key=lambda name:(metrics[name]['wis'],name))
    gains={ref:1-metrics[candidate]['wis']/metrics[ref]['wis'] for ref in REFS}
    advanced=all(v>=.01 for v in gains.values()) and coverage_pass(metrics[candidate],spec)
    return candidate,gains,bool(advanced)


def summary(frame,predictions,candidate,spec):
    metrics={name:model.metrics(frame,q,spec) for name,q in predictions.items()}
    return {'models':metrics,'paired':{ref:model.paired(frame,predictions[candidate],predictions[ref],spec) for ref in REFS},
            'relative_wis_gain':{ref:float(1-metrics[candidate]['wis']/metrics[ref]['wis']) for ref in REFS}}


def begin(phase):
    if not (OUT/'execution_lock.json').exists():raise ValueError('Freeze tested execution source before fitting')
    path=OUT/f'{phase}_attempt.json'
    if path.exists():raise FileExistsError('Attempt exists; retain it and use a fresh output directory for retry')
    write(path,{'started_at_utc':now(),'source_files':sources(),'source_sha256':digest(sources()),'spec_sha256':sha(SPEC),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'threads':1}})


def discover():
    spec=specification()
    if (OUT/'selection.json').exists():raise FileExistsError('Completed selection is frozen')
    begin('selection')
    discovery=pd.read_pickle(OLD/'discovery_data.pkl')
    inherited=json.loads((OLD/'selection.json').read_text());features=inherited['features']
    validate_manifest(inherited['input_manifest'])
    validate_population(discovery,features,[2022,2023])
    with threadpool_limits(limits=1):
        train,pt,folds=mechanics.crossfit(discovery,features)
        base=mechanics.original.fit_model(discovery.loc[discovery.year.eq(2022)],mechanics.BASE_CONFIG,features)
        validation=discovery.loc[discovery.year.eq(2023)].reset_index(drop=True)
        pv=mechanics.original.predict_model(validation,base)
        empirical=model.fit_empirical(train,pt,spec)
        predictions=model.empirical_predict(validation,pv,empirical)
        candidates={};projection={}
        for name,config in spec['candidates'].items():
            candidates[name]=model.fit_quantiles(train,pt,features,config,spec)
            predictions[name],projection[name]=model.quantile_predict(validation,pv,candidates[name],spec)
            print('candidate_fitted',name,flush=True)
    metrics={name:model.metrics(validation,q,spec) for name,q in predictions.items()}
    candidate,gains,advanced=select_candidate(metrics,spec)
    result={'selected':candidate,'all_selection_metrics':metrics,'relative_wis_gain':gains,'advancement_passed':bool(advanced),
        'coverage_gate_passed':coverage_pass(metrics[candidate],spec),'paired':{ref:model.paired(validation,predictions[candidate],predictions[ref],spec) for ref in REFS},
        'crossfit_blocks':folds,'features':features,'projection_diagnostics':projection,'selection_rows':len(validation),
        'source_files':sources(),'source_sha256':digest(sources()),'design_lock_sha256':sha(ARCHIVE/'design_lock.json'),'execution_lock_sha256':sha(OUT/'execution_lock.json'),'completed_at_utc':now()}
    save_pickle('selection_bundle.pkl',{'base':base,'empirical':empirical,'candidates':candidates})
    save_pickle('selection_forecasts.pkl',{'frame':validation,'point':pv,'quantiles':predictions})
    save_pickle('crossfit_residual_training.pkl',{'frame':train,'point':pt,'folds':folds})
    result['artifacts']={name:sha(OUT/name) for name in ['selection_bundle.pkl','selection_forecasts.pkl','crossfit_residual_training.pkl']}
    write(OUT/'selection.json',result)
    write(OUT/'selection_lock.json',{'selected':candidate,'advancement_passed':bool(advanced),'selection_sha256':sha(OUT/'selection.json'),'source_sha256':digest(sources()),'locked_at_utc':now(),'new_transfer_distributions_inspected':False})
    if advanced:
        final_frame=pd.concat([train,validation],ignore_index=True);final_point=np.concatenate([pt,pv])
        with threadpool_limits(limits=1):
            final={'candidate':model.fit_quantiles(final_frame,final_point,features,spec['candidates'][candidate],spec),
                   'empirical':model.fit_empirical(final_frame,final_point,spec)}
        save_pickle('final_distribution_bundle.pkl',final)
        write(OUT/'fit_lock.json',{'frozen_at_utc':now(),'selection_sha256':sha(OUT/'selection.json'),'model_sha256':sha(OUT/'final_distribution_bundle.pkl'),
            'base_production_model_sha256':sha(OLD/'candidate/model.pkl'),'fit_events':sorted(int(v) for v in final_frame.event_key.unique()),
            'fit_rows':len(final_frame),'source_sha256':digest(sources())})
    else:
        write(OUT/'results.json',{'status':'rejected_on_2023_distributional_screen','selected':candidate,'transfer_evaluated':False,
            'substantial_gate_passed':False,'promotion':False,'selection_sha256':sha(OUT/'selection.json'),
            'reason':'Requires at least1percent WIS improvement over both empirical references and no more than5pp undercoverage at80/90/95.','source_sha256':digest(sources())})
    print(json.dumps({'selected':candidate,'advancement_passed':bool(advanced),'relative_wis_gain':gains,'metrics':{name:{k:v for k,v in m.items() if k!='per_event'} for name,m in metrics.items()}},indent=2),flush=True)


def transfer():
    spec=specification()
    if (OUT/'results.json').exists():raise FileExistsError('Completed/rejected result is frozen')
    selection=json.loads((OUT/'selection.json').read_text());lock=json.loads((OUT/'fit_lock.json').read_text())
    selection_lock=json.loads((OUT/'selection_lock.json').read_text())
    assert selection_lock['advancement_passed'] and selection_lock['selection_sha256']==sha(OUT/'selection.json')
    assert selection_lock['source_sha256']==digest(sources())
    assert selection['advancement_passed'] and lock['selection_sha256']==sha(OUT/'selection.json')
    assert lock['model_sha256']==sha(OUT/'final_distribution_bundle.pkl') and lock['source_sha256']==digest(sources())
    assert lock['base_production_model_sha256']==sha(OLD/'candidate/model.pkl')
    begin('transfer')
    with (OUT/'final_distribution_bundle.pkl').open('rb') as f:bundle=pickle.load(f)
    with (OLD/'candidate/model.pkl').open('rb') as f:base=pickle.load(f)['model']
    source=OLD/'corrected_input_contract/transfer_data_and_forecasts.pkl'
    frame=pd.read_pickle(source)
    inherited=json.loads((OLD/'corrected_input_contract/results.json').read_text())
    validate_manifest(inherited['input_manifest'])
    validate_population(frame,selection['features'],[2024,2025,2026])
    with threadpool_limits(limits=1):
        point=mechanics.original.predict_model(frame,base)
        np.testing.assert_array_equal(point,frame.prediction_hgb_l15_i150.to_numpy())
        predictions=model.empirical_predict(frame,point,bundle['empirical'])
        candidate=selection['selected'];predictions[candidate],projection=model.quantile_predict(frame,point,bundle['candidate'],spec)
    assert len(frame)==55757 and frame.event_key.nunique()==61
    for q in predictions.values():np.testing.assert_array_equal(q[:,4],point)
    summaries={name:summary(frame.loc[mask],{k:v[mask] for k,v in predictions.items()},candidate,spec)
        for name,mask in [('2024',frame.year.eq(2024)),('2025',frame.year.eq(2025)),('2024_2025',frame.year.isin([2024,2025])),
                         ('2026',frame.year.eq(2026)),('2026_recent',frame.year.eq(2026)&frame.event_key.ge(202610))]}
    historical=summaries['2024_2025'];gates={}
    for ref in REFS:
        gates[f'historical_10pct_vs_{ref}']=historical['relative_wis_gain'][ref]>=.10
        gates[f'historical_block3_upper_negative_vs_{ref}']=historical['paired'][ref]['block3_ci95'][1]<0
        gates[f'each_year_wis_improves_vs_{ref}']=all(summaries[y]['relative_wis_gain'][ref]>0 for y in ['2024','2025','2026'])
    gates['coverage_each_year']=all(coverage_pass(summaries[y]['models'][candidate],spec)
        and all(summaries[y]['models'][candidate]['coverage'][str(q)]>=summaries[y]['models']['stratified_empirical']['coverage'][str(q)]-.03 for q in [.8,.9,.95]) for y in ['2024','2025','2026'])
    gates['point_unchanged']=True
    save_pickle('transfer_forecasts.pkl',{'frame':frame,'point':point,'quantiles':predictions})
    write(OUT/'results.json',{'status':'completed_distributional_transfer','selected':candidate,'transfer_evaluated':True,'summaries':summaries,'gates':gates,
        'substantial_gate_passed':all(gates.values()),'promotion':False,'point_mae_improvement_claimed':False,'source_sha256':digest(sources()),
        'selection_sha256':sha(OUT/'selection.json'),'fit_lock_sha256':sha(OUT/'fit_lock.json'),'forecasts_sha256':sha(OUT/'transfer_forecasts.pkl'),
        'original_forecasts_sha256':sha(source),'projection_diagnostics':projection,'completed_at_utc':now()})
    print(json.dumps({'substantial_gate_passed':all(gates.values()),'gates':gates,'summaries':{name:{'gains':s['relative_wis_gain'],'candidate':{k:v for k,v in s['models'][candidate].items() if k!='per_event'}} for name,s in summaries.items()}},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['discover','transfer']);args=parser.parse_args()
    {'discover':discover,'transfer':transfer}[args.phase]()
