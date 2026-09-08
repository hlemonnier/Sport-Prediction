"""Locked original-horizon gap experiment; no historical execution before review.

Suggested commit: research(f1-live): execute causal rival-gap comparison
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import pickle
import platform
import resource
import subprocess
import sys
import traceback

for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name]='1'

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from research.experiments.boundary_20260908.telemetry import data as old_data
from research.experiments.boundary_20260908.telemetry import run as old_run
from research.experiments.boundary_20260908.rival_gap_pilot_v2 import pilot

ROOT=old_data.ROOT
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/boundary_20260908/rival_gap_forecast'
PARENT=ROOT/'artifacts/research/boundary_20260908/telemetry/execution'
VERIFICATION=ROOT/'research/experiments/boundary_20260908/rival_gap_verification'
EVENTS=tuple(range(202201,202223))+tuple(range(202301,202323))
YEARS=(2022,2023)
LAGS=(2,0)
TARGET_COLUMNS=old_run.TARGET_COLUMNS
MODEL_NAMES=('base_hgb','gap_hgb','quality_hgb')
ADDED=('gap_values','gap_supported','gap_cutoff_ns','gap_lag_seconds','gap_provenance','original_row_sha256')


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return old_data._read_json(path)
def save(path,value):return old_data._write_json(path,value)
def record(path):return {'path':str(Path(path).resolve()),'sha256':old_data.sha(path)}
def progress(**value):print(json.dumps(value,allow_nan=False),flush=True)


def check(item):
    path=ROOT/item['path']
    if old_data.sha(path)!=item['sha256']:raise ValueError('Hash binding changed: '+str(path))
    if 'bytes' in item and path.stat().st_size!=item['bytes']:raise ValueError('Byte count changed: '+str(path))
    return path


def sources():
    from . import features
    paths={p for pattern in ('*.py','*.json','*.md') for p in HERE.glob(pattern)}
    paths.update(VERIFICATION/name for name in ('protocol.json','prefix.py','test_prefix.py'))
    statistics=read(VERIFICATION/'protocol.json')['statistics']
    statistical_source=ROOT/statistics['source_path']
    if old_data.sha(statistical_source)!=statistics['source_sha256']:
        raise ValueError('Independent statistical source changed')
    paths.add(statistical_source)
    # Every imported inherited module is in the closed telemetry source graph.
    inherited=read(PARENT/'design_lock.json')['sources']
    inherited.update(read(pilot.OUT/'design_lock.json')['source_files'])
    inherited.update(features.dependency_bindings())
    for name,digest in inherited.items():
        path=ROOT/name
        if old_data.sha(path)!=digest:raise ValueError('Inherited source changed: '+name)
        paths.add(path)
    return {str(p.relative_to(ROOT)):old_data.sha(p) for p in sorted(paths)}


def validate_spec(spec):
    from . import features
    required={
        'discovery':{'event_keys':list(EVENTS),'fit_year':2022,'selection_year':2023,
            'original_issuances':39220,'matched_rows':38370,'unmatched_retained':850,
            'fit_matched_rows':18363,'selection_issuances':20432,'selection_matched_rows':20007,'selection_unmatched_retained':425},
        'features':{'base_width':80,'gap_width':45,'primary_lag_seconds':2,'sensitivity_lag_seconds':0,
            'sensitivity_refit':False,'maximum_gap_age_seconds_at_cutoff':10,'maximum_quality_age_seconds':180},
        'models':{'names':list(MODEL_NAMES),'new_fits':2,'total_features':125,'loss':'absolute_error',
            'max_iter':150,'max_leaf_nodes':15,'learning_rate':.06,'min_samples_leaf':80,'l2_regularization':10,
            'early_stopping':False,'random_state':20260907,'residual_train_clip_seconds':5,'correction_clip_seconds':3,
            'base_selection_event_mae_seconds':.512080723333514},
        'selection_gate':{'candidate':'gap_hgb','references':['base_hgb','quality_hgb'],
            'minimum_relative_event_mae_reduction_each':.01,'event_ci_upper_negative':True,
            'circular_block3_ci_upper_negative':True,'all_leave_one_event_out_negative':True,
            'positive_zero_lag_gain_each':True,'promotion':False},
        'uncertainty':{'seed':20260907,'resamples_each':20000,'block_length':3},
        'execution':{'threads':1,'maximum_peak_rss_bytes':3*1024**3}}
    for section,items in required.items():
        for name,expected in items.items():
            actual=spec[section][name]
            if actual!=expected or (type(expected) is bool and type(actual) is not bool):
                raise ValueError('Frozen specification mismatch: '+section+'.'+name)
    if (tuple(features.FEATURE_NAMES)!=tuple(spec['features']['feature_order'])
            or tuple(features.CONTENT_INDICES)!=tuple(range(12))
            or tuple(features.QUALITY_INDICES)!=tuple(range(12,45))):raise ValueError('Gap feature layout changed')
    if ROOT/spec['inputs']['parent_execution']!=PARENT:raise ValueError('Parent execution changed')
    for name in ('timing_acquisition','pilot_result'):check(spec['inputs'][name])


def parent_graph(spec):
    """Read metadata and hash bound files only; no label or raw packet parsing."""
    graph={}
    def bind(item):
        path=check(item);graph[str(path)]=item['sha256'];return path
    paths={name:PARENT/name for name in ('design_lock.json','data_lock.json','fit_lock.json',
        'selection_issuance_lock.json','selection_lock.json','verification/result_v2.json')}
    for path in paths.values():bind(record(path))
    design,data,fit,issued,selected,verified=(read(paths[name]) for name in paths)
    declared=spec['inputs']
    if (old_data.sha(paths['data_lock.json'])!=declared['parent_data_lock_sha256']
            or old_data.sha(paths['verification/result_v2.json'])!=declared['parent_verification_sha256']
            or selected['selection']['sha256']!=declared['parent_selection_sha256']):raise ValueError('Declared parent changed')
    if verified['status']!='passed' or verified['selection_sha256']!=selected['selection']['sha256']:
        raise ValueError('Successful parent verification required')
    if (data['external_labels_attached'] is not False or data['model_fits']!=0
            or data['original_issuances_per_lag']!=39220 or check(data['design_lock'])!=paths['design_lock.json']
            or check(fit['data_lock'])!=paths['data_lock.json']):raise ValueError('Parent data/fit chronology invalid')
    if (issued['selection_labels_attached'] is not False or check(issued['fit_lock'])!=paths['fit_lock.json']
            or check(issued['data_lock'])!=paths['data_lock.json']):raise ValueError('Parent forecast closure invalid')
    if (fit['external_selection_labels_attached'] is not False or fit['fit_summary']['fit_year']!=2022
            or fit['fit_summary']['rows_per_fit']!=18363 or fit['fit_summary']['fit_latency_seconds']!=2):
        raise ValueError('Parent base must be fitted only on original 2022 rows')
    def old_bind(item):
        path=bind(item);expected=verified['bindings'].get(str(path.relative_to(ROOT)),{})
        if expected.get('sha256')!=item['sha256']:raise ValueError('Input is outside successful parent verification: '+str(path))
        return path
    old_bind(fit['models']);old_bind(selected['selection'])
    years={}
    for year,expected_rows,expected_matched in ((2022,18788,18363),(2023,20432,20007)):
        info=data['years'][str(year)];refs=info['original_references']
        if [r['event_key'] for r in refs]!=list(range(year*100+1,year*100+23)) or sum(r['rows'] for r in refs)!=expected_rows:
            raise ValueError('Original issuance inventory changed')
        for item in refs:old_bind(item)
        old_bind(info['feature_closure']);old_bind(info['input_validation'])
        label_item=record(PARENT/f'labels_{year}/label_closure.json');labels=read(old_bind(label_item))
        if year==2022 and label_item!=fit['training_labels']:raise ValueError('Parent training label closure changed')
        if (labels['feature_closure']!=info['feature_closure'] or labels['all_issuances']!=expected_rows
                or [r['event_key'] for r in labels['events']]!=[r['event_key'] for r in refs]
                or sum(r['matched'] for r in labels['events'])!=expected_matched):raise ValueError('Original label metadata changed')
        for ref,label in zip(refs,labels['events']):
            if label['rows']!=ref['rows'] or label['rows']!=label['matched']+label['unmatched']:raise ValueError('Label population differs')
            old_bind(label)
        years[str(year)]={'original_references':refs,'labels':label_item,'rows':expected_rows,'matched_rows':expected_matched}
    if set(issued['forecasts'])!={'0','2'}:raise ValueError('Both parent latency forecasts required')
    for item in issued['forecasts'].values():
        if item['rows']!=20432:raise ValueError('Parent forecast population changed')
        old_bind(item)
    acquired=read(bind(declared['timing_acquisition']));bind(declared['pilot_result'])
    if (acquired['status']!='complete_all_stream_jobs_recorded' or acquired['models_fitted']!=0
            or acquired['predictive_scores_computed']!=0):raise ValueError('Acquisition is not closed')
    sessions={r['event_key']:r for r in acquired['sessions']}
    if tuple(sorted(sessions))!=EVENTS:raise ValueError('Acquisition session population changed')
    streams=sorted((r for r in acquired['streams'] if r['stream']=='TimingData'),key=lambda r:r['event_key'])
    if tuple(r['event_key'] for r in streams)!=EVENTS:raise ValueError('Exactly 44 TimingData streams required')
    selected_streams=[]
    for stream in streams:
        if stream['status'] not in ('downloaded','reused_verified_pilot') or stream.get('source_status','downloaded')!='downloaded':
            raise ValueError('All 44 declared TimingData bodies must be available')
        if stream['session_path']!=sessions[stream['event_key']]['session_path']:raise ValueError('Wrong named race stream')
        item={'path':stream['decoded_body_path'],'sha256':stream['decoded_body_sha256'],'bytes':stream['decoded_body_bytes']}
        bind(item);selected_streams.append({'event_key':stream['event_key'],'session_path':stream['session_path'],
            'status':stream['status'],'stream':item})
    return graph,{'data_lock':record(paths['data_lock.json']),'fit_lock':record(paths['fit_lock.json']),
        'selection_lock':record(paths['selection_lock.json']),'verification':record(paths['verification/result_v2.json']),
        'years':years,'models':fit['models'],'reference_forecasts':issued['forecasts'],
        'timing_acquisition':declared['timing_acquisition'],'streams':selected_streams}


def resource_check():
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    if peak>3*1024**3:raise MemoryError('Peak RSS exceeds the frozen 3 GiB limit')
    return int(peak)


@contextmanager
def attempt(out,stage):
    out=Path(out);save(out/(stage+'_attempt.json'),{'stage':stage,'started_at_utc':now()})
    try:
        resource_check();yield;resource_check()
    except Exception as exc:
        save(out/(stage+'_failure.json'),{'stage':stage,'failed_at_utc':now(),'type':type(exc).__name__,
            'error':str(exc),'traceback':traceback.format_exc()});raise


def _freeze(out,review_path):
    out=Path(out)
    if (out/'design_lock.json').exists() or any(out.glob('*_failure.json')):raise FileExistsError('Existing execution is immutable')
    spec=read(HERE/'specification.json');validate_spec(spec);before=sources();review=read(review_path)
    if review.get('approved_for_execution_lock') is not True or review.get('source_files')!=before:
        raise ValueError('Exact-source independent review required')
    bindings,parent=parent_graph(spec)
    command=[sys.executable,'-m','pytest','-q','--import-mode=importlib','-p','no:cacheprovider',str(HERE)]
    tested=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    save(out/'pre_fit_tests.json',{'command':command,'exit_code':tested.returncode,'stdout':tested.stdout,
        'stderr':tested.stderr,'source_files':before,'completed_at_utc':now()})
    if tested.returncode or sources()!=before:raise ValueError('Tests failed or source changed during freeze')
    for path,digest in bindings.items():
        if old_data.sha(path)!=digest:raise ValueError('Input changed during freeze')
    save(out/'design_lock.json',{'closed_at_utc':now(),'sources':before,'inputs':bindings,'parent':parent,
        'specification':record(HERE/'specification.json'),'independent_review':record(review_path),
        'verification_protocol':record(VERIFICATION/'protocol.json'),
        'pre_fit_tests':record(out/'pre_fit_tests.json'),'historical_gap_builds_before_lock':0,'new_fits_before_lock':0,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
            'scipy':scipy.__version__,'sklearn':sklearn.__version__,'threads':1},
        'exposure':'Both seasons and old scores were already exposed. New forecast closure certifies execution order, not a fresh holdout.'})
    progress(stage='frozen',design_lock=record(out/'design_lock.json'),input_bindings=len(bindings))


def freeze(out,review_path):
    with attempt(out,'freeze'):_freeze(out,review_path)


def verify_design(out):
    out=Path(out);lock=read(out/'design_lock.json')
    if any(out.glob('*_failure.json')):raise ValueError('Preserved failed execution cannot advance')
    if sources()!=lock['sources']:raise ValueError('Source differs from reviewed design')
    for path,digest in lock['inputs'].items():
        if old_data.sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
    for key in ('specification','independent_review','pre_fit_tests','verification_protocol'):check(lock[key])
    spec=read(check(lock['specification']));validate_spec(spec)
    return lock,spec


def original_rows(item):
    rows=old_data._read_rows(check(item))
    if len(rows)!=item['rows'] or len({r['issuance_id'] for r in rows})!=len(rows):raise ValueError('Original issuance population changed')
    clocks=[]
    for row in rows:
        if any(k in row for k in (*TARGET_COLUMNS,*ADDED)) or any(k.startswith(('telemetry_','embedding_')) for k in row):
            raise ValueError('Original references must be target-free without telemetry additions')
        if row['event_key']!=item['event_key'] or row['year']!=item['event_key']//100 or type(row['issued_at_ns']) is not int:
            raise ValueError('Original event or exact clock metadata changed')
        if row['issuance_id']!=f"{row['event_key']}/{row['driver_id']}/{row['issued_after_lap_number']}/{row['issued_at_ns']}":
            raise ValueError('Original identity does not match its clock')
        if not isinstance(row['driver_id'],str) or any(k not in row for k in (*old_data.BASE_FEATURES,'forecast_naive_seconds')):
            raise ValueError('Original 80 features, driver and anchor required')
        clocks.append(row['issued_at_ns'])
    if clocks!=sorted(clocks):raise ValueError('Original issuance order must be chronological')
    return rows


def prepare_event(out,item,stream):
    from . import features
    rows=original_rows(item);entries={}
    for lag in LAGS:
        cursor=features.GapFeatureCursor(check(stream['stream']));built=[];supported=0
        try:
            for row in rows:
                cutoff=row['issued_at_ns']-lag*10**9;snapshot=cursor.query(row['driver_id'],cutoff_ns=cutoff)
                values=np.asarray(snapshot.gap_values,dtype=float)
                if values.shape!=(45,) or np.isinf(values).any() or type(snapshot.gap_supported) is not bool or snapshot.gap_cutoff_ns!=cutoff:
                    raise ValueError('Invalid gap feature snapshot')
                built.append({**row,'gap_values':values.tolist(),'gap_supported':snapshot.gap_supported,'gap_cutoff_ns':cutoff,
                    'gap_lag_seconds':lag,'gap_provenance':snapshot.provenance,'original_row_sha256':old_data.digest(row)})
                supported+=int(snapshot.gap_supported)
        finally:cursor.close()
        path=Path(out)/f"features_{item['event_key']//100}/{item['event_key']}_lag{lag}.jsonl"
        old_data._write_rows(path,built)
        entries[str(lag)]={**record(path),'rows':len(rows),'supported':supported,'fallback':len(rows)-supported}
    # This inspection may see later packets, but both immutable feature ledgers
    # already exist. It never determines eligibility or updates those features.
    inventory=pilot.inventory_stream(check(stream['stream']))
    return {'event_key':item['event_key'],'original':item,'source':stream,'ledgers':entries,'full_stream_inventory':inventory}


def prepare(out):
    out=Path(out);lock,spec=verify_design(out)
    with attempt(out,'prepare'):
        by_event={r['event_key']:r for r in lock['parent']['streams']};years={};count=0
        for year in YEARS:
            entries=[]
            for item in lock['parent']['years'][str(year)]['original_references']:
                entry=prepare_event(out,item,by_event[item['event_key']]);entries.append(entry);resource_check()
                progress(stage='gap_features_closed',event_key=item['event_key'],rows=item['rows'],
                    supported={lag:entry['ledgers'][lag]['supported'] for lag in ('2','0')})
            expected=lock['parent']['years'][str(year)]['rows']
            if sum(e['original']['rows'] for e in entries)!=expected:raise ValueError('Year issuance count changed')
            path=out/f'features_{year}/feature_closure.json'
            save(path,{'closed_at_utc':now(),'design_lock':record(out/'design_lock.json'),'events':entries,
                'issuances_per_lag':expected,'external_labels_read':False,'model_fits':0})
            years[str(year)]={'feature_closure':record(path)};count+=expected
        if count!=spec['discovery']['original_issuances']:raise ValueError('Original full population changed')
        verify_design(out)
        save(out/'data_lock.json',{'closed_at_utc':now(),'design_lock':record(out/'design_lock.json'),'years':years,
            'original_issuances_per_lag':count,'external_labels_read':False,'model_fits':0})
        progress(stage='prepared',events=44,original_issuances_per_lag=count,data_lock=record(out/'data_lock.json'))


def load_year(locked,year,lag):
    closure=read(check(locked['years'][str(year)]['feature_closure']));rows=[]
    if closure['design_lock']!=locked['design_lock'] or closure['external_labels_read'] is not False or closure['model_fits']!=0:
        raise ValueError('Year features did not close target-free')
    if [r['event_key'] for r in closure['events']]!=list(range(year*100+1,year*100+23)):raise ValueError('Year event population changed')
    for event in closure['events']:
        item=event['ledgers'][str(lag)];event_rows=old_data._read_rows(check(item));original=original_rows(event['original'])
        if len(event_rows)!=len(original) or len(event_rows)!=item['rows']:raise ValueError('Feature ledger population changed')
        for row,reference in zip(event_rows,original):
            if any(k in row for k in TARGET_COLUMNS):raise ValueError('Target entered feature ledger')
            if row['original_row_sha256']!=old_data.digest(reference) or old_data.digest({k:v for k,v in row.items() if k not in ADDED})!=old_data.digest(reference):
                raise ValueError('Original feature values changed')
            if row['gap_cutoff_ns']!=row['issued_at_ns']-lag*10**9 or row['gap_lag_seconds']!=lag:raise ValueError('Latency changed')
        rows.extend(event_rows)
    if len(rows)!=closure['issuances_per_lag']:raise ValueError('Loaded year population changed')
    return pd.DataFrame(rows)


def reference_year(lock,year):
    rows=[]
    for item in lock['parent']['years'][str(year)]['original_references']:rows.extend(original_rows(item))
    return pd.DataFrame(rows)


def load_saved_base(frame,item,lag):
    from . import models
    rows=old_data._read_rows(check(item))
    if len(rows)!=len(frame) or len(rows)!=item['rows']:raise ValueError('Saved base population changed')
    for saved,issued in zip(rows,frame.to_dict('records')):
        if any(saved[k]!=issued[k] for k in (*old_data.KEYS,'issuance_id','issued_at_ns')) or saved['lag_seconds']!=lag:
            raise ValueError('Saved base identity or latency changed')
    predictions=np.array([r['predictions']['base_hgb'] for r in rows],dtype=np.float64)
    if not np.isfinite(predictions).all() or (predictions<=0).any():raise ValueError('Saved baseline must be finite and positive')
    return models.SavedBaseForecasts(tuple(frame.issuance_id),tuple(int(v) for v in frame.gap_cutoff_ns),predictions,item['sha256'])


def forecast_rows(frame,predictions,lag):
    if set(predictions)!=set(MODEL_NAMES):raise ValueError('All three forecasts required')
    if any(k in frame for k in TARGET_COLUMNS):raise ValueError('Selection labels read before forecast closure')
    for value in predictions.values():
        if np.asarray(value).shape!=(len(frame),) or not np.isfinite(value).all() or (np.asarray(value)<=0).any():
            raise ValueError('Positive finite forecasts required for all original issuances')
    return [{**{k:row[k] for k in (*old_data.KEYS,'issuance_id','issued_at_ns','gap_supported','gap_cutoff_ns')},
        'lag_seconds':lag,'feature_row_sha256':old_data.digest(row),
        'predictions':{name:float(predictions[name][i]) for name in MODEL_NAMES}}
        for i,row in enumerate(frame.to_dict('records'))]


def load_forecasts(frame,item,lag):
    rows=old_data._read_rows(check(item))
    if len(rows)!=len(frame):raise ValueError('Forecasts lost original issuances')
    for saved,issued in zip(rows,frame.to_dict('records')):
        if saved['feature_row_sha256']!=old_data.digest(issued) or saved['issuance_id']!=issued['issuance_id'] or saved['lag_seconds']!=lag:
            raise ValueError('Saved forecast/feature binding changed')
    return {name:np.array([r['predictions'][name] for r in rows],dtype=np.float64) for name in MODEL_NAMES}


def select(out):
    from . import models,evaluate
    out=Path(out);lock,spec=verify_design(out)
    with attempt(out,'selection'):
        data_lock=read(out/'data_lock.json')
        if data_lock['design_lock']!=record(out/'design_lock.json') or data_lock['external_labels_read'] is not False or data_lock['model_fits']!=0:
            raise ValueError('All gap feature ledgers must close before supervision')
        train_frame=load_year(data_lock,2022,2);train_labels=lock['parent']['years']['2022']['labels']
        train_all=old_run.join_labels(train_frame,train_labels)
        training=train_all.loc[train_all.outcome_status.eq('matched')].copy()
        expected=old_run.join_labels(reference_year(lock,2022),train_labels)
        expected=expected.loc[expected.outcome_status.eq('matched')].copy()
        if len(training)!=spec['discovery']['fit_matched_rows']:raise ValueError('Training population changed')
        with check(lock['parent']['models']).open('rb') as f:old_bundle=pickle.load(f)
        base=old_bundle['models']['base_hgb']
        with threadpool_limits(limits=1):bundle=models.fit_models(training,expected_matched=expected,
            saved_base_model=base,saved_base_model_sha256=lock['parent']['models']['sha256'])
        resource_check()
        path=out/'models.pkl'
        with path.open('xb') as f:pickle.dump(bundle,f,protocol=pickle.HIGHEST_PROTOCOL)
        save(out/'fit_lock.json',{'closed_at_utc':now(),'data_lock':record(out/'data_lock.json'),'models':record(path),
            'training_labels':train_labels,'reused_base_bundle':lock['parent']['models'],
            'fit_summary':bundle['fit_summary'],'external_selection_labels_read':False})
        with path.open('rb') as f:bundle=pickle.load(f)
        frames={};refs={};forecasts={}
        expected_issued=reference_year(lock,2023)
        for lag in LAGS:
            frame=load_year(data_lock,2023,lag);old_run.assert_original_parity(frame,expected_issued)
            refs[lag]=load_saved_base(frame,lock['parent']['reference_forecasts'][str(lag)],lag)
            before=old_data.digest(frame.to_dict('records'))
            predictions=models.predict_models(bundle,frame,latency_seconds=lag,
                expected_issuances=expected_issued,saved_base_forecasts=refs[lag])
            if old_data.digest(frame.to_dict('records'))!=before:raise ValueError('Predict mutated target-free features')
            path=out/f'selection_lag{lag}_forecasts.jsonl';old_data._write_rows(path,forecast_rows(frame,predictions,lag))
            forecasts[str(lag)]={**record(path),'rows':len(frame)};frames[lag]=frame
            progress(stage='selection_forecasts_closed',lag_seconds=lag,rows=len(frame))
        verify_design(out)
        save(out/'selection_issuance_lock.json',{'closed_at_utc':now(),'fit_lock':record(out/'fit_lock.json'),
            'data_lock':record(out/'data_lock.json'),'forecasts':forecasts,
            'original_references':lock['parent']['years']['2023']['original_references'],
            'saved_base_forecasts':lock['parent']['reference_forecasts'],'external_selection_labels_read':False})
        # Only now does this runner read the previously exposed 2023 label rows.
        label_ref=lock['parent']['years']['2023']['labels']
        labeled={lag:old_run.join_labels(frame,label_ref) for lag,frame in frames.items()}
        expected_labeled=old_run.join_labels(expected_issued,label_ref)
        expected_matched=expected_labeled.loc[expected_labeled.outcome_status.eq('matched')].copy()
        predictions={lag:load_forecasts(frame,forecasts[str(lag)],lag) for lag,frame in frames.items()}
        summary=evaluate.evaluate_selection(labeled[2],predictions[2],labeled[0],predictions[0],
            expected_matched=expected_matched,expected_issuances=expected_issued,
            primary_references=refs[2],sensitivity_references=refs[0])
        verify_design(out)
        save(out/'selection.json',{'completed_at_utc':now(),'status':'retrospective_gap_research_no_promotion',
            'summary':summary,'design_lock':record(out/'design_lock.json'),'data_lock':record(out/'data_lock.json'),
            'fit_lock':record(out/'fit_lock.json'),'issuance_lock':record(out/'selection_issuance_lock.json'),
            'selection_labels':label_ref})
        save(out/'selection_lock.json',{'closed_at_utc':now(),'selection':record(out/'selection.json'),
            'design_lock':record(out/'design_lock.json'),'selected_candidate':'gap_hgb',
            'advances_to_later_evaluation':summary['advances_to_later_evaluation'],'promotion':False})
        progress(stage='selection_complete',advances_to_later_evaluation=summary['advances_to_later_evaluation'],
            selection=record(out/'selection.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','prepare','select'))
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--review',type=Path);args=parser.parse_args()
    if args.stage=='freeze' and args.review is None:parser.error('--review is required')
    if args.stage=='freeze':freeze(args.out,args.review)
    elif args.stage=='prepare':prepare(args.out)
    else:select(args.out)


if __name__=='__main__':main()
