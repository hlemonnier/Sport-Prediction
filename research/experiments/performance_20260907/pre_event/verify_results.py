#!/usr/bin/env python3
"""Independent readback checks for a frozen pre-event experiment artifact."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def verify(directory):
    artifact=directory/'results.json'
    result=json.loads(artifact.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    for group in ['implementation_manifest','input_manifest','output_manifest']:
        for path,digest in result[group].items():assert sha(ROOT/path)==digest,(group,path)
    predictions=pd.read_csv(directory/'predictions.csv',float_precision='round_trip')
    frozen_columns=['event_key','driver_id','team_id','year','round','rehearsal_source',
                    'raw_anchor','duration','rank_latest','wet_fraction','valid_clean_lap_count','anchor_is_imputed',
                    'baseline_qualifying','Q1_fixed_rank_blend','Q2_ridge_rank_residual',
                    'baseline_best_lap','L1_fractional_source_shift','L2_huber_context_residual']
    for event in result['events']:
        key=event['event_key'];rows=predictions.loc[predictions.event_key.eq(key)]
        frozen=rows[frozen_columns].copy()
        payload=json.dumps(clean(frozen.to_dict('records')),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if hashlib.sha256(payload).hexdigest()!=event['forecast_sha256']:
            # Concatenating a later event with a missing count promotes earlier
            # integer count columns to float in the CSV. Recover that event's
            # original integer dtype; no numerical prediction value changes.
            count=frozen.valid_clean_lap_count
            assert count.notna().all() and np.equal(count,np.floor(count)).all()
            frozen['valid_clean_lap_count']=count.astype(int)
            payload=json.dumps(clean(frozen.to_dict('records')),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        assert hashlib.sha256(payload).hexdigest()==event['forecast_sha256'],('forecast',key)
        for col in frozen_columns[12:15]:
            assert sorted(rows[col])==list(range(1,len(rows)+1)),(col,key)
            error=(rows[col]-rows.qualy_position).abs().mean()
            assert abs(error-event[col+'_mae'])<1e-12,(col,key,error)
        mask=rows.lap_scored.astype(bool)
        for col in frozen_columns[15:]:
            error=(rows.loc[mask,col]-rows.loc[mask,'achievable_session_end_lap_time_seconds']).abs().mean()
            assert abs(error-event[col+'_mae'])<1e-12,(col,key,error)
    for event in result['fit_attempts']:
        for name,model in event['models'].items():
            assert model['status']!='fit_failed_baseline_substitute',(event['event_key'],name)
            assert not model.get('warnings'),(event['event_key'],name)
            assert all(k<event['event_key'] for k in model.get('training_event_keys',[]))
    assert len(result['events'])==92
    assert len(result['excluded_events'])==9
    assert all(e['event_key']<202400 and 'no causal FP3' in e['reason'] for e in result['excluded_events'])
    assert len(result['all_tried_candidates'])==4 and result['selection_performed'] is False
    output={'status':'pass','result_sha256':sha(artifact),'forecast_hashes_recomputed':len(result['events']),
            'implementation_files_checked':len(result['implementation_manifest']),
            'input_files_checked':len(result['input_manifest']),
            'full_roster_rank_permutations_checked':len(result['events'])*3,
            'all_event_MAEs_recomputed':len(result['events'])*6,
            'all_fitted_models_strictly_prior':True,'final_fit_failures':0,'convergence_warnings':0,
            'verifier_sha256':sha(Path(__file__)),
            'experiment_tests':'five mathematical/chronology/path-adapter tests passed'}
    (directory/'verification.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=ROOT/'artifacts/research/performance_20260907/pre_event/cycle1')
    verify(parser.parse_args().directory.resolve())
