"""Repeat transfer with restored observed inputs; never refit or reselect."""
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('frontier_live_corrected_input_target',Path(__file__).with_name('live_frontier.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
OUT=m.OUT/'corrected_input_contract'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'results.json').exists():raise FileExistsError('Never overwrite corrected transfer results')
    selection=json.loads((m.OUT/'selection.json').read_text());lock=json.loads((m.OUT/'fit_lock.json').read_text())
    recovery_path=m.ROOT/'artifacts/research/frontier_20260907/input_recovery/manifest.json';recovery=json.loads(recovery_path.read_text())
    assert m.sha(m.HERE/'live_frontier.py')==lock['source_sha256'] and m.sha(m.SPEC)==lock['spec_sha256']
    assert m.sha(m.OUT/'selection.json')==lock['selection_sha256'] and m.sha(m.OUT/'fitted_models.pkl')==lock['models_sha256']==recovery['freeze']['model_sha256']
    m.write(OUT/'replay_lock.json',{'runner_sha256':m.sha(__file__),'selection_sha256':lock['selection_sha256'],'model_sha256':lock['models_sha256'],
        'recovery_manifest_sha256':m.sha(recovery_path),'change':'Restore nine already cached observed fields omitted by prior minimal CSV export; no parameter, selection or feature-definition changes','refit_permitted':False})
    old=pd.read_pickle(m.OUT/'transfer_data_and_forecasts.pkl');parts=[old.loc[old.event_key<202610].copy()]
    result0=json.loads((m.OUT/'results.json').read_text())
    inventory=[r for r in result0['input_manifest'] if r['event_key']<202610]
    for item in recovery['events']:
        raw=pd.read_csv(m.ROOT/item['path']);assert m.sha(m.ROOT/item['path'])==item['sha256']
        if not set(item['added_columns']).issubset(raw):raise ValueError('Required observed input columns absent')
        issued,frame=m.enrich(raw,item['event_key']);frame['year']=2026;parts.append(frame)
        inventory.append({'event_key':item['event_key'],'path':item['path'],'sha256':item['sha256'],'raw_rows':len(raw),'issuances':len(issued),'matched_rows':len(frame)})
    frame=pd.concat(parts,ignore_index=True)
    frame=frame.drop(columns=[c for c in frame if c.startswith('prediction_')])
    with (m.OUT/'fitted_models.pkl').open('rb') as f:models=pickle.load(f)
    predictions={}
    for family,name in selection['selected_by_family'].items():
        key=name.removesuffix('_half')
        if name.startswith('expert_'):p=frame[name].to_numpy()
        elif name.startswith('online_expert_eta'):p=m.online_policy(frame,float(name.removeprefix('online_expert_eta')))
        else:
            p=m.predict_model(frame,models[key])
            if name.endswith('_half'):p=.5*p+.5*frame.forecast_naive_seconds.to_numpy()
        predictions[name]=p
    prior=json.loads((m.ROOT/'artifacts/research/performance_20260907/live/corrected_cycle_1/selected_models.json').read_text())['final_ridge_model']
    predictions['previous_ridge']=m.base.ridge_predict(frame,prior)
    results={}
    for label,mask in [('2024',frame.year.eq(2024)),('2025',frame.year.eq(2025)),('2024_2025',frame.year.isin([2024,2025])),
                       ('2026',frame.year.eq(2026)),('2026_R1_R9',frame.year.eq(2026)&frame.event_key.lt(202610)),('2026_recent',frame.event_key.ge(202610))]:
        results[label]={name:m.diagnostics(frame.loc[mask],p[mask]) for name,p in predictions.items()}
    preferred=selection['preferred'];hist=results['2024_2025'][preferred];ridge=results['2024_2025']['previous_ridge']['candidate_mae']
    gate={'historical_gain_at_least_10pct':hist['relative_reduction']>=.1,'beat_previous_ridge_at_least_7pct':1-hist['candidate_mae']/ridge>=.07,
          'both_historical_years_improve':all(results[y][preferred]['delta']<0 for y in ['2024','2025']), 'block_ci_upper_negative':hist['block3_ci95'][1]<0,
          'loo_all_improve':hist['loo_max_delta']<0,'current_year_point_improves':results['2026'][preferred]['delta']<0}
    for name,p in predictions.items():frame['prediction_'+name]=p
    frame.to_pickle(OUT/'transfer_data_and_forecasts.pkl')
    result={'experiment':'live_frontier_cycle_1_corrected_observed_inputs','preferred':preferred,'selected_by_family':selection['selected_by_family'],
            'results':results,'gate':gate,'substantial_research_gate_passed':all(gate.values()),'fit_lock_sha256':m.sha(m.OUT/'fit_lock.json'),
            'selection_sha256':m.sha(m.OUT/'selection.json'),'input_manifest':inventory,'forecasts_sha256':m.sha(OUT/'transfer_data_and_forecasts.pkl'),
            'original_incomplete_input_result_sha256':m.sha(m.OUT/'results.json'),'recovery_manifest_sha256':m.sha(recovery_path),
            'model_bytes_unchanged':True,'feature_code_unchanged':True,'promotion':False,
            'evidence_role':'Amended retrospective transfer after source export omission diagnosed; all years previously exposed; no prospective evidence.'}
    m.write(OUT/'results.json',result)
    print(json.dumps({'gate':gate,'preferred':preferred,'results':{y:{n:round(r['relative_reduction']*100,4) for n,r in data.items()} for y,data in results.items()}},indent=2))


if __name__=='__main__':main()
