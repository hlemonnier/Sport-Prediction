"""Reconstruct final predictions and replay rich features under future poisoning."""
from pathlib import Path
import argparse
import importlib.util
import json
import pickle
import sys

import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('frontier_live_verification_target',Path(__file__).with_name('live_frontier.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def main(result_directory):
    results=json.loads((result_directory/'results.json').read_text())
    selection=json.loads((m.OUT/'selection.json').read_text())
    fit_lock=json.loads((m.OUT/'fit_lock.json').read_text())
    assert m.sha(m.OUT/'selection.json')==fit_lock['selection_sha256']
    assert m.sha(m.OUT/'fitted_models.pkl')==fit_lock['models_sha256']
    assert m.sha(m.HERE/'live_frontier.py')==fit_lock['source_sha256']
    frame=pd.read_pickle(result_directory/'transfer_data_and_forecasts.pkl')
    discovery=pd.read_pickle(m.OUT/'discovery_data.pkl')
    assert set(discovery.year)=={2022,2023} and set(frame.year)=={2024,2025,2026}
    inventory=selection['input_manifest']+results['input_manifest']
    assert len({i['event_key'] for i in inventory})==len(inventory)==105
    for item in inventory:assert m.sha(m.ROOT/item['path'])==item['sha256']
    with (m.OUT/'fitted_models.pkl').open('rb') as f:models=pickle.load(f)
    max_errors={}
    for name,model in models.items():
        p=m.predict_model(frame,model)
        selected_name=next(n for n in selection['selected_by_family'].values() if n.removesuffix('_half')==name)
        if selected_name.endswith('_half'):p=.5*p+.5*frame.forecast_naive_seconds.to_numpy()
        error=float(np.max(np.abs(p-frame['prediction_'+selected_name].to_numpy())))
        assert error<1e-10;max_errors[name]=error
    preferred=selection['preferred'];config=selection['configurations'][preferred.removesuffix('_half')]
    fitted=m.fit_model(discovery.loc[discovery.year==2022],config,selection['features'])
    v=discovery.loc[discovery.year==2023];p=m.predict_model(v,fitted)
    if preferred.endswith('_half'):p=.5*p+.5*v.forecast_naive_seconds.to_numpy()
    metric=m.diagnostics(v,p)['candidate_mae']
    assert abs(metric-selection['all_candidate_selection_metrics'][preferred]['candidate_mae'])<1e-10
    checked=[]
    for key in [202201,202506,202612]:
        item=next(i for i in inventory if i['event_key']==key);raw=pd.read_csv(m.ROOT/item['path'])
        issued,_=m.enrich(raw,key)
        for fraction in [.3,.65]:
            cutoff=float(raw.Time.quantile(fraction));a,_=m.enrich(raw.loc[raw.Time<=cutoff],key)
            bad=raw.copy();future=bad.Time>cutoff
            for col in ['LapTime','Sector1Time','Sector2Time','Sector3Time','SpeedI1','SpeedI2','SpeedST','Position','Stint']:
                if col in bad:bad.loc[future,col]=9999.
            bad.loc[future,'IsAccurate']=False;bad.loc[future,'Compound']='WET'
            b,_=m.enrich(bad,key);expected=issued.loc[issued.issued_at_timestamp<=cutoff].reset_index(drop=True)
            for x in [a,b.loc[b.issued_at_timestamp<=cutoff]]:
                pd.testing.assert_frame_equal(expected,x.reset_index(drop=True),check_exact=True)
                for model in models.values():np.testing.assert_array_equal(m.predict_model(expected,model),m.predict_model(x.reset_index(drop=True),model))
            checked.append({'event_key':key,'cutoff':cutoff,'issuances':len(expected)})
    reference=pd.read_csv(m.ROOT/'artifacts/research/performance_20260907/live/corrected_cycle_1/matched_forecasts.csv.gz',dtype={'driver_id':str})
    recent=pd.read_csv(m.ROOT/'artifacts/research/performance_20260907/live/recent_extension_final/matched_forecasts.csv.gz',dtype={'driver_id':str})
    old=pd.concat([reference.loc[reference.event_key.ge(202400)],recent],ignore_index=True)
    keys=m.KEYS+['target_lap_number','target_timestamp']
    merged=frame.merge(old[keys+['lap_time_seconds','forecast_naive_seconds']],on=keys,validate='one_to_one',suffixes=('','_previous'))
    assert len(merged)==len(frame)==len(old)
    for col in ['lap_time_seconds','forecast_naive_seconds']:np.testing.assert_allclose(merged[col],merged[col+'_previous'],atol=1e-10,rtol=0)
    metrics_checked=0
    for period, mask in [('2024',frame.year.eq(2024)),('2025',frame.year.eq(2025)),('2024_2025',frame.year.isin([2024,2025])),('2026',frame.year.eq(2026)),('2026_recent',frame.event_key.ge(202610))]:
        sub=frame.loc[mask]
        for name,reported in results['results'][period].items():
            residual=np.abs(sub['prediction_'+name]-sub.lap_time_seconds)
            value=residual.groupby(sub.event_key).mean().mean()
            assert abs(value-reported['candidate_mae'])<1e-12
            recomputed=m.diagnostics(sub,sub['prediction_'+name].to_numpy())
            for ci in ['event_ci95','block3_ci95']:np.testing.assert_array_equal(recomputed[ci],reported[ci])
            metrics_checked+=1
    payload={'status':'passed','results_sha256':m.sha(result_directory/'results.json'),'source_sha256':m.sha(m.HERE/'live_frontier.py'),
             'verifier_sha256':m.sha(__file__),'input_hashes_verified':len(inventory),'prediction_reconstruction_max_errors':max_errors,
             'preferred_selection_refit_reproduced':True,'real_prefix_poisoning_cases':checked,'population_exactly_matches_published_baselines':True,
             'transfer_rows':len(frame),'metric_and_interval_pairs_checked':metrics_checked,'research_only_no_prospective_claim':True}
    m.write(result_directory/'verification.json',payload);print(json.dumps(payload,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--result-directory',type=Path,default=m.OUT/'corrected_input_contract')
    main(parser.parse_args().result_directory.resolve())
