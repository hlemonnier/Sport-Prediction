"""Post-score diagnostics of immutable forecasts; no model fits or selection."""
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from research.experiments.boundary_20260908.distributional.execution import run

OUT=run.OUT/'diagnostics'
LEVELS=np.array([.025,.05,.1,.25,.5,.75,.9,.95,.975])


def load(name):
    with (run.OUT/name).open('rb') as file:return pickle.load(file)


def components(y,q):
    sharpness=np.zeros(len(y));under=np.zeros(len(y));over=np.zeros(len(y))
    for i,a in enumerate([.5,.2,.1,.05]):
        lower,upper=q[:,3-i],q[:,5+i]
        sharpness+=a/2*(upper-lower)
        under+=np.maximum(y-upper,0)
        over+=np.maximum(lower-y,0)
    return {'median_absolute_error_component':np.abs(y-q[:,4])/9,
            'weighted_width_component':sharpness/4.5,
            'underprediction_penalty_y_above_upper':under/4.5,
            'overprediction_penalty_y_below_lower':over/4.5}


def one(frame,predictions,selected,cuts):
    counts=frame.groupby('event_key').size();weight=1/(frame.event_key.map(counts).to_numpy()*len(counts))
    y=frame.lap_time_seconds.to_numpy();assert abs(weight.sum()-1)<1e-12
    models={}
    for name,q in predictions.items():
        parts=components(y,q);mean={k:float(weight@v) for k,v in parts.items()}
        error=y[:,None]-q;pin=np.maximum(LEVELS*error,(LEVELS-1)*error)
        wis=2/9*pin.sum(1);np.testing.assert_allclose(wis,sum(parts.values()),atol=1e-12)
        models[name]={'wis':float(weight@wis),'decomposition':mean,
                     'quantile_probability_y_le_q':(weight@((y[:,None]<=q).astype(float))).tolist(),
                     'calibration_probability_minus_nominal':(weight@((y[:,None]<=q).astype(float))-LEVELS).tolist(),
                     'weighted_wis_by_quantile':(2/9*(weight@pin)).tolist()}
    differences={ref:{'wis_delta':models[selected]['wis']-models[ref]['wis'],
                     'decomposition_delta':{k:models[selected]['decomposition'][k]-v for k,v in models[ref]['decomposition'].items()},
                     'quantile_wis_delta':(np.asarray(models[selected]['weighted_wis_by_quantile'])-models[ref]['weighted_wis_by_quantile']).tolist()}
                 for ref in run.REFS}
    volatility=np.maximum(frame.x_mad_3.to_numpy(),frame.x_mad_8.to_numpy())
    masks={'wet_compound':frame.wet_compound.to_numpy()>.5,'dry_compound':frame.wet_compound.to_numpy()<=.5,
           'stint_first_three_clean':frame.stint_clean_count.to_numpy()<=3,'stint_after_three_clean':frame.stint_clean_count.to_numpy()>3}
    quartile=np.searchsorted(cuts,volatility,side='right')
    masks.update({f'past_only_volatility_bin_{k}':quartile==k for k in range(4)})
    regimes={}
    for name,mask in masks.items():
        if not mask.any():continue
        share=float(weight[mask].sum());scores={}
        for name_model,q in predictions.items():
            total=sum(components(y,q).values());contribution=float(weight[mask]@total[mask])
            scores[name_model]={'wis_contribution_to_global':contribution,'wis_conditional_on_regime_under_global_weights':contribution/share}
        regimes[name]={'rows':int(mask.sum()),'events_with_rows':int(frame.loc[mask,'event_key'].nunique()),
                       'global_event_weight_share':share,'models':scores,'contribution_delta_vs_conditional':scores[selected]['wis_contribution_to_global']-scores['stratified_empirical']['wis_contribution_to_global']}
    return {'rows':len(frame),'events':len(counts),'models':models,'differences':differences,'regimes':regimes}


def main():
    if (OUT/'results.json').exists():raise FileExistsError('Diagnostics already frozen')
    result=json.loads((run.OUT/'results.json').read_text());selection=json.loads((run.OUT/'selection.json').read_text());selected=result['selected']
    selection_forecasts=load('selection_forecasts.pkl');transfer=load('transfer_forecasts.pkl')
    selected_bundle=load('selection_bundle.pkl');final=load('final_distribution_bundle.pkl')
    groups={'2023_selection':one(selection_forecasts['frame'],{k:selection_forecasts['quantiles'][k] for k in [*run.REFS,selected]},selected,selected_bundle['empirical']['cuts'])}
    frame=transfer['frame']
    for name,mask in [('2024',frame.year.eq(2024)),('2025',frame.year.eq(2025)),('2024_2025',frame.year.isin([2024,2025])),('2026',frame.year.eq(2026)),('2026_recent',frame.year.eq(2026)&frame.event_key.ge(202610))]:
        groups[name]=one(frame.loc[mask],{k:q[mask] for k,q in transfer['quantiles'].items()},selected,final['empirical']['cuts'])
        for key,m in groups[name]['models'].items():np.testing.assert_allclose(m['wis'],result['summaries'][name]['models'][key]['wis'],atol=1e-12)
    files=['results.json','selection.json','verification.json','evidence.json','selection_forecasts.pkl','transfer_forecasts.pkl','selection_bundle.pkl','final_distribution_bundle.pkl']
    run.write(OUT/'results.json',{'schema_version':'fixed_forecast_wis_diagnosis_v1','created_at_utc':run.now(),'status':'post_score_exploratory_diagnosis_not_model_selection',
        'source_sha256':run.sha(__file__),'inputs':{str((run.OUT/p).relative_to(run.ROOT)):run.sha(run.OUT/p) for p in files},'quantile_levels':LEVELS.tolist(),
        'weighting':'Each complete event has equal total weight. Regime WIS contributions retain these original weights; conditioned regime averages divide by regime weight share and do not reweight events within a regime.',
        'selected_candidate_unchanged':selected,'fits_executed':0,'promotion':False,'groups':groups})
    print(json.dumps({'groups':{name:{'deltas':v['differences']['stratified_empirical'],'selected_cdf':v['models'][selected]['quantile_probability_y_le_q'],'reference_cdf':v['models']['stratified_empirical']['quantile_probability_y_le_q']} for name,v in groups.items()}},indent=2))


if __name__=='__main__':main()
