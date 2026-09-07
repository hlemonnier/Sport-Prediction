"""Causal nonlinear and adaptive next-eligible-lap research, isolated from serving."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/research/frontier_20260907/live'
SPEC = HERE / 'specification.json'
BASE_PATH = ROOT / 'research/experiments/performance_20260907/live/run_experiment.py'
module_spec = importlib.util.spec_from_file_location('published_live_cycle', BASE_PATH)
base = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = base
module_spec.loader.exec_module(base)
SEED = 20260907
KEYS = base.KEYS
FEATURE_PREFIX = 'x_'
SECTORS = ['Sector1Time', 'Sector2Time', 'Sector3Time']
SPEEDS = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def finite(value, default=0.):
    z = base.numeric(value)
    return z if np.isfinite(z) else default


def enrich(raw, event_key):
    issued, matched = base.stream_event(raw, event_key)
    work = raw.copy()
    work['Time'] = pd.to_numeric(work.Time)
    work['_driver'] = work.DriverNumber.map(base.driver_key)
    work = work.sort_values(['Time', '_driver', 'LapNumber'], kind='mergesort')
    states, extra = {}, []
    issuance_keys = set(zip(issued.driver_id, issued.issued_after_lap_number, issued.issued_at_timestamp))
    for timestamp, batch in work.groupby('Time', sort=False):
        peers = {d: s['peer'] for d, s in states.items() if s['peer'] is not None}
        for row in batch.to_dict('records'):
            d, lap = row['_driver'], int(row['LapNumber'])
            s = states.setdefault(d, {'stint': np.nan, 'compound': '', 'history': deque(maxlen=12),
                                      'ewma': {}, 'peer': None, 'generation': 0})
            stint, compound = base.numeric(row.get('Stint')), str(row.get('Compound', 'UNKNOWN')).upper()
            pit_out = np.isfinite(base.numeric(row.get('PitOutTime')))
            pit_in = np.isfinite(base.numeric(row.get('PitInTime')))
            reset = s['generation'] == 0 or pit_out or (s['compound'] and compound != s['compound']) or (
                np.isfinite(stint) and np.isfinite(s['stint']) and stint != s['stint'])
            if reset:
                s['history'].clear(); s['ewma'].clear(); s['peer'] = None; s['generation'] += 1
                if not np.isfinite(stint): s['stint'] = np.nan
            if np.isfinite(stint): s['stint'] = stint
            s['compound'] = compound
            y = base.numeric(row.get('LapTime'))
            eligible = np.isfinite(y) and y > 0 and base.truth(row.get('IsAccurate', False)) and not (pit_in or pit_out) and not any(c in str(row.get('TrackStatus', '')) for c in '4567')
            if not eligible: continue
            previous = s['history'][-1] if s['history'] else None
            delta = (y - previous['y']) / (lap - previous['lap']) if previous else 0.
            s['peer'] = {'time': timestamp, 'lap': lap, 'y': y, 'delta': float(np.clip(delta, -5, 5))}
            item = {'lap': lap, 'y': y, **{k: base.numeric(row.get(k)) for k in SECTORS + SPEEDS}}
            s['history'].append(item)
            for alpha in [.25, .5, .75]:
                old = s['ewma'].get(alpha, y)
                s['ewma'][alpha] = y if abs(y-old) > 2 else old+alpha*(y-old)
            if (d, lap, timestamp) not in issuance_keys: continue
            h = list(s['history']); yy = np.array([i['y'] for i in h]); ll = np.array([i['lap'] for i in h])
            features = {'observed_lap': lap, 'position': finite(row.get('Position'), 11), 'lap_duration': y,
                        'fresh_tyre': float(base.truth(row.get('FreshTyre'))), 'history_count': len(h)}
            for k in range(1, 7):
                features[f'lag_{k}_gap'] = float(np.clip(yy[-k-1]-y, -15, 15)) if len(yy) > k else 0.
                features[f'lag_{k}_lap_distance'] = float(lap-ll[-k-1]) if len(yy) > k else 0.
            experts = {}
            for k in [2, 3, 5, 8]:
                v, l = yy[-k:], ll[-k:]
                med = float(np.median(v)); avg = float(v.mean())
                slope = float(np.dot(l-l.mean(), v-v.mean())/np.square(l-l.mean()).sum()) if len(v) >= 2 else 0.
                slope = float(np.clip(slope, -1, 1))
                features.update({f'mean_{k}_gap': float(np.clip(avg-y, -15, 15)), f'median_{k}_gap': float(np.clip(med-y, -15, 15)),
                    f'mad_{k}': float(np.median(np.abs(v-med))), f'trend_{k}': slope,
                    f'range_{k}': float(np.clip(v.max()-v.min(), 0, 20))})
                experts[f'mean_{k}'] = y+float(np.clip(avg-y, -3, 3))
                experts[f'median_{k}'] = y+float(np.clip(med-y, -3, 3))
                experts[f'trend_{k}'] = y+float(np.clip(avg+slope*(lap+1-l.mean())-y, -3, 3))
            for alpha, value in s['ewma'].items():
                features[f'reset_ewma_{alpha}_gap'] = value-y
                experts[f'reset_ewma_{alpha}'] = value
            for col in SECTORS+SPEEDS:
                current = finite(row.get(col))
                vals = np.array([v[col] for v in h[-5:]])
                vals = vals[np.isfinite(vals)]
                features[col] = current
                features[col+'_missing'] = float(not np.isfinite(base.numeric(row.get(col))))
                features[col+'_median_gap'] = float(np.clip(np.median(vals)-current, -30, 30)) if len(vals) else 0.
            for c in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']:
                features['compound_'+c] = float(compound == c)
            others = [p for driver, p in peers.items() if driver != d and 0 < timestamp-p['time'] <= 180 and abs(lap-p['lap']) <= 1]
            features['peer_count'] = len(others)
            features['peer_delta_mean'] = float(np.mean([p['delta'] for p in others])) if others else 0.
            features['peer_delta_median'] = float(np.median([p['delta'] for p in others])) if others else 0.
            features['peer_delta_mad'] = float(np.median(np.abs(np.array([p['delta'] for p in others])-features['peer_delta_median']))) if others else 0.
            features['relative_field_pace'] = float(np.clip(y-np.median([p['y'] for p in others]), -15, 15)) if others else 0.
            extra.append(dict(event_key=event_key, driver_id=d, issued_after_lap_number=lap, issued_at_timestamp=float(timestamp),
                **{FEATURE_PREFIX+k: float(v) for k, v in features.items()}, **{'expert_'+k: float(v) for k,v in experts.items()}))
    extra = pd.DataFrame(extra)
    enriched = issued.merge(extra, on=KEYS, validate='one_to_one')
    targets = matched[[*KEYS, 'target_lap_number', 'target_timestamp', 'lap_time_seconds', 'target_same_stint']]
    paired = enriched.merge(targets, on=KEYS, validate='one_to_one')
    assert len(enriched) == len(issued) and len(paired) == len(matched)
    return enriched, paired


def build(years):
    inventory, frames = [], []
    paths = sorted((ROOT/'data/f1/raw/weekends').glob('*/*/*_race_laps.csv'))
    if 2026 in years: paths += sorted((ROOT/'data/f1/performance_20260907/live_recent_final').glob('*_race_laps.csv'))
    seen = set()
    for p in paths:
        if 'live_recent_final' in str(p): year, rnd = 2026, int(p.name.split('_')[2])
        else: year, rnd = int(p.parts[-3]), int(p.parts[-2].split('_')[1])
        if year not in years: continue
        key = year*100+rnd
        assert key not in seen; seen.add(key)
        raw = pd.read_csv(p); issued, scored = enrich(raw, key); scored['year'] = year
        frames.append(scored)
        inventory.append({'event_key': key, 'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'raw_rows': len(raw), 'issuances': len(issued), 'matched_rows': len(scored)})
        print('features', key, len(scored), flush=True)
    return pd.concat(frames, ignore_index=True), inventory


def weights(frame):
    counts = frame.event_key.value_counts()
    return len(frame)/(len(counts)*frame.event_key.map(counts).to_numpy())


def diagnostics(frame, predictions, reference=None):
    ref = frame.forecast_naive_seconds.to_numpy() if reference is None else reference
    f = pd.DataFrame({'key': frame.event_key, 'base': np.abs(ref-frame.lap_time_seconds), 'candidate': np.abs(predictions-frame.lap_time_seconds)})
    ev = f.groupby('key')[['base','candidate']].mean();delta = (ev.candidate-ev.base).to_numpy()
    rng = np.random.default_rng(SEED); n=len(ev)
    draws = delta[rng.integers(n,size=(20000,n))].mean(axis=1)
    starts = rng.integers(n,size=(20000,int(np.ceil(n/3))))
    idx = ((starts[:,:,None]+np.arange(3))%n).reshape(20000,-1)[:,:n]
    blocks=delta[idx].mean(axis=1)
    return {'events': n, 'rows': len(frame), 'baseline_mae': float(ev.base.mean()), 'candidate_mae': float(ev.candidate.mean()),
        'relative_reduction': float(1-ev.candidate.mean()/ev.base.mean()), 'delta': float(delta.mean()),
        'event_ci95': np.quantile(draws,[.025,.975]).tolist(), 'block3_ci95': np.quantile(blocks,[.025,.975]).tolist(),
        'events_improved': int((delta<0).sum()), 'loo_max_delta': float(max((delta.sum()-v)/(n-1) for v in delta)) if n>1 else None,
        'per_event': [{'event_key': int(k), 'baseline_mae': float(r.base), 'candidate_mae': float(r.candidate)} for k,r in ev.iterrows()]}


def online_policy(frame, eta):
    """Full-information expert weighting; target becomes usable at its timestamp."""
    experts = ['forecast_naive_seconds']+[c for c in frame if c.startswith('expert_')]
    predicted = pd.Series(index=frame.index,dtype=float)
    for _, event in frame.groupby('event_key', sort=False):
        states={}; actions=[]
        for idx,row in event.iterrows():
            actions.append((float(row.issued_at_timestamp), 1, idx))
            actions.append((float(row.target_timestamp), 0, idx))
        for _, action, idx in sorted(actions):
            row=event.loc[idx];d=str(row.driver_id)
            losses=states.setdefault(d,np.zeros(len(experts)))
            if action==0:
                states[d]=.98*losses+np.minimum(np.abs(row[experts].to_numpy(float)-row.lap_time_seconds),3.)
            else:
                scores=-eta*losses;prob=np.exp(scores-scores.max());prob/=prob.sum()
                predicted.loc[idx]=float(prob@row[experts].to_numpy(float))
    return predicted.to_numpy()


def neural_fit(frame, features):
    import torch
    torch.set_num_threads(1);torch.manual_seed(SEED)
    x=frame[features].to_numpy(float);w=weights(frame)
    mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-8]=1
    xx=torch.tensor(np.clip((x-mean)/scale,-10,10),dtype=torch.float32)
    yy=torch.tensor(np.clip(frame.lap_time_seconds-frame.forecast_naive_seconds,-5,5).to_numpy(),dtype=torch.float32)
    ww=torch.tensor(w,dtype=torch.float32)
    model=torch.nn.Sequential(torch.nn.Linear(len(features),64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,1))
    skip=torch.nn.Linear(len(features),1,bias=False)
    optimizer=torch.optim.AdamW([*model.parameters(),*skip.parameters()],lr=.001,weight_decay=.01)
    rng=np.random.default_rng(SEED);history=[]
    for epoch in range(30):
        total=0.
        for indices in np.array_split(rng.permutation(len(frame)),int(np.ceil(len(frame)/1024))):
            optimizer.zero_grad();prediction=model(xx[indices]).squeeze(-1)+skip(xx[indices]).squeeze(-1)
            loss=(ww[indices]*torch.abs(prediction-yy[indices])).mean();loss.backward();optimizer.step();total+=float(loss.detach())*len(indices)
        history.append(total/len(frame))
    return {'kind':'neural','model':model.state_dict(),'skip':skip.state_dict(),'mean':mean,'scale':scale,'features':features,'training_losses':history}


def fit_model(frame, config, features):
    if config['kind']=='neural': return neural_fit(frame,features)
    if config['kind']=='hgb':
        model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.06,max_iter=config['iterations'],max_leaf_nodes=config['leaves'],min_samples_leaf=80,l2_regularization=10,early_stopping=False,random_state=SEED)
    else:
        model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=config['leaf'],max_features=.8,n_jobs=1,random_state=SEED)
    y=np.clip(frame.lap_time_seconds-frame.forecast_naive_seconds,-3 if config['kind']=='trees' else -5,3 if config['kind']=='trees' else 5)
    model.fit(frame[features],y,sample_weight=weights(frame))
    return {'kind':config['kind'],'model':model,'features':features}


def predict_model(frame, model):
    if model['kind']=='neural':
        import torch
        torch.set_num_threads(1)
        net=torch.nn.Sequential(torch.nn.Linear(len(model['features']),64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,1));net.load_state_dict(model['model'])
        skip=torch.nn.Linear(len(model['features']),1,bias=False);skip.load_state_dict(model['skip']);net.eval()
        x=torch.tensor(np.clip((frame[model['features']].to_numpy()-model['mean'])/model['scale'],-10,10),dtype=torch.float32)
        with torch.no_grad(): y=(net(x).squeeze(-1)+skip(x).squeeze(-1)).numpy()
    else:y=model['model'].predict(frame[model['features']])
    return frame.forecast_naive_seconds.to_numpy()+np.clip(y,-3,3)


def discover():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection.json').exists():raise FileExistsError('Completed discovery cannot be overwritten')
    write(OUT/'design_lock.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'spec_sha256':sha(SPEC),'source_sha256':sha(__file__),'base_source_sha256':sha(BASE_PATH),'specification':json.loads(SPEC.read_text())})
    data,inventory=build([2022,2023]);data.to_pickle(OUT/'discovery_data.pkl')
    fit=data.loc[data.year==2022];val=data.loc[data.year==2023]
    features=base.FEATURES+[c for c in data if c.startswith(FEATURE_PREFIX)]
    configs={f'hgb_l{l}_i{i}':{'kind':'hgb','leaves':l,'iterations':i} for l in [7,15,31] for i in [150,300]}
    configs.update({f'extra_trees_leaf{l}':{'kind':'trees','leaf':l} for l in [32,128]})
    configs['neural_residual']={'kind':'neural'}
    metrics={};predictions={};learned={}
    for col in [c for c in data if c.startswith('expert_')]:predictions[col]=val[col].to_numpy()
    for eta in [.1,.5,1.]:predictions[f'online_expert_eta{eta}']=online_policy(val,eta)
    for name,config in configs.items():
        model=fit_model(fit,config,features);learned[name]=model; p=predict_model(val,model)
        predictions[name]=p;predictions[name+'_half']=.5*p+.5*val.forecast_naive_seconds.to_numpy()
        print('selected-period-fit',name,diagnostics(val,p)['relative_reduction'],flush=True)
    for name,p in predictions.items():metrics[name]=diagnostics(val,p)
    families={'filters':[n for n in predictions if n.startswith('expert_')], 'online_expert':[n for n in predictions if n.startswith('online_')],
              'hgb':[n for n in predictions if n.startswith('hgb_')], 'trees':[n for n in predictions if n.startswith('extra_')], 'neural':[n for n in predictions if n.startswith('neural_')]}
    selected={family:min(names,key=lambda name:metrics[name]['candidate_mae']) for family,names in families.items()}
    preferred=min(selected.values(),key=lambda name:metrics[name]['candidate_mae'])
    lock={'preferred':preferred,'selected_by_family':selected,'all_candidate_selection_metrics':metrics,'configurations':configs,'features':features,'input_manifest':inventory,'design_sha256':sha(OUT/'design_lock.json')}
    write(OUT/'selection.json',lock)
    models={}
    for name in selected.values():
        key=name.removesuffix('_half')
        if key in configs:models[key]=fit_model(data,configs[key],features)
    with (OUT/'fitted_models.pkl').open('wb') as f:pickle.dump(models,f)
    write(OUT/'fit_lock.json',{'selection_sha256':sha(OUT/'selection.json'),'models_sha256':sha(OUT/'fitted_models.pkl'),'training_years':[2022,2023],'source_sha256':sha(__file__),'spec_sha256':sha(SPEC)})
    print(json.dumps({'preferred':preferred,'families':selected,'selection_metrics':{n:metrics[n]['relative_reduction'] for n in selected.values()}},indent=2),flush=True)


def transfer():
    if (OUT/'results.json').exists():raise FileExistsError('Completed transfer cannot be overwritten')
    lock=json.loads((OUT/'fit_lock.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
    assert sha(__file__)==lock['source_sha256'] and sha(SPEC)==lock['spec_sha256'] and sha(OUT/'fitted_models.pkl')==lock['models_sha256'] and sha(OUT/'selection.json')==lock['selection_sha256']
    with (OUT/'fitted_models.pkl').open('rb') as f:models=pickle.load(f)
    frame,inventory=build([2024,2025,2026]);predictions={}
    for family,name in selection['selected_by_family'].items():
        key=name.removesuffix('_half')
        if name.startswith('expert_'):p=frame[name].to_numpy()
        elif name.startswith('online_expert_eta'):p=online_policy(frame,float(name.removeprefix('online_expert_eta')))
        else:
            p=predict_model(frame,models[key])
            if name.endswith('_half'):p=.5*p+.5*frame.forecast_naive_seconds.to_numpy()
        predictions[name]=p
    prior=json.loads((ROOT/'artifacts/research/performance_20260907/live/corrected_cycle_1/selected_models.json').read_text())['final_ridge_model']
    predictions['previous_ridge']=base.ridge_predict(frame,prior)
    results={}
    for label,mask in [('2024',frame.year.eq(2024)),('2025',frame.year.eq(2025)),('2024_2025',frame.year.isin([2024,2025])),('2026',frame.year.eq(2026)),('2026_recent',frame.event_key.ge(202610))]:
        subset=frame.loc[mask];results[label]={name:diagnostics(subset,p[mask]) for name,p in predictions.items()}
    preferred=selection['preferred'];hist=results['2024_2025'][preferred]
    ridge=results['2024_2025']['previous_ridge']['candidate_mae']
    gate={'historical_gain_at_least_10pct':hist['relative_reduction']>=.1,'beat_previous_ridge_at_least_7pct':1-hist['candidate_mae']/ridge>=.07,
          'both_historical_years_improve':all(results[y][preferred]['delta']<0 for y in ['2024','2025']), 'block_ci_upper_negative':hist['block3_ci95'][1]<0,
          'loo_all_improve':hist['loo_max_delta']<0,'current_year_point_improves':results['2026'][preferred]['delta']<0}
    for name,p in predictions.items():frame['prediction_'+name]=p
    frame.to_pickle(OUT/'transfer_data_and_forecasts.pkl')
    result={'experiment':'live_frontier_cycle_1','preferred':preferred,'selected_by_family':selection['selected_by_family'],'results':results,'gate':gate,'substantial_research_gate_passed':all(gate.values()),
            'fit_lock_sha256':sha(OUT/'fit_lock.json'),'selection_sha256':sha(OUT/'selection.json'),'input_manifest':inventory,'forecasts_sha256':sha(OUT/'transfer_data_and_forecasts.pkl'),'promotion':False}
    write(OUT/'results.json',result)
    print(json.dumps({'gate':gate,'preferred':preferred,'results':{y:{n:round(m['relative_reduction']*100,4) for n,m in r.items()} for y,r in results.items()}},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['discover','transfer']);args=parser.parse_args()
    discover() if args.phase=='discover' else transfer()
