#!/usr/bin/env python3
"""Four frozen, inexpensive pre-event challengers on chronological local events."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings
from datetime import datetime, timezone
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
PYTHON = ROOT / 'research/projects/F1/rising_qualification_prediction/Python'
sys.path[:0] = [str(ROOT), str(PYTHON)]

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge, HuberRegressor
from run_qualifying_pairwise_challenger_backtest import (
    _event_inference_frame, _qualifying_target_frame, _resolve_snapshot_path,
)
import run_qualifying_pairwise_challenger_backtest as canonical_runner
from packages.f1.features.qualifying_lap import build_quality_aware_rehearsal_features
from packages.f1.models.ultimate_lap_time.achievable import (
    ACTUAL_LAP_COLUMN, robust_huber_location,
)

HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / 'specification.json'
QUAL_FEATURES = ['rank_latest', 'rank_earlier_delta', 'rank_prior_delta',
                 'log_clean_count', 'spread_fraction', 'imputed', 'is_sq', 'rank_low_evidence']
LAP_FEATURES = ['rank_latest', 'rank_earlier_delta', 'rank_prior_delta',
                'log_clean_count', 'spread_fraction', 'is_sq', 'relative_anchor',
                'session_evolution_fraction', 'wet_fraction', 'field_iqr_fraction']
QUAL_MODELS = ['baseline_qualifying', 'Q1_fixed_rank_blend', 'Q2_ridge_rank_residual']
LAP_MODELS = ['baseline_best_lap', 'L1_fractional_source_shift', 'L2_huber_context_residual']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def canonical(value) -> bytes:
    return json.dumps(clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def lower_median(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    return float(x[(len(x)-1)//2]) if len(x) else 0.0


def permutation(values, fallback) -> np.ndarray:
    # Stable latest-rank tie-break, never a target- or driver-name-based signal.
    order = np.lexsort((np.asarray(fallback), np.asarray(values)))
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order)+1)
    return ranks


def prior_rank(history: pd.DataFrame, driver: str, team: str) -> float:
    if history.empty:
        return 0.5
    rows = history.loc[history.driver_id.eq(driver)].sort_values('event_key').tail(6)
    if rows.empty:
        rows = history.loc[history.team_id.eq(team)].groupby('event_key')['actual_rank_fraction'].mean().tail(6)
        values = rows.to_numpy(dtype=float)
    else:
        values = rows.actual_rank_fraction.to_numpy(dtype=float)
    if not len(values):
        return 0.5
    weights = 2.0 ** (-np.arange(len(values)-1, -1, -1)/3.0)
    return float(np.average(values, weights=weights))


def relocated_snapshot_path(root: Path, event_dir: Path, value: object) -> Path:
    """Recover obsolete absolute checkout prefixes without changing raw metadata."""
    candidate=Path(str(value)).expanduser()
    choices=[candidate if candidate.is_absolute() else root/candidate,event_dir/candidate.name]
    for choice in choices:
        choice=choice.resolve()
        if choice.is_file() and root.resolve() in choice.parents:
            return choice
    raise FileNotFoundError(f'no repository snapshot for {value!r} in {event_dir}')


def inference_features(event_dir: Path, history: pd.DataFrame):
    # Process-local adapter only: the canonical source and snapshots stay immutable.
    with patch.object(canonical_runner,'_resolve_snapshot_path',relocated_snapshot_path):
        frame, info, paths, target_path = _event_inference_frame(ROOT, event_dir)
    frame = frame.reset_index(drop=True)
    metadata = json.loads((event_dir/'weekend_metadata.json').read_text())
    q = next(s for s in metadata['sessions'] if s['session_type']=='qualifying')
    before = [s for s in metadata['sessions'] if int(s['session_order']) < int(q['session_order'])]
    latest = max((s for s in before if s['session_name'].lower()=='practice 3'
                  or s['session_type']=='sprint_qualifying'), key=lambda s:int(s['session_order']))
    earlier = [s for s in before if int(s['session_order']) < int(latest['session_order'])
               and 'practice' in s['session_name'].lower()]
    n = len(frame)
    frame['rank_latest'] = (frame.latest_qualifying_rehearsal_rank.astype(float)-1)/max(n-1, 1)
    frame['raw_anchor'] = pd.to_numeric(frame.valid_clean_best_seconds, errors='coerce')
    frame['duration'] = float(frame.raw_anchor.median())
    if not np.isfinite(frame.duration.iloc[0]):
        raise ValueError('no finite raw rehearsal duration')
    frame['rank_earlier'] = frame.rank_latest
    frame['session_evolution_fraction'] = 0.0
    if earlier:
        previous = max(earlier, key=lambda s:int(s['session_order']))
        path = relocated_snapshot_path(ROOT,event_dir,previous['laps_path'])
        laps = pd.read_csv(path); laps['rehearsal_source']=previous['session_name']
        ef = build_quality_aware_rehearsal_features(laps, entrants=frame[['driver_id','team_id']], official_session_timing=True)
        ef = ef.set_index('driver_id').reindex(frame.driver_id)
        anchors = pd.to_numeric(ef.valid_clean_best_seconds,errors='coerce')
        earlier_ranks = (anchors.rank(method='average')-1)/max(anchors.notna().sum()-1,1)
        frame['rank_earlier'] = earlier_ranks.to_numpy()
        frame['rank_earlier'] = frame.rank_earlier.fillna(frame.rank_latest)
        previous_median = float(anchors.median())
        frame['session_evolution_fraction'] = (frame.duration.iloc[0]-previous_median)/frame.duration.iloc[0]
        paths.append(path)
    frame['rank_prior'] = [prior_rank(history,str(d),str(t)) for d,t in zip(frame.driver_id,frame.team_id)]
    frame['rank_earlier_delta'] = frame.rank_earlier-frame.rank_latest
    frame['rank_prior_delta'] = frame.rank_prior-frame.rank_latest
    count = pd.to_numeric(frame.valid_clean_lap_count,errors='coerce').fillna(0).clip(lower=0)
    frame['log_clean_count'] = np.log1p(count)
    frame['spread_fraction'] = pd.to_numeric(frame.best_two_spread_seconds,errors='coerce')/frame.duration
    frame['imputed'] = frame.anchor_is_imputed.astype(float)
    frame['is_sq'] = float(info['rehearsal_source']=='sprint_qualifying')
    frame['rank_low_evidence'] = frame.rank_latest/(1+count)
    frame['relative_anchor'] = (frame.raw_anchor-frame.duration)/frame.duration
    frame['wet_fraction'] = float(frame.best_lap_compound.fillna('').astype(str).str.upper().isin(['INTER','INTERMEDIATE','WET']).mean())
    frame['field_iqr_fraction'] = float((frame.raw_anchor.quantile(.75)-frame.raw_anchor.quantile(.25))/frame.duration.iloc[0])
    frame['year'] = info['year']; frame['round'] = info['round']
    return frame,info,paths,target_path


def fitted_residual(history, current, columns, target, *, huber=False):
    usable = history.loc[pd.to_numeric(history[target],errors='coerce').notna()].copy()
    if usable.event_key.nunique()<12:
        return None, {'status':'baseline_fallback_insufficient_events','training_events':int(usable.event_key.nunique())}
    x = usable[columns].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    z = current[columns].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    count = usable.groupby('event_key').event_key.transform('size').to_numpy(dtype=float)
    weights = 1/count
    finite = np.isfinite(x)
    centers = np.divide(np.where(finite,x,0).T@weights,finite.T@weights,
                        out=np.zeros(len(columns)),where=(finite.T@weights)>0)
    x = np.where(finite,x,centers); z = np.where(np.isfinite(z),z,centers)
    scales = np.sqrt(np.average((x-centers)**2,axis=0,weights=weights)); scales[scales<1e-8]=1
    model = HuberRegressor(epsilon=1.35,alpha=1,max_iter=1000) if huber else Ridge(alpha=10)
    y = usable[target].to_numpy(dtype=float)
    if huber:y = np.clip(y,-25,25)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit((x-centers)/scales,y,sample_weight=weights)
    pred = model.predict((z-centers)/scales)
    info = {'status':'fitted','training_events':int(usable.event_key.nunique()),
            'training_event_keys':sorted(usable.event_key.unique().astype(int).tolist()),
            'coefficients':dict(zip(columns,model.coef_.tolist())), 'intercept':float(model.intercept_),
            'centers':centers.tolist(),'scales':scales.tolist(),
            'warnings':[str(w.message) for w in caught]}
    return pred,info


def predict_event(frame: pd.DataFrame, history: pd.DataFrame):
    if not history.empty and history.event_key.max()>=frame.event_key.iloc[0]:
        raise ValueError('history must precede event')
    if ACTUAL_LAP_COLUMN in frame or 'qualy_position' in frame:
        raise ValueError('current inputs contain target columns')
    p = frame[['event_key','driver_id','team_id','year','round','rehearsal_source',
               'raw_anchor','duration','rank_latest','wet_fraction','valid_clean_lap_count','anchor_is_imputed']].copy()
    p['baseline_qualifying'] = frame.latest_qualifying_rehearsal_rank.astype(int)
    blend = np.where(frame.is_sq.eq(1),.85*frame.rank_latest+.15*frame.rank_prior,
                     .65*frame.rank_latest+.20*frame.rank_earlier+.15*frame.rank_prior)
    p['Q1_fixed_rank_blend'] = permutation(blend,p.baseline_qualifying)
    fit = {}
    try:
        ridge,fit['Q2_ridge_rank_residual'] = fitted_residual(history,frame,QUAL_FEATURES,'rank_residual') if not history.empty else (None,{'status':'baseline_fallback_empty_history'})
    except Exception as exc:
        ridge=None;fit['Q2_ridge_rank_residual']={'status':'fit_failed_baseline_substitute','error':repr(exc)}
    p['Q2_ridge_rank_residual'] = permutation(frame.rank_latest+(np.clip(ridge,-.35,.35) if ridge is not None else 0),p.baseline_qualifying)
    same = history.loc[history.year.eq(frame.year.iloc[0]) & history.rehearsal_source.eq(frame.rehearsal_source.iloc[0])] if not history.empty else history
    shifts=[];fractions=[]
    if not same.empty:
        for _,rows in same.groupby('event_key',sort=True):
            shift=robust_huber_location(rows[ACTUAL_LAP_COLUMN]-rows.raw_anchor)
            if np.isfinite(shift):shifts.append(shift);fractions.append(shift/rows.duration.iloc[0])
    p['baseline_best_lap'] = frame.raw_anchor+lower_median(shifts)
    p['L1_fractional_source_shift'] = frame.raw_anchor+lower_median(fractions)*frame.duration
    try:
        adjustment,fit['L2_huber_context_residual'] = fitted_residual(history,frame,LAP_FEATURES,'lap_residual_percent',huber=True) if not history.empty else (None,{'status':'baseline_fallback_empty_history'})
    except Exception as exc:
        adjustment=None;fit['L2_huber_context_residual']={'status':'fit_failed_baseline_substitute','error':repr(exc)}
    p['L2_huber_context_residual'] = (frame.raw_anchor+np.clip(adjustment,-25,25)*frame.duration/100) if adjustment is not None else p.baseline_best_lap
    for col in QUAL_MODELS:
        assert sorted(p[col].tolist())==list(range(1,len(p)+1)),col
    return p,fit


def paired(values,seed):
    values=np.asarray(values,dtype=float);n=len(values)
    if not n:return None
    rng=np.random.default_rng(seed)
    means=values[rng.integers(n,size=(20000,n))].mean(axis=1)
    loo=(values.sum()-values)/(n-1) if n>1 else np.array([np.nan])
    # Circular block sampling preserves the sample size and permits every start.
    starts=rng.integers(n,size=(20000,int(np.ceil(n/3))))
    indices=((starts[...,None]+np.arange(3))%n).reshape(20000,-1)[:,:n]
    bm=values[indices].mean(axis=1)
    return {'events':n,'delta_mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
            'bootstrap_fraction_improving':float((means<0).mean()),
            'loo_min':float(loo.min()),'loo_max':float(loo.max()),
            'event_wins':int((values<0).sum()),'event_ties':int((values==0).sum()),
            'three_event_circular_block_ci95':np.quantile(bm,[.025,.975]).tolist(),
            'interpretation':'historical paired resampling frequency, not a posterior or promotion test'}


def run(out: Path):
    spec=json.loads(SPEC_PATH.read_text());spec_sha=digest(SPEC_PATH)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError(out/'results.json')
    inputs={};history_parts=[];forecasts=[];events=[];excluded=[];attempts=[];inventory={}
    dirs=sorted((ROOT/'data/f1/raw/weekends').glob('20*/round_*'))
    dirs=[d for d in dirs if d.is_dir() and (d/'weekend_metadata.json').is_file() and 2022<=int(d.parent.name)<=2026]
    for d in dirs:inventory[d.parent.name]=inventory.get(d.parent.name,0)+1
    for event_dir in dirs:
        event_key=int(event_dir.parent.name)*100+int(event_dir.name.split('_')[1])
        inputs[str((event_dir/'weekend_metadata.json').relative_to(ROOT))]=digest(event_dir/'weekend_metadata.json')
        history=pd.concat(history_parts,ignore_index=True) if history_parts else pd.DataFrame()
        try:
            frame,info,paths,target_path=inference_features(event_dir,history)
            for path in paths:
                if path != target_path:
                    inputs[str(path.relative_to(ROOT))]=digest(path)
            frozen,fit=predict_event(frame,history)
            # No target I/O above this point. Retain a cryptographic forecast binding.
            forecast_sha=hashlib.sha256(canonical(frozen.to_dict('records'))).hexdigest()
            target,target_info=_qualifying_target_frame(target_path)
            inputs[str(target_path.relative_to(ROOT))]=digest(target_path)
            if set(frame.driver_id)!=set(target.driver_id):
                excluded.append({'event_key':event_key,'reason':'inference_target_roster_mismatch',
                                 'forecast_rows':len(frame),'target_rows':len(target),
                                 'missing_in_inference':sorted(set(target.driver_id)-set(frame.driver_id)),
                                 'extra_in_inference':sorted(set(frame.driver_id)-set(target.driver_id)),
                                 'frozen_forecast_sha256':forecast_sha})
                continue
            scored=frozen.merge(target[['driver_id','qualy_position',ACTUAL_LAP_COLUMN]],on='driver_id',validate='one_to_one')
            if not np.isfinite(scored.qualy_position).all():raise ValueError('nonfinite qualifying position')
            metrics={'event_key':event_key,**info,'forecast_sha256':forecast_sha,
                     'target_read_after_forecast_frozen':True,'training_event_count':int(history.event_key.nunique()) if not history.empty else 0}
            for col in QUAL_MODELS:
                metrics[col+'_mae']=float((scored[col]-scored.qualy_position).abs().mean())
                metrics[col+'_kendall']=float(kendalltau(scored[col],scored.qualy_position).statistic)
                metrics[col+'_top3_overlap']=len(set(scored.nsmallest(3,col).driver_id)&set(scored.nsmallest(3,'qualy_position').driver_id))/3
            lap_mask=np.isfinite(scored[ACTUAL_LAP_COLUMN]) & np.isfinite(scored[LAP_MODELS]).all(axis=1)
            metrics['lap_scored_rows']=int(lap_mask.sum());metrics['lap_forecast_rows']=len(scored)
            metrics['lap_target_missing_rows']=int(scored[ACTUAL_LAP_COLUMN].isna().sum())
            metrics['lap_raw_rehearsal_missing_rows']=int(scored.raw_anchor.isna().sum())
            metrics['wet_rehearsal_fraction']=float(frame.wet_fraction.iloc[0])
            metrics['clean_lap_count_median']=float(frame.valid_clean_lap_count.median())
            for col in LAP_MODELS:
                metrics[col+'_mae']=float((scored.loc[lap_mask,col]-scored.loc[lap_mask,ACTUAL_LAP_COLUMN]).abs().mean())
            scored['lap_scored']=lap_mask;forecasts.append(scored);events.append(metrics)
            attempts.append({'event_key':event_key,'models':fit})
            labeled=frame.merge(target[['driver_id','qualy_position',ACTUAL_LAP_COLUMN]],on='driver_id',validate='one_to_one')
            labeled['actual_rank_fraction']=(labeled.qualy_position-1)/max(len(labeled)-1,1)
            labeled['rank_residual']=labeled.actual_rank_fraction-labeled.rank_latest
            labeled['lap_residual_percent']=100*(labeled[ACTUAL_LAP_COLUMN]-labeled.raw_anchor)/labeled.duration
            history_parts.append(labeled)
            print(event_key,'rows',len(frame),'prior',metrics['training_event_count'],flush=True)
        except Exception as exc:
            excluded.append({'event_key':event_key,'reason':repr(exc)})
            print(event_key,'EXCLUDED',repr(exc),flush=True)
    event_frame=pd.DataFrame(events);predictions=pd.concat(forecasts,ignore_index=True)
    summaries={}
    for label,mask in [('warmup_2022_2023',event_frame.year.le(2023)),('2024',event_frame.year.eq(2024)),
                       ('2025',event_frame.year.eq(2025)),('primary_2024_2025',event_frame.year.isin([2024,2025])),
                       ('exposed_2026',event_frame.year.eq(2026))]:
        ev=event_frame.loc[mask].sort_values('event_key');summary={'events':len(ev),'event_keys':ev.event_key.tolist(),
            'roster_rows':int(ev.field_size.sum()),'lap_scored_rows':int(ev.lap_scored_rows.sum()),
            'lap_target_missing_rows':int(ev.lap_target_missing_rows.sum()),'lap_raw_rehearsal_missing_rows':int(ev.lap_raw_rehearsal_missing_rows.sum()),
            'point_metrics':{},'paired':{}}
        for col in QUAL_MODELS+LAP_MODELS:summary['point_metrics'][col]=float(ev[col+'_mae'].mean())
        for col,base in [(c,QUAL_MODELS[0]) for c in QUAL_MODELS[1:]]+[(c,LAP_MODELS[0]) for c in LAP_MODELS[1:]]:
            summary['paired'][col]=paired((ev[col+'_mae']-ev[base+'_mae']).to_numpy(),spec['seed'])
        summaries[label]=summary
    error_structure={}
    primary=event_frame.loc[event_frame.year.isin([2024,2025])].copy()
    primary['wet_rehearsal']=primary.wet_rehearsal_fraction.gt(.2)
    for group in ['rehearsal_source','wet_rehearsal']:
        error_structure[group]=[{str(group):str(key),'events':len(g),
             **{c:float(g[c+'_mae'].mean()) for c in QUAL_MODELS+LAP_MODELS}} for key,g in primary.groupby(group)]
    source_files={str(SPEC_PATH.relative_to(ROOT)):spec_sha,str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}
    # Hash every loaded project Python module, not only the entry-point imports.
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            path=Path(path).resolve()
            if ROOT in path.parents and path.is_file() and path.suffix=='.py' and '.venv' not in str(path.relative_to(ROOT)):
                source_files[str(path.relative_to(ROOT))]=digest(path)
    result={'schema_version':'f1_pre_event_performance_experiment_v1','generated_at':datetime.now(timezone.utc).isoformat(),
            'specification_sha256':spec_sha,'specification':spec,'inventory_events':inventory,'summaries':summaries,
            'baseline_error_structure':error_structure,'events':events,'excluded_events':excluded,'fit_attempts':attempts,
            'all_tried_candidates':list(spec['candidates']),'selection_performed':False,'promotion':False,
            'execution_repairs': ['Initial attempt preserved at cycle1_execution.log: obsolete absolute snapshot prefixes skipped older files; final manifest encountered generated PyTorch _ops.py. Same candidate specification rerun with event-local snapshot relocation and physical-source-only hashing.'],
            'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
                       'thread_limits':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}},
            'implementation_manifest':source_files,'input_manifest':inputs}
    for name,obj in [('predictions.csv',predictions),('events.csv',event_frame)]:obj.to_csv(out/name,index=False)
    result['output_manifest']={str((out/name).relative_to(ROOT)):digest(out/name) for name in ['predictions.csv','events.csv']}
    (out/'results.json').write_text(json.dumps(clean(result),indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(clean(summaries),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/research/performance_20260907/pre_event/cycle1')
    run(parser.parse_args().output.resolve())
