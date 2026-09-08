"""One bounded dynamic hierarchical pre-qualifying research experiment."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
OLD_PATH=ROOT/'research/experiments/performance_20260907/pre_event/run_experiment.py'
old_spec=importlib.util.spec_from_file_location('prior_pre_event',OLD_PATH)
old=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old)
HERE=Path(__file__).resolve().parent
SPEC_PATH=HERE/'spec.json'
REFERENCE=ROOT/'artifacts/research/performance_20260907/pre_event/cycle1/results.json'
PREDICTIONS=REFERENCE.with_name('predictions.csv')
COMPARATORS=['baseline_qualifying','Q1_fixed_rank_blend','Q2_ridge_rank_residual']
NUMERIC=['earlier_delta','prior_delta','latest','imputed','spread','potential_delta','sq_earlier','sq_prior']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(value):return old.canonical(value)


def model_features(frame):
    n=len(frame);x=pd.DataFrame(index=frame.index)
    x['earlier_delta']=19*(frame.rank_earlier-frame.rank_latest)/5
    x['prior_delta']=19*(frame.rank_prior-frame.rank_latest)/5
    x['latest']=frame.rank_latest
    x['imputed']=frame.anchor_is_imputed.astype(float)
    x['spread']=pd.to_numeric(frame.best_two_spread_seconds,errors='coerce').div(frame.duration).clip(0,.1)*100
    anchors=pd.to_numeric(frame.latent_potential_adjusted_anchor_seconds,errors='coerce')
    potential=(anchors.rank(method='average')-1)/max(n-1,1)
    x['potential_delta']=19*(potential-frame.rank_latest)/5
    x['sq_earlier']=x.earlier_delta*frame.is_sq
    x['sq_prior']=x.prior_delta*frame.is_sq
    x=x.replace([np.inf,-np.inf],np.nan).fillna(0.)
    x=x-x.mean(axis=0)
    return x


def design(frame,teams,drivers):
    x=frame[NUMERIC].to_numpy(dtype=float)
    groups=[]
    for column,levels in [('team_id',teams),('driver_id',drivers)]:
        z=frame[column].astype(str).to_numpy()[:,None]==np.asarray(levels)[None,:]
        z=z.astype(float)
        # Center separately within each event; no label is used in this design.
        for _,idx in frame.groupby('event_key',sort=False).indices.items():z[idx]-=z[idx].mean(axis=0)
        groups.append(z)
    return np.column_stack([x,*groups])


def objective(beta,x,y,w,penalty):
    r=np.abs(y-x@beta)
    return float(np.sum(w*np.where(r<=3,.5*r*r,3*(r-1.5)))+.5*np.dot(penalty*beta,beta))


def solve(x,y,w,penalty):
    beta=np.zeros(x.shape[1]);previous=objective(beta,x,y,w,penalty)
    for iteration in range(40):
        r=np.abs(y-x@beta);robust=np.minimum(1,3/np.maximum(r,1e-12))
        weight=w*robust
        new=np.linalg.solve(x.T@(weight[:,None]*x)+np.diag(penalty),x.T@(weight*y))
        value=objective(new,x,y,w,penalty)
        assert value<=previous+1e-7,(value,previous)
        if np.max(np.abs(new-beta))<1e-8:
            beta=new;break
        beta=new;previous=value
    gradient=x.T@(w*np.clip(x@beta-y,-3,3))+penalty*beta
    return beta,{'iterations':iteration+1,'objective':objective(beta,x,y,w,penalty),
                 'gradient_max_abs':float(np.max(np.abs(gradient)))}


def predict(frame,history,config):
    if 'qualy_position' in frame:raise ValueError('current target supplied')
    key=int(frame.event_key.iloc[0]);year=key//100
    if not history.empty and int(history.event_key.max())>=key:raise ValueError('history is not earlier')
    same=history.loc[history.event_key.floordiv(100).eq(year)].copy() if not history.empty else history
    if same.empty:return frame.latest_qualifying_rehearsal_rank.to_numpy(dtype=int),{'status':'zero_state_season_start','training_event_keys':[]}
    teams=sorted(set(same.team_id.astype(str))|set(frame.team_id.astype(str)))
    drivers=sorted(set(same.driver_id.astype(str))|set(frame.driver_id.astype(str)))
    x=design(same,teams,drivers);z=design(frame,teams,drivers)
    keys=sorted(same.event_key.unique());age={k:len(keys)-i for i,k in enumerate(keys)}
    weights=2.**(-same.event_key.map(age).to_numpy()/config['half_life_events'])
    weights*=20/same.groupby('event_key').event_key.transform('size').to_numpy()
    penalty=config['lambda']*np.array([1.]*len(NUMERIC)+[2.]*len(teams)+[4.]*len(drivers))
    beta,diagnostic=solve(x,same.residual_equivalent_positions.to_numpy(),weights,penalty)
    correction=config['strength']*np.clip(z@beta,-5,5)/19
    ranks=old.permutation(frame.rank_latest+correction,frame.latest_qualifying_rehearsal_rank)
    return ranks,{'status':'fitted',**diagnostic,'training_event_keys':list(map(int,keys)),
                  'team_levels':teams,'driver_levels':drivers,'coefficients':beta.tolist(),
                  'mean_abs_correction_equivalent_positions':float(np.mean(np.abs(correction))*19)}


def grid(spec):
    g=spec['hyperparameter_grid']
    return {f'h{h}_l{l:g}_s{s:g}':{'half_life_events':h,'lambda':l,'strength':s}
            for h in g['half_life_events'] for l in g['lambda'] for s in g['strength']}


def select(events,configs):
    selection=[e for e in events if e['year']==2023]
    assert selection
    scores={name:float(np.mean([e['models'][name]['mae'] for e in selection])) for name in configs}
    best=min(scores.values());ties=[name for name in scores if scores[name]<=best+1e-12]
    chosen=min(ties,key=lambda name:(-configs[name]['lambda'],configs[name]['strength'],configs[name]['half_life_events']))
    return {'selected':chosen,'parameters':configs[chosen],'selection_event_keys':[e['event_key'] for e in selection],
            'selection_variant_event_MAE':scores,'target_boundary':'all selected event keys <202400'}


def main(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError(out/'results.json')
    spec=json.loads(SPEC_PATH.read_text());spec_hash=sha(SPEC_PATH);code_hash=sha(Path(__file__))
    ref=json.loads(REFERENCE.read_text())
    assert sha(PREDICTIONS)==ref['output_manifest'][str(PREDICTIONS.relative_to(ROOT))]
    reference_source_drift=[{'path':p,'historical_sha256':h,'current_sha256':sha(ROOT/p)}
                            for p,h in ref['implementation_manifest'].items() if sha(ROOT/p)!=h]
    sources={str(SPEC_PATH.relative_to(ROOT)):spec_hash,str(Path(__file__).relative_to(ROOT)):code_hash}
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value:
            path=Path(value).resolve()
            if path.is_file() and ROOT in path.parents and path.suffix=='.py' and '.venv' not in str(path.relative_to(ROOT)):
                sources[str(path.relative_to(ROOT))]=sha(path)
    reference=pd.read_csv(PREDICTIONS,float_precision='round_trip',usecols=['event_key','driver_id',*COMPARATORS])
    configs=grid(spec);parts=[];records=[];events=[];excluded=[];inputs={};lock=None;failed=[]
    paths=sorted((ROOT/'data/f1/raw/weekends').glob('20*/round_*/weekend_metadata.json'))
    paths=[p for p in paths if 2023<=int(p.parent.parent.name)<=2026]
    for path in paths:
        year=int(path.parent.parent.name);key=year*100+int(path.parent.name.split('_')[1])
        if year>=2024 and lock is None:
            lock=select(events,configs)
            (out/'selection_lock.json').write_text(json.dumps(lock,indent=2,allow_nan=False)+'\n')
        history=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
        same=history.loc[history.event_key.floordiv(100).eq(year)] if not history.empty else history
        inputs[str(path.relative_to(ROOT))]=sha(path)
        try:
            frame,info,files,target_path=old.inference_features(path.parent,same)
            numeric=model_features(frame)
            for name in NUMERIC:frame[name]=numeric[name]
            active=configs if year==2023 else {lock['selected']:lock['parameters']}
            pred=frame[['event_key','driver_id']].copy();fits={}
            for name,config in active.items():
                try:
                    pred[name],fits[name]=predict(frame,history,config)
                except Exception as exc:
                    failed.append({'event_key':key,'variant':name,'error':repr(exc)})
                    raise
            pred_hash=hashlib.sha256(pack(pred.to_dict('records'))).hexdigest()
            for f in files:
                if f!=target_path:inputs[str(f.relative_to(ROOT))]=sha(f)
            target,_=old._qualifying_target_frame(target_path);inputs[str(target_path.relative_to(ROOT))]=sha(target_path)
            assert inputs[str(target_path.relative_to(ROOT))]==ref['input_manifest'][str(target_path.relative_to(ROOT))],'reference target bytes changed'
            assert set(frame.driver_id)==set(target.driver_id),'target roster mismatch'
            scored=pred.merge(target[['driver_id','qualy_position']],on='driver_id',validate='one_to_one')
            comparator=reference.loc[reference.event_key.eq(key),['driver_id',*COMPARATORS]]
            assert set(scored.driver_id)==set(comparator.driver_id),'reference roster mismatch'
            scored=scored.merge(comparator,on='driver_id',validate='one_to_one',suffixes=('','_reference'))
            models={}
            for name in [*active,*COMPARATORS]:
                assert sorted(scored[name].tolist())==list(range(1,len(scored)+1))
                models[name]={'mae':float((scored[name]-scored.qualy_position).abs().mean()),
                              'kendall':float(kendalltau(scored[name],scored.qualy_position).statistic),
                              'top3_overlap':len(set(scored.nsmallest(3,name).driver_id)&set(scored.nsmallest(3,'qualy_position').driver_id))/3}
            if year>=2024:scored['selected_dynamic']=scored[lock['selected']]
            scored['year']=year;records.append(scored)
            events.append({'event_key':key,'year':year,'rows':len(scored),'source':info['rehearsal_source'],
                           'models':models,'fits':fits,'forecast_sha256':pred_hash})
            labeled=frame.merge(target[['driver_id','qualy_position',old.ACTUAL_LAP_COLUMN]],on='driver_id',validate='one_to_one')
            labeled['actual_rank_fraction']=(labeled.qualy_position-1)/max(len(labeled)-1,1)
            labeled['residual_equivalent_positions']=19*(labeled.actual_rank_fraction-labeled.rank_latest)
            parts.append(labeled)
            print(key,'scored',len(scored),'active',len(active),flush=True)
        except Exception as exc:
            excluded.append({'event_key':key,'error':repr(exc)})
            print(key,'EXCLUDED',repr(exc),flush=True)
    assert lock is not None
    assert not failed,failed
    assert len([e for e in events if e['year']==2024])==24
    assert len([e for e in events if e['year']==2025])==24
    assert len([e for e in events if e['year']==2026])==9
    summaries={}
    for label,years in [('2024',[2024]),('2025',[2025]),('pooled_2024_2025',[2024,2025]),('exposed_2026',[2026])]:
        es=[e for e in events if e['year'] in years];name=lock['selected'];models=[name,*COMPARATORS]
        summary={'events':len(es),'rows':sum(e['rows'] for e in es),
                 'event_keys':[e['event_key'] for e in es],
                 'mean_MAE':{m:float(np.mean([e['models'][m]['mae'] for e in es])) for m in models},
                 'mean_Kendall':{m:float(np.mean([e['models'][m]['kendall'] for e in es])) for m in models},
                 'paired_vs':{}}
        for comparator in COMPARATORS:
            delta=[e['models'][name]['mae']-e['models'][comparator]['mae'] for e in es]
            summary['paired_vs'][comparator]=old.paired(delta,20260908)
        summaries[label]=summary
    output=pd.concat(records,ignore_index=True);output.to_csv(out/'predictions.csv',index=False)
    for p,h in sources.items():assert sha(ROOT/p)==h,('source changed during run',p)
    assert sha(SPEC_PATH)==spec_hash and sha(Path(__file__))==code_hash
    result={'schema_version':'dynamic_hierarchical_rehearsal_bias_v1','generated_at':datetime.now(timezone.utc).isoformat(),
            'specification':spec,'specification_sha256':spec_hash,'selection':lock,'summaries':summaries,
            'events':events,'excluded_events':excluded,'failed_variants':failed,'all_tried_configurations':configs,
            'promotion':False,'implementation_manifest':sources,'input_manifest':inputs,
            'reference_artifact':{'path':str(REFERENCE.relative_to(ROOT)),'sha256':sha(REFERENCE)},
            'reference_predictions':{'path':str(PREDICTIONS.relative_to(ROOT)),'sha256':sha(PREDICTIONS)},
            'reference_source_drift':reference_source_drift,
            'reference_scope':'Compare immutable previously verified prediction bytes on unchanged target bytes; historical broad source closure includes five modules changed by subsequent frontier integration and is not relabelled as current.',
            'output_manifest':{str((out/p).relative_to(ROOT)):sha(out/p) for p in ['predictions.csv','selection_lock.json']}}
    result['material_success_screen']=bool(summaries['pooled_2024_2025']['paired_vs']['Q2_ridge_rank_residual']['delta_mean']<=-.15
                                            and summaries['exposed_2026']['paired_vs']['Q1_fixed_rank_blend']['delta_mean']<=0)
    (out/'results.json').write_text(json.dumps(old.clean(result),indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'selected':lock,'summaries':summaries,'material_success_screen':result['material_success_screen']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/research/boundary_20260908/pre_event/cycle1')
    main(parser.parse_args().output.resolve())
