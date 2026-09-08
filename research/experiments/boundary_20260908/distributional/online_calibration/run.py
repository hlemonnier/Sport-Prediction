"""Frozen four-policy online quantile calibration on inherited original issuances."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from research.experiments.boundary_20260908.distributional.execution import run as parent
from research.experiments.boundary_20260908.distributional.execution import model as scores
from research.experiments.boundary_20260908.distributional.online_calibration import model
HERE=Path(__file__).resolve().parent
ROOT=parent.ROOT
OUT=ROOT/'artifacts/research/boundary_20260908/distributional/online_calibration'
OLD=parent.OUT
SPEC=HERE/'specification.json'
STATIC=('global_empirical','stratified_empirical','selected_quantile')
sha,write,now,digest=parent.sha,parent.write,parent.now,parent.digest


def sources():
    paths=[*sorted(HERE.glob('*.py')),SPEC,Path(scores.__file__),Path(parent.__file__)]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def spec():
    lock=json.loads((OUT/'design_lock.json').read_text());assert lock['spec_sha256']==sha(SPEC)
    for p,s in lock['inputs'].items():assert sha(ROOT/p)==s
    frozen=json.loads((OUT/'execution_lock.json').read_text());assert frozen['source_sha256']==digest(sources())
    return json.loads(SPEC.read_text()),parent.specification()


def load(filename):
    with (OLD/filename).open('rb') as f:return pickle.load(f)


def save(filename,obj):
    with (OUT/filename).open('wb') as f:pickle.dump(obj,f)


def original(phase):
    data=load('selection_forecasts.pkl' if phase=='selection' else 'transfer_forecasts.pkl')
    chosen=json.loads((OLD/'selection.json').read_text())['selected']
    return data['frame'],{**{k:data['quantiles'][k] for k in STATIC[:2]},'selected_quantile':data['quantiles'][chosen]}


def begin(phase):
    if (OUT/f'{phase}_attempt.json').exists():raise FileExistsError('Preserve existing attempt')
    write(OUT/f'{phase}_attempt.json',{'started_at_utc':now(),'source_sha256':digest(sources()),'spec_sha256':sha(SPEC)})


def summarize(frame,predictions,candidate,refs,metric_spec):
    metrics={k:scores.metrics(frame,q,metric_spec) for k,q in predictions.items()}
    return {'models':metrics,'relative_wis_gain':{r:1-metrics[candidate]['wis']/metrics[r]['wis'] for r in refs},'paired':{r:scores.paired(frame,predictions[candidate],predictions[r],metric_spec) for r in refs}}


def discover():
    protocol,metric_spec=spec();begin('selection');frame,pred=original('selection');assert set(frame.year)=={2023}
    audit={};diagnostics={}
    for name,config in protocol['variants'].items():
        for base,prefix in [('selected_quantile','quantile'),('stratified_empirical','conditional')]:
            key=prefix+'__'+name;pred[key],audit[key],diagnostics[key]=model.replay(frame,pred[base],config)
        print('replayed',name,flush=True)
    metrics={k:scores.metrics(frame,q,metric_spec) for k,q in pred.items()}
    chosen=min(protocol['variants'],key=lambda n:(metrics['quantile__'+n]['wis'],n));best_cond=min(protocol['variants'],key=lambda n:(metrics['conditional__'+n]['wis'],n))
    candidate='quantile__'+chosen;refs=list(dict.fromkeys([*STATIC,'conditional__'+chosen,'conditional__'+best_cond]))
    summary=summarize(frame,pred,candidate,refs,metric_spec)
    advanced=all(v>=.01 for v in summary['relative_wis_gain'].values()) and parent.coverage_pass(metrics[candidate],metric_spec)
    save('selection_forecasts.pkl',{'predictions':pred,'audit':audit})
    result={'selected':chosen,'selected_key':candidate,'best_conditional':best_cond,'references':refs,'advancement_passed':bool(advanced),'coverage_passed':parent.coverage_pass(metrics[candidate],metric_spec),'summary':summary,'diagnostics':diagnostics,'spec_sha256':sha(SPEC),'source_sha256':digest(sources()),'forecast_sha256':sha(OUT/'selection_forecasts.pkl'),'completed_at_utc':now()}
    write(OUT/'selection.json',result)
    write(OUT/'selection_lock.json',{'selected':chosen,'best_conditional':best_cond,'references':refs,'selection_sha256':sha(OUT/'selection.json'),'source_sha256':digest(sources()),'advancement_passed':bool(advanced),'locked_at_utc':now(),'new_transfer_calibration_scores_inspected':False})
    if not advanced:write(OUT/'results.json',{'status':'rejected_on_2023_online_calibration_screen','selected':chosen,'selection_sha256':sha(OUT/'selection.json'),'transfer_evaluated':False,'substantial_gate_passed':False,'promotion':False,'point_mae_gain_claimed':False,'source_sha256':digest(sources())})
    print(json.dumps({'selected':chosen,'advanced':bool(advanced),'gains':summary['relative_wis_gain'],'wis':{k:v['wis'] for k,v in metrics.items()}},indent=2),flush=True)


def transfer():
    protocol,metric_spec=spec()
    if (OUT/'results.json').exists():raise FileExistsError('Frozen result exists')
    selection=json.loads((OUT/'selection.json').read_text());lock=json.loads((OUT/'selection_lock.json').read_text())
    assert lock['advancement_passed'] and lock['selection_sha256']==sha(OUT/'selection.json') and lock['source_sha256']==digest(sources())
    begin('transfer');frame,pred=original('transfer');assert set(frame.year)=={2024,2025,2026};audit={};diags={}
    chosen=selection['selected'];candidate=selection['selected_key'];refs=selection['references']
    for name,base,prefix in [(chosen,'selected_quantile','quantile'),*[(n,'stratified_empirical','conditional') for n in dict.fromkeys([chosen,selection['best_conditional']])]]:
        key=prefix+'__'+name;pred[key],audit[key],diags[key]=model.replay(frame,pred[base],protocol['variants'][name])
    summaries={name:summarize(frame.loc[mask],{k:q[mask] for k,q in pred.items()},candidate,refs,metric_spec) for name,mask in [('2024',frame.year.eq(2024)),('2025',frame.year.eq(2025)),('2024_2025',frame.year.isin([2024,2025])),('2026',frame.year.eq(2026)),('2026_recent',frame.year.eq(2026)&frame.event_key.ge(202610))]}
    historic=summaries['2024_2025'];gates={}
    for ref in refs:
        threshold=.1 if ref in STATIC[:2] else 0.
        gates['historical_gain_vs_'+ref]=historic['relative_wis_gain'][ref]>=threshold if threshold else historic['relative_wis_gain'][ref]>0
        gates['historical_negative_block_ci_vs_'+ref]=historic['paired'][ref]['block3_ci95'][1]<0
        gates['each_year_improves_vs_'+ref]=all(summaries[y]['relative_wis_gain'][ref]>0 for y in ['2024','2025','2026'])
    gates['original_coverage_constraint']=all(summaries[y]['models'][candidate]['coverage'][str(q)]>=max(q-.05,summaries[y]['models']['stratified_empirical']['coverage'][str(q)]-.03) for y in ['2024','2025','2026'] for q in [.8,.9,.95]);gates['fixed_point']=True
    save('transfer_forecasts.pkl',{'predictions':pred,'audit':audit})
    write(OUT/'results.json',{'status':'completed_online_calibration_transfer','selected':chosen,'selected_key':candidate,'references':refs,'transfer_evaluated':True,'summaries':summaries,'gates':gates,'substantial_gate_passed':all(gates.values()),'promotion':False,'point_mae_gain_claimed':False,'forecast_sha256':sha(OUT/'transfer_forecasts.pkl'),'selection_sha256':sha(OUT/'selection.json'),'source_sha256':digest(sources()),'diagnostics':diags,'completed_at_utc':now()})
    print(json.dumps({'gates':gates,'gains':{k:v['relative_wis_gain'] for k,v in summaries.items()}},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['discover','transfer']);args=p.parse_args();{'discover':discover,'transfer':transfer}[args.phase]()
