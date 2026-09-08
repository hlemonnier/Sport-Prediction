"""Independent issue-driven heap replay of delayed quantile feedback."""
import heapq
import json
import pickle
import numpy as np
from research.experiments.boundary_20260908.distributional.online_calibration import run
from research.experiments.boundary_20260908.distributional.execution import verify as metrics_oracle


def independent_replay(frame,base,config):
    frame=frame.reset_index(drop=True);out=np.zeros_like(base);latest=np.full(len(frame),-np.inf);seen=np.zeros(len(frame),int)
    levels=np.array([.025,.05,.1,.25,.5,.75,.9,.95,.975]);eta=config['eta_seconds'];shrink=1-eta*config['shrink_per_second']
    for _,race in frame.groupby('event_key'):
        state={};time={};counts={};heap=[]
        for issued,group in race.groupby('issued_at_timestamp',sort=True):
            while heap and heap[0][0]<issued:
                arrival=heap[0][0];feedback={}
                while heap and heap[0][0]==arrival:
                    _,index,context=heapq.heappop(heap)
                    g=levels-(frame.at[index,'lap_time_seconds']<=out[index]);g[4]=0.
                    feedback.setdefault(context,[]).append(g)
                for context,gradient in feedback.items():
                    state[context]=shrink*state.get(context,np.zeros(9))+eta*np.array(gradient).mean(0)
                    time[context]=arrival;counts[context]=counts.get(context,0)+len(gradient)
            for index,row in group.iterrows():
                context='all' if config['context']=='race_global' else row.compound
                x=base[index]-base[index,4]+state.get(context,np.zeros(9))
                out[index]=np.r_[np.sort(np.minimum(x[:4],0)),0,np.sort(np.maximum(x[5:],0))]+base[index,4]
                latest[index]=time.get(context,-np.inf);seen[index]=counts.get(context,0)
                heapq.heappush(heap,(row.target_timestamp,index,context))
    return out,latest,seen


def load(filename):
    with (run.OUT/filename).open('rb') as f:return pickle.load(f)


def verify_phase(phase,selection,protocol,metric_spec):
    frame,static=run.original(phase);saved=load(phase+'_forecasts.pkl');candidate=selection['selected_key']
    for name,q in static.items():np.testing.assert_array_equal(q,saved['predictions'][name])
    max_error=0.;policies=0
    for key,audit in saved['audit'].items():
        prefix,variant=key.split('__');base='selected_quantile' if prefix=='quantile' else 'stratified_empirical'
        expected,latest,seen=independent_replay(frame,static[base],protocol['variants'][variant])
        np.testing.assert_array_equal(expected,saved['predictions'][key]);np.testing.assert_array_equal(latest,audit['latest_feedback_timestamp']);np.testing.assert_array_equal(seen,audit['resolved_outcomes_seen'])
        assert (latest<frame.issued_at_timestamp).all();policies+=1
    summary=selection['summary'] if phase=='selection' else None
    if phase=='selection':
        metrics_oracle.check_forecasts(frame,static['selected_quantile'][:,4],saved['predictions'],summary['models'],metric_spec)
        for ref in selection['references']:metrics_oracle.independent_paired(frame,saved['predictions'][candidate],saved['predictions'][ref],summary['paired'][ref],metric_spec)
    else:
        result=json.loads((run.OUT/'results.json').read_text())
        for name,mask in [('2024',frame.year.eq(2024)),('2025',frame.year.eq(2025)),('2024_2025',frame.year.isin([2024,2025])),('2026',frame.year.eq(2026)),('2026_recent',frame.year.eq(2026)&frame.event_key.ge(202610))]:
            summary=result['summaries'][name]
            metrics_oracle.check_forecasts(frame.loc[mask],static['selected_quantile'][mask,4],{k:q[mask] for k,q in saved['predictions'].items()},summary['models'],metric_spec)
            for ref in selection['references']:metrics_oracle.independent_paired(frame.loc[mask],saved['predictions'][candidate][mask],saved['predictions'][ref][mask],summary['paired'][ref],metric_spec)
    return {'rows':len(frame),'policies_replayed':policies,'forecast_max_absolute_error':max_error,'latest_feedback_strictly_prior':True,'point_max_absolute_change':0.}


def main():
    protocol,metric_spec=run.spec();selection=json.loads((run.OUT/'selection.json').read_text());lock=json.loads((run.OUT/'selection_lock.json').read_text());result=json.loads((run.OUT/'results.json').read_text())
    assert lock['selection_sha256']==run.sha(run.OUT/'selection.json')==result['selection_sha256']
    assert selection['forecast_sha256']==run.sha(run.OUT/'selection_forecasts.pkl')
    metrics=selection['summary']['models']
    best=min(protocol['variants'],key=lambda n:(metrics['quantile__'+n]['wis'],n));best_cond=min(protocol['variants'],key=lambda n:(metrics['conditional__'+n]['wis'],n))
    assert best==selection['selected']==lock['selected'] and best_cond==selection['best_conditional']==lock['best_conditional']
    refs=list(dict.fromkeys([*run.STATIC,'conditional__'+best,'conditional__'+best_cond]));assert refs==selection['references']==lock['references']
    candidate='quantile__'+best
    advanced=all(1-metrics[candidate]['wis']/metrics[r]['wis']>=.01 for r in refs) and all(metrics[candidate]['coverage'][str(q)]>=q-.05 for q in [.8,.9,.95])
    assert advanced==selection['advancement_passed']==lock['advancement_passed']
    phases={'selection':verify_phase('selection',selection,protocol,metric_spec)}
    if advanced:
        assert result['transfer_evaluated'] and result['forecast_sha256']==run.sha(run.OUT/'transfer_forecasts.pkl')
        phases['transfer']=verify_phase('transfer',selection,protocol,metric_spec)
        hist=result['summaries']['2024_2025'];gates={}
        for ref in refs:
            gain=1-hist['models'][candidate]['wis']/hist['models'][ref]['wis']
            gates['historical_gain_vs_'+ref]=gain>=.1 if ref in run.STATIC[:2] else gain>0
            gates['historical_negative_block_ci_vs_'+ref]=hist['paired'][ref]['block3_ci95'][1]<0
            gates['each_year_improves_vs_'+ref]=all(result['summaries'][y]['models'][candidate]['wis']<result['summaries'][y]['models'][ref]['wis'] for y in ['2024','2025','2026'])
        gates['original_coverage_constraint']=all(result['summaries'][y]['models'][candidate]['coverage'][str(q)]>=max(q-.05,result['summaries'][y]['models']['stratified_empirical']['coverage'][str(q)]-.03) for y in ['2024','2025','2026'] for q in [.8,.9,.95]);gates['fixed_point']=True
        assert result['gates']==gates and all(gates.values())==result['substantial_gate_passed']
    else:
        assert not result['transfer_evaluated'] and not result['substantial_gate_passed']
        assert not (run.OUT/'transfer_attempt.json').exists() and not (run.OUT/'transfer_forecasts.pkl').exists()
    run.write(run.OUT/'verification.json',{'status':'passed','verified_at_utc':run.now(),'source_sha256':run.digest(run.sources()),'selection_sha256':run.sha(run.OUT/'selection.json'),'results_sha256':run.sha(run.OUT/'results.json'),'checks':phases,'fits_executed':0,'selection_and_gates_recomputed':True})
    print(json.dumps({'status':'passed','checks':phases,'transfer_evaluated':advanced}),flush=True)


if __name__=='__main__':main()
