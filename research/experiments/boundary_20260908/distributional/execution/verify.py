"""Independent interval-score, empirical-CDF, chronology and serialized forecast audit.

Fits only the four inherited base regressors again for deterministic OOF replay;
never selects a model or fits any additional quantile candidate.
"""
import json
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.experiments.boundary_20260908.distributional.execution import run,model


def independent_wis(y,q):
    y=np.asarray(y);q=np.asarray(q)
    result=.5*np.abs(y-q[:,4])
    for i,alpha in enumerate([.5,.2,.1,.05]):
        lo,hi=q[:,3-i],q[:,5+i]
        result+=alpha/2*(hi-lo)+np.maximum(lo-y,0)+np.maximum(y-hi,0)
    return result/4.5


def independent_quantile(x,w,levels):
    support=np.unique(x);cdf=np.asarray([w[x<=z].sum()/w.sum() for z in support])
    cdf[-1]=1.
    return support[np.searchsorted(cdf,levels,side='left')]


def project(q):
    result=q.copy();result[:4]=np.sort(np.clip(q[:4],None,0));result[4]=0.;result[5:]=np.sort(np.clip(q[5:],0,None));return result


def verify_empirical(frame,point,bundle,spec):
    residual=frame.lap_time_seconds.to_numpy()-point
    events=frame.event_key.to_numpy();counts={v:int((events==v).sum()) for v in np.unique(events)}
    w=np.asarray([1/counts[v] for v in events]);w*=len(w)/w.sum()
    levels=spec['quantile_levels']
    # A sorted cumulative implementation is needed for full data; the slow
    # support-summing oracle is independently checked in focused unit tests.
    def inverse(x,weights):
        order=np.argsort(x);cumulative=np.cumsum(weights[order]);positions=np.searchsorted(cumulative,np.asarray(levels)*cumulative[-1],side='left')
        return x[order[np.minimum(positions,len(x)-1)]]
    np.testing.assert_array_equal(bundle['global'],project(inverse(residual,w)))
    vol=np.maximum(frame.x_mad_3.to_numpy(),frame.x_mad_8.to_numpy())
    order=np.argsort(vol);cum=np.cumsum(w[order]);cuts=vol[order[np.searchsorted(cum,np.array([.25,.5,.75])*cum[-1])]]
    np.testing.assert_array_equal(cuts,bundle['cuts'])
    groups=(frame.wet_compound.to_numpy()>.5)*8+(frame.stint_clean_count.to_numpy()<=3)*4+np.searchsorted(cuts,vol,side='right')
    for cell in np.unique(groups):
        take=groups==cell;sw=w[take];neff=sw.sum()**2/(sw**2).sum();a=neff/(neff+200)
        mix=(1-a)*w/w.sum()+np.where(take,a*w/sw.sum(),0.)
        np.testing.assert_array_equal(bundle['cells'][str(cell)],project(inverse(residual,mix)))
        np.testing.assert_allclose(bundle['cell_diagnostics'][str(cell)]['effective_n'],neff,rtol=1e-12)
    return len(bundle['cells'])


def check_forecasts(frame,point,predictions,expected,spec):
    result={}
    for name,q in predictions.items():
        assert q.shape==(len(frame),9) and np.isfinite(q).all() and (np.diff(q,axis=1)>=0).all()
        np.testing.assert_array_equal(q[:,4],point)
        score=independent_wis(frame.lap_time_seconds,q)
        np.testing.assert_allclose(score,model.wis_rows(frame.lap_time_seconds,q,spec),rtol=1e-13,atol=1e-13)
        means=pd.Series(score).groupby(frame.event_key.to_numpy()).mean()
        metric=expected[name];np.testing.assert_allclose(means.mean(),metric['wis'],rtol=0,atol=1e-12)
        mae=pd.Series(np.abs(frame.lap_time_seconds.to_numpy()-point)).groupby(frame.event_key.to_numpy()).mean().mean()
        np.testing.assert_allclose(mae,metric['point_mae'],rtol=0,atol=1e-12)
        for i,level in enumerate(spec['central_interval_coverages']):
            y=frame.lap_time_seconds.to_numpy();lo,hi=q[:,3-i],q[:,5+i]
            coverage=pd.Series(((lo<=y)&(y<=hi)).astype(float)).groupby(frame.event_key.to_numpy()).mean().mean()
            width=pd.Series(hi-lo).groupby(frame.event_key.to_numpy()).mean().mean()
            np.testing.assert_allclose(coverage,metric['coverage'][str(level)],atol=1e-12)
            np.testing.assert_allclose(width,metric['mean_width_seconds'][str(level)],atol=1e-12)
        result[name]={'rows':len(frame),'events':len(means),'wis':float(means.mean()),'point_mae':float(mae),'median_max_absolute_difference':0.}
    return result


def independent_paired(frame,points,reference,expected,spec):
    delta=independent_wis(frame.lap_time_seconds,points)-independent_wis(frame.lap_time_seconds,reference)
    events=pd.Series(delta).groupby(frame.event_key.to_numpy()).mean()
    rng=np.random.default_rng(spec['uncertainty']['seed']);B=spec['uncertainty']['resamples'];draws=np.zeros((2,B))
    for year in sorted(set(events.index//100)):
        values=events.loc[events.index//100==year].to_numpy();n=len(values)
        draws[0]+=values[rng.integers(n,size=(B,n))].mean(1)*n/len(events)
        starts=rng.integers(n,size=(B,(n+2)//3));ix=np.concatenate([starts[:,:,None]+j for j in range(3)],axis=2).reshape(B,-1)[:,:n]%n
        draws[1]+=values[ix].mean(1)*n/len(events)
    np.testing.assert_allclose(events.mean(),expected['candidate_minus_reference'],atol=1e-12)
    np.testing.assert_allclose(np.quantile(draws[0],[.025,.975]),expected['event_ci95'],atol=1e-12)
    np.testing.assert_allclose(np.quantile(draws[1],[.025,.975]),expected['block3_ci95'],atol=1e-12)


def load(name):
    with (run.OUT/name).open('rb') as f:return pickle.load(f)


def main():
    spec=run.specification();selection=json.loads((run.OUT/'selection.json').read_text());result=json.loads((run.OUT/'results.json').read_text())
    assert selection['source_sha256']==run.digest(run.sources())
    assert result['selection_sha256']==run.sha(run.OUT/'selection.json')
    lock=json.loads((run.OUT/'selection_lock.json').read_text())
    assert lock['selection_sha256']==run.sha(run.OUT/'selection.json') and lock['selected']==selection['selected']
    for name,value in selection['artifacts'].items():assert run.sha(run.OUT/name)==value
    discovery=pd.read_pickle(run.OLD/'discovery_data.pkl');features=selection['features']
    inherited=json.loads((run.OLD/'selection.json').read_text())
    raw_count=run.validate_manifest(inherited['input_manifest'])
    assert features==inherited['features'];run.validate_population(discovery,features,[2022,2023])
    saved=load('crossfit_residual_training.pkl');bundle=load('selection_bundle.pkl');forecasts=load('selection_forecasts.pkl')
    with threadpool_limits(limits=1):
        frame,point,folds=run.mechanics.crossfit(discovery,features)
        pd.testing.assert_frame_equal(frame,saved['frame']);np.testing.assert_array_equal(point,saved['point']);assert folds==saved['folds']==selection['crossfit_blocks']
        assert set(frame.year)=={2022}
        for fold in folds:assert max(fold['fit_events'])<min(fold['prediction_events'])
        validation=discovery.loc[discovery.year.eq(2023)].reset_index(drop=True);pd.testing.assert_frame_equal(validation,forecasts['frame'])
        refit=run.mechanics.original.fit_model(discovery.loc[discovery.year.eq(2022)],run.mechanics.BASE_CONFIG,features)
        vp=run.mechanics.original.predict_model(validation,refit)
        np.testing.assert_array_equal(vp,forecasts['point'])
        np.testing.assert_array_equal(vp,run.mechanics.original.predict_model(validation,bundle['base']))
        cells=verify_empirical(frame,point,bundle['empirical'],spec)
        replay=model.empirical_predict(validation,vp,bundle['empirical'])
        quantile_count=0
        for name,quantiles in bundle['candidates'].items():
            assert quantiles['features']==features and quantiles['fit_rows']==len(frame)
            for q,learner in quantiles['models'].items():
                params=learner.get_params();assert params['quantile']==float(q) and params['loss']=='quantile'
                assert params['max_iter']==spec['candidates'][name]['iterations'] and params['max_leaf_nodes']==spec['candidates'][name]['leaves']
                for k,v in spec['hgb'].items():assert params[k]==v
                quantile_count+=1
            replay[name],_=model.quantile_predict(validation,vp,quantiles,spec)
        for name,q in replay.items():np.testing.assert_array_equal(q,forecasts['quantiles'][name])
    verified_selection=check_forecasts(validation,vp,replay,selection['all_selection_metrics'],spec)
    chosen=min(spec['candidates'],key=lambda n:(verified_selection[n]['wis'],n));assert chosen==selection['selected']
    coverage=all(selection['all_selection_metrics'][chosen]['coverage'][str(q)]>=q-.05 for q in [.8,.9,.95])
    gains={ref:1-verified_selection[chosen]['wis']/verified_selection[ref]['wis'] for ref in run.REFS}
    advanced=coverage and all(x>=.01 for x in gains.values());assert advanced==selection['advancement_passed']==lock['advancement_passed']
    for ref in run.REFS:independent_paired(validation,replay[chosen],replay[ref],selection['paired'][ref],spec)
    transfer_verified={}
    if advanced:
        assert result['transfer_evaluated'];final=load('final_distribution_bundle.pkl');forecast=load('transfer_forecasts.pkl')
        assert result['forecasts_sha256']==run.sha(run.OUT/'transfer_forecasts.pkl')
        frame_final=pd.concat([frame,validation],ignore_index=True);point_final=np.r_[point,vp]
        assert set(frame_final.year)=={2022,2023}
        verify_empirical(frame_final,point_final,final['empirical'],spec)
        fl=json.loads((run.OUT/'fit_lock.json').read_text());assert fl['fit_rows']==len(frame_final) and fl['fit_events']==sorted(map(int,frame_final.event_key.unique()))
        assert fl['model_sha256']==run.sha(run.OUT/'final_distribution_bundle.pkl')
        transfer=forecast['frame'];raw=pd.read_pickle(run.OLD/'corrected_input_contract/transfer_data_and_forecasts.pkl');pd.testing.assert_frame_equal(transfer,raw)
        run.validate_population(transfer,features,[2024,2025,2026])
        raw_count+=run.validate_manifest(json.loads((run.OLD/'corrected_input_contract/results.json').read_text())['input_manifest'])
        np.testing.assert_array_equal(forecast['point'],raw.prediction_hgb_l15_i150.to_numpy())
        replay=model.empirical_predict(transfer,forecast['point'],final['empirical']);replay[chosen],_=model.quantile_predict(transfer,forecast['point'],final['candidate'],spec)
        for name,q in replay.items():np.testing.assert_array_equal(q,forecast['quantiles'][name])
        masks={'2024':transfer.year.eq(2024),'2025':transfer.year.eq(2025),'2024_2025':transfer.year.isin([2024,2025]),'2026':transfer.year.eq(2026),'2026_recent':transfer.year.eq(2026)&transfer.event_key.ge(202610)}
        for name,mask in masks.items():
            summary=result['summaries'][name];transfer_verified[name]=check_forecasts(transfer.loc[mask],forecast['point'][mask],{k:q[mask] for k,q in replay.items()},summary['models'],spec)
            for ref in run.REFS:independent_paired(transfer.loc[mask],replay[chosen][mask],replay[ref][mask],summary['paired'][ref],spec)
        gates={};h=result['summaries']['2024_2025']
        for ref in run.REFS:
            gates['historical_10pct_vs_'+ref]=1-h['models'][chosen]['wis']/h['models'][ref]['wis']>=.1
            gates['historical_block3_upper_negative_vs_'+ref]=h['paired'][ref]['block3_ci95'][1]<0
            gates['each_year_wis_improves_vs_'+ref]=all(result['summaries'][y]['models'][chosen]['wis']<result['summaries'][y]['models'][ref]['wis'] for y in ['2024','2025','2026'])
        gates['coverage_each_year']=all(result['summaries'][y]['models'][chosen]['coverage'][str(q)]>=max(q-.05,result['summaries'][y]['models']['stratified_empirical']['coverage'][str(q)]-.03) for y in ['2024','2025','2026'] for q in [.8,.9,.95]);gates['point_unchanged']=True
        assert gates==result['gates'] and all(gates.values())==result['substantial_gate_passed']
    else:
        assert not result['transfer_evaluated'] and not result['substantial_gate_passed']
        for name in ['transfer_attempt.json','transfer_forecasts.pkl','final_distribution_bundle.pkl','fit_lock.json']:assert not (run.OUT/name).exists()
    run.write(run.OUT/'verification.json',{'verified_at_utc':run.now(),'status':'passed','source_sha256':run.digest(run.sources()),'execution_lock_sha256':run.sha(run.OUT/'execution_lock.json'),
        'selection_sha256':run.sha(run.OUT/'selection.json'),'results_sha256':run.sha(run.OUT/'results.json'),'raw_inputs_verified':raw_count,'crossfit_rows_replayed':len(frame),'base_models_refit_for_verification':4,'selection_quantile_models_replayed':quantile_count,'empirical_cells_rebuilt':cells,'forecast_replay_max_absolute_error':0.,'fixed_point_max_absolute_change':0.,'selection':verified_selection,'transfer':transfer_verified,'promotion':False})
    print(json.dumps({'status':'passed','selection_models_replayed':quantile_count,'raw_inputs_verified':raw_count,'transfer_evaluated':advanced}),flush=True)


if __name__=='__main__':main()
