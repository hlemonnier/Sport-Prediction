"""Counterexamples for scoring, empirical distributions and causal model inputs."""
import json
import numpy as np
import pandas as pd
import pytest
from research.experiments.boundary_20260908.distributional.execution import model, run

SPEC=json.loads(run.SPEC.read_text())


def frame(n=500):
    rng=np.random.default_rng(987)
    return pd.DataFrame({'event_key':np.repeat([202201,202202],n//2),
        'lap_time_seconds':90+rng.normal(size=n), 'wet_compound':np.repeat([0,1],n//2),
        'stint_clean_count':5, 'x_mad_3':rng.random(n), 'x_mad_8':rng.random(n)})


def test_wis_interval_definition_equals_two_ninths_pinball():
    rng=np.random.default_rng(73);q=np.sort(rng.normal(size=(100,9)),axis=1);y=rng.normal(size=100)*4
    wis=.5*np.abs(y-q[:,4])
    for i,a in enumerate([.5,.2,.1,.05]):
        lo,hi=q[:,3-i],q[:,5+i]
        interval=hi-lo+2/a*np.maximum(lo-y,0)+2/a*np.maximum(y-hi,0)
        wis+=a/2*interval
    np.testing.assert_allclose(model.wis_rows(y,q,SPEC),wis/4.5,rtol=1e-14)
    np.testing.assert_allclose(wis/4.5,2/9*model.pinball_rows(y,q,SPEC['quantile_levels']).sum(1))
    # A point mass has finite WIS equal to absolute error, not infinite NLL.
    np.testing.assert_allclose(model.wis_rows(y,np.zeros((100,9)),SPEC),np.abs(y))


def test_weighted_inverse_cdf_minimizes_pinball_with_zero_weights_and_ties():
    x=np.array([10.,-2.,0.,0.,999.]);w=np.array([1.,3.,1.,2.,0.]);levels=np.array([0.,.025,.25,.5,.75,.975,1.])
    actual=model.weighted_quantile(x,w,levels)
    np.testing.assert_array_equal(actual,[-2,-2,-2,0,0,10,10])
    for q,prediction in zip(levels,actual):
        grid=np.unique(np.r_[x,np.linspace(-3,11,100)])
        loss=lambda z:np.sum(w*np.maximum(q*(x-z),(q-1)*(x-z)))
        assert loss(prediction)<=min(map(loss,grid))+1e-12
    np.testing.assert_array_equal(actual,model.weighted_quantile(x,100*w,levels))
    with pytest.raises(ValueError):model.weighted_quantile(x,np.zeros(5),levels)
    with pytest.raises(ValueError):model.weighted_quantile(x,w,[float('nan')])


def test_conditional_reference_is_cdf_mixture_not_quantile_average():
    data=frame(800);data['x_mad_3']=1.;data['x_mad_8']=1.
    data['lap_time_seconds']=np.r_[np.full(400,89.),np.full(400,99.)]
    empirical=model.fit_empirical(data,np.full(800,90.),SPEC)
    q=empirical['cells']['3'];weight=400/(400+200)
    # Dry cell all at -1; global half -1, half +9: mixture .8333 mass at -1.
    assert q[5]==0 # The 75th percentile is -1, clipped above the fixed median.
    assert q[6]==9 # The 90th percentile exceeds the 0.8333 lower mass.
    assert q[7]==9 and q[8]==9
    assert q[7] != (weight*(-1)+(1-weight)*9)
    predictions=model.empirical_predict(data,np.full(800,90.),empirical)
    assert np.isfinite(predictions['stratified_empirical']).all()
    empty=data.iloc[:2].copy();empty['stint_clean_count']=1
    np.testing.assert_array_equal(model.empirical_predict(empty,[90,90],empirical)['stratified_empirical'],np.tile(empirical['global']+90,(2,1)))


def test_coherence_projection_preserves_point_and_is_idempotent():
    raw=np.array([[1,-2,-3,4,100,3,-5,2,1.]])
    out=model.project_offsets(raw)
    np.testing.assert_array_equal(out,[[-3,-2,0,0,0,0,1,2,3]])
    np.testing.assert_array_equal(model.project_offsets(out),out)
    assert (np.diff(out,axis=1)>=0).all()
    with pytest.raises(ValueError):model.project_offsets(np.full((1,9),np.nan))


def test_event_balancing_is_not_row_weighted():
    data=pd.DataFrame({'event_key':[202201,202202,202202,202202],'lap_time_seconds':[9.,1.,1.,1.]})
    q=np.zeros((4,9));metrics=model.metrics(data,q,SPEC)
    assert metrics['wis']==5 and metrics['point_mae']==5
    assert not np.isclose(metrics['wis'],3)


def test_forecasting_cannot_read_targets_or_mutate_reference_fit():
    data=frame();point=np.full(len(data),90.);features=['wet_compound','stint_clean_count','x_mad_3','x_mad_8']
    fitted=model.fit_quantiles(data,point,features,{'leaves':7,'iterations':2},SPEC)
    empirical=model.fit_empirical(data,point,SPEC)
    test=data.iloc[:15].copy();poison=test.copy();poison['lap_time_seconds']=1e9
    for name,bundle in [('quantile',fitted),('empirical',empirical)]:
        if name=='quantile':
            a=model.quantile_predict(test,point[:15],bundle,SPEC)[0];b=model.quantile_predict(poison,point[:15],bundle,SPEC)[0]
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[:,4],point[:15])
        else:
            a=model.empirical_predict(test,point[:15],bundle);b=model.empirical_predict(poison,point[:15],bundle)
            for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_crossfit_expanding_2022_never_reads_future_labels(monkeypatch):
    mechanics=run.mechanics
    data=pd.DataFrame({'event_key':list(range(202201,202223))+[202301,202302],
                       'year':[2022]*22+[2023]*2,'lap_time_seconds':np.arange(24.)})
    calls=[]
    def fit(f,config,features):
        calls.append(set(f.event_key));return {'maxevent':f.event_key.max(),'value':f.lap_time_seconds.sum()}
    def predict(f,b):
        assert b['maxevent']<f.event_key.min();return np.full(len(f),b['value'])
    monkeypatch.setattr(mechanics.original,'fit_model',fit);monkeypatch.setattr(mechanics.original,'predict_model',predict)
    train,p,ledger=mechanics.crossfit(data,[])
    poisoned=data.copy();poisoned.loc[poisoned.year.eq(2023),'lap_time_seconds']=1e9
    train2,p2,ledger2=mechanics.crossfit(poisoned,[])
    np.testing.assert_array_equal(p,p2);pd.testing.assert_frame_equal(train,train2)
    assert all(max(x['fit_events'])<min(x['prediction_events']) for x in ledger)
    assert all(202301 not in events for events in calls)


def test_manifest_is_fail_closed_and_verifies_flat_provider_records(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path);p=tmp_path/'laps.csv';p.write_text('test')
    item={'event_key':202201,'path':'laps.csv','sha256':run.sha(p)}
    assert run.validate_manifest([item])==1
    with pytest.raises(ValueError):run.validate_manifest([{'laps':item}])
    with pytest.raises(ValueError):run.validate_manifest([item,item])
    p.write_text('changed')
    with pytest.raises(ValueError):run.validate_manifest([item])


def test_gate_requires_both_references_and_coverage():
    assert not run.coverage_pass({'coverage':{'0.8':.8,'0.9':.9,'0.95':.899}},SPEC)
    assert run.coverage_pass({'coverage':{'0.8':.75,'0.9':.85,'0.95':.90}},SPEC)


def test_independent_empirical_oracle_on_tied_support():
    from research.experiments.boundary_20260908.distributional.execution.verify import independent_quantile
    x=np.array([-1.,5.,-1.,2.,10.]);w=np.array([1.,3.,2.,5.,.5]);levels=np.array(SPEC['quantile_levels'])
    np.testing.assert_array_equal(model.weighted_quantile(x,w,levels),independent_quantile(x,w,levels))


def test_selection_requires_improvement_over_each_strong_reference():
    metrics={name:{'wis':1.0,'coverage':{'0.8':.8,'0.9':.9,'0.95':.95}} for name in [*SPEC['candidates'],*run.REFS]}
    names=list(SPEC['candidates']);metrics[names[0]]['wis']=.98
    selected,gains,advanced=run.select_candidate(metrics,SPEC)
    assert selected==names[0] and advanced
    metrics['stratified_empirical']['wis']=.975
    assert not run.select_candidate(metrics,SPEC)[2]
    metrics['stratified_empirical']['wis']=1.0
    metrics[names[0]]['coverage']['0.95']=.89
    assert not run.select_candidate(metrics,SPEC)[2]
