"""Synthetic objective/population contracts; no real CatBoost or historical data."""
import copy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from . import models as m


def frame(n=8,events=(202201,202202)):
    x=pd.DataFrame(np.arange(n*80,dtype=float).reshape(n,80)/1000,columns=m.BASE_FEATURES)
    # Deliberately unequal events make row weighting mistakes observable.
    event_values=[events[0]]*(n-len(events)+1)+list(events[1:])
    x['event_key']=event_values;x['driver_id']='44'
    x['issued_after_lap_number']=np.arange(n)+1
    x['issued_at_timestamp']=100.+np.arange(n)*90.
    x['issued_at_ns']=(x.issued_at_timestamp*1_000_000_000).astype('int64')
    x['issuance_id']=['synthetic-'+str(i) for i in range(n)]
    x['forecast_naive_seconds']=90.
    x['year']=2022;x['outcome_status']='matched'
    x['lap_time_seconds']=90.+np.linspace(-10,10,n)
    x['target_lap_number']=x.issued_after_lap_number+1
    x['target_timestamp']=x.issued_at_timestamp+100.
    x['target_at_ns']=x.issued_at_ns+100_000_000_000
    x['target_id']=['future-'+str(i) for i in range(n)]
    x['target_same_stint']=[bool(i%2) for i in range(n)]
    return x


def issued(f):
    return f.drop(columns=[c for c in f if c.startswith('target_') or c in ('lap_time_seconds','outcome_status')])


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(m,'TRAIN_ROWS',8)
    monkeypatch.setattr(m,'TRAIN_EVENTS',(202201,202202))
    return frame()


@pytest.fixture
def backend():
    calls=[]
    class Estimator:
        def __init__(self,**kwargs):
            self.parameters=kwargs;self.fit_calls=[];self.predict_calls=[]
            calls.append(self)
        def fit(self,x,y,**kwargs):
            self.fit_calls.append((x.copy(),y.copy(),copy.deepcopy(kwargs)))
            self.feature_names_=list(x.columns);self.tree_count_=1000
            # Deliberately mutate borrowed inputs; candidates/input frames must be isolated.
            x.iloc[:,:]=999.;y[:]=999.;kwargs['sample_weight'][:]=999.
            return self
        def predict(self,x,*,thread_count):
            assert thread_count == 1
            self.predict_calls.append(x.copy())
            return np.full(len(x),-10. if self.parameters['boosting_type']=='Plain' else 10.)
    return Estimator,calls


def test_exact_two_objectives_and_unclipped_original_matrix(small,backend):
    constructor,calls=backend;before=small.copy(deep=True)
    result=m.fit_candidates(small,small.copy(),constructor)
    assert tuple(result)==('plain','ordered') and len(calls)==2
    expected=dict(loss_function='MAE',iterations=1000,depth=6,learning_rate=.03,l2_leaf_reg=10,
        grow_policy='SymmetricTree',task_type='CPU',thread_count=1,use_best_model=False,
        leaf_estimation_method='Exact',leaf_estimation_iterations=1,boost_from_average=True,
        bootstrap_type='No',random_strength=0,border_count=254,nan_mode='Min',has_time=False,
        random_seed=20260908,allow_writing_files=False,verbose=False)
    for model,scheme in zip(calls,('Plain','Ordered')):
        assert model.parameters=={**expected,'boosting_type':scheme}
        assert len(model.fit_calls)==1
        x,y,kwargs=model.fit_calls[0]
        assert list(x.columns)==list(m.BASE_FEATURES) and x.shape==(8,80)
        np.testing.assert_array_equal(x,before[list(m.BASE_FEATURES)])
        np.testing.assert_array_equal(y,np.clip(before.lap_time_seconds-90.,-5.,5.))
        assert set(kwargs)=={'sample_weight'} # no eval_set/early stopping/masks
        np.testing.assert_array_equal(kwargs['sample_weight'],np.array([4/7]*7+[4.]))
        assert kwargs['sample_weight'].mean()==pytest.approx(1.)
    pd.testing.assert_frame_equal(small,before)


def test_production_sized_synthetic_population_and_each_event_total(backend):
    data=frame(m.TRAIN_ROWS,m.TRAIN_EVENTS)
    _,calls=backend;m.fit_candidates(data,data.copy(),backend[0])
    assert len(calls)==2 and all(len(model.fit_calls[0][1])==18363 for model in calls)
    weights=calls[0].fit_calls[0][2]['sample_weight']
    totals=pd.Series(weights).groupby(data.event_key).sum().to_numpy()
    np.testing.assert_allclose(totals,np.full(22,18363/22),rtol=1e-14)
    assert weights.sum()==pytest.approx(18363)


def test_prediction_all_rows_anchor_and_bounds_no_new_fit(small,backend):
    models=m.fit_candidates(small,small.copy(),backend[0]);query=issued(small)
    before=query.copy(deep=True)
    points=m.predict_candidates(models,query,query.copy())
    np.testing.assert_array_equal(points['plain'],np.full(8,87.))
    np.testing.assert_array_equal(points['ordered'],np.full(8,93.))
    assert all(p.dtype==np.float64 for p in points.values())
    assert all(len(x.fit_calls)==1 and len(x.predict_calls)==1 for x in backend[1])
    pd.testing.assert_frame_equal(query,before)


@pytest.mark.parametrize('column,value',[
    ('issuance_id','altered'),('driver_id','22'),('issued_at_ns',101),
    ('issued_at_timestamp',100.000000001),('forecast_naive_seconds',91.),
    (m.BASE_FEATURES[0],-.1),('event_key',202202),('issued_after_lap_number',30)])
def test_original_identity_values_cannot_drift(small,backend,column,value):
    changed=small.copy();changed.loc[0,column]=value
    with pytest.raises(ValueError):m.fit_candidates(changed,small,backend[0])
    assert backend[1]==[]


@pytest.mark.parametrize('what',['missing','duplicate','permuted','wrong_year','wrong_event','unmatched'])
def test_training_population_immutable_before_backend(small,backend,what):
    actual=small.copy();expected=small.copy()
    if what=='missing':actual=actual.iloc[:-1]
    if what=='duplicate':actual=pd.concat([actual.iloc[:-1],actual.iloc[[0]]],ignore_index=True)
    if what=='permuted':actual=actual.iloc[::-1].reset_index(drop=True)
    if what=='wrong_year':actual['year']=2023
    if what=='wrong_event':actual['event_key']=202203;expected=actual.copy()
    if what=='unmatched':actual.loc[0,'outcome_status']='unmatched'
    with pytest.raises(ValueError):m.fit_candidates(actual,expected,backend[0])
    assert not backend[1]


@pytest.mark.parametrize('column,value',[
    ('lap_time_seconds',91.),('target_lap_number',90),('target_timestamp',1000.),
    ('target_same_stint',True),('target_id','changed'),('target_at_ns',1)])
def test_original_target_pairing_immutable(small,backend,column,value):
    actual=small.copy();actual.loc[0,column]=value
    with pytest.raises(ValueError):m.fit_candidates(actual,small,backend[0])
    assert not backend[1]


@pytest.mark.parametrize('column,value',[
    ('lap_time_seconds',np.nan),('lap_time_seconds',np.inf),('lap_time_seconds',0.),
    ('target_timestamp',100.),('target_lap_number',1),('target_at_ns',100_000_000_000),
    ('target_same_stint','False'),('issued_at_ns',True)])
def test_structurally_invalid_matched_inputs_rejected(small,backend,column,value):
    actual=small.copy();actual[column]=actual[column].astype(object);actual.loc[0,column]=value
    with pytest.raises(ValueError):m.fit_candidates(actual,actual.copy(),backend[0])
    assert not backend[1]


@pytest.mark.parametrize('column',['target_timestamp','lap_time_seconds','outcome_status','LapTime','target_any_future_value'])
def test_prediction_refuses_labels_before_estimator_use(small,backend,column):
    models=m.fit_candidates(small,small.copy(),backend[0]);query=issued(small)
    query[column]=object()
    with pytest.raises(ValueError,match='target-free'):m.predict_candidates(models,query,issued(small))
    assert all(not model.predict_calls for model in backend[1])


@pytest.mark.parametrize('fault',['dropped','reordered','feature','anchor','id','expected_labeled'])
def test_prediction_preserves_every_independent_original_issuance(small,backend,fault):
    models=m.fit_candidates(small,small.copy(),backend[0]);query=issued(small);expected=query.copy()
    if fault=='dropped':query=query.iloc[:-1]
    if fault=='reordered':query=query.iloc[::-1]
    if fault=='feature':query.loc[0,m.BASE_FEATURES[0]]+=1
    if fault=='anchor':query.loc[0,'forecast_naive_seconds']+=1
    if fault=='id':query.loc[0,'issuance_id']='different'
    if fault=='expected_labeled':expected['target_id']='poison'
    with pytest.raises(ValueError):m.predict_candidates(models,query,expected)
    assert all(not model.predict_calls for model in backend[1])


@pytest.mark.parametrize('fault',['wrong_keys','feature_order','few_trees','nan','infinity','shape','nonpositive'])
def test_bad_serialized_models_and_outputs_fail_closed(small,backend,fault):
    models=m.fit_candidates(small,small.copy(),backend[0]);query=issued(small)
    model=models['plain']
    if fault=='wrong_keys':models['other']=models.pop('ordered')
    if fault=='feature_order':model.feature_names_=model.feature_names_[::-1]
    if fault=='few_trees':model.tree_count_=999
    if fault=='nan':model.predict=lambda x,**kwargs:np.full(len(x),np.nan)
    if fault=='infinity':model.predict=lambda x,**kwargs:np.full(len(x),np.inf)
    if fault=='shape':model.predict=lambda x,**kwargs:np.zeros((len(x),1))
    if fault=='nonpositive':query['forecast_naive_seconds']=1.
    with pytest.raises(ValueError):m.predict_candidates(models,query,query.copy())


def test_backend_lazy_version_guard_and_invalid_input_never_imports(small,monkeypatch,backend):
    monkeypatch.setitem(sys.modules,'catboost',SimpleNamespace(__version__='1.2.9',CatBoostRegressor=backend[0]))
    with pytest.raises(ValueError,match='1.2.10'):m.fit_candidates(small,small.copy())
    assert not backend[1]
    monkeypatch.setitem(sys.modules,'catboost',SimpleNamespace(__version__='1.2.10',CatBoostRegressor=backend[0]))
    assert set(m.fit_candidates(small,small.copy()))=={'plain','ordered'}


def test_duplicate_feature_names_and_missing_exact_clock_fail(small,backend):
    for actual in (pd.concat([small,small[[m.BASE_FEATURES[0]]]],axis=1),small.drop(columns='issued_at_ns')):
        with pytest.raises(ValueError):m.fit_candidates(actual,small,backend[0])
    assert not backend[1]


def test_parameter_objects_are_independent_and_no_unfrozen_candidates():
    x=m.parameters('plain');x['depth']=1
    assert m.parameters('ordered')['depth']==m.parameters('plain')['depth']==6
    with pytest.raises(ValueError):m.parameters('sweep')
