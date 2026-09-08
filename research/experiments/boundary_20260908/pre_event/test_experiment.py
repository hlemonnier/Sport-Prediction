import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

s=importlib.util.spec_from_file_location('boundary_pre_event',Path(__file__).with_name('run_experiment.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_quadratic_regime_matches_penalized_normal_equations():
    x=np.array([[1.,0.],[0.,1.],[1.,1.]]);y=np.array([.2,-.1,.05]);w=np.array([1.,2.,3.]);penalty=np.array([2.,4.])
    beta,diagnostic=m.solve(x,y,w,penalty)
    expected=np.linalg.solve(x.T@(w[:,None]*x)+np.diag(penalty),x.T@(w*y))
    np.testing.assert_allclose(beta,expected,rtol=0,atol=1e-12)
    assert diagnostic['gradient_max_abs']<1e-10


def test_season_reset_ignores_all_prior_season_response_values():
    frame=pd.DataFrame({'event_key':[202401,202401],'driver_id':['A','B'],'latest_qualifying_rehearsal_rank':[1,2]})
    hist=pd.DataFrame({'event_key':[202324],'residual_equivalent_positions':[1e12]})
    p,diagnostic=m.predict(frame,hist,{'lambda':.5,'half_life_events':3,'strength':1})
    assert p.tolist()==[1,2] and diagnostic['status']=='zero_state_season_start'


def test_reject_current_labels_and_nonprior_history():
    frame=pd.DataFrame({'event_key':[202401],'qualy_position':[1]})
    with pytest.raises(ValueError,match='target'):m.predict(frame,pd.DataFrame(),{})
    with pytest.raises(ValueError,match='earlier'):m.predict(frame.drop(columns='qualy_position'),pd.DataFrame({'event_key':[202401]}),{})


def test_selection_ignores_later_year_metric_changes():
    c={'A':{'lambda':1,'strength':1,'half_life_events':3},'B':{'lambda':2,'strength':1,'half_life_events':3}}
    e=[{'year':2023,'event_key':202301,'models':{'A':{'mae':2},'B':{'mae':3}}},
       {'year':2024,'event_key':202401,'models':{'A':{'mae':99},'B':{'mae':0}}}]
    assert m.select(e,c)['selected']=='A'
