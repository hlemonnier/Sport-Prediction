import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('pre_event_research',Path(__file__).with_name('run_experiment.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_rank_head_preserves_roster_and_rehearsal_ties():
    assert runner.permutation([.2,.1,.1],[3,2,1]).tolist()==[3,2,1]


def test_current_targets_and_future_history_rejected():
    frame=pd.DataFrame({'event_key':[202405],'qualy_position':[1]})
    with pytest.raises(ValueError,match='target columns'):
        runner.predict_event(frame,pd.DataFrame())
    with pytest.raises(ValueError,match='precede'):
        runner.predict_event(frame,pd.DataFrame({'event_key':[202405]}))


def test_event_weighted_ridge_is_invariant_to_repeated_rows_in_one_event():
    h=pd.DataFrame({'event_key':np.repeat(np.arange(202401,202413),3),
                    'x':np.tile([0.,1.,2.],12),'y':np.tile([0.,1.,1.5],12)})
    current=pd.DataFrame({'x':[.5,1.5]})
    p,_=runner.fitted_residual(h,current,['x'],'y')
    more=pd.concat([h,h.loc[h.event_key.eq(202401)]]*1,ignore_index=True)
    q,_=runner.fitted_residual(more,current,['x'],'y')
    np.testing.assert_allclose(p,q,atol=1e-12)


def test_historical_prior_excludes_other_driver_if_own_history_exists():
    h=pd.DataFrame({'event_key':[202401,202402,202403],'driver_id':['A','B','A'],
                    'team_id':['X','X','X'],'actual_rank_fraction':[.2,1.,.4]})
    value=runner.prior_rank(h,'A','X')
    assert .3<value<.4
    assert runner.lower_median([1,2,3,4])==2


def test_old_checkout_prefix_resolves_only_existing_same_event_basename(tmp_path):
    event=tmp_path/'event';event.mkdir()
    file=event/'03_practice_3_laps.csv';file.write_text('driver\nA\n')
    assert runner.relocated_snapshot_path(tmp_path,event,'/old/checkout/03_practice_3_laps.csv')==file
    with pytest.raises(FileNotFoundError):
        runner.relocated_snapshot_path(tmp_path,event,'/old/checkout/04_qualifying_laps.csv')
