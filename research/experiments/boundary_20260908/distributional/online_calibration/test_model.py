import numpy as np
import pandas as pd
import pytest
from research.experiments.boundary_20260908.distributional.online_calibration import model

CFG={'eta_seconds':.03,'shrink_per_second':.2,'context':'race_global'}


def data(issues,targets,y=None,compound=None):
    n=len(issues)
    return pd.DataFrame({'event_key':202301,'driver_id':list(map(str,range(n))),'issued_at_timestamp':issues,'target_timestamp':targets,'lap_time_seconds':y if y is not None else np.full(n,100.),'compound':compound or ['HARD']*n,'target_same_stint':False})


def quantiles(n):return np.tile(np.array([86,87,88,89,90,91,92,93,94.]),(n,1))


def test_simultaneous_peer_and_own_labels_are_strictly_prior_only():
    frame=data([0,1,1,1.1],[1,2,2.1,2.2]);q=quantiles(4)
    pred,audit,_=model.replay(frame,q,CFG)
    np.testing.assert_array_equal(pred[:3],q[:3])
    assert pred[3,6]>q[3,6] and audit['latest_feedback_timestamp'][3]==1
    assert (audit['latest_feedback_timestamp']<frame.issued_at_timestamp).all()


def test_future_target_poison_cannot_change_earlier_forecasts():
    frame=data([0,1,2,3],[2.5,3.5,4.5,5.5]);q=quantiles(4)
    a=model.replay(frame,q,CFG)[0]
    poisoned=frame.copy();poisoned.loc[poisoned.target_timestamp.ge(3),'lap_time_seconds']=-1e6
    b=model.replay(poisoned,q,CFG)[0]
    np.testing.assert_array_equal(a,b) # No poisoned label has arrived by last issuance.
    changed=frame.copy();changed.loc[0,'lap_time_seconds']=-1e6
    c=model.replay(changed,q,CFG)[0]
    np.testing.assert_array_equal(a[:3],c[:3]);assert not np.array_equal(a[3],c[3])


def test_frozen_issued_quantile_defines_delayed_feedback():
    frame=data([0,1,2.1,3.1],[3,2,4,5],y=[91.01,100,100,100]);q=quantiles(4)
    pred,audit,_=model.replay(frame,q,CFG)
    # Label1 at2 first moves q75 by+.03*.75=.0225, then delayed label0 at3
    # must compare91.01 to issued91, not the updated91.0225.
    expected=.0225*(1-.03*.2)+.03*.75
    assert pred[3,5]==pytest.approx(91+expected)
    assert audit['resolved_outcomes_seen'][3]==2


def test_equal_time_label_batches_use_mean_gradient_independent_of_order():
    frame=data([0,0,2.1],[2,2,3],y=[100,80,100]);q=quantiles(3)
    a=model.replay(frame,q,CFG)[0]
    assert a[2,5]==pytest.approx(91+.03*(.75-.5))
    order=[1,0,2];b=model.replay(frame.iloc[order].reset_index(drop=True),q[order],CFG)[0]
    np.testing.assert_array_equal(a[2],b[2])


def test_compound_at_issuance_and_no_future_stint_filter():
    frame=data([0,2,3],[1,4,5],compound=['SOFT','HARD','SOFT']);q=quantiles(3)
    config={**CFG,'context':'race_compound'};a=model.replay(frame,q,config)[0]
    np.testing.assert_array_equal(a[1],q[1]);assert a[2,5]>q[2,5]
    frame['target_same_stint']=True;b=model.replay(frame,q,config)[0]
    np.testing.assert_array_equal(a,b)


def test_new_event_resets_state_and_coherence_point_are_exact():
    frame=data([0,2,0,2],[1,3,1,3]);frame['event_key']=[202301,202301,202302,202302];q=quantiles(4)
    pred,_,d=model.replay(frame,q,CFG)
    np.testing.assert_array_equal(pred[:2],pred[2:]);np.testing.assert_array_equal(pred[:,4],q[:,4])
    assert (np.diff(pred,axis=1)>=0).all() and np.isfinite(pred).all() and d['resolved_outcomes']==4


def test_impossible_timing_rejected_without_filtering():
    frame=data([0,1],[0,2])
    with pytest.raises(ValueError):model.replay(frame,quantiles(2),CFG)


def test_wrong_sign_and_nonfinite_update_parameters_rejected():
    frame=data([0,2],[1,3]);q=quantiles(2)
    for eta,lam in [(-.03,-.2),(-.03,.2),(.03,-.2),(float('nan'),.2),(.03,float('inf')),(0,.2)]:
        with pytest.raises(ValueError):model.replay(frame,q,{**CFG,'eta_seconds':eta,'shrink_per_second':lam})
