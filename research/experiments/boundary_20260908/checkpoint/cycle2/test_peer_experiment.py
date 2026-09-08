import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.checkpoint import run_experiment as c1
from research.experiments.boundary_20260908.checkpoint.test_experiment import fixture as small_fixture
from research.experiments.boundary_20260908.checkpoint.cycle2 import run_experiment as exp


def fixture():
    pieces=[]
    source=small_fixture().loc[lambda f:f.DriverNumber.eq("1")]
    for driver in range(1,6):
        frame=source.copy();shift=(driver-1)*2.
        frame.DriverNumber=str(driver);frame.Position=driver
        frame.Time+=shift;frame.PitInTime+=shift;frame.PitOutTime+=shift
        pieces.append(frame)
    return pd.concat(pieces,ignore_index=True)


def enriched(raw):
    issued,matched,_=c1.checkpoint_event(raw,202201)
    peer=exp.peer_features(raw,issued)
    return issued.merge(peer,on=c1.KEYS,validate="one_to_one"),matched.merge(peer,on=c1.KEYS,validate="one_to_one")


def test_new_peers_are_strictly_later_than_original_point_and_before_checkpoint():
    issued,_=enriched(fixture())
    row=issued.loc[issued.driver_id.eq("1") & issued.checkpoint_lap.eq(5)].iloc[0]
    assert row.a_new_peer_count==4
    assert row.prior_time<row.peer_evidence_max_timestamp<row.checkpoint_time
    assert row.a_paired_change_count>=3


def test_same_timestamp_other_car_observation_is_excluded():
    raw=fixture();mask=raw.DriverNumber.eq("2") & raw.LapNumber.eq(5)
    raw.loc[mask,"Time"]=452.;raw.loc[mask,"PitInTime"]=np.nan;raw.loc[mask,"IsAccurate"]=True
    before,_=enriched(raw)
    raw.loc[mask,"LapTime"]=999.
    after,_=enriched(raw)
    pick=lambda f:f.loc[f.driver_id.eq("1") & f.checkpoint_lap.eq(5),[c for c in f if c.startswith("a_")]]
    pd.testing.assert_frame_equal(pick(before),pick(after),check_exact=True)


def test_truncation_and_future_poison_preserve_peer_features():
    raw=fixture();cutoff=542.
    full,_=enriched(raw);prefix,_=enriched(raw.loc[raw.Time<=cutoff])
    poisoned=raw.copy();poisoned.loc[poisoned.Time>cutoff,"LapTime"]=9999.
    after,_=enriched(poisoned)
    expected=full.loc[full.checkpoint_time<=cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,prefix.reset_index(drop=True),check_exact=True)
    pd.testing.assert_frame_equal(expected,after.loc[after.checkpoint_time<=cutoff].reset_index(drop=True),check_exact=True)


def test_no_support_exactly_returns_cycle1_and_eligible_points_never_change():
    _,frame=enriched(fixture());base=c1.base_points(frame)
    previous=base.copy();previous[~frame.eligible]+=.25
    class Extreme:
        def predict(self,x):return np.full(len(x),999.)
    result=exp.predict(frame,base,previous,Extreme(),{"prediction_clip_seconds":1.})
    gate=exp.active(frame)
    np.testing.assert_array_equal(result[~gate],previous[~gate])
    np.testing.assert_array_equal(result[frame.eligible],base[frame.eligible])
    np.testing.assert_allclose(result[gate]-previous[gate],1.)


def test_feature_matrix_does_not_depend_on_future_target_values_or_flags():
    _,frame=enriched(fixture());base=c1.base_points(frame)
    before=exp.features(frame,base,base)
    changed=frame.copy();changed.target_seconds=99999.;changed.target_stint_transition=~changed.target_stint_transition
    pd.testing.assert_frame_equal(before,exp.features(changed,base,base),check_exact=True)
