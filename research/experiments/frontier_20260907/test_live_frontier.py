import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

path=Path(__file__).with_name('live_frontier.py')
spec=importlib.util.spec_from_file_location('frontier_live_under_test',path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def raw_stream():
    return pd.DataFrame([{'DriverNumber':str(driver),'LapNumber':lap,'Time':lap*100.+offset,
        'LapTime':90.+.07*lap+.03*driver,'IsAccurate':True,'Stint':1.,'Compound':'MEDIUM',
        'PitInTime':np.nan,'PitOutTime':np.nan,'TrackStatus':'1','TyreLife':lap,
        'Sector1Time':30.+.02*lap,'Sector2Time':30.,'Sector3Time':30.,'Position':driver}
        for lap in range(1,13) for driver,offset in [(1,0),(2,0),(3,1),(4,2)]])


def test_enriched_features_are_prefix_causal_and_target_population_matches():
    raw=raw_stream();full,paired=m.enrich(raw,202201)
    old,old_paired=m.base.stream_event(raw,202201)
    pd.testing.assert_frame_equal(full[old.columns],old)
    assert len(paired)==len(old_paired)
    truncated,_=m.enrich(raw.loc[raw.Time<=702],202201)
    pd.testing.assert_frame_equal(full.loc[full.issued_at_timestamp<=702].reset_index(drop=True),truncated.reset_index(drop=True))
    bad=raw.copy();mask=bad.Time>702
    bad.loc[mask,'LapTime']=12345.;bad.loc[mask,'Sector1Time']=900.;bad.loc[mask,'Position']=99
    bad.loc[mask,'IsAccurate']=False;bad.loc[mask,'Stint']=42
    poisoned,_=m.enrich(bad,202201)
    pd.testing.assert_frame_equal(full.loc[full.issued_at_timestamp<=702].reset_index(drop=True),poisoned.loc[poisoned.issued_at_timestamp<=702].reset_index(drop=True))


def test_same_timestamp_peer_rows_cannot_influence_each_other():
    raw=raw_stream();original,_=m.enrich(raw,202201)
    raw.loc[(raw.DriverNumber=='2') & (raw.LapNumber==6),'LapTime']=97.
    changed,_=m.enrich(raw,202201)
    row=(original.driver_id=='1') & (original.issued_after_lap_number==6)
    pd.testing.assert_frame_equal(original.loc[row].reset_index(drop=True),changed.loc[row].reset_index(drop=True))


def test_policy_uses_rewards_only_after_target_receipt():
    _,frame=m.enrich(raw_stream(),202201)
    p=m.online_policy(frame,.5)
    corrupt=frame.copy();corrupt.loc[corrupt.target_timestamp>702,'lap_time_seconds']=10000
    q=m.online_policy(corrupt,.5)
    np.testing.assert_array_equal(p[frame.issued_at_timestamp<=702],q[frame.issued_at_timestamp<=702])


def test_missing_stint_is_forward_only_and_no_target_feature_names():
    raw=raw_stream();raw.loc[raw.LapNumber.ge(5),'Stint']=np.nan
    a,_=m.enrich(raw,202201);raw.loc[raw.LapNumber.ge(9),'Stint']=2
    b,_=m.enrich(raw,202201)
    pd.testing.assert_frame_equal(a.loc[a.issued_after_lap_number.lt(9)].reset_index(drop=True),b.loc[b.issued_after_lap_number.lt(9)].reset_index(drop=True))
    features=m.base.FEATURES+[c for c in a if c.startswith(m.FEATURE_PREFIX)]
    assert not any('target' in name or 'lap_time_seconds' in name for name in features)
    assert np.isfinite(a[features].to_numpy()).all()
