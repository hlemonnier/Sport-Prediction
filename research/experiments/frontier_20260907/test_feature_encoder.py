import numpy as np
import pandas as pd
import pytest

from research.experiments.frontier_20260907.live_frontier import enrich
from research.experiments.frontier_20260907.feature_encoder import observed_features


def complete_stream():
    return pd.DataFrame([{'DriverNumber':str(d),'LapNumber':lap,'Time':100.*lap+d,
        'LapTime':90.+.03*lap+.02*d,'IsAccurate':True,'Stint':1.,'Compound':'MEDIUM',
        'PitInTime':np.nan,'PitOutTime':np.nan,'TrackStatus':'1','TyreLife':float(lap),
        'Sector1Time':30.+.03*lap,'Sector2Time':30.,'Sector3Time':30.,
        'SpeedI1':280.,'SpeedI2':270.,'SpeedFL':290.,'SpeedST':300.,'Position':float(d),'FreshTyre':True}
        for lap in range(1,10) for d in range(1,5)])


def test_inference_features_match_frozen_experiment_exactly():
    raw=complete_stream();expected,_=enrich(raw,202201)
    pd.testing.assert_frame_equal(expected,observed_features(raw,202201),check_exact=True)


def test_forecasts_are_possible_before_any_future_target_exists():
    raw=complete_stream();empty=observed_features(raw.loc[raw.LapNumber.le(2)],202201)
    assert empty.empty
    prefix=observed_features(raw.loc[raw.LapNumber.le(3)],202201)
    assert len(prefix)==4 and prefix.issued_after_lap_number.eq(3).all()
    full=observed_features(raw,202201)
    pd.testing.assert_frame_equal(prefix,full.loc[full.issued_after_lap_number.eq(3)].reset_index(drop=True),check_exact=True)
    assert not any(c.startswith('target_') or c=='lap_time_seconds' for c in prefix)


def test_required_observed_schema_cannot_be_silently_zero_filled():
    raw=complete_stream()
    with pytest.raises(ValueError,match='Sector1Time'):observed_features(raw.drop(columns=['Sector1Time']),202201)
    raw['Sector1Time']=pd.to_timedelta(raw.Sector1Time,unit='s')
    with pytest.raises(TypeError,match='explicit numeric seconds'):observed_features(raw,202201)
