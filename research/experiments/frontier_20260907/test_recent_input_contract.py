import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('recent_input_recovery_test',Path(__file__).with_name('recover_recent_inputs.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def frames():
    old=pd.DataFrame({'DriverNumber':['1','2'],'LapNumber':[4,4],'Time':[400.,401.],
                      'LapTime':[90.,91.],'PitInTime':[np.nan,np.nan],'PitOutTime':[np.nan,np.nan],
                      'IsAccurate':[True,True],'Stint':[1.,1.],'Compound':['HARD','SOFT']})
    native=old.copy()
    for col in ['Sector1Time','Sector2Time','Sector3Time']:native[col]=30.
    for col in ['Time','LapTime','PitInTime','PitOutTime','Sector1Time','Sector2Time','Sector3Time']:
        native[col]=pd.to_timedelta(native[col],unit='s')
    for col in ['SpeedI1','SpeedI2','SpeedFL','SpeedST']:native[col]=280.
    native['FreshTyre']=True;native['Position']=[1.,2.]
    return old,native


def test_recovery_preserves_all_original_observations_and_explicit_seconds():
    old,native=frames();original=old.copy();out=m.merge_observed_fields(old,native)
    pd.testing.assert_frame_equal(out[original.columns],original)
    assert out.Sector1Time.tolist()==[30.,30.]
    assert set(m.EXTRA).issubset(out)


def test_missing_schema_and_unlabelled_duration_units_are_rejected():
    old,native=frames()
    with pytest.raises(ValueError,match='schema'):m.merge_observed_fields(old,native.drop(columns='Sector1Time'))
    native['Sector1Time']=30_000_000_000
    with pytest.raises(TypeError,match='units'):m.merge_observed_fields(old,native)


def test_revised_outcomes_or_changed_roster_cannot_silently_enter_recovery():
    old,native=frames();native['LapTime']=pd.to_timedelta([89.,91.],unit='s')
    with pytest.raises(AssertionError):m.merge_observed_fields(old,native)
    old,native=frames()
    with pytest.raises(ValueError,match='population'):m.merge_observed_fields(old,native.iloc[:1])
