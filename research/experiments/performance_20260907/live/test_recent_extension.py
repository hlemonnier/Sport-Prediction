import numpy as np
import pandas as pd
import pytest

from research.experiments.performance_20260907.live.run_recent_extension import native_laps_to_seconds


def test_native_lap_timedeltas_become_seconds_and_missing_pit_times_stay_missing():
    frame=pd.DataFrame(dict(DriverNumber=["1"],Driver=["VER"],LapNumber=[4.],Stint=[1.],Compound=["MEDIUM"],TyreLife=[4.],TrackStatus=["1"],IsAccurate=[True]))
    frame["Time"]=pd.to_timedelta([3690.125],unit="s")
    frame["LapTime"]=pd.to_timedelta([90.123],unit="s")
    frame["PitInTime"]=pd.to_timedelta([np.nan],unit="s")
    frame["PitOutTime"]=pd.to_timedelta([np.nan],unit="s")
    result=native_laps_to_seconds(frame)
    assert result.Time.iloc[0]==pytest.approx(3690.125)
    assert result.LapTime.iloc[0]==pytest.approx(90.123)
    assert result.PitInTime.isna().all() and result.PitOutTime.isna().all()
    frame["Time"]=3690125000000
    with pytest.raises(TypeError,match="timedelta"):
        native_laps_to_seconds(frame)
