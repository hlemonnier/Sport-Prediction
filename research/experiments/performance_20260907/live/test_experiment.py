import numpy as np
import pandas as pd

from research.experiments.performance_20260907.live.run_experiment import (
    FEATURES, attach_selected_forecasts, ridge_fit, ridge_predict, stream_event,
)


def stream():
    return pd.DataFrame([
        dict(DriverNumber=driver,LapNumber=lap,Time=100.*lap,
             LapTime=90.+driver*.2+lap*.03,Compound="MEDIUM",Stint=1.,TyreLife=lap,
             IsAccurate=True,TrackStatus="1",PitInTime=np.nan,PitOutTime=np.nan)
        for lap in range(1,9) for driver in range(1,5)
    ])


def test_future_target_and_quality_poisoning_does_not_change_any_mechanism():
    raw=stream()
    full,matched=stream_event(raw,202201)
    model=ridge_fit(matched,10.)
    prefix,_=stream_event(raw,202201,end_at=500.)
    poisoned=raw.copy()
    future=poisoned.Time>500
    poisoned.loc[future,"LapTime"]=999999.
    poisoned.loc[future,"IsAccurate"]=False
    poisoned.loc[future,"Stint"]=15.
    poison,_=stream_event(poisoned,202201)
    expected=full.loc[full.issued_at_timestamp<=500].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected,prefix.reset_index(drop=True),check_exact=True)
    pd.testing.assert_frame_equal(expected,poison.reset_index(drop=True),check_exact=True)
    np.testing.assert_array_equal(ridge_predict(expected,model),ridge_predict(poison,model))
    assert set(FEATURES).issubset(expected)


def test_timestamp_ties_are_order_invariant_and_never_use_tied_other_cars():
    raw=stream()
    left,_=stream_event(raw,202201)
    right,_=stream_event(raw.iloc[::-1],202201)
    pd.testing.assert_frame_equal(left,right,check_exact=True)
    assert left.common_other_drivers.gt(0).all()
    assert left.common_evidence_max_timestamp.lt(left.issued_at_timestamp).all()


def test_unknown_stint_is_not_backward_filled_and_pit_out_resets_local_memory():
    raw=stream()
    raw.loc[raw.LapNumber<6,"Stint"]=np.nan
    raw.loc[raw.LapNumber==6,"PitOutTime"]=599.
    raw.loc[raw.LapNumber>=6,"Stint"]=2.
    full,_=stream_event(raw,202201)
    prefix,_=stream_event(raw,202201,end_at=500.)
    pd.testing.assert_frame_equal(full.loc[full.issued_at_timestamp<=500].reset_index(drop=True),prefix,check_exact=True)
    first_new=full.loc[full.issued_after_lap_number==7]
    assert first_new.stint_clean_count.eq(1).all()
    assert first_new.own_last_delta.eq(0.).all()
    assert first_new.robust_trend_w1.eq(first_new.forecast_naive_seconds).all()


def test_issuance_is_not_removed_when_future_target_is_ineligible_or_absent():
    raw=stream()
    emitted,_=stream_event(raw,202201,end_at=300.)
    assert len(emitted)==4
    spoiled=raw.copy()
    spoiled.loc[spoiled.LapNumber>3,"IsAccurate"]=False
    unchanged,matched=stream_event(spoiled,202201)
    pd.testing.assert_frame_equal(emitted,unchanged,check_exact=True)
    assert matched.empty


def test_attaching_candidate_outputs_preserves_ridge_input_units_and_predictions():
    _,matched=stream_event(stream(),202201)
    model=ridge_fit(matched,10.)
    expected=ridge_predict(matched,model)
    selected={"common_increment":"common_increment_w0.25", "ridge_correction":"ridge_correction_a10"}
    out=attach_selected_forecasts(matched,selected,model)
    pd.testing.assert_frame_equal(matched[FEATURES],out[FEATURES],check_exact=True)
    np.testing.assert_array_equal(expected,out.prediction_ridge_correction)
    assert out.common_increment.abs().lt(1.).all()
    assert out.prediction_common_increment.gt(80.).all()
