import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.checkpoint import run_experiment as exp


def fixture():
    rows = []
    for driver in ["1", "2"]:
        for lap in range(1, 11):
            timestamp = lap*90.0+int(driver)*2.0
            rows.append(dict(DriverNumber=driver, LapNumber=lap, Time=timestamp,
                             LapTime=90.+.05*lap, Sector1Time=29.+.05*lap,
                             Sector2Time=31., Sector3Time=30., SpeedI1=270., SpeedI2=280.,
                             SpeedFL=290., SpeedST=310., Position=int(driver), FreshTyre=True,
                             IsAccurate=lap not in [5, 6, 7], Compound="HARD" if lap >= 6 else "SOFT",
                             Stint=2. if lap >= 6 else 1., TyreLife=lap-5 if lap >= 6 else lap,
                             PitInTime=timestamp-.2 if lap == 5 else np.nan,
                             PitOutTime=timestamp-60 if lap == 6 else np.nan,
                             TrackStatus="4" if lap == 7 else "1"))
    return pd.DataFrame(rows)


def test_match_older_checkpoints_before_refreshing_current_eligible_issuance():
    issued, matched, _ = exp.checkpoint_event(fixture(), 202201)
    d = matched.loc[matched.driver_id.eq("1")]
    multiple = d.loc[d.checkpoint_lap.isin([4, 5, 6, 7])]
    assert multiple.target_lap.tolist() == [8, 8, 8, 8]
    assert multiple.prior_lap.tolist() == [4, 4, 4, 4]
    assert d.loc[d.checkpoint_lap.eq(8), "target_lap"].item() == 9
    assert (matched.target_time > matched.checkpoint_time).all()
    assert len(issued) == 16 and len(matched) == 14


def test_prefix_and_future_poison_preserve_all_features_and_checkpoint_population():
    raw = fixture(); cutoff = 634.0
    full, _, _ = exp.checkpoint_event(raw, 202201)
    prefix, _, _ = exp.checkpoint_event(raw.loc[raw.Time <= cutoff], 202201)
    poison = raw.copy()
    mask = poison.Time > cutoff
    poison.loc[mask, ["LapTime", "Sector2Time", "TyreLife", "SpeedI2"]] = 9999.
    poison.loc[mask, "IsAccurate"] = False
    changed, _, _ = exp.checkpoint_event(poison, 202201)
    expected = full.loc[full.checkpoint_time <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, prefix, check_exact=True)
    pd.testing.assert_frame_equal(expected, changed.loc[changed.checkpoint_time <= cutoff].reset_index(drop=True), check_exact=True)
    np.testing.assert_array_equal(exp.base_points(expected), exp.base_points(prefix))


def test_future_same_row_pit_value_neither_enters_features_nor_announces_its_presence():
    raw = fixture()
    mask = raw.LapNumber.eq(5)
    raw.loc[mask, "PitInTime"] = raw.loc[mask, "Time"]+.9
    a, _, _ = exp.checkpoint_event(raw, 202201)
    absent = raw.copy(); absent.loc[mask, "PitInTime"] = np.nan
    b, _, _ = exp.checkpoint_event(absent, 202201)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert a.loc[a.checkpoint_lap.eq(5), "c_pit_in"].eq(0).all()


def test_no_future_target_is_required_to_issue_a_checkpoint():
    raw = fixture(); partial = raw.loc[raw.LapNumber <= 7]
    issued, matched, _ = exp.checkpoint_event(partial, 202201)
    assert (issued.checkpoint_lap == 7).sum() == 2
    assert not matched.checkpoint_lap.eq(7).any()
    assert not any(c.startswith("target_") for c in issued)


def test_gate_preserves_original_eligible_points_even_for_an_extreme_expert():
    _, matched, _ = exp.checkpoint_event(fixture(), 202201)
    reference = exp.base_points(matched)
    class ExtremeExpert:
        def predict(self, x): return np.full(len(x), 1000.)
    result = exp.predict_expert(matched, reference, ExtremeExpert(), {"prediction_clip_seconds":3})
    np.testing.assert_array_equal(result[matched.eligible], reference[matched.eligible])
    np.testing.assert_allclose(result[~matched.eligible]-reference[~matched.eligible], 3.)


def test_target_balanced_metric_does_not_count_a_repeated_outcome_as_two_targets():
    frame = pd.DataFrame(dict(event_key=[202201]*3, driver_id=["1"]*3,
                              target_lap=[8, 8, 9], target_seconds=[100., 100., 100.]))
    reference = np.array([110., 110., 100.]); candidate = np.array([100., 100., 100.])
    assert exp.metrics(frame, reference, candidate)["baseline_mae"] == 20/3
    assert exp.metrics(frame, reference, candidate, True)["baseline_mae"] == 5.


def test_duplicate_completed_observation_is_rejected():
    raw = fixture()
    with pytest.raises(ValueError, match="chronology"):
        exp.checkpoint_event(pd.concat([raw, raw.iloc[[5]]]), 202201)
