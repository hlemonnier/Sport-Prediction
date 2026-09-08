import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from anchors import observed_anchors, points, KEYS
spec = importlib.util.spec_from_file_location('anchor_run', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def raw_laps():
    return pd.DataFrame({'DriverNumber': [1] * 6, 'LapNumber': np.arange(1, 7),
                         'Time': np.arange(1, 7) * 100., 'LapTime': [90., 91., 92., 110., 92., 93.],
                         'IsAccurate': [True] * 6, 'Stint': [1.] * 6, 'Compound': ['SOFT'] * 6,
                         'PitInTime': [np.nan] * 6, 'PitOutTime': [np.nan] * 6, 'TrackStatus': [1] * 6})


def test_medians_and_gaps_are_unclipped():
    out = observed_anchors(raw_laps(), 202201)
    row = out.loc[out.issued_after_lap_number.eq(4)].iloc[0]
    assert row.a_median3_seconds == 92 and row.a_gap3 == -18
    assert row.a_median5_seconds == 91.5 and row.a_gap5 == -18.5


def test_future_rows_cannot_change_earlier_anchor_features():
    raw = raw_laps(); prefix = observed_anchors(raw.loc[raw.Time.le(400)], 202201)
    raw.loc[raw.Time.gt(400), ['LapTime', 'Stint']] = [9000., 10.]
    raw.loc[raw.Time.gt(400), 'Compound'] = 'WET'
    full = observed_anchors(raw, 202201)
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), full.loc[full.issued_at_timestamp.le(400)].reset_index(drop=True))


def test_future_pit_metadata_is_ignored_and_missing_state_carries_forward():
    raw = raw_laps(); original = observed_anchors(raw, 202201)
    raw.loc[2, 'PitInTime'] = 9000.
    raw.loc[2, ['Stint', 'Compound']] = [np.nan, None]
    changed = observed_anchors(raw, 202201)
    pd.testing.assert_frame_equal(original, changed)


def test_observed_pit_and_known_stint_change_reset_history():
    raw = raw_laps(); raw.loc[3, 'PitOutTime'] = 399.; raw.loc[3:, 'Stint'] = 2.
    out = observed_anchors(raw, 202201)
    row = out.loc[out.issued_after_lap_number.eq(5)].iloc[0]
    assert row.a_count3 == 1 and row.a_median3_seconds == 92 and row.a_generation == 2


def test_gate_preserves_baseline_outside_observed_support_and_gap():
    frame = observed_anchors(raw_laps(), 202201)
    baseline = frame.a_observed_seconds.to_numpy() - .2
    prediction, diagnostic = points(frame, np.zeros(len(frame)), baseline, {'window': 3, 'policy': 'gate'})
    gate = (frame.a_count3 >= 3) & (frame.a_gap3.abs() >= 1)
    np.testing.assert_array_equal(prediction[~gate], baseline[~gate])
    assert prediction[3] == 92 and diagnostic['gate_rows'] == int(gate.sum())
    assert diagnostic['invalid_point_fallbacks'] == 0


def test_nonpositive_points_fall_back_and_invalid_chronology_is_rejected():
    frame = observed_anchors(raw_laps(), 202201); baseline = frame.a_observed_seconds.to_numpy()
    prediction, diagnostic = points(frame, np.full(len(frame), -1000), baseline, {'window': 5, 'policy': 'full'})
    np.testing.assert_array_equal(prediction, baseline)
    assert diagnostic['invalid_point_fallbacks'] == len(frame)
    raw = raw_laps(); raw.loc[1, 'LapNumber'] = 1
    with pytest.raises(ValueError, match='chronology'): observed_anchors(raw, 202201)


def test_selection_ties_prefer_predeclared_gate_then_shorter_window():
    candidates = {'m5full': {'policy': 'full', 'window': 5}, 'm5gate': {'policy': 'gate', 'window': 5},
                  'm3gate': {'policy': 'gate', 'window': 3}}
    metrics = {m: {'candidate_mae': .5} for m in candidates}
    assert runner.choose(metrics, candidates) == 'm3gate'
