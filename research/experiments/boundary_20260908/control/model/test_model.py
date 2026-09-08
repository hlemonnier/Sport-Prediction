import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.control.model import features as f


def stream(records):
    return pd.DataFrame([dict(Time=t, sequence=i, Status=s, status_present=present)
                         for i, (t, s, present) in enumerate(records)])


def point(records, now, prior=0., lag=15.):
    times, codes = f.status_records(stream(records))
    return f.one_point(times, codes, now, prior, lag)


def test_strict_latency_excludes_equal_clock():
    records = [(0., "1", True), (100., "4", True)]
    assert point(records, 115.)["g_current_1"] == 1
    assert point(records, 115.001)["g_current_4"] == 1
    assert point(records, 100., lag=0.)["g_current_1"] == 1


def test_message_only_update_does_not_replace_status_or_age():
    result = point([(0., "4", True), (20., "", False)], 45.)
    assert result["g_current_4"] == 1
    assert result["control_observed_time"] == 0.
    assert result["g_last_update_age"] == 30.


def test_unknown_never_clear_or_backfilled():
    records = [(0., "1", True), (20., "4", True), (40., "3", True), (80., "1", True)]
    result = point(records, 70.)
    assert result["g_known"] == 0 and result["g_current_unknown"] == 1
    assert result["g_current_1"] == 0
    assert result["g_sc_duration300"] == 20.
    assert result["g_sc_exit_missing"] == 1  # An unknown update is not an observed SC exit.


def test_vsc_ending_remains_neutralized_and_episode_does_not_restart():
    records = [(0., "1", True), (20., "6", True), (40., "7", True), (100., "1", True)]
    result = point(records, 100.)
    assert result["g_vsc_active"] == 1
    assert result["g_vsc_entry_age"] == 65.
    assert result["g_vsc_current_duration_lower_bound"] == 65.
    completed = point(records, 135.)
    assert completed["g_vsc_completed_duration"] == 80.
    assert completed["g_vsc_exit_age"] == 20.


def test_unknown_entry_is_left_censored_not_completed_episode():
    records = [(0., "", True), (20., "4", True), (50., "1", True)]
    result = point(records, 80.)
    assert result["g_sc_entry_left_censored"] == 1
    assert result["g_sc_completed_duration_missing"] == 1
    assert result["g_sc_duration300"] == 30.


def test_same_clock_uses_last_source_record_only_after_latency():
    records = [(0., "1", True), (20., "4", True), (20., "1", True)]
    result = point(records, 50.)
    assert result["g_current_1"] == 1 and result["g_sc_duration300"] == 0.


def test_prefix_and_future_poison_invariance():
    records = [(0., "1", True), (20., "4", True), (100., "5", True), (200., "1", True)]
    actual = point(records, 90., prior=30.)
    assert actual == point(records[:2], 90., prior=30.)
    poisoned = records[:2]+[(100., "3", True), (200., "7", True)]
    assert actual == point(poisoned, 90., prior=30.)


def test_duration_window_and_prior_horizon_are_clipped_at_observed_cutoff():
    result = point([(0., "1", True), (100., "4", True), (500., "1", True)], 450., prior=350.)
    assert result["g_sc_duration300"] == 300.
    assert result["g_sc_duration_since_prior"] == 100.
    assert result["g_sc_current_duration_lower_bound"] == 335.


def test_activation_preserves_eligible_and_unknown():
    frame = pd.DataFrame(dict(eligible=[True, False, False, False], g_known=[1., 0., 1., 1.],
        g_current_1=[0., 0., 1., 1.], g_nonclear_duration300=[10., 10., 0., 10.]))
    assert f.activation(frame, {"scope": "all_known_ineligible"}).tolist() == [False, False, True, True]
    assert f.activation(frame, {"scope": "flag_context"}).tolist() == [False, False, False, True]


@pytest.mark.parametrize("now, prior, lag", [(10., 11., 0.), (10., 1., -1.), (np.nan, 0., 15.)])
def test_invalid_clock_rejected(now, prior, lag):
    with pytest.raises(ValueError):
        point([(0., "1", True)], now, prior, lag)


def test_model_feature_columns_ignore_predictions_targets_and_peer_cycle2():
    from research.experiments.boundary_20260908.control.model import run_experiment as run
    frame = pd.read_pickle(run.CACHE_PATH).head(8).copy()
    extras = f.event_features(stream([(0., "1", True)]), frame)
    frame = pd.concat([frame, extras], axis=1)
    expected = run.model_features(frame)
    for name in ("target_seconds", "target_time", "target_lap", "prediction_control_l7_all", "a_fake_future"):
        frame[name] = np.arange(len(frame))*99999.
    pd.testing.assert_frame_equal(expected, run.model_features(frame))
    assert not any(c.startswith("a_") for c in expected)
