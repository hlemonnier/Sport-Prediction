import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.online_residual.run import (
    CONTEXT, assimilate, metric,
)


def frame():
    rows = [
        ("1", 1., 2., 92., 1),
        ("1", 2., 4., 95., 2),
        ("2", 2., 3., 93., 2),
        ("3", 2.5, 5., 97., 2),
        ("4", 3.5, 6., 99., 3),
    ]
    return pd.DataFrame([
        dict.fromkeys(CONTEXT, 0.) | {
            "event_key": 202201, "driver_id": driver,
            "issued_at_timestamp": issued, "target_timestamp": target,
            "issued_after_lap_number": lap, "target_lap_number": lap+1,
            "lap_time_seconds": y, "forecast_naive_seconds": 90.,
            "stint_generation": 1, "compound": "MEDIUM", "target_same_stint": True,
        } for driver, issued, target, y, lap in rows
    ])


def test_own_completion_is_available_but_equal_timestamp_peer_is_not():
    data = frame()
    x = assimilate(data, np.full(len(data), 90.))
    assert x.loc[1, "r_own_last"] == 2.
    assert x.loc[2, "r_peer_count"] == 0.
    assert x.loc[3, "r_peer_median"] == 2.
    assert x.loc[4, "r_peer_median"] == 2.5


def test_future_outcomes_cannot_change_earlier_features():
    data = frame()
    expected = assimilate(data, np.full(len(data), 90.))
    poisoned = data.copy()
    poisoned.loc[poisoned.target_timestamp > 2.5, "lap_time_seconds"] = -1e9
    poisoned.loc[poisoned.target_timestamp > 2.5, "target_same_stint"] = False
    actual = assimilate(poisoned, np.full(len(data), 90.))
    pd.testing.assert_frame_equal(expected.loc[data.issued_at_timestamp <= 2.5],
                                  actual.loc[data.issued_at_timestamp <= 2.5], check_exact=True)


def test_stint_event_and_unresolved_target_boundaries():
    data = frame()
    data.loc[1, "stint_generation"] = 2
    x = assimilate(data, np.full(len(data), 90.))
    assert x.loc[1, "r_own_count"] == 0.
    data.loc[0, "target_same_stint"] = False
    x = assimilate(data, np.full(len(data), 90.))
    assert x.loc[3, "r_peer_count"] == 0.
    second = data.copy()
    second["event_key"] = 202202
    joined = pd.concat([data, second], ignore_index=True)
    both = assimilate(joined, np.full(len(joined), 90.))
    np.testing.assert_array_equal(both.iloc[:len(data)].to_numpy(), both.iloc[len(data):].to_numpy())
    data.loc[0, "target_timestamp"] = data.loc[0, "issued_at_timestamp"]
    with pytest.raises(ValueError, match="strictly later"):
        assimilate(data, np.full(len(data), 90.))


def test_metrics_weight_events_equally_and_pair_common_population():
    data = pd.DataFrame({"event_key": [1,2,2,2], "lap_time_seconds": [0.,0.,0.,0.]})
    report = metric(data, [0.,2.,2.,2.], [2.,1.,1.,1.])
    assert report["baseline_mae"] == 1.5
    assert report["candidate_mae"] == 1.
    assert report["events_won"] == 1
    assert report["loo_max_delta"] == 1.
