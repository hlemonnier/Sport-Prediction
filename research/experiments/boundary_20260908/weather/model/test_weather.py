import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.weather.model import run_experiment as w


def stream(times, **overrides):
    values = {c:np.full(len(times), 1.0) for c in w.CHANNELS}
    values.update(Rainfall=np.zeros(len(times)), WindDirection=np.zeros(len(times)))
    values.update(overrides)
    return pd.DataFrame({"Time":times, **values})


def test_strict_cutoff_excludes_equal_and_future_records():
    weather = stream([99., 100., 101.], AirTemp=[10., 900., 999.])
    features, sources = w.weather_features(weather, [160.])
    assert features.w_AirTemp.item() == 10
    assert sources.AirTemp.item() == 99
    assert features.w_AirTemp_age_seconds.item() == 61


def test_each_channel_keeps_own_age_and_stale_value_is_unavailable():
    weather = stream([0., 100., 700.], AirTemp=[20., np.nan, np.nan])
    features, sources = w.weather_features(weather, [200., 800.])
    assert sources.AirTemp.tolist() == [0., 0.]
    assert features.w_AirTemp.tolist() == [20., 0.]
    assert features.w_AirTemp_missing.tolist() == [0., 1.]
    assert features.w_AirTemp_age_seconds.tolist() == [200., 800.]
    assert features.w_Humidity_missing.tolist() == [0., 0.]


def test_no_weather_remains_missing_and_never_backfilled():
    features, sources = w.weather_features(stream([200.]), [100., 260.])
    assert features.w_any_channel_available.tolist() == [0., 0.]
    assert features.w_AirTemp_never_observed.tolist() == [1., 1.]
    assert sources.isna().all().all()


def test_causal_slope_uses_actual_elapsed_minutes_and_rain_window():
    weather = stream([0., 60., 120., 360., 420., 480., 1000.],
                     AirTemp=[0., 1., 2., 6., 7., 8., 100.], Rainfall=[1., 1., 0., 0., 1., 0., 1.])
    features, _ = w.weather_features(weather, [480., 1000.])
    assert features.w_AirTemp_slope_per_minute.tolist() == [1., 0.]
    assert features.w_AirTemp_slope_missing.tolist() == [0., 1.]
    # At t480 cutoff420 excludes the rain transition at420.
    assert features.w_rain_sample_count.iloc[0] == 4
    assert features.w_rain_positive_count.iloc[0] == 2
    assert features.w_rain_changes.iloc[0] == 1
    # At t1000 the window starts100: transitions120->360->420->480 only.
    assert features.w_rain_sample_count.iloc[1] == 4
    assert features.w_rain_positive_count.iloc[1] == 1
    assert features.w_rain_changes.iloc[1] == 2


def test_wind_direction_wraps_without_discontinuity():
    a, _ = w.weather_features(stream([10.], WindDirection=[0.], WindSpeed=[4.]), [100.])
    b, _ = w.weather_features(stream([10.], WindDirection=[360.], WindSpeed=[4.]), [100.])
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert a.w_wind_x.item() == 4


def test_future_weather_poison_and_truncation_leave_issued_features_identical():
    weather = stream(np.arange(0., 1001., 60.), Rainfall=np.arange(17)%2)
    full, _ = w.weather_features(weather, [150., 300., 500.])
    prefix = weather.loc[weather.Time < 440.].copy()
    truncated, _ = w.weather_features(prefix, [150., 300., 500.])
    pd.testing.assert_frame_equal(full, truncated)
    poisoned = weather.copy()
    poisoned.loc[poisoned.Time >= 440., "AirTemp"] = 99999
    poisoned.loc[poisoned.Time >= 440., "Rainfall"] = 1
    altered, _ = w.weather_features(poisoned, [150., 300., 500.])
    pd.testing.assert_frame_equal(full, altered)


def test_all_four_variants_preserve_base_without_support_and_rain_gate():
    class Dummy:
        def predict(self, x): return np.full(len(x), 2.)
    frame = pd.DataFrame({"prior_naive_seconds":[90., 91., 92.],
                          **{"p_"+c:np.zeros(3) for c in w.c1.BASE_FEATURES}})
    weather, _ = w.weather_features(stream([100., 200.], Rainfall=[0., 1.]), [50., 190., 300.])
    base = np.array([90.25, 91.25, 92.25])
    for variant in w.SPEC["family"]["grid"]:
        point = w.candidate_points(frame, weather, base, Dummy(), variant)
        assert point[0] == base[0]
        assert point[2] == 94.
        assert point[1] == (base[1] if variant["scope"] == "observed_recent_rain" else 93.)


def test_augmentation_has_no_target_dependency_or_legacy_mutation():
    frame = pd.DataFrame({"event_key":[202201, 202201, 202202], "checkpoint_time":[200., 300., 400.],
                          "target_seconds":[90., 91., 92.], **{"p_"+c:np.arange(3.) for c in w.c1.BASE_FEATURES}}, index=[4, 3, 9])
    before = frame.copy(deep=True)
    weather = {202201:stream([0., 100., 200.]), 202202:stream([0., 200.])}
    result = w.augment(frame, weather)
    changed = frame.drop(columns="target_seconds")
    pd.testing.assert_frame_equal(result, w.augment(changed, weather))
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(w.model_features(frame, result)[w.c1.BASE_FEATURES], w.c1.prior_features(frame))


def test_invalid_weather_order_and_rain_values_rejected():
    with pytest.raises(ValueError, match="unique"):
        w.weather_features(stream([10., 10.]), [100.])
    with pytest.raises(ValueError, match="binary"):
        w.weather_features(stream([10.], Rainfall=[2.]), [100.])
