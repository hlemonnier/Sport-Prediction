from __future__ import annotations

from pathlib import Path

import pandas as pd

from packages.f1.features.weather import (
    OpenMeteoWeatherProvider,
    WeatherLocation,
    apply_f1_weather_to_features,
    resolve_f1_weather_location,
)


def _sample_payload() -> dict[str, object]:
    return {
        "hourly": {
            "time": ["2026-05-24T12:00", "2026-05-24T13:00", "2026-05-24T14:00"],
            "temperature_2m": [20.0, 21.0, 20.5],
            "relative_humidity_2m": [70.0, 72.0, 74.0],
            "precipitation_probability": [20.0, 80.0, 50.0],
            "precipitation": [0.0, 1.2, 0.4],
            "rain": [0.0, 1.0, 0.3],
            "cloud_cover": [60.0, 90.0, 80.0],
            "wind_speed_10m": [10.0, 12.0, 14.0],
            "wind_gusts_10m": [18.0, 25.0, 22.0],
        }
    }


def test_open_meteo_forecast_uses_cache(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_get(url: str, timeout_seconds: float) -> dict[str, object]:
        calls.append(url)
        assert timeout_seconds > 0
        return _sample_payload()

    provider = OpenMeteoWeatherProvider(cache_dir=tmp_path, http_get=fake_get)
    location = WeatherLocation("Circuit de Monaco", 43.7347, 7.4206, "Europe/Monaco")

    first = provider.fetch_forecast(location, forecast_days=3)
    second = provider.fetch_forecast(location, forecast_days=3)

    assert len(calls) == 1
    assert "api.open-meteo.com/v1/forecast" in calls[0]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.summary()["weather_wet_risk"] == 0.8


def test_open_meteo_event_summary_filters_hourly_window(tmp_path: Path) -> None:
    provider = OpenMeteoWeatherProvider(cache_dir=tmp_path, http_get=lambda _url, _timeout: _sample_payload())
    location = WeatherLocation("Circuit de Monaco", 43.7347, 7.4206, "Europe/Monaco")

    snapshot = provider.fetch_historical(
        location,
        start_date="2026-05-24",
        end_date="2026-05-24",
    )
    summary = snapshot.summary(start="2026-05-24T13:00", end="2026-05-24T13:00")

    assert summary["weather_kind"] == "historical"
    assert summary["weather_hour_count"] == 1
    assert summary["weather_precipitation_sum"] == 1.2
    assert summary["weather_wind_gusts_10m_max"] == 25.0


def test_f1_weather_resolves_static_circuit_location() -> None:
    location = resolve_f1_weather_location("Monaco Grand Prix")

    assert location is not None
    assert location.name == "Circuit de Monaco"
    assert location.timezone == "Europe/Monaco"


def test_f1_weather_summary_updates_weather_priors() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["VER", "NOR"],
            "track_weather_uncertainty": [0.10, 0.20],
            "track_weather_uncertainty_prior": [0.15, 0.25],
            "race_generation_variance_prior": [0.30, 0.40],
        }
    )

    out = apply_f1_weather_to_features(
        frame,
        {
            "weather_available": True,
            "weather_wet_risk": 0.80,
            "weather_precipitation_sum": 1.6,
            "weather_rain_sum": 1.3,
            "weather_wind_gusts_10m_max": 25.0,
        },
    )

    assert list(out["track_weather_uncertainty_prior"]) == [0.80, 0.80]
    assert list(out["track_weather_uncertainty"]) == [0.80, 0.80]
    assert float(out["race_generation_variance_prior"].iloc[0]) > 0.30
    assert "open_meteo_wet_risk" in out.columns
