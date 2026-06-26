import json

from f1_platform.weather import fetch_open_meteo_forecast, resolve_circuit_location


def test_resolve_circuit_location_matches_fastf1_schedule_names():
    circuit, matched_by = resolve_circuit_location("Spielberg", "Austria", "Austrian Grand Prix")

    assert circuit is not None
    assert circuit.id == "red-bull-ring"
    assert circuit.name == "Red Bull Ring"
    assert matched_by in {"spielberg", "austria", "red bull ring"}


def test_fetch_open_meteo_forecast_marks_no_api_key(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {
                    "current": {
                        "time": "2026-06-26T15:00",
                        "temperature_2m": 23.4,
                        "precipitation": 0,
                        "wind_speed_10m": 11.2,
                    },
                    "current_units": {"temperature_2m": "°C", "wind_speed_10m": "km/h"},
                    "hourly": {
                        "time": ["2026-06-26T15:00", "2026-06-26T16:00"],
                        "temperature_2m": [23.4, 22.9],
                        "precipitation_probability": [5, 10],
                        "precipitation": [0, 0.2],
                        "rain": [0, 0.1],
                        "wind_speed_10m": [11.2, 12.0],
                        "wind_gusts_10m": [18.2, 19.0],
                    },
                }
            ).encode("utf-8")

    monkeypatch.setattr("f1_platform.weather.urlopen", lambda request, timeout: FakeResponse())
    circuit, _ = resolve_circuit_location("Spielberg")
    assert circuit is not None

    payload = fetch_open_meteo_forecast(circuit, forecast_days=3)

    assert payload["source"] == "open-meteo"
    assert payload["requiresApiKey"] is False
    assert payload["authentication"] == "none"
    assert payload["current"]["temperature_2m"] == 23.4
    assert payload["summary"]["totalPrecipitationMm"] == 0.2
    assert payload["circuit"]["id"] == "red-bull-ring"
