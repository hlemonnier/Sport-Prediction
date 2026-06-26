"""No-key weather support for F1 circuits via Open-Meteo."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .schemas import JsonObject


@dataclass(frozen=True, slots=True)
class CircuitLocation:
    id: str
    name: str
    latitude: float
    longitude: float
    timezone: str
    aliases: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        payload = asdict(self)
        payload["aliases"] = list(self.aliases)
        return payload


CIRCUIT_LOCATIONS: tuple[CircuitLocation, ...] = (
    CircuitLocation("bahrain", "Bahrain International Circuit", 26.0325, 50.5106, "Asia/Bahrain", ("bahrain", "sakhir")),
    CircuitLocation("jeddah", "Jeddah Corniche Circuit", 21.6319, 39.1044, "Asia/Riyadh", ("jeddah", "saudi arabia")),
    CircuitLocation("albert-park", "Albert Park", -37.8497, 144.9680, "Australia/Melbourne", ("melbourne", "australia", "albert park")),
    CircuitLocation("suzuka", "Suzuka Circuit", 34.8431, 136.5410, "Asia/Tokyo", ("suzuka", "japan")),
    CircuitLocation("shanghai", "Shanghai International Circuit", 31.3389, 121.2197, "Asia/Shanghai", ("shanghai", "china")),
    CircuitLocation("miami", "Miami International Autodrome", 25.9581, -80.2389, "America/New_York", ("miami", "miami gardens")),
    CircuitLocation("imola", "Autodromo Enzo e Dino Ferrari", 44.3439, 11.7167, "Europe/Rome", ("imola", "emilia-romagna")),
    CircuitLocation("monaco", "Circuit de Monaco", 43.7347, 7.4206, "Europe/Monaco", ("monaco", "monte carlo")),
    CircuitLocation("villeneuve", "Circuit Gilles Villeneuve", 45.5000, -73.5228, "America/Toronto", ("montreal", "montréal", "canada", "villeneuve")),
    CircuitLocation("barcelona", "Circuit de Barcelona-Catalunya", 41.5700, 2.2611, "Europe/Madrid", ("barcelona", "catalunya", "spain")),
    CircuitLocation("red-bull-ring", "Red Bull Ring", 47.2197, 14.7647, "Europe/Vienna", ("spielberg", "austria", "red bull ring")),
    CircuitLocation("silverstone", "Silverstone Circuit", 52.0786, -1.0169, "Europe/London", ("silverstone", "great britain", "britain", "uk")),
    CircuitLocation("hungaroring", "Hungaroring", 47.5789, 19.2486, "Europe/Budapest", ("hungaroring", "hungary", "budapest")),
    CircuitLocation("spa", "Circuit de Spa-Francorchamps", 50.4372, 5.9714, "Europe/Brussels", ("spa", "belgium", "francorchamps")),
    CircuitLocation("zandvoort", "Circuit Zandvoort", 52.3888, 4.5409, "Europe/Amsterdam", ("zandvoort", "netherlands", "dutch")),
    CircuitLocation("monza", "Autodromo Nazionale Monza", 45.6156, 9.2811, "Europe/Rome", ("monza", "italy")),
    CircuitLocation("baku", "Baku City Circuit", 40.3725, 49.8533, "Asia/Baku", ("baku", "azerbaijan")),
    CircuitLocation("marina-bay", "Marina Bay Street Circuit", 1.2914, 103.8640, "Asia/Singapore", ("singapore", "marina bay")),
    CircuitLocation("cota", "Circuit of the Americas", 30.1328, -97.6411, "America/Chicago", ("austin", "united states", "cota", "americas")),
    CircuitLocation("mexico-city", "Autodromo Hermanos Rodriguez", 19.4042, -99.0907, "America/Mexico_City", ("mexico", "mexico city")),
    CircuitLocation("interlagos", "Interlagos", -23.7036, -46.6997, "America/Sao_Paulo", ("sao paulo", "são paulo", "brazil", "interlagos")),
    CircuitLocation("las-vegas", "Las Vegas Strip Circuit", 36.1147, -115.1728, "America/Los_Angeles", ("las vegas", "vegas")),
    CircuitLocation("losail", "Lusail International Circuit", 25.4900, 51.4542, "Asia/Qatar", ("qatar", "lusail", "losail")),
    CircuitLocation("yas-marina", "Yas Marina Circuit", 24.4672, 54.6031, "Asia/Dubai", ("abu dhabi", "yas marina", "united arab emirates")),
)

OPEN_METEO_FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"


def resolve_circuit_location(*values: Any) -> tuple[CircuitLocation, str] | tuple[None, None]:
    haystack = " ".join(str(value) for value in values if value is not None).strip()
    if not haystack:
        return None, None
    normalized = _normalize(haystack)
    for circuit in CIRCUIT_LOCATIONS:
        for alias in (circuit.id, circuit.name, *circuit.aliases):
            if _normalize(alias) in normalized:
                return circuit, alias
    return None, None


def fetch_open_meteo_forecast(
    circuit: CircuitLocation,
    *,
    forecast_days: int = 7,
    timeout_seconds: float = 20.0,
) -> JsonObject:
    days = max(1, min(16, int(forecast_days)))
    params = {
        "latitude": f"{circuit.latitude:.6f}",
        "longitude": f"{circuit.longitude:.6f}",
        "current": ",".join(
            (
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "weather_code",
                "wind_speed_10m",
                "wind_gusts_10m",
            )
        ),
        "hourly": ",".join(
            (
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation_probability",
                "precipitation",
                "rain",
                "weather_code",
                "cloud_cover",
                "wind_speed_10m",
                "wind_gusts_10m",
            )
        ),
        "forecast_days": str(days),
        "timezone": circuit.timezone,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    url = f"{OPEN_METEO_FORECAST_ENDPOINT}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "sport-prediction-f1-platform/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - fixed public API endpoint.
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Open-Meteo returned a non-object JSON payload")
    return _weather_payload(circuit, payload, url=url)


def _weather_payload(circuit: CircuitLocation, payload: JsonObject, *, url: str) -> JsonObject:
    current = payload.get("current") if isinstance(payload.get("current"), dict) else {}
    hourly_rows = _hourly_rows(payload)
    return {
        "source": "open-meteo",
        "authentication": "none",
        "requiresApiKey": False,
        "url": url,
        "circuit": circuit.to_dict(),
        "current": dict(current),
        "summary": _summarize_hourly(hourly_rows[:24]),
        "hourly": hourly_rows[:48],
        "rawUnits": payload.get("current_units", {}),
    }


def _hourly_rows(payload: JsonObject) -> list[JsonObject]:
    hourly = payload.get("hourly") if isinstance(payload.get("hourly"), dict) else {}
    times = hourly.get("time")
    if not isinstance(times, list):
        return []
    rows: list[JsonObject] = []
    for index, raw_time in enumerate(times):
        row: JsonObject = {"time": raw_time}
        for key, values in hourly.items():
            if key == "time":
                continue
            if isinstance(values, list) and index < len(values):
                row[key] = values[index]
        rows.append(row)
    return rows


def _summarize_hourly(rows: list[JsonObject]) -> JsonObject:
    return {
        "hours": len(rows),
        "avgTemperatureC": _mean(_numbers(rows, "temperature_2m")),
        "maxTemperatureC": _max(_numbers(rows, "temperature_2m")),
        "minTemperatureC": _min(_numbers(rows, "temperature_2m")),
        "maxRainMm": _max(_numbers(rows, "rain")),
        "totalPrecipitationMm": _sum(_numbers(rows, "precipitation")),
        "maxPrecipitationProbability": _max(_numbers(rows, "precipitation_probability")),
        "avgWindSpeedKmh": _mean(_numbers(rows, "wind_speed_10m")),
        "maxWindGustKmh": _max(_numbers(rows, "wind_gusts_10m")),
    }


def _numbers(rows: list[JsonObject], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _sum(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values), 3)


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return round(max(values), 3)


def _min(values: list[float]) -> float | None:
    if not values:
        return None
    return round(min(values), 3)


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
