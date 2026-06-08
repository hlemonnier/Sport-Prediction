"""Open-Meteo weather provider with forecast and historical JSON caching.

The provider is intentionally sport-agnostic. F1 circuits, football stadiums,
and future sports should pass a WGS84 location plus an event time window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "rain",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "wind_speed_10m",
    "wind_gusts_10m",
)

JsonGet = Callable[[str, float], Mapping[str, Any]]


@dataclass(frozen=True)
class WeatherLocation:
    name: str
    latitude: float
    longitude: float
    timezone: str = "auto"

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WeatherSnapshot:
    source: str
    kind: str
    location: WeatherLocation
    requested_start: str | None
    requested_end: str | None
    fetched_at: str
    payload: Mapping[str, Any]
    cache_hit: bool = False
    cache_path: str | None = None

    def hourly_rows(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> list[dict[str, object]]:
        hourly = self.payload.get("hourly")
        if not isinstance(hourly, Mapping):
            return []
        times = hourly.get("time")
        if not isinstance(times, list):
            return []

        start_dt = _parse_datetime(start) if start is not None else None
        end_dt = _parse_datetime(end) if end is not None else None
        rows: list[dict[str, object]] = []
        for idx, raw_time in enumerate(times):
            row_time = _parse_datetime(str(raw_time))
            if start_dt is not None and row_time < start_dt:
                continue
            if end_dt is not None and row_time > end_dt:
                continue
            row: dict[str, object] = {"time": str(raw_time)}
            for key, values in hourly.items():
                if key == "time":
                    continue
                if isinstance(values, list) and idx < len(values):
                    row[str(key)] = values[idx]
            rows.append(row)
        return rows

    def summary(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> dict[str, object]:
        summary = summarize_hourly_weather(self.hourly_rows(start=start, end=end))
        summary.update(
            {
                "weather_source": self.source,
                "weather_kind": self.kind,
                "weather_location_name": self.location.name,
                "weather_latitude": self.location.latitude,
                "weather_longitude": self.location.longitude,
                "weather_timezone": self.location.timezone,
                "weather_cache_hit": self.cache_hit,
            }
        )
        return summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: datetime | str | date) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _date_text(value: datetime | str | date) -> str:
    return _parse_datetime(value).date().isoformat()


def _json_get(url: str, timeout_seconds: float) -> Mapping[str, Any]:
    request = Request(url, headers={"User-Agent": "sport-prediction-weather/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - controlled public API URL.
        raw = response.read().decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Open-Meteo returned a non-object JSON payload.")
    return payload


def _coerce_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / float(len(clean))


def _sum(values: Iterable[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean)


def _max(values: Iterable[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return max(clean)


def _feature_value(rows: list[dict[str, object]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        parsed = _coerce_float(row.get(column))
        if parsed is not None:
            values.append(parsed)
    return values


def summarize_hourly_weather(rows: list[dict[str, object]]) -> dict[str, object]:
    """Convert hourly weather rows into model-friendly event-level features."""

    if not rows:
        return {
            "weather_available": False,
            "weather_hour_count": 0,
            "weather_wet_risk": 0.0,
        }

    precipitation_probability = _feature_value(rows, "precipitation_probability")
    precipitation = _feature_value(rows, "precipitation")
    rain = _feature_value(rows, "rain")
    temperature = _feature_value(rows, "temperature_2m")
    humidity = _feature_value(rows, "relative_humidity_2m")
    wind_speed = _feature_value(rows, "wind_speed_10m")
    wind_gusts = _feature_value(rows, "wind_gusts_10m")
    cloud_cover = _feature_value(rows, "cloud_cover")
    pressure = _feature_value(rows, "pressure_msl")
    weather_codes = _feature_value(rows, "weather_code")

    precip_probability_max = _max(precipitation_probability)
    precipitation_sum = _sum(precipitation) or 0.0
    rain_sum = _sum(rain) or 0.0
    probability_component = (
        min(1.0, max(0.0, precip_probability_max / 100.0))
        if precip_probability_max is not None
        else 0.0
    )
    precipitation_component = min(1.0, max(0.0, precipitation_sum / 5.0))
    rain_component = min(1.0, max(0.0, rain_sum / 3.0))
    wet_risk = max(probability_component, precipitation_component, rain_component)

    return {
        "weather_available": True,
        "weather_hour_count": len(rows),
        "weather_start_time": rows[0].get("time"),
        "weather_end_time": rows[-1].get("time"),
        "weather_temperature_2m_mean": _mean(temperature),
        "weather_relative_humidity_2m_mean": _mean(humidity),
        "weather_precipitation_probability_max": precip_probability_max,
        "weather_precipitation_sum": precipitation_sum,
        "weather_rain_sum": rain_sum,
        "weather_wet_risk": wet_risk,
        "weather_cloud_cover_mean": _mean(cloud_cover),
        "weather_pressure_msl_mean": _mean(pressure),
        "weather_wind_speed_10m_mean": _mean(wind_speed),
        "weather_wind_gusts_10m_max": _max(wind_gusts),
        "weather_code_max": _max(weather_codes),
    }


class OpenMeteoWeatherProvider:
    """Open-Meteo forecast and archive client with file-system cache."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        timeout_seconds: float = 20.0,
        forecast_ttl_seconds: int = 3600,
        http_get: JsonGet | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.timeout_seconds = float(timeout_seconds)
        self.forecast_ttl_seconds = int(forecast_ttl_seconds)
        self.http_get = http_get or _json_get
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_cache_root(cls, cache_root: str | Path | None) -> "OpenMeteoWeatherProvider":
        root = Path(cache_root).expanduser() if cache_root else Path(".cache")
        return cls(root / "weather" / "open_meteo")

    def fetch_forecast(
        self,
        location: WeatherLocation,
        *,
        forecast_days: int = 7,
        hourly_variables: Iterable[str] | None = None,
        force_refresh: bool = False,
    ) -> WeatherSnapshot:
        days = max(1, min(16, int(forecast_days)))
        params = self._base_params(location, hourly_variables)
        params["forecast_days"] = str(days)
        return self._fetch(
            endpoint=FORECAST_ENDPOINT,
            params=params,
            kind="forecast",
            location=location,
            requested_start=None,
            requested_end=None,
            ttl_seconds=self.forecast_ttl_seconds,
            force_refresh=force_refresh,
        )

    def fetch_historical(
        self,
        location: WeatherLocation,
        *,
        start_date: datetime | str | date,
        end_date: datetime | str | date,
        hourly_variables: Iterable[str] | None = None,
        force_refresh: bool = False,
    ) -> WeatherSnapshot:
        params = self._base_params(location, hourly_variables)
        params["start_date"] = _date_text(start_date)
        params["end_date"] = _date_text(end_date)
        return self._fetch(
            endpoint=HISTORICAL_ENDPOINT,
            params=params,
            kind="historical",
            location=location,
            requested_start=params["start_date"],
            requested_end=params["end_date"],
            ttl_seconds=None,
            force_refresh=force_refresh,
        )

    def fetch_event_weather(
        self,
        location: WeatherLocation,
        *,
        start: datetime | str,
        end: datetime | str,
        hourly_variables: Iterable[str] | None = None,
        force_refresh: bool = False,
    ) -> WeatherSnapshot:
        start_dt = _parse_datetime(start)
        end_dt = _parse_datetime(end)
        today = datetime.now().date()
        if end_dt.date() < today:
            return self.fetch_historical(
                location,
                start_date=start_dt,
                end_date=end_dt,
                hourly_variables=hourly_variables,
                force_refresh=force_refresh,
            )

        days_until_end = (end_dt.date() - today).days + 1
        return self.fetch_forecast(
            location,
            forecast_days=max(1, min(16, days_until_end)),
            hourly_variables=hourly_variables,
            force_refresh=force_refresh,
        )

    def _base_params(
        self,
        location: WeatherLocation,
        hourly_variables: Iterable[str] | None,
    ) -> dict[str, str]:
        variables = tuple(hourly_variables or DEFAULT_HOURLY_VARIABLES)
        return {
            "latitude": f"{float(location.latitude):.6f}",
            "longitude": f"{float(location.longitude):.6f}",
            "hourly": ",".join(variables),
            "timezone": location.timezone or "auto",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }

    def _cache_file(self, endpoint: str, params: Mapping[str, str]) -> Path | None:
        if self.cache_dir is None:
            return None
        canonical = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(
        self,
        path: Path | None,
        ttl_seconds: int | None,
    ) -> Mapping[str, Any] | None:
        if path is None or not path.exists():
            return None
        if ttl_seconds is not None:
            age_seconds = datetime.now().timestamp() - path.stat().st_mtime
            if age_seconds > ttl_seconds:
                return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
        except Exception:
            return None
        payload = cached.get("payload") if isinstance(cached, Mapping) else None
        return payload if isinstance(payload, Mapping) else None

    def _write_cache(self, path: Path | None, url: str, payload: Mapping[str, Any]) -> None:
        if path is None:
            return
        envelope = {
            "source": "open_meteo",
            "url": url,
            "fetched_at": _utc_now(),
            "payload": payload,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(envelope, handle, ensure_ascii=False, indent=2)

    def _fetch(
        self,
        *,
        endpoint: str,
        params: Mapping[str, str],
        kind: str,
        location: WeatherLocation,
        requested_start: str | None,
        requested_end: str | None,
        ttl_seconds: int | None,
        force_refresh: bool,
    ) -> WeatherSnapshot:
        query = urlencode(dict(params))
        url = f"{endpoint}?{query}"
        cache_file = self._cache_file(endpoint, params)
        if not force_refresh:
            cached = self._read_cache(cache_file, ttl_seconds)
            if cached is not None:
                return WeatherSnapshot(
                    source="open_meteo",
                    kind=kind,
                    location=location,
                    requested_start=requested_start,
                    requested_end=requested_end,
                    fetched_at=_utc_now(),
                    payload=cached,
                    cache_hit=True,
                    cache_path=str(cache_file) if cache_file else None,
                )

        payload = self.http_get(url, self.timeout_seconds)
        self._write_cache(cache_file, url, payload)
        return WeatherSnapshot(
            source="open_meteo",
            kind=kind,
            location=location,
            requested_start=requested_start,
            requested_end=requested_end,
            fetched_at=_utc_now(),
            payload=payload,
            cache_hit=False,
            cache_path=str(cache_file) if cache_file else None,
        )
