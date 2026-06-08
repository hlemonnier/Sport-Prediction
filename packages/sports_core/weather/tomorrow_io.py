"""Tomorrow.io provider contract.

Tomorrow.io is intentionally a second provider: Open-Meteo remains the default
free provider, while Tomorrow.io is reserved for paid nowcasting/live strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .open_meteo import WeatherLocation, WeatherSnapshot


@dataclass(frozen=True)
class TomorrowIOConfig:
    api_key: str
    base_url: str = "https://api.tomorrow.io/v4"


class TomorrowIOWeatherProvider:
    """Contract placeholder for premium nowcasting integration."""

    def __init__(self, config: TomorrowIOConfig) -> None:
        self.config = config

    def fetch_event_weather(
        self,
        location: WeatherLocation,
        *,
        start: object,
        end: object,
        fields: tuple[str, ...] = (),
    ) -> WeatherSnapshot:
        raise NotImplementedError(
            "Tomorrow.io is reserved for the live nowcasting phase; use Open-Meteo first."
        )

    def request_params(self, location: WeatherLocation, fields: tuple[str, ...]) -> Mapping[str, object]:
        return {
            "location": f"{location.latitude},{location.longitude}",
            "fields": fields,
            "timezone": location.timezone,
        }
