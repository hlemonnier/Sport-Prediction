"""Configuration objects for match result prediction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictionConfig:
    league: str
    season: int
    round_number: int
    mode: str
    data_source: str | None = None
    train_seasons: list[int] | None = None
    cache_dir: str | None = None
    football_model: str = "dixon"
    football_calibration: str = "auto"
    shadow_eval: bool = True
    weather_enabled: bool = False
    weather_provider: str = "open_meteo"
    weather_latitude: float | None = None
    weather_longitude: float | None = None
    weather_timezone: str | None = None
    weather_cache_dir: str | None = None
    weather_hours_before: float = 2.0
    weather_hours_after: float = 3.0
