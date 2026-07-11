"""F1 session prediction configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class PredictionConfig:
    source: str
    mode: str
    year: int
    round_number: int
    train_seasons: List[int]
    include_standings: bool
    cache_dir: Optional[str]
    meeting_name: Optional[str]
    country_name: Optional[str]
    weekends_dir: Optional[str]
    enable_dl_candidates: bool = False
    compare_families: List[str] = field(default_factory=lambda: ["ml"])
    dl_device: str = "auto"
    dl_arch: str = "mlp_tabular_v1"
    dl_hyperparams: dict[str, Any] = field(default_factory=dict)
    dl_seed: int = 42
    disable_runsim_features: bool = False
    disable_circuit_features: bool = True
    f1_model: str = "auto"
    f1_listwise: str = "pl_gumbel"
    f1_pl_samples: int = 2000
    f1_pl_temperature: float = 1.0
    f1_listwise_seed: int = 42
    shadow_eval: bool = True
    f1_mode: str = "offline"
    f1_live_source: str = "auto"
    f1_live_model: str = "ssm_v1"
    f1_live_horizon_laps: int = 10
    f1_live_seed: int = 42
    f1_live_cache_dir: Optional[str] = None
    f1_live_replay_path: Optional[str] = None
    f1_live_calibration_path: Optional[str] = None
    f1_live_replay_cutoff_lap: Optional[int] = None
    season_weight_year: Optional[int] = None
    season_weight_multiplier: float = 1.0
    race_delta_constraint_mode: str = "constrained"
    race_information_horizon: str = "auto"
    weather_enabled: bool = False
    weather_provider: str = "open_meteo"
    weather_latitude: Optional[float] = None
    weather_longitude: Optional[float] = None
    weather_timezone: Optional[str] = None
    weather_start: Optional[str] = None
    weather_end: Optional[str] = None
    weather_cache_dir: Optional[str] = None
