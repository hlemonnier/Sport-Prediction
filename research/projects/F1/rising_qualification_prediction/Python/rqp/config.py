"""Configuration objects for predictions."""

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
    disable_circuit_features: bool = False
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
    season_weight_year: Optional[int] = None
    season_weight_multiplier: float = 1.0


@dataclass
class PredictionResult:
    version: str
    table: "object"  # pandas DataFrame
    notes: List[str]
    model_name: str
    model_family: str
    device_used: Optional[str]
    dl_available: bool
    candidate_leaderboard: List[dict[str, Any]]
    extras: dict[str, Any] = field(default_factory=dict)
