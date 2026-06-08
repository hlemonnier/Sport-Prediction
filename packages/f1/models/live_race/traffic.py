"""Traffic-loss priors for the live-race simulator.

The model is deliberately small and deterministic.  It gives the Phase 5
simulator a shared traffic proxy that depends on the live state, circuit
overtaking prior, position, gaps, and track status without pretending to model
the full multi-car game yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from packages.f1.models.live_race.action_space import ACTION_PIT_NOW, StrategyAction
from packages.f1.models.live_race.environment import StrategyState


@dataclass(frozen=True)
class TrafficModelConfig:
    base_loss_seconds: float = 0.10
    close_gap_loss_seconds: float = 0.42
    position_loss_seconds: float = 0.22
    chaos_loss_seconds: float = 0.28
    pit_rejoin_multiplier: float = 1.25
    yellow_multiplier: float = 0.70
    sc_vsc_multiplier: float = 0.20
    red_multiplier: float = 0.0
    max_loss_seconds: float = 4.0


def _finite(value: object, default: float) -> float:
    try:
        if value is None or not np.isfinite(float(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _metadata_float(state: StrategyState, key: str, default: float = float("nan")) -> float:
    metadata = state.metadata if isinstance(state.metadata, Mapping) else {}
    return _finite(metadata.get(key), default)


def overtake_difficulty_proxy(state: StrategyState, *, scenario_overtake_propensity: Optional[float] = None) -> float:
    """Return a 0..1 proxy where 1 means overtaking is very difficult."""

    if scenario_overtake_propensity is not None:
        return float(np.clip(1.0 - float(scenario_overtake_propensity), 0.0, 1.0))
    if state.circuit_overtaking_difficulty is not None:
        return float(np.clip(float(state.circuit_overtaking_difficulty), 0.0, 1.0))
    if state.track_overtake_propensity is not None:
        return float(np.clip(1.0 - float(state.track_overtake_propensity), 0.0, 1.0))
    return 0.55


def estimate_traffic_loss_seconds(
    state: StrategyState,
    *,
    action: Optional[StrategyAction] = None,
    config: TrafficModelConfig | None = None,
    scenario_multiplier: float = 1.0,
    scenario_overtake_propensity: Optional[float] = None,
) -> float:
    """Estimate one-lap traffic loss in seconds.

    Inputs are live-available proxies only.  ``gap_to_car_ahead_seconds`` may be
    supplied in state metadata; otherwise we use gap-to-leader as a weak pressure
    signal and fall back to a neutral value.
    """

    cfg = config or TrafficModelConfig()
    difficulty = overtake_difficulty_proxy(state, scenario_overtake_propensity=scenario_overtake_propensity)
    position = _finite(state.position, 10.0)
    position_factor = float(np.clip((position - 1.0) / 19.0, 0.0, 1.0))

    gap_ahead = _metadata_float(state, "gap_to_car_ahead_seconds", float("nan"))
    if not np.isfinite(gap_ahead):
        gap_ahead = _finite(state.gap_to_leader_seconds, float("nan"))
    if np.isfinite(gap_ahead):
        close_gap_pressure = float(np.clip(1.0 - (gap_ahead / 3.0), 0.0, 1.0))
    else:
        close_gap_pressure = 0.40

    chaos = _finite(state.track_chaos_index, _metadata_float(state, "track_chaos_index", 0.25))
    raw = (
        float(cfg.base_loss_seconds)
        + (float(cfg.close_gap_loss_seconds) * difficulty * close_gap_pressure)
        + (float(cfg.position_loss_seconds) * difficulty * position_factor)
        + (float(cfg.chaos_loss_seconds) * float(np.clip(chaos, 0.0, 1.0)))
    )

    if action is not None and action.action_type == ACTION_PIT_NOW:
        raw *= float(cfg.pit_rejoin_multiplier)
    if state.is_red:
        raw *= float(cfg.red_multiplier)
    elif state.is_sc_vsc:
        raw *= float(cfg.sc_vsc_multiplier)
    elif state.is_yellow:
        raw *= float(cfg.yellow_multiplier)

    raw *= max(0.0, float(scenario_multiplier))
    return float(np.clip(raw, 0.0, float(cfg.max_loss_seconds)))


def traffic_diagnostics(
    state: StrategyState,
    *,
    action: Optional[StrategyAction] = None,
    config: TrafficModelConfig | None = None,
    scenario_multiplier: float = 1.0,
    scenario_overtake_propensity: Optional[float] = None,
) -> dict[str, object]:
    return {
        "traffic_loss_seconds": estimate_traffic_loss_seconds(
            state,
            action=action,
            config=config,
            scenario_multiplier=scenario_multiplier,
            scenario_overtake_propensity=scenario_overtake_propensity,
        ),
        "overtake_difficulty_proxy": overtake_difficulty_proxy(
            state,
            scenario_overtake_propensity=scenario_overtake_propensity,
        ),
        "scenario_multiplier": float(scenario_multiplier),
    }


__all__ = [
    "TrafficModelConfig",
    "estimate_traffic_loss_seconds",
    "overtake_difficulty_proxy",
    "traffic_diagnostics",
]
