"""Pit-loss priors for the live-race simulator and simulator-backed planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.traffic import overtake_difficulty_proxy


@dataclass(frozen=True)
class PitLossConfig:
    green_loss_seconds: float = 21.0
    yellow_loss_seconds: float = 15.5
    sc_vsc_loss_seconds: float = 11.0
    red_loss_seconds: float = 0.0
    monaco_multiplier: float = 1.12
    high_overtake_multiplier: float = 0.94
    traffic_loss_sensitivity: float = 0.45
    position_rejoin_sensitivity: float = 0.05
    min_loss_seconds: float = 6.0
    max_loss_seconds: float = 35.0


def _finite(value: object, default: float) -> float:
    try:
        if value is None or not np.isfinite(float(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _circuit_key(state: StrategyState) -> str:
    metadata = state.metadata if isinstance(state.metadata, Mapping) else {}
    for key in ("circuit_id", "circuit_key", "circuit_name", "location", "event_name"):
        value = metadata.get(key)
        if value is not None:
            return str(value).strip().lower().replace(" ", "_")
    return ""


def estimate_pit_loss_seconds(
    state: StrategyState,
    *,
    config: PitLossConfig | None = None,
    traffic_loss_seconds: float = 0.0,
    scenario_multiplier: float = 1.0,
    scenario_offset_seconds: float = 0.0,
    scenario_overtake_propensity: Optional[float] = None,
) -> float:
    """Estimate pit-lane race-time loss under the current track-status proxy."""

    explicit = _finite(state.pit_loss_estimate_seconds, float("nan"))
    if np.isfinite(explicit):
        base = explicit
    else:
        cfg = config or PitLossConfig()
        if state.is_red:
            base = float(cfg.red_loss_seconds)
        elif state.is_sc_vsc:
            base = float(cfg.sc_vsc_loss_seconds)
        elif state.is_yellow:
            base = float(cfg.yellow_loss_seconds)
        else:
            base = float(cfg.green_loss_seconds)

    cfg = config or PitLossConfig()
    circuit = _circuit_key(state)
    difficulty = overtake_difficulty_proxy(state, scenario_overtake_propensity=scenario_overtake_propensity)
    circuit_multiplier = 1.0
    if "monaco" in circuit or "monte_carlo" in circuit:
        circuit_multiplier *= float(cfg.monaco_multiplier)
    elif difficulty < 0.35:
        circuit_multiplier *= float(cfg.high_overtake_multiplier)

    position = _finite(state.position, 10.0)
    position_component = float(cfg.position_rejoin_sensitivity) * float(np.clip(position - 1.0, 0.0, 19.0))
    traffic_component = float(cfg.traffic_loss_sensitivity) * max(0.0, float(traffic_loss_seconds))

    loss = (
        (float(base) * circuit_multiplier * max(0.0, float(scenario_multiplier)))
        + traffic_component
        + position_component
        + float(scenario_offset_seconds)
    )
    return float(np.clip(loss, float(cfg.min_loss_seconds), float(cfg.max_loss_seconds)))


def pit_loss_diagnostics(
    state: StrategyState,
    *,
    config: PitLossConfig | None = None,
    traffic_loss_seconds: float = 0.0,
    scenario_multiplier: float = 1.0,
    scenario_offset_seconds: float = 0.0,
    scenario_overtake_propensity: Optional[float] = None,
) -> dict[str, object]:
    return {
        "pit_loss_seconds": estimate_pit_loss_seconds(
            state,
            config=config,
            traffic_loss_seconds=traffic_loss_seconds,
            scenario_multiplier=scenario_multiplier,
            scenario_offset_seconds=scenario_offset_seconds,
            scenario_overtake_propensity=scenario_overtake_propensity,
        ),
        "circuit_key": _circuit_key(state),
        "overtake_difficulty_proxy": overtake_difficulty_proxy(
            state,
            scenario_overtake_propensity=scenario_overtake_propensity,
        ),
        "scenario_multiplier": float(scenario_multiplier),
        "scenario_offset_seconds": float(scenario_offset_seconds),
    }


__all__ = ["PitLossConfig", "estimate_pit_loss_seconds", "pit_loss_diagnostics"]
