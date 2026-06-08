"""Deterministic DP/MPC-style planner for live F1 race strategy.

This is the Phase 3 non-DL challenger.  It is intentionally not branded as the
future calibrated simulator: transitions are a practical deterministic
approximation from the current live state, compound priors, track-status flags,
and optional deterministic strategy-adapter scores.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.live_race.action_space import (
    ACTION_PIT_NEXT_LAP,
    ACTION_PIT_NOW,
    ACTION_STAY_OUT,
    DRY_COMPOUNDS,
    ActionMaskConfig,
    StrategyAction,
    build_action_space,
    build_legal_action_mask,
    normalize_compound,
)
from packages.f1.models.live_race.environment import (
    RewardConfig,
    StrategyReward,
    StrategyState,
    StrategyTransition,
)
from packages.f1.models.live_race.state import compound_deg_prior
from packages.f1.models.live_race.strategy import BaselineStrategyPolicyAdapter, StrategyPolicyAdapter


class TransitionModel(Protocol):
    def step(self, state: StrategyState, action: StrategyAction) -> StrategyTransition:
        ...


@dataclass(frozen=True)
class DeterministicTransitionConfig:
    """Parameters for the pre-simulator single-car transition approximation."""

    default_base_lap_seconds: float = 90.0
    default_pit_loss_seconds: float = 21.0
    sc_vsc_pit_loss_seconds: float = 11.0
    yellow_pit_loss_seconds: float = 15.5
    soft_life_laps: float = 16.0
    medium_life_laps: float = 22.0
    hard_life_laps: float = 28.0
    inter_life_laps: float = 14.0
    wet_life_laps: float = 12.0
    soft_lap_delta_seconds: float = -0.35
    medium_lap_delta_seconds: float = 0.0
    hard_lap_delta_seconds: float = 0.28
    inter_lap_delta_seconds: float = 1.25
    wet_lap_delta_seconds: float = 2.20
    conservative_lap_delta_seconds: float = 0.08
    aggressive_lap_delta_seconds: float = -0.14
    conservative_deg_multiplier: float = 0.86
    aggressive_deg_multiplier: float = 1.18
    tyre_cliff_quadratic_seconds: float = 0.035
    fuel_burn_lap_gain_seconds: float = 0.018
    green_pit_track_position_penalty_seconds: float = 3.0
    yellow_pit_track_position_penalty_seconds: float = 1.2
    sc_vsc_pit_track_position_penalty_seconds: float = 0.35
    mandatory_dry_change_finish_penalty_seconds: float = 35.0
    pit_next_commitment_penalty_seconds: float = 0.12


@dataclass(frozen=True)
class PlannerConfig:
    horizon_laps: int = 12
    discount: float = 1.0
    action_mask: ActionMaskConfig = ActionMaskConfig()
    transition: DeterministicTransitionConfig = DeterministicTransitionConfig()
    reward: RewardConfig = RewardConfig(position_gain_weight=0.0)
    strategy_score_weight: float = 0.05
    include_pit_next_lap: bool = True
    compounds: tuple[str, ...] = DRY_COMPOUNDS


@dataclass(frozen=True)
class PlannerResult:
    action: StrategyAction
    value: float
    action_values: dict[str, float]
    plan: tuple[StrategyAction, ...]
    diagnostics: dict[str, object]


def compound_service_life(compound: object, cfg: DeterministicTransitionConfig | None = None) -> float:
    config = cfg or DeterministicTransitionConfig()
    normalized = normalize_compound(compound)
    if normalized == "SOFT":
        return float(config.soft_life_laps)
    if normalized == "MEDIUM":
        return float(config.medium_life_laps)
    if normalized == "HARD":
        return float(config.hard_life_laps)
    if normalized == "INTER":
        return float(config.inter_life_laps)
    if normalized == "WET":
        return float(config.wet_life_laps)
    return float(config.medium_life_laps)


def compound_lap_delta(compound: object, cfg: DeterministicTransitionConfig | None = None) -> float:
    config = cfg or DeterministicTransitionConfig()
    normalized = normalize_compound(compound)
    if normalized == "SOFT":
        return float(config.soft_lap_delta_seconds)
    if normalized == "MEDIUM":
        return float(config.medium_lap_delta_seconds)
    if normalized == "HARD":
        return float(config.hard_lap_delta_seconds)
    if normalized == "INTER":
        return float(config.inter_lap_delta_seconds)
    if normalized == "WET":
        return float(config.wet_lap_delta_seconds)
    return 0.0


def _mode_lap_delta(action: StrategyAction, cfg: DeterministicTransitionConfig) -> float:
    if action.mode == "aggressive":
        return float(cfg.aggressive_lap_delta_seconds)
    return float(cfg.conservative_lap_delta_seconds)


def _mode_deg_multiplier(action: StrategyAction, cfg: DeterministicTransitionConfig) -> float:
    if action.mode == "aggressive":
        return float(cfg.aggressive_deg_multiplier)
    return float(cfg.conservative_deg_multiplier)


def _finite(value: Optional[float], default: float) -> float:
    try:
        if value is None or not np.isfinite(float(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _circuit_tyre_multiplier(state: StrategyState) -> float:
    tyre = _finite(state.circuit_tyre_degradation, 0.55)
    return float(np.clip(0.75 + (0.65 * tyre), 0.75, 1.45))


def _overtaking_difficulty(state: StrategyState) -> float:
    if state.circuit_overtaking_difficulty is not None:
        return float(np.clip(state.circuit_overtaking_difficulty, 0.0, 1.0))
    if state.track_overtake_propensity is not None:
        return float(np.clip(1.0 - state.track_overtake_propensity, 0.0, 1.0))
    return 0.55


def _pit_loss_seconds(state: StrategyState, cfg: DeterministicTransitionConfig) -> float:
    explicit = _finite(state.pit_loss_estimate_seconds, float("nan"))
    if np.isfinite(explicit):
        return max(1.0, explicit)
    if state.is_sc_vsc:
        return float(cfg.sc_vsc_pit_loss_seconds)
    if state.is_yellow:
        return float(cfg.yellow_pit_loss_seconds)
    return float(cfg.default_pit_loss_seconds)


def _pit_track_position_penalty(state: StrategyState, cfg: DeterministicTransitionConfig) -> float:
    difficulty = _overtaking_difficulty(state)
    if state.is_sc_vsc:
        base = float(cfg.sc_vsc_pit_track_position_penalty_seconds)
    elif state.is_yellow:
        base = float(cfg.yellow_pit_track_position_penalty_seconds)
    else:
        base = float(cfg.green_pit_track_position_penalty_seconds)
    return float(base * difficulty)


def _base_clean_lap_seconds(state: StrategyState, cfg: DeterministicTransitionConfig) -> float:
    if state.next_lap_mean is not None and np.isfinite(float(state.next_lap_mean)):
        return float(state.next_lap_mean)
    return float(cfg.default_base_lap_seconds + _finite(state.pace_penalty_mean, 0.0))


def estimate_clean_lap_seconds(
    state: StrategyState,
    action: StrategyAction,
    *,
    config: DeterministicTransitionConfig | None = None,
) -> float:
    """Estimate one green-flag lap before pit-lane loss is applied."""

    cfg = config or DeterministicTransitionConfig()
    compound = action.compound if action.action_type == ACTION_PIT_NOW else state.compound
    compound = normalize_compound(compound)
    tyre_age = 0 if action.action_type == ACTION_PIT_NOW else int(state.tyre_age)
    life = compound_service_life(compound, cfg)
    deg_rate = max(0.0, _finite(state.deg_rate_mean, compound_deg_prior(compound)))
    deg_rate *= _circuit_tyre_multiplier(state) * _mode_deg_multiplier(action, cfg)
    age_penalty = deg_rate * float(max(0, tyre_age))
    cliff_start = 0.72 * max(1.0, life)
    cliff_age = max(0.0, float(tyre_age) - cliff_start)
    cliff_penalty = float(cfg.tyre_cliff_quadratic_seconds) * (cliff_age**2)
    fuel_gain = float(cfg.fuel_burn_lap_gain_seconds) * float(max(0, state.lap_number))

    return float(
        _base_clean_lap_seconds(state, cfg)
        + compound_lap_delta(compound, cfg)
        + _mode_lap_delta(action, cfg)
        + age_penalty
        + cliff_penalty
        - fuel_gain
    )


class DeterministicStrategyTransitionModel:
    """Single-car deterministic transition approximation used before Phase 5."""

    limitations = (
        "single_car_no_traffic_response",
        "deterministic_no_random_sc_vsc_sampling",
        "pit_loss_and_tyre_degradation_are_hand_tuned_priors",
        "position_reward_is_proxy_not_counterfactual_race_order",
    )

    def __init__(
        self,
        *,
        config: DeterministicTransitionConfig | None = None,
        action_space: Optional[Sequence[StrategyAction]] = None,
        action_mask_config: ActionMaskConfig | None = None,
        reward_config: RewardConfig | None = None,
    ) -> None:
        self.config = config or DeterministicTransitionConfig()
        self.action_space = tuple(action_space or build_action_space(compounds=DRY_COMPOUNDS, include_pit_next_lap=True))
        self.action_mask_config = action_mask_config or ActionMaskConfig()
        self.reward_config = reward_config or RewardConfig(position_gain_weight=0.0)

    def step(self, state: StrategyState, action: StrategyAction) -> StrategyTransition:
        legal_mask = build_legal_action_mask(
            state,
            action_space=self.action_space,
            config=self.action_mask_config,
        )
        if not legal_mask.is_legal(action):
            reward = StrategyReward(
                value=-float(self.reward_config.illegal_action_penalty),
                components={"illegal_action_penalty": float(self.reward_config.illegal_action_penalty)},
                note=legal_mask.reason_for(action),
            )
            return StrategyTransition(
                state_t=state,
                action_t=action,
                reward_t=reward,
                state_t1=state,
                done=bool(state.remaining_laps == 0),
                legal_action_mask=legal_mask,
                metadata={
                    "transition_model": "deterministic_pre_sim_v1",
                    "limitations": self.limitations,
                    "invalid_action_reason": legal_mask.reason_for(action),
                },
            )

        elapsed = estimate_clean_lap_seconds(state, action, config=self.config)
        metadata = dict(state.metadata)
        next_compound = state.compound
        next_tyre_age = int(state.tyre_age) + 1
        next_stint = int(state.stint_id)
        used_compounds = tuple(state.used_compounds)
        position_proxy = state.position

        if action.action_type == ACTION_PIT_NOW:
            next_compound = normalize_compound(action.compound)
            elapsed += _pit_loss_seconds(state, self.config)
            elapsed += _pit_track_position_penalty(state, self.config)
            next_tyre_age = 0
            next_stint += 1
            if next_compound not in used_compounds:
                used_compounds = (*used_compounds, next_compound)
            if position_proxy is not None and not state.is_sc_vsc:
                position_proxy = int(position_proxy) + max(1, int(round(2.0 * _overtaking_difficulty(state))))
            metadata.pop("forced_pit_next_compound", None)
            metadata.pop("forced_pit_next_mode", None)
        elif action.action_type == ACTION_PIT_NEXT_LAP:
            metadata["forced_pit_next_compound"] = normalize_compound(action.compound)
            metadata["forced_pit_next_mode"] = action.mode
            elapsed += float(self.config.pit_next_commitment_penalty_seconds)
        else:
            metadata.pop("forced_pit_next_compound", None)
            metadata.pop("forced_pit_next_mode", None)

        next_lap = int(state.lap_number) + 1
        remaining = None if state.remaining_laps is None else max(0, int(state.remaining_laps) - 1)
        race_time = None
        if state.race_time_seconds is not None:
            race_time = float(state.race_time_seconds) + max(0.1, float(elapsed))

        next_state = replace(
            state,
            lap_number=next_lap,
            remaining_laps=remaining,
            stint_id=next_stint,
            compound=next_compound,
            tyre_age=next_tyre_age,
            used_compounds=used_compounds,
            race_time_seconds=race_time,
            position=position_proxy,
            next_lap_mean=float(elapsed),
            metadata={
                **metadata,
                "available_through_lap": next_lap,
                "transition_model": "deterministic_pre_sim_v1",
            },
        )

        components = {
            "estimated_lap_seconds": float(elapsed),
            "pit_loss_seconds": float(_pit_loss_seconds(state, self.config)) if action.action_type == ACTION_PIT_NOW else 0.0,
            "pit_track_position_penalty_seconds": float(_pit_track_position_penalty(state, self.config))
            if action.action_type == ACTION_PIT_NOW
            else 0.0,
            "illegal_action_penalty": 0.0,
        }
        reward = StrategyReward(
            value=-float(elapsed),
            components=components,
            note="deterministic_pre_sim_transition",
        )
        done = bool(remaining == 0)
        return StrategyTransition(
            state_t=state,
            action_t=action,
            reward_t=reward,
            state_t1=next_state,
            done=done,
            legal_action_mask=legal_mask,
            metadata={
                "transition_model": "deterministic_pre_sim_v1",
                "limitations": self.limitations,
            },
        )


class DeterministicStrategyPlanner:
    """Finite-horizon dynamic-programming planner for the live strategy state."""

    def __init__(
        self,
        *,
        config: PlannerConfig | None = None,
        transition_model: Optional[TransitionModel] = None,
        strategy_adapter: Optional[StrategyPolicyAdapter] = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.action_space = build_action_space(
            compounds=self.config.compounds,
            include_pit_next_lap=bool(self.config.include_pit_next_lap),
        )
        self.strategy_adapter = strategy_adapter or BaselineStrategyPolicyAdapter()
        self.transition_model = transition_model or DeterministicStrategyTransitionModel(
            config=self.config.transition,
            action_space=self.action_space,
            action_mask_config=self.config.action_mask,
            reward_config=self.config.reward,
        )
        self._memo: dict[tuple[object, ...], tuple[float, tuple[StrategyAction, ...]]] = {}

    @property
    def limitations(self) -> tuple[str, ...]:
        return tuple(getattr(self.transition_model, "limitations", ()))

    def plan(self, state: StrategyState | Mapping[str, object]) -> PlannerResult:
        start = state if isinstance(state, StrategyState) else StrategyState.from_mapping(state)
        self._memo = {}
        legal_mask = build_legal_action_mask(
            start,
            action_space=self.action_space,
            config=self.config.action_mask,
        )
        action_values: dict[str, float] = {}
        best_action: Optional[StrategyAction] = None
        best_value = -float("inf")
        best_plan: tuple[StrategyAction, ...] = ()

        depth = min(max(1, int(self.config.horizon_laps)), max(1, int(start.remaining_laps or self.config.horizon_laps)))
        for action in legal_mask.legal_actions:
            transition = self.transition_model.step(start, action)
            future_value, future_plan = self._search(transition.state_t1, depth - 1)
            value = float(transition.reward_t.value) + self._adapter_bonus(start, action) + (
                float(self.config.discount) * future_value
            )
            action_values[action.key] = float(value)
            if value > best_value:
                best_value = float(value)
                best_action = action
                best_plan = (action, *future_plan)

        if best_action is None:
            fallback = StrategyAction(ACTION_STAY_OUT)
            best_action = fallback
            best_value = -float(self.config.reward.illegal_action_penalty)
            best_plan = (fallback,)

        return PlannerResult(
            action=best_action,
            value=float(best_value),
            action_values=action_values,
            plan=best_plan,
            diagnostics={
                "planner": "deterministic_dp_pre_sim_v1",
                "horizon_laps": int(depth),
                "legal_action_count": int(legal_mask.legal_count),
                "limitations": self.limitations,
                "used_strategy_adapter_scores": bool(self.strategy_adapter is not None and self.config.strategy_score_weight != 0.0),
            },
        )

    def value_action(self, state: StrategyState | Mapping[str, object], action: StrategyAction) -> float:
        start = state if isinstance(state, StrategyState) else StrategyState.from_mapping(state)
        depth = min(max(1, int(self.config.horizon_laps)), max(1, int(start.remaining_laps or self.config.horizon_laps)))
        transition = self.transition_model.step(start, action)
        future_value, _ = self._search(transition.state_t1, depth - 1)
        return float(transition.reward_t.value) + self._adapter_bonus(start, action) + (
            float(self.config.discount) * future_value
        )

    def _search(self, state: StrategyState, depth: int) -> tuple[float, tuple[StrategyAction, ...]]:
        if depth <= 0 or bool(state.remaining_laps == 0):
            return self._terminal_value(state), ()
        key = self._memo_key(state, depth)
        cached = self._memo.get(key)
        if cached is not None:
            return cached

        legal_mask = build_legal_action_mask(
            state,
            action_space=self.action_space,
            config=self.config.action_mask,
        )
        best_value = -float("inf")
        best_plan: tuple[StrategyAction, ...] = ()
        for action in legal_mask.legal_actions:
            transition = self.transition_model.step(state, action)
            future_value, future_plan = self._search(transition.state_t1, depth - 1)
            value = float(transition.reward_t.value) + self._adapter_bonus(state, action) + (
                float(self.config.discount) * future_value
            )
            if value > best_value:
                best_value = float(value)
                best_plan = (action, *future_plan)
        if not np.isfinite(best_value):
            best_value = -float(self.config.reward.illegal_action_penalty)
        self._memo[key] = (float(best_value), best_plan)
        return self._memo[key]

    def _memo_key(self, state: StrategyState, depth: int) -> tuple[object, ...]:
        forced = state.metadata.get("forced_pit_next_compound")
        return (
            int(depth),
            int(state.lap_number),
            int(state.remaining_laps or 0),
            int(state.stint_id),
            state.compound,
            int(state.tyre_age),
            tuple(sorted(state.used_compounds)),
            bool(state.is_sc_vsc),
            bool(state.is_yellow),
            bool(state.is_red),
            round(_finite(state.deg_rate_mean, 0.04), 4),
            round(_finite(state.circuit_overtaking_difficulty, 0.55), 3),
            round(_finite(state.circuit_tyre_degradation, 0.55), 3),
            forced,
        )

    def _terminal_value(self, state: StrategyState) -> float:
        used_dry = {compound for compound in state.used_compounds if compound in DRY_COMPOUNDS}
        current_dry = state.compound in DRY_COMPOUNDS
        if current_dry and len(used_dry) < 2 and bool(self.config.action_mask.enforce_dry_mandatory_change):
            return -float(self.config.transition.mandatory_dry_change_finish_penalty_seconds)
        return 0.0

    def _adapter_bonus(self, state: StrategyState, action: StrategyAction) -> float:
        weight = float(self.config.strategy_score_weight)
        if weight == 0.0 or self.strategy_adapter is None:
            return 0.0
        try:
            row = pd.DataFrame([_state_to_adapter_row(state)])
            scores = self.strategy_adapter.evaluate_actions(row)
            if scores.empty:
                return 0.0
            first = scores.iloc[0]
            if action.action_type == ACTION_STAY_OUT:
                score = float(first.get("score_stay_out", 0.0))
            elif action.action_type == ACTION_PIT_NEXT_LAP:
                score = float(first.get("score_pit_next_lap", 0.0))
            else:
                score = float(first.get("score_pit_now", 0.0))
            return float(weight * score)
        except Exception:
            return 0.0


def _state_to_adapter_row(state: StrategyState) -> dict[str, object]:
    return {
        "driver_id": state.driver_id,
        "compound": state.compound,
        "tyre_age": state.tyre_age,
        "stint_id": state.stint_id,
        "deg_rate_mean": state.deg_rate_mean,
        "pace_penalty_mean": state.pace_penalty_mean,
        "track_status": state.track_status,
        "lap_last": state.lap_number,
        "race_total_laps": state.total_laps,
        "remaining_laps": state.remaining_laps,
        "pit_loss_estimate_seconds": state.pit_loss_estimate_seconds,
        "next_lap_mean": state.next_lap_mean,
        "is_red": state.is_red,
        "is_sc_vsc": state.is_sc_vsc,
        "is_yellow": state.is_yellow,
        "is_greenish": state.is_greenish,
    }


__all__ = [
    "DeterministicStrategyPlanner",
    "DeterministicStrategyTransitionModel",
    "DeterministicTransitionConfig",
    "PlannerConfig",
    "PlannerResult",
    "TransitionModel",
    "compound_lap_delta",
    "compound_service_life",
    "estimate_clean_lap_seconds",
]
