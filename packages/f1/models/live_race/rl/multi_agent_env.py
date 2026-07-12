"""Deterministic multi-agent live-race strategy environment.

Phase 8 deliberately stays dependency-light.  The environment below is not a
full neural simulator; it is a simultaneous 4-8 car game around the existing
``StrategyState`` and ``StrategyAction`` contracts.  It is useful for testing
traffic, undercut response, and synchronized-pit failure modes before any heavy
RL stack is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Callable, Mapping, Optional, Protocol, Sequence

import numpy as np

from packages.f1.models.live_race.action_space import (
    ACTION_PIT_NOW,
    ACTION_STAY_OUT,
    ActionMaskConfig,
    LegalActionMask,
    StrategyAction,
    build_action_space,
    build_legal_action_mask,
    normalize_compound,
)
from packages.f1.models.live_race.environment import StrategyReward, StrategyState
from packages.f1.models.live_race.pit_loss import PitLossConfig, estimate_pit_loss_seconds
from packages.f1.models.live_race.rl.replay_buffer import StrategyActionIndex
from packages.f1.models.live_race.state import compound_deg_prior
from packages.f1.models.live_race.traffic import TrafficModelConfig, overtake_difficulty_proxy


class MultiAgentPolicy(Protocol):
    def select_action(self, state: StrategyState) -> StrategyAction:
        ...


@dataclass(frozen=True)
class MultiAgentRaceConfig:
    """Configuration for a simplified 4-8 car simultaneous strategy game."""

    model_id: str = "multi_agent_live_race_env_v1"
    seed: int = 7
    min_cars: int = 4
    max_cars: int = 8
    base_lap_seconds: float = 90.0
    fuel_burn_lap_gain_seconds: float = 0.025
    tyre_degradation_multiplier: float = 2.8
    tyre_cliff_start_age: float = 24.0
    tyre_cliff_quadratic_seconds: float = 0.030
    traffic_gap_seconds: float = 2.2
    traffic_loss_seconds: float = 1.35
    dirty_air_loss_seconds: float = 0.35
    rejoin_traffic_loss_seconds: float = 0.90
    undercut_pressure_seconds: float = 1.25
    synchronized_pit_penalty_seconds: float = 5.0
    synchronized_pit_threshold_fraction: float = 0.34
    recent_pit_cooldown_laps: int = 1
    position_reward_weight: float = 1.5
    illegal_action_penalty: float = 250.0
    min_lap_seconds: float = 45.0
    max_lap_seconds: float = 220.0
    noise_std_seconds: float = 0.025
    compounds: tuple[str, ...] = ("MEDIUM", "HARD")
    modes: tuple[str, ...] = ("conservative",)
    include_pit_next_lap: bool = False
    compound_effects_seconds: Mapping[str, float] = field(
        default_factory=lambda: {
            "SOFT": -0.25,
            "MEDIUM": 0.0,
            "HARD": 0.18,
            "INTER": 1.10,
            "WET": 2.20,
        }
    )
    action_mask: ActionMaskConfig = field(
        default_factory=lambda: ActionMaskConfig(
            allow_pit_next_lap=False,
            allow_same_compound_pit=False,
            default_available_compounds=("MEDIUM", "HARD"),
            mandatory_stop_window_laps=2,
        )
    )
    traffic: TrafficModelConfig = field(default_factory=TrafficModelConfig)
    pit_loss: PitLossConfig = field(default_factory=PitLossConfig)


@dataclass(frozen=True)
class MultiAgentRaceState:
    """Aligned per-car state for one simultaneous strategy decision."""

    cars: tuple[StrategyState, ...]
    lap_number: int
    scenario_id: str = "traffic_heavy_synthetic"
    seed: int = 7
    pit_counts_by_lap: Mapping[int, int] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cars = tuple(self.cars)
        if not cars:
            raise ValueError("multi-agent race state requires at least one car")
        driver_ids = [str(car.driver_id) for car in cars]
        if len(set(driver_ids)) != len(driver_ids):
            raise ValueError("driver_id values must be unique in a multi-agent state")
        object.__setattr__(self, "cars", cars)
        object.__setattr__(self, "lap_number", int(self.lap_number))
        object.__setattr__(self, "pit_counts_by_lap", {int(k): int(v) for k, v in dict(self.pit_counts_by_lap).items()})
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def car_count(self) -> int:
        return len(self.cars)

    @property
    def driver_ids(self) -> tuple[str, ...]:
        return tuple(str(car.driver_id) for car in self.cars)

    @property
    def done(self) -> bool:
        return all((car.remaining_laps or 0) <= 0 for car in self.cars)

    def car_by_driver(self) -> dict[str, StrategyState]:
        return {str(car.driver_id): car for car in self.cars}

    def fingerprint(self) -> str:
        return _stable_digest(
            {
                "lap_number": self.lap_number,
                "scenario_id": self.scenario_id,
                "seed": self.seed,
                "pit_counts_by_lap": dict(sorted((int(k), int(v)) for k, v in self.pit_counts_by_lap.items())),
                "cars": [_car_payload(car) for car in self.cars],
            }
        )


@dataclass(frozen=True)
class MultiAgentStepResult:
    """One simultaneous multi-car transition."""

    state_t: MultiAgentRaceState
    actions: tuple[StrategyAction, ...]
    rewards: tuple[StrategyReward, ...]
    state_t1: MultiAgentRaceState
    done: bool
    legal_action_masks: tuple[LegalActionMask, ...]
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def action_by_driver(self) -> dict[str, StrategyAction]:
        return {driver_id: action for driver_id, action in zip(self.state_t.driver_ids, self.actions)}

    @property
    def reward_by_driver(self) -> dict[str, float]:
        return {driver_id: float(reward.value) for driver_id, reward in zip(self.state_t.driver_ids, self.rewards)}

    def to_payload(self) -> dict[str, object]:
        return {
            "state_t": self.state_t.fingerprint(),
            "state_t1": self.state_t1.fingerprint(),
            "actions": [action.to_payload() for action in self.actions],
            "rewards": [reward.to_payload() for reward in self.rewards],
            "done": bool(self.done),
            "diagnostics": _json_safe(dict(self.diagnostics)),
        }

    def fingerprint(self) -> str:
        return _stable_digest(self.to_payload())


@dataclass(frozen=True)
class MultiAgentRolloutResult:
    """Rollout payload for self-play and seeded replay tests."""

    initial_state: MultiAgentRaceState
    transitions: tuple[MultiAgentStepResult, ...]
    policy_name: str = "policy"
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    @property
    def final_state(self) -> MultiAgentRaceState:
        return self.transitions[-1].state_t1 if self.transitions else self.initial_state

    @property
    def action_keys_by_driver(self) -> dict[str, list[str]]:
        out = {driver_id: [] for driver_id in self.initial_state.driver_ids}
        for transition in self.transitions:
            for driver_id, action in zip(transition.state_t.driver_ids, transition.actions):
                out[str(driver_id)].append(action.key)
        return out

    @property
    def pit_lap_histogram(self) -> dict[int, int]:
        histogram: dict[int, int] = {}
        for transition in self.transitions:
            pit_count = sum(1 for action in transition.actions if action.action_type == ACTION_PIT_NOW)
            if pit_count:
                histogram[int(transition.state_t.lap_number)] = int(pit_count)
        return histogram

    @property
    def max_same_lap_pit_count(self) -> int:
        histogram = self.pit_lap_histogram
        return int(max(histogram.values())) if histogram else 0

    @property
    def illegal_action_count(self) -> int:
        return int(sum(int(transition.diagnostics.get("illegal_action_count", 0)) for transition in self.transitions))

    @property
    def synchronized_pit_rate(self) -> float:
        if not self.transitions:
            return 0.0
        sync_steps = sum(1 for transition in self.transitions if bool(transition.diagnostics.get("synchronized_pit_pattern")))
        return float(sync_steps / len(self.transitions))

    @property
    def coordination_penalty_seconds(self) -> float:
        return float(sum(float(transition.diagnostics.get("coordination_penalty_seconds", 0.0)) for transition in self.transitions))

    @property
    def team_time_seconds(self) -> float:
        return float(sum(float(car.race_time_seconds or 0.0) for car in self.final_state.cars))

    @property
    def team_reward(self) -> float:
        return float(sum(float(reward.value) for transition in self.transitions for reward in transition.rewards))

    def metrics(self) -> dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "transitions": int(len(self.transitions)),
            "team_time_seconds": float(self.team_time_seconds),
            "team_reward": float(self.team_reward),
            "illegal_action_count": int(self.illegal_action_count),
            "pit_lap_histogram": dict(sorted(self.pit_lap_histogram.items())),
            "max_same_lap_pit_count": int(self.max_same_lap_pit_count),
            "synchronized_pit_rate": float(self.synchronized_pit_rate),
            "coordination_penalty_seconds": float(self.coordination_penalty_seconds),
            "final_order": [car.driver_id for car in sorted(self.final_state.cars, key=_race_time)],
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> str:
        return _stable_digest(
            {
                "initial_state": self.initial_state.fingerprint(),
                "policy_name": self.policy_name,
                "transitions": [transition.fingerprint() for transition in self.transitions],
            }
        )


class MultiAgentLiveRaceEnv:
    """Seeded simultaneous multi-car environment for traffic-heavy strategy tests."""

    limitations = (
        "simplified_four_to_eight_car_game_not_full_grid",
        "central_order_and_traffic_are_proxy_models",
        "pit_loss_and_degradation_are_hand_tuned_until_calibrated",
        "seeded_replay_is_deterministic_for_policy_comparison",
    )

    def __init__(
        self,
        *,
        config: MultiAgentRaceConfig | None = None,
        action_space: Optional[Sequence[StrategyAction]] = None,
    ) -> None:
        self.config = config or MultiAgentRaceConfig()
        self.action_space = tuple(
            action_space
            or build_action_space(
                compounds=self.config.compounds,
                modes=self.config.modes,
                include_pit_next_lap=self.config.include_pit_next_lap,
            )
        )
        self.action_index = StrategyActionIndex.from_action_space(self.action_space)
        self.model_id = self.config.model_id

    def validate_start_state(self, state: MultiAgentRaceState) -> None:
        car_count = int(state.car_count)
        if car_count < int(self.config.min_cars) or car_count > int(self.config.max_cars):
            raise ValueError(
                f"Phase 8 starts with {self.config.min_cars}-{self.config.max_cars} cars; got {car_count}"
            )
        laps = {int(car.lap_number) for car in state.cars}
        if len(laps) != 1:
            raise ValueError("all cars must be aligned to the same live lap")

    def reset(self, state: Optional[MultiAgentRaceState] = None) -> MultiAgentRaceState:
        initial = state or build_traffic_heavy_scenario(seed=int(self.config.seed))
        self.validate_start_state(initial)
        return self._refresh_public_metadata(initial)

    def legal_masks(self, state: MultiAgentRaceState) -> tuple[LegalActionMask, ...]:
        return tuple(
            build_legal_action_mask(car, action_space=self.action_space, config=self.config.action_mask)
            for car in state.cars
        )

    def step(
        self,
        state: MultiAgentRaceState,
        actions: Mapping[str, StrategyAction | str] | Sequence[StrategyAction | str],
    ) -> MultiAgentStepResult:
        self.validate_start_state(state)
        state = self._refresh_public_metadata(state)
        proposed = self._normalise_actions(state, actions)
        legal_masks = self.legal_masks(state)
        applied, illegal_count, illegal_penalties = self._apply_legal_fallbacks(proposed, legal_masks)

        pit_count = sum(1 for action in applied if action.action_type == ACTION_PIT_NOW)
        sync_threshold = self.sync_pit_threshold(state.car_count)
        sync_excess = max(0, pit_count - sync_threshold)
        sync_penalty_each = (
            float(self.config.synchronized_pit_penalty_seconds) * float(sync_excess)
            if sync_excess > 0
            else 0.0
        )
        old_order = _order_indices_by_race_time(state.cars)
        old_position_by_index = {idx: pos + 1 for pos, idx in enumerate(old_order)}
        pit_by_index = {idx for idx, action in enumerate(applied) if action.action_type == ACTION_PIT_NOW}

        next_cars: list[StrategyState] = []
        rewards: list[StrategyReward] = []
        step_components: list[dict[str, float]] = []

        for idx, (car, action) in enumerate(zip(state.cars, applied)):
            components = self._step_components(
                state,
                car_index=idx,
                action=action,
                old_position=int(old_position_by_index[idx]),
                pit_by_index=pit_by_index,
                synchronized_pit_penalty=sync_penalty_each,
            )
            elapsed = float(components["race_time_delta_seconds"])
            next_car = self._next_car_state(car, action, elapsed=elapsed, components=components)
            next_cars.append(next_car)
            step_components.append(components)

        provisional_state = MultiAgentRaceState(
            cars=tuple(next_cars),
            lap_number=int(state.lap_number) + 1,
            scenario_id=state.scenario_id,
            seed=state.seed,
            pit_counts_by_lap={**dict(state.pit_counts_by_lap), int(state.lap_number): int(pit_count)},
            metadata={
                **dict(state.metadata),
                "last_action_keys": tuple(action.key for action in applied),
                "last_pit_count": int(pit_count),
            },
        )
        next_state = self._refresh_public_metadata(provisional_state)
        new_position_by_driver = {str(car.driver_id): int(car.position or idx + 1) for idx, car in enumerate(next_state.cars)}

        for idx, (car, next_car, action, components) in enumerate(zip(state.cars, next_state.cars, applied, step_components)):
            old_position = int(old_position_by_index[idx])
            new_position = int(new_position_by_driver.get(str(car.driver_id), old_position))
            position_gain = float(old_position - new_position)
            illegal_penalty = float(illegal_penalties[idx])
            value = (
                -float(components["race_time_delta_seconds"])
                + (float(self.config.position_reward_weight) * position_gain)
                - illegal_penalty
            )
            rewards.append(
                StrategyReward(
                    value=float(value),
                    components={
                        **components,
                        "position_gain": float(position_gain),
                        "illegal_action_penalty": float(illegal_penalty),
                    },
                    note="multi_agent_live_race_transition",
                )
            )

        diagnostics = {
            "transition_model": self.model_id,
            "scenario_id": state.scenario_id,
            "seed": int(self.config.seed),
            "car_count": int(state.car_count),
            "pit_count": int(pit_count),
            "sync_pit_threshold": int(sync_threshold),
            "synchronized_pit_pattern": bool(sync_excess > 0),
            "synchronized_pit_excess": int(sync_excess),
            "coordination_penalty_seconds": float(sync_penalty_each * pit_count),
            "illegal_action_count": int(illegal_count),
            "action_keys": [action.key for action in applied],
            "limitations": self.limitations,
        }
        return MultiAgentStepResult(
            state_t=state,
            actions=tuple(applied),
            rewards=tuple(rewards),
            state_t1=next_state,
            done=bool(next_state.done),
            legal_action_masks=legal_masks,
            diagnostics=diagnostics,
        )

    def rollout(
        self,
        state: MultiAgentRaceState,
        *,
        policy: MultiAgentPolicy | Callable[[StrategyState], StrategyAction] | object,
        max_laps: Optional[int] = None,
        policy_name: Optional[str] = None,
    ) -> MultiAgentRolloutResult:
        current = self.reset(state)
        limit = int(max_laps) if max_laps is not None else int(max(car.remaining_laps or 0 for car in current.cars))
        transitions: list[MultiAgentStepResult] = []
        for _ in range(max(0, limit)):
            actions = self._select_policy_actions(policy, current)
            transition = self.step(current, actions)
            transitions.append(transition)
            current = transition.state_t1
            if transition.done:
                break
        name = policy_name or getattr(policy, "model_id", policy.__class__.__name__)
        return MultiAgentRolloutResult(
            initial_state=state,
            transitions=tuple(transitions),
            policy_name=str(name),
            diagnostics={"transition_model": self.model_id, "limitations": self.limitations},
        )

    def sync_pit_threshold(self, car_count: int) -> int:
        raw = int(np.floor(float(car_count) * float(self.config.synchronized_pit_threshold_fraction)))
        return max(1, raw)

    def _select_policy_actions(self, policy: object, state: MultiAgentRaceState) -> tuple[StrategyAction, ...]:
        if hasattr(policy, "select_actions"):
            raw = policy.select_actions(state)  # type: ignore[attr-defined]
            return self._normalise_actions(state, raw)
        selected: list[StrategyAction] = []
        for car in state.cars:
            if hasattr(policy, "select_action"):
                action = policy.select_action(car)  # type: ignore[attr-defined]
            else:
                action = policy(car)  # type: ignore[operator]
            selected.append(action if isinstance(action, StrategyAction) else StrategyAction.from_key(action))
        return tuple(selected)

    def _normalise_actions(
        self,
        state: MultiAgentRaceState,
        actions: Mapping[str, StrategyAction | str] | Sequence[StrategyAction | str],
    ) -> tuple[StrategyAction, ...]:
        if isinstance(actions, Mapping):
            out = []
            for driver_id in state.driver_ids:
                raw = actions.get(str(driver_id), StrategyAction(ACTION_STAY_OUT))
                out.append(raw if isinstance(raw, StrategyAction) else StrategyAction.from_key(raw))
            return tuple(out)
        out = tuple(action if isinstance(action, StrategyAction) else StrategyAction.from_key(action) for action in actions)
        if len(out) != state.car_count:
            raise ValueError("number of actions must match car count")
        return out

    def _apply_legal_fallbacks(
        self,
        actions: tuple[StrategyAction, ...],
        legal_masks: tuple[LegalActionMask, ...],
    ) -> tuple[tuple[StrategyAction, ...], int, tuple[float, ...]]:
        applied: list[StrategyAction] = []
        penalties: list[float] = []
        illegal_count = 0
        for action, mask in zip(actions, legal_masks):
            if mask.is_legal(action):
                applied.append(action)
                penalties.append(0.0)
                continue
            illegal_count += 1
            fallback = StrategyAction(ACTION_STAY_OUT)
            if not mask.is_legal(fallback):
                legal = mask.legal_actions
                fallback = legal[0] if legal else fallback
            applied.append(fallback)
            penalties.append(float(self.config.illegal_action_penalty))
        return tuple(applied), int(illegal_count), tuple(penalties)

    def _step_components(
        self,
        state: MultiAgentRaceState,
        *,
        car_index: int,
        action: StrategyAction,
        old_position: int,
        pit_by_index: set[int],
        synchronized_pit_penalty: float,
    ) -> dict[str, float]:
        car = state.cars[car_index]
        old_order = _order_indices_by_race_time(state.cars)
        order_pos = old_order.index(car_index)
        ahead_index = old_order[order_pos - 1] if order_pos > 0 else None
        behind_index = old_order[order_pos + 1] if order_pos + 1 < len(old_order) else None
        gap_ahead = _gap_ahead_seconds(state.cars, car_index, old_order)
        gap_behind = _gap_behind_seconds(state.cars, car_index, old_order)
        difficulty = overtake_difficulty_proxy(car)
        traffic_pressure = _traffic_pressure(gap_ahead, difficulty, self.config)

        pit_now = action.action_type == ACTION_PIT_NOW
        cleared_by_car_ahead_pitting = bool(ahead_index is not None and ahead_index in pit_by_index and not pit_now)
        if cleared_by_car_ahead_pitting:
            traffic_loss = traffic_pressure * 0.25
        elif pit_now:
            traffic_loss = float(self.config.rejoin_traffic_loss_seconds) * (0.35 + difficulty)
            traffic_loss += float(self.config.dirty_air_loss_seconds) * float(np.clip((old_position - 1) / 7.0, 0.0, 1.0))
        else:
            traffic_loss = traffic_pressure

        undercut_pressure = 0.0
        if (
            not pit_now
            and behind_index is not None
            and behind_index in pit_by_index
            and np.isfinite(gap_behind)
            and gap_behind <= float(self.config.traffic_gap_seconds) * 1.5
            and int(car.tyre_age) >= 14
        ):
            undercut_pressure = float(self.config.undercut_pressure_seconds) * float(1.0 - min(1.0, gap_behind / 4.0))

        compound = normalize_compound(action.compound) if pit_now else normalize_compound(car.compound)
        if compound == "UNKNOWN":
            compound = "MEDIUM"
        tyre_age_for_lap = 0 if pit_now else int(car.tyre_age)
        lap_time = self._green_lap_seconds(
            car,
            action,
            compound=compound,
            tyre_age=tyre_age_for_lap,
            traffic_loss=float(traffic_loss),
            undercut_pressure=float(undercut_pressure),
        )
        pit_loss = 0.0
        if pit_now:
            pit_loss = estimate_pit_loss_seconds(
                car,
                config=self.config.pit_loss,
                traffic_loss_seconds=float(traffic_loss),
                scenario_overtake_propensity=1.0 - difficulty,
            )
        elapsed = float(lap_time + pit_loss + (synchronized_pit_penalty if pit_now else 0.0))
        elapsed = float(np.clip(elapsed, float(self.config.min_lap_seconds), float(self.config.max_lap_seconds)))
        return {
            "event_lap_baseline": float(_event_lap_baseline(car, self.config)),
            "green_lap_seconds": float(lap_time),
            "race_time_delta_seconds": float(elapsed),
            "traffic_loss": float(traffic_loss),
            "traffic_pressure": float(traffic_pressure),
            "undercut_pressure": float(undercut_pressure),
            "pit_loss": float(pit_loss),
            "synchronized_pit_penalty": float(synchronized_pit_penalty if pit_now else 0.0),
            "overtake_difficulty_proxy": float(difficulty),
        }

    def _green_lap_seconds(
        self,
        car: StrategyState,
        action: StrategyAction,
        *,
        compound: str,
        tyre_age: int,
        traffic_loss: float,
        undercut_pressure: float,
    ) -> float:
        baseline = _event_lap_baseline(car, self.config)
        pace = _finite(car.pace_penalty_mean, 0.0)
        compound_effect = _mapping_float(self.config.compound_effects_seconds, compound, 0.0)
        deg_rate = compound_deg_prior(compound) if action.action_type == ACTION_PIT_NOW else _finite(
            car.deg_rate_mean,
            compound_deg_prior(compound),
        )
        degradation = max(0.0, float(deg_rate)) * float(tyre_age) * float(self.config.tyre_degradation_multiplier)
        cliff_age = max(0.0, float(tyre_age) - float(self.config.tyre_cliff_start_age))
        cliff = float(self.config.tyre_cliff_quadratic_seconds) * (cliff_age**2)
        fuel = -float(self.config.fuel_burn_lap_gain_seconds) * float(max(0, car.lap_number))
        noise = _deterministic_normal(
            seed=int(self.config.seed),
            scenario_id=str(car.metadata.get("multi_agent_scenario_id", "")),
            driver_id=str(car.driver_id),
            lap_number=int(car.lap_number) + 1,
            std=float(self.config.noise_std_seconds),
        )
        lap = baseline + pace + compound_effect + degradation + cliff + fuel + traffic_loss + undercut_pressure + noise
        return float(np.clip(lap, float(self.config.min_lap_seconds), float(self.config.max_lap_seconds)))

    def _next_car_state(
        self,
        car: StrategyState,
        action: StrategyAction,
        *,
        elapsed: float,
        components: Mapping[str, float],
    ) -> StrategyState:
        pit_now = action.action_type == ACTION_PIT_NOW
        next_compound = normalize_compound(action.compound) if pit_now else normalize_compound(car.compound)
        if next_compound == "UNKNOWN":
            next_compound = "MEDIUM"
        used = tuple(car.used_compounds)
        if pit_now and next_compound not in used:
            used = (*used, next_compound)
        race_time = float(car.race_time_seconds or 0.0) + float(elapsed)
        remaining = None if car.remaining_laps is None else max(0, int(car.remaining_laps) - 1)
        metadata = {
            **dict(car.metadata or {}),
            "available_through_lap": int(car.lap_number) + 1,
            "ignored_future_columns": tuple(car.metadata.get("ignored_future_columns", ())),
            "last_multi_agent_components": {str(k): float(v) for k, v in components.items()},
            "last_multi_agent_action_key": action.key,
            "transition_model": self.model_id,
        }
        return replace(
            car,
            lap_number=int(car.lap_number) + 1,
            remaining_laps=remaining,
            stint_id=int(car.stint_id) + (1 if pit_now else 0),
            compound=next_compound,
            tyre_age=0 if pit_now else int(car.tyre_age) + 1,
            used_compounds=used,
            race_time_seconds=float(race_time),
            deg_rate_mean=compound_deg_prior(next_compound) if pit_now else float(car.deg_rate_mean),
            next_lap_mean=float(elapsed),
            metadata=metadata,
        )

    def _refresh_public_metadata(self, state: MultiAgentRaceState) -> MultiAgentRaceState:
        order = _order_indices_by_race_time(state.cars)
        rank_by_index = {idx: pos + 1 for pos, idx in enumerate(order)}
        recent_pits = _recent_pit_count(state.pit_counts_by_lap, state.lap_number, int(self.config.recent_pit_cooldown_laps))
        recent_pit_fraction = float(recent_pits / max(1, state.car_count))
        refreshed: list[StrategyState] = []
        for idx, car in enumerate(state.cars):
            gap_ahead = _gap_ahead_seconds(state.cars, idx, order)
            gap_behind = _gap_behind_seconds(state.cars, idx, order)
            difficulty = overtake_difficulty_proxy(car)
            metadata = {
                **dict(car.metadata or {}),
                "ignored_future_columns": tuple(car.metadata.get("ignored_future_columns", ())),
                "multi_agent_model_id": self.model_id,
                "multi_agent_scenario_id": state.scenario_id,
                "multi_agent_seed": int(state.seed),
                "multi_agent_car_index": int(idx),
                "multi_agent_car_count": int(state.car_count),
                "multi_agent_recent_pit_fraction": float(recent_pit_fraction),
                "multi_agent_recent_pit_count": int(recent_pits),
                "multi_agent_sync_pit_threshold": int(self.sync_pit_threshold(state.car_count)),
                "gap_to_car_ahead_seconds": None if not np.isfinite(gap_ahead) else float(gap_ahead),
                "gap_to_car_behind_seconds": None if not np.isfinite(gap_behind) else float(gap_behind),
                "multi_agent_traffic_pressure": float(_traffic_pressure(gap_ahead, difficulty, self.config)),
            }
            refreshed.append(
                replace(
                    car,
                    position=int(rank_by_index[idx]),
                    gap_to_leader_seconds=float(max(0.0, _race_time(car) - _race_time(state.cars[order[0]]))),
                    metadata=metadata,
                )
            )
        return replace(state, cars=tuple(refreshed), lap_number=int(refreshed[0].lap_number))


def build_traffic_heavy_scenario(
    *,
    car_count: int = 6,
    seed: int = 7,
    scenario_id: str = "traffic_heavy_synthetic",
    start_lap: int = 5,
    total_laps: int = 14,
    base_lap_seconds: float = 90.0,
    starting_tyre_age: int = 18,
    gap_seconds: float = 0.70,
) -> MultiAgentRaceState:
    """Build a deterministic 4-8 car traffic-heavy Phase 8 scenario."""

    if car_count < 4 or car_count > 8:
        raise ValueError("traffic-heavy Phase 8 scenarios are intentionally limited to 4-8 cars")
    cars: list[StrategyState] = []
    for idx in range(int(car_count)):
        driver_id = f"car_{idx + 1}"
        race_time = (float(base_lap_seconds) * float(start_lap)) + (float(idx) * float(gap_seconds))
        cars.append(
            StrategyState(
                event_key=202608,
                driver_id=driver_id,
                lap_number=int(start_lap),
                total_laps=int(total_laps),
                remaining_laps=max(0, int(total_laps) - int(start_lap)),
                stint_id=1,
                compound="MEDIUM",
                tyre_age=int(starting_tyre_age),
                used_compounds=("MEDIUM",),
                race_time_seconds=float(race_time),
                gap_to_leader_seconds=float(idx) * float(gap_seconds),
                position=idx + 1,
                track_status="1",
                is_greenish=True,
                pace_penalty_mean=0.08 * float(idx % 3),
                deg_rate_mean=0.13,
                next_lap_mean=float(base_lap_seconds),
                pit_loss_estimate_seconds=20.5,
                circuit_overtaking_difficulty=0.88,
                circuit_tyre_degradation=0.92,
                circuit_safety_car_probability=0.18,
                circuit_strategy_variance=0.70,
                track_overtake_propensity=0.12,
                track_chaos_index=0.55,
                metadata={
                    "available_compounds": ("MEDIUM", "HARD"),
                    "pit_lane_open": True,
                    "event_lap_baseline_seconds": float(base_lap_seconds),
                    "circuit_id": "phase8_low_overtake_synthetic",
                    "ignored_future_columns": tuple(),
                    "multi_agent_scenario_id": scenario_id,
                    "multi_agent_seed": int(seed),
                },
            )
        )
    return MultiAgentRaceState(
        cars=tuple(cars),
        lap_number=int(start_lap),
        scenario_id=str(scenario_id),
        seed=int(seed),
        metadata={"scenario_type": "traffic_heavy", "simplified_car_count": int(car_count)},
    )


build_synthetic_multi_agent_scenario = build_traffic_heavy_scenario


def _race_time(car: StrategyState) -> float:
    if car.race_time_seconds is not None:
        return float(car.race_time_seconds)
    if car.position is not None:
        return float(car.position) * 2.0
    return 0.0


def _order_indices_by_race_time(cars: Sequence[StrategyState]) -> list[int]:
    return sorted(range(len(cars)), key=lambda idx: (_race_time(cars[idx]), str(cars[idx].driver_id)))


def _gap_ahead_seconds(cars: Sequence[StrategyState], idx: int, order: Sequence[int]) -> float:
    pos = list(order).index(idx)
    if pos <= 0:
        return float("inf")
    return max(0.0, _race_time(cars[idx]) - _race_time(cars[order[pos - 1]]))


def _gap_behind_seconds(cars: Sequence[StrategyState], idx: int, order: Sequence[int]) -> float:
    pos = list(order).index(idx)
    if pos + 1 >= len(order):
        return float("inf")
    return max(0.0, _race_time(cars[order[pos + 1]]) - _race_time(cars[idx]))


def _traffic_pressure(gap_ahead: float, difficulty: float, config: MultiAgentRaceConfig) -> float:
    if not np.isfinite(gap_ahead):
        return 0.0
    gap_factor = float(np.clip(1.0 - (float(gap_ahead) / max(0.1, float(config.traffic_gap_seconds))), 0.0, 1.0))
    return float((config.traffic_loss_seconds * difficulty * gap_factor) + (config.dirty_air_loss_seconds * gap_factor))


def _recent_pit_count(pit_counts_by_lap: Mapping[int, int], lap_number: int, cooldown_laps: int) -> int:
    start = int(lap_number) - max(0, int(cooldown_laps))
    return int(sum(int(count) for lap, count in pit_counts_by_lap.items() if start <= int(lap) < int(lap_number)))


def _event_lap_baseline(car: StrategyState, config: MultiAgentRaceConfig) -> float:
    metadata = car.metadata if isinstance(car.metadata, Mapping) else {}
    value = _finite(metadata.get("event_lap_baseline_seconds"), float("nan"))
    if not np.isfinite(value):
        value = _finite(car.next_lap_mean, float(config.base_lap_seconds))
    return float(value if np.isfinite(value) else config.base_lap_seconds)


def _mapping_float(mapping: Mapping[object, object], key: object, default: float = 0.0) -> float:
    if key in mapping:
        return _finite(mapping.get(key), default)
    text = str(key)
    if text in mapping:
        return _finite(mapping.get(text), default)
    return float(default)


def _finite(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        numeric = float(value)
        if not np.isfinite(numeric):
            return float(default)
        return numeric
    except Exception:
        return float(default)


def _stable_uint64(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _deterministic_normal(*, seed: int, scenario_id: str, driver_id: str, lap_number: int, std: float) -> float:
    if float(std) <= 0.0:
        return 0.0
    rng = np.random.default_rng(_stable_uint64(seed, scenario_id, driver_id, int(lap_number), "phase8_noise"))
    return float(rng.normal(0.0, float(std)))


def _car_payload(car: StrategyState) -> dict[str, object]:
    metadata = car.metadata if isinstance(car.metadata, Mapping) else {}
    public_keys = (
        "multi_agent_car_index",
        "multi_agent_car_count",
        "multi_agent_recent_pit_fraction",
        "multi_agent_recent_pit_count",
        "multi_agent_sync_pit_threshold",
        "gap_to_car_ahead_seconds",
        "gap_to_car_behind_seconds",
        "multi_agent_traffic_pressure",
        "last_multi_agent_action_key",
    )
    return {
        "driver_id": car.driver_id,
        "lap_number": int(car.lap_number),
        "remaining_laps": car.remaining_laps,
        "compound": car.compound,
        "tyre_age": int(car.tyre_age),
        "used_compounds": list(car.used_compounds),
        "race_time_seconds": car.race_time_seconds,
        "position": car.position,
        "gap_to_leader_seconds": car.gap_to_leader_seconds,
        "metadata": {key: _json_safe(metadata.get(key)) for key in public_keys if key in metadata},
    }


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    try:
        if value is None or np.isnan(value):  # type: ignore[arg-type]
            return None
    except Exception:
        pass
    return value


def _stable_digest(payload: Mapping[str, object]) -> str:
    text = json.dumps(_json_safe(dict(payload)), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "MultiAgentLiveRaceEnv",
    "MultiAgentPolicy",
    "MultiAgentRaceConfig",
    "MultiAgentRaceState",
    "MultiAgentRolloutResult",
    "MultiAgentStepResult",
    "build_synthetic_multi_agent_scenario",
    "build_traffic_heavy_scenario",
]
