"""Seedable live-race simulator for F1 strategy planning.

This is the Phase 5 world model.  It remains intentionally single-car and
proxy-based, but every transition exposes the same typed ``StrategyTransition``
contract used by replay, DP/MPC, and later RL policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from typing import Callable, Iterable, Mapping, Optional, Protocol, Sequence

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
    LEAKAGE_CONTRACT_VERSION,
    RewardConfig,
    StrategyReward,
    StrategyState,
    StrategyTransition,
    build_replay_transitions,
)
from packages.f1.models.live_race.pit_loss import PitLossConfig, estimate_pit_loss_seconds
from packages.f1.models.live_race.state import compound_deg_prior, parse_track_status
from packages.f1.models.live_race.traffic import TrafficModelConfig, estimate_traffic_loss_seconds


class SimulatorPolicy(Protocol):
    def select_action(self, state: StrategyState) -> StrategyAction:
        ...


@dataclass(frozen=True)
class SimulatorScenario:
    """Deterministic scenario perturbation for simulator rollouts."""

    scenario_id: str = "base"
    seed_offset: int = 0
    baseline_offset_seconds: float = 0.0
    lap_baseline_offsets: Mapping[int, float] = field(default_factory=dict)
    driver_pace_offset_seconds: float = 0.0
    degradation_multiplier: float = 1.0
    compound_effect_offsets: Mapping[str, float] = field(default_factory=dict)
    fuel_burn_multiplier: float = 1.0
    traffic_multiplier: float = 1.0
    pit_loss_multiplier: float = 1.0
    pit_loss_offset_seconds: float = 0.0
    track_status_offsets: Mapping[str, float] = field(default_factory=dict)
    weather_offset_seconds: float = 0.0
    noise_std_seconds: float = 0.18
    forced_track_status: Optional[str] = None
    wet_track: bool = False
    overtake_propensity: Optional[float] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RaceSimulatorConfig:
    model_id: str = "live_race_simulator_v1"
    seed: int = 7
    default_total_laps: int = 58
    default_base_lap_seconds: float = 90.0
    event_lap_slope_seconds: float = 0.0
    fuel_burn_lap_gain_seconds: float = 0.035
    tyre_cliff_start_ratio: float = 0.72
    tyre_cliff_quadratic_seconds: float = 0.032
    min_lap_seconds: float = 45.0
    max_lap_seconds: float = 220.0
    compound_effects_seconds: Mapping[str, float] = field(
        default_factory=lambda: {
            "SOFT": -0.35,
            "MEDIUM": 0.0,
            "HARD": 0.28,
            "INTER": 1.25,
            "WET": 2.20,
        }
    )
    track_status_offsets_seconds: Mapping[str, float] = field(
        default_factory=lambda: {
            "green": 0.0,
            "yellow": 2.4,
            "sc_vsc": 8.5,
            "red": 45.0,
            "ambiguous": 0.8,
        }
    )
    dry_on_wet_penalty_seconds: float = 4.0
    wet_compound_on_wet_offset_seconds: float = 0.8
    conservative_lap_delta_seconds: float = 0.06
    aggressive_lap_delta_seconds: float = -0.12
    conservative_degradation_multiplier: float = 0.88
    aggressive_degradation_multiplier: float = 1.18
    action_mask: ActionMaskConfig = ActionMaskConfig()
    reward: RewardConfig = RewardConfig(position_gain_weight=0.0)
    traffic: TrafficModelConfig = TrafficModelConfig()
    pit_loss: PitLossConfig = PitLossConfig()
    compounds: tuple[str, ...] = DRY_COMPOUNDS
    include_pit_next_lap: bool = True


@dataclass(frozen=True)
class SimulatorStepBreakdown:
    event_lap_baseline: float
    driver_pace_state: float
    tyre_degradation_state: float
    compound_effect: float
    fuel_proxy: float
    traffic_loss: float
    track_status_offset: float
    weather_offset: float
    random_noise: float
    pit_loss: float
    elapsed_seconds: float

    def to_payload(self) -> dict[str, float]:
        return {
            "event_lap_baseline": float(self.event_lap_baseline),
            "driver_pace_state": float(self.driver_pace_state),
            "tyre_degradation_state": float(self.tyre_degradation_state),
            "compound_effect": float(self.compound_effect),
            "fuel_proxy": float(self.fuel_proxy),
            "traffic_loss": float(self.traffic_loss),
            "track_status_offset": float(self.track_status_offset),
            "weather_offset": float(self.weather_offset),
            "random_noise": float(self.random_noise),
            "pit_loss": float(self.pit_loss),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def _finite(value: object, default: float) -> float:
    try:
        if value is None or not np.isfinite(float(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _mapping_float(mapping: Mapping[object, object], key: object, default: float = 0.0) -> float:
    if key in mapping:
        return _finite(mapping.get(key), default)
    text_key = str(key)
    if text_key in mapping:
        return _finite(mapping.get(text_key), default)
    return float(default)


def _stable_uint64(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _deterministic_normal(*, seed: int, scenario_id: str, state: StrategyState, lap_number: int, std: float) -> float:
    if float(std) <= 0.0:
        return 0.0
    raw_seed = _stable_uint64(seed, scenario_id, state.event_key, state.driver_id, int(lap_number), "lap_noise")
    rng = np.random.default_rng(raw_seed)
    return float(rng.normal(0.0, float(std)))


def _status_key(state: StrategyState) -> str:
    if state.is_red:
        return "red"
    if state.is_sc_vsc:
        return "sc_vsc"
    if state.is_yellow:
        return "yellow"
    if state.is_greenish:
        return "green"
    return "ambiguous"


def _scenario_status_state(state: StrategyState, scenario: SimulatorScenario) -> StrategyState:
    if not scenario.forced_track_status:
        return state
    flags = parse_track_status(scenario.forced_track_status)
    return replace(
        state,
        track_status=str(scenario.forced_track_status),
        is_red=bool(flags.is_red),
        is_sc_vsc=bool(flags.is_sc_vsc),
        is_yellow=bool(flags.is_yellow),
        is_greenish=bool(flags.is_greenish),
    )


def _service_life(compound: str) -> float:
    if compound == "SOFT":
        return 16.0
    if compound == "MEDIUM":
        return 22.0
    if compound == "HARD":
        return 28.0
    if compound == "INTER":
        return 14.0
    if compound == "WET":
        return 12.0
    return 20.0


def _mode_lap_delta(action: StrategyAction, config: RaceSimulatorConfig) -> float:
    if action.mode == "aggressive":
        return float(config.aggressive_lap_delta_seconds)
    return float(config.conservative_lap_delta_seconds)


def _mode_deg_multiplier(action: StrategyAction, config: RaceSimulatorConfig) -> float:
    if action.mode == "aggressive":
        return float(config.aggressive_degradation_multiplier)
    return float(config.conservative_degradation_multiplier)


def _points_proxy(position: Optional[int]) -> float:
    if position is None:
        return 0.0
    table = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0, 6: 8.0, 7: 6.0, 8: 4.0, 9: 2.0, 10: 1.0}
    return float(table.get(int(position), 0.0))


class LiveRaceSimulator:
    """Deterministic, seedable single-car simulator."""

    limitations = (
        "single_car_traffic_and_final_order_are_proxy_only",
        "hand_tuned_pace_tyre_pit_loss_priors_until_calibrated",
        "same_seed_scenarios_are_deterministic_but_not_stochastic_weather_forecasts",
    )

    def __init__(
        self,
        *,
        config: RaceSimulatorConfig | None = None,
        scenario: SimulatorScenario | None = None,
        action_space: Optional[Sequence[StrategyAction]] = None,
    ) -> None:
        self.config = config or RaceSimulatorConfig()
        self.scenario = scenario or SimulatorScenario()
        self.action_space = tuple(
            action_space
            or build_action_space(
                compounds=self.config.compounds,
                include_pit_next_lap=bool(self.config.include_pit_next_lap),
            )
        )
        self.model_id = self.config.model_id

    def with_scenario(self, scenario: SimulatorScenario) -> "LiveRaceSimulator":
        return LiveRaceSimulator(config=self.config, scenario=scenario, action_space=self.action_space)

    def step(self, state: StrategyState, action: StrategyAction) -> StrategyTransition:
        state = self._normalize_state_metadata(state)
        action = action if isinstance(action, StrategyAction) else StrategyAction.from_key(action)
        legal_mask = build_legal_action_mask(state, action_space=self.action_space, config=self.config.action_mask)
        if state.remaining_laps is not None and int(state.remaining_laps) <= 0:
            return StrategyTransition(
                state_t=state,
                action_t=action,
                reward_t=StrategyReward(
                    value=0.0,
                    components={
                        "race_time_delta_seconds": 0.0,
                        "illegal_action_penalty": 0.0,
                    },
                    note="race_complete_no_transition",
                ),
                state_t1=state,
                done=True,
                legal_action_mask=legal_mask,
                metadata={
                    "transition_model": self.model_id,
                    "scenario_id": self.scenario.scenario_id,
                    "terminal_noop": True,
                    "limitations": self.limitations,
                },
            )

        operational_fallback_executed = bool(
            legal_mask.operational_fallback_applied
            and legal_mask.is_selectable(action)
        )
        if not legal_mask.is_legal(action) and not operational_fallback_executed:
            penalty = float(self.config.reward.illegal_action_penalty)
            return StrategyTransition(
                state_t=state,
                action_t=action,
                reward_t=StrategyReward(
                    value=-penalty,
                    components={"illegal_action_penalty": penalty, "race_time_delta_seconds": 0.0},
                    note=legal_mask.reason_for(action),
                ),
                state_t1=state,
                done=bool(state.remaining_laps == 0),
                legal_action_mask=legal_mask,
                metadata={
                    "transition_model": self.model_id,
                    "scenario_id": self.scenario.scenario_id,
                    "invalid_action_reason": legal_mask.reason_for(action),
                    "limitations": self.limitations,
                },
            )

        status_state = _scenario_status_state(state, self.scenario)
        next_lap = int(state.lap_number) + 1
        compound_for_lap = (
            normalize_compound(action.compound) if action.action_type == ACTION_PIT_NOW else normalize_compound(state.compound)
        )
        if compound_for_lap == "UNKNOWN":
            compound_for_lap = "MEDIUM"
        tyre_age_for_lap = 0 if action.action_type == ACTION_PIT_NOW else int(state.tyre_age)
        deg_rate_for_lap = compound_deg_prior(compound_for_lap) if action.action_type == ACTION_PIT_NOW else _finite(
            state.deg_rate_mean,
            compound_deg_prior(compound_for_lap),
        )

        traffic_loss = estimate_traffic_loss_seconds(
            status_state,
            action=action,
            config=self.config.traffic,
            scenario_multiplier=self.scenario.traffic_multiplier,
            scenario_overtake_propensity=self.scenario.overtake_propensity,
        )
        pit_loss = 0.0
        if action.action_type == ACTION_PIT_NOW:
            pit_loss = estimate_pit_loss_seconds(
                status_state,
                config=self.config.pit_loss,
                traffic_loss_seconds=traffic_loss,
                scenario_multiplier=self.scenario.pit_loss_multiplier,
                scenario_offset_seconds=self.scenario.pit_loss_offset_seconds,
                scenario_overtake_propensity=self.scenario.overtake_propensity,
            )

        breakdown = self._breakdown(
            status_state,
            action,
            next_lap=next_lap,
            compound=compound_for_lap,
            tyre_age=tyre_age_for_lap,
            deg_rate=deg_rate_for_lap,
            traffic_loss=traffic_loss,
            pit_loss=pit_loss,
        )
        elapsed = max(0.1, float(breakdown.elapsed_seconds))

        next_state = self._next_state(
            status_state,
            action,
            next_lap=next_lap,
            elapsed=elapsed,
            breakdown=breakdown,
            compound_for_lap=compound_for_lap,
        )
        reward = StrategyReward(
            value=-float(elapsed)
            - (
                float(self.config.reward.illegal_action_penalty)
                if operational_fallback_executed
                else 0.0
            ),
            components={
                **breakdown.to_payload(),
                "race_time_delta_seconds": float(elapsed),
                "estimated_lap_seconds": float(elapsed - pit_loss),
                "illegal_action_penalty": (
                    float(self.config.reward.illegal_action_penalty)
                    if operational_fallback_executed
                    else 0.0
                ),
                "points_proxy": _points_proxy(next_state.position),
            },
            note=(
                "operational_fallback_nonlegal_safety_transition"
                if operational_fallback_executed
                else "live_race_simulator_transition"
            ),
        )
        return StrategyTransition(
            state_t=state,
            action_t=action,
            reward_t=reward,
            state_t1=next_state,
            done=bool(next_state.remaining_laps == 0),
            legal_action_mask=legal_mask,
            metadata={
                "transition_model": self.model_id,
                "scenario_id": self.scenario.scenario_id,
                "seed": int(self.config.seed),
                "constraint_legal_action": bool(
                    legal_mask.is_legal(action)
                ),
                "operational_fallback_executed": bool(
                    operational_fallback_executed
                ),
                "breakdown": breakdown.to_payload(),
                "limitations": self.limitations,
                "final_order_is_proxy_only": True,
            },
        )

    def simulate_action_sequence(
        self,
        state: StrategyState,
        actions: Sequence[StrategyAction],
        *,
        stop_on_done: bool = True,
    ) -> tuple[StrategyTransition, ...]:
        current = state
        transitions: list[StrategyTransition] = []
        for action in actions:
            transition = self.step(current, action)
            transitions.append(transition)
            current = transition.state_t1
            if stop_on_done and transition.done:
                break
        return tuple(transitions)

    def simulate_policy(
        self,
        start_state: StrategyState | Mapping[str, object],
        *,
        policy: Optional[SimulatorPolicy | Callable[[StrategyState], StrategyAction]] = None,
        actions: Optional[Sequence[StrategyAction]] = None,
        max_laps: Optional[int] = None,
    ) -> tuple[StrategyTransition, ...]:
        state = start_state if isinstance(start_state, StrategyState) else StrategyState.from_mapping(start_state)
        remaining = state.remaining_laps if state.remaining_laps is not None else max(0, int(self.config.default_total_laps) - state.lap_number)
        limit = int(max_laps) if max_laps is not None else int(max(0, remaining))
        provided_actions = tuple(actions or ())

        transitions: list[StrategyTransition] = []
        current = state
        for idx in range(limit):
            if idx < len(provided_actions):
                action = provided_actions[idx]
            else:
                action = self._select_policy_action(policy, current)
            transition = self.step(current, action)
            transitions.append(transition)
            current = transition.state_t1
            if transition.done:
                break
        return tuple(transitions)

    def replay_race(
        self,
        laps: pd.DataFrame,
        *,
        policy: Optional[SimulatorPolicy | Callable[[StrategyState], StrategyAction]] = None,
        driver_id: Optional[str] = None,
        start_lap: Optional[int] = None,
    ) -> tuple[StrategyTransition, ...]:
        """Replay a single driver's race from lap 1 or any live replay lap.

        Observed adjacent-lap actions seed the replay when no policy is
        supplied; the simulator then rolls the state to the remaining horizon
        with the supplied policy or legal stay-out fallback.
        """

        if not isinstance(laps, pd.DataFrame):
            raise TypeError("laps must be a pandas DataFrame")
        if laps.empty:
            return ()
        frame = laps.copy()
        if driver_id is not None and "driver_id" in frame.columns:
            frame = frame[frame["driver_id"].astype(str) == str(driver_id)].copy()
        if frame.empty:
            return ()
        if "driver_id" in frame.columns and frame["driver_id"].astype(str).nunique() > 1:
            raise ValueError("replay_race requires one driver or an explicit driver_id")
        event_col = "event_key" if "event_key" in frame.columns else (
            "meeting_key" if "meeting_key" in frame.columns else None
        )
        if event_col is not None and frame[event_col].dropna().astype(str).nunique() > 1:
            raise ValueError("replay_race requires event-isolated input")
        lap_col = "lap_number" if "lap_number" in frame.columns else "LapNumber"
        frame[lap_col] = pd.to_numeric(frame[lap_col], errors="coerce")
        frame = frame[frame[lap_col].notna()].sort_values(lap_col, kind="mergesort")
        if start_lap is not None:
            frame = frame[frame[lap_col] >= int(start_lap)].copy()
        if frame.empty:
            return ()

        rows = [row for _, row in frame.iterrows()]
        start_state = StrategyState.from_mapping(rows[0])
        actions: list[StrategyAction] = []
        if policy is None:
            observed_transitions = build_replay_transitions(
                frame,
                action_space=self.action_space,
                action_mask_config=self.config.action_mask,
            )
            for observed in observed_transitions:
                actions.append(observed.action_t)
                actions.extend(
                    StrategyAction(ACTION_STAY_OUT)
                    for _ in range(max(0, int(observed.metadata.get("elapsed_laps", 1)) - 1))
                )
        return self.simulate_policy(start_state, policy=policy, actions=actions)

    def compare_policies_same_seed(
        self,
        start_state: StrategyState,
        policies: Mapping[str, SimulatorPolicy | Callable[[StrategyState], StrategyAction]],
        *,
        max_laps: Optional[int] = None,
    ) -> dict[str, dict[str, object]]:
        """Run policies under the identical configured seed/scenario."""

        out: dict[str, dict[str, object]] = {}
        for name, policy in policies.items():
            transitions = self.simulate_policy(start_state, policy=policy, max_laps=max_laps)
            total_time = float(sum(t.reward_t.components.get("race_time_delta_seconds", 0.0) for t in transitions))
            illegal = int(sum(0 if t.is_action_legal() else 1 for t in transitions))
            final_state = transitions[-1].state_t1 if transitions else start_state
            out[str(name)] = {
                "transitions": int(len(transitions)),
                "horizon_time_seconds": total_time,
                "illegal_action_count": illegal,
                "final_position_proxy": final_state.position,
                "points_proxy": _points_proxy(final_state.position),
                "action_keys": [transition.action_t.key for transition in transitions],
                "scenario_id": self.scenario.scenario_id,
                "seed": int(self.config.seed),
            }
        return out

    def _normalize_state_metadata(self, state: StrategyState) -> StrategyState:
        metadata = dict(state.metadata or {})
        metadata.setdefault("ignored_future_columns", tuple())
        metadata.setdefault("available_through_lap", int(state.lap_number))
        metadata.setdefault(
            "leakage_contract_version",
            state.metadata.get("leakage_contract_version", LEAKAGE_CONTRACT_VERSION),
        )
        if self.scenario.wet_track:
            metadata["weather_is_wet"] = True
        return replace(state, metadata=metadata)

    def _event_lap_baseline(self, state: StrategyState, *, next_lap: int) -> float:
        metadata = state.metadata if isinstance(state.metadata, Mapping) else {}
        if "event_lap_baseline_by_lap" in metadata:
            # A complete per-lap map is indistinguishable here from future
            # actual race outcomes.  The state contract has no first-seen/as-of
            # certificate for individual map entries, so accepting it would let
            # policy evaluation consume future information.  Use the causal
            # scalar estimate available at the current state cutoff instead.
            raise ValueError(
                "event_lap_baseline_by_lap is prohibited in causal policy simulation; "
                "supply event_lap_baseline_seconds known at the state cutoff"
            )

        explicit = _finite(metadata.get("event_lap_baseline_seconds"), float("nan"))
        if not np.isfinite(explicit):
            explicit = _finite(metadata.get("baseline_lap_seconds"), float("nan"))
        if not np.isfinite(explicit) and state.next_lap_mean is not None:
            explicit = max(1.0, float(state.next_lap_mean) - _finite(state.pace_penalty_mean, 0.0))
        if not np.isfinite(explicit):
            explicit = float(self.config.default_base_lap_seconds) + (float(self.config.event_lap_slope_seconds) * float(next_lap))
        return float(explicit + self.scenario.baseline_offset_seconds + _mapping_float(self.scenario.lap_baseline_offsets, int(next_lap), 0.0))

    def _tyre_degradation_state(self, state: StrategyState, action: StrategyAction, *, compound: str, tyre_age: int, deg_rate: float) -> float:
        tyre_prior = _finite(state.circuit_tyre_degradation, 0.55)
        circuit_multiplier = float(np.clip(0.75 + (0.65 * tyre_prior), 0.75, 1.45))
        rate = max(0.0, float(deg_rate)) * circuit_multiplier * _mode_deg_multiplier(action, self.config)
        rate *= max(0.0, float(self.scenario.degradation_multiplier))
        age_penalty = rate * float(max(0, tyre_age))
        cliff_start = float(self.config.tyre_cliff_start_ratio) * _service_life(compound)
        cliff_age = max(0.0, float(tyre_age) - cliff_start)
        return float(age_penalty + (float(self.config.tyre_cliff_quadratic_seconds) * (cliff_age**2)))

    def _compound_effect(self, compound: str) -> float:
        base = _mapping_float(self.config.compound_effects_seconds, compound, 0.0)
        offset = _mapping_float(self.scenario.compound_effect_offsets, compound, 0.0)
        return float(base + offset)

    def _track_status_offset(self, state: StrategyState) -> float:
        key = _status_key(state)
        base = _mapping_float(self.config.track_status_offsets_seconds, key, 0.0)
        offset = _mapping_float(self.scenario.track_status_offsets, key, 0.0)
        return float(base + offset)

    def _weather_offset(self, compound: str) -> float:
        offset = float(self.scenario.weather_offset_seconds)
        if self.scenario.wet_track:
            if compound in DRY_COMPOUNDS:
                offset += float(self.config.dry_on_wet_penalty_seconds)
            else:
                offset += float(self.config.wet_compound_on_wet_offset_seconds)
        return float(offset)

    def _breakdown(
        self,
        state: StrategyState,
        action: StrategyAction,
        *,
        next_lap: int,
        compound: str,
        tyre_age: int,
        deg_rate: float,
        traffic_loss: float,
        pit_loss: float,
    ) -> SimulatorStepBreakdown:
        event_lap_baseline = self._event_lap_baseline(state, next_lap=next_lap)
        driver_pace_state = _finite(state.pace_penalty_mean, 0.0) + float(self.scenario.driver_pace_offset_seconds)
        tyre_degradation_state = self._tyre_degradation_state(
            state,
            action,
            compound=compound,
            tyre_age=tyre_age,
            deg_rate=deg_rate,
        )
        compound_effect = self._compound_effect(compound)
        fuel_proxy = -float(self.config.fuel_burn_lap_gain_seconds) * float(max(0, state.lap_number)) * float(
            self.scenario.fuel_burn_multiplier
        )
        track_status_offset = self._track_status_offset(state)
        weather_offset = self._weather_offset(compound)
        random_noise = _deterministic_normal(
            seed=int(self.config.seed) + int(self.scenario.seed_offset),
            scenario_id=self.scenario.scenario_id,
            state=state,
            lap_number=next_lap,
            std=float(self.scenario.noise_std_seconds),
        )
        lap_time = (
            event_lap_baseline
            + driver_pace_state
            + tyre_degradation_state
            + compound_effect
            + _mode_lap_delta(action, self.config)
            + fuel_proxy
            + float(traffic_loss)
            + track_status_offset
            + weather_offset
            + random_noise
        )
        lap_time = float(np.clip(lap_time, float(self.config.min_lap_seconds), float(self.config.max_lap_seconds)))
        elapsed = float(lap_time + max(0.0, float(pit_loss)))
        return SimulatorStepBreakdown(
            event_lap_baseline=float(event_lap_baseline),
            driver_pace_state=float(driver_pace_state),
            tyre_degradation_state=float(tyre_degradation_state),
            compound_effect=float(compound_effect + _mode_lap_delta(action, self.config)),
            fuel_proxy=float(fuel_proxy),
            traffic_loss=float(traffic_loss),
            track_status_offset=float(track_status_offset),
            weather_offset=float(weather_offset),
            random_noise=float(random_noise),
            pit_loss=float(pit_loss),
            elapsed_seconds=float(elapsed),
        )

    def _next_state(
        self,
        state: StrategyState,
        action: StrategyAction,
        *,
        next_lap: int,
        elapsed: float,
        breakdown: SimulatorStepBreakdown,
        compound_for_lap: str,
    ) -> StrategyState:
        metadata = dict(state.metadata or {})
        next_compound = state.compound
        next_tyre_age = int(state.tyre_age) + 1
        next_stint = int(state.stint_id)
        used_compounds = tuple(state.used_compounds)
        deg_rate_mean = float(state.deg_rate_mean)
        position_proxy = state.position

        if action.action_type == ACTION_PIT_NOW:
            next_compound = compound_for_lap
            next_tyre_age = 0
            next_stint += 1
            deg_rate_mean = compound_deg_prior(next_compound)
            if next_compound not in used_compounds:
                used_compounds = (*used_compounds, next_compound)
            metadata.pop("forced_pit_next_compound", None)
            metadata.pop("forced_pit_next_mode", None)
            if position_proxy is not None and not state.is_sc_vsc:
                loss_positions = max(1, int(round(max(1.0, breakdown.pit_loss) / 8.0)))
                position_proxy = min(20, int(position_proxy) + loss_positions)
        elif action.action_type == ACTION_PIT_NEXT_LAP:
            metadata["forced_pit_next_compound"] = normalize_compound(action.compound)
            metadata["forced_pit_next_mode"] = action.mode
        else:
            metadata.pop("forced_pit_next_compound", None)
            metadata.pop("forced_pit_next_mode", None)

        remaining = None if state.remaining_laps is None else max(0, int(state.remaining_laps) - 1)
        race_time = float(elapsed)
        if state.race_time_seconds is not None:
            race_time = float(state.race_time_seconds) + float(elapsed)

        metadata.update(
            {
                "available_through_lap": int(next_lap),
                "transition_model": self.model_id,
                "simulator_scenario_id": self.scenario.scenario_id,
                "last_simulator_breakdown": breakdown.to_payload(),
            }
        )
        if self.scenario.metadata:
            metadata["scenario_metadata"] = dict(self.scenario.metadata)

        return replace(
            state,
            lap_number=int(next_lap),
            remaining_laps=remaining,
            stint_id=int(next_stint),
            compound=next_compound,
            tyre_age=int(next_tyre_age),
            used_compounds=used_compounds,
            race_time_seconds=float(race_time),
            position=position_proxy,
            deg_rate_mean=float(deg_rate_mean),
            next_lap_mean=float(elapsed),
            metadata=metadata,
        )

    def _select_policy_action(
        self,
        policy: Optional[SimulatorPolicy | Callable[[StrategyState], StrategyAction]],
        state: StrategyState,
    ) -> StrategyAction:
        if policy is None:
            stay = StrategyAction(ACTION_STAY_OUT)
            legal_mask = build_legal_action_mask(state, action_space=self.action_space, config=self.config.action_mask)
            if legal_mask.is_legal(stay):
                return stay
            legal = legal_mask.legal_actions
            return legal[0] if legal else stay
        if hasattr(policy, "select_action"):
            action = policy.select_action(state)  # type: ignore[union-attr]
        elif hasattr(policy, "plan"):
            action = policy.plan(state).action  # type: ignore[union-attr]
        else:
            action = policy(state)  # type: ignore[misc]
        return action if isinstance(action, StrategyAction) else StrategyAction.from_key(action)


def default_strategy_scenarios(seed: int = 7) -> tuple[SimulatorScenario, ...]:
    """Named stress scenarios required by the Phase 6 planner diagnostics."""

    return (
        SimulatorScenario(
            scenario_id="monaco_low_overtake",
            seed_offset=11,
            traffic_multiplier=1.45,
            pit_loss_multiplier=1.10,
            overtake_propensity=0.12,
            metadata={"diagnostic": "low-overtake/Monaco-style traffic"},
        ),
        SimulatorScenario(
            scenario_id="high_overtake",
            seed_offset=23,
            traffic_multiplier=0.70,
            pit_loss_multiplier=0.95,
            overtake_propensity=0.78,
            metadata={"diagnostic": "high-overtake circuit"},
        ),
        SimulatorScenario(
            scenario_id="wet",
            seed_offset=37,
            degradation_multiplier=1.18,
            weather_offset_seconds=1.2,
            wet_track=True,
            noise_std_seconds=0.35,
            metadata={"diagnostic": "wet track proxy; points/final order remain proxy-only"},
        ),
        SimulatorScenario(
            scenario_id="sc_vsc",
            seed_offset=41,
            forced_track_status="4",
            pit_loss_multiplier=0.72,
            traffic_multiplier=0.35,
            track_status_offsets={"sc_vsc": 4.0},
            metadata={"diagnostic": "SC/VSC discounted pit-loss window"},
        ),
    )


def simulation_trace_frame(transitions: Iterable[StrategyTransition]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, transition in enumerate(transitions):
        components = transition.reward_t.components
        rows.append(
            {
                "index": int(idx),
                "driver_id": transition.state_t.driver_id,
                "lap_number": int(transition.state_t1.lap_number),
                "action_key": transition.action_t.key,
                "compound": transition.state_t1.compound,
                "tyre_age": int(transition.state_t1.tyre_age),
                "race_time_seconds": transition.state_t1.race_time_seconds,
                "position_proxy": transition.state_t1.position,
                "lap_time_seconds": components.get("estimated_lap_seconds"),
                "race_time_delta_seconds": components.get("race_time_delta_seconds"),
                "pit_loss_seconds": components.get("pit_loss"),
                "traffic_loss_seconds": components.get("traffic_loss"),
                "track_status_offset_seconds": components.get("track_status_offset"),
                "weather_offset_seconds": components.get("weather_offset"),
                "scenario_id": transition.metadata.get("scenario_id"),
                "is_action_legal": transition.is_action_legal(),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "LiveRaceSimulator",
    "RaceSimulatorConfig",
    "SimulatorPolicy",
    "SimulatorScenario",
    "SimulatorStepBreakdown",
    "default_strategy_scenarios",
    "simulation_trace_frame",
]
