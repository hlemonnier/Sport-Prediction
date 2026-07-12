from __future__ import annotations

from packages.f1.models.live_race.action_space import (
    ACTION_PIT_NEXT_LAP,
    ACTION_PIT_NOW,
    ACTION_STAY_OUT,
    StrategyAction,
    build_legal_action_mask,
)
from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.mpc import SimulatorMPCStrategyPlanner
from packages.f1.models.live_race.planner import PlannerConfig, SimulatorDPPlanner, SimulatorMPCConfig, SimulatorMPCPlanner
from packages.f1.models.live_race.simulator import RaceSimulatorConfig


def _state(**overrides: object) -> StrategyState:
    base = {
        "event_key": 202601,
        "driver_id": "16",
        "lap_number": 20,
        "total_laps": 58,
        "remaining_laps": 38,
        "stint_id": 1,
        "compound": "SOFT",
        "tyre_age": 24,
        "used_compounds": ("SOFT",),
        "race_time_seconds": 1800.0,
        "position": 6,
        "track_status": "1",
        "is_greenish": True,
        "pace_penalty_mean": 0.8,
        "deg_rate_mean": 0.20,
        "next_lap_mean": 90.0,
        "circuit_tyre_degradation": 0.95,
        "circuit_overtaking_difficulty": 0.35,
        "metadata": {
            "available_compounds": ("SOFT", "MEDIUM", "HARD"),
            "pit_lane_open": True,
            "event_lap_baseline_seconds": 89.5,
            "ignored_future_columns": tuple(),
        },
    }
    base.update(overrides)
    return StrategyState(**base)


def _planner() -> SimulatorMPCPlanner:
    return SimulatorMPCPlanner(
        config=SimulatorMPCConfig(
            horizon_laps=4,
            branch_limit=7,
            max_sequences=160,
            race_time_weight=1.0,
            downside_cvar_weight=0.05,
            simulator=RaceSimulatorConfig(seed=55),
        )
    )


def test_simulator_mpc_uses_named_scenarios_and_beats_stay_out_sequence() -> None:
    planner = _planner()
    state = _state()

    result = planner.plan(state)
    stay_sequence = tuple(StrategyAction(ACTION_STAY_OUT) for _ in range(4))
    stay_score = planner.score_action_sequence(state, stay_sequence)

    assert any(action.action_type in {ACTION_PIT_NOW, ACTION_PIT_NEXT_LAP} for action in result.plan)
    assert result.value >= stay_score.utility
    assert set(result.diagnostics["scenario_ids"]) == {"monaco_low_overtake", "high_overtake", "wet", "sc_vsc"}
    assert result.diagnostics["points_and_final_order_are_proxy_only"] is True
    assert "downside_cvar_seconds" in result.diagnostics["best_sequence"]


def test_simulator_mpc_does_not_recommend_unavailable_compounds() -> None:
    planner = _planner()
    state = _state(metadata={"available_compounds": ("MEDIUM", "HARD"), "ignored_future_columns": tuple()})

    result = planner.plan(state)
    mask = build_legal_action_mask(state, action_space=planner.action_space, config=planner.config.action_mask)

    assert mask.is_legal(result.action)
    assert result.action.compound != "SOFT"
    assert "plan_selected_from_legal_action_sequences" in result.diagnostics["illegal_action_proof"]


def test_simulator_mpc_does_not_recommend_impossible_late_pit_windows() -> None:
    planner = _planner()
    state = _state(
        remaining_laps=2,
        lap_number=56,
        compound="HARD",
        tyre_age=5,
        used_compounds=("SOFT", "HARD"),
        metadata={"available_compounds": ("SOFT", "MEDIUM", "HARD"), "ignored_future_columns": tuple()},
    )

    result = planner.plan(state)

    assert result.action.action_type == ACTION_STAY_OUT


def test_simulator_mpc_wrapper_replans_to_contract_action() -> None:
    planner = _planner()
    state = _state()

    result = SimulatorMPCStrategyPlanner(planner=planner).replan(state)

    assert isinstance(result.action, StrategyAction)
    assert result.diagnostics["planner"] == "simulator_mpc_v1"


def test_simulator_dp_wrapper_uses_simulator_transition_model() -> None:
    planner = SimulatorDPPlanner(config=PlannerConfig(horizon_laps=3, strategy_score_weight=0.0))
    result = planner.plan(_state(remaining_laps=5))

    assert result.diagnostics["planner"] == "simulator_dp_v1"
    assert result.diagnostics["transition_model"] == "live_race_simulator_v1"


def test_dp_mandatory_compound_penalty_applies_only_at_race_finish() -> None:
    planner = SimulatorDPPlanner(config=PlannerConfig(horizon_laps=3, strategy_score_weight=0.0))

    assert planner._terminal_value(_state(remaining_laps=20, used_compounds=("SOFT",))) == 0.0
    assert planner._terminal_value(_state(remaining_laps=0, used_compounds=("SOFT",))) < 0.0
    assert (
        planner._terminal_value(
            _state(
                remaining_laps=0,
                compound="MEDIUM",
                used_compounds=("INTER", "MEDIUM"),
            )
        )
        == 0.0
    )


def test_mpc_branch_limit_preserves_pit_next_action_family() -> None:
    planner = SimulatorMPCPlanner(
        config=SimulatorMPCConfig(horizon_laps=2, branch_limit=3, max_sequences=20)
    )
    state = _state()
    mask = build_legal_action_mask(state, action_space=planner.action_space, config=planner.config.action_mask)

    selected = planner._ordered_candidate_actions(mask.legal_actions)

    assert {action.action_type for action in selected} == {
        ACTION_STAY_OUT,
        ACTION_PIT_NOW,
        ACTION_PIT_NEXT_LAP,
    }


def test_mpc_action_space_can_select_wet_compounds_when_track_is_wet() -> None:
    planner = SimulatorMPCPlanner(
        config=SimulatorMPCConfig(horizon_laps=2, branch_limit=8, max_sequences=40)
    )
    state = _state(
        metadata={
            "available_compounds": ("INTER", "WET"),
            "pit_lane_open": True,
            "weather_is_wet": True,
            "ignored_future_columns": tuple(),
        }
    )
    mask = build_legal_action_mask(state, action_space=planner.action_space, config=planner.config.action_mask)

    assert any(action.compound in {"INTER", "WET"} for action in mask.legal_actions)


def test_action_mask_fails_closed_without_inventory_or_pit_lane_state() -> None:
    planner = SimulatorMPCPlanner(
        config=SimulatorMPCConfig(horizon_laps=2, branch_limit=8, max_sequences=40)
    )
    state = _state(metadata={"ignored_future_columns": tuple()})
    mask = build_legal_action_mask(
        state,
        action_space=planner.action_space,
        config=planner.config.action_mask,
    )

    assert all(not action.is_pit_action for action in mask.legal_actions)
