from __future__ import annotations

import pandas as pd
import pytest

from packages.f1.models.live_race.action_space import ACTION_PIT_NOW, ACTION_STAY_OUT, StrategyAction
from packages.f1.models.live_race.simulator import (
    LiveRaceSimulator,
    RaceSimulatorConfig,
    SimulatorScenario,
    simulation_trace_frame,
)
from packages.f1.models.live_race.simulator_calibration import (
    build_simulator_calibration_report,
    pit_loss_calibration,
)
from packages.f1.models.live_race.state import compound_deg_prior


def _state(**overrides: object):
    base = {
        "event_key": 202601,
        "driver_id": "44",
        "lap_number": 1,
        "total_laps": 5,
        "remaining_laps": 4,
        "stint_id": 1,
        "compound": "SOFT",
        "tyre_age": 12,
        "used_compounds": ("SOFT",),
        "race_time_seconds": 90.0,
        "position": 5,
        "track_status": "1",
        "is_greenish": True,
        "pace_penalty_mean": 0.4,
        "deg_rate_mean": 0.08,
        "next_lap_mean": 91.0,
        "circuit_tyre_degradation": 0.80,
        "circuit_overtaking_difficulty": 0.45,
        "metadata": {
            "available_compounds": ("SOFT", "MEDIUM", "HARD"),
            "pit_lane_open": True,
            "event_lap_baseline_seconds": 90.0,
            "ignored_future_columns": tuple(),
        },
    }
    base.update(overrides)
    from packages.f1.models.live_race.environment import StrategyState

    return StrategyState(**base)


def _race_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_key": 202601,
                "driver_id": "44",
                "lap_number": 1,
                "total_laps": 5,
                "remaining_laps": 4,
                "stint_id": 1,
                "compound": "SOFT",
                "tyre_age": 0,
                "used_compounds": "SOFT",
                "available_compounds": "SOFT,MEDIUM,HARD",
                "pit_lane_open": True,
                "track_status": "1",
                "lap_time_seconds": 90.0,
                "race_time_seconds": 90.0,
                "timestamp": 1.0,
                "observed_action": "stay_out",
                "final_position": 5,
            },
            {
                "event_key": 202601,
                "driver_id": "44",
                "lap_number": 2,
                "total_laps": 5,
                "remaining_laps": 3,
                "stint_id": 1,
                "compound": "SOFT",
                "tyre_age": 1,
                "used_compounds": "SOFT",
                "available_compounds": "SOFT,MEDIUM,HARD",
                "pit_lane_open": True,
                "track_status": "1",
                "lap_time_seconds": 90.8,
                "race_time_seconds": 180.8,
                "timestamp": 2.0,
                "observed_action": "pit_now",
                "next_compound": "HARD",
                "pit_loss_seconds": 21.0,
                "final_position": 5,
            },
            {
                "event_key": 202601,
                "driver_id": "44",
                "lap_number": 3,
                "total_laps": 5,
                "remaining_laps": 2,
                "stint_id": 2,
                "compound": "HARD",
                "tyre_age": 0,
                "used_compounds": "SOFT,HARD",
                "available_compounds": "SOFT,MEDIUM,HARD",
                "pit_lane_open": True,
                "track_status": "1",
                "lap_time_seconds": 112.0,
                "race_time_seconds": 292.8,
                "timestamp": 3.0,
                "observed_action": "stay_out",
                "final_position": 5,
            },
        ]
    )


def test_live_race_simulator_is_seed_deterministic_for_same_scenario() -> None:
    state = _state()
    action = StrategyAction(ACTION_STAY_OUT)
    config = RaceSimulatorConfig(seed=123)
    scenario = SimulatorScenario(scenario_id="base", noise_std_seconds=0.25)

    left = LiveRaceSimulator(config=config, scenario=scenario).step(state, action)
    right = LiveRaceSimulator(config=config, scenario=scenario).step(state, action)

    assert left.reward_t.components["race_time_delta_seconds"] == right.reward_t.components["race_time_delta_seconds"]
    assert left.state_t1.fingerprint() == right.state_t1.fingerprint()


def test_terminal_simulator_step_is_a_done_noop_without_a_phantom_lap() -> None:
    state = _state(
        lap_number=5,
        remaining_laps=0,
        compound="HARD",
        used_compounds=("SOFT", "HARD"),
    )

    transition = LiveRaceSimulator().step(state, StrategyAction(ACTION_STAY_OUT))

    assert transition.done is True
    assert transition.state_t1 == state
    assert transition.reward_t.value == 0.0
    assert transition.metadata["terminal_noop"] is True
    assert transition.legal_action_mask.constraint_feasible is False


def test_single_car_simulator_advances_nonlegal_operational_fallback_with_penalty() -> None:
    state = _state(
        remaining_laps=2,
        compound="SOFT",
        used_compounds=("SOFT",),
        metadata={
            **_state().metadata,
            "available_compounds": ("SOFT",),
            "compound_inventory_known": True,
            "mandatory_compound_change_required": True,
        },
    )
    fallback = StrategyAction(ACTION_STAY_OUT)

    transition = LiveRaceSimulator().step(state, fallback)

    assert transition.state_t1.lap_number == state.lap_number + 1
    assert transition.state_t1.remaining_laps == 1
    assert transition.metadata["constraint_legal_action"] is False
    assert transition.metadata["operational_fallback_executed"] is True
    assert transition.reward_t.components["illegal_action_penalty"] > 0.0
    assert transition.reward_t.note == (
        "operational_fallback_nonlegal_safety_transition"
    )


def test_pit_action_applies_pit_loss_and_resets_tyre_and_degradation_prior() -> None:
    state = _state(compound="SOFT", tyre_age=20, used_compounds=("SOFT",), deg_rate_mean=0.16)
    action = StrategyAction(ACTION_PIT_NOW, compound="HARD")

    transition = LiveRaceSimulator(config=RaceSimulatorConfig(seed=9)).step(state, action)

    assert transition.is_action_legal()
    assert transition.state_t1.compound == "HARD"
    assert transition.state_t1.tyre_age == 0
    assert "HARD" in transition.state_t1.used_compounds
    assert transition.state_t1.deg_rate_mean == compound_deg_prior("HARD")
    assert transition.reward_t.components["pit_loss"] > 0.0
    assert transition.reward_t.components["race_time_delta_seconds"] > transition.reward_t.components["estimated_lap_seconds"]


def test_simulator_can_replay_from_lap_one_and_any_live_lap() -> None:
    simulator = LiveRaceSimulator(config=RaceSimulatorConfig(seed=17))
    full = simulator.replay_race(_race_rows())
    mid = simulator.replay_race(_race_rows(), start_lap=2)

    assert len(full) == 4
    assert len(mid) == 3
    assert full[-1].state_t1.lap_number == 5
    assert mid[0].state_t.lap_number == 2
    trace = simulation_trace_frame(full)
    assert set(trace["scenario_id"]) == {"base"}
    assert trace["race_time_delta_seconds"].notna().all()


def test_simulator_replay_rejects_cross_event_input() -> None:
    first = _race_rows()
    second = _race_rows()
    second["event_key"] = 202602

    with pytest.raises(ValueError, match="event-isolated"):
        LiveRaceSimulator().replay_race(
            pd.concat([first, second], ignore_index=True)
        )


def test_simulator_rejects_uncertified_future_lap_baseline_map() -> None:
    state = _state(
        metadata={
            **_state().metadata,
            "event_lap_baseline_by_lap": {2: 89.0, 3: 88.5},
        }
    )

    with pytest.raises(ValueError, match="event_lap_baseline_by_lap is prohibited"):
        LiveRaceSimulator().step(state, StrategyAction(ACTION_STAY_OUT))


def test_simulator_calibration_report_exposes_phase5_metric_groups() -> None:
    result = build_simulator_calibration_report(_race_rows(), simulator=LiveRaceSimulator(config=RaceSimulatorConfig(seed=3)))

    assert result.metrics["one_step_lap_time"]["available"] is True
    assert result.metrics["pit_loss"]["available"] is True
    assert result.metrics["pit_loss"]["matching"] == "transition_aligned_explicit_observed_pit_loss"
    assert result.metrics["track_status"]["available"] is True
    assert result.metrics["final_order_proxy"]["proxy_only"] is True


def test_pit_loss_estimate_is_not_accepted_as_observed_calibration_truth() -> None:
    laps = pd.DataFrame({"pit_loss_estimate_seconds": [21.0]})
    rows = [
        {
            "is_pit_action": True,
            "predicted_pit_loss_seconds": 21.0,
            "observed_pit_loss_seconds": float("nan"),
        }
    ]

    metrics = pit_loss_calibration(rows, laps)

    assert metrics["available"] is False
    assert metrics["reason"] == "no_explicit_observed_pit_loss_column"


def test_simulator_calibration_uses_the_same_semi_markov_pit_transition() -> None:
    laps = pd.DataFrame(
        {
            "event_key": [202601] * 4,
            "driver_id": ["44"] * 4,
            "lap_number": [1, 2, 3, 4],
            "total_laps": [6] * 4,
            "remaining_laps": [5, 4, 3, 2],
            "stint_id": [1, 1, 2, 2],
            "compound": ["MEDIUM", "MEDIUM", "HARD", "HARD"],
            "tyre_age": [4, 5, 0, 1],
            "used_compounds": ["MEDIUM", "MEDIUM", "MEDIUM,HARD", "MEDIUM,HARD"],
            "available_compounds": ["MEDIUM,HARD"] * 4,
            "pit_lane_open": [True] * 4,
            "behavior_action_probability": [0.2] * 4,
            "race_time_seconds": [90.0, 202.0, 310.0, 401.0],
            "is_pit_in_lap": [False, True, False, False],
            "is_pit_out_lap": [False, False, True, False],
            "observed_pit_loss_seconds": [float("nan"), 21.0, float("nan"), float("nan")],
        }
    )

    result = build_simulator_calibration_report(
        laps,
        simulator=LiveRaceSimulator(config=RaceSimulatorConfig(seed=3)),
    )

    assert result.metrics["one_step_lap_time"]["matching"] == "semi_markov_decision_transition"
    assert result.metrics["one_step_lap_time"]["replay_transition_count"] == 1
    assert len(result.rows) == 1
    assert result.rows[0]["transition_kind"] == "pit_stop_semi_markov"
    assert result.rows[0]["elapsed_laps"] == 3
    assert result.rows[0]["lap_number"] == 1
    assert result.rows[0]["next_lap_number"] == 4
    assert result.metrics["pit_loss"]["available"] is True
