"""Moment, shared-horizon and multi-step state regressions from the math audit."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import packages.f1.models.live_race.predict as live
from packages.f1.data.schemas.session import PredictionConfig
from packages.f1.models.live_race.action_space import StrategyAction
from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.mpc import SimulatorMPCStrategyPlanner
from packages.f1.models.live_race.pit_loss import PitLossConfig
from packages.f1.models.live_race.planner import (
    DeterministicStrategyTransitionModel, DeterministicTransitionConfig,
    SimulatorMPCConfig, SimulatorMPCPlanner,
)
from packages.f1.models.live_race.simulator import LiveRaceSimulator, RaceSimulatorConfig, SimulatorScenario
from packages.f1.models.live_race.simulator_calibration import one_step_lap_time_calibration
from packages.f1.models.live_race.sources import LiveSourceResult, _standardize_laps
from packages.f1.models.live_race.state import (
    BaselineModel,
    FilterConfig,
    FilterState,
    advance_sampled_state,
    predict_state,
    sample_filter_state,
)
from packages.f1.models.live_race.traffic import TrafficModelConfig


def test_sampled_trajectory_one_step_and_cumulative_moments_match_linear_gaussian_process():
    cfg = FilterConfig(phi=0.8, q_pace=0.15, q_deg=0.04, r_obs=0.2)
    state = FilterState(np.array([0.2, 0.03]), np.array([[0.4, 0.025], [0.025, 0.015]]))
    rng = np.random.default_rng(762)
    samples = sample_filter_state(state, rng, size=80000)
    totals = np.zeros(len(samples))
    a = np.array([[1., 1.], [0., cfg.phi]])
    # Independent augmented-state calculation for [x_t, sum_{j<=t} y_j].
    f = np.zeros((3, 3))
    f[:2, :2], f[2, :2], f[2, 2] = a, a[0], 1.
    g = np.array([[1., 0.], [0., 1.], [1., 0.]])
    q = np.diag([cfg.q_pace**2, cfg.q_deg**2])
    covariance = np.zeros((3, 3))
    covariance[:2, :2] = state.cov
    expected = np.r_[state.mean, 0.]
    for step in range(8):
        samples = advance_sampled_state(samples, cfg, rng)
        totals += samples[:, 0] + rng.normal(0., np.sqrt(cfg.r_obs), len(samples))
        expected = f @ expected
        covariance = f @ covariance @ f.T + g @ q @ g.T
        covariance[2, 2] += cfg.r_obs
        if step == 0:
            one_mean, one_cov = predict_state(state, cfg)
            np.testing.assert_allclose(samples.mean(axis=0), one_mean, atol=0.009)
            np.testing.assert_allclose(np.cov(samples.T), one_cov, rtol=0.025, atol=0.002)
    assert totals.mean() == pytest.approx(expected[2], abs=0.06)
    assert totals.var() == pytest.approx(covariance[2, 2], rel=0.025)


@pytest.fixture
def fixed_green_no_pit(monkeypatch):
    monkeypatch.setattr(live, "_sample_strategy_template", lambda **_: live.StrategyTemplate("hold", None, 0.))
    monkeypatch.setattr(live, "_advance_rollout_regime", lambda *args: "green")


def _snapshot(laps=(10, 10), times=(900., 910.)):
    return pd.DataFrame(dict(driver_id=["A", "B"], lap_last=laps, race_time_seconds=times,
                             compound=["MEDIUM"]*2, tyre_age=[0]*2))


def _forecast(snapshot, *, pace=(0., 0.), variance=1e-12, horizon=10, samples=50):
    cfg = FilterConfig(phi=0., q_pace=1e-9, q_deg=1e-9, r_obs=1e-9)
    states = {driver: FilterState(np.array([mean, 0.]), np.diag([variance, 0.]))
              for driver, mean in zip(["A", "B"], pace)}
    return live._mc_position_distribution(snapshot, states, BaselineModel({}, 90., 0.), cfg,
                                           horizon, 42, requested_samples=samples, max_mc_work=10000000)


def test_actual_order_monte_carlo_matches_constant_unknown_pace_probability(fixed_green_no_pit):
    result, _ = _forecast(_snapshot(), variance=1., samples=3000)
    expected = norm.cdf(10. / np.sqrt(2. * 10.**2))
    # Six binomial SEs; the old repeated-covariance process misses by ~0.12.
    assert result.set_index("driver_id").loc["A", "p_win_H"] == pytest.approx(expected, abs=0.045)


def test_asynchronous_crossings_are_compared_at_one_future_distance(fixed_green_no_pit):
    result, summary = _forecast(_snapshot((10, 9), (900., 819.)), pace=(0., -1.))
    assert result.set_index("driver_id").loc["B", "p_win_H"] == 1.
    assert result["forecast_target_lap"].tolist() == [20, 20]
    assert result["forecast_laps_from_last_observation"].tolist() == [10, 11]
    assert summary["work_per_sample"] == 21
    assert "common_race_distance" in summary["position_distribution_semantics"]


def test_true_lap_deficit_requires_extra_distance_to_catch_up(fixed_green_no_pit):
    # Both last crossings happened around the same clock time, a true lap gap.
    result, _ = _forecast(_snapshot((10, 9), (900., 902.)), pace=(0., -1.))
    assert result.set_index("driver_id").loc["A", "p_win_H"] == 1.


@pytest.mark.parametrize("schedule", [{"total_laps": [12, 12]}, {"remaining_laps": [2, 3]}])
def test_shared_horizon_is_capped_by_known_remaining_race_distance(fixed_green_no_pit, schedule):
    snapshot = _snapshot((10, 9), (900., 819.)).assign(**schedule)
    result, summary = _forecast(snapshot, pace=(0., -1.), horizon=20)
    assert result["forecast_laps_from_last_observation"].tolist() == [2, 3]
    assert summary["forecast_target_lap"] == 12
    assert summary["forecast_horizon_laps_effective"] == 2
    assert summary["scheduled_total_laps"] == 12
    assert summary["scheduled_horizon_known"] is True


def test_finished_race_has_no_phantom_rollout_laps(fixed_green_no_pit, monkeypatch):
    snapshot = _snapshot().assign(total_laps=[10, 10])
    monkeypatch.setattr(live, "advance_sampled_state", lambda *_: pytest.fail("terminal field advanced"))
    result, summary = _forecast(snapshot)
    assert result["forecast_laps_from_last_observation"].tolist() == [0, 0]
    assert summary["forecast_horizon_laps_effective"] == 0


def test_conflicting_schedule_evidence_is_rejected(fixed_green_no_pit):
    with pytest.raises(ValueError, match="inconsistent scheduled race distance"):
        _forecast(_snapshot().assign(total_laps=[12, 13]))


def _state_row(**overrides):
    row = dict(event_key=202601, driver_id="A", lap_number=10, total_laps=58,
               stint_id=1, compound="MEDIUM", tyre_age=10, used_compounds=("SOFT", "MEDIUM"),
               race_time_seconds=900., position=5, track_status="1", pace_penalty_mean=0.,
               deg_rate_mean=0., next_lap_mean=90., available_compounds=("SOFT", "MEDIUM", "HARD"),
               pit_lane_open=True, is_box_lap=False)
    return {**row, **overrides}


def _zero_effect_simulator():
    config = RaceSimulatorConfig(fuel_burn_lap_gain_seconds=0., conservative_lap_delta_seconds=0.,
        aggressive_lap_delta_seconds=0., compound_effects_seconds={}, tyre_cliff_quadratic_seconds=0.,
        traffic=TrafficModelConfig(max_loss_seconds=0.),
        pit_loss=PitLossConfig(green_loss_seconds=20., position_rejoin_sensitivity=0.))
    scenario = SimulatorScenario(noise_std_seconds=0., degradation_multiplier=0.)
    return LiveRaceSimulator(config=config, scenario=scenario)


def test_normal_mapping_path_keeps_pit_cost_out_of_all_later_baselines():
    simulator = _zero_effect_simulator()
    state = StrategyState.from_mapping(_state_row())
    transitions = simulator.simulate_action_sequence(state, [StrategyAction("pit_now", compound="HARD"),
        StrategyAction("stay_out"), StrategyAction("stay_out")])
    assert all(t.is_action_legal() for t in transitions)
    assert [t.reward_t.components["elapsed_seconds"] for t in transitions] == [110., 90., 90.]
    assert [t.reward_t.components["event_lap_baseline"] for t in transitions] == [90., 90., 90.]
    assert sum(t.reward_t.value for t in transitions) == -290.
    assert transitions[0].state_t1.next_lap_mean == 90.
    round_trip = StrategyState.from_mapping(transitions[0].state_t1.as_feature_dict())
    assert simulator.step(round_trip, StrategyAction("stay_out")).reward_t.value == -90.


def test_mapping_preserves_explicit_baseline_and_applies_scenario_offsets_once():
    simulator = _zero_effect_simulator()
    simulator = LiveRaceSimulator(config=replace(simulator.config, event_lap_slope_seconds=0.2),
        scenario=replace(simulator.scenario, baseline_offset_seconds=3.))
    row = _state_row(next_lap_mean=145., baseline_lap=90.)
    transitions = simulator.simulate_action_sequence(StrategyState.from_mapping(row), [StrategyAction("stay_out")]*3)
    np.testing.assert_allclose([t.reward_t.components["event_lap_baseline"] for t in transitions], [93., 93.2, 93.4])
    nested = StrategyState.from_mapping(_state_row(metadata={"event_lap_baseline_seconds": 88.}))
    assert simulator.step(nested, StrategyAction("stay_out")).reward_t.components["event_lap_baseline"] == 91.


def test_simulator_mpc_mapping_route_scores_one_time_pit_cost():
    simulator = _zero_effect_simulator()
    planner = SimulatorMPCPlanner(config=SimulatorMPCConfig(horizon_laps=3,
        simulator=simulator.config, downside_cvar_weight=0.), scenarios=[simulator.scenario])
    scored = planner.score_action_sequence(_state_row(), [StrategyAction("pit_now", compound="HARD"),
        StrategyAction("stay_out"), StrategyAction("stay_out")])
    assert scored.expected_time_seconds == 290.
    wrapper = SimulatorMPCStrategyPlanner(planner=planner)
    result = wrapper.replan(_state_row())
    assert result.value == -270.


def test_legacy_deterministic_dp_keeps_transient_costs_out_of_baseline():
    transition_model = DeterministicStrategyTransitionModel(config=DeterministicTransitionConfig(
        hard_lap_delta_seconds=0., medium_lap_delta_seconds=0., conservative_lap_delta_seconds=0.,
        # Isolate pit/baseline accounting even after compound-prior resets.
        conservative_deg_multiplier=0.,
        fuel_burn_lap_gain_seconds=0., tyre_cliff_quadratic_seconds=0.,
        default_pit_loss_seconds=20., green_pit_track_position_penalty_seconds=0.))
    state = StrategyState.from_mapping(_state_row())
    first = transition_model.step(state, StrategyAction("pit_now", compound="HARD"))
    second = transition_model.step(first.state_t1, StrategyAction("stay_out"))
    assert first.reward_t.value == -110.
    assert second.reward_t.value == -90.
    round_trip = StrategyState.from_mapping(first.state_t1.as_feature_dict())
    assert transition_model.step(round_trip, StrategyAction("stay_out")).reward_t.value == -90.


def test_semi_markov_pit_calibrator_counts_one_pit_cost_across_three_laps():
    rows = []
    for lap, stint, compound, pin, pout, elapsed in [
        (10, 1, "MEDIUM", False, False, 900.),
        (11, 1, "MEDIUM", True, False, 1010.),
        (12, 2, "HARD", False, True, 1100.),
        (13, 2, "HARD", False, False, 1190.),
    ]:
        rows.append(_state_row(lap_number=lap, stint_id=stint, compound=compound,
            is_pit_in_lap=pin, is_pit_out_lap=pout, is_box_lap=pin or pout,
            race_time_seconds=elapsed, timestamp=elapsed))
    metrics, calibrated = one_step_lap_time_calibration(pd.DataFrame(rows), simulator=_zero_effect_simulator())
    assert metrics["available"] is True
    assert len(calibrated) == 1
    assert calibrated[0]["elapsed_laps"] == 3
    assert calibrated[0]["predicted_pit_loss_seconds"] == 20.
    assert calibrated[0]["predicted_elapsed_seconds"] == calibrated[0]["actual_elapsed_seconds"] == 290.


def _run_trace(monkeypatch, observations):
    config = PredictionConfig(source="local", mode="race", year=2026, round_number=1,
        train_seasons=[], include_standings=False, cache_dir=None, meeting_name=None,
        country_name=None, weekends_dir=None, f1_live_horizon_laps=1)
    monkeypatch.setattr(live, "load_live_observations", lambda _: LiveSourceResult(observations, "local", []))
    monkeypatch.setattr(live, "_write_trace", lambda *args, **kwargs: {})
    monkeypatch.setattr(live, "evaluate_live_replay", lambda *args, **kwargs: {"available": False})
    return live.run_live_race_prediction(config)


def test_filter_resets_on_outlap_and_counts_every_completed_lap_for_tyre_age(monkeypatch):
    rows = []
    for lap, stint, compound, box, track in [
        (1, 1, "SOFT", False, "1"), (2, 1, "SOFT", False, "1"),
        (3, 1, "SOFT", True, "1"), (4, 2, "HARD", True, "1"),
        (5, 2, "HARD", False, "4"), (6, 2, "HARD", False, "1"),
    ]:
        rows.append(dict(event_key=202601, driver_id="A", lap_number=lap, stint_id=stint,
            compound=compound, is_box_lap=box, is_accurate=True, track_status=track,
            lap_time_seconds=90., race_time_seconds=90.*lap, timestamp=90.*lap))
    result = _run_trace(monkeypatch, pd.DataFrame(rows))
    assert result.trace["reset_applied"].tolist() == [True, False, False, True, False, False]
    assert result.trace["tyre_age"].tolist() == [1, 2, 3, 1, 2, 3]
    assert result.trace.iloc[3]["deg_rate_mean"] == pytest.approx(0.03 * 0.9)
    scrubbed = pd.DataFrame(rows).assign(tyre_age=[4, 5, 6, 8, 9, 10])
    assert _run_trace(monkeypatch, scrubbed).trace["tyre_age"].tolist() == [4, 5, 6, 8, 9, 10]


def test_scheduled_distance_survives_source_and_snapshot_without_inference(monkeypatch):
    raw = pd.DataFrame(dict(DriverNumber=["A", "B"], LapNumber=[10, 9], LapTime=[90., 90.],
        Time=[900., 819.], Compound=["MEDIUM"]*2, TrackStatus=["1"]*2, IsAccurate=[True]*2,
        scheduled_laps=[12, 12]))
    standardized = _standardize_laps(raw, event_key=202601, source_used="local")
    result = _run_trace(monkeypatch, standardized)
    assert result.snapshot["total_laps"].tolist() == [12., 12.]
    assert result.summary["scheduled_total_laps"] == 12
    no_schedule = _standardize_laps(raw.drop(columns="scheduled_laps"), event_key=202601, source_used="local")
    assert "total_laps" not in no_schedule
    assert _run_trace(monkeypatch, no_schedule).summary["scheduled_horizon_known"] is False
