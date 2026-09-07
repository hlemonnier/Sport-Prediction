"""Independent tyre-clock and compound-state transition regressions."""
from dataclasses import replace

import pytest

from packages.f1.models.live_race.action_space import StrategyAction
from packages.f1.models.live_race.environment import StrategyState
from packages.f1.models.live_race.pit_loss import PitLossConfig
from packages.f1.models.live_race.planner import DeterministicStrategyTransitionModel, DeterministicTransitionConfig
from packages.f1.models.live_race.simulator import LiveRaceSimulator, RaceSimulatorConfig, SimulatorScenario
from packages.f1.models.live_race.state import compound_deg_prior
from packages.f1.models.live_race.traffic import TrafficModelConfig


def _state():
    return StrategyState.from_mapping(dict(event_key=202601,driver_id='A',lap_number=10,total_laps=58,
        stint_id=1,compound='SOFT',tyre_age=10,used_compounds=('SOFT',),race_time_seconds=900.,position=5,
        track_status='1',pace_penalty_mean=0.,deg_rate_mean=.5,next_lap_mean=90.,
        available_compounds=('SOFT','MEDIUM','HARD'),pit_lane_open=True,is_box_lap=False,
        circuit_tyre_degradation=0.,metadata={'event_lap_baseline_seconds':90.}))


@pytest.fixture(params=['simulator','deterministic'])
def model(request):
    if request.param=='simulator':
        return LiveRaceSimulator(config=RaceSimulatorConfig(fuel_burn_lap_gain_seconds=0.,
            compound_effects_seconds={},conservative_lap_delta_seconds=0.,
            conservative_degradation_multiplier=1.,tyre_cliff_quadratic_seconds=0.,
            traffic=TrafficModelConfig(max_loss_seconds=0.),
            pit_loss=PitLossConfig(green_loss_seconds=20.,position_rejoin_sensitivity=0.)),
            scenario=SimulatorScenario(noise_std_seconds=0.))
    return DeterministicStrategyTransitionModel(config=DeterministicTransitionConfig(
        fuel_burn_lap_gain_seconds=0.,soft_lap_delta_seconds=0.,medium_lap_delta_seconds=0.,
        hard_lap_delta_seconds=0.,conservative_lap_delta_seconds=0.,conservative_deg_multiplier=1.,
        tyre_cliff_quadratic_seconds=0.,default_pit_loss_seconds=20.,
        green_pit_track_position_penalty_seconds=0.,pit_next_commitment_penalty_seconds=0.))


@pytest.mark.parametrize('compound',['HARD','MEDIUM'])
def test_pit_now_completes_one_new_tyre_lap_and_counts_degradation_once(model,compound):
    state=_state()
    first=model.step(state,StrategyAction('pit_now',compound=compound))
    second=model.step(first.state_t1,StrategyAction('stay_out'))
    third=model.step(second.state_t1,StrategyAction('stay_out'))
    assert all(t.is_action_legal() for t in (first,second,third))
    assert [t.state_t1.lap_number for t in (first,second,third)]==[11,12,13]
    assert [t.state_t1.tyre_age for t in (first,second,third)]==[1,2,3]
    rate=compound_deg_prior(compound)
    assert first.state_t1.deg_rate_mean==rate
    assert first.reward_t.value==pytest.approx(-110.)
    # Circuit multiplier at supplied tyre severity0 is0.75. Every full lap
    # after the pit contributes exactly one further age increment.
    assert second.reward_t.value==pytest.approx(-(90.+rate*.75))
    assert third.reward_t.value==pytest.approx(-(90.+2*rate*.75))
    round_trip=StrategyState.from_mapping(first.state_t1.as_feature_dict())
    assert model.step(round_trip,StrategyAction('stay_out')).reward_t.value==pytest.approx(second.reward_t.value)


def test_pit_next_is_a_commitment_then_a_separate_full_new_tyre_lap(model):
    original=_state()
    commitment=model.step(original,StrategyAction('pit_next_lap',compound='HARD'))
    assert commitment.is_action_legal()
    assert commitment.state_t1.compound=='SOFT'
    assert commitment.state_t1.tyre_age==11
    assert commitment.state_t1.deg_rate_mean==.5
    assert commitment.state_t1.used_compounds==('SOFT',)
    assert commitment.state_t1.metadata['forced_pit_next_compound']=='HARD'
    assert commitment.reward_t.value==pytest.approx(-(90.+10*.5*.75))
    # No mount at the commitment boundary: the documented forced PIT_NOW
    # executes on the next transition and then drives a full new-compound lap.
    executed=model.step(commitment.state_t1,StrategyAction('pit_now',compound='HARD'))
    assert executed.is_action_legal()
    assert executed.state_t1.lap_number==12
    assert executed.state_t1.tyre_age==1
    assert executed.state_t1.deg_rate_mean==compound_deg_prior('HARD')
    assert 'HARD' in executed.state_t1.used_compounds
    assert 'forced_pit_next_compound' not in executed.state_t1.metadata
    assert executed.reward_t.value==pytest.approx(-110.)


def test_pit_never_carries_old_compound_degradation_into_new_stint(model):
    high=_state()
    low=replace(high,deg_rate_mean=.001)
    action=StrategyAction('pit_now',compound='HARD')
    high_after=model.step(high,action).state_t1
    low_after=model.step(low,action).state_t1
    assert high_after.deg_rate_mean==low_after.deg_rate_mean==compound_deg_prior('HARD')
    assert model.step(high_after,StrategyAction('stay_out')).reward_t.value==pytest.approx(
        model.step(low_after,StrategyAction('stay_out')).reward_t.value)
