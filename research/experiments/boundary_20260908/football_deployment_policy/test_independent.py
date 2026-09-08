"""Synthetic independent-kernel and fixed-policy gate contracts; no data reads."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math

import numpy as np
import pytest

from . import independent as ind


def identity():
    return {'schema': 'football_calibrator_state_v1', 'method': 'identity',
            'class_functions': [{'kind': 'identity'} for _ in range(3)],
            'nonidentity_class_floor': 1e-12, 'platt_input_clip': [1e-12, 1-1e-12]}


def goal_state():
    return {'schema': 'football_dc_state_v1', 'attack': {'A': .1, 'B': -.1},
            'defense': {'A': .05, 'B': -.05}, 'home_intercept': math.log(1.4),
            'away_intercept': math.log(1.1), 'rho': -.07, 'diagnostics': {}}


@pytest.mark.parametrize('rates', [(.05, .05), (.05, 6.), (6., .05), (1.35, 1.05), (6., 6.), (100., 100.)])
@pytest.mark.parametrize('position', [0., .5, 1.])
def test_full_finite_grid_and_refinement_match_canonical(rates, position):
    from packages.football.mrp.score_distribution import build_score_distribution
    lh, la = rates
    lo, hi = max(-1/lh, -1/la), min(1., 1/(lh*la))
    rho = lo + (hi-lo)*position
    actual, expected = ind.raw_joint(lh, la, rho), build_score_distribution(lh, la, rho)
    np.testing.assert_allclose(actual['matrix'], expected.matrix, rtol=0, atol=2e-16)
    for selected in ([.2, .6, .2], [.999999, .0000005, .0000005], [0., 1., 0.]):
        actual, expected_final = ind.reconcile(ind.raw_joint(lh, la, rho), selected), expected.reconcile(tuple(selected))
        assert actual['dimensions'] == [len(expected_final.matrix), len(expected_final.matrix[0])]
        np.testing.assert_allclose(actual['matrix'], expected_final.matrix, rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual['final_HDA'], expected_final.outcome_probabilities, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual['expected_goals'], expected_final.expected_goals, rtol=0, atol=1e-10)
        assert actual['mode'][:2] == list(expected_final.most_likely_scoreline[:2])
        assert actual['tail_probability_bound'] <= ind.TAIL


def test_saved_parameter_replay_and_neutral_team():
    from packages.football.mrp import deployment
    from packages.football.mrp.data import FixtureRecord
    from packages.football.mrp.training import ProbabilityCalibrator, _identity
    state = goal_state()
    native = deployment.restore_dc(state)
    for home, away in [('A', 'B'), ('unseen', 'B'), ('A', 'other')]:
        fixture = FixtureRecord('m', datetime(2025, 1, 1), 2024, 'epl', None, home, away)
        expected = deployment.joint_output(native, fixture, ProbabilityCalibrator('identity', [_identity]*3))
        actual = ind.replay_output(state, identity(), home, away)
        for key in actual:
            if key == 'reconciled':
                assert actual[key] is expected[key]
            else:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)


@pytest.mark.parametrize('policy', ['off', 'platt', 'isotonic'])
def test_actual_synthetic_calibrator_export_matches_native(policy):
    from packages.football.mrp import deployment, training
    rng = np.random.default_rng(142)
    probabilities = rng.dirichlet([2, 3, 4], size=300)
    labels = [int(rng.choice(3, p=p)) for p in probabilities]
    native = training.fit_probability_calibrator_from_rows(list(map(tuple, probabilities)), labels, [], policy=policy)
    assert native.method == ('identity' if policy == 'off' else policy)
    state = deployment.serialize_calibrator(native)
    for p in [*[list(x) for x in probabilities[:25]], [1., 0., 0.], [0., 0., 1.], [.01, .01, .01]]:
        np.testing.assert_allclose(ind.apply_calibrator(state, p), native.apply(tuple(p)), rtol=0, atol=1e-12)


@pytest.mark.parametrize('rates', [(0, 1, 0), (-1, 1, 0), (101, 1, 0), (1, 1, 1.1), (2, 2, -.6), (math.nan, 1, 0), (1, math.inf, 0)])
def test_invalid_kernel_fails(rates):
    with pytest.raises(ValueError):
        ind.raw_joint(*rates)


@pytest.mark.parametrize('mutation', [lambda s: s.update(method='other'), lambda s: s.update(class_functions=[]),
                                     lambda s: s.update(nonidentity_class_floor=1e-6), lambda s: s.update(platt_input_clip=[0, 1]),
                                     lambda s: s['class_functions'][0].update(kind='platt')])
def test_invalid_calibrator_state_fails(mutation):
    state = identity(); mutation(state)
    with pytest.raises((ValueError, KeyError)):
        ind.apply_calibrator(state, [.5, .25, .25])


def test_one_knot_isotonic_zero_floor_and_invalid_knots():
    state = identity(); state['method'] = 'isotonic'
    item = {'kind': 'isotonic', 'params': {'out_of_bounds': 'clip'},
            'X_thresholds_': [.5], 'y_thresholds_': [0.], 'X_min_': .5, 'X_max_': .5}
    state['class_functions'][0] = item
    actual = ind.apply_calibrator(state, [1., 0., 0.])
    np.testing.assert_allclose(actual, [1/3]*3, rtol=0, atol=1e-15)
    item['X_thresholds_'] = [.5, .4]; item['y_thresholds_'] = [.1, .2]
    with pytest.raises(ValueError):
        ind.apply_calibrator(state, [.5, .25, .25])


def test_joint_distribution_does_not_accept_target_argument():
    state = goal_state()
    first = ind.replay_output(state, identity(), 'A', 'B')
    for goals in ([0, 0], [4, 2], [999, 999]):
        before = deepcopy(first)
        ind.score_loss(first['matrix'], goals)
        assert first == before
    assert ind.replay_output(state, identity(), 'A', 'B') == first
    with pytest.raises(TypeError):
        ind.raw_joint(1., 1., 0., goals=[99, 99])


def test_zero_outside_and_invalid_score_have_distinct_semantics():
    matrix = [[.25, .25], [.5, 0.]]
    assert ind.score_loss(matrix, [1, 1])['cause'] == 'zero_mass'
    assert ind.score_loss(matrix, [2, 0])['cause'] == 'out_of_support'
    assert ind.score_loss(matrix, [1, 0])['nll'] == -math.log(.5)
    for goals in ([True, 0], [-1, 0], [1., 0], [0]):
        with pytest.raises(ValueError):
            ind.score_loss(matrix, goals)


def test_moment_mode_and_low_score_tau_math():
    matrix = [[.2, .2], [.2, .2], [.2, 0.]]
    summary = ind.matrix_summary(matrix)
    assert summary['mode'] == [0, 0, .2]
    np.testing.assert_allclose(summary['expected_goals'], [.8, .4], rtol=0, atol=1e-15)
    plain, corrected = ind.raw_joint(1.4, 1.1, 0), ind.raw_joint(1.4, 1.1, -.1)
    np.testing.assert_allclose(corrected['expected_goals'], [1.4, 1.1], rtol=0, atol=1e-11)
    assert corrected['matrix'][0][0] > plain['matrix'][0][0]
    assert corrected['matrix'][0][1] < plain['matrix'][0][1]


def test_history_clock_window_future_payload_poison_and_dst():
    class Poison(dict):
        def __getitem__(self, key):
            if key == 'payload':
                raise AssertionError('Future payload was accessed')
            return super().__getitem__(key)
    cutoff = ind.local_midnight('2024-04-01', 'E0')
    rows = [Poison(match_id=str(i), league='E0', day=day, season=2023) for i, day in enumerate(['2024-03-30', '2024-03-31', '2024-04-01', '2024-04-02'])]
    assert [r['match_id'] for r in ind.history_membership(rows, cutoff, 'E0')] == ['0', '1']
    assert (ind.local_midnight('2024-04-01', 'E0')-ind.local_midnight('2024-03-31', 'E0')).total_seconds() == 23*3600
    first = (cutoff.astimezone(ind.ZoneInfo('Europe/London')).date()-timedelta(days=1095))
    rows = [dict(match_id=str(i), league='E0', day=d.isoformat(), season=d.year) for i, d in enumerate([first-timedelta(days=1), first])]
    assert [r['match_id'] for r in ind.history_membership(rows, cutoff, 'E0')] == ['1']


@pytest.mark.parametrize('count', [0, 20, 74, 100, 210, 331])
def test_partition_membership_exact_native_and_boundary_purge(count):
    from packages.football.mrp.data import MatchRecord
    from packages.football.mrp.protocol import chronological_populations
    from dataclasses import asdict
    rows = []
    for i in range(count):
        d = datetime(2023, 1, 1)+timedelta(days=i//3)
        rows.append(MatchRecord(str(i), d, 2022, 'epl', None, 'A', 'B', 1, 0, None, None,
                                result_available_at=d+timedelta(days=1 if i % 9 else 7)))
    expected = chronological_populations(rows)
    actual = ind.chronological_partitions([asdict(r) for r in rows])
    assert actual['excluded_boundary_ids'] == expected.excluded_ids
    for key in ('fit', 'calibration', 'selection', 'test'):
        assert actual[key] == [r.match_id for r in getattr(expected, key)]


def scoring_fixture():
    rows = []
    for league in ('E0', 'SP1', 'I1'):
        for season in (2024, 2025):
            for i in range(3):
                rows.append({'match_id': f'{league}:{season}:{i}', 'league': league, 'season': season,
                             'day': f'{season}-08-{1+i:02d}', 'goals': [1, 0], 'label': 0,
                             'outputs': {ind.REFERENCE: {'hda': [.5, .25, .25], 'matrix': [[.25, .25], [.5, 0.]]},
                                         ind.CANDIDATE: {'hda': [.7, .15, .15], 'matrix': [[.15, .15], [.7, 0.]]}}})
    spec = {'population': {'rows': len(rows), 'countries': ['E0', 'SP1', 'I1'], 'season_start_years': [2024, 2025],
                           'rows_each_country_season': 3, 'ordered_original_match_ids_sha256': ind.digest([r['match_id'] for r in rows])},
            'uncertainty': {'blocks_calendar_days': [1, 7, 28], 'gate_block_days': 28, 'resamples': 50, 'seed': 20260908},
            'acceptance': {'one_x_two': {'minimum_pooled_relative_nll_gain': .02}}}
    return rows, spec


def test_all_fixed_gates_and_shared_score_draws():
    rows, spec = scoring_fixture()
    report = ind.evaluate(rows, spec)
    assert report['passes_all_gates'] is True and report['promotion'] is False
    for value in report['paired'].values():
        assert value['one_x_two'] == value['joint_score']
    assert report['metrics'][ind.REFERENCE]['log_loss'] == -math.log(.5)
    json.dumps(report, allow_nan=False)


def test_joint_veto_even_when_hda_improves():
    rows, spec = scoring_fixture()
    for row in rows:
        row['outputs'][ind.CANDIDATE]['matrix'] = [[.15, .15, 0.], [.1, 0., 0.], [.6, 0., 0.]]
    report = ind.evaluate(rows, spec)
    assert report['checks']['one_x_two_minimum_gain'] is True
    assert report['checks']['pooled_joint_score_nonworse'] is False
    assert report['passes_all_gates'] is False


@pytest.mark.parametrize('goals', [[1, 1], [99, 0]])
def test_nonfinite_joint_preserves_population_and_has_no_fake_interval(goals):
    rows, spec = scoring_fixture()
    rows[0]['goals'] = goals; rows[0]['label'] = 1 if goals[0] == goals[1] else 0
    report = ind.evaluate(rows, spec)
    assert report['rows'] == 18 and len(report['nonfinite_losses']) == 2
    assert report['metrics'][ind.CANDIDATE]['joint_score_log_loss'] is None
    assert report['paired']['28']['joint_score']['percentile_95_interval'] is None
    assert report['passes_all_gates'] is False
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('mutation', [lambda rows: rows.pop(), lambda rows: rows.reverse(),
                                     lambda rows: rows[0].update(match_id=rows[1]['match_id']),
                                     lambda rows: rows[0].update(label=True),
                                     lambda rows: rows[0]['outputs'][ind.CANDIDATE].update(hda=[.8, .1, .1])])
def test_population_alignment_and_output_guards(mutation):
    rows, spec = scoring_fixture(); mutation(rows)
    with pytest.raises(ValueError):
        ind.evaluate(rows, spec)
