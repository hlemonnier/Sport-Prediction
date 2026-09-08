"""Synthetic fixed-policy score and population contracts.

Suggested commit: test(football): guard paired deployment score gates.
"""
from copy import deepcopy
from datetime import date, timedelta
import json

import pytest

from . import scoring


def cohort():
    rows = []
    for country in ('E0', 'I1', 'SP1'):
        for season in (2024, 2025):
            for i in range(4):
                rows.append({'match_id': f'{country}:{season}:{i}', 'league': country, 'season': season,
                    'day': (date(season, 8, 1)+timedelta(days=i*35)).isoformat(), 'goals': [1, 0], 'label': 0,
                    'outputs': {scoring.REFERENCE: {'matrix': [[.2, .3], [.3, .2]], 'hda': [.3, .4, .3]},
                        scoring.CANDIDATE: {'matrix': [[.1, .2], [.6, .1]], 'hda': [.6, .2, .2]}}})
    spec = {'population': {'rows': len(rows), 'countries': ['E0', 'I1', 'SP1'], 'season_start_years': [2024, 2025],
        'rows_each_country_season': 4, 'ordered_original_match_ids_sha256': scoring.digest([r['match_id'] for r in rows])},
        'uncertainty': {'resamples': 200, 'seed': 20260908, 'blocks_calendar_days': [1, 7, 28], 'gate_block_days': 28},
        'acceptance': {'one_x_two': {'minimum_pooled_relative_nll_gain': .02}}}
    return rows, spec


def test_improvement_is_paired_and_same_draws_for_identical_loss_deltas():
    rows, spec = cohort()
    report = scoring.evaluate(rows, spec)
    assert report['passes_all_gates']
    for pair in report['paired'].values():
        assert pair['one_x_two'] == pair['joint_score']
    assert report['promotion'] is False


@pytest.mark.parametrize('score,cause', [([8, 0], 'out_of_support'), ([2, 0], 'zero_mass')])
def test_zero_scores_are_not_clipped_expanded_or_dropped(score, cause):
    rows, spec = cohort()
    for row in rows:
        row['goals'] = score
        if cause == 'zero_mass':
            for output in row['outputs'].values():
                output['matrix'].append([0., 0.])
    original = deepcopy(rows)
    report = scoring.evaluate(rows, spec)
    assert rows == original and report['rows'] == len(rows)
    assert not report['passes_all_gates']
    assert all(x['joint_cause'] == cause for x in report['nonfinite_losses'])
    assert report['paired']['28']['joint_score']['percentile_95_interval'] is None
    json.dumps(report, allow_nan=False)


def test_better_hda_can_have_worse_joint_forecast_and_is_vetoed():
    rows, spec = cohort()
    for row in rows:
        row['outputs'][scoring.CANDIDATE]['matrix'] = [[.1, .2, 0], [.1, .1, 0], [.5, 0, 0]]
    report = scoring.evaluate(rows, spec)
    assert report['checks']['one_x_two_minimum_gain']
    assert not report['checks']['pooled_joint_score_nonworse']
    assert not report['passes_all_gates']


@pytest.mark.parametrize('change', ['drop', 'duplicate', 'reorder', 'country', 'label', 'fractional', 'incoherent', 'extra_policy'])
def test_population_and_score_tampering_fails(change):
    rows, spec = cohort()
    if change == 'drop': rows.pop()
    elif change == 'duplicate': rows[0] = deepcopy(rows[1])
    elif change == 'reorder': rows.reverse()
    elif change == 'country': rows[0]['league'] = 'FR'
    elif change == 'label': rows[0]['label'] = 1
    elif change == 'fractional': rows[0]['goals'] = [1.1, 0]
    elif change == 'incoherent': rows[0]['outputs'][scoring.CANDIDATE]['hda'] = [.2, .2, .6]
    elif change == 'extra_policy': rows[0]['outputs']['extra'] = {}
    with pytest.raises(ValueError): scoring.evaluate(rows, spec)
