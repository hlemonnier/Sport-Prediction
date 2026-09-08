from copy import deepcopy

import pytest

from . import data


def fixture(day='2022-01-01', home='A', league='E0'):
    return {'match_id': f'{league}:2022:{home}:B', 'league': league, 'season': 2022,
            'home': home, 'away': 'B', 'day': day,
            'forecast_cutoff_utc': day+'T00:00:00',
            'fit_cutoff_utc': '2022-01-01T00:00:00', 'fit_id': 'frozen',
            'result_available_at': '2022-01-02T00:00:00', 'label': 0,
            'probabilities': {name: [.51, .26, .23] for name in data.INTERNAL_REFERENCES}}


def quote(row):
    return {'match_id': row['match_id'], 'day': row['day'], 'avg': ['2', '3', '4'],
            'b365': ['2.1', '3.1', '3.8'], 'quote_observed_at': None,
            'receipt_certified': False, 'product': 'provider_preclosing_snapshot'}


def test_issuance_is_target_free_and_preserves_incumbent_vectors():
    original = fixture();before = deepcopy(original)
    row = data.issue_row(original, quote(original))
    assert 'label' not in row and 'goals' not in row and row['labels_attached'] is False
    assert row['quote']['quote_observed_at'] is None
    assert row['market_ready'] and len(row['probabilities']) == 9
    assert all(row['probabilities'][name] == original['probabilities'][name] for name in data.INTERNAL_REFERENCES)
    assert original == before


def test_closing_odds_or_outcomes_cannot_change_issuance():
    source = fixture();a = data.issue_row(source, quote(source))
    source.update(label=2, goals=[1000, 1000], AvgCH=100000)
    assert data.issue_row(source, quote(source)) == a


def test_missing_snapshot_falls_back_for_every_market_reference():
    source = fixture();q = quote(source);q['avg'][0] = None
    row = data.issue_row(source, q)
    assert not row['market_ready']
    assert all(row['probabilities'][name] == row['probabilities'][data.INTERNAL]
               for name in ('avg_normalized', 'avg_power', 'b365_normalized', 'b365_power', 'arithmetic_half'))


@pytest.mark.parametrize('field', ['match_id', 'day'])
def test_odds_join_rejects_wrong_fixture_or_date(field):
    source = fixture();q = quote(source);q[field] = 'wrong'
    with pytest.raises(ValueError, match='identity/date'): data.issue_row(source, q)


def test_frozen_vectors_are_reused_exactly_and_disagreement_fails():
    source = fixture();row = deepcopy(source);row['probabilities'].pop(data.INTERNAL)
    reused = data.reuse_incumbent(row, source)
    assert reused['probabilities'][data.INTERNAL] == source['probabilities'][data.INTERNAL]
    source['probabilities']['dc365_elo50'] = [.5, .25, .25]
    with pytest.raises(ValueError, match='vector mismatch'): data.reuse_incumbent(row, source)


def test_training_uses_parsed_utc_strict_cutoffs_before_inspecting_labels_or_prices():
    source = fixture();row = data.issue_row(source, quote(source))
    class Poison(dict):
        def __getitem__(self, key):
            if key in ('market_ready', 'label', 'probabilities'): raise AssertionError('Unavailable value read')
            return super().__getitem__(key)
    future = Poison({**row, 'match_id': 'future', 'forecast_cutoff_utc': '2022-01-03T00:00:00',
                     'result_available_at': '2022-01-04T00:00:00'})
    equal = Poison({**row, 'match_id': 'equal', 'result_available_at': '2022-01-02T02:00:00+02:00'})
    before = {**row, 'result_available_at': '2022-01-02T00:59:59+01:00'}
    source['result_available_at'] = before['result_available_at']
    accepted, labels, missing = data.training_rows([before, equal, future], {row['match_id']: source}, '2022-01-02T00:00:00Z')
    assert accepted == [before] and labels == [0] and missing == []


def test_common_training_exclusion_and_label_identity_validation():
    source = fixture();row = data.issue_row(source, quote(source));row['market_ready'] = False
    accepted, labels, missing = data.training_rows([row], {}, '2022-01-03T00:00:00')
    assert accepted == [] and labels == [] and missing == [row['match_id']]
    row['market_ready'] = True;source['day'] = 'wrong'
    with pytest.raises(ValueError, match='identity/clock'): data.training_rows([row], {row['match_id']: source}, '2022-01-03T00:00:00')


def test_duplicate_ids_cannot_double_training_weight():
    source = fixture();row = data.issue_row(source, quote(source))
    with pytest.raises(ValueError, match='Duplicate training'):
        data.training_rows([row, row], {row['match_id']: source}, '2022-01-03T00:00:00')


def test_future_append_and_result_perturbation_leave_training_membership_unchanged():
    source = fixture();row = data.issue_row(source, quote(source))
    before = data.training_rows([row], {row['match_id']: source}, '2022-01-03T00:00:00')
    future = {**row, 'match_id': 'future', 'forecast_cutoff_utc': '2100-01-01T00:00:00',
              'result_available_at': '2100-01-02T00:00:00', 'market_ready': False}
    assert data.training_rows([row, future], {row['match_id']: source, 'future': {'label': object()}}, '2022-01-03T00:00:00') == before
