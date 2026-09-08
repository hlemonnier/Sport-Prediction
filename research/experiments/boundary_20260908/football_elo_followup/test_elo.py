"""Synthetic equations, strict chronology, delegation and readout membership.

Suggested commit: test(football): verify causal matched Elo calibration
"""
from copy import deepcopy
from datetime import date, timedelta
import json
import math

import numpy as np
import pytest

from . import elo
from research.experiments.boundary_20260908.football_past_market import data as parent


class Poison(dict):
    def get(self, *args, **kwargs):
        raise AssertionError('Unadmitted payload was inspected')


class OutcomePoison(dict):
    def get(self, key, *args):
        if key in ('FTHG', 'FTAG', 'FTR'):
            raise AssertionError('Outcome was inspected')
        return super().get(key, *args)


def raw(day='2019-08-01', home='A', away='B', league='E0', season=2019,
        odds=(2., 3., 4.), goals=(1, 0), payload=None):
    keys = ('BbAvH', 'BbAvD', 'BbAvA') if season < 2019 else ('AvgH', 'AvgD', 'AvgA')
    values = {key: str(value) for key, value in zip(keys, odds)}
    values.update(FTHG=str(goals[0]), FTAG=str(goals[1]),
                  FTR='H' if goals[0] > goals[1] else 'D' if goals[0] == goals[1] else 'A')
    return parent.RawFixture(f'{league}:{season}:{home}:{away}', league, season, home, away,
        date.fromisoformat(day), values if payload is None else payload,
        'synthetic.csv', 'a' * 64, 2, 'b' * 64)


def fixture(row):
    return parent._metadata(row)


def expected_home(rh=1000., ra=1000.):
    return 1 / (1 + 10 ** (-(rh - ra + 80) / 400))


def admitted_ids(cursor):
    return [row['match_id'] for batch in cursor.updates for row in batch['rows']]


@pytest.mark.parametrize('kind', [None, True, 'unknown', 'Result', 14])
def test_only_two_fixed_processes(kind):
    with pytest.raises(ValueError):
        elo.EloCursor(parent.Archive(()), 1, kind)
    with pytest.raises(ValueError):
        elo.calibration_rows(parent.Archive(()), 1, '2022-01-01', kind)


@pytest.mark.parametrize('delay', [0, 2, 8, True, 1.0, '1'])
@pytest.mark.parametrize('kind', elo.KINDS)
def test_delays_use_exact_parent_contract(delay, kind):
    with pytest.raises(ValueError):
        elo.EloCursor(parent.Archive(()), delay, kind)
    with pytest.raises(ValueError):
        elo.calibration_rows(parent.Archive(()), delay, '2022-01-01', kind)


def test_odds_query_and_calibration_are_direct_delegations(monkeypatch):
    query_output, calibration_output = {'exact': object()}, (object(), object(), object())
    calls = []
    def query(cursor, fixtures, cutoff):
        calls.append((cursor.archive, fixtures, cutoff))
        return query_output
    monkeypatch.setattr(parent.EloCursor, 'query', query)
    monkeypatch.setattr(parent, 'calibration_rows', lambda *args: calibration_output)
    archive = parent.Archive(())
    fixtures = []
    assert elo.EloCursor(archive, 1).query(fixtures, '2022-01-01') is query_output
    assert calls == [(archive, fixtures, '2022-01-01')]
    assert elo.calibration_rows(archive, 1, '2022-01-01') is calibration_output


@pytest.mark.parametrize('delay', [1, 7])
def test_actual_odds_cursor_and_calibration_bits_equal_parent(delay):
    rows = (raw('2018-08-01', season=2018), raw('2019-08-01', away='C'),
            raw('2019-08-01', home='D'), raw('2020-08-01', season=2020, goals=(1, 1)),
            raw('2021-08-01', season=2021, home='B', goals=(0, 1)))
    original = parent.EloCursor(parent.Archive(deepcopy(rows)), delay)
    delegated = elo.EloCursor(parent.Archive(deepcopy(rows)), delay, 'odds')
    for cutoff in ('2018-08-01', '2019-08-01', '2019-08-10', '2021-08-12'):
        assert parent.canonical(original.query([fixture(rows[1])], cutoff)) == parent.canonical(delegated.query([fixture(rows[1])], cutoff))
        assert parent.canonical(original.updates) == parent.canonical(delegated.updates)
    a = parent.calibration_rows(parent.Archive(deepcopy(rows)), delay, '2022-01-01')
    b = elo.calibration_rows(parent.Archive(deepcopy(rows)), delay, '2022-01-01', 'odds')
    assert parent.canonical(a) == parent.canonical(b)


@pytest.mark.parametrize('goals,score,outcome', [((1, 0), 1., 0), ((1, 1), .5, 1), ((0, 1), 0., 2)])
def test_result_one_step_k14_equation_and_no_home_advantage_in_readout(goals, score, outcome):
    row = raw(goals=goals)
    cursor = elo.EloCursor(parent.Archive((row,)), 1, 'result')
    delta = 14 * (score - expected_home())
    result = cursor.query([fixture(row)], '2019-08-04')
    assert cursor.ratings['E0', 'A'] == pytest.approx(1000 + delta, abs=1e-13)
    assert cursor.ratings['E0', 'B'] == pytest.approx(1000 - delta, abs=1e-13)
    assert result['x'] == pytest.approx([2 * delta / 400], abs=1e-15)
    update = cursor.updates[0]['rows'][0]
    assert update['outcome'] == outcome and update['observed_score'] == score
    assert update['a0_utc'] == '2019-08-01T23:00:00+00:00'
    assert update['goals'] == list(goals)
    assert result['provenance']['K'] == 14 and result['provenance']['kind'] == 'result'
    assert math.fsum(cursor.ratings.values()) == pytest.approx(2000.)


def test_shared_team_equal_clock_batch_is_atomic_and_order_invariant():
    a, b = raw(goals=(1, 0)), raw(away='C', goals=(0, 1))
    da, db = 14 * (1 - expected_home()), -14 * expected_home()
    outputs = []
    for rows in ((a, b), (b, a)):
        cursor = elo.EloCursor(parent.Archive(rows), 1, 'result')
        outputs.append(cursor.query([fixture(b)], '2019-08-04'))
        assert cursor.ratings['E0', 'A'] == pytest.approx(1000 + da + db)
        assert outputs[-1]['x'][0] == pytest.approx((da + 2 * db) / 400)
        assert all(row['rating_home_before'] == 1000 for row in cursor.updates[0]['rows'])
        assert math.fsum(cursor.ratings.values()) == pytest.approx(3000.)
    assert parent.canonical(outputs[0]) == parent.canonical(outputs[1])


def test_atomic_batch_does_not_commit_half_an_invalid_batch():
    first, bad = raw(away='B'), raw(away='C')
    bad.payload['FTR'] = 'D'
    cursor = elo.EloCursor(parent.Archive((first, bad)), 1, 'result')
    with pytest.raises(ValueError, match='disagrees'):
        cursor.query([fixture(first)], '2019-08-04')
    assert cursor.ratings == {} and cursor.updates == [] and cursor.index == 0


@pytest.mark.parametrize('league', parent.LEAGUES)
@pytest.mark.parametrize('day,delay,elapsed_hours', [('2022-03-26', 1, 23), ('2022-10-29', 1, 25),
                                                  ('2022-03-20', 7, 167), ('2022-10-23', 7, 169)])
def test_result_query_uses_strict_calendar_dst_availability(league, day, delay, elapsed_hours):
    row = raw(day, league=league, season=2021)
    a0 = parent.midnight(row.day + timedelta(days=1), league)
    clock = parent.availability(row.day, league, delay)
    assert (clock - a0).total_seconds() == elapsed_hours * 3600
    cursor = elo.EloCursor(parent.Archive((row,)), delay, 'result')
    assert cursor.query([fixture(row)], clock)['x'] == [0.]
    assert cursor.updates == []
    assert cursor.query([fixture(row)], clock + timedelta(microseconds=1))['x'] != [0.]


def test_future_current_equal_time_poison_never_read_even_with_cached_quote():
    past = raw('2020-01-01', home='Past', season=2019)
    rows = [past, raw('2020-01-02', home='Equal', payload=Poison()),
            raw('2020-01-04', home='Current', payload=Poison()),
            raw('2025-01-01', home='Future', season=2024, payload=Poison())]
    archive = parent.Archive(tuple(rows))
    for row in rows[1:]:
        archive._quotes[row.match_id] = {'invalid': 'prepopulated by a later query elsewhere'}
    cursor = elo.EloCursor(archive, 1, 'result')
    result = cursor.query([fixture(rows[1])], '2020-01-04T01:00:00+01:00')
    assert admitted_ids(cursor) == [past.match_id]
    assert result['x'] == [pytest.approx((1000 - cursor.ratings['E0', 'B']) / 400)]


def test_appending_future_rows_does_not_change_prior_snapshot_or_journal():
    past = raw('2020-01-01')
    extended = parent.Archive((past, raw('2025-01-01', home='Future', season=2024, payload=Poison())))
    one = elo.EloCursor(parent.Archive((deepcopy(past),)), 1, 'result')
    two = elo.EloCursor(extended, 1, 'result')
    assert parent.canonical(one.query([fixture(past)], '2020-01-05')) == parent.canonical(two.query([fixture(past)], '2020-01-05'))
    assert parent.canonical(one.updates) == parent.canonical(two.updates)


@pytest.mark.parametrize('invalid', [True, '', 'nan', 'inf', '0', '1', None])
def test_missing_quote_skips_result_payload_and_matches_odds_membership(invalid):
    valid = raw(home='Valid')
    missing = raw(home='Missing', payload=OutcomePoison(AvgH=invalid, AvgD='3', AvgA='4'))
    cursors = [elo.EloCursor(parent.Archive(deepcopy((valid, missing))), 1, kind) for kind in elo.KINDS]
    for cursor in cursors:
        cursor.query([fixture(valid)], '2019-08-04')
    assert admitted_ids(cursors[0]) == admitted_ids(cursors[1]) == [valid.match_id]
    assert cursors[0].missing == cursors[1].missing == [missing.match_id]


def test_odds_never_reads_outcomes_but_admitted_result_requires_them():
    base = raw()
    row = raw(payload=OutcomePoison(base.payload))
    assert elo.EloCursor(parent.Archive((row,)), 1, 'odds').query([fixture(row)], '2019-08-04')['x']
    with pytest.raises(AssertionError, match='Outcome'):
        elo.EloCursor(parent.Archive((row,)), 1, 'result').query([fixture(row)], '2019-08-04')


def test_quotes_affect_result_membership_but_not_its_valid_update_value():
    results = []
    for odds in ((2, 3, 4), (25, 8, 1.05)):
        row = raw(odds=odds)
        results.append(elo.EloCursor(parent.Archive((row,)), 1, 'result').query([fixture(row)], '2019-08-04'))
    assert results[0]['x'] == results[1]['x']


def test_unwindowed_history_league_isolation_neutral_entry_and_returning_team():
    old = raw('2017-01-01', season=2017)
    current = raw('2024-01-01', season=2023, home='C', away='A', goals=(0, 1))
    cursor = elo.EloCursor(parent.Archive((old, current)), 1, 'result')
    targets = [fixture(old), fixture(raw(league='I1')), fixture(raw(home='Future', away='Other'))]
    first = cursor.query(targets, '2023-01-01')
    assert first['x'][0] != 0 and first['x'][1:] == [0., 0.]
    assert len(cursor.ratings) == 2
    old_a = cursor.ratings['E0', 'A']
    cursor.query([fixture(current)], '2024-01-04')
    last = cursor.updates[-1]['rows'][0]
    assert last['rating_away_before'] == old_a and last['rating_home_before'] == 1000
    assert cursor.ratings['E0', 'A'] > old_a


def test_repeat_clock_is_idempotent_backward_clock_rejected_and_snapshot_immutable():
    a = raw('2019-08-01')
    b = raw('2019-09-01', home='C')
    cursor = elo.EloCursor(parent.Archive((a, b)), 1, 'result')
    first = cursor.query([fixture(a)], '2019-08-04')
    saved = deepcopy(first)
    assert first == cursor.query([fixture(a)], '2019-08-04')
    cursor.query([fixture(b)], '2019-09-04')
    assert first == saved
    with pytest.raises(ValueError, match='backward'):
        cursor.query([fixture(a)], '2019-08-04')


@pytest.mark.parametrize('delay', [1, 7])
def test_calibration_uses_original_issue_before_its_result_and_fixed_seasons(delay):
    warm = raw('2018-07-01', season=2018)
    a = raw('2019-08-01', away='C', goals=(0, 1))
    b = raw('2020-08-01', season=2020, away='D', goals=(1, 1))
    missing = raw('2021-02-01', season=2020, home='Missing', payload=OutcomePoison(AvgH='', AvgD='3', AvgA='4'))
    equal = raw('2021-12-31', season=2021, home='Equal', payload=Poison())
    future = raw('2025-01-01', season=2024, home='Future', payload=Poison())
    rows = (warm, a, b, missing, equal, future)
    cutoff = parent.availability(equal.day, equal.league, delay)
    x, y, details = elo.calibration_rows(parent.Archive(rows), delay, cutoff, 'result')
    oracle = elo.EloCursor(parent.Archive(deepcopy(rows)), delay, 'result')
    expected = [oracle.query([fixture(row)], parent.midnight(row.day, row.league))['x'][0] for row in (a, b)]
    assert x == expected and y == [2, 1]
    assert [r['match_id'] for r in details['rows']] == [a.match_id, b.match_id]
    assert details['excluded_missing_quote_ids'] == [missing.match_id]
    assert warm.match_id not in [r['match_id'] for r in details['rows']]
    odds = elo.calibration_rows(parent.Archive(deepcopy(rows)), delay, cutoff, 'odds')
    assert [r['match_id'] for r in odds[2]['rows']] == [r['match_id'] for r in details['rows']]
    assert odds[1] == y and odds[2]['excluded_missing_quote_ids'] == details['excluded_missing_quote_ids']
    changed = deepcopy(a)
    changed.payload.update(FTHG='4', FTAG='0', FTR='H')
    altered = elo.calibration_rows(parent.Archive((deepcopy(warm), changed)), delay, cutoff, 'result')
    assert altered[0][0] == x[0] and altered[1][0] == 0
    assert json.loads(json.dumps(details, allow_nan=False)) == details


def test_calibration_captures_x_before_current_payload_access(monkeypatch):
    row = raw('2019-08-01')
    events = []
    old_query, old_quote, old_outcome = elo.EloCursor.query, parent._quote, parent._outcome
    def query(self, fixtures, cutoff):
        events.append('query')
        return old_query(self, fixtures, cutoff)
    def quote(archive, received):
        events.append('quote')
        return old_quote(archive, received)
    def outcome(received):
        events.append('outcome')
        return old_outcome(received)
    monkeypatch.setattr(elo.EloCursor, 'query', query)
    monkeypatch.setattr(parent, '_quote', quote)
    monkeypatch.setattr(parent, '_outcome', outcome)
    x, y, _ = elo.calibration_rows(parent.Archive((row,)), 1, '2022-01-01', 'result')
    assert events == ['query', 'quote', 'outcome']
    assert x == [0.] and y == [0]


def test_nonfinite_extreme_expected_score_is_stable():
    row = raw()
    cursor = elo.EloCursor(parent.Archive((row,)), 1, 'result')
    cursor.ratings = {('E0', 'A'): 1e6, ('E0', 'B'): -1e6}
    result = cursor.query([fixture(row)], '2019-08-04')
    assert np.isfinite(result['x']).all()
    assert cursor.updates[0]['rows'][0]['delta'] == 0.
