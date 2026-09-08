"""Independent replay contract tests; synthetic data, no historical fits/scores."""
from copy import deepcopy
import csv
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from . import data, independent, run, verify


def spec():
    result = deepcopy(run.read(run.SPEC))
    result['uncertainty']['resamples'] = 32
    return result


def scored(specification, *, transfer=False):
    names = list(dict.fromkeys(specification['candidates'] + specification['selection']['references']))
    rows = []
    for league in (('E0', 'I1', 'SP1') if transfer else ('E0',)):
        for season in ((2024, 2025) if transfer else (2022, 2023)):
            for i in range(9):
                label = i % 3
                p = [.09, .09, .09]; p[label] = .82
                q = [.19, .19, .19]; q[label] = .62
                rows.append({'match_id': f'{league}:{season}:{i}', 'league': league,
                             'season': season, 'day': (date(season, 8, 1)+timedelta(days=i*13)).isoformat(),
                             'label': label, 'probabilities': {name: list(p if name in specification['candidates'] else q) for name in names}})
    if transfer:
        next(row for row in rows if row['league'] == 'I1' and row['season'] == 2024)['match_id'] = 'I1:2024:Fiorentina:Inter'
    return rows


@pytest.mark.parametrize('actual,expected', [(True, 1), (False, 0), (1, True), (0, False),
                                            (np.bool_(True), True), (True, 1.)])
def test_close_requires_actual_booleans(actual, expected):
    with pytest.raises(ValueError):
        verify.close(actual, expected)


def test_close_preserves_exact_schema_and_finite_tolerance():
    verify.close({'x': [1., True, None]}, {'x': [1.+1e-12, True, None]})
    for actual, expected in [({'x': 1, 'extra': 0}, {'x': 1}), ([1, 2], [1]),
                             (float('nan'), float('nan')), (1.+1e-8, 1.)]:
        with pytest.raises(ValueError):
            verify.close(actual, expected)


@pytest.mark.parametrize('phase,selected', [('selection', None), ('selection', 'past_market_shot50'),
                                         ('transfer', 'past_market')])
def test_independent_full_report_replays_runner_math(phase, selected):
    specification = spec()
    rows = scored(specification, transfer=phase == 'transfer')
    report = run.make_report(rows, specification, phase, selected)
    verify.audit_report(rows, report, specification, phase,
                        candidate_inventory=None if selected is None else [selected])
    assert report['passes_all_gates'] is True


@pytest.mark.parametrize('field,value', [('rows', 1), ('population_sha256', '0'*64),
                                       ('passes_all_gates', 1), ('promotion', 0)])
def test_report_population_and_boolean_contract_cannot_be_forged(field, value):
    specification = spec(); rows = scored(specification)
    report = run.make_report(rows, specification, 'selection')
    report[field] = value
    with pytest.raises(ValueError):
        verify.audit_report(rows, report, specification, 'selection')


def test_changed_candidate_winner_and_interval_are_detected():
    specification = spec(); rows = scored(specification)
    report = run.make_report(rows, specification, 'selection')
    altered = deepcopy(report)
    altered['selected_candidate'] = specification['candidates'][1]
    with pytest.raises(ValueError, match='selection'):
        verify.audit_report(rows, altered, specification, 'selection')
    altered = deepcopy(report)
    altered['comparisons']['past_market'][run.INCUMBENT]['28']['percentile_95_interval'][1] += .01
    with pytest.raises(ValueError, match='numeric'):
        verify.audit_report(rows, altered, specification, 'selection')


def test_primary_cannot_omit_the_second_candidate():
    specification = spec(); rows = scored(specification)
    one_candidate = run.make_report(rows, specification, 'selection', 'past_market')
    with pytest.raises(ValueError, match='candidate inventory'):
        verify.audit_report(rows, one_candidate, specification, 'selection')


def test_interval_gate_uses_independent_sign_even_inside_numeric_tolerance(monkeypatch):
    specification = spec(); rows = scored(specification)
    report = run.make_report(rows, specification, 'selection')
    report['comparisons']['past_market'][run.INCUMBENT]['28']['percentile_95_interval'][1] = -1e-12
    original = independent.paired_uncertainty
    def replay(rows, candidate, reference, days, specification):
        result = original(rows, candidate, reference, days, specification)
        if candidate == 'past_market' and reference == run.INCUMBENT and days == 28:
            result['percentile_95_interval'][1] = 1e-12
        return result
    monkeypatch.setattr(independent, 'paired_uncertainty', replay)
    with pytest.raises(ValueError):
        verify.audit_report(rows, report, specification, 'selection')


def test_terminal_failure_prevents_any_source_or_result_read(tmp_path, monkeypatch):
    (tmp_path/'selection_delay1_failure.json').write_text('{}')
    monkeypatch.setattr(verify, 'read', lambda path: (_ for _ in ()).throw(AssertionError('read after failure')))
    with pytest.raises(ValueError, match='terminal failed'):
        verify.verify(tmp_path, 'selection', 1)


def test_country_season_deterioration_and_resumption_sensitivity_replay():
    specification = spec(); rows = scored(specification, transfer=True)
    for row in rows:
        if row['league'] == 'I1' and row['season'] == 2025:
            row['probabilities']['past_market'] = [1/3]*3
    report = run.make_report(rows, specification, 'transfer', 'past_market')
    verify.audit_report(rows, report, specification, 'transfer')
    assert report['passes_all_gates'] is False
    assert report['resumed_fixture_excluded']['rows'] == len(rows)-1
    report['resumed_fixture_excluded']['rows'] += 1
    with pytest.raises(ValueError):
        verify.audit_report(rows, report, specification, 'transfer')


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return {'path': str(path), 'sha256': verify.sha(path), 'bytes': path.stat().st_size}


def csv_fixture(tmp_path, league='E0', season=2022, days=('26/03/2022', '29/10/2022')):
    path = tmp_path/f'{league}_{season}_{season+1}.csv'
    columns = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'AvgH', 'AvgD', 'AvgA']
    rows = []
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for i, day in enumerate(days):
            row = dict(zip(columns, [league, day, f'A{i}', 'B', '1', '0', 'H', '2', '3', '4']))
            rows.append(row); writer.writerow(row)
    binding = {'path': str(path), 'sha256': verify.sha(path), 'bytes': path.stat().st_size}
    raw = tuple(data.RawFixture(f'{league}:{season}:{row["HomeTeam"]}:B', league, season,
                               row['HomeTeam'], 'B', data.datetime.strptime(row['Date'], '%d/%m/%Y').date(),
                               row, str(path), binding['sha256'], number, data.digest(row))
                for number, row in enumerate(rows, 2))
    return binding, data.Archive(raw)


@pytest.mark.parametrize('league', data.LEAGUES)
def test_csv_metadata_reconstruction_has_exact_dst_clocks_and_raw_hashes(tmp_path, league):
    binding, archive = csv_fixture(tmp_path, league)
    specification = {'inputs': {'cached_csv_files': [binding]}, 'quotes': {'verified_rows': 2}}
    expected = data.metadata_rows(archive)
    records = verify.raw_archive(specification, expected)
    assert [{k: v for k, v in row.items() if k != 'payload'} for row in records] == expected
    for row in records:
        assert row['source_row_hash'] == data.digest(row['payload'])
    altered = deepcopy(expected); altered[0]['availability_by_delay']['1'] = altered[0]['a0_utc']
    with pytest.raises(ValueError, match='metadata replay'):
        verify.raw_archive(specification, altered)


class Poison(dict):
    def get(self, *args, **kwargs):
        raise AssertionError('Future payload access')


@pytest.mark.parametrize('delay', [1, 7])
def test_elo_all_prefixes_match_atomic_operational_cursor(tmp_path, delay):
    binding, archive = csv_fixture(tmp_path, days=('26/03/2022', '26/03/2022', '29/03/2022'))
    # Make A0 appear twice at one availability time to detect sequential updates.
    original = list(archive.rows)
    second = original[1]
    payload = {**second.payload, 'HomeTeam': 'A0', 'AwayTeam': 'C', 'AvgH': '4', 'AvgA': '2'}
    original[1] = data.RawFixture('E0:2022:A0:C', 'E0', 2022, 'A0', 'C', second.day,
                                  payload, second.source_path, second.source_sha256, 3, data.digest(payload))
    archive = data.Archive(tuple(original))
    records = [{**data._metadata(row), 'payload': row.payload} for row in archive.rows]
    clocks = []
    for row in records:
        clock = independent.availability(row['day'], 'E0', delay)[1]
        clocks += [clock, clock+timedelta(microseconds=1)]
    later = {**records[0], 'match_id': 'E0:2022:Future:B', 'day': '2023-01-01', 'payload': Poison()}
    states = verify.elo_states(records+[later], reversed(clocks), delay)
    cursor = data.EloCursor(archive, delay)
    for cutoff in sorted(set(clocks)):
        cursor.query([], cutoff)
        assert states[independent.utc(cutoff)].keys() == cursor.ratings.keys()
        for team, rating in cursor.ratings.items():
            assert states[independent.utc(cutoff)][team] == pytest.approx(rating, abs=1e-11)
    assert verify.elo_states(list(reversed(records)), clocks, delay) == verify.elo_states(records, clocks, delay)


def reference_files(tmp_path):
    raw = {'match_id': 'E0:2022:A:B', 'league': 'E0', 'season': 2022, 'home': 'A', 'away': 'B',
           'day': '2022-08-05', 'forecast_cutoff_utc': '2022-08-04T23:00:00',
           'fit_cutoff_utc': '2022-07-31T23:00:00', 'fit_id': '0', 'result_available_at': '2022-08-05T23:00:00',
           'probabilities': {name: [.4123456789123456, .25, .3376543210876544] for name in data.INTERNAL}}
    xg = {'match_id': raw['match_id'], 'forecast_cutoff_utc': raw['forecast_cutoff_utc'],
          'fit_cutoff_utc': raw['fit_cutoff_utc'], 'probabilities': {'xg_add90': [.4, .3, .3]}}
    original = write_json(tmp_path/'original.json', {'predictions': [raw]})
    xb = write_json(tmp_path/'xg.json', [xg])
    specification = {'inputs': {'football_selection': original, 'xg_selection_issued': xb},
                     'selection': {'ordered_original_match_ids_sha256': verify.digest([raw['match_id']])}}
    reference = deepcopy(raw)
    reference['probabilities']['xg_add90_selection_safeguard'] = list(xg['probabilities']['xg_add90'])
    return specification, [reference]


def test_reference_bit_conservation_and_changed_clock_are_detected(tmp_path):
    specification, refs = reference_files(tmp_path)
    verify.reference_check(specification, 'selection', refs)
    changed = deepcopy(refs)
    changed[0]['probabilities'][run.INCUMBENT][0] = np.nextafter(changed[0]['probabilities'][run.INCUMBENT][0], 1.).item()
    with pytest.raises(ValueError, match='bits'):
        verify.reference_check(specification, 'selection', changed)
    changed = deepcopy(refs); changed[0]['fit_cutoff_utc'] = '2022-08-01T00:00:00'
    with pytest.raises(ValueError, match='clock'):
        verify.reference_check(specification, 'selection', changed)


def test_bound_rejects_source_mutation(tmp_path):
    binding = write_json(tmp_path/'source.json', {'synthetic': True})
    inventory = {}
    verify.bound(binding, inventory)
    assert inventory[binding['path']] == binding['sha256']
    Path(binding['path']).write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        verify.bound(binding, inventory)


def test_supported_blend_call_matches_keyword_only_independent_api():
    # Exercise the verifier's actual AST call sites; a positional supported flag
    # previously escaped pure-kernel tests and would crash the first real replay.
    import ast
    tree = ast.parse(Path(verify.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'fallback_or_blend']
    assert len(calls) == 2
    for call in calls:
        assert len(call.args) == 2
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        result = independent.fallback_or_blend([.6, .2, .2], [.4, .3, .3], **kwargs)
        np.testing.assert_array_equal(result, [.5, .25, .25])
