"""Decision and execution boundaries that can invalidate historical evidence."""
from copy import deepcopy
from datetime import date, timedelta

import pytest

from . import run


def specification():
    spec = deepcopy(run.read(run.SPEC))
    spec['uncertainty']['resamples'] = 40
    return spec


def forecasts(spec, transfer=False):
    names = list(dict.fromkeys(spec['candidates'] + spec['selection']['references']))
    records = []
    countries = ('E0', 'I1', 'SP1') if transfer else ('E0',)
    for country in countries:
        for season in (2022, 2023):
            for i in range(20):
                label = i % 3
                baseline = [0.2] * 3
                baseline[label] = 0.6
                strong = [0.1] * 3
                strong[label] = 0.8
                records.append({'match_id': f'{country}:{season}:{i}', 'league': country, 'season': season,
                    'day': (date(season, 8, 1) + timedelta(days=i * 7)).isoformat(), 'label': label,
                    'probabilities': {name: list(strong if name in spec['candidates'] else baseline) for name in names}})
    return records


def test_ties_follow_declared_order_and_all_reference_checks_pass():
    spec = specification()
    result = run.make_report(forecasts(spec), spec, 'selection')
    assert result['selected_candidate'] == spec['candidates'][0]
    assert result['passes_all_gates']
    assert set(result['checks']) == set(spec['selection']['references'])


@pytest.mark.parametrize('reference', run.read(run.SPEC)['selection']['references'])
def test_every_strong_reference_can_veto_selection(reference):
    spec = specification()
    records = forecasts(spec)
    for row in records:
        row['probabilities'][reference] = list(row['probabilities'][spec['candidates'][0]])
    result = run.make_report(records, spec, 'selection')
    assert not result['passes_all_gates']
    assert not result['checks'][reference]['minimum_gain_met']
    assert not result['checks'][reference]['negative_28day_upper']


def test_locked_winner_is_not_reselected_during_sensitivity():
    spec = specification()
    rows = forecasts(spec)
    winner = spec['candidates'][1]
    for row in rows:
        row['probabilities'][winner] = list(row['probabilities'][run.INCUMBENT])
    report = run.make_report(rows, spec, 'selection', selected=winner)
    assert report['selected_candidate'] == winner
    assert list(report['comparisons']) == [winner]
    assert not report['passes_all_gates']


def test_transfer_cannot_hide_a_bad_country_season_in_pooled_gain():
    spec = specification()
    rows = forecasts(spec, transfer=True)
    winner = spec['candidates'][0]
    for row in rows:
        if row['league'] == 'I1' and row['season'] == 2023:
            row['probabilities'][winner] = [1 / 3] * 3
    report = run.make_report(rows, spec, 'transfer', winner)
    check = report['checks'][run.INCUMBENT]
    assert check['minimum_gain_met']
    assert not check['no_country_season_nll_deterioration']
    assert not check['country_season_brier_nonworse']
    assert not report['passes_all_gates']


def test_probability_floor_cannot_silently_repair_invalid_forecasts():
    with pytest.raises(ValueError):
        run.probability_vectors([[0.0, 0.4, 0.6]])


def test_outputs_are_exclusive_and_failures_are_preserved(tmp_path):
    run.save(tmp_path / 'design_lock.json', {})
    with pytest.raises(RuntimeError):
        with run.stage_attempt(tmp_path, 'selection_delay1'):
            raise RuntimeError('synthetic failure')
    assert run.read(tmp_path / 'selection_delay1_failure.json')['advancement_allowed'] is False
    with pytest.raises(FileExistsError):
        with run.stage_attempt(tmp_path, 'selection_delay1'):
            pytest.fail('an existing attempt must not be reused')


def test_resource_limit_rejects_overrun(monkeypatch):
    spec = specification()
    monkeypatch.setattr(run, 'peak_rss', lambda: spec['resources']['maximum_peak_rss_bytes'] + 1)
    with pytest.raises(RuntimeError, match='resource'):
        run.check_resources(run.time.monotonic(), spec)


def test_predecessor_failure_prevents_reading_or_fitting_transfer(tmp_path):
    run.save(tmp_path / 'selection_delay1_decision_lock.json', {'passes_all_gates': False})
    with pytest.raises(ValueError, match='did not pass'):
        run.predecessor(tmp_path, 'selection_delay1')


def test_predecessor_requires_independent_verification(tmp_path, monkeypatch):
    run.save(tmp_path / 'design_lock.json', {})
    run.save(tmp_path / 'selection_delay1_issuance_lock.json', {})
    run.save(tmp_path / 'selection_delay1.json', {'summary': {'passes_all_gates': True, 'selected_candidate': 'past_market'},
        'issuance_lock_sha256': run.sha(tmp_path / 'selection_delay1_issuance_lock.json')})
    run.save(tmp_path / 'selection_delay1_decision_lock.json', {'passes_all_gates': True, 'selected_candidate': 'past_market',
        'design_lock_sha256': run.sha(tmp_path / 'design_lock.json'), 'result_sha256': run.sha(tmp_path / 'selection_delay1.json')})
    monkeypatch.setattr(run, 'verify_phase_files', lambda *_: None)
    with pytest.raises(FileNotFoundError):
        run.predecessor(tmp_path, 'selection_delay1')
