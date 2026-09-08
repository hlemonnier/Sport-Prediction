"""Fixed comparisons, fit budget, exact reuse and advancement contracts."""
from copy import deepcopy
from datetime import date, timedelta
import json

import pytest

from . import run


def specification():
    spec = deepcopy(run.read(run.SPEC))
    spec['uncertainty']['resamples'] = 40
    return spec


def forecasts(spec, phase='selection'):
    rows = []
    for league in (('E0',) if phase == 'selection' else ('E0', 'I1', 'SP1')):
        for season in ((2022, 2023) if phase == 'selection' else (2024, 2025)):
            for i in range(20):
                label = i % 3
                weak, strong = [0.2] * 3, [0.1] * 3
                weak[label], strong[label] = 0.6, 0.8
                rows.append({'match_id': f'{league}:{season}:{i}', 'league': league, 'season': season,
                    'day': (date(season, 8, 1) + timedelta(days=i * 7)).isoformat(), 'label': label,
                    'probabilities': {name: list(strong if name == run.CANDIDATE else weak)
                        for name in [run.CANDIDATE, *spec[phase]['references']]}})
    return rows


@pytest.mark.parametrize('phase,reference', [(phase, ref) for phase in ('selection', 'transfer')
    for ref in run.read(run.SPEC)[phase]['references']])
def test_every_reference_can_veto_advancement(phase, reference):
    spec = specification()
    rows = forecasts(spec, phase)
    passed = run.make_report(rows, spec, phase, run.CANDIDATE)
    assert passed['passes_all_gates'] is True
    for row in rows:
        row['probabilities'][reference] = list(row['probabilities'][run.CANDIDATE])
    failed = run.make_report(rows, spec, phase, run.CANDIDATE)
    assert failed['passes_all_gates'] is False
    assert failed['checks'][reference]['minimum_gain_met'] is False
    assert failed['checks'][reference]['negative_28day_upper'] is False
    assert failed['promotion'] is False


def test_bad_country_season_cannot_be_hidden_by_pooled_improvement():
    spec = specification()
    rows = forecasts(spec, 'transfer')
    for row in rows:
        if row['league'] == 'I1' and row['season'] == 2025:
            row['probabilities'][run.CANDIDATE] = [1 / 3] * 3
    report = run.make_report(rows, spec, 'transfer', run.CANDIDATE)
    check = report['checks'][run.CONTROL]
    assert check['minimum_gain_met'] is True
    assert check['no_country_season_nll_deterioration'] is False
    assert check['country_season_brier_nonworse'] is False
    assert report['passes_all_gates'] is False


@pytest.mark.parametrize('phase,delay,expected', [
    ('selection', 1, []), ('selection', 7, ['selection_delay1']),
    ('transfer', 1, ['selection_delay1', 'selection_delay7']),
    ('transfer', 7, ['selection_delay1', 'selection_delay7', 'transfer_delay1'])])
def test_stage_order(phase, delay, expected):
    assert run.required_predecessors(phase, delay) == expected


@pytest.mark.parametrize('phase,delay', [('other', 1), ('selection', 0), ('selection', True), ('selection', 1.0), ('transfer', 8)])
def test_no_extra_phase_or_delay(phase, delay):
    with pytest.raises(ValueError):
        run.required_predecessors(phase, delay)


def test_only_three_new_readout_fits_and_exact_primary_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'check_resources', lambda *_: {})
    original = tmp_path / 'original.json'
    original.write_bytes(b'{"model":{"tag":"original"},"closed_at_utc":"old"}\n')
    spec = {'inputs': {'primary_market_readout': {'path': original.name, 'sha256': run.sha(original), 'bytes': original.stat().st_size}}}
    calls = []
    def calibration(_archive, delay, cutoff, *, kind):
        calls.append((kind, delay, cutoff))
        return [0., 1., -1.], [0, 1, 2], {'kind': kind}
    monkeypatch.setattr(run.elo, 'calibration_rows', calibration)
    fits = []
    def fitted(x, y):
        fits.append((x, y))
        return {'tag': f'fit{len(fits)}'}
    monkeypatch.setattr(run.model, 'fit_ordered_logit', fitted)
    primary = run.readouts(tmp_path, object(), spec, 'selection', 1, '2022-08-05', 0.)
    assert primary['odds'] == {'tag': 'original'}
    assert run.readout_path(tmp_path, 'odds', 1).read_bytes() == original.read_bytes()
    assert calls == [('result', 1, '2022-08-05')]
    run.readouts(tmp_path, object(), spec, 'selection', 7, '2022-08-05', 0.)
    assert calls == [('result', 1, '2022-08-05'), ('odds', 7, '2022-08-05'), ('result', 7, '2022-08-05')]
    for delay in (1, 7):
        run.readouts(tmp_path, object(), spec, 'transfer', delay, '2022-08-05', 0.)
    assert len(fits) == 3
    with pytest.raises(FileExistsError):
        run.readouts(tmp_path, object(), spec, 'selection', 1, '2022-08-05', 0.)
    assert len(fits) == 3


def test_saved_reference_control_bits_and_order_survive(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    rows = [{'match_id': f'E0:2022:{i}', 'day': '2022-08-05',
             'probabilities': {'legacy': [0.5112345678901234, 0.2, 0.2887654321098766]}} for i in range(2)]
    old = deepcopy(rows)
    for row in old:
        row['probabilities'].update({name: [0.4, 0.3, 0.3] for name in run.OLD_CONTROLS})
    spec = {'inputs': {}, 'selection': {'rows': 2, 'ordered_original_match_ids_sha256': run.digest([r['match_id'] for r in rows])}}
    for key, payload in [('references_selection', rows), ('primary_forecasts', old)]:
        path = tmp_path / f'{key}.json'
        path.write_text(json.dumps(payload))
        spec['inputs'][key] = {'path': path.name, 'sha256': run.sha(path), 'bytes': path.stat().st_size}
    copied = run.reference_rows(spec, 'selection')
    assert [r['match_id'] for r in copied] == [r['match_id'] for r in rows]
    assert copied == old
    # Mutating the target-free legacy receipt cannot silently alter the original.
    changed = deepcopy(rows)
    changed[0]['probabilities']['legacy'] = [0.5, 0.2, 0.3]
    path = tmp_path / 'references_selection.json'
    path.write_text(json.dumps(changed))
    spec['inputs']['references_selection'].update(sha256=run.sha(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match='reference probabilities'):
        run.reference_rows(spec, 'selection')


def test_failed_predecessor_stops_before_readout_or_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'verify_prepared', lambda *_: None)
    visited = []
    def reject(_out, stage):
        visited.append(stage)
        raise ValueError('Failed fixed gate')
    monkeypatch.setattr(run, 'predecessor', reject)
    monkeypatch.setattr(run, 'fit_and_issue', lambda *_: pytest.fail('No fit/forecast after failed predecessor'))
    with pytest.raises(ValueError, match='Failed fixed gate'):
        run.execute(tmp_path, 'selection', 7)
    assert visited == ['selection_delay1']
    assert not list(tmp_path.iterdir())


def test_primary_issuance_reuses_candidate_and_closes_before_labels(tmp_path, monkeypatch):
    from ..football_past_market.test_data import raw, fixture
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run.parent, 'ROOT', tmp_path)
    monkeypatch.setattr(run.data, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'check_resources', lambda *_: {})
    class CurrentQuotePoison(dict):
        def get(self, key, *args):
            if key.startswith(('Avg', 'BbAv')):
                raise AssertionError('Current fixture odds must never be inspected')
            return super().get(key, *args)
    earlier = raw('2022-08-01', season=2021)
    active = raw('2022-08-05', season=2022, payload=CurrentQuotePoison(FTHG='2', FTAG='0', FTR='H'))
    unsupported = raw('2022-08-05', season=2022, home='New', away='Team',
                      payload=CurrentQuotePoison(FTHG='0', FTAG='0', FTR='D'))
    archive = run.data.Archive((earlier, active, unsupported))
    spec = specification()
    refs = []
    for item in (active, unsupported):
        row = fixture(item)
        row['probabilities'] = {name: [0.4, 0.3, 0.3] for name in spec['selection']['references'] if name != run.CONTROL}
        refs.append(row)
    previous = deepcopy(refs)
    for i, row in enumerate(previous):
        row['support'] = i == 0
        row['probabilities'][run.CANDIDATE] = [0.5112345678901234, 0.2, 0.2887654321098766] if i == 0 else [0.4, 0.3, 0.3]
    original = tmp_path / 'original_issued.json'
    run.save(original, previous)
    spec['inputs']['primary_forecasts'] = run.file_record(original)
    spec_path = tmp_path / 'spec.json'
    run.save(spec_path, spec)
    monkeypatch.setattr(run, 'SPEC', spec_path)
    monkeypatch.setattr(run, 'parent_specification', lambda _: {})
    monkeypatch.setattr(run.data, 'load_archive', lambda *_args, **_kwargs: archive)
    run.save(tmp_path / 'references_selection.json', refs)
    run.save(tmp_path / 'data_lock.json', {})
    fitted = {'kind': 'ordered_logit_v1', 'theta': [1.0, -0.5, 1.0], 'beta': 1.0,
              't1': -0.5, 't2': 0.5, 'gap': 1.0, 'floor': 1e-12}
    for kind in ('odds', 'result'):
        run.save(run.readout_path(tmp_path, kind, 1), {'model': fitted})
    # Odds is deliberately not a valid predictor: the primary must copy its
    # saved vector, never regenerate it from coefficients or rounded values.
    monkeypatch.setattr(run, 'readouts', lambda *_: {'odds': {'invalid': True}, 'result': fitted})
    monkeypatch.setattr(run.model, 'fit_strength', lambda *_: pytest.fail('Strength fits are forbidden'))
    monkeypatch.setattr(run.model, 'fit_ordered_logit', lambda *_: pytest.fail('Readout fit was already supplied'))
    attach = run.data.attach_labels
    observed = []
    def labels(rows, source, *, forecast_closure_path):
        closure = run.read(forecast_closure_path)
        saved = run.read(tmp_path / closure['forecasts']['path'])
        assert len(saved) == 2 and closure['labels_attached'] is False
        assert all('label' not in row for row in saved)
        observed.append(closure['forecasts']['sha256'])
        return attach(rows, source, forecast_closure_path=forecast_closure_path)
    monkeypatch.setattr(run.data, 'attach_labels', labels)
    scored = run.fit_and_issue(tmp_path, 'selection', 1)
    assert [row['label'] for row in scored] == [0, 1]
    assert len(observed) == 1
    assert [row['probabilities'][run.CANDIDATE] for row in scored] == [row['probabilities'][run.CANDIDATE] for row in previous]
    assert scored[1]['probabilities'][run.CONTROL] == refs[1]['probabilities'][run.INCUMBENT]
    state = run.read(tmp_path / 'selection_delay1_states.json')[0]
    assert state['support_training_ids'] == [earlier.match_id]
    assert state['supported'] == [True, False]
    assert set(state['ratings']) == {'result'}
    assert run.read(tmp_path / 'selection_delay1_issuance_lock.json')['optimizer_fits'] == 1
