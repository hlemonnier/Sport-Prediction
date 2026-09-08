"""Synthetic orchestration checks; no historical artifact or candidate fit."""
from copy import deepcopy

import numpy as np
import pytest

from . import data, run
from .test_data import fixture, quote


def test_all_four_fits_share_past_rows_and_current_labels_are_unread(monkeypatch):
    past = []
    for index in range(3):
        source = fixture(home=f'past{index}', league='SP1');source['label'] = index
        past.append(source)
    current = [fixture(day='2022-01-03', home='currentA'), fixture(day='2022-01-03', home='currentB')]
    for source in current:
        source.update(fit_cutoff_utc='2022-01-03T00:00:00', result_available_at='2022-01-04T00:00:00', label=object())
    issued = [data.issue_row(row, quote(row)) for row in past+current]
    calls = []
    class FakePool:
        def predict(self, market, prior=None): return np.asarray(market)
        def serialize(self): return {'synthetic': True}
    def fitted(m, p, y, family, internal):
        calls.append((deepcopy(m), deepcopy(y), family, internal));return FakePool()
    monkeypatch.setattr(run.model, 'fit', fitted)
    spec = {'selection': {'league': 'E0', 'seasons': [2022], 'expected_rows': 2}}
    predictions, fits = run.fit_and_issue(issued, data.unique(past+current), spec)
    assert len(calls) == 4 and all(call[:2] == calls[0][:2] for call in calls)
    assert calls[0][1] == [0, 1, 2]
    assert len(predictions) == 2 and len(fits) == 1
    assert set(fits[0]['training_ids']) == {row['match_id'] for row in past}
    assert all(len(row['probabilities']) == 13 and 'label' not in row for row in predictions)


def test_selection_checks_every_reference_and_does_not_advance_on_a_weaker_claim(monkeypatch):
    from research.experiments.boundary_20260908.football import run as inherited
    references = ['raw_market', 'calibrated_market', 'strong_internal']
    spec = {'candidates': ['pool_shared', 'pool_classwise'], 'references': references,
            'selection': {'minimum_relative_nll_gain_vs_every_reference': .005, 'seasons': [2022]},
            'uncertainty': {'blocks_calendar_days': [28]}}
    losses = {name: 1. for name in references};losses.update(pool_shared=.994, pool_classwise=.995)
    monkeypatch.setattr(inherited.b, 'metrics', lambda _rows, name: {'log_loss': losses[name]})
    monkeypatch.setattr(inherited, 'uncertainty', lambda *_: {'percentile_95_interval': [-.02, -.001]})
    rows = [{'match_id': 'synthetic', 'season': 2022}]
    report = run.selection_report(rows, spec)
    assert report['advances_to_later_evaluation'] and set(report['all_reference_gate_checks']) == set(references)
    losses['calibrated_market'] = .993
    report = run.selection_report(rows, spec)
    assert not report['advances_to_later_evaluation']
    assert report['decision'] == 'stop_no_new_transfer_or_promotion'
    losses['calibrated_market'] = 1.
    monkeypatch.setattr(inherited, 'uncertainty', lambda *_: {'percentile_95_interval': [-.02, 0.]})
    assert not run.selection_report(rows, spec)['advances_to_later_evaluation']


def test_no_preparation_without_the_reviewed_design_lock(tmp_path):
    with pytest.raises(FileNotFoundError): run.prepare(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_forecasts_are_closed_before_final_selection_label_attachment(tmp_path, monkeypatch):
    out = tmp_path/'output';out.mkdir();lane = tmp_path/'source';lane.mkdir()
    source = fixture();forecast = data.issue_row(source, quote(source))
    run.save(lane/'specification.json', {})
    run.save(out/'design_lock.json', {})
    run.save(out/'prepared_inputs.json', [forecast])
    run.save(out/'incumbent_training_reconstruction.json', {})
    run.save(out/'data_lock.json', {'design_lock_sha256': data.sha(out/'design_lock.json'),
        'prepared_inputs_sha256': data.sha(out/'prepared_inputs.json'),
        'incumbent_reconstruction_sha256': data.sha(out/'incumbent_training_reconstruction.json')})
    monkeypatch.setattr(run, 'HERE', lane)
    monkeypatch.setattr(run, 'verify_lock', lambda *_: None)
    monkeypatch.setattr(data, 'load_feature_cache', lambda *_: [source])
    monkeypatch.setattr(run, 'fit_and_issue', lambda *_: ([forecast], []))
    original = data.label_for
    seen = []
    def guarded(row, sources):
        lock = run.read(out/'selection_issuance_lock.json')
        assert lock['issued_sha256'] == data.sha(out/'selection_issued.json')
        assert lock['current_row_labels_attached'] is False
        assert all('label' not in r for r in run.read(out/'selection_issued.json'))
        seen.append(row['match_id']);return original(row, sources)
    monkeypatch.setattr(data, 'label_for', guarded)
    monkeypatch.setattr(run, 'selection_report', lambda *_: {'selected_candidate': 'pool_shared', 'advances_to_later_evaluation': False})
    run.select(out)
    assert seen == [source['match_id']]
    assert run.read(out/'selection_lock.json')['advances_to_later_evaluation'] is False
    with pytest.raises(FileExistsError): run.select(out)
