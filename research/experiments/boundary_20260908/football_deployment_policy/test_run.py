"""Synthetic execution closure and failed-attempt contracts.

Suggested commit: test(football): protect immutable deployment assessment stages.
"""
import json

import pytest

from . import run


def test_save_is_exclusive_and_standard_json(tmp_path):
    path = tmp_path / 'saved.json'
    run.save(path, {'value': 2})
    with pytest.raises(FileExistsError): run.save(path, {'value': 3})
    assert json.loads(path.read_text()) == {'value': 2}
    with pytest.raises(ValueError): run.save(tmp_path / 'nonfinite.json', {'value': float('inf')})


def test_failed_attempt_is_terminal_and_preserved(tmp_path):
    with pytest.raises(RuntimeError, match='solver'):
        with run.attempt(tmp_path, 'forecast'):
            raise RuntimeError('solver failed')
    failure = run.read(tmp_path / 'forecast_failure.json')
    assert failure['advancement_allowed'] is False
    with pytest.raises(ValueError, match='terminal'):
        with run.attempt(tmp_path, 'score'): pass
    assert not (tmp_path / 'score_attempt.json').exists()


def test_completed_attempt_cannot_be_repeated(tmp_path):
    with run.attempt(tmp_path, 'prepare'): pass
    with pytest.raises(FileExistsError):
        with run.attempt(tmp_path, 'prepare'): pass


def test_resource_limits_fail_explicitly(monkeypatch):
    monkeypatch.setattr(run.time, 'monotonic', lambda: 12.)
    spec = {'resources': {'maximum_stage_wall_seconds': 10, 'maximum_peak_rss_bytes': 10**15}}
    with pytest.raises(RuntimeError): run.resources(0., spec)


def test_modified_bound_artifact_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    path = tmp_path / 'artifact.json'
    run.save(path, {'closed': True})
    record = run.file_record(path)
    assert run.bound(record) == path
    path.write_text('{"closed":false}')
    with pytest.raises(ValueError, match='changed'): run.bound(record)


def test_failure_arriving_during_scoring_prevents_evaluation_closure(tmp_path, monkeypatch):
    """A terminal marker must be checked again after the scoring callback."""
    from types import SimpleNamespace

    monkeypatch.setattr(run, 'ROOT', tmp_path)
    inputs = SimpleNamespace(spec={})
    design_checks = []

    def verify_design(out, *, inputs=None):
        design_checks.append(out)
        run.refuse_failure(out)
        return {}, inputs or frozen_inputs

    frozen_inputs = inputs
    monkeypatch.setattr(run, 'verify_design', verify_design)
    monkeypatch.setattr(run, 'verify_prepared', lambda out, inputs: {})
    monkeypatch.setattr(run.data, 'attach_labels', lambda *args, **kwargs: [])
    monkeypatch.setattr(run, 'resources', lambda *args: {'elapsed_seconds': 0., 'peak_rss_bytes': 1})
    for name, value in [('design_lock.json', {}), ('data_lock.json', {}), ('forecasts.json', [])]:
        run.save(tmp_path / name, value)
    run.save(tmp_path / 'forecast_closure.json', {
        'design_lock_sha256': run.sha(tmp_path / 'design_lock.json'),
        'data_lock_sha256': run.sha(tmp_path / 'data_lock.json'),
        'forecasts': run.file_record(tmp_path / 'forecasts.json'), 'blocks': [],
        'rows': 0, 'labels_attached': False,
    })

    def scoring_callback(rows, spec):
        run.save(tmp_path / 'late_failure.json', {'advancement_allowed': False})
        return {'metrics': {}, 'checks': {'synthetic': True}, 'passes_all_gates': True}

    monkeypatch.setattr(run.scoring, 'evaluate', scoring_callback)
    with pytest.raises(ValueError, match='terminal'):
        run.score(tmp_path)
    assert len(design_checks) == 2
    assert (tmp_path / 'scored_rows.json').exists()
    assert (tmp_path / 'late_failure.json').exists()
    assert run.read(tmp_path / 'score_failure.json')['advancement_allowed'] is False
    assert not (tmp_path / 'evaluation.json').exists()
