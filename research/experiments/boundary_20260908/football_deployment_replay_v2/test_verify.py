"""Synthetic-only narrow metadata amendment and immutable failure exceptions."""
from copy import deepcopy
import json

import pytest

from . import verify as v


def save(path, value, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return {'path': str(path.relative_to(root)), 'sha256': v.original.sha(path), 'bytes': path.stat().st_size}


def test_adds_only_three_independently_derived_fields(tmp_path, monkeypatch):
    state = {'attack': {'A': 0.}, 'defense': {'A': 0.}, 'rho': 0.}
    artifact = save(tmp_path/'old_feature.json', {'state': state}, tmp_path)
    descriptors = [{'league': 'E0', 'saved_full_model': state, 'saved_full_lineage': {'fit_match_ids': ['a', 'b'], 'cutoff_utc': '2026-01-01'}, 'unchanged': [1, 2]}]
    before = deepcopy(descriptors)
    archive, fixtures, inventory = object(), object(), object()
    monkeypatch.setattr(v, '_ORIGINAL_REFERENCE_INPUTS', lambda *_: (archive, fixtures, descriptors, inventory))
    result = v.corrected_reference_inputs({'inputs': {'feature_artifacts': {'E0': artifact}}}, tmp_path, {artifact['path']: artifact['sha256']}, {})
    assert result[0] is archive and result[1] is fixtures and result[3] is inventory
    assert descriptors == before
    line = result[2][0]['saved_full_lineage']
    assert line['source_artifact'] == artifact['path'] and line['source_artifact_sha256'] == artifact['sha256']
    assert line['goal_state_sha256'] == v.original.ind.digest(state)
    for key in v.LINEAGE_KEYS:
        line.pop(key)
    assert result[2] == before


@pytest.mark.parametrize('kind', ['already_present', 'wrong_input_hash', 'changed_artifact'])
def test_metadata_amendment_does_not_relax_other_input_checks(tmp_path, monkeypatch, kind):
    artifact = save(tmp_path/'old_feature.json', {}, tmp_path)
    descriptor = {'league': 'E0', 'saved_full_model': {}, 'saved_full_lineage': {}}
    inputs = {artifact['path']: artifact['sha256']}
    if kind == 'already_present':
        descriptor['saved_full_lineage']['source_artifact'] = 'wrong'
    elif kind == 'wrong_input_hash':
        inputs[artifact['path']] = '0'*64
    else:
        (tmp_path/'old_feature.json').write_text('{ }')
    monkeypatch.setattr(v, '_ORIGINAL_REFERENCE_INPUTS', lambda *_: ([], [], [descriptor], []))
    with pytest.raises(ValueError):
        v.corrected_reference_inputs({'inputs': {'feature_artifacts': {'E0': artifact}}}, tmp_path, inputs, {})


def failure(tmp_path):
    out = tmp_path/'parent'; out.mkdir()
    record = save(out/'verification_failure.json', {'exception_type': 'ValueError', 'message': v.FAILURE_MESSAGE,
                                                   'advancement_allowed': False, 'failed_at_utc': '2026-09-08T00:00:00Z'}, tmp_path)
    return out, record


def test_exact_pinned_failure_remains_unchanged(tmp_path):
    out, record = failure(tmp_path)
    before = (out/'verification_failure.json').read_bytes()
    v.permit_pinned_failure(out, record, root=tmp_path)
    assert (out/'verification_failure.json').read_bytes() == before


@pytest.mark.parametrize('kind', ['another_marker', 'missing', 'changed_bytes', 'wrong_reason', 'wrong_type', 'truthy_flag', 'other_directory'])
def test_no_other_failure_or_mutation_is_permitted(tmp_path, kind):
    out, record = failure(tmp_path)
    path = out/'verification_failure.json'
    if kind == 'another_marker':
        save(out/'forecast_failure.json', {}, tmp_path)
    elif kind == 'missing':
        path.unlink()
    elif kind == 'changed_bytes':
        path.write_text(path.read_text()+' ')
    elif kind == 'other_directory':
        other = tmp_path/'other'; other.mkdir()
        record = save(other/'verification_failure.json', v.original.read(path), tmp_path)
    else:
        payload = v.original.read(path)
        payload.update({'message': 'Failed forecast'} if kind == 'wrong_reason' else {'exception_type': 'RuntimeError'} if kind == 'wrong_type' else {'advancement_allowed': 0})
        record = save(path, payload, tmp_path)
    with pytest.raises(ValueError):
        v.permit_pinned_failure(out, record, root=tmp_path)


def test_parent_replay_guards_run_and_functions_restore(tmp_path, monkeypatch):
    out, record = failure(tmp_path)
    old_inputs, old_failure = v.original.reference_inputs, v.original.check_failures
    monkeypatch.setattr(v.original, 'reject_operational_imports', lambda: None)
    def verify(parent, root):
        assert parent == out and root == tmp_path
        assert v.original.reference_inputs is v.corrected_reference_inputs
        v.original.check_failures(parent)
        return {'status': 'passed', 'refits': 0}
    monkeypatch.setattr(v.original, 'verify', verify)
    assert v.replay_parent(out, record, root=tmp_path) == {'status': 'passed', 'refits': 0}
    assert v.original.reference_inputs is old_inputs and v.original.check_failures is old_failure


def test_other_parent_replay_failure_is_never_swallowed(tmp_path, monkeypatch):
    out, record = failure(tmp_path)
    old_inputs, old_failure = v.original.reference_inputs, v.original.check_failures
    monkeypatch.setattr(v.original, 'reject_operational_imports', lambda: None)
    def broken(*args, **kwargs):
        raise ValueError('Actual matrix replay mismatch')
    monkeypatch.setattr(v.original, 'verify', broken)
    with pytest.raises(ValueError, match='Actual matrix'):
        v.replay_parent(out, record, root=tmp_path)
    assert v.original.reference_inputs is old_inputs and v.original.check_failures is old_failure


def synthetic_v2_closure(tmp_path, monkeypatch):
    parent, failure_record = failure(tmp_path)
    out = tmp_path/'corrective'; out.mkdir()
    here = tmp_path/'research/experiments/boundary_20260908/football_deployment_replay_v2'; here.mkdir(parents=True)
    old_verify = save(tmp_path/'old/verify.py', {'synthetic_source': 'verify'}, tmp_path)
    old_math = save(tmp_path/'old/independent.py', {'synthetic_source': 'math'}, tmp_path)
    proposal = save(here/'README.md', {'synthetic_proposal': True}, tmp_path)
    inputs = {'verification_failure.json': failure_record}
    for name in ('design_lock.json', 'data_lock.json', 'forecast_closure.json', 'evaluation.json', 'verification_attempt.json'):
        inputs[name] = save(parent/name, {'synthetic_closed': name}, tmp_path)
    spec = {'inputs': inputs, 'parent_out': str(parent.relative_to(tmp_path)), 'new_fits': 0, 'new_forecasts': 0,
            'new_calibrations': 0, 'numerical_scoring_or_gate_changes': False, 'preserve_original_terminal_attempt': True,
            'all_other_failures_fatal': True, 'only_correction': {'fields': list(v.LINEAGE_KEYS)},
            'permitted_failure': {'filename': 'verification_failure.json', 'exception_type': 'ValueError', 'message': v.FAILURE_MESSAGE},
            'original_verifier': old_verify, 'original_independent_arithmetic': old_math, 'proposal': proposal,
            'maximum_peak_rss_bytes': 2**40, 'maximum_stage_wall_seconds': 3600}
    save(here/'specification.json', spec, tmp_path)
    save(here/'verify.py', {'synthetic_source': 'v2'}, tmp_path)
    sources = v.sources(tmp_path)
    tests = save(out/'pre_fit_tests.json', {'source_files': sources, 'exit_code': 0}, tmp_path)
    save(out/'design_lock.json', {'sources': sources, 'pre_fit_tests': tests,
                                'review': {'approved_for_replay': True, 'source_files': sources},
                                'locked_at_utc': '2026-09-08T01:00:00Z', 'new_fits': 0, 'new_forecasts': 0}, tmp_path)
    monkeypatch.setattr(v.original, 'reject_operational_imports', lambda: None)
    def replay(path, record, root):
        assert path == parent and record == failure_record and root == tmp_path
        v.permit_pinned_failure(path, record, root=root)
        return {'status': 'passed', 'passes_all_gates': True, 'refits': 0}
    monkeypatch.setattr(v, 'replay_parent', replay)
    return out


def test_complete_separate_v2_closure_preserves_parent_failure(tmp_path, monkeypatch):
    out = synthetic_v2_closure(tmp_path, monkeypatch)
    result = v.verify(out, root=tmp_path)
    assert result['status'] == 'passed' and result['production_activated'] is False
    assert result['original_protocol_status'] == 'terminal_verification_failure_preserved'
    assert result['new_fits'] == result['new_forecasts'] == result['new_calibrations'] == 0


@pytest.mark.parametrize('kind', ['unreviewed', 'truthy_review', 'failed_tests', 'source_changed', 'late_failure', 'parent_changed'])
def test_corrective_closure_is_not_a_generic_bypass(tmp_path, monkeypatch, kind):
    out = synthetic_v2_closure(tmp_path, monkeypatch)
    design = v.original.read(out/'design_lock.json')
    if kind in ('unreviewed', 'truthy_review'):
        design['review']['approved_for_replay'] = False if kind == 'unreviewed' else 1
        save(out/'design_lock.json', design, tmp_path)
    elif kind == 'failed_tests':
        tests = v.original.read(tmp_path/design['pre_fit_tests']['path']); tests['exit_code'] = 1
        design['pre_fit_tests'] = save(tmp_path/design['pre_fit_tests']['path'], tests, tmp_path)
        save(out/'design_lock.json', design, tmp_path)
    elif kind == 'source_changed':
        (tmp_path/'old/verify.py').write_text('changed')
    elif kind == 'parent_changed':
        (tmp_path/'parent/evaluation.json').write_text('changed')
    else:
        def fail_late(*args, **kwargs):
            save(out/'unexpected_failure.json', {}, tmp_path)
            return {'status': 'passed', 'passes_all_gates': True}
        monkeypatch.setattr(v, 'replay_parent', fail_late)
    with pytest.raises(ValueError):
        v.verify(out, root=tmp_path)
