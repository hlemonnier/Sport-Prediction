"""Independent runner regressions using synthetic files and mocked fit kernels.

No historical inputs, predictions, labels or optimizer fits are read or run.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from . import run


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return record(path)


def record(path):
    return {'path': str(path), 'sha256': run.sha(path), 'bytes': path.stat().st_size}


def protocol():
    return deepcopy(run.read(Path(run.__file__).with_name('specification.json')))


def synthetic_references(spec):
    names = [n for n in spec['selection']['references'] if n not in (*run.OLD_CONTROLS, run.CONTROL)]
    rows = []
    for home in ('A', 'New'):
        rows.append({'match_id': f'E0:2022:{home}:B', 'league': 'E0', 'season': 2022,
                     'home': home, 'away': 'B', 'day': '2022-08-05',
                     'forecast_cutoff_utc': '2022-08-04T23:00:00',
                     'fit_cutoff_utc': '2022-07-31T23:00:00', 'fit_id': 'old',
                     'result_available_at': '2022-08-05T23:00:00',
                     'probabilities': {name: [.4123456789123456, .25, .3376543210876544] for name in names}})
    return rows


def closed_parent(tmp_path):
    spec = protocol()
    refs = synthetic_references(spec)
    old = deepcopy(refs)
    for i, row in enumerate(old):
        row['support'] = not bool(i)
        row['probabilities'][run.CANDIDATE] = [.51, .29, .20] if i == 0 else list(row['probabilities'][run.INCUMBENT])
        row['probabilities']['outcome_control'] = [.45, .30, .25] if i == 0 else list(row['probabilities'][run.INCUMBENT])
        row['probabilities']['outcome_control_shot50'] = [.4311728394561728, .275, .2938271605438272] if i == 0 else list(row['probabilities'][run.INCUMBENT])
    spec['inputs']['references_selection'] = write(tmp_path/'parent_refs.json', refs)
    spec['inputs']['primary_forecasts'] = write(tmp_path/'parent_forecasts.json', old)
    # Noncanonical whitespace deliberately tests byte reuse instead of reserialization.
    readout = tmp_path/'parent_readout.json'
    readout.write_text('{ "model" : { "kind" : "odds" }, "old_time" : "kept" }\n\n')
    spec['inputs']['primary_market_readout'] = record(readout)
    spec['selection']['rows'] = len(refs)
    spec['selection']['ordered_original_match_ids_sha256'] = run.digest([r['match_id'] for r in refs])
    return spec, refs, old


def test_reference_extraction_retains_old_control_bits_without_changing_parent(tmp_path):
    spec, refs, old = closed_parent(tmp_path)
    before = Path(spec['inputs']['primary_forecasts']['path']).read_bytes()
    extracted = run.reference_rows(spec, 'selection')
    assert [r['match_id'] for r in extracted] == [r['match_id'] for r in refs]
    for row, previous in zip(extracted, old):
        for name in run.OLD_CONTROLS:
            assert np.asarray(row['probabilities'][name]).tobytes() == np.asarray(previous['probabilities'][name]).tobytes()
        assert run.CANDIDATE not in row['probabilities']
    assert Path(spec['inputs']['primary_forecasts']['path']).read_bytes() == before


@pytest.mark.parametrize('change', ['clock', 'vector', 'missing_row', 'duplicate_row', 'label'])
def test_reference_contract_rejects_coherently_rehashed_wrong_metadata(tmp_path, change):
    spec, refs, old = closed_parent(tmp_path)
    if change == 'clock': old[0]['forecast_cutoff_utc'] = '2022-08-05T00:00:00'
    if change == 'vector': old[0]['probabilities'][run.INCUMBENT][0] = np.nextafter(old[0]['probabilities'][run.INCUMBENT][0], 1.).item()
    if change == 'missing_row': old.pop()
    if change == 'duplicate_row': old.append(deepcopy(old[0]))
    if change == 'label':
        refs[0]['label'] = 0
        old[0]['label'] = 0
    spec['inputs']['primary_forecasts'] = write(tmp_path/'parent_forecasts.json', old)
    spec['inputs']['references_selection'] = write(tmp_path/'parent_refs.json', refs)
    with pytest.raises(ValueError):
        run.reference_rows(spec, 'selection')


def runner_fixture(tmp_path, monkeypatch):
    spec, _, old = closed_parent(tmp_path)
    extracted = run.reference_rows(spec, 'selection')
    specpath = tmp_path/'spec.json'; write(specpath, spec)
    out = tmp_path/'execution'; out.mkdir()
    write(out/'references_selection.json', extracted)
    write(out/'data_lock.json', {'synthetic': True})
    monkeypatch.setattr(run, 'SPEC', specpath)
    monkeypatch.setattr(run, 'file_record', record)
    monkeypatch.setattr(run, 'check_resources', lambda *_: {'synthetic': True})
    monkeypatch.setattr(run, 'parent_specification', lambda *_: {'synthetic': True})
    monkeypatch.setattr(run.data, 'load_archive', lambda *_a, **_kw: object())
    monkeypatch.setattr(run.data, 'validate_reference_metadata', lambda *_: None)
    monkeypatch.setattr(run.data, 'quote_rows', lambda *_: ([{'match_id': 'old_quote'}], {}))
    monkeypatch.setattr(run.data, 'support_for', lambda rows, _: [r['home'] != 'New' for r in rows])
    calls = {'fits': [], 'queries': [], 'labels': 0}
    def calibration(_archive, delay, cutoff, *, kind):
        return [0., .2, -.2], [0, 1, 2], {'kind': kind, 'extra_days': delay, 'calibration_cutoff_utc': cutoff}
    monkeypatch.setattr(run.elo, 'calibration_rows', calibration)
    def fit(x, y):
        calls['fits'].append((list(x), list(y)))
        return {'kind': 'fitted', 'serial': len(calls['fits'])}
    monkeypatch.setattr(run.model, 'fit_ordered_logit', fit)
    monkeypatch.setattr(run.model, 'fit_strength', lambda *_: pytest.fail('Forbidden strength fit'))
    class Cursor:
        def __init__(self, _archive, delay, *, kind):
            self.kind = kind
        def query(self, rows, cutoff):
            calls['queries'].append(self.kind)
            return {'x': [0.]*len(rows), 'provenance': {'kind': self.kind}}
    monkeypatch.setattr(run.elo, 'EloCursor', Cursor)
    def predict(model, x):
        if model['kind'] == 'odds':
            pytest.fail('Primary market candidate must not be reconstructed')
        return np.tile([.3, .3, .4], (len(x), 1))
    monkeypatch.setattr(run.model, 'predict_ordered_logit', predict)
    def attach(rows, _archive, *, forecast_closure_path):
        # This is the only external scoring-label access in this synthetic run.
        closure = run.read(forecast_closure_path)
        assert closure['labels_attached'] is False
        assert run.sha(closure['forecasts']['path']) == closure['forecasts']['sha256']
        assert run.read(closure['forecasts']['path']) == rows
        assert (out/'selection_delay1_states.json').exists()
        assert set(closure['readouts']) == {'odds', 'result'}
        assert all(Path(item['path']).exists() for item in closure['readouts'].values())
        calls['labels'] += 1
        return [{**r, 'label': 0} for r in rows]
    monkeypatch.setattr(run.data, 'attach_labels', attach)
    return out, spec, old, calls


def test_primary_full_issuance_reuses_exact_candidate_and_closes_before_labels(tmp_path, monkeypatch):
    out, spec, old, calls = runner_fixture(tmp_path, monkeypatch)
    rows = run.fit_and_issue(out, 'selection', 1)
    assert calls == {'fits': [([0., .2, -.2], [0, 1, 2])], 'queries': ['result'], 'labels': 1}
    assert [r['match_id'] for r in rows] == [r['match_id'] for r in old]
    for row, previous in zip(rows, old):
        for name in (run.CANDIDATE, *run.OLD_CONTROLS):
            assert np.asarray(row['probabilities'][name]).tobytes() == np.asarray(previous['probabilities'][name]).tobytes()
    assert rows[1]['probabilities'][run.CONTROL] == rows[1]['probabilities'][run.INCUMBENT]
    assert run.read(out/'selection_delay1_issuance_lock.json')['optimizer_fits'] == 1
    assert run.readout_path(out, 'odds', 1).read_bytes() == Path(spec['inputs']['primary_market_readout']['path']).read_bytes()


def test_all_stages_need_only_three_readout_fits_and_transfer_never_refits(tmp_path, monkeypatch):
    out, spec, _, calls = runner_fixture(tmp_path, monkeypatch)
    run.readouts(out, object(), spec, 'selection', 1, '2022-08-04T23:00:00', 0)
    assert len(calls['fits']) == 1
    run.readouts(out, object(), spec, 'selection', 7, '2022-08-04T23:00:00', 0)
    assert len(calls['fits']) == 3
    for delay in (1, 7):
        run.readouts(out, object(), spec, 'transfer', delay, '2022-08-04T23:00:00', 0)
    assert len(calls['fits']) == 3


@pytest.mark.parametrize('bad_support', [0, 1, 'true', None])
def test_primary_support_requires_exact_boolean_parent_contract(tmp_path, monkeypatch, bad_support):
    out, spec, old, _ = runner_fixture(tmp_path, monkeypatch)
    old[0]['support'] = bad_support
    spec['inputs']['primary_forecasts'] = write(tmp_path/'parent_forecasts.json', old)
    write(Path(run.SPEC), spec)
    with pytest.raises(ValueError, match='support changed'):
        run.fit_and_issue(out, 'selection', 1)


@pytest.mark.parametrize('phase,delay,expected', [('selection',1,[]), ('selection',7,['selection_delay1']),
    ('transfer',1,['selection_delay1','selection_delay7']),
    ('transfer',7,['selection_delay1','selection_delay7','transfer_delay1'])])
def test_exact_conditional_stage_order(phase, delay, expected):
    assert run.required_predecessors(phase, delay) == expected


@pytest.mark.parametrize('delay', [True, False, 1., 7., '1', 0, 8])
def test_stage_delay_rejects_bool_and_numeric_aliases(delay):
    with pytest.raises(ValueError):
        run.required_predecessors('selection', delay)


def test_terminal_failure_blocks_before_data_or_predecessor_reads(tmp_path, monkeypatch):
    (tmp_path/'prepare_failure.json').write_text('{}')
    monkeypatch.setattr(run, 'read', lambda *_: pytest.fail('Read after terminal failure'))
    with pytest.raises(ValueError, match='failed stage'):
        run.verify_design(tmp_path)
    with pytest.raises(ValueError, match='failed execution'):
        run.predecessor(tmp_path, 'selection_delay1')


def test_bound_input_rejects_changed_bytes_even_if_valid_json(tmp_path):
    item = write(tmp_path/'input.json', {'x': 1})
    Path(item['path']).write_text('{"x": 2}')
    with pytest.raises(ValueError, match='Bound input changed'):
        run.bound(item)


def input_graph(tmp_path, monkeypatch, parent_gate=False):
    keys = ('primary_forecasts', 'primary_market_readout', 'parent_primary_issuance_lock',
            'references_selection', 'references_transfer')
    inputs = {key: write(tmp_path/f'{key}.json', {'synthetic': key}) for key in keys}
    inputs['parent_specification'] = write(tmp_path/'parent_spec.json', {})
    inputs['parent_primary_result'] = write(tmp_path/'parent_result.json', {'passes_all_gates': False})
    inputs['parent_design_lock'] = write(tmp_path/'parent_design.json', {'sources': {}, 'inputs': {}})
    receipt = {'status': 'passed', 'passes_all_gates': parent_gate,
               'result_sha256': inputs['parent_primary_result']['sha256'],
               'bindings': {inputs[key]['path']: inputs[key]['sha256'] for key in keys}}
    inputs['parent_primary_verification'] = write(tmp_path/'parent_verification.json', receipt)
    monkeypatch.setattr(run.parent, 'input_bindings', lambda _: {})
    return {'inputs': inputs}, receipt


def test_verified_failed_parent_is_valid_reuse_not_a_passing_new_predecessor(tmp_path, monkeypatch):
    specification, _ = input_graph(tmp_path, monkeypatch)
    closed = run.input_bindings(specification)
    assert all(closed[item['path']] == item['sha256'] for item in specification['inputs'].values())


@pytest.mark.parametrize('wrong', [0, True, None])
def test_parent_failure_status_must_remain_exact_false(tmp_path, monkeypatch, wrong):
    specification, _ = input_graph(tmp_path, monkeypatch, parent_gate=wrong)
    with pytest.raises(ValueError, match='failed-family'):
        run.input_bindings(specification)


def test_fresh_rehash_cannot_replace_parent_verified_readout(tmp_path, monkeypatch):
    specification, _ = input_graph(tmp_path, monkeypatch)
    path = Path(specification['inputs']['primary_market_readout']['path'])
    specification['inputs']['primary_market_readout'] = write(path, {'different': 'readout'})
    with pytest.raises(ValueError, match='verified graph'):
        run.input_bindings(specification)


@pytest.mark.parametrize('which', ['decision', 'result', 'verification'])
@pytest.mark.parametrize('truthy', [1, 'true'])
def test_predecessor_requires_true_in_every_independent_receipt(tmp_path, which, truthy):
    decision = {'passes_all_gates': True}
    result = {'summary': {'passes_all_gates': True}}
    verification = {'passes_all_gates': True, 'status': 'passed'}
    if which == 'result': result['summary']['passes_all_gates'] = truthy
    elif which == 'decision': decision['passes_all_gates'] = truthy
    else: verification['passes_all_gates'] = truthy
    write(tmp_path/'selection_delay1_decision_lock.json', decision)
    write(tmp_path/'selection_delay1.json', result)
    write(tmp_path/'selection_delay1_verification.json', verification)
    with pytest.raises(ValueError, match='independently pass'):
        run.predecessor(tmp_path, 'selection_delay1')
