"""Offline orchestration checks; all transport calls are synthetic."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import pytest

from research.experiments.boundary_20260908.telemetry import acquire as a


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline test attempted a network request')
    monkeypatch.setattr(requests, 'get', forbidden)


@pytest.fixture
def scope(tmp_path, monkeypatch):
    original = json.loads(a.CONTRACT.read_text())
    pilot_spec = json.loads((a.TRANSPORT.parent / 'specification.json').read_text())
    root = tmp_path
    here = root / 'research/experiments/boundary_20260908/telemetry'
    pilot = root / 'artifacts/research/boundary_20260908/telemetry_pilot'
    out, data = root / 'artifacts/new', root / 'data/new'
    here.mkdir(parents=True)
    data.mkdir(parents=True)
    monkeypatch.setattr(a, 'ROOT', root)
    monkeypatch.setattr(a, 'HERE', here)
    monkeypatch.setattr(a, 'CONTRACT', here / 'acquisition_contract.json')
    monkeypatch.setattr(a, 'PILOT', pilot)
    monkeypatch.setattr(a, 'OUT', out)
    monkeypatch.setattr(a, 'DATA', data)
    monkeypatch.setattr(a.shutil, 'disk_usage', lambda _: SimpleNamespace(free=13*1024**3))
    bindings = []
    def bound(path, content):
        put(root / path, content)
        bindings.append({'path': path, 'sha256': a.sha(root / path)})
    parent = 'artifacts/parent.json'
    put(root / parent, {'sessions': original['sessions']})
    original['parent_manifest'] = {'path': parent, 'sha256': a.sha(root / parent)}
    rows = []
    for event in (202201, 202301):
        body = root / f'data/pilot/{event}.body'
        body.parent.mkdir(parents=True, exist_ok=True)
        body.write_bytes(b'synthetic pilot')
        rows.append({'event_key': event, 'status': 'downloaded_unparsed', 'attempts': [],
            'decoded_body_path': str(body.relative_to(root)), 'decoded_body_sha256': a.sha(body),
            'decoded_body_bytes': body.stat().st_size})
    prior_receipt = root / 'data/pilot/receipt.json'
    put(prior_receipt, {'closed': True})
    bound(str((pilot / 'acquisition.json').relative_to(root)), {'streams': rows})
    bound(str((pilot / 'diagnostics.json').relative_to(root)), {'status': 'passed'})
    bound(str((pilot / 'independent_verification.json').relative_to(root)), {
        'status': 'passed', 'bindings': {str(prior_receipt.relative_to(root)): {
            'sha256': a.sha(prior_receipt), 'bytes': prior_receipt.stat().st_size}}})
    bound('research/experiments/boundary_20260908/telemetry_pilot/specification.json', pilot_spec)
    original['bindings'] = bindings
    put(a.CONTRACT, original)
    monkeypatch.setattr(a, 'sources', lambda: {'synthetic/acquire.py': 'fixed-source'})
    put(out / 'acquisition_review.json', {'status': 'passed', 'source_files': a.sources()})
    return original


def test_valid_scope_reuses_exactly_two_without_new_requests(scope):
    spec, rows = a.validate()
    assert len(spec['sessions']) == 44
    assert [r['event_key'] for r in rows] == [202201, 202301]
    assert all(r['status'] == 'reused_verified_pilot' and r['new_requests'] == 0 for r in rows)


@pytest.mark.parametrize('key,value', [
    ('expected_sessions', 43), ('maximum_new_http_requests', 85),
    ('reuse_event_keys', [202201]), ('max_parallel_requests', 3),
    ('max_requests_per_event', 3), ('max_http_body_bytes', 2**27),
    ('max_decoded_body_bytes', 2**27), ('request_timeout_read_seconds', 31),
    ('request_timeout_connect_seconds', 16), ('max_request_elapsed_seconds', 151),
    ('follow_redirects', True), ('follow_redirects', 0),
    ('stream', 'Position.z'), ('base_urls', ['https://unreviewed.invalid']),
    ('minimum_free_disk_bytes', float(12*1024**3)),
])
def test_contract_scope_and_typed_transport_limits_fail_closed(scope, key, value):
    scope[key] = value
    put(a.CONTRACT, scope)
    with pytest.raises(ValueError):
        a.validate()


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'later_year', 'reordered'])
def test_mapping_must_equal_original44(scope, mutation):
    if mutation == 'missing': scope['sessions'].pop()
    elif mutation == 'duplicate': scope['sessions'][-1] = scope['sessions'][0]
    elif mutation == 'later_year': scope['sessions'][0]['event_key'] = 202401
    else: scope['sessions'].reverse()
    put(a.CONTRACT, scope)
    with pytest.raises(ValueError): a.validate()


@pytest.mark.parametrize('path', [
    'data/pilot/receipt.json', 'data/pilot/202201.body',
    'artifacts/parent.json',
    'research/experiments/boundary_20260908/telemetry_pilot/specification.json',
])
def test_closed_pilot_source_receipt_body_and_parent_drift(scope, path):
    with (a.ROOT / path).open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError): a.validate()


def test_insufficient_disk_stops_before_requests(scope, monkeypatch):
    monkeypatch.setattr(a.shutil, 'disk_usage', lambda _: SimpleNamespace(free=12*1024**3-1))
    with pytest.raises(ValueError, match='headroom'): a.validate()


def test_dry_run_is_read_only_and_has_fixed_42_job_budget(scope, capsys):
    before = {str(p): a.sha(p) for p in a.ROOT.rglob('*') if p.is_file()}
    a.main(dry_run=True)
    result = json.loads(capsys.readouterr().out)
    assert (result['sessions'], result['new_stream_jobs'], result['reused_streams']) == (44, 42, 2)
    assert result['max_new_http_requests'] == 84 and result['max_parallel_requests'] == 2
    assert result['model_fits'] == result['predictive_scores'] == 0
    assert before == {str(p): a.sha(p) for p in a.ROOT.rglob('*') if p.is_file()}


@pytest.mark.parametrize('location', ['lock', 'manifest', 'progress', 'body', 'receipt', 'source'])
def test_global_output_guard_precedes_all_requests(scope, monkeypatch, location):
    paths = {'lock': a.OUT/'acquisition_lock.json', 'manifest': a.OUT/'acquisition.json',
        'progress': a.OUT/'acquired_202322.json', 'body': a.DATA/'202322_CarData.z.attempt1.httpbody',
        'receipt': a.DATA/'202322_CarData.z.attempt1.receipt.json',
        'source': a.DATA/'202322_CarData.z.source.json'}
    put(paths[location], {'preserve': True})
    before = paths[location].read_bytes()
    monkeypatch.setattr(a, 'transport', lambda: pytest.fail('Transport constructed before global guard'))
    with pytest.raises(FileExistsError): a.main()
    assert paths[location].read_bytes() == before


@pytest.mark.parametrize('bad_review', [None, {'status': 'failed'},
    {'status': 'passed', 'source_files': {'different': 'source'}}])
def test_source_bound_review_required_before_requests(scope, bad_review, monkeypatch):
    path = a.OUT / 'acquisition_review.json'
    if bad_review is None: path.unlink()
    else: put(path, bad_review)
    monkeypatch.setattr(a, 'transport', lambda: pytest.fail('Unreviewed transport constructed'))
    with pytest.raises((ValueError, FileNotFoundError)): a.main()
    assert not (a.OUT/'acquisition_lock.json').exists()


def fake_fetch(job, spec, *, unavailable=False, second=False):
    row = {'event_key': job['event_key'], 'session_path': job['session_path'],
           'stream': 'CarData.z', 'status': 'unavailable', 'attempts': []}
    for ordinal in range(1, 3 if unavailable or second else 2):
        body = a.DATA / f"{job['event_key']}_CarData.z.attempt{ordinal}.httpbody"
        body.write_bytes(b'closed synthetic body')
        attempt = {'requested_url': spec['base_urls'][ordinal-1]+'/static/'+job['session_path']+'CarData.z.jsonStream',
            'http_status': 404 if unavailable or second and ordinal == 1 else 200,
            'wire_body_complete': True, 'wire_body_path': a.rel(body),
            'wire_body_sha256': a.sha(body), 'wire_body_bytes': body.stat().st_size}
        row['attempts'].append(attempt)
        put(a.DATA/f"{job['event_key']}_CarData.z.attempt{ordinal}.receipt.json", attempt)
    if not unavailable:
        row.update(status='downloaded_unparsed', decoded_body_path=a.rel(body),
                   decoded_body_sha256=a.sha(body), decoded_body_bytes=body.stat().st_size)
    put(a.DATA/f"{job['event_key']}_CarData.z.source.json", row)
    return row


def test_full_offline_orchestration_keeps_failures_reuse_and_two_worker_cap(scope, monkeypatch):
    seen, active, peak = [], 0, 0
    guard = threading.Lock()
    def fetch(job, spec):
        nonlocal active, peak
        with guard:
            active += 1; peak = max(active, peak); seen.append(job['event_key'])
        time.sleep(.002)
        row = fake_fetch(job, spec, unavailable=job['event_key'] == 202322,
                         second=job['event_key'] == 202202)
        with guard: active -= 1
        return row
    monkeypatch.setattr(a, 'transport', lambda: SimpleNamespace(fetch=fetch))
    a.main()
    result = a.read(a.OUT/'acquisition.json')
    assert len(seen) == len(set(seen)) == 42 and 202201 not in seen and 202301 not in seen
    assert peak == 2 and len(result['streams']) == 44
    assert result['unavailable_streams'] == 1 and result['new_http_attempts'] == 44
    assert len(list(a.OUT.glob('acquired_*.json'))) == 42
    assert a.read(a.OUT/'acquisition_lock.json')['acquisition_review_sha256'] == a.sha(a.OUT/'acquisition_review.json')
    assert len(result['output_bindings']) == 42*3+4
    for path, binding in result['output_bindings'].items():
        assert a.sha(a.ROOT/path) == binding['sha256']
        assert (a.ROOT/path).stat().st_size == binding['bytes']


@pytest.mark.parametrize('fault', ['receipt', 'source', 'body', 'url', 'event', 'status', 'success', 'decoded_path'])
def test_job_closure_rejects_corrupt_or_misbound_outputs(scope, fault):
    job = scope['sessions'][1]
    row = fake_fetch(job, scope)
    if fault == 'receipt': put(a.DATA/f"{job['event_key']}_CarData.z.attempt1.receipt.json", {})
    elif fault == 'source': put(a.DATA/f"{job['event_key']}_CarData.z.source.json", {})
    elif fault == 'body': (a.ROOT/row['decoded_body_path']).write_bytes(b'corrupt')
    elif fault == 'url': row['attempts'][0]['requested_url'] = 'https://unapproved.invalid'
    elif fault == 'event': row['event_key'] = 202401
    elif fault == 'status': row['status'] = 'verified'
    elif fault == 'success':
        row['attempts'][0]['wire_body_complete'] = False
        put(a.DATA/f"{job['event_key']}_CarData.z.attempt1.receipt.json", row['attempts'][0])
        put(a.DATA/f"{job['event_key']}_CarData.z.source.json", row)
    else:
        row['decoded_body_path'] = 'outside.body'
        put(a.DATA/f"{job['event_key']}_CarData.z.source.json", row)
    with pytest.raises(ValueError): a.close_new_row(row, job, scope)


def test_write_is_exclusive_and_rejects_nonfinite_json(tmp_path):
    path = tmp_path/'out.json'
    a.write(path, {'a': 1})
    with pytest.raises(FileExistsError): a.write(path, {'a': 2})
    assert a.read(path) == {'a': 1}
    with pytest.raises(ValueError): a.write(tmp_path/'invalid.json', {'a': float('nan')})


def test_transport_module_is_isolated_and_frozen_source_unchanged():
    before = a.sha(a.TRANSPORT)
    first, second = a.transport(), a.transport()
    assert first is not second and first.DATA == second.DATA == a.DATA
    assert first.fetch.__module__ == 'telemetry44_transport'
    assert a.sha(a.TRANSPORT) == before
