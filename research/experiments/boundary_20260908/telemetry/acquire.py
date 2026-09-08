"""Acquire the fixed44 discovery streams with the already tested HTTP transport."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CONTRACT = HERE / 'acquisition_contract.json'
OUT = ROOT / 'artifacts/research/boundary_20260908/telemetry'
DATA = ROOT / 'data/f1/boundary_20260908/telemetry'
PILOT = ROOT / 'artifacts/research/boundary_20260908/telemetry_pilot'
TRANSPORT = ROOT / 'research/experiments/boundary_20260908/telemetry_pilot/acquire.py'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def validate():
    spec = read(CONTRACT)
    for row in [spec['parent_manifest'], *spec['bindings']]:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Frozen source/input drift: ' + row['path'])
    parents = read(ROOT / spec['parent_manifest']['path'])['sessions']
    sessions = spec['sessions']
    if sessions != parents or len(sessions) != 44 or len({r['event_key'] for r in sessions}) != 44:
        raise ValueError('Discovery session mapping differs from frozen original44')
    if {r['event_key']//100 for r in sessions} != {2022, 2023}:
        raise ValueError('Later years are outside discovery acquisition')
    if (spec['expected_sessions'] != 44 or spec['reuse_event_keys'] != [202201, 202301]
            or spec['maximum_new_http_requests'] != 84):
        raise ValueError('Discovery acquisition count differs from fixed scope')
    prior = read(ROOT / 'research/experiments/boundary_20260908/telemetry_pilot/specification.json')
    for key in ('base_urls', 'follow_redirects', 'elapsed_deadline_semantics',
                'elapsed_deadline_limitations', 'max_decoded_body_bytes', 'max_http_body_bytes',
                'max_parallel_requests', 'max_request_elapsed_seconds', 'max_requests_per_event',
                'request_timeout_connect_seconds', 'request_timeout_read_seconds', 'stream'):
        if type(spec[key]) is not type(prior[key]) or spec[key] != prior[key]:
            raise ValueError('Reviewed HTTP transport contract changed: ' + key)
    if (type(spec['minimum_free_disk_bytes']) is not int or
            spec['minimum_free_disk_bytes'] != 12*1024**3 or
            shutil.disk_usage(ROOT).free < spec['minimum_free_disk_bytes']):
        raise ValueError('Insufficient fixed acquisition headroom')
    pilot = read(PILOT / 'acquisition.json')
    verification = read(PILOT / 'independent_verification.json')
    if read(PILOT / 'diagnostics.json')['status'] != 'passed' or verification['status'] != 'passed':
        raise ValueError('Pilot is not completely verified')
    # Recheck the closed receipt/source bindings, not only their earlier verdict.
    for path, binding in verification['bindings'].items():
        source = ROOT / path
        if sha(source) != binding['sha256'] or source.stat().st_size != binding['bytes']:
            raise ValueError('Verified pilot evidence changed: ' + path)
    reused = []
    for row in pilot['streams']:
        if row['event_key'] not in spec['reuse_event_keys'] or row['status'] != 'downloaded_unparsed':
            raise ValueError('Unexpected pilot stream')
        path = ROOT / row['decoded_body_path']
        if sha(path) != row['decoded_body_sha256'] or path.stat().st_size != row['decoded_body_bytes']:
            raise ValueError('Verified pilot body changed')
        for attempt in row['attempts']:
            if 'wire_body_path' in attempt and sha(ROOT / attempt['wire_body_path']) != attempt['wire_body_sha256']:
                raise ValueError('Pilot response receipt binding changed')
        reused.append({**row, 'status': 'reused_verified_pilot', 'new_requests': 0})
    if sorted(r['event_key'] for r in reused) != spec['reuse_event_keys']:
        raise ValueError('Missing verified pilot')
    return spec, reused


def transport():
    """Configure an isolated module instance; never edit the frozen helper file."""
    module_spec = importlib.util.spec_from_file_location('telemetry44_transport', TRANSPORT)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.DATA = DATA
    return module


def sources():
    paths = [Path(__file__), CONTRACT, TRANSPORT,
             HERE / 'test_acquire.py', HERE / 'ACQUISITION.md']
    return {rel(path): sha(path) for path in paths}


def preflight_outputs(jobs):
    """Fail before *any* job starts if this scope was already attempted."""
    for path in (OUT / 'acquisition_lock.json', OUT / 'acquisition.json'):
        if path.exists():
            raise FileExistsError('Existing acquisition output: ' + str(path))
    for job in jobs:
        event = job['event_key']
        paths = [OUT / f'acquired_{event}.json', *DATA.glob(f'{event}_CarData.z.*')]
        if any(path.exists() for path in paths):
            raise FileExistsError('Existing acquisition artifacts for event ' + str(event))


def close_new_row(row, job, spec):
    """Bind every completed or failed attempt without treating failure as success."""
    if (row['event_key'] != job['event_key'] or row['session_path'] != job['session_path']
            or row['stream'] != 'CarData.z' or
            row['status'] not in ('unavailable', 'downloaded_unparsed') or
            not 1 <= len(row['attempts']) <= spec['max_requests_per_event']):
        raise ValueError('Acquisition result does not match its fixed job')
    bindings = {}
    def bind(path, expected_sha=None, expected_bytes=None):
        digest, size = sha(path), path.stat().st_size
        if (expected_sha is not None and digest != expected_sha or
                expected_bytes is not None and size != expected_bytes):
            raise ValueError('Closed response body changed')
        bindings[rel(path)] = {'sha256': digest, 'bytes': size}
    for ordinal, attempt in enumerate(row['attempts'], 1):
        expected_url = spec['base_urls'][ordinal-1] + '/static/' + job['session_path'] + 'CarData.z.jsonStream'
        if attempt['requested_url'] != expected_url:
            raise ValueError('Attempt URL is outside its fixed provider order')
        receipt = DATA / f"{job['event_key']}_CarData.z.attempt{ordinal}.receipt.json"
        if read(receipt) != attempt:
            raise ValueError('Attempt receipt differs from result')
        bind(receipt)
        if 'wire_body_path' in attempt:
            body = DATA / f"{job['event_key']}_CarData.z.attempt{ordinal}.httpbody"
            if attempt['wire_body_path'] != rel(body):
                raise ValueError('Attempt body is outside its fixed output path')
            bind(body, attempt['wire_body_sha256'], attempt['wire_body_bytes'])
    source = DATA / f"{job['event_key']}_CarData.z.source.json"
    if read(source) != row:
        raise ValueError('Source receipt differs from result')
    bind(source)
    if row['status'] == 'downloaded_unparsed':
        final = row['attempts'][-1]
        if final.get('http_status') != 200 or final.get('wire_body_complete') is not True or 'error' in final:
            raise ValueError('Downloaded result lacks a complete successful attempt')
        decoded = ROOT / row['decoded_body_path']
        wire = ROOT / final['wire_body_path']
        if decoded not in (wire, wire.with_suffix('.decoded.jsonStream')):
            raise ValueError('Decoded body is outside its fixed output path')
        if not 0 < row['decoded_body_bytes'] <= spec['max_decoded_body_bytes']:
            raise ValueError('Decoded body size is outside the contract')
        bind(decoded, row['decoded_body_sha256'], row['decoded_body_bytes'])
    return bindings


def main(dry_run=False):
    spec, rows = validate()
    jobs = [r for r in spec['sessions'] if r['event_key'] not in spec['reuse_event_keys']]
    if len(jobs) != 42:
        raise ValueError('Expected42 new streams')
    summary = {'sessions': 44, 'new_stream_jobs': 42, 'reused_streams': 2,
               'max_new_http_requests': 84, 'max_parallel_requests': 2,
               'source_files': sources(), 'model_fits': 0, 'predictive_scores': 0}
    if dry_run:
        print(json.dumps(summary, sort_keys=True))
        return
    preflight_outputs(jobs)
    review_path = OUT / 'acquisition_review.json'
    review = read(review_path)
    if review.get('status') != 'passed' or review.get('source_files') != summary['source_files']:
        raise ValueError('A passing review of the exact acquisition sources is required')
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    write(OUT / 'acquisition_lock.json', {**summary, 'frozen_at_utc': now(), 'contract_sha256': sha(CONTRACT),
          'acquisition_review_sha256': sha(review_path)})
    client = transport()
    output_bindings = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(client.fetch, job, spec): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            output_bindings.update(close_new_row(row, futures[future], spec))
            rows.append(row)
            # Immutable per-event progress survives an interruption; retries need
            # a separate reviewed output scope, never overwritten partial data.
            write(OUT / f"acquired_{row['event_key']}.json", row)
    if summary['source_files'] != sources():
        raise ValueError('Acquisition source changed while requests ran')
    rows.sort(key=lambda row: row['event_key'])
    if [r['event_key'] for r in rows] != [r['event_key'] for r in spec['sessions']]:
        raise ValueError('Final acquisition population changed')
    for row in rows:
        if row['status'] != 'unavailable' and sha(ROOT / row['decoded_body_path']) != row['decoded_body_sha256']:
            raise ValueError('Closed response body changed')
    for path, binding in output_bindings.items():
        if sha(ROOT / path) != binding['sha256'] or (ROOT / path).stat().st_size != binding['bytes']:
            raise ValueError('Closed acquisition receipt changed')
    result = {**summary, 'completed_at_utc': now(), 'status': 'all_discovery_attempts_recorded',
              'contract_sha256': sha(CONTRACT), 'acquisition_lock_sha256': sha(OUT / 'acquisition_lock.json'),
              'sessions': spec['sessions'], 'streams': rows,
              'output_bindings': output_bindings,
              'new_http_attempts': sum(len(r['attempts']) for r in rows if r['status'] != 'reused_verified_pilot'),
              'unavailable_streams': sum(r['status'] == 'unavailable' for r in rows)}
    write(OUT / 'acquisition.json', result)
    print(json.dumps({'manifest': rel(OUT / 'acquisition.json'), 'sha256': sha(OUT / 'acquisition.json'),
                      'unavailable_streams': result['unavailable_streams'], 'new_http_attempts': result['new_http_attempts']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args().dry_run)
