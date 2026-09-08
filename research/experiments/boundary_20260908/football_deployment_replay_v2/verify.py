"""Correct only the frozen deployment verifier's three-field lineage omission.

No fitting, regeneration, operational imports, changed matrices or changed gates.
The failed original verification remains immutable and terminal in its own stage.
Suggested commit: fix(research): replay missing deployment lineage metadata.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
from unittest.mock import patch

from research.experiments.boundary_20260908.football_deployment_policy import verify as original

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts/research/boundary_20260908/football_deployment_replay_v2'
SPEC = HERE / 'specification.json'
LINEAGE_KEYS = ('goal_state_sha256', 'source_artifact', 'source_artifact_sha256')
FAILURE_MESSAGE = 'Closed metadata differs from independent reconstruction'
_ORIGINAL_REFERENCE_INPUTS = original.reference_inputs


def corrected_reference_inputs(spec, root, inputs, bindings):
    """The original full graph rebuild plus exactly three independently derived keys."""
    archive, fixtures, descriptors, inventory = _ORIGINAL_REFERENCE_INPUTS(spec, root, inputs, bindings)
    corrected = deepcopy(descriptors)
    for descriptor in corrected:
        lineage = descriptor['saved_full_lineage']
        if any(key in lineage for key in LINEAGE_KEYS):
            raise ValueError('The inherited metadata schema no longer has the frozen omission')
        record = spec['inputs']['feature_artifacts'][descriptor['league']]
        if inputs.get(record['path']) != record['sha256']:
            raise ValueError('The provenance artifact lacks its original input lock')
        original.bound(record, root, bindings)
        lineage.update(goal_state_sha256=original.ind.digest(descriptor['saved_full_model']),
                       source_artifact=record['path'], source_artifact_sha256=record['sha256'])
    return archive, fixtures, corrected, inventory


def permit_pinned_failure(parent_out, record, *, root):
    """Permit this one closed bookkeeping failure, never a numerical/fit failure."""
    parent_out, root = Path(parent_out), Path(root)
    failure = parent_out/'verification_failure.json'
    if set(parent_out.glob('*_failure.json')) != {failure}:
        raise ValueError('Only the exact preserved original verification failure is allowed')
    if original.bound(record, root, {}) != failure:
        raise ValueError('Pinned failure points to a different original stage')
    payload = original.read(failure)
    if payload.get('exception_type') != 'ValueError' or payload.get('message') != FAILURE_MESSAGE or payload.get('advancement_allowed') is not False:
        raise ValueError('Original failure is not the exact metadata-bookkeeping rejection')


def replay_parent(parent_out, failure_record, *, root):
    """All original guards/replays execute; only two narrow functions are wrapped."""
    original.reject_operational_imports()
    parent_out = Path(parent_out)
    permit_pinned_failure(parent_out, failure_record, root=root)
    def guarded_failure_check(path):
        if Path(path) != parent_out:
            raise ValueError('Failure exception attempted outside the pinned parent directory')
        permit_pinned_failure(parent_out, failure_record, root=root)
    with patch.object(original, 'reference_inputs', corrected_reference_inputs), patch.object(original, 'check_failures', guarded_failure_check):
        result = original.verify(parent_out, root=root)
    permit_pinned_failure(parent_out, failure_record, root=root)
    return result


def sources(root=ROOT):
    root = Path(root)
    here = root/'research/experiments/boundary_20260908/football_deployment_replay_v2'
    spec = original.read(here/'specification.json')
    paths = {p for pattern in ('*.py', '*.json', '*.md') for p in here.glob(pattern)}
    paths.update(root/spec[key]['path'] for key in ('original_verifier', 'original_independent_arithmetic'))
    return {str(path.relative_to(root)): original.sha(path) for path in sorted(paths)}


def verify(out=OUT, *, root=ROOT):
    """Read-only corrective replay after its separate exact-source closure."""
    started = time.monotonic()
    root, out = Path(root).resolve(), Path(out).resolve()
    original.reject_operational_imports(); original.check_failures(out)
    bindings = {}
    def local(path):
        record = {'path': str(path.relative_to(root)), 'sha256': original.sha(path), 'bytes': path.stat().st_size}
        return original.read(original.bound(record, root, bindings))
    design = local(out/'design_lock.json')
    actual_sources = sources(root)
    if design['sources'] != actual_sources:
        raise ValueError('Corrective source inventory changed')
    for path, expected in actual_sources.items():
        original.bound({'path': path, 'sha256': expected}, root, bindings)
    spec_path = root/'research/experiments/boundary_20260908/football_deployment_replay_v2/specification.json'
    spec = local(spec_path)
    if any(type(spec[key]) is not int or spec[key] != 0 for key in ('new_fits', 'new_forecasts', 'new_calibrations')) or spec['numerical_scoring_or_gate_changes'] is not False or spec['preserve_original_terminal_attempt'] is not True or spec['all_other_failures_fatal'] is not True:
        raise ValueError('Corrective replay cannot change fitting, forecasts, numerical gates or terminal history')
    if tuple(spec['only_correction']['fields']) != LINEAGE_KEYS or spec['permitted_failure'] != {'filename': 'verification_failure.json', 'exception_type': 'ValueError', 'message': FAILURE_MESSAGE}:
        raise ValueError('Corrective amendment scope differs')
    if any(type(design[key]) is not int or design[key] != 0 for key in ('new_fits', 'new_forecasts')):
        raise ValueError('Replay source closure contains unauthorized fitting or forecasts')
    review = design['review']
    if review.get('approved_for_replay') is not True or review.get('source_files') != actual_sources:
        raise ValueError('No exact-source approval for the corrective replay')
    tests = original.read(original.bound(design['pre_fit_tests'], root, bindings))
    if type(tests['exit_code']) is not int or tests['exit_code'] != 0 or tests['source_files'] != actual_sources:
        raise ValueError('Corrective synthetic tests did not pass for these source bytes')
    for key in ('original_verifier', 'original_independent_arithmetic', 'proposal'):
        original.bound(spec[key], root, bindings)
    parent_out = root/spec['parent_out']
    if out == parent_out or parent_out in out.parents:
        raise ValueError('Corrective artifacts require a separate original-stage directory')
    for name, record in spec['inputs'].items():
        if original.bound(record, root, bindings) != parent_out/name:
            raise ValueError('Pinned original closure points outside the declared parent stage')
    failure_record = spec['inputs']['verification_failure.json']
    failure = original.read(root/failure_record['path'])
    if original.ind.utc(failure['failed_at_utc']) > original.ind.utc(design['locked_at_utc']):
        raise ValueError('The corrective design must follow the preserved failure')
    result = replay_parent(parent_out, failure_record, root=root)
    original.check_failures(out)
    if sources(root) != actual_sources or any(original.sha(root/path) != expected for path, expected in bindings.items()):
        raise ValueError('Corrective sources or pinned original closures changed during replay')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(peak if sys.platform == 'darwin' else peak*1024)
    elapsed = time.monotonic()-started
    if peak > spec['maximum_peak_rss_bytes'] or elapsed > spec['maximum_stage_wall_seconds']:
        raise RuntimeError('Corrective replay resource bound exceeded')
    return {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'design_lock_sha256': original.sha(out/'design_lock.json'),
            'source_files': actual_sources, 'bindings': bindings,
            'original_terminal_failure': failure_record,
            'original_protocol_status': 'terminal_verification_failure_preserved',
            'corrected_fields': list(LINEAGE_KEYS), 'independent_verification': result,
            'passes_all_gates': result['passes_all_gates'], 'production_activated': False,
            'new_fits': 0, 'new_calibrations': 0, 'new_forecasts': 0,
            'resources': {'elapsed_seconds': elapsed, 'peak_rss_bytes': peak},
            'scope': 'Separate zero-fit replay with all original independent numerical/causal/closure/gate checks; only three expected provenance fields and one exact immutable failure exception differ.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(); out = args.out.resolve()
    original.check_failures(out)
    def save(name, value):
        with (out/name).open('x') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    def timeout(*_):
        raise TimeoutError('Corrective replay wall limit exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(original.read(SPEC)['maximum_stage_wall_seconds'])
    save('verification_attempt.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                                      'design_lock_sha256': original.sha(out/'design_lock.json')})
    try:
        result = verify(out)
        save('verification.json', result)
    except BaseException as exc:
        save('verification_failure.json', {'failed_at_utc': datetime.now(timezone.utc).isoformat(),
             'exception_type': type(exc).__name__, 'message': str(exc), 'advancement_allowed': False})
        raise
    print(json.dumps({'status': result['status'], 'sha256': original.sha(out/'verification.json'),
                      'passes_all_gates': result['passes_all_gates']}), flush=True)


if __name__ == '__main__':
    main()
