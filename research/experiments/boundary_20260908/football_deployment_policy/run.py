"""Execute the single frozen full-history deployment-policy assessment.

Suggested commit: research(football): execute full-history deployment assessment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

from threadpoolctl import threadpool_limits

from . import data, runtime, scoring
from packages.football.mrp.deployment import restore_dc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts/research/boundary_20260908/football_deployment_policy'
SPEC = HERE / 'specification.json'
sha, digest = data.sha, data.digest


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def file_record(path):
    return {'path': relative(path), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def bound(record):
    path = ROOT / record['path']
    if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
        raise ValueError('Closed artifact changed: ' + record['path'])
    return path


def environment():
    return {'python': platform.python_version(), 'executable_sha256': sha(Path(sys.executable).resolve()),
        'platform': platform.platform(), 'threads': 1,
        'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'threadpoolctl')}}


def sources():
    spec = read(SPEC)
    old = read(ROOT / spec['inputs']['old_evaluation']['path'])['source_files']
    paths = {ROOT / p for p in old}
    paths.update(ROOT / item['path'] for item in spec['inputs']['canonical_global_caller_sources'])
    paths.update((ROOT / 'packages/football/mrp').glob('*.py'))
    paths.add(ROOT / 'packages/football/tests/test_deployment.py')
    paths.update(p for pattern in ('*.py', '*.md', '*.json') for p in HERE.glob(pattern))
    return {relative(p): sha(p) for p in sorted(paths)}


def resources(started, spec):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(rss if sys.platform == 'darwin' else rss * 1024)
    elapsed = time.monotonic() - started
    if rss > spec['resources']['maximum_peak_rss_bytes'] or elapsed > spec['resources']['maximum_stage_wall_seconds']:
        raise RuntimeError('Declared execution resource bound exceeded')
    return {'elapsed_seconds': elapsed, 'peak_rss_bytes': rss}


def refuse_failure(out):
    if any(Path(out).glob('*_failure.json')):
        raise ValueError('A failed execution stage is terminal in this directory')


@contextmanager
def attempt(out, stage):
    refuse_failure(out)
    save(out / f'{stage}_attempt.json', {'started_at_utc': now(), 'stage': stage})
    try:
        yield
    except BaseException as exc:
        save(out / f'{stage}_failure.json', {'failed_at_utc': now(), 'exception_type': type(exc).__name__,
            'message': str(exc), 'advancement_allowed': False})
        raise


def verify_design(out, *, inputs=None):
    refuse_failure(out)
    lock = read(out / 'design_lock.json')
    if lock['sources'] != sources() or lock['runtime'] != environment():
        raise ValueError('Frozen source/runtime changed')
    inputs = data.load_inputs(read(SPEC), ROOT) if inputs is None else inputs
    if lock['inputs'] != inputs.bindings:
        raise ValueError('Frozen input graph changed')
    for path, expected in lock['inputs'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen input bytes changed: ' + path)
    for key in ('review', 'pre_fit_tests'):
        bound(lock[key])
    for item in lock['reference_sources'].values():
        bound(item)
    return lock, inputs


def freeze(out, review_path):
    started = time.monotonic()
    spec, before = read(SPEC), sources()
    with attempt(out, 'freeze'):
        review = read(review_path)
        if review.get('approved_for_execution_lock') is not True or review.get('source_files') != before:
            raise ValueError('Independent review does not approve these exact sources')
        inputs = data.load_inputs(spec, ROOT)
        # Serialization feasibility, without training goal reads or scoring.
        # The source-bound existing states are complete; reuse is fixed here.
        for block in inputs.blocks:
            restore_dc(block['saved_full_model'])
        command = [sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib', '-p', 'no:cacheprovider', relative(HERE), 'packages/football/tests/test_deployment.py']
        tests = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        save(out / 'pre_fit_tests.json', {'command': command, 'exit_code': tests.returncode,
            'stdout': tests.stdout, 'stderr': tests.stderr, 'source_files': before,
            'scope': 'synthetic_only_no_historical_fits_or_scoring', 'completed_at_utc': now()})
        if tests.returncode or sources() != before:
            raise ValueError('Synthetic tests failed or reviewed sources changed')
        snapshots = {}
        for path in before:
            destination = out / 'reference_sources' / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write((ROOT / path).read_bytes())
            snapshots[path] = file_record(destination)
            if snapshots[path]['sha256'] != before[path]:
                raise ValueError('Source snapshot changed while copied')
        if sources() != before or any(sha(ROOT / path) != expected for path, expected in inputs.bindings.items()):
            raise ValueError('Source or input bytes changed before design closure')
        save(out / 'design_lock.json', {'locked_at_utc': now(), 'sources': before,
            'inputs': inputs.bindings, 'runtime': environment(), 'review': file_record(review_path),
            'pre_fit_tests': file_record(out / 'pre_fit_tests.json'), 'reference_sources': snapshots,
            'candidate_state_branch': 'reuse_all_36_saved_full_history_states',
            'branch_basis': 'Complete source-bound state serialization and synthetic caller parity before external target attachment.',
            'historical_new_fits_before_lock': 0, 'new_scores_before_lock': False,
            'resources': resources(started, spec)})
    print(json.dumps({'stage': 'frozen', 'sha256': sha(out / 'design_lock.json')}), flush=True)


def prepare(out):
    started = time.monotonic()
    lock, inputs = verify_design(out)
    with attempt(out, 'prepare'):
        payloads = {'archive_metadata': data.metadata_rows(inputs), 'fixture_metadata': inputs.fixture_rows,
            'blocks_metadata': inputs.blocks, 'block_inventory': inputs.block_inventory}
        for name, value in payloads.items():
            save(out / f'{name}.json', value)
        verify_design(out, inputs=inputs)
        save(out / 'data_lock.json', {'closed_at_utc': now(), 'design_lock_sha256': sha(out / 'design_lock.json'),
            'files': {name: file_record(out / f'{name}.json') for name in payloads},
            'rows': len(inputs.fixture_rows), 'blocks': len(inputs.blocks), 'new_fits': 0, 'new_scores': False,
            'resources': resources(started, inputs.spec)})
    print(json.dumps({'stage': 'prepared', 'sha256': sha(out / 'data_lock.json')}), flush=True)


def verify_prepared(out, inputs):
    lock = read(out / 'data_lock.json')
    if lock['design_lock_sha256'] != sha(out / 'design_lock.json'):
        raise ValueError('Prepared source lock differs')
    for item in lock['files'].values():
        bound(item)
    if read(bound(lock['files']['blocks_metadata'])) != inputs.blocks:
        raise ValueError('Prepared block inventory changed')
    return lock


def forecast(out):
    started = time.monotonic()
    design, inputs = verify_design(out)
    verify_prepared(out, inputs)
    if design['candidate_state_branch'] != 'reuse_all_36_saved_full_history_states':
        raise ValueError('Unexpected precommitted state branch')
    with attempt(out, 'forecast'), threadpool_limits(limits=1):
        issued, blocks = {}, []
        counts = {'prefix_goal_fits': 0, 'full_history_goal_fits': 0,
            'reference_calibration_policy_calls': 0, 'candidate_learned_calibration_fits': 0, 'candidate_parity_goal_fits': 0}
        for index, descriptor in enumerate(inputs.blocks):
            block = data.prepare_block(inputs, descriptor)
            result = runtime.execute_block(block['history'], block['fixtures'], block['saved_full_model'],
                block['fit_cutoff_utc'], block['expected_partitions'], saved_full_lineage=block['saved_full_lineage'],
                expected_reference_probabilities=block['expected_reference_probabilities'],
                expected_candidate_probabilities=block['expected_candidate_probabilities'], shadow_eval=False)
            result.update(block_id=block['block_id'], league=block['league'], fit_id=block['fit_id'],
                fit_cutoff_utc=block['fit_cutoff_utc'], fixture_metadata=block['fixture_metadata'],
                history_metadata=block['history_metadata'], input_sources=block['input_sources'],
                expected_partitions=block['expected_partitions'], saved_full_lineage=block['saved_full_lineage'],
                design_lock_sha256=sha(out / 'design_lock.json'))
            for meta, reference, candidate in zip(block['fixture_metadata'], result['reference']['fixtures'], result['candidate']['fixtures'], strict=True):
                if meta['match_id'] != reference['match_id'] or meta['match_id'] != candidate['match_id'] or meta['match_id'] in issued:
                    raise ValueError('Block forecast identities differ or overlap')
                issued[meta['match_id']] = {**meta, 'block_id': block['block_id'],
                    'outputs': {scoring.REFERENCE: reference, scoring.CANDIDATE: candidate}}
            for key in counts:
                counts[key] += result['fit_counts'][key]
            save(out / 'blocks' / f'{index:02d}.json', result)
            blocks.append(file_record(out / 'blocks' / f'{index:02d}.json'))
            print(json.dumps({'stage': 'forecast', 'completed_blocks': index+1, 'total_blocks': len(inputs.blocks),
                'block_id': block['block_id'], 'rows': len(issued), 'resources': resources(started, inputs.spec)}), flush=True)
        nblocks = inputs.spec['population']['fit_blocks']
        if counts != {'prefix_goal_fits': nblocks, 'full_history_goal_fits': 0,
                'reference_calibration_policy_calls': nblocks, 'candidate_learned_calibration_fits': 0, 'candidate_parity_goal_fits': 0}:
            raise ValueError('Fixed fit/calibration budget differs')
        ids = [r['match_id'] for r in inputs.fixture_rows]
        if set(issued) != set(ids):
            raise ValueError('Forecast population incomplete')
        rows = [issued[x] for x in ids]
        save(out / 'forecasts.json', rows)
        verify_design(out, inputs=inputs)
        save(out / 'forecast_closure.json', {'closed_at_utc': now(), 'forecasts': file_record(out / 'forecasts.json'),
            'blocks': blocks, 'rows': len(rows), 'labels_attached': False, 'fit_counts': counts,
            'design_lock_sha256': sha(out / 'design_lock.json'), 'data_lock_sha256': sha(out / 'data_lock.json'),
            'resources': resources(started, inputs.spec)})
    print(json.dumps({'stage': 'forecast_closed', 'sha256': sha(out / 'forecast_closure.json')}), flush=True)


def score(out):
    started = time.monotonic()
    _, inputs = verify_design(out)
    verify_prepared(out, inputs)
    with attempt(out, 'score'):
        closure_record = file_record(out / 'forecast_closure.json')
        closure = read(out / 'forecast_closure.json')
        if closure['design_lock_sha256'] != sha(out / 'design_lock.json') or closure['data_lock_sha256'] != sha(out / 'data_lock.json'):
            raise ValueError('Forecast closure lineage differs')
        for item in closure['blocks']:
            bound(item)
        rows = data.attach_labels(read(bound(closure['forecasts'])), inputs,
            forecast_closure_path=out / 'forecast_closure.json')
        save(out / 'scored_rows.json', rows)
        scored_record = file_record(out / 'scored_rows.json')
        report = scoring.evaluate(rows, inputs.spec)
        verify_design(out, inputs=inputs)
        verify_prepared(out, inputs)
        for item in [closure_record, scored_record, closure['forecasts'], *closure['blocks']]:
            bound(item)
        save(out / 'evaluation.json', {'created_at_utc': now(), 'report': report,
            'forecast_closure': closure_record,
            'scored_rows': scored_record, 'design_lock': file_record(out / 'design_lock.json'),
            'resources': resources(started, inputs.spec), 'production_activated': False})
    print(json.dumps({'stage': 'scored', 'sha256': sha(out / 'evaluation.json'),
        'metrics': report['metrics'], 'checks': report['checks'], 'passes_all_gates': report['passes_all_gates']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'prepare', 'forecast', 'score'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    def timeout(*_):
        raise TimeoutError('Fixed stage wall limit exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(read(SPEC)['resources']['maximum_stage_wall_seconds'])
    if args.stage == 'freeze':
        if args.review is None:
            parser.error('--review is required for freeze')
        freeze(args.out, args.review)
    else:
        globals()[args.stage](args.out)


if __name__ == '__main__':
    main()
