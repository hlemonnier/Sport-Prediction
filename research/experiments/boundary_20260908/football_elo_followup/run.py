"""Execute the single frozen Elo-Odds follow-up without strength refitting.

Suggested commit: research(football): execute the fixed Elo-Odds follow-up.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import signal
import subprocess
import sys
import time

from threadpoolctl import threadpool_limits

from . import elo
from ..football_past_market import data, model, run as parent

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts/research/boundary_20260908/football_elo_followup'
SPEC = HERE / 'specification.json'
CANDIDATE = 'elo_odds_reference'
CONTROL = 'elo_result_k14'
INCUMBENT = 'shot_strength_90d_ridge0.1'
OLD_CONTROLS = ('outcome_control', 'outcome_control_shot50')

read, sha, digest, now, utc = parent.read, parent.sha, parent.digest, parent.now, parent.utc
save, relative, file_record = parent.save, parent.relative, parent.file_record
runtime, check_resources, stage_attempt = parent.runtime, parent.check_resources, parent.stage_attempt
make_report = parent.make_report


def bound(item):
    path = ROOT / item['path']
    if sha(path) != item['sha256'] or ('bytes' in item and path.stat().st_size != item['bytes']):
        raise ValueError('Bound input changed: ' + item['path'])
    return path


def parent_specification(spec):
    return read(bound(spec['inputs']['parent_specification']))


def input_bindings(spec):
    result = parent.input_bindings(parent_specification(spec))
    for item in spec['inputs'].values():
        bound(item)
        result[item['path']] = item['sha256']
    design = read(bound(spec['inputs']['parent_design_lock']))
    verification = read(bound(spec['inputs']['parent_primary_verification']))
    if verification.get('status') != 'passed' or verification.get('result_sha256') != spec['inputs']['parent_primary_result']['sha256']:
        raise ValueError('The original candidate has not been independently verified')
    # A failed model-selection gate is an observed result, not an execution
    # failure. The new protocol expressly reuses a comparator from that result.
    if verification.get('passes_all_gates') is not False:
        raise ValueError('The frozen parent failed-family status differs')
    for group in (design['sources'], design['inputs'], verification['bindings']):
        for path, expected in group.items():
            if sha(ROOT / path) != expected or (path in result and result[path] != expected):
                raise ValueError('Inherited verified graph differs: ' + path)
            result[path] = expected
    for key in ('primary_forecasts', 'primary_market_readout', 'parent_primary_issuance_lock', 'references_selection', 'references_transfer'):
        item = spec['inputs'][key]
        if verification['bindings'].get(item['path']) != item['sha256']:
            raise ValueError('Parent verification does not bind ' + key)
    return result


def sources():
    spec = read(SPEC)
    inherited = read(bound(spec['inputs']['parent_design_lock']))['sources']
    paths = [ROOT / path for path in inherited]
    paths.extend(path for pattern in ('*.py', '*.md', '*.json') for path in HERE.glob(pattern))
    return {relative(path): sha(path) for path in sorted(set(paths))}


def verify_design(out):
    if any(out.glob('*_failure.json')):
        raise ValueError('A failed stage is terminal for this execution directory')
    lock = read(out / 'design_lock.json')
    if lock['sources'] != sources() or lock['runtime'] != runtime():
        raise ValueError('Frozen source or runtime changed')
    if lock['inputs'] != input_bindings(read(SPEC)):
        raise ValueError('Frozen input graph changed')
    for key in ('review', 'pre_fit_tests', 'synthetic_feasibility'):
        bound(lock[key])
    return lock


def freeze(out, review_path, feasibility_path):
    if (out / 'design_lock.json').exists():
        raise ValueError('Existing immutable design lock')
    spec = read(SPEC)
    bound(spec['proposal'])
    before = sources()
    review = read(review_path)
    if review.get('approved_for_execution_lock') is not True or review.get('source_files') != before:
        raise ValueError('Independent execution review does not approve these exact files')
    feasibility = read(feasibility_path)
    if feasibility.get('scope') != 'synthetic_only_no_historical_inputs' or feasibility.get('limits', {}).get('result') != 'passed':
        raise ValueError('Synthetic feasibility has not passed')
    for path in (HERE / 'elo.py', Path(model.__file__)):
        if feasibility.get('source_files', {}).get(relative(path)) != sha(path):
            raise ValueError('Synthetic feasibility used different model code')
    inputs = input_bindings(spec)
    command = [sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib', '-p', 'no:cacheprovider', relative(HERE)]
    tested = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    save(out / 'pre_fit_tests.json', {'command': command, 'exit_code': tested.returncode,
         'stdout': tested.stdout, 'stderr': tested.stderr, 'completed_at_utc': now(), 'source_files': before})
    if tested.returncode or sources() != before:
        raise ValueError('Pre-fit tests failed or source changed during tests')
    save(out / 'design_lock.json', {'locked_at_utc': now(), 'sources': before, 'inputs': inputs,
         'runtime': runtime(), 'review': file_record(review_path), 'pre_fit_tests': file_record(out / 'pre_fit_tests.json'),
         'synthetic_feasibility': file_record(feasibility_path), 'historical_new_fits_before_lock': 0,
         'new_scores_before_lock': False, 'maximum_new_readout_fits': 3, 'strength_fits': 0})
    print({'stage': 'frozen', 'sha256': sha(out / 'design_lock.json')}, flush=True)


def reference_rows(spec, phase):
    rows = read(bound(spec['inputs'][f'references_{phase}']))
    if phase == 'selection':
        old = read(bound(spec['inputs']['primary_forecasts']))
        if [r['match_id'] for r in old] != [r['match_id'] for r in rows]:
            raise ValueError('Original candidate and reference identities differ')
        for row, previous in zip(rows, old, strict=True):
            if any(previous[key] != value for key, value in row.items() if key != 'probabilities'):
                raise ValueError('Original candidate metadata differs')
            if any(previous['probabilities'][key] != value for key, value in row['probabilities'].items()):
                raise ValueError('Original reference probabilities differ')
            for name in OLD_CONTROLS:
                row['probabilities'][name] = data._vector(previous['probabilities'][name])
    if len(rows) != spec[phase]['rows'] or digest([r['match_id'] for r in rows]) != spec[phase]['ordered_original_match_ids_sha256']:
        raise ValueError('Original full ordered population changed')
    for row in rows:
        if any(key in row for key in ('label', 'outcome', 'goals', 'y')):
            raise ValueError('References contain scoring labels')
    return rows


def verify_prepared(out):
    verify_design(out)
    lock = read(out / 'data_lock.json')
    if lock['design_lock_sha256'] != sha(out / 'design_lock.json'):
        raise ValueError('Prepared design differs')
    for item in lock['files'].values():
        bound(item)
    return lock


def prepare(out):
    verify_design(out)
    spec = read(SPEC)
    started = time.monotonic()
    with stage_attempt(out, 'prepare'):
        archive = data.load_archive(parent_specification(spec), root=ROOT)
        refs = {phase: reference_rows(spec, phase) for phase in ('selection', 'transfer')}
        support = {}
        for rows in refs.values():
            data.validate_reference_metadata(archive, rows)
        for delay in (1, 7):
            count = 0
            for cutoff in sorted({r['forecast_cutoff_utc'] for r in refs['selection']}, key=utc):
                batch = [r for r in refs['selection'] if r['forecast_cutoff_utc'] == cutoff]
                history, _ = data.quote_rows(archive, cutoff, delay)
                count += sum(data.support_for(batch, history))
            support[str(delay)] = int(count)
        if support != {'1': spec['support']['expected_selection_supported_primary'], '7': spec['support']['expected_selection_supported_sensitivity']}:
            raise ValueError('Fixed support population changed')
        for phase, rows in refs.items():
            save(out / f'references_{phase}.json', rows)
        save(out / 'archive_metadata.json', data.metadata_rows(archive))
        verify_design(out)
        save(out / 'data_lock.json', {'closed_at_utc': now(), 'design_lock_sha256': sha(out / 'design_lock.json'),
             'files': {name: file_record(out / f'{name}.json') for name in ('references_selection', 'references_transfer', 'archive_metadata')},
             'selection_support': support, 'new_fits': 0, 'new_scores': False,
             'resources': check_resources(started, spec), 'references_labels_attached': False})
    print({'stage': 'prepared', 'selection_support': support}, flush=True)


def readout_path(out, kind, delay):
    return out / f'elo_{kind}_readout_delay{delay}.json'


def readouts(out, archive, spec, phase, delay, first_cutoff, started):
    if phase == 'selection':
        for kind in ('odds', 'result'):
            path = readout_path(out, kind, delay)
            if kind == 'odds' and delay == 1:
                # Preserve the exact historical readout bytes, including its
                # original creation time. This is reuse, not a new fit.
                source = bound(spec['inputs']['primary_market_readout'])
                with path.open('xb') as stream:
                    stream.write(source.read_bytes())
                if sha(path) != spec['inputs']['primary_market_readout']['sha256']:
                    raise ValueError('Primary readout bytes changed')
            else:
                x, y, training = elo.calibration_rows(archive, delay, first_cutoff, kind=kind)
                fitted = model.fit_ordered_logit(x, y)
                save(path, {'kind': kind, 'model': fitted, 'training': training, 'x': list(map(float, x)),
                     'y': list(map(int, y)), 'closed_at_utc': now(), 'resources': check_resources(started, spec)})
    return {kind: read(readout_path(out, kind, delay))['model'] for kind in ('odds', 'result')}


def fit_and_issue(out, phase, delay):
    spec = read(SPEC)
    refs = read(out / f'references_{phase}.json')
    archive = data.load_archive(parent_specification(spec), root=ROOT)
    data.validate_reference_metadata(archive, refs)
    stage = f'{phase}_delay{delay}'
    started = time.monotonic()
    first = min((r['forecast_cutoff_utc'] for r in read(out / 'references_selection.json')), key=utc)
    fitted = readouts(out, archive, spec, phase, delay, first, started)
    primary = phase == 'selection' and delay == 1
    old = {r['match_id']: r for r in read(bound(spec['inputs']['primary_forecasts']))} if primary else None
    cursors = {kind: elo.EloCursor(archive, delay, kind=kind) for kind in (('result',) if primary else ('odds', 'result'))}
    results, states = [], []
    clocks = sorted({row['forecast_cutoff_utc'] for row in refs}, key=utc)
    for index, cutoff in enumerate(clocks):
        batch = [row for row in refs if row['forecast_cutoff_utc'] == cutoff]
        history, _ = data.quote_rows(archive, cutoff, delay)
        history_ids = [row['match_id'] for row in history]
        if len(set(history_ids)) != len(history_ids) or set(history_ids) & {r['match_id'] for r in batch}:
            raise ValueError('Current fixture or duplicate in support history')
        supported = data.support_for(batch, history)
        queried = {kind: cursor.query(batch, cutoff) for kind, cursor in cursors.items()}
        probabilities = {kind: model.predict_ordered_logit(fitted[kind], snapshot['x']) for kind, snapshot in queried.items()}
        state_id = f'{index:04d}'
        for i, (row, support) in enumerate(zip(batch, supported, strict=True)):
            new = deepcopy(row)
            new.update({'support': bool(support), 'elo_state_id': state_id})
            incumbent = row['probabilities'][INCUMBENT]
            candidate = (data._vector(old[row['match_id']]['probabilities'][CANDIDATE]) if primary
                         else probabilities['odds'][i].tolist() if support else list(incumbent))
            if primary and (old[row['match_id']]['support'] is not bool(support) or (not support and candidate != incumbent)):
                raise ValueError('Primary candidate fallback or support changed')
            new['probabilities'].update({CANDIDATE: candidate,
                CONTROL: probabilities['result'][i].tolist() if support else list(incumbent)})
            parent.probability_vectors(list(new['probabilities'].values()))
            if set(new['probabilities']) != {CANDIDATE, *spec[phase]['references']}:
                raise ValueError('Fixed probability inventory changed')
            results.append(new)
        states.append({'state_id': state_id, 'cutoff_utc': cutoff, 'batch_ids': [r['match_id'] for r in batch],
             'supported': list(map(bool, supported)), 'support_training_ids': history_ids,
             'support_training_ids_sha256': digest(history_ids), 'ratings': queried})
        check_resources(started, spec)
        if index % 40 == 0 or index + 1 == len(clocks):
            print({'stage': stage, 'clock': index + 1, 'total': len(clocks), 'elapsed_seconds': time.monotonic() - started}, flush=True)
    by_id = {row['match_id']: row for row in results}
    results = [by_id[r['match_id']] for r in refs]
    if len(by_id) != len(refs):
        raise ValueError('Issuance population changed')
    save(out / f'{stage}_issued.json', results)
    save(out / f'{stage}_states.json', states)
    save(out / f'{stage}_issuance_lock.json', {'closed_at_utc': now(), 'forecasts': file_record(out / f'{stage}_issued.json'),
         'rows': len(results), 'labels_attached': False, 'data_lock_sha256': sha(out / 'data_lock.json'),
         'states': file_record(out / f'{stage}_states.json'),
         'readouts': {kind: file_record(readout_path(out, kind, delay)) for kind in ('odds', 'result')},
         'optimizer_fits': (1 if primary else 2) if phase == 'selection' else 0,
         'resources': check_resources(started, spec)})
    return data.attach_labels(results, archive, forecast_closure_path=out / f'{stage}_issuance_lock.json')


def verify_phase_files(out, stage):
    closure = read(out / f'{stage}_issuance_lock.json')
    if closure['labels_attached'] is not False or closure['data_lock_sha256'] != sha(out / 'data_lock.json'):
        raise ValueError('Invalid forecast closure')
    for item in [closure['forecasts'], closure['states'], *closure['readouts'].values()]:
        bound(item)
    return closure


def predecessor(out, stage):
    if any(out.glob('*_failure.json')):
        raise ValueError('A failed execution cannot authorize advancement')
    decision = read(out / f'{stage}_decision_lock.json')
    result = read(out / f'{stage}.json')
    verification = read(out / f'{stage}_verification.json')
    if (decision['passes_all_gates'] is not True or result['summary']['passes_all_gates'] is not True
            or verification.get('passes_all_gates') is not True or verification.get('status') != 'passed'):
        raise ValueError('Predecessor did not independently pass')
    if decision['selected_candidate'] != CANDIDATE or result['summary']['selected_candidate'] != CANDIDATE:
        raise ValueError('The fixed candidate changed')
    if decision['design_lock_sha256'] != sha(out / 'design_lock.json') or result['design_lock_sha256'] != sha(out / 'design_lock.json'):
        raise ValueError('Predecessor design changed')
    if decision['result_sha256'] != sha(out / f'{stage}.json') or verification['result_sha256'] != sha(out / f'{stage}.json'):
        raise ValueError('Predecessor result changed')
    if result['issuance_lock_sha256'] != sha(out / f'{stage}_issuance_lock.json'):
        raise ValueError('Predecessor issuance changed')
    verify_phase_files(out, stage)
    for path, expected in verification['bindings'].items():
        if sha(ROOT / path) != expected:
            raise ValueError('Predecessor independent binding changed')
    return CANDIDATE


def required_predecessors(phase, delay):
    if phase not in ('selection', 'transfer') or type(delay) is not int or delay not in (1, 7):
        raise ValueError('Unknown fixed phase or delay')
    stages = ['selection_delay1'] if delay == 7 or phase == 'transfer' else []
    if phase == 'transfer':
        stages.append('selection_delay7')
        if delay == 7:
            stages.append('transfer_delay1')
    return stages


def execute(out, phase, delay):
    started = time.monotonic()
    previous = required_predecessors(phase, delay)
    verify_prepared(out)
    spec = read(SPEC)
    for stage in previous:
        predecessor(out, stage)
    stage = f'{phase}_delay{delay}'
    with stage_attempt(out, stage), threadpool_limits(limits=1):
        rows = fit_and_issue(out, phase, delay)
        summary = make_report(rows, spec, phase, CANDIDATE)
        verify_prepared(out)
        save(out / f'{stage}.json', {'completed_at_utc': now(), 'summary': summary,
             'resources': check_resources(started, spec), 'status': 'retrospective_reused_dates_no_promotion',
             'design_lock_sha256': sha(out / 'design_lock.json'), 'data_lock_sha256': sha(out / 'data_lock.json'),
             'issuance_lock_sha256': sha(out / f'{stage}_issuance_lock.json')})
        save(out / f'{stage}_decision_lock.json', {'closed_at_utc': now(), 'selected_candidate': CANDIDATE,
             'passes_all_gates': summary['passes_all_gates'], 'result_sha256': sha(out / f'{stage}.json'),
             'design_lock_sha256': sha(out / 'design_lock.json')})
    print({'stage': stage, 'nll': {name: values['log_loss'] for name, values in summary['metrics'].items()},
           'checks': summary['checks'], 'passes_all_gates': summary['passes_all_gates']}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'prepare', 'selection', 'sensitivity', 'transfer'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--feasibility', type=Path)
    parser.add_argument('--delay', type=int, choices=(1, 7), default=1)
    args = parser.parse_args()
    if args.stage == 'freeze':
        if args.review is None or args.feasibility is None:
            parser.error('freeze requires --review and --feasibility')
        freeze(args.out, args.review, args.feasibility)
        return
    def deadline(_signum, _frame):
        raise TimeoutError('Predeclared stage wall limit reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(read(SPEC)['resources']['maximum_stage_wall_seconds'])
    try:
        if args.stage == 'prepare':
            prepare(args.out)
        else:
            execute(args.out, 'transfer' if args.stage == 'transfer' else 'selection',
                    args.delay if args.stage == 'transfer' else (7 if args.stage == 'sensitivity' else 1))
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
