"""Execute the fixed past-market experiment with immutable phase closures.

Suggested commit: research(football): execute delayed past-market strength tests.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'

import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from . import data, model
from research.experiments.performance_20260907.football import benchmark

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts/research/boundary_20260908/football_past_market'
SPEC = HERE / 'specification.json'
INCUMBENT = 'shot_strength_90d_ridge0.1'
CONTROLS = ('outcome_control', 'outcome_control_shot50', 'elo_odds_reference')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def utc(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def file_record(path):
    return {'path': relative(path), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def runtime():
    return {'python': platform.python_version(), 'executable_sha256': sha(Path(sys.executable).resolve()),
            'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'threadpoolctl': threadpoolctl.__version__,
            'threads': 1}


def input_bindings(spec):
    result = {}
    def visit(value):
        if isinstance(value, dict):
            if {'path', 'bytes', 'sha256'} <= value.keys():
                path = ROOT / value['path']
                if path.stat().st_size != value['bytes'] or sha(path) != value['sha256']:
                    raise ValueError('Specified input changed: ' + str(path))
                result[value['path']] = value['sha256']
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(spec['inputs'])
    for name, expected in spec['inputs']['inherited_source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Inherited source changed: ' + name)
        result[name] = expected
    # Check the original verification graph, rather than accepting fresh hashes alone.
    prior = read(ROOT / spec['inputs']['football_verification']['path'])
    if prior.get('status') != 'passed':
        raise ValueError('Original football verification did not pass')
    for key, field in [('football_selection', 'selection_sha256'), ('football_transfer', 'evaluation_sha256')]:
        if prior[field] != spec['inputs'][key]['sha256']:
            raise ValueError('Original football verification does not bind the reference')
    parent_sources = read(ROOT / spec['inputs']['football_selection']['path'])['source_files']
    if digest(parent_sources) != prior['source_sha256']:
        raise ValueError('Original football source receipt mismatch')
    for name, expected in parent_sources.items():
        if result.get(name) != expected:
            raise ValueError('Inherited source is not bound by the original receipt: ' + name)
    xg = read(ROOT / spec['inputs']['xg_verification']['path'])
    if xg.get('status') != 'passed':
        raise ValueError('Original xG verification did not pass')
    for key in ('xg_selection_issued', 'xg_selection_lock', 'xg_selection_result'):
        item = spec['inputs'][key]
        if xg['bindings'].get(item['path']) != item['sha256']:
            raise ValueError('Original xG verification does not bind the safeguard')
    for item in spec['inputs']['cached_csv_files']:
        manifest = read(ROOT / item['acquisition_manifest'])
        original = {row['name']: row['sha256'] for row in manifest['files']}
        if original.get(Path(item['path']).name) != item['sha256']:
            raise ValueError('Original acquisition manifest differs')
    return result


def sources():
    spec = read(SPEC)
    paths = [*HERE.glob('*.py'), *HERE.glob('*.json'), *HERE.glob('*.md')]
    paths += [ROOT / name for name in spec['inputs']['inherited_source_sha256']]
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
        item = lock[key]
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen execution evidence changed: ' + key)
    return lock


def freeze(out, review_path, feasibility_path):
    if (out / 'design_lock.json').exists():
        raise ValueError('Existing immutable design lock')
    before = sources()
    review = read(review_path)
    if review.get('approved_for_execution_lock') is not True or review.get('source_files') != before:
        raise ValueError('Independent execution review does not approve these exact files')
    feasibility = read(feasibility_path)
    if feasibility.get('limits', {}).get('result') != 'passed' or feasibility.get('scope') != 'synthetic_only_no_historical_inputs':
        raise ValueError('Synthetic feasibility has not passed')
    if feasibility.get('source_files', {}).get(relative(HERE / 'model.py')) != sha(HERE / 'model.py'):
        raise ValueError('Synthetic feasibility used different model code')
    spec = read(SPEC)
    if sha(ROOT / spec['proposal']['path']) != spec['proposal']['sha256']:
        raise ValueError('Predeclared proposal changed')
    bindings = input_bindings(spec)
    command = [sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib', '-p', 'no:cacheprovider', relative(HERE)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    save(out / 'pre_fit_tests.json', {'command': command, 'exit_code': result.returncode,
         'stdout': result.stdout, 'stderr': result.stderr, 'completed_at_utc': now(), 'source_files': before})
    if result.returncode or before != sources():
        raise ValueError('Pre-fit tests failed or sources changed during tests')
    save(out / 'design_lock.json', {'locked_at_utc': now(), 'sources': before, 'inputs': bindings,
         'runtime': runtime(), 'review': file_record(review_path),
         'pre_fit_tests': file_record(out / 'pre_fit_tests.json'),
         'synthetic_feasibility': file_record(feasibility_path),
         'historical_new_candidate_fits_before_lock': 0, 'new_candidate_scores_before_lock': 0})
    print(json.dumps({'stage': 'frozen', 'sha256': sha(out / 'design_lock.json')}), flush=True)


@contextmanager
def stage_attempt(out, stage):
    save(out / f'{stage}_attempt.json', {'started_at_utc': now(), 'design_lock_sha256': sha(out / 'design_lock.json')})
    try:
        yield
    except BaseException as error:
        save(out / f'{stage}_failure.json', {'failed_at_utc': now(), 'exception_type': type(error).__name__,
             'message': str(error), 'advancement_allowed': False})
        raise


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def check_resources(started, spec):
    elapsed = time.monotonic() - started
    rss = peak_rss()
    if rss > spec['resources']['maximum_peak_rss_bytes'] or elapsed > spec['resources']['maximum_stage_wall_seconds']:
        raise RuntimeError('Predeclared stage resource bound exceeded')
    return {'elapsed_seconds': elapsed, 'peak_rss_bytes': rss}


def verify_prepared(out):
    verify_design(out)
    lock = read(out / 'data_lock.json')
    if lock['design_lock_sha256'] != sha(out / 'design_lock.json'):
        raise ValueError('Prepared design differs')
    for item in lock['files'].values():
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Prepared input changed')
    return lock


def prepare(out):
    verify_design(out)
    spec = read(SPEC)
    started = time.monotonic()
    with stage_attempt(out, 'prepare'):
        archive = data.load_archive(spec, root=ROOT)
        metadata = data.metadata_rows(archive)
        references = {phase: data.reference_rows(spec, phase, root=ROOT) for phase in ('selection', 'transfer')}
        for rows in references.values():
            data.validate_reference_metadata(archive, rows)
        support_counts = {}
        for delay in (1, 7):
            count = 0
            rows = references['selection']
            for cutoff in sorted({row['forecast_cutoff_utc'] for row in rows}, key=utc):
                batch = [row for row in rows if row['forecast_cutoff_utc'] == cutoff]
                training, _ = data.quote_rows(archive, cutoff, delay)
                count += sum(data.support_for(batch, training))
            support_counts[str(delay)] = int(count)
        expected = spec['quotes']['expected_selection_support']
        if support_counts != {'1': expected['primary'], '7': expected['sensitivity']}:
            raise ValueError('Observed support differs from predeclared metadata check')
        for phase, rows in references.items():
            if len(rows) != spec[phase]['rows'] or digest([row['match_id'] for row in rows]) != spec[phase]['ordered_original_match_ids_sha256']:
                raise ValueError('Original ordered population changed: ' + phase)
            if len({row['forecast_cutoff_utc'] for row in rows}) != spec[phase]['unique_forecast_clocks']:
                raise ValueError('Original forecast clocks changed')
            save(out / f'references_{phase}.json', rows)
        save(out / 'archive_metadata.json', metadata)
        verify_design(out)
        save(out / 'data_lock.json', {'closed_at_utc': now(), 'design_lock_sha256': sha(out / 'design_lock.json'),
             'files': {name: file_record(out / f'{name}.json') for name in ('references_selection', 'references_transfer', 'archive_metadata')},
             'selection_support': support_counts, 'historical_candidate_fits': 0, 'candidate_scores_computed': False,
             'resources': check_resources(started, spec),
             'references_labels_attached': False,
             'preparation_limit': 'Support checks inspect only admitted earlier quotes; no candidate is fitted and no outcome score is computed.'})
    print(json.dumps({'stage': 'prepared', 'support': support_counts}), flush=True)


def probability_vectors(values):
    return benchmark.validate_probabilities(values)


def fit_and_issue(out, phase, delay, selected=None):
    spec = read(SPEC)
    references = read(out / f'references_{phase}.json')
    archive = data.load_archive(spec, root=ROOT)
    stage = f'{phase}_delay{delay}'
    started = time.monotonic()
    fit_directory = out / f'{stage}_fits'
    fit_directory.mkdir(exist_ok=False)
    if phase == 'selection':
        x, y, diagnostics = data.calibration_rows(archive, delay, min((row['forecast_cutoff_utc'] for row in references), key=utc))
        readout = model.fit_ordered_logit(x, y)
        save(out / f'elo_readout_delay{delay}.json', {'model': readout, 'training': diagnostics,
             'x': list(map(float, x)), 'y': list(map(int, y)), 'closed_at_utc': now(), 'resources': check_resources(started, spec)})
    else:
        readout = read(out / f'elo_readout_delay{delay}.json')['model']
    cursor = data.EloCursor(archive, delay)
    results, fit_records = [], []
    names = spec['candidates'] if selected is None else [selected]
    clocks = sorted({row['forecast_cutoff_utc'] for row in references}, key=utc)
    for index, cutoff in enumerate(clocks):
        batch = [row for row in references if row['forecast_cutoff_utc'] == cutoff]
        training, diagnostics = data.training_rows(archive, cutoff, delay)
        training_ids = [row['match_id'] for row in training]
        if len(training_ids) != len(set(training_ids)) or set(training_ids) & {row['match_id'] for row in batch}:
            raise ValueError('Duplicate or current fixture in admitted history')
        supported = data.support_for(batch, training)
        fitted = {kind: model.fit_strength(training, kind) for kind in ('past_market', 'outcome_control')}
        active = [row for row, support in zip(batch, supported) if support]
        predicted = {kind: model.predict_strength(fitted[kind], active) if active else np.empty((0, 3)) for kind in fitted}
        elo = cursor.query(batch, cutoff)
        elo_p = model.predict_ordered_logit(readout, elo['x'])
        probability_vectors(elo_p)
        cursor_index = 0
        fit_id = f'{index:04d}'
        for i, (row, support) in enumerate(zip(batch, supported)):
            new = deepcopy(row)
            new['support'] = bool(support)
            new['past_market_fit_id'] = fit_id
            incumbent = row['probabilities'][INCUMBENT]
            if not support:
                additions = {name: list(incumbent) for name in [*names, *CONTROLS]}
            else:
                past = predicted['past_market'][cursor_index].tolist()
                outcome = predicted['outcome_control'][cursor_index].tolist()
                additions = {'past_market': past,
                             'past_market_shot50': [(a + b) / 2 for a, b in zip(past, incumbent)],
                             'outcome_control': outcome,
                             'outcome_control_shot50': [(a + b) / 2 for a, b in zip(outcome, incumbent)],
                             'elo_odds_reference': elo_p[i].tolist()}
                additions = {name: additions[name] for name in [*names, *CONTROLS]}
                cursor_index += 1
            probability_vectors(list(additions.values()))
            new['probabilities'].update(additions)
            results.append(new)
        fit_path = fit_directory / f'{fit_id}.json'
        save(fit_path, {'fit_id': fit_id, 'cutoff_utc': cutoff, 'extra_days': delay,
             'training_ids': training_ids, 'training_ids_sha256': digest(training_ids),
             'training_rows_sha256': digest(training), 'training_diagnostics': diagnostics,
             'weights': [row['weight'] for row in training], 'models': fitted,
             'batch_ids': [row['match_id'] for row in batch], 'supported': list(map(bool, supported)),
             'elo': elo, 'resources': check_resources(started, spec)})
        fit_records.append(file_record(fit_path))
        if index % 20 == 0 or index + 1 == len(clocks):
            print(json.dumps({'stage': stage, 'clock': index + 1, 'total_clocks': len(clocks),
                  'history': len(training), 'elapsed_seconds': time.monotonic() - started}), flush=True)
    order = {row['match_id']: i for i, row in enumerate(references)}
    results.sort(key=lambda row: order[row['match_id']])
    if [row['match_id'] for row in results] != [row['match_id'] for row in references]:
        raise ValueError('Issuance population changed')
    save(out / f'{stage}_issued.json', results)
    save(out / f'{stage}_fit_index.json', fit_records)
    closure = {'closed_at_utc': now(), 'forecasts': file_record(out / f'{stage}_issued.json'),
               'rows': len(results), 'labels_attached': False, 'data_lock_sha256': sha(out / 'data_lock.json'),
               'fit_index': file_record(out / f'{stage}_fit_index.json'),
               'elo_readout': file_record(out / f'elo_readout_delay{delay}.json'),
               'optimizer_fits': 2 * len(clocks) + int(phase == 'selection'), 'resources': check_resources(started, spec)}
    save(out / f'{stage}_issuance_lock.json', closure)
    return data.attach_labels(results, archive, forecast_closure_path=out / f'{stage}_issuance_lock.json')


def uncertainty(rows, candidate, reference, block_days, spec):
    adjusted = [{**row, 'season': f"{row['league']}:{row['season']}",
                 'probabilities': {**row['probabilities'], 'dc_equal': row['probabilities'][reference]}} for row in rows]
    result = benchmark.paired_uncertainty(adjusted, candidate, spec, block_days)
    result['difference_candidate_minus_reference'] = result.pop('difference_candidate_minus_dc_equal')
    result['reference'] = reference
    return result


def make_report(rows, spec, phase, selected=None):
    candidates = spec['candidates'] if selected is None else [selected]
    refs = spec[phase]['references']
    names = [*candidates, *refs]
    metrics = {name: benchmark.metrics(rows, name) for name in names}
    if selected is None:
        selected = min(candidates, key=lambda name: metrics[name]['log_loss'])
    comparisons = {candidate: {ref: {str(days): uncertainty(rows, candidate, ref, days, spec)
                   for days in spec['uncertainty']['blocks_calendar_days']} for ref in refs} for candidate in candidates}
    by_country = {country: {name: benchmark.metrics([row for row in rows if row['league'] == country], name) for name in names}
                  for country in sorted({row['league'] for row in rows})}
    by_country_season = {f'{country}:{season}': {name: benchmark.metrics([row for row in rows if row['league'] == country and row['season'] == season], name) for name in names}
                         for country, season in sorted({(row['league'], row['season']) for row in rows})}
    checks = {}
    for ref in refs:
        check = {'relative_nll_gain': 1 - metrics[selected]['log_loss'] / metrics[ref]['log_loss'],
                 'minimum_gain_met': metrics[selected]['log_loss'] <= (1 - spec[phase]['minimum_relative_nll_gain_each_reference']) * metrics[ref]['log_loss'],
                 'negative_28day_upper': comparisons[selected][ref]['28']['percentile_95_interval'][1] < 0}
        if phase == 'transfer':
            check.update({'each_country_improves': all(block[selected]['log_loss'] < block[ref]['log_loss'] for block in by_country.values()),
                          'no_country_season_nll_deterioration': all(block[selected]['log_loss'] <= block[ref]['log_loss'] for block in by_country_season.values()),
                          'pooled_brier_nonworse': metrics[selected]['brier_sum_classes'] <= metrics[ref]['brier_sum_classes'],
                          'country_season_brier_nonworse': all(block[selected]['brier_sum_classes'] <= block[ref]['brier_sum_classes'] for block in by_country_season.values())})
        checks[ref] = check
    passed = all(all(value for key, value in check.items() if key != 'relative_nll_gain') for check in checks.values())
    result = {'rows': len(rows), 'population_sha256': digest([row['match_id'] for row in rows]),
              'selected_candidate': selected, 'metrics': metrics, 'comparisons': comparisons,
              'by_country': by_country, 'by_country_season': by_country_season,
              'checks': checks, 'passes_all_gates': bool(passed), 'promotion': False}
    if phase == 'transfer':
        subset = [row for row in rows if row['match_id'] != 'I1:2024:Fiorentina:Inter']
        result['resumed_fixture_excluded'] = {'rows': len(subset),
             'metrics': {name: benchmark.metrics(subset, name) for name in names},
             'paired_28day': {ref: uncertainty(subset, selected, ref, 28, spec) for ref in refs}}
    return result


def predecessor(out, stage):
    if (out / f'{stage}_failure.json').exists():
        raise ValueError('Failed predecessor cannot authorize advancement')
    lock = read(out / f'{stage}_decision_lock.json')
    if lock['passes_all_gates'] is not True or lock['design_lock_sha256'] != sha(out / 'design_lock.json'):
        raise ValueError('Predecessor did not pass under this design')
    if lock['result_sha256'] != sha(out / f'{stage}.json'):
        raise ValueError('Predecessor result changed')
    result = read(out / f'{stage}.json')
    if result['summary']['passes_all_gates'] is not True or result['summary']['selected_candidate'] != lock['selected_candidate']:
        raise ValueError('Predecessor decision mismatch')
    if result['issuance_lock_sha256'] != sha(out / f'{stage}_issuance_lock.json'):
        raise ValueError('Predecessor issuance changed')
    verify_phase_files(out, stage)
    verification = read(out / f'{stage}_verification.json')
    if verification.get('status') != 'passed' or verification.get('result_sha256') != sha(out / f'{stage}.json'):
        raise ValueError('Predecessor has not passed independent verification')
    return lock['selected_candidate']


def verify_phase_files(out, stage):
    closure = read(out / f'{stage}_issuance_lock.json')
    if closure['labels_attached'] is not False or closure['data_lock_sha256'] != sha(out / 'data_lock.json'):
        raise ValueError('Invalid forecast closure')
    for key in ('forecasts', 'fit_index', 'elo_readout'):
        item = closure[key]
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Closed phase input changed: ' + key)
    index = read(ROOT / closure['fit_index']['path'])
    for item in index:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Closed fit changed')
    return closure


def execute(out, phase, delay):
    started = time.monotonic()
    verify_prepared(out)
    spec = read(SPEC)
    selected = None
    if phase == 'selection' and delay == 7:
        selected = predecessor(out, 'selection_delay1')
    elif phase == 'transfer':
        selected = predecessor(out, 'selection_delay1')
        if predecessor(out, 'selection_delay7') != selected:
            raise ValueError('Sensitivity reselected the candidate')
    stage = f'{phase}_delay{delay}'
    with stage_attempt(out, stage), threadpool_limits(limits=1):
        rows = fit_and_issue(out, phase, delay, selected)
        summary = make_report(rows, spec, phase, selected)
        verify_prepared(out)
        save(out / f'{stage}.json', {'completed_at_utc': now(), 'summary': summary,
             'resources': check_resources(started, spec),
             'status': 'retrospective_reused_dates_no_promotion', 'design_lock_sha256': sha(out / 'design_lock.json'),
             'data_lock_sha256': sha(out / 'data_lock.json'),
             'issuance_lock_sha256': sha(out / f'{stage}_issuance_lock.json')})
        save(out / f'{stage}_decision_lock.json', {'closed_at_utc': now(),
             'selected_candidate': summary['selected_candidate'], 'passes_all_gates': summary['passes_all_gates'],
             'result_sha256': sha(out / f'{stage}.json'), 'design_lock_sha256': sha(out / 'design_lock.json')})
    print(json.dumps({'stage': stage, 'selected': summary['selected_candidate'],
          'nll': {name: values['log_loss'] for name, values in summary['metrics'].items()},
          'checks': summary['checks'], 'passes_all_gates': summary['passes_all_gates']}), flush=True)


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
    else:
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
