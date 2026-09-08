"""Freeze, materialize causal inputs, then run the fixed selection experiment.

Historical fitting is possible only after an independently reviewed source lock.
No transfer implementation is enabled by this first selection runner.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from . import data, model

ROOT, HERE = data.ROOT, data.HERE
OUT = ROOT/'artifacts/research/boundary_20260908/football_information'


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def relative(path): return str(Path(path).resolve().relative_to(ROOT))
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False);stream.write('\n')


def sources():
    paths = list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+list(HERE.glob('*.md'))
    # Bind every dependency used by unchanged incumbent reconstruction/scoring.
    parent = read(ROOT/'artifacts/research/boundary_20260908/football/selection.json')
    for name, expected in parent['source_files'].items():
        if data.sha(ROOT/name) != expected:
            raise ValueError('Inherited reconstruction source changed: '+name)
    paths += [ROOT/name for name in parent['source_files']]
    return {relative(path): data.sha(path) for path in sorted(set(paths))}


def input_bindings(spec):
    bindings = dict(spec['inputs'])
    for relative_path in list(bindings):
        if 'source_manifest.json' not in relative_path: continue
        manifest = read(ROOT/relative_path)
        for item in manifest['files']:
            bindings['data/football/performance_20260907/'+item['name']] = item['sha256']
    return bindings


def verify_lock(out):
    lock = read(out/'design_lock.json')
    if lock['sources'] != sources(): raise ValueError('Source differs from reviewed execution lock')
    for path, expected in lock['inputs'].items():
        if data.sha(ROOT/path) != expected: raise ValueError('Frozen input changed: '+path)
    if lock['independent_review']['sha256'] != data.sha(ROOT/lock['independent_review']['path']):
        raise ValueError('Independent review receipt changed')
    return lock


def freeze(out, review_path):
    if (out/'design_lock.json').exists(): raise ValueError('Existing immutable execution lock')
    spec = read(HERE/'specification.json');before = sources()
    if data.sha(ROOT/spec['proposal']['path']) != spec['proposal']['sha256']:
        raise ValueError('Original proposal changed')
    review = read(review_path)
    if review.get('approved_for_execution_lock') is not True or review.get('source_files') != before:
        raise ValueError('Independent review must explicitly approve these exact sources')
    command = [sys.executable, '-m', 'pytest', '-q', '--import-mode=importlib', '-p', 'no:cacheprovider', relative(HERE)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if before != sources(): raise ValueError('Sources changed during tests')
    save(out/'pre_fit_tests.json', {'command': command, 'exit_code': result.returncode,
         'stdout': result.stdout, 'stderr': result.stderr, 'source_files': before, 'completed_at_utc': now()})
    if result.returncode: raise ValueError('Pre-fit tests failed')
    bindings = input_bindings(spec)
    for path, expected in bindings.items():
        if data.sha(ROOT/path) != expected: raise ValueError('Input hash mismatch: '+path)
    save(out/'design_lock.json', {'locked_at_utc': now(), 'sources': before, 'inputs': bindings,
         'proposal_sha256': spec['proposal']['sha256'],
         'independent_review': {'path': relative(review_path), 'sha256': data.sha(review_path)},
         'pre_fit_tests_sha256': data.sha(out/'pre_fit_tests.json'),
         'historical_new_candidate_fits_before_lock': 0, 'new_candidate_scores_before_lock': 0,
         'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'threads': 1}})
    print(json.dumps({'stage': 'frozen', 'design_lock_sha256': data.sha(out/'design_lock.json')}), flush=True)


def prepare(out):
    verify_lock(out);spec = read(HERE/'specification.json')
    save(out/'prepare_attempt.json', {'started_at_utc': now(), 'design_lock_sha256': data.sha(out/'design_lock.json')})
    feature_rows = data.load_feature_cache(spec);frozen = data.load_frozen_incumbents(spec)
    chosen = [r for r in feature_rows if r['season'] in spec['training_season_start_years']]
    pending = [r for r in chosen if r['match_id'] not in frozen]
    if len(pending) != 3800 or len(frozen) != 760 or len(chosen) != 4560:
        raise ValueError('Unexpected fixed training/selection incumbent population')
    # This reconstructs only the previously selected, unchanged shot component.
    # Candidate pooling models are not fitted during input materialization.
    from research.experiments.boundary_20260908.football import run as inherited
    with threadpool_limits(limits=1):
        generated, fits = inherited.predict_corrections(feature_rows, pending,
            {data.INTERNAL: tuple(spec['internal_configuration'])}, 'market_training_reconstruction')
    generated = data.unique(generated);rows = []
    for row in chosen:
        if row['match_id'] in frozen:
            rows.append(data.reuse_incumbent(row, frozen[row['match_id']]))
        else:
            rows.append(data.reuse_incumbent(row, generated[row['match_id']]))
    quotes, bindings = data.load_quotes(spec, spec['training_season_start_years'])
    issued = data.build_issuance(rows, quotes)
    selected = [r for r in issued if r['league'] == 'E0' and r['season'] in (2022, 2023)]
    if {r['match_id'] for r in selected} != set(frozen) or not all(r['market_ready'] for r in selected):
        raise ValueError('Complete frozen selection cohort changed; do not score')
    for row in selected:
        for name in data.INTERNAL_REFERENCES:
            if row['probabilities'][name] != frozen[row['match_id']]['probabilities'][name]:
                raise ValueError('Frozen selection vector was altered')
    save(out/'prepared_inputs.json', issued)
    save(out/'incumbent_training_reconstruction.json', {'fits': fits, 'generated_rows': len(pending),
         'verbatim_reused_selection_rows': len(frozen), 'configuration': spec['internal_configuration']})
    verify_lock(out)
    save(out/'data_lock.json', {'closed_at_utc': now(), 'design_lock_sha256': data.sha(out/'design_lock.json'),
         'prepared_inputs_sha256': data.sha(out/'prepared_inputs.json'), 'rows': len(issued),
         'incumbent_reconstruction_sha256': data.sha(out/'incumbent_training_reconstruction.json'),
         'quote_input_bindings': bindings, 'selection_rows': len(selected), 'labels_attached': False,
         'pooling_models_fitted': 0, 'new_candidate_scores_computed': False,
         'timing_limit': 'Coefficient clocks and carried midnight predictions do not certify availability before unknown historical quote times.'})
    print(json.dumps({'stage': 'prepared', 'rows': len(issued), 'selection_rows': len(selected)}), flush=True)


def fit_and_issue(issued, sources_by_id, spec):
    selected = [r for r in issued if r['league'] == spec['selection']['league'] and r['season'] in spec['selection']['seasons']]
    if len(selected) != spec['selection']['expected_rows'] or not all(r['market_ready'] for r in selected):
        raise ValueError('Selection needs the same complete predeclared cohort')
    results = [];fits = []
    keys = sorted({(r['fit_cutoff_utc'], r['fit_id']) for r in selected}, key=lambda x: (data.utc(x[0]), x[1]))
    for cutoff, fit_id in keys:
        batch = [r for r in selected if r['fit_id'] == fit_id and r['fit_cutoff_utc'] == cutoff]
        training, y, missing = data.training_rows(issued, sources_by_id, cutoff)
        if not training: raise ValueError('No common past training population')
        if {r['match_id'] for r in training} & {r['match_id'] for r in batch}:
            raise ValueError('Current batch entered training')
        market = [r['probabilities']['avg_power'] for r in training]
        prior = [r['probabilities'][data.INTERNAL] for r in training]
        fitted = {};predictions = {}
        for family in ('shared', 'classwise'):
            for internal in (False, True):
                name = ('pool_' if internal else 'market_')+family
                fitted[name] = model.fit(market, prior if internal else None, y, family, internal)
                predictions[name] = fitted[name].predict([r['probabilities']['avg_power'] for r in batch],
                    [r['probabilities'][data.INTERNAL] for r in batch] if internal else None)
        for i, row in enumerate(batch):
            new = deepcopy(row)
            new['probabilities'].update({name: values[i].tolist() for name, values in predictions.items()})
            new['pool_fit_id'] = fit_id;new['pool_fit_cutoff_utc'] = cutoff;results.append(new)
        fits.append({'fit_id': fit_id, 'cutoff_utc': cutoff, 'training_ids': [r['match_id'] for r in training],
             'training_ids_sha256': digest([r['match_id'] for r in training]), 'excluded_missing_snapshot_ids': missing,
             'latest_training_result_available_at': max((r['result_available_at'] for r in training), key=data.utc),
             'models': {name: fitted[name].serialize() for name in fitted}})
        print(json.dumps({'selection_fit': fit_id, 'past_rows': len(training)}), flush=True)
    return sorted(results, key=lambda r: (data.utc(r['forecast_cutoff_utc']), r['match_id'])), fits


def selection_report(rows, spec):
    from research.experiments.boundary_20260908.football import run as inherited
    names = spec['candidates']+spec['references']
    metrics = {name: inherited.b.metrics(rows, name) for name in names}
    selected = min(spec['candidates'], key=lambda name: (metrics[name]['log_loss'], name))
    comparisons = {candidate: {reference: {str(days): inherited.uncertainty(rows, candidate, reference, days, spec)
                   for days in spec['uncertainty']['blocks_calendar_days']} for reference in spec['references']}
                   for candidate in spec['candidates']}
    checks = {reference: {'relative_nll_gain': 1-metrics[selected]['log_loss']/metrics[reference]['log_loss'],
             'minimum_gain_met': metrics[selected]['log_loss'] <= (1-spec['selection']['minimum_relative_nll_gain_vs_every_reference'])*metrics[reference]['log_loss'],
             'negative_upper_28day_interval': comparisons[selected][reference]['28']['percentile_95_interval'][1] < 0}
             for reference in spec['references']}
    advanced = all(r['minimum_gain_met'] and r['negative_upper_28day_interval'] for r in checks.values())
    return {'n': len(rows), 'population_sha256': digest([r['match_id'] for r in rows]),
            'metrics': metrics, 'candidate_comparisons': comparisons, 'selected_candidate': selected,
            'all_reference_gate_checks': checks, 'advances_to_later_evaluation': bool(advanced),
            'decision': 'eligible_for_locked_later_research' if advanced else 'stop_no_new_transfer_or_promotion',
            'by_season': {str(year): {name: inherited.b.metrics([r for r in rows if r['season'] == year], name)
                          for name in names} for year in spec['selection']['seasons']}}


def select(out):
    verify_lock(out);spec = read(HERE/'specification.json');lock = read(out/'data_lock.json')
    if lock['design_lock_sha256'] != data.sha(out/'design_lock.json') or lock['prepared_inputs_sha256'] != data.sha(out/'prepared_inputs.json'):
        raise ValueError('Prepared input lock mismatch')
    if lock['incumbent_reconstruction_sha256'] != data.sha(out/'incumbent_training_reconstruction.json'):
        raise ValueError('Incumbent reconstruction receipt changed')
    save(out/'selection_attempt.json', {'started_at_utc': now(), 'data_lock_sha256': data.sha(out/'data_lock.json')})
    issued = read(out/'prepared_inputs.json');sources_by_id = data.unique(data.load_feature_cache(spec))
    with threadpool_limits(limits=1):
        predictions, fits = fit_and_issue(issued, sources_by_id, spec)
    save(out/'selection_issued.json', predictions);save(out/'selection_fits.json', fits)
    save(out/'selection_issuance_lock.json', {'closed_at_utc': now(), 'rows': len(predictions),
         'issued_sha256': data.sha(out/'selection_issued.json'), 'fits_sha256': data.sha(out/'selection_fits.json'),
         'data_lock_sha256': data.sha(out/'data_lock.json'), 'current_row_labels_attached': False,
         'earlier_selection_labels_used_only_after_strict_availability': True})
    # Outcome attachment occurs only after every immutable forecast is closed.
    labeled = [{**r, 'label': data.label_for(r, sources_by_id)} for r in predictions]
    summary = selection_report(labeled, spec)
    verify_lock(out)
    save(out/'selection.json', {'completed_at_utc': now(), 'status': 'retrospective_snapshot_product_no_promotion',
         'summary': summary, 'design_lock_sha256': data.sha(out/'design_lock.json'),
         'data_lock_sha256': data.sha(out/'data_lock.json'),
         'issuance_lock_sha256': data.sha(out/'selection_issuance_lock.json')})
    save(out/'selection_lock.json', {'closed_at_utc': now(), 'selected_candidate': summary['selected_candidate'],
         'advances_to_later_evaluation': summary['advances_to_later_evaluation'],
         'selection_sha256': data.sha(out/'selection.json'), 'design_lock_sha256': data.sha(out/'design_lock.json'),
         'data_lock_sha256': data.sha(out/'data_lock.json')})
    print(json.dumps({'stage': 'selection_complete', 'summary': summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'prepare', 'select'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    if args.stage == 'freeze':
        if args.review is None: parser.error('--review is required before execution lock')
        freeze(args.out, args.review)
    elif args.stage == 'prepare': prepare(args.out)
    else: select(args.out)


if __name__ == '__main__': main()
