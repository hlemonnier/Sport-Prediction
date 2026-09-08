"""Replay the promoted global default on all closed fixtures without fitting.

The fit seams return independently verified saved states after checking exact
training membership/content. This checks production selection and complete
output transport; synthetic integration tests exercise actual fresh fitting.
Suggested commit: feat(football): activate the verified full-history default.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

from threadpoolctl import threadpool_limits

from packages.football.mrp import deployment, prediction
from packages.football.mrp.config import PredictionConfig
from packages.football.mrp.data import FixtureRecord, LocalFootballData, MatchRecord
from research.experiments.boundary_20260908.football_deployment_policy import verify as reference
from research.experiments.boundary_20260908.football_deployment_policy.runtime import _check_caller_outputs

ROOT = Path(__file__).resolve().parents[4]
PARENT = ROOT / 'artifacts/research/boundary_20260908/football_deployment_policy'
REPLAY = ROOT / 'artifacts/research/boundary_20260908/football_deployment_replay_v2'
OUT = ROOT / 'artifacts/research/boundary_20260908/football_deployment_integration'


def record(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': reference.sha(path), 'bytes': path.stat().st_size}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def source_files():
    paths = [*ROOT.glob('packages/football/mrp/*.py'), Path(__file__),
        ROOT/'packages/football/tests/test_default_deployment.py', ROOT/'apps/api/src/index.js',
        ROOT/'research/projects/football/Match Result Prediction/Python/run_experiment.py',
        ROOT/'research/projects/football/Match Result Prediction/Python/mrp/__init__.py']
    return {str(p.relative_to(ROOT)): reference.sha(p) for p in paths}


def execute():
    before = source_files()
    approved_record = record(REPLAY/'verification.json')
    approved = reference.read(REPLAY/'verification.json')
    if approved['status'] != 'passed' or approved['passes_all_gates'] is not True:
        raise ValueError('Complete independent verification and every fixed gate must pass')
    design = reference.read(PARENT/'design_lock.json')
    spec = reference.read(ROOT/'research/experiments/boundary_20260908/football_deployment_policy/specification.json')
    bindings = {}
    archive, _, _, _ = reference.reference_inputs(spec, ROOT, design['inputs'], bindings)
    archive = {r['match_id']: r for r in archive}
    closure_record = record(PARENT/'forecast_closure.json')
    closure = reference.read(PARENT/'forecast_closure.json')
    counts = {'prefix_state_injections': 0, 'full_history_state_injections': 0, 'calibrator_restorations': 0}
    records, issued_ids, maximum = [], [], 0.
    with threadpool_limits(limits=1):
        for item in closure['blocks']:
            block = reference.read(reference.bound(item, ROOT, bindings))
            candidate, prior = block['candidate'], block['reference']
            history = []
            for identity in candidate['lineage']['fit_match_ids']:
                values = reference.match_record(archive[identity])
                for key in ('date', 'result_available_at'):
                    values[key] = datetime.fromisoformat(values[key])
                history.append(MatchRecord(**values))
            fixtures = [FixtureRecord(r['match_id'], datetime.fromisoformat(r['forecast_cutoff_utc']), r['season'],
                'epl' if r['league'] == 'E0' else r['league'], None, r['home'], r['away']) for r in block['fixture_metadata']]
            config = PredictionConfig(league=fixtures[0].league, season=fixtures[0].season,
                round_number=1, mode='matchday', shadow_eval=False, weather_enabled=False)
            assert config.football_model == 'dixon' and config.football_calibration == 'auto' and config.goal_strength_half_life_days is None
            dataset = LocalFootballData(None, {}, history, fixtures)
            called = {'prefix': 0, 'full': 0, 'calibrator': 0}
            def prefix_fit(rows, notes, **kwargs):
                called['prefix'] += 1
                if [m.match_id for m in rows] != prior['lineage']['goal_fit_ids'] or kwargs.get('half_life_days') is not None:
                    raise ValueError('Global assessment fit membership or policy changed')
                expected = block['expected_partitions']['fit']['data_sha256']
                if deployment.canonical_sha256([asdict(r) for r in rows]) != expected:
                    raise ValueError('Global assessment fit content changed')
                return deployment.restore_dc(prior['model_state'])
            def full_fit(rows, notes, **kwargs):
                called['full'] += 1
                if deployment.canonical_sha256([asdict(r) for r in rows]) != candidate['lineage']['match_records_sha256'] or kwargs.get('half_life_days') is not None:
                    raise ValueError('Global forecast fit omitted or changed admitted history')
                restored = deployment.restore_dc(candidate['model_state'])
                # Old saved states omit this redundant diagnostic list. The fit
                # seam represents a fresh native fit, which emits it. Reattach
                # only the already verified lineage; learned parameters do not
                # change, and this is explicitly recorded as state reuse.
                ids = [r.match_id for r in rows]
                if restored.diagnostics.get('fit_match_ids', ids) != ids:
                    raise ValueError('Saved fit diagnostic identities conflict with verified lineage')
                restored.diagnostics['fit_match_ids'] = ids
                return restored
            def calibrate(rows, model, notes, **kwargs):
                called['calibrator'] += 1
                if [m.match_id for m in rows] != prior['lineage']['calibration_ids'] or kwargs.get('policy') != 'auto':
                    raise ValueError('Global assessment calibration changed')
                return deployment.restore_calibrator(prior['calibrator_state'])
            with patch.object(prediction, 'load_local_football_data', lambda _: (dataset, [])), \
                    patch.object(prediction, 'fit_dixon_coles', prefix_fit), \
                    patch.object(deployment, 'fit_dixon_coles', full_fit), \
                    patch.object(prediction, 'fit_probability_calibrator_with_policy', calibrate), \
                    patch.object(prediction, 'train_gradient_boosting_model', lambda *args: None):
                result = prediction.run_prediction(config)
            if called != {'prefix': 1, 'full': 1, 'calibrator': 1}:
                raise ValueError('Global default did not select exactly one separate full-history forecast model')
            _check_caller_outputs(candidate['fixtures'], result)
            if result.diagnostics['fixture_policy'] != 'dc_full_admitted_equal_off':
                raise ValueError('Global default effective fixture policy differs')
            if result.diagnostics['assessment_model']['protocol'] != prior['diagnostics']['protocol']:
                raise ValueError('Global default changed historical assessment lineage')
            if result.diagnostics['fixture_model']['lineage']['match_records_sha256'] != candidate['lineage']['match_records_sha256']:
                raise ValueError('Global default fixture lineage differs')
            for state in candidate['fixtures']:
                emitted = result.diagnostics['fixture_distributions'][state['match_id']]
                maximum = max(maximum, reference.close(emitted['score_probability_matrix'], state['matrix'], atol=1e-12))
            for k, source in [('prefix_state_injections', 'prefix'), ('full_history_state_injections', 'full'), ('calibrator_restorations', 'calibrator')]:
                counts[k] += called[source]
            ids = [r['match_id'] for r in result.rows]
            issued_ids.extend(ids)
            records.append({'block_id': block['block_id'], 'rows': len(ids), 'ordered_ids_sha256': reference.ind.digest(ids),
                'complete_output_parity': True, 'fit_input_content_parity': True, 'assessment_lineage_preserved': True})
    expected_ids = [r['match_id'] for r in reference.read(reference.bound(closure['forecasts'], ROOT, bindings))]
    if len(issued_ids) != len(expected_ids) or set(issued_ids) != set(expected_ids) or len(set(issued_ids)) != len(issued_ids):
        raise ValueError('Global forecast replay population differs')
    if source_files() != before or any(reference.sha(ROOT/p) != h for p,h in bindings.items()):
        raise ValueError('Integration source or historical artifact changed during replay')
    for item in (approved_record, closure_record):
        reference.bound(item, ROOT, {})
    return {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_files': before, 'independent_verification': approved_record, 'forecast_closure': closure_record,
        'rows': len(issued_ids), 'blocks': records, 'state_reuse': counts, 'historical_optimizer_fits': 0,
        'restored_diagnostics_note': 'Old full-history states omit fit_match_ids; the replay fit seam reattaches the independently verified full-history lineage to satisfy the fresh-fit diagnostic contract. Learned parameters are unchanged.',
        'maximum_matrix_error': maximum, 'default_policy': 'dc_full_admitted_equal_off',
        'scope': 'Actual global default caller and admission/fit seams with independently verified saved historical states; no historical refits or new performance estimates.'}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    start = time.monotonic()
    if any(OUT.glob('*_failure.json')):
        raise ValueError('Failed integration replay is terminal')
    save(OUT/'integration_attempt.json', {'started_at_utc': datetime.now(timezone.utc).isoformat()})
    try:
        result = execute()
        result['elapsed_seconds'] = time.monotonic()-start
        save(OUT/'integration.json', result)
    except BaseException as exc:
        save(OUT/'integration_failure.json', {'error': str(exc), 'exception_type': type(exc).__name__})
        raise
    print(json.dumps({'status': result['status'], 'rows': result['rows'], 'maximum_matrix_error': result['maximum_matrix_error']}))


if __name__ == '__main__':
    main()
