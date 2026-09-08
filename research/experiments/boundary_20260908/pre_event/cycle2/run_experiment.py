"""Frozen matched-lap measurement experiment; all scored dates retrospective."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / 'research/projects/F1/rising_qualification_prediction/Python')]
from measurement import predict
from run_qualifying_pairwise_challenger_backtest import _qualifying_target_frame
from measurement import clean_laps

REFERENCE = ROOT / 'artifacts/research/performance_20260907/pre_event/cycle1/results.json'
COMPARATORS = ['baseline_qualifying', 'Q1_fixed_rank_blend', 'Q2_ridge_rank_residual']
FEATURES = ['event_key', 'driver_id', 'team_id', 'year', 'raw_anchor', 'rehearsal_source', *COMPARATORS]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + '\n')


def select(events, candidates):
    eligible = [e for e in events if e['year'] == 2023]
    assert len(eligible) == 16
    scores = {name: float(np.mean([e['models'][name]['mae'] for e in eligible])) for name in candidates}
    ties = [name for name in candidates if scores[name] <= min(scores.values()) + 1e-12]
    chosen = min(ties, key=lambda name: (-candidates[name]['penalty_strength'], candidates[name]['blend_strength']))
    return {'selected': chosen, 'parameters': candidates[chosen], 'selection_event_keys': [e['event_key'] for e in eligible],
            'selection_variant_event_MAE': scores, 'selection_year': 2023}


def paired(values):
    values = np.asarray(values, dtype=float); n = len(values)
    rng = np.random.default_rng(20260908)
    bootstrap = values[rng.integers(n, size=(20000, n))].mean(axis=1)
    starts = rng.integers(n, size=(20000, int(np.ceil(n / 3))))
    indices = ((starts[..., None] + np.arange(3)) % n).reshape(20000, -1)[:, :n]
    blocks = values[indices].mean(axis=1)
    loo = (values.sum() - values) / (n - 1)
    return {'events': n, 'delta_mean': values.mean(), 'ci95': np.quantile(bootstrap, [.025, .975]),
            'three_event_circular_block_ci95': np.quantile(blocks, [.025, .975]),
            'loo_min': loo.min(), 'loo_max': loo.max(), 'event_wins': int((values < 0).sum()),
            'event_ties': int((values == 0).sum()), 'bootstrap_fraction_improving': (bootstrap < 0).mean(),
            'interpretation': 'retrospective paired resampling frequency; not a posterior or promotion test'}


def latest_pre_q(metadata):
    qualifying = next(s for s in metadata['sessions'] if s['session_type'] == 'qualifying')
    candidates = [s for s in metadata['sessions'] if int(s['session_order']) < int(qualifying['session_order'])
                  and s.get('completed', True) is True
                  and (s['session_name'].lower() == 'practice 3' or s['session_type'] == 'sprint_qualifying')]
    if not candidates:
        raise ValueError('no completed causal FP3/SprintQualifying rehearsal')
    return max(candidates, key=lambda s: int(s['session_order'])), qualifying


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists() or (output / 'forecasts_before_targets.jsonl').exists():
        raise FileExistsError('Use a new output directory; existing forecast/result is immutable')
    spec_path = HERE / 'spec.json'; spec = json.loads(spec_path.read_text())
    ref = json.loads(REFERENCE.read_text()); prior_path = REFERENCE.with_name('predictions.csv')
    assert sha(prior_path) == ref['output_manifest'][str(prior_path.relative_to(ROOT))]
    reference = pd.read_csv(prior_path, usecols=FEATURES, float_precision='round_trip')
    assert set(reference.columns) == set(FEATURES) and 'qualy_position' not in reference
    source = {str(spec_path.relative_to(ROOT)): sha(spec_path)}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.is_file() and ROOT in path.parents and path.suffix == '.py' and '.venv' not in str(path.relative_to(ROOT)):
                source[str(path.relative_to(ROOT))] = sha(path)
    drift = [{'path': p, 'historical_sha256': h, 'current_sha256': sha(ROOT / p)}
             for p, h in ref['implementation_manifest'].items() if sha(ROOT / p) != h]
    inputs = {}; rows = []; events = []; excluded = []; failures = []; lock = None
    for metadata_path in sorted((ROOT / 'data/f1/raw/weekends').glob('20*/round_*/weekend_metadata.json')):
        year = int(metadata_path.parent.parent.name)
        if not 2023 <= year <= 2026: continue
        key = year * 100 + int(metadata_path.parent.name.split('_')[1])
        if year >= 2024 and lock is None:
            lock = select(events, spec['candidates'])
            save(output / 'selection_lock.json', lock)
        inputs[str(metadata_path.relative_to(ROOT))] = sha(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        try:
            latest, qualifying = latest_pre_q(metadata)
        except ValueError as exc:
            excluded.append({'event_key': key, 'error': str(exc)}); continue
        frame = reference.loc[reference.event_key.eq(key)].reset_index(drop=True).copy()
        assert len(frame) >= 19 and not frame.driver_id.duplicated().any(), key
        path = metadata_path.parent / Path(latest['laps_path']).name
        target_path = metadata_path.parent / Path(qualifying['results_path']).name
        inputs[str(path.relative_to(ROOT))] = sha(path)
        assert inputs[str(path.relative_to(ROOT))] == ref['input_manifest'][str(path.relative_to(ROOT))]
        laps = clean_laps(pd.read_csv(path))
        active = spec['candidates'] if year == 2023 else {lock['selected']: lock['parameters']}
        frozen = frame[['event_key', 'driver_id', *COMPARATORS]].copy(); diagnostics = {}
        feature_hash = hashlib.sha256(canonical(frame.to_dict('records'))).hexdigest()
        for name, config in active.items():
            try:
                frozen[name], diagnostics[name] = predict(frame, laps, **config)
            except Exception as exc:
                failures.append({'event_key': key, 'variant': name, 'error': repr(exc)})
                save(output / 'failed_attempts.json', failures)
                raise
        assert feature_hash == hashlib.sha256(canonical(frame.to_dict('records'))).hexdigest()
        forecast_payload = canonical(frozen.to_dict('records'))
        forecast_hash = hashlib.sha256(forecast_payload).hexdigest()
        # This append/close happens before the target CSV is opened below.
        with (output / 'forecasts_before_targets.jsonl').open('ab') as handle:
            handle.write(canonical({'event_key': key, 'forecast_sha256': forecast_hash,
                                    'predictions': frozen.to_dict('records')}) + b'\n')
        target, _ = _qualifying_target_frame(target_path)
        inputs[str(target_path.relative_to(ROOT))] = sha(target_path)
        assert inputs[str(target_path.relative_to(ROOT))] == ref['input_manifest'][str(target_path.relative_to(ROOT))]
        assert set(target.driver_id) == set(frame.driver_id), (key, 'target roster mismatch')
        scored = frozen.merge(target[['driver_id', 'qualy_position']], on='driver_id', validate='one_to_one')
        scored['year'] = year
        models = {}
        for name in [*active, *COMPARATORS]:
            assert sorted(scored[name].tolist()) == list(range(1, len(scored) + 1))
            models[name] = {'mae': float((scored[name] - scored.qualy_position).abs().mean()),
                            'kendall': float(kendalltau(scored[name], scored.qualy_position).statistic),
                            'top3_overlap': len(set(scored.nsmallest(3, name).driver_id)
                                                & set(scored.nsmallest(3, 'qualy_position').driver_id)) / 3}
        if year >= 2024: scored['selected_measurement'] = scored[lock['selected']]
        rows.append(scored)
        events.append({'event_key': key, 'year': year, 'rows': len(scored), 'models': models,
                       'fits': diagnostics, 'forecast_sha256': forecast_hash, 'feature_sha256': feature_hash,
                       'rehearsal_session_order': int(latest['session_order']),
                       'target_session_order': int(qualifying['session_order']),
                       'rehearsal_source': str(frame.rehearsal_source.iloc[0])})
        print(key, 'scored', len(scored), {m: v['mae'] for m, v in models.items()}, flush=True)
    assert not failures and lock is not None
    assert {y: sum(e['year'] == y for e in events) for y in [2023, 2024, 2025, 2026]} == {2023: 16, 2024: 24, 2025: 24, 2026: 9}
    summaries = {}
    for label, years in [('2024', [2024]), ('2025', [2025]), ('pooled_2024_2025', [2024, 2025]), ('exposed_2026', [2026])]:
        sample = [e for e in events if e['year'] in years]; selected = lock['selected']
        summaries[label] = {'events': len(sample), 'rows': sum(e['rows'] for e in sample),
                            'event_keys': [e['event_key'] for e in sample],
                            'mean_MAE': {m: np.mean([e['models'][m]['mae'] for e in sample]) for m in [selected, *COMPARATORS]},
                            'mean_Kendall': {m: np.mean([e['models'][m]['kendall'] for e in sample]) for m in [selected, *COMPARATORS]},
                            'paired_vs': {m: paired([e['models'][selected]['mae'] - e['models'][m]['mae'] for e in sample]) for m in COMPARATORS}}
    for path, digest in source.items(): assert sha(ROOT / path) == digest, ('source changed during run', path)
    pd.concat(rows, ignore_index=True).to_csv(output / 'predictions.csv', index=False)
    result = {'schema_version': 'matched_lap_hierarchical_measurement_v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
              'specification': spec, 'specification_sha256': sha(spec_path), 'selection': lock,
              'summaries': summaries, 'events': events, 'excluded_events': excluded, 'failed_variants': failures,
              'all_tried_configurations': spec['candidates'], 'promotion': False,
              'material_success_screen': bool(summaries['pooled_2024_2025']['paired_vs']['Q2_ridge_rank_residual']['delta_mean'] <= -.15
                                               and summaries['exposed_2026']['paired_vs']['Q1_fixed_rank_blend']['delta_mean'] <= 0),
              'implementation_manifest': source, 'input_manifest': inputs,
              'reference_artifact': {'path': str(REFERENCE.relative_to(ROOT)), 'sha256': sha(REFERENCE)},
              'reference_predictions': {'path': str(prior_path.relative_to(ROOT)), 'sha256': sha(prior_path)},
              'reference_source_drift': drift,
              'reference_scope': 'Immutable previous forecast/causal feature bytes and unchanged target/lap bytes. Old broad source closure is not relabelled as current.',
              'output_manifest': {str((output / p).relative_to(ROOT)): sha(output / p)
                                  for p in ['predictions.csv', 'selection_lock.json', 'forecasts_before_targets.jsonl']}}
    save(output / 'results.json', result)
    print(json.dumps(clean({'selection': lock, 'summaries': summaries, 'material_success_screen': result['material_success_screen']}), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/research/boundary_20260908/pre_event/cycle2')
    main(parser.parse_args().output.resolve())
