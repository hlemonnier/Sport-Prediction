"""Verify the failed discovery without opening transfer labels or fitting models."""
from pathlib import Path
import hashlib
import importlib.util
import json
import pickle
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts/research/boundary_20260908/anchor'
OLD = ROOT / 'artifacts/research/frontier_20260907/live'
sys.path.insert(0, str(HERE))
from anchors import observed_anchors, KEYS, EXTRA_FEATURES


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
def near(a, b): np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def main():
    lock = load(OUT / 'design_lock.json'); selection = load(OUT / 'selection.json')
    assert sha(OUT / 'design_lock.json') == selection['design_lock_sha256']
    for name, value in lock['implementation_manifest'].items(): assert sha(ROOT / name) == value
    for name, field in [('selection_predictions.pkl', 'predictions_sha256'), ('selection_predictions.csv', 'predictions_csv_sha256'),
                        ('discovery_anchors.pkl', 'discovery_anchors_sha256'), ('selection_models.pkl', 'selection_models_sha256')]:
        assert sha(OUT / name) == selection[field]
    assert sha(OLD / 'discovery_data.pkl') == lock['discovery_data_sha256']
    assert sha(OLD / 'selection.json') == lock['original_selection_sha256']
    for item in selection['input_manifest']: assert sha(ROOT / item['path']) == item['sha256']
    original = pd.read_pickle(OLD / 'discovery_data.pkl')
    enriched = pd.read_pickle(OUT / 'discovery_anchors.pkl')
    points = pd.read_pickle(OUT / 'selection_predictions.pkl')
    assert len(original) == len(enriched) and set(enriched.year) == {2022, 2023}
    old = original.set_index(KEYS).sort_index(); new = enriched.set_index(KEYS).sort_index()
    pd.testing.assert_frame_equal(old, new[old.columns])
    assert len(lock['features']) == 86 and lock['features'][-6:] == EXTRA_FEATURES
    val = enriched.loc[enriched.year.eq(2023)].reset_index(drop=True)
    assert len(val) == 20007 and set(points.year) == {2023} and not points.duplicated(KEYS).any()
    for key in [*KEYS, 'target_lap_number', 'target_timestamp', 'lap_time_seconds', 'target_same_stint']:
        np.testing.assert_array_equal(val[key], points[key])
    assert (val.a_evidence_max_timestamp <= val.issued_at_timestamp).all()
    assert (val.target_timestamp > val.issued_at_timestamp).all()
    with (OUT / 'selection_models.pkl').open('rb') as handle: models = pickle.load(handle)
    baseline = val.forecast_naive_seconds.to_numpy() + np.clip(models['baseline'].predict(val[lock['features'][:80]]), -3, 3)
    near(baseline, points.baseline_hgb_seconds)
    truth = val.lap_time_seconds.to_numpy(); errors = np.abs(baseline - truth)
    scores = {}; gate_counts = {}
    for name, config in lock['specification']['candidates'].items():
        window = config['window']
        candidate = val[f'a_median{window}_seconds'].to_numpy() + models['anchors'][window].predict(val[lock['features']])
        gate = np.ones(len(val), dtype=bool) if config['policy'] == 'full' else (
            (val[f'a_count{window}'].to_numpy() >= 3) & (np.abs(val[f'a_gap{window}'].to_numpy()) >= 1))
        use = gate & np.isfinite(candidate) & (candidate > 0)
        predicted = np.where(use, candidate, baseline)
        near(predicted, points['prediction_' + name])
        np.testing.assert_array_equal(predicted[~gate], baseline[~gate])
        gate_counts[name] = int(gate.sum())
        ev = pd.DataFrame({'event_key': val.event_key, 'base': errors,
                            'candidate': np.abs(predicted - truth)}).groupby('event_key').mean()
        delta = (ev.candidate - ev.base).to_numpy(); n = len(delta)
        rng = np.random.default_rng(20260908)
        boot = delta[rng.integers(n, size=(20000, n))].mean(axis=1)
        starts = rng.integers(n, size=(20000, int(np.ceil(n / 3))))
        indices = ((starts[..., None] + np.arange(3)) % n).reshape(20000, -1)[:, :n]
        loo = (delta.sum() - delta) / (n - 1)
        stats = selection['candidate_metrics'][name]
        near(ev.base.mean(), stats['baseline_mae']); near(ev.candidate.mean(), stats['candidate_mae'])
        near(delta.mean(), stats['delta']); near(1 - ev.candidate.mean() / ev.base.mean(), stats['relative_gain'])
        near(np.quantile(boot, [.025, .975]), stats['event_ci95'])
        near(np.quantile(delta[indices].mean(axis=1), [.025, .975]), stats['block3_ci95'])
        near([loo.min(), loo.max()], [stats['loo_min_delta'], stats['loo_max_delta']])
        assert int((delta < 0).sum()) == stats['events_won']
        scores[name] = float(ev.candidate.mean())
    configs = lock['specification']['candidates']
    ties = [name for name, value in scores.items() if value <= min(scores.values()) + 1e-12]
    assert selection['selected'] == min(ties, key=lambda name: (configs[name]['policy'] != 'gate', configs[name]['window']))
    best = selection['candidate_metrics'][selection['selected']]
    passed = best['relative_gain'] >= .005 and best['loo_max_delta'] < 0
    assert selection['validation_advancement_screen_passed'] is passed is False
    assert not (OUT / 'frozen_model.pkl').exists() and not (OUT / 'transfer_predictions.pkl').exists()
    prefix_cases = []
    for key in [202201, 202301]:
        item = next(x for x in selection['input_manifest'] if x['event_key'] == key)
        raw = pd.read_csv(ROOT / item['path']); cutoff = float(raw.Time.median())
        full = observed_anchors(raw, key)
        prefix = observed_anchors(raw.loc[raw.Time.le(cutoff)], key)
        expected = full.loc[full.issued_at_timestamp.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix.reset_index(drop=True), expected)
        poisoned = raw.copy()
        poisoned.loc[poisoned.Time.gt(cutoff), 'LapTime'] = 9000.
        poisoned.loc[poisoned.Time.gt(cutoff), 'Compound'] = 'WET'
        future = observed_anchors(poisoned, key)
        pd.testing.assert_frame_equal(future.loc[future.issued_at_timestamp.le(cutoff)].reset_index(drop=True), expected)
        prefix_cases.append({'event_key': key, 'prefix_rows': len(prefix), 'prefix_and_future_poison': 'PASS'})
    selected_points = points['prediction_' + selection['selected']].to_numpy()
    gap = np.abs(val.a_gap3.to_numpy()); diagnostic = []
    for name, mask in [('anchor_gap_below1', gap < 1), ('anchor_gap_1to3', (gap >= 1) & (gap <= 3)), ('anchor_gap_above3', gap > 3)]:
        diagnostic.append({'group': name, 'rows': int(mask.sum()), 'baseline_row_mae': float(errors[mask].mean()),
                            'selected_row_mae': float(np.abs(selected_points[mask] - truth[mask]).mean()),
                            'row_mae_delta': float(np.mean(np.abs(selected_points[mask] - truth[mask]) - errors[mask]))})
    write(OUT / 'selection_failure_diagnostics.json', {'scope': '2023-only fixed observed-gap groups; no variants selected or transfer labels opened', 'groups': diagnostic})
    report = {'status': 'PASS', 'selection_sha256': sha(OUT / 'selection.json'), 'verifier_sha256': sha(Path(__file__)),
              'implementation_files': len(lock['implementation_manifest']), 'input_files': len(selection['input_manifest']),
              'immutable_original_rows': len(enriched), 'legacy_features_unchanged': 80, 'new_features': 6,
              'selection_rows': len(val), 'selection_events': int(val.event_key.nunique()),
              'model_forecasts_reconstructed': len(val) * 5, 'four_candidate_metrics_reconstructed': True,
              'gate_rows': gate_counts, 'real_causal_prefix_cases': prefix_cases,
              'advancement_screen_passed': False, 'transfer_scored': False}
    write(OUT / 'verification.json', report)
    result = {'experiment_id': lock['specification']['experiment_id'], 'status': 'rejected_at_2023_selection',
              'selected': selection['selected'], 'validation_advancement_screen_passed': False,
              'candidate_selection_metrics': selection['candidate_metrics'], 'transfer_scored': False,
              'composition_eligible_branch': 'unchanged_current_HGB', 'promotion': False,
              'all_evaluation_dates_previously_exposed': True,
              'selection_sha256': sha(OUT / 'selection.json'), 'design_lock_sha256': sha(OUT / 'design_lock.json'),
              'verification_sha256': sha(OUT / 'verification.json'), 'selection_predictions_sha256': sha(OUT / 'selection_predictions.pkl'),
              'failure_diagnostics_sha256': sha(OUT / 'selection_failure_diagnostics.json')}
    write(OUT / 'results.json', result)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
