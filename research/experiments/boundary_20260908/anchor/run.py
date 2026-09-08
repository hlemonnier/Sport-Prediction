"""Frozen four-candidate anchor experiment, with selection preceding transfer."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from anchors import EXTRA_FEATURES, KEYS, observed_anchors, points

OUT = ROOT / 'artifacts/research/boundary_20260908/anchor'
OLD = ROOT / 'artifacts/research/frontier_20260907/live'
BASE_PARAMETERS = dict(loss='absolute_error', learning_rate=.06, max_iter=150, max_leaf_nodes=15,
                       min_samples_leaf=80, l2_regularization=10, early_stopping=False, random_state=20260907)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
def load(path): return json.loads(Path(path).read_text())


def snapshot_sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in [HERE / 'run.py', HERE / 'anchors.py', HERE / 'specification.json']}


def check_sources(lock):
    for path, digest in lock['implementation_manifest'].items(): assert sha(ROOT / path) == digest, path


def weights(frame):
    counts = frame.event_key.value_counts()
    return len(frame) / (len(counts) * frame.event_key.map(counts).to_numpy())


def metrics(frame, prediction, baseline):
    rows = pd.DataFrame({'event_key': frame.event_key.to_numpy(),
                         'base': np.abs(np.asarray(baseline) - frame.lap_time_seconds.to_numpy()),
                         'candidate': np.abs(np.asarray(prediction) - frame.lap_time_seconds.to_numpy())})
    ev = rows.groupby('event_key').mean(); delta = (ev.candidate - ev.base).to_numpy(); n = len(ev)
    rng = np.random.default_rng(20260908)
    bootstrap = delta[rng.integers(n, size=(20000, n))].mean(axis=1)
    starts = rng.integers(n, size=(20000, int(np.ceil(n / 3))))
    indices = ((starts[..., None] + np.arange(3)) % n).reshape(20000, -1)[:, :n]
    loo = (delta.sum() - delta) / (n - 1) if n > 1 else np.array([np.nan])
    return {'events': n, 'rows': len(frame), 'baseline_mae': float(ev.base.mean()),
            'candidate_mae': float(ev.candidate.mean()), 'relative_gain': float(1 - ev.candidate.mean() / ev.base.mean()),
            'delta': float(delta.mean()), 'event_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'block3_ci95': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
            'loo_min_delta': float(loo.min()) if n > 1 else None, 'loo_max_delta': float(loo.max()) if n > 1 else None,
            'events_won': int((delta < 0).sum()), 'events_tied': int((delta == 0).sum()),
            'row_baseline_mae': float(rows.base.mean()), 'row_candidate_mae': float(rows.candidate.mean()),
            'per_event': [{'event_key': int(k), 'baseline_mae': float(e.base), 'candidate_mae': float(e.candidate)} for k, e in ev.iterrows()]}


def attach(frame, inventory):
    frames = []; manifest = []
    expected_keys = set(frame.event_key.unique()); seen = set()
    for item in inventory:
        key = item['event_key']; assert key in expected_keys and key not in seen; seen.add(key)
        path = ROOT / item['path']; assert sha(path) == item['sha256'], path
        if key >= 202610: assert 'recent_full_observations' in item['path'], 'narrow transfer input forbidden'
        raw = pd.read_csv(path)
        anchor = observed_anchors(raw, key)
        existing = frame.loc[frame.event_key.eq(key)].copy()
        old_hash = hashlib.sha256(pd.util.hash_pandas_object(existing, index=False).values.tobytes()).hexdigest()
        joined = existing.merge(anchor, on=KEYS, how='left', validate='one_to_one', sort=False)
        assert joined.a_observed_seconds.notna().all() and len(joined) == len(existing)
        np.testing.assert_array_equal(joined.a_observed_seconds, joined.forecast_naive_seconds)
        assert (joined.a_evidence_max_timestamp <= joined.issued_at_timestamp).all()
        assert old_hash == hashlib.sha256(pd.util.hash_pandas_object(joined[existing.columns], index=False).values.tobytes()).hexdigest()
        assert (joined.target_timestamp > joined.issued_at_timestamp).all()
        manifest.append({**item, 'anchor_rows': len(anchor), 'matched_anchor_rows': len(joined),
                         'new_anchor_history_future_pit_timestamps_ignored': sum(int((pd.to_numeric(raw[c], errors='coerce') > raw.Time).sum()) for c in ['PitInTime', 'PitOutTime'])})
        frames.append(joined)
        print('anchor-input', key, len(joined), flush=True)
    assert seen == expected_keys
    result = pd.concat(frames, ignore_index=True)
    assert not result.duplicated(KEYS).any() and np.isfinite(result[EXTRA_FEATURES]).all().all()
    return result, manifest


def fit(frame, features, window=None):
    target = frame.lap_time_seconds.to_numpy() - (frame.forecast_naive_seconds.to_numpy() if window is None else frame[f'a_median{window}_seconds'].to_numpy())
    if window is None: target = np.clip(target, -5, 5)
    model = HistGradientBoostingRegressor(**BASE_PARAMETERS)
    model.fit(frame[features], target, sample_weight=weights(frame))
    return model


def choose(candidate_metrics, candidates):
    best = min(m['candidate_mae'] for m in candidate_metrics.values())
    ties = [n for n, m in candidate_metrics.items() if m['candidate_mae'] <= best + 1e-12]
    return min(ties, key=lambda n: (candidates[n]['policy'] != 'gate', candidates[n]['window']))


def point_output(frame, baseline, predictions):
    columns = [*KEYS, 'year', 'target_lap_number', 'target_timestamp', 'lap_time_seconds', 'target_same_stint',
               'forecast_naive_seconds', *[c for c in frame if c.startswith('a_')]]
    out = frame[columns].copy(); out['baseline_hgb_seconds'] = baseline
    for name, value in predictions.items(): out['prediction_' + name] = value
    return out


def discover():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'design_lock.json').exists(): raise FileExistsError('existing experiment is immutable')
    old_selection = load(OLD / 'selection.json'); spec = load(HERE / 'specification.json')
    features = old_selection['features']; assert len(features) == 80
    lock = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'implementation_manifest': snapshot_sources(),
            'specification': spec, 'features': features + EXTRA_FEATURES,
            'original_selection_sha256': sha(OLD / 'selection.json'),
            'discovery_data_sha256': sha(OLD / 'discovery_data.pkl')}
    write(OUT / 'design_lock.json', lock)
    original = pd.read_pickle(OLD / 'discovery_data.pkl')
    frame, inventory = attach(original, old_selection['input_manifest'])
    train = frame.loc[frame.year.eq(2022)].reset_index(drop=True)
    validation = frame.loc[frame.year.eq(2023)].reset_index(drop=True)
    assert len(frame) == len(train) + len(validation)
    baseline_model = fit(train, features)
    baseline = validation.forecast_naive_seconds.to_numpy() + np.clip(baseline_model.predict(validation[features]), -3, 3)
    baseline_mae = metrics(validation, baseline, baseline)['baseline_mae']
    expected = old_selection['all_candidate_selection_metrics']['hgb_l15_i150']['candidate_mae']
    np.testing.assert_allclose(baseline_mae, expected, rtol=0, atol=1e-12)
    learned = {}; residual = {}; candidate_metrics = {}; diagnostics = {}; predictions = {}
    for window in [3, 5]:
        learned[window] = fit(train, lock['features'], window)
        residual[window] = learned[window].predict(validation[lock['features']])
        print('anchor-model-fit', window, flush=True)
    for name, config in spec['candidates'].items():
        predictions[name], diagnostics[name] = points(validation, residual[config['window']], baseline, config)
        candidate_metrics[name] = metrics(validation, predictions[name], baseline)
        print('selection', name, candidate_metrics[name]['relative_gain'], flush=True)
    selected = choose(candidate_metrics, spec['candidates'])
    best = candidate_metrics[selected]
    advancement = best['relative_gain'] >= .005 and best['loo_max_delta'] < 0
    outputs = point_output(validation, baseline, predictions)
    outputs.to_pickle(OUT / 'selection_predictions.pkl')
    outputs.to_csv(OUT / 'selection_predictions.csv', index=False)
    frame.to_pickle(OUT / 'discovery_anchors.pkl')
    with (OUT / 'selection_models.pkl').open('wb') as handle: pickle.dump({'baseline': baseline_model, 'anchors': learned}, handle)
    selection = {'selected': selected, 'selected_configuration': spec['candidates'][selected],
                 'candidate_metrics': candidate_metrics, 'candidate_diagnostics': diagnostics,
                 'validation_advancement_screen_passed': bool(advancement), 'selection_year': 2023,
                 'selection_event_keys': sorted(int(k) for k in validation.event_key.unique()),
                 'training_years': [2022], 'training_event_keys': sorted(int(k) for k in train.event_key.unique()),
                 'input_manifest': inventory, 'design_lock_sha256': sha(OUT / 'design_lock.json'),
                 'predictions_sha256': sha(OUT / 'selection_predictions.pkl'),
                 'predictions_csv_sha256': sha(OUT / 'selection_predictions.csv'),
                 'discovery_anchors_sha256': sha(OUT / 'discovery_anchors.pkl'),
                 'selection_models_sha256': sha(OUT / 'selection_models.pkl'),
                 'all_dates_previously_exposed': True}
    check_sources(lock)
    write(OUT / 'selection.json', selection)
    print(json.dumps({'selected': selected, 'validation_advancement_screen_passed': bool(advancement),
                      'selected_metrics': best}, indent=2), flush=True)


def transfer():
    if (OUT / 'results.json').exists(): raise FileExistsError('existing transfer is immutable')
    selected = load(OUT / 'selection.json'); lock = load(OUT / 'design_lock.json')
    assert selected['validation_advancement_screen_passed'], 'failed predeclared validation screen'
    check_sources(lock); assert sha(OUT / 'design_lock.json') == selected['design_lock_sha256']
    assert sha(OUT / 'discovery_anchors.pkl') == selected['discovery_anchors_sha256']
    frame = pd.read_pickle(OUT / 'discovery_anchors.pkl')
    config = selected['selected_configuration']; features = lock['features']
    model = fit(frame, features, config['window'])
    with (OUT / 'frozen_model.pkl').open('wb') as handle: pickle.dump(model, handle)
    write(OUT / 'fit_lock.json', {'selection_sha256': sha(OUT / 'selection.json'), 'model_sha256': sha(OUT / 'frozen_model.pkl'),
                                 'fit_years': [2022, 2023], 'fit_event_keys': sorted(int(k) for k in frame.event_key.unique()),
                                 'fit_rows': len(frame), 'base_production_model_sha256': sha(OLD / 'candidate/model.pkl')})
    # Frozen fit above precedes opening the transfer features and targets.
    corrected_path = OLD / 'corrected_input_contract/results.json'; corrected = load(corrected_path)
    old_data_path = OLD / 'corrected_input_contract/transfer_data_and_forecasts.pkl'
    assert sha(old_data_path) == corrected['forecasts_sha256']
    original = pd.read_pickle(old_data_path)
    data, inventory = attach(original, corrected['input_manifest'])
    with (OLD / 'candidate/model.pkl').open('rb') as handle: production = pickle.load(handle)['model']
    base_features = production['features']; assert base_features == features[:80]
    baseline = data.forecast_naive_seconds.to_numpy() + np.clip(production['model'].predict(data[base_features]), -3, 3)
    np.testing.assert_allclose(baseline, data.prediction_hgb_l15_i150, rtol=0, atol=1e-12)
    predictions, diagnostics = points(data, model.predict(data[features]), baseline, config)
    scores = {}
    for name, mask in [('2024', data.year.eq(2024)), ('2025', data.year.eq(2025)),
                       ('2024_2025', data.year.isin([2024, 2025])), ('2026', data.year.eq(2026)),
                       ('2026_recent', data.event_key.ge(202610))]:
        scores[name] = metrics(data.loc[mask], predictions[mask], baseline[mask])
    historical = scores['2024_2025']
    substantial = (historical['relative_gain'] >= .1 and historical['block3_ci95'][1] < 0 and historical['loo_max_delta'] < 0
                   and all(scores[y]['relative_gain'] > 0 for y in ['2024', '2025', '2026']))
    point_output(data, baseline, {selected['selected']: predictions}).to_pickle(OUT / 'transfer_predictions.pkl')
    point_output(data, baseline, {selected['selected']: predictions}).to_csv(OUT / 'transfer_predictions.csv', index=False)
    check_sources(lock)
    result = {'experiment_id': lock['specification']['experiment_id'], 'selected': selected['selected'],
              'selected_configuration': config, 'metrics': scores, 'prediction_diagnostics': diagnostics,
              'substantial_success_screen_passed': bool(substantial), 'promotion': False,
              'all_dates_previously_exposed': True, 'input_manifest': inventory,
              'implementation_manifest': lock['implementation_manifest'], 'design_lock_sha256': sha(OUT / 'design_lock.json'),
              'selection_sha256': sha(OUT / 'selection.json'), 'fit_lock_sha256': sha(OUT / 'fit_lock.json'),
              'frozen_model_sha256': sha(OUT / 'frozen_model.pkl'),
              'reference_corrected_results_sha256': sha(corrected_path), 'reference_transfer_data_sha256': sha(old_data_path),
              'predictions_sha256': sha(OUT / 'transfer_predictions.pkl'), 'predictions_csv_sha256': sha(OUT / 'transfer_predictions.csv')}
    write(OUT / 'results.json', result)
    print(json.dumps({'selected': selected['selected'], 'metrics': scores, 'substantial_success_screen_passed': bool(substantial)}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['discover', 'transfer'])
    {'discover': discover, 'transfer': transfer}[parser.parse_args().command]()
