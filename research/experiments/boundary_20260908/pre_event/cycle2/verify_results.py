"""Independent numerical/provenance verification of the immutable cycle2 result."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'artifacts/research/boundary_20260908/pre_event/cycle2'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def near(a, b): np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
def pack(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    path = OUT / 'results.json'
    r = json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    for group in ['implementation_manifest', 'input_manifest', 'output_manifest']:
        for name, digest in r[group].items(): assert sha(ROOT / name) == digest, (group, name)
    for group in ['reference_artifact', 'reference_predictions']:
        assert sha(ROOT / r[group]['path']) == r[group]['sha256']
    current = pd.read_csv(OUT / 'predictions.csv', float_precision='round_trip')
    ref = pd.read_csv(ROOT / r['reference_predictions']['path'], float_precision='round_trip')
    assert not current.duplicated(['event_key', 'driver_id']).any()
    frozen = [json.loads(line) for line in (OUT / 'forecasts_before_targets.jsonl').read_text().splitlines()]
    assert [x['event_key'] for x in frozen] == [e['event_key'] for e in r['events']]
    assert json.loads((OUT / 'selection_lock.json').read_text()) == r['selection']
    comparators = r['specification']['comparators']; selected = r['selection']['selected']
    counts = {'events': 0, 'rank_permutations': 0, 'event_metrics': 0, 'forecast_hashes': 0,
              'fitted_predictions_reconstructed': 0, 'whole_event_fallbacks': 0, 'unsupported_driver_scores_checked': 0}
    diagnostic_rows = []
    for event, forecast in zip(r['events'], frozen):
        key = event['event_key']; rows = current.loc[current.event_key.eq(key)]
        prior = ref.loc[ref.event_key.eq(key)].set_index('driver_id').loc[rows.driver_id]
        assert set(rows.driver_id) == set(prior.index) and len(rows) == event['rows']
        near(rows.qualy_position, prior.qualy_position)
        for name in comparators: near(rows[name], prior[name])
        assert event['rehearsal_session_order'] < event['target_session_order']
        assert hashlib.sha256(pack(forecast['predictions'])).hexdigest() == event['forecast_sha256'] == forecast['forecast_sha256']
        models = list(r['all_tried_configurations']) if event['year'] == 2023 else [selected]
        reconstructed = rows[['event_key', 'driver_id', *comparators, *models]].copy()
        for name in ['event_key', *comparators, *models]: reconstructed[name] = reconstructed[name].astype(int)
        assert pack(reconstructed.to_dict('records')) == pack(forecast['predictions'])
        for name in [*models, *comparators]:
            assert sorted(rows[name].tolist()) == list(range(1, len(rows) + 1))
            near((rows[name] - rows.qualy_position).abs().mean(), event['models'][name]['mae'])
            near(kendalltau(rows[name], rows.qualy_position).statistic, event['models'][name]['kendall'])
            near(len(set(rows.nsmallest(3, name).driver_id) & set(rows.nsmallest(3, 'qualy_position').driver_id)) / 3,
                 event['models'][name]['top3_overlap'])
            counts['rank_permutations'] += 1; counts['event_metrics'] += 3
        for name, fit in event['fits'].items():
            if fit['status'].startswith('whole_event_Q2_fallback'):
                near(rows[name], rows.Q2_ridge_rank_residual)
                counts['whole_event_fallbacks'] += 1
                continue
            n = len(rows); base = (rows.Q2_ridge_rank_residual.to_numpy() - 1) * 19 / (n - 1)
            assert fit['drivers'] == rows.driver_id.tolist()
            reliability = np.array(fit['reliability_multiplier'])
            assert np.all((reliability >= 0) & (reliability <= 1))
            assert np.all(np.array(fit['conditional_leave_lap_out_sensitivity']) >= 0)
            for i, d in enumerate(fit['drivers']):
                support = fit['support'][d]
                expected_support = support['matched_laps'] >= 2 and support['peer_drivers'] >= 3 and support['component_drivers'] >= 6
                assert support['supported'] is expected_support
                if not expected_support:
                    assert reliability[i] == 0
                    counts['unsupported_driver_scores_checked'] += 1
            adjustment = r['all_tried_configurations'][name]['blend_strength'] * reliability * np.clip(fit['innovation_equivalent_positions'], -5, 5)
            near(adjustment, fit['applied_adjustment_equivalent_positions'])
            order = np.lexsort((rows.Q2_ridge_rank_residual.to_numpy(), base + adjustment))
            rank = np.empty(n, dtype=int); rank[order] = np.arange(1, n + 1)
            near(rank, rows[name])
            counts['fitted_predictions_reconstructed'] += 1
            if name == selected:
                diagnostic_rows.append({'event_key': key, 'year': event['year'],
                                        'supported_drivers': sum(v['supported'] for v in fit['support'].values()),
                                        'mean_abs_adjustment': float(np.mean(np.abs(adjustment))),
                                        'max_abs_adjustment': float(np.max(np.abs(adjustment))),
                                        'changed_driver_ranks_vs_Q2': int((rank != rows.Q2_ridge_rank_residual.to_numpy()).sum()),
                                        'mean_reliability': float(reliability.mean())})
        counts['events'] += 1; counts['forecast_hashes'] += 1
    selection = [e for e in r['events'] if e['year'] == 2023]
    config = r['all_tried_configurations']
    scores = {m: np.mean([e['models'][m]['mae'] for e in selection]) for m in config}
    for name, score in scores.items(): near(score, r['selection']['selection_variant_event_MAE'][name])
    ties = [m for m in scores if scores[m] <= min(scores.values()) + 1e-12]
    assert selected == min(ties, key=lambda m: (-config[m]['penalty_strength'], config[m]['blend_strength']))
    assert all(k // 100 == 2023 for k in r['selection']['selection_event_keys'])
    for summary in r['summaries'].values():
        events = [e for e in r['events'] if e['event_key'] in summary['event_keys']]
        assert len(events) == summary['events'] and sum(e['rows'] for e in events) == summary['rows']
        for m, score in summary['mean_MAE'].items():
            near(np.mean([e['models'][m]['mae'] for e in events]), score)
            near(np.mean([e['models'][m]['kendall'] for e in events]), summary['mean_Kendall'][m])
        for m, stats in summary['paired_vs'].items():
            d = np.array([e['models'][selected]['mae'] - e['models'][m]['mae'] for e in events]); n = len(d)
            rng = np.random.default_rng(20260908)
            boot = d[rng.integers(n, size=(20000, n))].mean(axis=1)
            starts = rng.integers(n, size=(20000, int(np.ceil(n / 3))))
            indices = ((starts[..., None] + np.arange(3)) % n).reshape(20000, -1)[:, :n]
            loo = (d.sum() - d) / (n - 1)
            near(d.mean(), stats['delta_mean']); near(np.quantile(boot, [.025, .975]), stats['ci95'])
            near(np.quantile(d[indices].mean(axis=1), [.025, .975]), stats['three_event_circular_block_ci95'])
            near([loo.min(), loo.max()], [stats['loo_min'], stats['loo_max']])
            near((boot < 0).mean(), stats['bootstrap_fraction_improving'])
            assert int((d < 0).sum()) == stats['event_wins'] and int((d == 0).sum()) == stats['event_ties']
    assert not r['failed_variants'] and r['promotion'] is False and r['material_success_screen'] is False
    pd.DataFrame(diagnostic_rows).to_csv(OUT / 'selected_measurement_diagnostics.csv', index=False)
    verification = {'status': 'PASS', 'results_sha256': sha(path), 'verifier_sha256': sha(Path(__file__)),
                    'checks': counts, 'rows': len(current), 'selection_events': len(selection),
                    'implementation_files': len(r['implementation_manifest']), 'input_files': len(r['input_manifest']),
                    'output_files': len(r['output_manifest']),
                    'selected_diagnostics_sha256': sha(OUT / 'selected_measurement_diagnostics.csv'),
                    'limits': 'Verifies recorded chronology and exact pre-target forecast bytes; execution-order proof additionally requires source review. Measurement sensitivity is not calibrated uncertainty.'}
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__': main()
