"""Read-only verification of the frozen cycle; writes only its verification JSON."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts/research/boundary_20260908/pre_event/cycle1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def main():
    result_path = OUT / 'results.json'
    result = json.loads(result_path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    for group in ['implementation_manifest', 'input_manifest', 'output_manifest']:
        for path, digest in result[group].items():
            assert sha(ROOT / path) == digest, (group, path)
    for group in ['reference_artifact', 'reference_predictions']:
        assert sha(ROOT / result[group]['path']) == result[group]['sha256'], group
    current = pd.read_csv(OUT / 'predictions.csv', float_precision='round_trip')
    reference = pd.read_csv(ROOT / result['reference_predictions']['path'], float_precision='round_trip')
    assert not current.duplicated(['event_key', 'driver_id']).any()
    assert current.qualy_position.notna().all()
    selected = result['selection']['selected']
    comparators = result['specification']['evaluation']['comparators']
    counts = {'events': 0, 'rank_permutations': 0, 'event_metrics': 0, 'forecast_hashes': 0, 'fitted_histories': 0}
    for event in result['events']:
        key = event['event_key']
        rows = current.loc[current.event_key.eq(key)]
        prior = reference.loc[reference.event_key.eq(key)].set_index('driver_id').loc[rows.driver_id]
        assert len(rows) == event['rows'] and set(rows.driver_id) == set(prior.index)
        close(rows.qualy_position, prior.qualy_position)
        for model in comparators:
            close(rows[model], prior[model])
        models = list(result['all_tried_configurations']) if event['year'] == 2023 else [selected]
        # CSV concatenation promotes early-only integer columns to floats; restore their original types.
        frozen = rows[['event_key', 'driver_id', *models]].copy()
        for col in ['event_key', *models]:
            frozen[col] = frozen[col].astype(int)
        payload = json.dumps(frozen.to_dict('records'), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        assert hashlib.sha256(payload).hexdigest() == event['forecast_sha256']
        counts['forecast_hashes'] += 1
        for model in [*models, *comparators]:
            assert sorted(rows[model].tolist()) == list(range(1, len(rows) + 1))
            close((rows[model] - rows.qualy_position).abs().mean(), event['models'][model]['mae'])
            close(kendalltau(rows[model], rows.qualy_position).statistic, event['models'][model]['kendall'])
            close(len(set(rows.nsmallest(3, model).driver_id) & set(rows.nsmallest(3, 'qualy_position').driver_id)) / 3,
                  event['models'][model]['top3_overlap'])
            counts['rank_permutations'] += 1
            counts['event_metrics'] += 3
        for fit in event['fits'].values():
            assert all(k < key and k // 100 == key // 100 for k in fit['training_event_keys'])
            counts['fitted_histories'] += fit['status'] == 'fitted'
        counts['events'] += 1
    selection = [e for e in result['events'] if e['year'] == 2023]
    configs = result['all_tried_configurations']
    scores = {m: float(np.mean([e['models'][m]['mae'] for e in selection])) for m in configs}
    for model, value in scores.items():
        close(value, result['selection']['selection_variant_event_MAE'][model])
    ties = [m for m in scores if scores[m] <= min(scores.values()) + 1e-12]
    assert selected == min(ties, key=lambda m: (-configs[m]['lambda'], configs[m]['strength'], configs[m]['half_life_events']))
    assert json.loads((OUT / 'selection_lock.json').read_text()) == result['selection']
    for summary in result['summaries'].values():
        events = [e for e in result['events'] if e['event_key'] in summary['event_keys']]
        assert summary['events'] == len(events)
        assert summary['rows'] == sum(e['rows'] for e in events)
        for model, value in summary['mean_MAE'].items():
            close(np.mean([e['models'][model]['mae'] for e in events]), value)
            close(np.mean([e['models'][model]['kendall'] for e in events]), summary['mean_Kendall'][model])
        for model, metrics in summary['paired_vs'].items():
            delta = np.array([e['models'][selected]['mae'] - e['models'][model]['mae'] for e in events])
            n = len(delta)
            rng = np.random.default_rng(20260908)
            samples = delta[rng.integers(n, size=(20000, n))].mean(axis=1)
            starts = rng.integers(n, size=(20000, int(np.ceil(n / 3))))
            indices = ((starts[..., None] + np.arange(3)) % n).reshape(20000, -1)[:, :n]
            blocks = delta[indices].mean(axis=1)
            loo = (delta.sum() - delta) / (n - 1)
            close(delta.mean(), metrics['delta_mean'])
            close(np.quantile(samples, [.025, .975]), metrics['ci95'])
            close(np.quantile(blocks, [.025, .975]), metrics['three_event_circular_block_ci95'])
            close((samples < 0).mean(), metrics['bootstrap_fraction_improving'])
            close([loo.min(), loo.max()], [metrics['loo_min'], metrics['loo_max']])
            assert metrics['event_wins'] == (delta < 0).sum()
            assert metrics['event_ties'] == (delta == 0).sum()
    assert not result['failed_variants'] and result['promotion'] is False
    assert result['material_success_screen'] is False
    report = {'status': 'PASS', 'results_sha256': sha(result_path), 'checks': counts,
              'rows': len(current), 'implementation_files': len(result['implementation_manifest']),
              'input_files': len(result['input_manifest']), 'output_files': len(result['output_manifest']),
              'selection_variants': len(configs), 'selection_events': len(selection),
              'comparison_scope': 'Exact prior forecast and target rows; prior source drift remains explicitly disclosed.',
              'limitations': 'Verification checks recorded chronology and before-label forecast hashes; execution ordering additionally requires source review.',
              'verifier_sha256': sha(Path(__file__))}
    (OUT / 'verification.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
