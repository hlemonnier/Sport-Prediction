"""Fixed paired evaluation of already closed deployment score distributions.

Suggested commit: research(football): evaluate complete deployment forecasts.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
import math

import numpy as np

REFERENCE = 'dc_frozen_prefix_auto'
CANDIDATE = 'dc_full_admitted_equal_off'
POLICIES = (REFERENCE, CANDIDATE)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def losses(rows, policy):
    """The saved matrix is immutable: scoring never builds or extends support."""
    result = []
    for row in rows:
        goals, label = row['goals'], row['label']
        if len(goals) != 2 or any(type(x) is not int or x < 0 for x in goals):
            raise ValueError('Nonnegative integer score required')
        h, a = goals
        if type(label) is not int or label != (0 if h > a else 1 if h == a else 2):
            raise ValueError('Outcome and score disagree')
        output = row['outputs'][policy]
        p = np.asarray(output['hda'], dtype=float)
        matrix = np.asarray(output['matrix'], dtype=float)
        if (p.shape != (3,) or matrix.ndim != 2 or min(matrix.shape, default=0) < 1
                or not np.isfinite(p).all() or not np.isfinite(matrix).all()
                or (p < 0).any() or (matrix < 0).any()
                or abs(p.sum()-1) > 1e-10 or abs(matrix.sum()-1) > 1e-10):
            raise ValueError('Invalid complete probability output')
        home, away = np.indices(matrix.shape)
        marginal = np.asarray([matrix[home > away].sum(), matrix[home == away].sum(), matrix[home < away].sum()])
        if np.max(np.abs(marginal-p)) > 1e-10:
            raise ValueError('HDA and score matrix disagree')
        inside = h < matrix.shape[0] and a < matrix.shape[1]
        mass = float(matrix[h, a]) if inside else 0.
        winner = int(p.argmax())
        result.append({'nll': -math.log(float(p[label])) if p[label] > 0 else None,
            'joint': -math.log(mass) if mass > 0 else None,
            'joint_cause': None if mass > 0 else 'zero_mass' if inside else 'out_of_support',
            'brier': float(np.square(p-np.eye(3)[label]).sum()),
            'confidence': float(p[winner]), 'correct': int(winner == label), 'p': p.tolist()})
    return result


def metrics(rows, scores):
    n = len(rows)
    def mean(key):
        values = [r[key] for r in scores]
        return float(np.mean(values)) if all(x is not None for x in values) else None
    bins, ece = [], 0.
    for k in range(10):
        group = [r for r in scores if k/10 <= r['confidence'] and (r['confidence'] < (k+1)/10 if k < 9 else r['confidence'] <= 1)]
        if group:
            confidence = float(np.mean([r['confidence'] for r in group]))
            accuracy = float(np.mean([r['correct'] for r in group]))
            bins.append({'lower': k/10, 'upper': (k+1)/10, 'n': len(group), 'mean_confidence': confidence, 'accuracy': accuracy})
            ece += len(group)/n * abs(confidence-accuracy)
    nll, joint = mean('nll'), mean('joint')
    return {'n': n, 'log_loss': nll, 'brier_sum_classes': mean('brier'),
        'joint_score_log_loss': joint, 'status': 'finite' if nll is not None and joint is not None else 'nonfinite_loss',
        'accuracy': mean('correct'), 'top_label_ece_10_bins': ece, 'calibration_bins': bins,
        'class_mean_probability_minus_frequency':
            (np.asarray([s['p'] for s in scores])-np.eye(3)[[r['label'] for r in rows]]).mean(axis=0).tolist()}


def paired(rows, scores, days, spec):
    """Both losses use the very same sampled calendar blocks in each stratum."""
    count, seed = spec['uncertainty']['resamples'], spec['uncertainty']['seed']
    if type(count) is not int or count <= 0 or type(seed) is not int or seed < 0 or days not in (1, 7, 28):
        raise ValueError('Invalid fixed bootstrap settings')
    delta = {}
    result = {}
    for name, key in [('one_x_two', 'nll'), ('joint_score', 'joint')]:
        pairs = list(zip(scores[CANDIDATE], scores[REFERENCE], strict=True))
        finite = all(c[key] is not None and r[key] is not None for c, r in pairs)
        if finite:
            delta[name] = np.asarray([c[key]-r[key] for c, r in pairs])
        result[name] = {'difference_candidate_minus_reference': None, 'percentile_95_interval': None,
            'bootstrap_fraction_difference_below_zero': None, 'block_calendar_days': days,
            'blocks_by_season': {}, 'resamples': count, 'reference': REFERENCE, 'status': 'nonfinite_loss'}
    rng = np.random.default_rng(seed)
    boot = {key: np.zeros(count) for key in delta}
    for stratum in sorted({f"{r['league']}:{r['season']}" for r in rows}):
        indices = [i for i, r in enumerate(rows) if f"{r['league']}:{r['season']}" == stratum]
        dates = [date.fromisoformat(rows[i]['day']) for i in indices]
        anchor = min(dates)
        blocks = {}
        for i, day in zip(indices, dates, strict=True):
            blocks.setdefault((day-anchor).days//days, []).append(i)
        sizes = np.asarray([len(b) for b in blocks.values()])
        selected = rng.integers(0, len(blocks), size=(count, len(blocks)))
        for name, values in delta.items():
            sums = np.asarray([sum(values[b]) for b in blocks.values()])
            boot[name] += len(indices)/len(rows) * sums[selected].sum(axis=1)/sizes[selected].sum(axis=1)
            result[name]['blocks_by_season'][stratum] = len(blocks)
    for name, values in delta.items():
        result[name].update(status='finite', difference_candidate_minus_reference=float(values.mean()),
            percentile_95_interval=np.quantile(boot[name], [.025, .975]).tolist(),
            bootstrap_fraction_difference_below_zero=float(np.mean(boot[name] < 0)))
    return result


def evaluate(rows, spec):
    population = spec['population']
    ids = [r['match_id'] for r in rows]
    if (len(rows) != population['rows'] or len(set(ids)) != len(ids)
            or any(not isinstance(x, str) or not x for x in ids)
            or digest(ids) != population['ordered_original_match_ids_sha256']):
        raise ValueError('Complete ordered fixed population required')
    counts = Counter((r['league'], r['season']) for r in rows)
    expected = {(country, season): population['rows_each_country_season']
        for country in population['countries'] for season in population['season_start_years']}
    if counts != expected or any(type(r['season']) is not int for r in rows):
        raise ValueError('Country-season population differs')
    if any(set(r['outputs']) != set(POLICIES) for r in rows):
        raise ValueError('Exactly the two fixed policies required')
    scores = {p: losses(rows, p) for p in POLICIES}
    report = {'rows': len(rows), 'population_sha256': digest(ids),
        'metrics': {p: metrics(rows, scores[p]) for p in POLICIES},
        'by_country': {}, 'by_country_season': {},
        'paired': {str(days): paired(rows, scores, days, spec) for days in spec['uncertainty']['blocks_calendar_days']},
        'promotion': False, 'nonfinite_losses': []}
    for policy in POLICIES:
        for row, score in zip(rows, scores[policy], strict=True):
            if score['joint'] is None or score['nll'] is None:
                report['nonfinite_losses'].append({'match_id': row['match_id'], 'policy': policy,
                    'joint_cause': score['joint_cause'], 'one_x_two_zero': score['nll'] is None})
    for country in population['countries']:
        for season in [None, *population['season_start_years']]:
            idx = [i for i, r in enumerate(rows) if r['league'] == country and (season is None or r['season'] == season)]
            bucket = {p: metrics([rows[i] for i in idx], [scores[p][i] for i in idx]) for p in POLICIES}
            report['by_country' if season is None else 'by_country_season'][country if season is None else f'{country}:{season}'] = bucket
    def better(bucket, key, strict=False):
        c, r = bucket[CANDIDATE][key], bucket[REFERENCE][key]
        return c is not None and r is not None and (c < r if strict else c <= r)
    def interval_pass(name, strict=False):
        pair = report['paired'][str(spec['uncertainty']['gate_block_days'])][name]
        return pair['status'] == 'finite' and (pair['percentile_95_interval'][1] < 0 if strict else pair['percentile_95_interval'][1] <= 0)
    c, r = report['metrics'][CANDIDATE]['log_loss'], report['metrics'][REFERENCE]['log_loss']
    gain = (r-c)/r if c is not None and r is not None and r > 0 else None
    report['relative_one_x_two_nll_gain'] = gain
    report['checks'] = {
        'complete_population': True, 'all_losses_finite': not report['nonfinite_losses'],
        'one_x_two_minimum_gain': gain is not None and gain >= spec['acceptance']['one_x_two']['minimum_pooled_relative_nll_gain'],
        'one_x_two_paired28_upper_negative': interval_pass('one_x_two', True),
        'each_country_one_x_two_improves': all(better(v, 'log_loss', True) for v in report['by_country'].values()),
        'each_country_season_one_x_two_nonworse': all(better(v, 'log_loss') for v in report['by_country_season'].values()),
        'each_country_season_brier_nonworse': all(better(v, 'brier_sum_classes') for v in report['by_country_season'].values()),
        'pooled_joint_score_nonworse': better(report['metrics'], 'joint_score_log_loss'),
        'joint_score_paired28_upper_nonpositive': interval_pass('joint_score'),
        'each_country_joint_score_nonworse': all(better(v, 'joint_score_log_loss') for v in report['by_country'].values())}
    report['passes_all_gates'] = all(report['checks'].values())
    return report
