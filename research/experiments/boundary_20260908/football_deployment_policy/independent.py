"""Independent deployment-policy arithmetic: no operational imports or fitting.

Suggested commit: research(football): verify deployment score distributions independently.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import bisect
import hashlib
import json
import math
from zoneinfo import ZoneInfo

import numpy as np

REFERENCE = 'dc_frozen_prefix_auto'
CANDIDATE = 'dc_full_admitted_equal_off'
POLICIES = (REFERENCE, CANDIDATE)
ZONES = {'E0': 'Europe/London', 'SP1': 'Europe/Madrid', 'I1': 'Europe/Rome'}
TAIL = 1e-12


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def finite(value, name='number'):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def normalize(values):
    if len(values) != 3:
        raise ValueError('Exactly three H/D/A probabilities required')
    values = [finite(x, 'probability') for x in values]
    total = math.fsum(values)
    if min(values) < 0 or total <= 0 or not math.isfinite(total):
        raise ValueError('Nonnegative probabilities with positive finite mass required')
    return [x / total for x in values]


def rho_support(lh, la, rho):
    lh, la, rho = (finite(x) for x in (lh, la, rho))
    if min(lh, la) <= 0 or max(lh, la) > 100:
        raise ValueError('Independent finite kernel supports rates in (0,100]')
    if not max(-1 / lh, -1 / la) <= rho <= min(1., 1 / (lh * la)):
        raise ValueError('Rho violates nonnegative full score support')
    return lh, la, rho


def dc_rates(state, home, away):
    if not isinstance(home, str) or not home or not isinstance(away, str) or not away or home == away:
        raise ValueError('Distinct nonempty team IDs required')
    if state.get('schema', 'football_dc_state_v1') != 'football_dc_state_v1':
        raise ValueError('Unsupported goal state schema')
    for kind in ('attack', 'defense'):
        if not isinstance(state[kind], dict) or any(not isinstance(k, str) or not k for k in state[kind]):
            raise ValueError('Goal strengths must use nonempty team keys')
        for value in state[kind].values():
            finite(value, kind)
    home_log = finite(state['home_intercept']) + state['attack'].get(home, 0.) + state['defense'].get(away, 0.)
    away_log = finite(state['away_intercept']) + state['attack'].get(away, 0.) + state['defense'].get(home, 0.)
    try:
        lh, la = math.exp(home_log), math.exp(away_log)
    except OverflowError as exc:
        raise ValueError('Nonfinite model rates') from exc
    rho_support(lh, la, state['rho'])
    return lh, la


def _poisson(rate, tolerance, minimum=1):
    """Tail bound from decreasing probability ratios, without 1-CDF cancellation."""
    values = [math.exp(-rate)]
    for k in range(1001):
        following = values[-1] * rate / (k + 1)
        ratio = rate / (k + 2)
        bound = following / (1 - ratio) if ratio < 1 else math.inf
        if k >= minimum and bound <= tolerance:
            return values, bound
        if k == 1000:
            break
        values.append(following)
    raise ValueError('Poisson support did not converge')


def matrix_summary(matrix):
    if not isinstance(matrix, (list, tuple)) or not matrix or not isinstance(matrix[0], (list, tuple)) or not matrix[0]:
        raise ValueError('Nonempty rectangular matrix required')
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError('Ragged score matrix')
    cells = [(h, a, finite(p, 'score mass')) for h, row in enumerate(matrix) for a, p in enumerate(row)]
    if any(p < 0 for _, _, p in cells) or abs(math.fsum(p for _, _, p in cells) - 1) > 1e-10:
        raise ValueError('Invalid normalized score matrix')
    hda = [math.fsum(p for h, a, p in cells if (0 if h > a else 1 if h == a else 2) == k) for k in range(3)]
    mode = max(cells, key=lambda x: (x[2], -x[0], -x[1]))
    return {'final_HDA': hda, 'dimensions': [len(matrix), width],
            'expected_goals': [math.fsum(h*p for h, _, p in cells), math.fsum(a*p for _, a, p in cells)],
            'mode': list(mode)}


def raw_joint(lh, la, rho, *, tail_tolerance=TAIL):
    lh, la, rho = rho_support(lh, la, rho)
    tolerance = finite(tail_tolerance)
    if not 1e-25 <= tolerance < 1e-3:
        raise ValueError('Unsupported tail tolerance')
    home, home_bound = _poisson(lh, tolerance / 2)
    away, away_bound = _poisson(la, tolerance / 2)
    size = max(len(home), len(away))
    for values, rate in ((home, lh), (away, la)):
        while len(values) < size:
            values.append(values[-1] * rate / len(values))
    matrix = [[h*a for a in away] for h in home]
    factors = ((1-lh*la*rho, 1+lh*rho), (1+la*rho, 1-rho))
    for h in range(2):
        for a in range(2):
            matrix[h][a] *= factors[h][a]
            if matrix[h][a] < 0:
                raise ValueError('Negative low-score mass')
    total = math.fsum(p for row in matrix for p in row)
    omitted = max(0., 1-total)
    if omitted > tolerance*1.01+1e-15 or total <= 0:
        raise ValueError('Invalid tail-controlled mass')
    matrix = [[p/total for p in row] for row in matrix]
    return {'matrix': matrix, 'omitted_probability_mass': omitted,
            'tail_probability_bound': home_bound+away_bound,
            'source_parameters': [lh, la, rho], 'reconciled': False, **matrix_summary(matrix)}


def reconcile(raw, selected):
    targets = normalize(selected)
    regions = matrix_summary(raw['matrix'])['final_HDA']
    if any(r <= 0 and q > 0 for r, q in zip(regions, targets)):
        raise ValueError('Selected mass assigned to empty region')
    amplification = max(q/r for q, r in zip(targets, regions) if r > 0)
    bound = raw['tail_probability_bound'] * amplification
    if bound > TAIL:
        tighter = TAIL / amplification / 2
        return reconcile(raw_joint(*raw['source_parameters'], tail_tolerance=tighter), targets)
    matrix = [[p * targets[0 if h > a else 1 if h == a else 2] / regions[0 if h > a else 1 if h == a else 2]
               for a, p in enumerate(row)] for h, row in enumerate(raw['matrix'])]
    return {**raw, 'matrix': matrix, 'tail_probability_bound': bound, 'reconciled': True, **matrix_summary(matrix)}


def _sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def apply_calibrator(state, probabilities):
    if state.get('schema', 'football_calibrator_state_v1') != 'football_calibrator_state_v1':
        raise ValueError('Unsupported calibrator schema')
    method = state['method']
    if state.get('nonidentity_class_floor') != 1e-12 or state.get('platt_input_clip') != [1e-12, 1-1e-12]:
        raise ValueError('Serialized calibration numerical semantics differ')
    functions = state['class_functions']
    if method not in ('identity', 'platt', 'isotonic') or len(functions) != 3:
        raise ValueError('Unsupported calibrator method/functions')
    values = []
    for value, function in zip(normalize(probabilities), functions):
        kind = function['kind']
        if kind == 'identity':
            result = value
        elif kind == 'platt' and method == 'platt':
            if function['classes_'] != [0, 1] or function['n_features_in_'] != 1 or len(function['coef_']) != 1 or len(function['coef_'][0]) != 1 or len(function['intercept_']) != 1:
                raise ValueError('Only binary one-logit Platt calibration is supported')
            clipped = min(1-1e-12, max(1e-12, value))
            z = finite(function['coef_'][0][0]) * math.log(clipped/(1-clipped)) + finite(function['intercept_'][0])
            result = _sigmoid(finite(z))
        elif kind == 'isotonic' and method == 'isotonic':
            x, y = function['X_thresholds_'], function['y_thresholds_']
            if not len(x) or len(x) != len(y):
                raise ValueError('Aligned nonempty isotonic knots required')
            x, y = [finite(v) for v in x], [finite(v) for v in y]
            if any(b <= a for a, b in zip(x, x[1:])) or any(v < 0 or v > 1 for v in y) or any(b < a for a, b in zip(y, y[1:])):
                raise ValueError('Invalid isotonic knots')
            if finite(function['X_min_']) != x[0] or finite(function['X_max_']) != x[-1]:
                raise ValueError('Isotonic range differs from its knots')
            if function.get('params', {}).get('out_of_bounds', 'clip') != 'clip':
                raise ValueError('Only the canonical clipped isotonic extrapolation is supported')
            j = bisect.bisect_right(x, value)
            result = y[0] if j == 0 else y[-1] if j == len(x) else y[j-1]+(y[j]-y[j-1])*(value-x[j-1])/(x[j]-x[j-1])
        else:
            raise ValueError('Calibrator class function differs from method')
        if not math.isfinite(result) or result < 0:
            raise ValueError('Invalid calibrated mass')
        values.append(max(1e-12, result) if method != 'identity' else result)
    return normalize(values)


def replay_output(goal_state, calibrator_state, home, away):
    lh, la = dc_rates(goal_state, home, away)
    raw = raw_joint(lh, la, goal_state['rho'])
    selected = apply_calibrator(calibrator_state, raw['final_HDA'])
    final = reconcile(raw, selected)
    output = {**final, 'lambda_home': lh, 'lambda_away': la, 'rho': goal_state['rho'],
              'raw_hda': raw['final_HDA'], 'selected_hda': selected,
              'raw_omitted_probability_mass': raw['omitted_probability_mass']}
    output['hda'] = output.pop('final_HDA')
    return output


def score_loss(matrix, goals):
    matrix_summary(matrix)
    if len(goals) != 2 or any(type(g) is not int or g < 0 for g in goals):
        raise ValueError('Observed score requires two nonnegative integers')
    h, a = goals
    if h >= len(matrix) or a >= len(matrix[0]):
        return {'status': 'nonfinite_loss', 'cause': 'out_of_support', 'probability': 0., 'nll': None}
    probability = matrix[h][a]
    if probability <= 0:
        return {'status': 'nonfinite_loss', 'cause': 'zero_mass', 'probability': probability, 'nll': None}
    return {'status': 'finite', 'cause': None, 'probability': probability, 'nll': -math.log(probability)}


def utc(value):
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(value, datetime):
        raise ValueError('Datetime required')
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def local_midnight(day, league):
    day = date.fromisoformat(day) if isinstance(day, str) else day
    return datetime.combine(day, datetime.min.time(), ZoneInfo(ZONES[league])).astimezone(timezone.utc)


def history_membership(records, cutoff, league):
    """Metadata admission precedes any scored payload access; local-day window."""
    cutoff = utc(cutoff)
    end_day = cutoff.astimezone(ZoneInfo(ZONES[league])).date()
    first_day = end_day - timedelta(days=1095)
    selected = []
    for row in records:
        if row['league'] != league:
            continue
        day = date.fromisoformat(row['day']) if isinstance(row['day'], str) else row['day']
        clock = local_midnight(day, league)
        available = local_midnight(day+timedelta(days=1), league)
        if first_day <= day < end_day and clock < cutoff and available <= cutoff:
            selected.append(row)
    if len({r['match_id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate admitted identity')
    return sorted(selected, key=lambda r: (local_midnight(r['day'], league), r['season'], r['match_id']))


def chronological_partitions(records):
    """Whole UTC-day tail splitting with the original availability purge."""
    def key(r):
        return utc(r['date']), r['season'], r.get('round_number') if isinstance(r.get('round_number'), int) else 9999, r['match_id']
    rows = sorted(records, key=key)
    if len({r['match_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate chronological identity')
    def split(values, n):
        if not values:
            return [], []
        i = max(0, len(values)-n)
        boundary = utc(values[i]['date']).date()
        while i and utc(values[i-1]['date']).date() == boundary:
            i -= 1
        tail = values[i:]
        cutoff = min(utc(r['date']) for r in tail)
        return [r for r in values[:i] if utc(r['result_available_at']) <= cutoff], tail
    n = len(rows)
    prefix, test = split(rows, max(30, math.ceil(.2*n)))
    prefix, selection = split(prefix, max(30, math.ceil(.15*n)))
    fit, calibration = split(prefix, max(30, math.ceil(.15*n)))
    if not (len(fit) >= 45 and min(len(calibration), len(selection), len(test)) >= 30):
        fit, calibration = split(rows, max(30, math.ceil(.2*n)))
        if len(fit) < 45 or len(calibration) < 30:
            fit, calibration = rows, []
        selection, test = [], []
    groups = {'fit': fit, 'calibration': calibration, 'selection': selection, 'test': test}
    used = {r['match_id'] for values in groups.values() for r in values}
    return {**{name: [r['match_id'] for r in values] for name, values in groups.items()},
            'excluded_boundary_ids': [r['match_id'] for r in rows if r['match_id'] not in used]}


def _row_scores(rows, policy):
    scored = []
    for row in rows:
        if type(row['label']) is not int or row['label'] not in (0, 1, 2):
            raise ValueError('Invalid H/D/A label')
        out = row['outputs'][policy]
        summary = matrix_summary(out['matrix'])
        p = out['hda']
        normalized = normalize(p)
        if max(abs(a-b) for a, b in zip(p, normalized)) > 1e-10 or max(abs(a-b) for a, b in zip(p, summary['final_HDA'])) > 1e-10:
            raise ValueError('Stored HDA does not equal emitted matrix marginals')
        score = score_loss(out['matrix'], row['goals'])
        h, a = row['goals']
        if row['label'] != (0 if h > a else 1 if h == a else 2):
            raise ValueError('Score and outcome disagree')
        true_mass = p[row['label']]
        winner = max(range(3), key=lambda k: (p[k], -k))
        scored.append({'nll': -math.log(true_mass) if true_mass > 0 else None,
                       'joint': score['nll'], 'joint_cause': score['cause'],
                       'brier': math.fsum((value-int(k == row['label']))**2 for k, value in enumerate(p)),
                       'confidence': p[winner], 'correct': int(winner == row['label']), 'p': p})
    return scored


def _metrics(rows, scored):
    n = len(rows)
    def mean(values):
        return math.fsum(values)/n if all(v is not None for v in values) else None
    bins, ece = [], 0.
    for k in range(10):
        bucket = [r for r in scored if k/10 <= r['confidence'] and (r['confidence'] < (k+1)/10 if k < 9 else r['confidence'] <= 1)]
        if bucket:
            confidence = math.fsum(r['confidence'] for r in bucket)/len(bucket)
            accuracy = math.fsum(r['correct'] for r in bucket)/len(bucket)
            ece += len(bucket)/n*abs(confidence-accuracy)
            bins.append({'lower': k/10, 'upper': (k+1)/10, 'n': len(bucket), 'mean_confidence': confidence, 'accuracy': accuracy})
    joint = mean([r['joint'] for r in scored])
    nll = mean([r['nll'] for r in scored])
    return {'n': n, 'log_loss': nll, 'brier_sum_classes': mean([r['brier'] for r in scored]),
            'joint_score_log_loss': joint, 'status': 'finite' if joint is not None and nll is not None else 'nonfinite_loss',
            'accuracy': mean([r['correct'] for r in scored]), 'top_label_ece_10_bins': ece,
            'calibration_bins': bins,
            'class_mean_probability_minus_frequency': [math.fsum(s['p'][k]-int(r['label'] == k) for r, s in zip(rows, scored))/n for k in range(3)]}


def paired_losses(rows, differences, days, spec):
    """Paired one-score arrays; RNG identity is the same for both score types."""
    if type(days) is not int or days not in (1, 7, 28) or len(rows) != len(differences) or not rows:
        raise ValueError('Invalid declared paired population/block size')
    count, seed = spec['uncertainty']['resamples'], spec['uncertainty']['seed']
    if type(count) is not int or count <= 0 or type(seed) is not int or seed < 0:
        raise ValueError('Invalid bootstrap settings')
    result = {'difference_candidate_minus_reference': None, 'percentile_95_interval': None,
              'bootstrap_fraction_difference_below_zero': None, 'block_calendar_days': days,
              'blocks_by_season': {}, 'resamples': count, 'reference': REFERENCE,
              'status': 'nonfinite_loss'}
    if any(x is None for x in differences):
        return result
    differences = [finite(x, 'loss delta') for x in differences]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[f"{row['league']}:{row['season']}"].append(i)
    rng, draws_total = np.random.default_rng(seed), np.zeros(count)
    for group in sorted(groups):
        indices = groups[group]
        dates = [date.fromisoformat(rows[i]['day']) for i in indices]
        first = min(dates)
        blocks = {}
        for i, day in zip(indices, dates):
            blocks.setdefault((day-first).days//days, []).append(differences[i])
        sums = np.asarray([sum(x) for x in blocks.values()])
        sizes = np.asarray([len(x) for x in blocks.values()])
        selected = rng.integers(0, len(blocks), size=(count, len(blocks)))
        draws_total += len(indices)/len(rows)*(sums[selected].sum(1)/sizes[selected].sum(1))
        result['blocks_by_season'][group] = len(blocks)
    return {**result, 'status': 'finite', 'difference_candidate_minus_reference': math.fsum(differences)/len(rows),
            'percentile_95_interval': np.quantile(draws_total, [.025, .975]).tolist(),
            'bootstrap_fraction_difference_below_zero': float(np.mean(draws_total < 0))}


def evaluate(rows, spec):
    """Full fixed population; nonfinite score loss preserves rows and vetoes gates."""
    population = spec['population']
    ids = [r['match_id'] for r in rows]
    if len(rows) != population['rows'] or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('The complete unique declared population is required')
    if digest(ids) != population['ordered_original_match_ids_sha256']:
        raise ValueError('Ordered scoring population differs from the frozen cohort')
    countries = population['countries']
    seasons = population['season_start_years']
    expected = {(league, season) for league in countries for season in seasons}
    actual = {(r['league'], r['season']) for r in rows}
    if actual != expected or any(type(r['season']) is not int for r in rows):
        raise ValueError('Country/season scope differs from the fixed cohort')
    for key in expected:
        if sum((r['league'], r['season']) == key for r in rows) != population['rows_each_country_season']:
            raise ValueError('Incomplete country-season population')
    if any(set(r['outputs']) != set(POLICIES) for r in rows):
        raise ValueError('Exactly the fixed two policies must be scored')
    scored = {p: _row_scores(rows, p) for p in POLICIES}
    result = {'rows': len(rows), 'population_sha256': digest(ids),
              'metrics': {p: _metrics(rows, scored[p]) for p in POLICIES},
              'by_country': {}, 'by_country_season': {}, 'paired': {}, 'promotion': False,
              'nonfinite_losses': []}
    for p in POLICIES:
        for row, score in zip(rows, scored[p]):
            if score['joint'] is None or score['nll'] is None:
                result['nonfinite_losses'].append({'match_id': row['match_id'], 'policy': p,
                                                  'joint_cause': score['joint_cause'], 'one_x_two_zero': score['nll'] is None})
    for league in countries:
        for season in [None, *seasons]:
            indices = [i for i, r in enumerate(rows) if r['league'] == league and (season is None or r['season'] == season)]
            values = {p: _metrics([rows[i] for i in indices], [scored[p][i] for i in indices]) for p in POLICIES}
            target = result['by_country'] if season is None else result['by_country_season']
            target[league if season is None else f'{league}:{season}'] = values
    for days in spec['uncertainty']['blocks_calendar_days']:
        result['paired'][str(days)] = {}
        for name, score_key in [('one_x_two', 'nll'), ('joint_score', 'joint')]:
            delta = [c[score_key]-r[score_key] if c[score_key] is not None and r[score_key] is not None else None
                     for c, r in zip(scored[CANDIDATE], scored[REFERENCE])]
            result['paired'][str(days)][name] = paired_losses(rows, delta, days, spec)
    def improve(values, key, strict=False):
        c, r = values[CANDIDATE][key], values[REFERENCE][key]
        return c is not None and r is not None and (c < r if strict else c <= r)
    def upper(name, strict=False):
        pair = result['paired'][str(spec['uncertainty']['gate_block_days'])][name]
        return pair['status'] == 'finite' and (pair['percentile_95_interval'][1] < 0 if strict else pair['percentile_95_interval'][1] <= 0)
    c, r = result['metrics'][CANDIDATE]['log_loss'], result['metrics'][REFERENCE]['log_loss']
    gain = (r-c)/r if c is not None and r is not None and r > 0 else None
    result['relative_one_x_two_nll_gain'] = gain
    result['checks'] = {
        'complete_population': True,
        'all_losses_finite': not result['nonfinite_losses'],
        'one_x_two_minimum_gain': gain is not None and gain >= spec['acceptance']['one_x_two']['minimum_pooled_relative_nll_gain'],
        'one_x_two_paired28_upper_negative': upper('one_x_two', True),
        'each_country_one_x_two_improves': all(improve(v, 'log_loss', True) for v in result['by_country'].values()),
        'each_country_season_one_x_two_nonworse': all(improve(v, 'log_loss') for v in result['by_country_season'].values()),
        'each_country_season_brier_nonworse': all(improve(v, 'brier_sum_classes') for v in result['by_country_season'].values()),
        'pooled_joint_score_nonworse': improve(result['metrics'], 'joint_score_log_loss'),
        'joint_score_paired28_upper_nonpositive': upper('joint_score'),
        'each_country_joint_score_nonworse': all(improve(v, 'joint_score_log_loss') for v in result['by_country'].values())}
    result['passes_all_gates'] = all(result['checks'].values())
    return result
