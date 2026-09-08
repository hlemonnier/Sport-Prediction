"""Independent arithmetic and chronology for the frozen past-market protocol.

No import of this experiment's model/data/runner, no optimizer, and no implicit
file reads. Inputs are caller-supplied, source-bound metadata and coefficients.
Suggested commit: research(football): independently replay past-market forecasts
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
import math
from numbers import Real
from zoneinfo import ZoneInfo

import numpy as np

LEAGUES = ('E0', 'I1', 'SP1')
TIMEZONES = {'E0': 'Europe/London', 'I1': 'Europe/Rome', 'SP1': 'Europe/Madrid'}
FLOOR = 1e-12


def finite(value, name='number'):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name+' must be numeric, not a boolean')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name+' must be finite')
    return result


def utc(value):
    """Legacy naive forecast timestamps mean UTC, never machine local time."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime):
        raise ValueError('A datetime or ISO datetime string is required')
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def source_day(value):
    if isinstance(value, str):
        value = date.fromisoformat(value)
    if type(value) is not date:
        raise ValueError('Canonical source day must be a date or ISO date')
    return value


def availability(day, league, extra_days):
    """A0 and quote availability as aware UTC: local D+1 and D+2/D+8."""
    if league not in TIMEZONES or type(extra_days) is not int or extra_days not in (1, 7):
        raise ValueError('A declared league and one/seven extra calendar days are required')
    day = source_day(day)
    zone = ZoneInfo(TIMEZONES[league])
    a0 = datetime.combine(day+timedelta(days=1), time.min, zone).astimezone(timezone.utc)
    available = datetime.combine(day+timedelta(days=1+extra_days), time.min, zone).astimezone(timezone.utc)
    return a0, available


def admitted(day, league, cutoff, extra_days, *, history_days=1095):
    if type(history_days) is not int or history_days != 1095:
        raise ValueError('The frozen local-calendar history is1095 days')
    cutoff, day = utc(cutoff), source_day(day)
    a0, available = availability(day, league, extra_days)
    current = cutoff.astimezone(ZoneInfo(TIMEZONES[league])).date()
    return current-timedelta(days=1095) <= day < current and available < cutoff


def simplex(values, *, floor=False):
    raw = np.asarray(values)
    if raw.shape != (3,) or raw.dtype.kind not in 'fiu':
        raise ValueError('Expected one numeric H/D/A vector')
    p = raw.astype(np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or abs(math.fsum(p)-1.) > 1e-10:
        raise ValueError('Invalid probability simplex')
    if floor:
        p = np.maximum(p, FLOOR)
        p /= math.fsum(p)
    return p


def _softmax(logits):
    values = [finite(v, 'logit') for v in logits]
    largest = max(values)
    masses = [math.exp(v-largest) for v in values]
    denominator = math.fsum(masses)
    return np.array([v/denominator for v in masses], dtype=np.float64)


def _sigmoid(value):
    if value >= 0:
        e = math.exp(-value)
        return 1./(1.+e)
    e = math.exp(value)
    return e/(1.+e)


def devig(odds, method='power'):
    """Independent power root via bisection; normalized inverse for Elo.

    Unlike the fitting helper's Brent solver, this uses deterministic bisection
    and stdlib exponentials. It returns the raw normalized simplex; the main
    strength target applies the separate fixed probability floor.
    """
    if len(odds) != 3:
        raise ValueError('Exactly three decimal odds are required')
    values = [finite(o, 'odds') for o in odds]
    if min(values) <= 1:
        raise ValueError('Decimal odds must be strictly greater than one')
    if method not in ('normalized', 'power'):
        raise ValueError('Only fixed power/normalized de-vig is allowed')
    result, power = _cached_devig(tuple(values), method)
    # The shared cache stores immutable tuples, never caller-mutable arrays.
    return np.array(result, dtype=np.float64), power


@lru_cache(maxsize=32768)
def _cached_devig(values, method):
    """Bounded cache entered only after numeric admission and validation."""
    logs = [-math.log(o) for o in values]
    if method == 'normalized':
        return tuple(_softmax(logs)), None
    def residual(power):
        return math.fsum(math.exp(power*x) for x in logs)-1.
    high = 1.
    for _ in range(1024):
        if residual(high) <= 0:
            break
        high *= 2.
    else:
        raise ValueError('Power root could not be bracketed')
    low = 0.
    for _ in range(180):
        middle = low+(high-low)/2.
        if middle in (low, high):
            break
        if residual(middle) > 0:
            low = middle
        else:
            high = middle
    power = low+(high-low)/2.
    return tuple(_softmax([power*x for x in logs])), power


def _field(row, name):
    return row[name] if isinstance(row, Mapping) else getattr(row, name)


def _odds(raw, season):
    names = ('BbAvH', 'BbAvD', 'BbAvA') if season in (2017, 2018) else ('AvgH', 'AvgD', 'AvgA')
    output = []
    for name in names:
        value = raw.get(name)
        if isinstance(value, (bool, np.bool_)) or value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or number <= 1:
            return None
        output.append(number)
    return output


def _outcome(raw):
    label = raw.get('FTR')
    if label not in ('H', 'D', 'A'):
        raise ValueError('An admitted quote-valid fixture must have a canonical FTR')
    goals = []
    for key in ('FTHG', 'FTAG'):
        value = raw.get(key)
        if isinstance(value, (bool, np.bool_)) or not str(value).isdecimal():
            raise ValueError('Admitted goals must be nonnegative integers')
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Admitted goals must be nonnegative integers') from exc
        if number < 0:
            raise ValueError('Admitted goals must be nonnegative integers')
        goals.append(number)
    expected = 'H' if goals[0] > goals[1] else 'A' if goals[0] < goals[1] else 'D'
    if label != expected:
        raise ValueError('Admitted goals and FTR disagree')
    return ('H', 'D', 'A').index(label)


def history_membership(records, cutoff, *, extra_days, include_outcomes=True, windowed=True):
    """Reconstruct metadata-first membership, odds and objective weights.

    ``records`` may contain plain mappings or the data lane's RawFixture
    objects: match_id/league/season/home/away/day/payload. The payload is never
    fetched until the strict availability and optional fixed1095-day window
    pass. Accepted rows are sorted by availability and ID. Missing quotes are reported, then skipped
    before outcome access. ``windowed=False,include_outcomes=False`` is solely
    the all-history, quote-only input to causal Elo replay.

    Raw weights are2**(-elapsed_age_from_A0/365). Objective weights divide each
    by its league's total and by3. Empty leagues are explicitly reported; the
    fitting caller must require all3, rather than silently renormalizing them.
    """
    if type(include_outcomes) is not bool or type(windowed) is not bool:
        raise ValueError('History mode flags must be explicit booleans')
    if not windowed and include_outcomes:
        raise ValueError('All-history mode is reserved for quote-only Elo updates')
    cutoff = utc(cutoff)
    # Validate the lag even if there are no rows.
    if type(extra_days) is not int or extra_days not in (1, 7):
        raise ValueError('Only1/7 extra calendar days are declared')
    seen, accepted, missing, metadata_admitted = set(), [], [], []
    for row in records:
        league = _field(row, 'league')
        day = source_day(_field(row, 'day'))
        a0, available = availability(day, league, extra_days)
        if available >= cutoff or (windowed and not admitted(day, league, cutoff, extra_days)):
            continue
        match_id = _field(row, 'match_id')
        if not isinstance(match_id, str) or not match_id or match_id in seen:
            raise ValueError('Admitted history requires unique canonical match IDs')
        seen.add(match_id)
        season = _field(row, 'season')
        if isinstance(season, (bool, np.bool_)) or not isinstance(season, (int, np.integer)) or not 2017 <= season <= 2025:
            raise ValueError('Only declared2017–2025 season starts are supported')
        metadata_admitted.append(match_id)
        raw = _field(row, 'payload')
        if not isinstance(raw, Mapping):
            raise ValueError('Admitted raw payload must be a mapping')
        odds = _odds(raw, season)
        if odds is None:
            missing.append(match_id)
            continue
        home, away = _field(row, 'home'), _field(row, 'away')
        if not isinstance(home, str) or not home or not isinstance(away, str) or not away or home == away:
            raise ValueError('Admitted teams must be distinct nonempty names')
        outcome = _outcome(raw) if include_outcomes else None
        power_q, power = devig(odds, 'power')
        normalized_q, _ = devig(odds, 'normalized')
        age = (cutoff-a0).total_seconds()/86400.
        weight = 2.**(-age/365.)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Admitted elapsed recency weights must be finite and positive')
        item = {'match_id': match_id, 'league': league, 'season': int(season), 'home': home,
                'away': away, 'day': day.isoformat(), 'a0_utc': a0.isoformat(),
                'available_at_utc': available.isoformat(), 'quote_available_at_utc': available.isoformat(), 'odds': odds,
                'q': simplex(power_q, floor=True).tolist(), 'q_normalized': normalized_q.tolist(),
                'power': power, 'weight': weight}
        if include_outcomes:
            item['outcome'] = outcome
        accepted.append(item)
    accepted.sort(key=lambda row: (row['quote_available_at_utc'], row['match_id']))
    totals = {league: math.fsum(r['weight'] for r in accepted if r['league'] == league) for league in LEAGUES}
    objective = [r['weight']/(3.*totals[r['league']]) for r in accepted]
    return {'rows': accepted, 'training_ids': [r['match_id'] for r in accepted],
            'metadata_admitted_ids': metadata_admitted, 'missing_quote_ids': missing,
            'raw_weights': [r['weight'] for r in accepted], 'objective_weights': objective,
            'missing_leagues': [league for league in LEAGUES if totals[league] == 0],
            'cutoff_utc': cutoff.isoformat(), 'extra_days': extra_days}


def strength_predict(payload, league, home, away):
    """Replay actual centered effects, independent of fitted free coordinates."""
    if not isinstance(payload, dict) or payload.get('kind') != 'past_market_strength_v1':
        raise ValueError('Expected serialized past-market strength parameters')
    if payload.get('league_order') != list(LEAGUES) or set(payload.get('leagues', {})) != set(LEAGUES):
        raise ValueError('Exact declared league parameters are required')
    if league not in LEAGUES or not isinstance(home, str) or not isinstance(away, str) or home == away:
        raise ValueError('A declared league and distinct team names are required')
    parameters = payload['leagues'][league]
    teams = parameters['teams']
    if not isinstance(teams, list) or not teams or not all(isinstance(t, str) and t for t in teams) or teams != sorted(set(teams)):
        raise ValueError('Actual teams must be sorted unique names')
    if home not in teams or away not in teams:
        raise ValueError('Unknown teams require an explicit caller-owned incumbent fallback')
    effects = []
    for name in ('s', 'v'):
        values = parameters[name]
        if not isinstance(values, list) or len(values) != len(teams):
            raise ValueError('Actual team effects must align to the team list')
        values = [finite(x, name) for x in values]
        if abs(math.fsum(values)) > 1e-10 * max(1., math.fsum(abs(x) for x in values)):
            raise ValueError('Actual team effects must be centered')
        effects.append(values)
    h, d = finite(parameters['h'], 'home intercept'), finite(parameters['d'], 'draw intercept')
    i, j = teams.index(home), teams.index(away)
    s, v = effects
    difference = h+s[i]-s[j]
    return simplex(_softmax([difference/2., d+v[i]+v[j], -difference/2.]), floor=True)


def strength_objective_audit(payload, rows):
    """Unfloored penalized CE and Helmert free gradient, without refitting.

    Targets/weights must be the independently admitted rows. The model's
    diagnostic target hashes or normalized weights are never trusted here.
    Actual effects determine logits and penalty; explicit Helmert projection
    independently maps accumulated effect gradients to saved free coordinates.
    """
    if not isinstance(rows, list) or not rows or payload.get('target_kind') not in ('past_market', 'outcome_control'):
        raise ValueError('Nonempty admitted rows and a declared target kind are required')
    if payload.get('penalty') != .002 or payload.get('floor') != FLOOR:
        raise ValueError('Saved objective penalty/output floor differ from the protocol')
    roster = {league: sorted({r[key] for r in rows if r['league'] == league for key in ('home', 'away')}) for league in LEAGUES}
    blocks, theta = {}, []
    for league in LEAGUES:
        names = roster[league]
        if len(names) < 2 or payload['leagues'][league]['teams'] != names:
            raise ValueError('Saved teams differ from actual admitted history')
        # Reuse only this independent file's parameter validation, never model.py.
        strength_predict(payload, league, names[0], names[1])
        saved = payload['leagues'][league]
        n = len(names)
        sf, vf = [], []
        for key, target in (('s_free', sf), ('v_free', vf)):
            if not isinstance(saved[key], list) or len(saved[key]) != n-1:
                raise ValueError('Invalid saved free coefficient shape')
            target.extend(finite(x, key) for x in saved[key])
        # Explicit Helmert construction via scalar sums, not the fitted basis.
        for free, actual in ((sf, saved['s']), (vf, saved['v'])):
            reconstructed = [0.]*n
            for j, coefficient in enumerate(free):
                denominator = math.sqrt((j+1)*(j+2))
                for i in range(j+1):
                    reconstructed[i] += coefficient/denominator
                reconstructed[j+1] -= (j+1)*coefficient/denominator
            if not np.allclose(reconstructed, actual, rtol=0, atol=1e-10):
                raise ValueError('Actual and free centered coefficients disagree')
        h, d = saved['h'], saved['d']
        theta.extend([h, d, *sf, *vf])
        blocks[league] = {'saved': saved, 'index': {name:i for i,name in enumerate(names)},
            's_free': sf, 'v_free': vf, 'h_gradient': [], 'd_gradient': [],
            's_gradient': [[] for _ in names], 'v_gradient': [[] for _ in names]}
    if not np.array_equal(np.array(theta), np.asarray(payload['theta'])):
        raise ValueError('Serialized global theta differs from its league blocks')
    if any(r['league'] not in LEAGUES for r in rows):
        raise ValueError('Unknown training league')
    weights = [finite(r['weight'], 'raw recency weight') for r in rows]
    if min(weights) <= 0:
        raise ValueError('Training weights must be positive')
    denominators = {league: math.fsum(w for row,w in zip(rows,weights) if row['league']==league) for league in LEAGUES}
    losses = []
    for row, weight in zip(rows, weights):
        block = blocks[row['league']]
        saved, index = block['saved'], block['index']
        i, j = index[row['home']], index[row['away']]
        if i == j:
            raise ValueError('Training teams must differ')
        difference = saved['h']+saved['s'][i]-saved['s'][j]
        logits = [difference/2., saved['d']+saved['v'][i]+saved['v'][j], -difference/2.]
        if not all(math.isfinite(z) for z in logits):
            raise ValueError('Nonfinite training logits')
        if payload['target_kind'] == 'past_market':
            target = simplex(row['q'])
        else:
            label = row['outcome']
            if isinstance(label, (bool,np.bool_)) or not isinstance(label,(int,np.integer)) or label not in (0,1,2):
                raise ValueError('Invalid admitted outcome')
            target = np.array([float(k == label) for k in range(3)])
        largest = max(logits)
        shifted = [z-largest for z in logits]
        log_denominator = math.log(math.fsum(math.exp(z) for z in shifted))
        logp = [z-log_denominator for z in shifted]
        alpha = weight/(3.*denominators[row['league']])
        losses.append(-alpha*math.fsum(float(q)*lp for q,lp in zip(target,logp)))
        mass = math.fsum(target)
        residual = [alpha*(math.exp(lp)*mass-float(q)) for q,lp in zip(target,logp)]
        gc, gd = (residual[0]-residual[2])/2., residual[1]
        block['h_gradient'].append(gc); block['d_gradient'].append(gd)
        block['s_gradient'][i].append(gc); block['s_gradient'][j].append(-gc)
        block['v_gradient'][i].append(gd); block['v_gradient'][j].append(gd)
    penalty, gradient = [], []
    for league in LEAGUES:
        block = blocks[league]; saved = block['saved']
        h, d = saved['h'], saved['d']
        prior_draw = math.log(2/3)
        penalty.extend([h*h, (d-prior_draw)**2, *(s*s for s in saved['s']), *(v*v for v in saved['v'])])
        gradient.extend([math.fsum(block['h_gradient'])+.002*h,
                         math.fsum(block['d_gradient'])+.002*(d-prior_draw)])
        for effect, free in ((block['s_gradient'],block['s_free']), (block['v_gradient'],block['v_free'])):
            actual_gradient = [math.fsum(v) for v in effect]
            for j, coefficient in enumerate(free):
                projected = (math.fsum(actual_gradient[:j+1])-(j+1)*actual_gradient[j+1])/math.sqrt((j+1)*(j+2))
                gradient.append(projected+.002*coefficient)
    objective = math.fsum(losses)+.001*math.fsum(penalty)
    if not math.isfinite(objective) or not all(math.isfinite(g) for g in gradient):
        raise ValueError('Nonfinite independently reconstructed objective/gradient')
    return {'objective': objective, 'gradient': gradient, 'gradient_max': max(map(abs,gradient)),
            'rows': len(rows), 'free_theta': theta}


def ordered_logit_predict(payload, rating_home, rating_away):
    """Stable ordinal H/D/A readout, with positive threshold separation."""
    if not isinstance(payload, dict) or payload.get('kind') != 'ordered_logit_v1':
        raise ValueError('Expected serialized ordered-logit readout')
    beta, low, high = [finite(payload[k], k) for k in ('beta', 't1', 't2')]
    gap = finite(payload['gap'], 'threshold gap')
    if beta < 0 or gap < 1e-6 or high <= low or not math.isclose(high-low, gap, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Ordered-logit constraints are not satisfied')
    x = (finite(rating_home)-finite(rating_away))/400.
    center = beta*x
    if not math.isfinite(center):
        raise ValueError('Nonfinite ordered-logit linear predictor')
    a, b = low-center, high-center
    # sigma(b)-sigma(a)=sigma(b)*sigma(-a)*(1-exp(-gap)).
    # This avoids subtracting two near-one probabilities for a rare draw.
    p_away = _sigmoid(a)
    p_home = _sigmoid(-b)
    p_draw = _sigmoid(b)*_sigmoid(-a)*(-math.expm1(-gap))
    return simplex([p_home, p_draw, p_away], floor=True)


def ordered_objective_audit(payload, x, outcomes):
    """Unfloored readout NLL, analytic gradient and bound KKT mapping."""
    ordered_logit_predict(payload, 1000., 1000.)
    beta, low, gap = [finite(payload[k],k) for k in ('beta','t1','gap')]
    theta = [beta,low,gap]
    if payload.get('theta') != theta or payload.get('floor') != FLOOR:
        raise ValueError('Readout serialization differs from the fixed contract')
    if len(x) != len(outcomes) or not len(x):
        raise ValueError('Aligned nonempty readout features and outcomes are required')
    values = [finite(v,'prequential rating difference') for v in x]
    if not all(isinstance(y,(int,np.integer)) and not isinstance(y,(bool,np.bool_)) and y in (0,1,2) for y in outcomes):
        raise ValueError('Readout outcomes must be H/D/A integer indices')
    if set(outcomes) != {0,1,2}:
        raise ValueError('Readout calibration requires all three classes')
    def log_sigmoid(v):
        return -max(-v,0.)-math.log1p(math.exp(-abs(v)))
    losses, gradients = [], [[],[],[]]
    gap_term = math.log(-math.expm1(-gap))
    inverse_gap_expm1 = math.exp(-gap)/(-math.expm1(-gap))
    for value,label in zip(values,outcomes):
        a = low-beta*value; b = a+gap
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError('Nonfinite readout logits')
        if label == 0:
            lp, da, dg = log_sigmoid(-b), -_sigmoid(b), -_sigmoid(b)
        elif label == 2:
            lp, da, dg = log_sigmoid(a), _sigmoid(-a), 0.
        else:
            lp = log_sigmoid(b)+log_sigmoid(-a)+gap_term
            da, dg = _sigmoid(-b)-_sigmoid(a), _sigmoid(-b)+inverse_gap_expm1
        losses.append(-lp)
        gradients[0].append(value*da); gradients[1].append(-da); gradients[2].append(-dg)
    gradient = [math.fsum(g)/len(values) for g in gradients]
    lower = [0., -math.inf, 1e-6]
    projected = [t-max(t-g,bound) for t,g,bound in zip(theta,gradient,lower)]
    return {'objective': math.fsum(losses)/len(values), 'gradient': gradient,
            'kkt_residual': max(map(abs,projected)), 'rows': len(values)}


def fallback_or_blend(prediction, incumbent, *, supported, blend=False):
    """Strict flags and exact unchanged fallback; no incumbent normalization."""
    if type(supported) is not bool or type(blend) is not bool:
        raise ValueError('Support and blend must be explicit booleans')
    incumbent = simplex(incumbent)
    if not supported:
        return incumbent.copy()
    prediction = simplex(prediction)
    return .5*prediction+.5*incumbent if blend else prediction.copy()


def elo_atomic_update(ratings, observations):
    """Update a single already-admitted equal-clock batch from its prestate.

    Ratings use (league,team) keys. Observations carry league/home/away/q and
    optional match_id. Membership and clock grouping are separate functions;
    this pure numerical primitive must not silently serialize same-clock games.
    """
    before = {}
    for key, value in ratings.items():
        if not isinstance(key, tuple) or len(key) != 2 or key[0] not in LEAGUES or not isinstance(key[1], str) or not key[1]:
            raise ValueError('Ratings must use declared league/team keys')
        before[key] = finite(value, 'rating')
    changes = defaultdict(list)
    seen = set()
    for row in observations:
        league, home, away = row['league'], row['home'], row['away']
        if league not in LEAGUES or not isinstance(home, str) or not home or not isinstance(away, str) or not away or home == away:
            raise ValueError('Invalid Elo fixture metadata')
        if 'match_id' in row:
            match = row['match_id']
            if not isinstance(match, str) or not match or match in seen:
                raise ValueError('Elo batch match IDs must be unique nonempty strings')
            seen.add(match)
        h, a = (league, home), (league, away)
        rh, ra = before.get(h, 1000.), before.get(a, 1000.)
        q = simplex(row['q'])
        expected = _sigmoid(math.log(10.)*((rh-ra+80.)/400.))
        delta = 175.*(float(q[0])+.5*float(q[1])-expected)
        changes[h].append(delta)
        changes[a].append(-delta)
    result = dict(before)
    for key, deltas in changes.items():
        result[key] = before.get(key, 1000.)+math.fsum(deltas)
        if not math.isfinite(result[key]):
            raise ValueError('Nonfinite updated Elo rating')
    return result


def elo_replay(records, cutoff, *, extra_days):
    """Replay all strictly admitted historical quotes; no rolling Elo reset."""
    admitted_rows = history_membership(records, cutoff, extra_days=extra_days,
                                       include_outcomes=False, windowed=False)['rows']
    grouped = defaultdict(list)
    for row in admitted_rows:
        grouped[utc(row['available_at_utc'])].append({**row, 'q': row['q_normalized']})
    ratings = {}
    for clock in sorted(grouped):
        ratings = elo_atomic_update(ratings, sorted(grouped[clock], key=lambda row: row['match_id']))
    return ratings


def _scored(rows, name):
    if not rows:
        raise ValueError('A nonempty original scored population is required')
    probabilities, labels = [], []
    for row in rows:
        label = row['label']
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or label not in (0, 1, 2):
            raise ValueError('Outcome must be an integer H/D/A index')
        p = simplex(row['probabilities'][name])
        if min(p) <= 0:
            raise ValueError('Scored forecasts must be strictly positive without new clipping')
        probabilities.append(p)
        labels.append(label)
    return np.asarray(probabilities), np.asarray(labels, dtype=np.int64)


def metrics(rows, name):
    """Unweighted fixture NLL, class-summed Brier, top-label ECE and bias."""
    p, y = _scored(rows, name)
    n = len(rows)
    losses, brier, confidence, correct = [], [], [], []
    for prediction, label in zip(p, y):
        losses.append(-math.log(float(prediction[label])))
        brier.append(math.fsum((float(value)-int(k == label))**2 for k, value in enumerate(prediction)))
        winner = int(np.argmax(prediction))
        confidence.append(float(prediction[winner]))
        correct.append(int(winner == label))
    bins, ece = [], 0.
    for k in range(10):
        indices = [i for i, c in enumerate(confidence) if c >= k/10 and (c < (k+1)/10 if k < 9 else c <= 1)]
        if indices:
            mean_confidence = math.fsum(confidence[i] for i in indices)/len(indices)
            accuracy = math.fsum(correct[i] for i in indices)/len(indices)
            ece += len(indices)/n*abs(mean_confidence-accuracy)
            bins.append({'lower': k/10, 'upper': (k+1)/10, 'n': len(indices), 'mean_confidence': mean_confidence, 'accuracy': accuracy})
    return {'n': n, 'log_loss': math.fsum(losses)/n, 'brier_sum_classes': math.fsum(brier)/n,
            'accuracy': math.fsum(correct)/n, 'top_label_ece_10_bins': ece, 'calibration_bins': bins,
            'class_mean_probability_minus_frequency': [math.fsum(float(pred[k])-int(label == k) for pred, label in zip(p, y))/n for k in range(3)]}


def paired_uncertainty(rows, candidate, reference, days, spec):
    """Independent paired fixed-calendar blocks within league-season strata.

    Resampling preserves the inherited first-occurrence block order and its
    fixture-weighted mixture of stratum means. Empty calendar blocks are not
    sampled; observed blocks carry their original fixture multiplicity.
    """
    p, labels = _scored(rows, candidate)
    q, _ = _scored(rows, reference)
    if type(days) is not int or days not in (1, 7, 28):
        raise ValueError('Only declared1/7/28-calendar-day blocks are supported')
    n_boot = spec['uncertainty']['resamples']
    seed = spec['uncertainty']['seed']
    if type(n_boot) is not int or n_boot <= 0 or type(seed) is not int or seed < 0:
        raise ValueError('Positive resample count and nonnegative integer seed required')
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        if row['league'] not in LEAGUES or isinstance(row['season'], bool):
            raise ValueError('Canonical league-season metadata is required')
        strata[f"{row['league']}:{row['season']}"].append(i)
    random = np.random.default_rng(seed)
    sampled_means = np.zeros(n_boot)
    all_deltas, counts_by_stratum = [], {}
    for key in sorted(strata):
        indices = strata[key]
        calendar = [source_day(rows[i]['day']) for i in indices]
        anchor = min(calendar)
        blocks = {}
        for i, day in zip(indices, calendar):
            delta = -math.log(float(p[i, labels[i]]))+math.log(float(q[i, labels[i]]))
            all_deltas.append(delta)
            block = (day-anchor).days//days
            blocks.setdefault(block, []).append(delta)
        totals = np.array([sum(values) for values in blocks.values()])
        counts = np.array([len(values) for values in blocks.values()])
        draws = random.integers(0, len(blocks), size=(n_boot, len(blocks)))
        sampled_means += len(indices)/len(rows)*(totals[draws].sum(1)/counts[draws].sum(1))
        counts_by_stratum[key] = len(blocks)
    return {'difference_candidate_minus_reference': float(np.mean(all_deltas)),
            'percentile_95_interval': np.quantile(sampled_means, [.025, .975]).tolist(),
            'bootstrap_fraction_difference_below_zero': float(np.mean(sampled_means < 0)),
            'block_calendar_days': days, 'blocks_by_season': counts_by_stratum,
            'resamples': n_boot, 'reference': reference,
            'note': 'Paired, season-stratified fixed-block resampling; conditional historical uncertainty, not a posterior probability or season-transfer guarantee'}
