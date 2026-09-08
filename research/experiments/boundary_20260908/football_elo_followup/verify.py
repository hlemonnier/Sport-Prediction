"""Independent fixed Elo follow-up replay; no fitting or operational imports.

Suggested commit: research(football): independently verify the fixed Elo follow-up.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from research.experiments.boundary_20260908.football_past_market import independent as ind
from research.experiments.boundary_20260908.football_past_market import verify as parent

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts/research/boundary_20260908/football_elo_followup'
CANDIDATE = 'elo_odds_reference'
CONTROL = 'elo_result_k14'
INCUMBENT = 'shot_strength_90d_ridge0.1'
KINDS = ('odds', 'result')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _kind(kind):
    if kind not in KINDS:
        raise ValueError('Only the frozen odds/result rating processes are supported')
    return kind


def _delay(delay):
    if type(delay) is not int or delay not in (1, 7):
        raise ValueError('Only the frozen 1/7 extra local-calendar-day delays are supported')
    return delay


def reject_operational_imports():
    for lane, names in [('football_past_market', ('run', 'model', 'data')),
                        ('football_elo_followup', ('run', 'elo'))]:
        for name in names:
            if f'research.experiments.boundary_20260908.{lane}.{name}' in sys.modules:
                raise ValueError('Historical verifier must run without operational imports')


def check_failures(out):
    if any(Path(out).glob('*_failure.json')):
        raise ValueError('A terminal failure prevents an eligible verification')


def result_atomic_update(ratings, observations):
    """K=14 updates from one pre-batch state; no market target or fit involved."""
    before = {}
    for key, value in ratings.items():
        if not isinstance(key, tuple) or len(key) != 2 or key[0] not in ind.LEAGUES or not isinstance(key[1], str) or not key[1]:
            raise ValueError('Ratings require declared league/team keys')
        before[key] = ind.finite(value, 'rating')
    changes, seen = defaultdict(list), set()
    for row in observations:
        league, home, away = row['league'], row['home'], row['away']
        if league not in ind.LEAGUES or not isinstance(home, str) or not home or not isinstance(away, str) or not away or home == away:
            raise ValueError('Result-Elo requires distinct valid teams')
        mid = row['match_id']
        if not isinstance(mid, str) or not mid or mid in seen:
            raise ValueError('Result-Elo batch identities must be unique')
        seen.add(mid)
        label = row['outcome']
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or label not in (0, 1, 2):
            raise ValueError('Result-Elo outcome must be an H/D/A integer index')
        h, a = (league, home), (league, away)
        rh, ra = before.get(h, 1000.), before.get(a, 1000.)
        z = math.log(10.) * ((rh - ra + 80.) / 400.)
        if not math.isfinite(z):
            raise ValueError('Nonfinite rating difference')
        expected = 1. / (1. + math.exp(-z)) if z >= 0 else math.exp(z) / (1. + math.exp(z))
        delta = 14. * ((1., .5, 0.)[int(label)] - expected)
        changes[h].append(delta)
        changes[a].append(-delta)
    result = dict(before)
    for key, delta in changes.items():
        result[key] = before.get(key, 1000.) + math.fsum(delta)
        if not math.isfinite(result[key]):
            raise ValueError('Nonfinite updated rating')
    return result


def replay_states(records, clocks, delay, kind):
    """Independent metadata-first prefix replay; future numeric payloads stay unread.

    Quotes define the same matched update membership in both processes. Ratings
    accumulate without expiration; the separate support mask uses 1095 days.
    """
    _kind(kind); _delay(delay)
    events = sorted(records, key=lambda r: (ind.availability(r['day'], r['league'], delay)[1], r['match_id']))
    states, position, ratings, seen = {}, 0, {}, set()
    for cutoff in sorted(set(map(ind.utc, clocks))):
        while position < len(events):
            clock = ind.availability(events[position]['day'], events[position]['league'], delay)[1]
            if clock >= cutoff:
                break
            batch = []
            while position < len(events) and ind.availability(events[position]['day'], events[position]['league'], delay)[1] == clock:
                row = events[position]
                position += 1
                mid = row['match_id']
                if not isinstance(mid, str) or not mid or mid in seen:
                    raise ValueError('Admitted raw identities must be unique')
                seen.add(mid)
                odds = ind._odds(row['payload'], row['season'])
                if odds is None:
                    continue
                item = {key: row[key] for key in ('match_id', 'league', 'home', 'away')}
                if kind == 'odds':
                    item['q'] = ind.devig(odds, 'normalized')[0]
                else:
                    if ind.availability(row['day'], row['league'], delay)[0] >= cutoff:
                        raise ValueError('A result was not available at the replay cutoff')
                    item['outcome'] = ind._outcome(row['payload'])
                batch.append(item)
            ratings = ind.elo_atomic_update(ratings, batch) if kind == 'odds' else result_atomic_update(ratings, batch)
        states[cutoff] = dict(ratings)
    return states


def support_history(records, cutoff, delay):
    """Rolling support membership only; never reads goals or future quotes."""
    _delay(delay)
    cutoff = ind.utc(cutoff)
    accepted, seen = [], set()
    for row in records:
        if not ind.admitted(row['day'], row['league'], cutoff, delay):
            continue
        if row['match_id'] in seen:
            raise ValueError('Duplicate support identity')
        seen.add(row['match_id'])
        if ind._odds(row['payload'], row['season']) is not None:
            accepted.append(row)
    return sorted(accepted, key=lambda row: (ind.availability(row['day'], row['league'], delay)[1], row['match_id']))


def support_teams(records, cutoff, delay):
    return {(r['league'], team) for r in support_history(records, cutoff, delay) for team in (r['home'], r['away'])}


def calibration_population(records, cutoff, delay):
    """Reconstruct fixed pre-2022, quote-valid readout rows before labels."""
    _delay(delay)
    cutoff = ind.utc(cutoff)
    eligible = [r for r in records if r['season'] in (2019, 2020, 2021)
                and ind.availability(r['day'], r['league'], delay)[1] < cutoff]
    eligible.sort(key=lambda r: (ind.utc(r['forecast_cutoff_utc']), r['match_id']))
    if len({r['match_id'] for r in eligible}) != len(eligible):
        raise ValueError('Duplicate calibration identity')
    return [r for r in eligible if ind._odds(r['payload'], r['season']) is not None]


def audit_readout(record, records, delay, kind, first_selection_cutoff, *, states=None):
    """Replay matched prequential features and the saved convex optimum; no refit."""
    _kind(kind); _delay(delay)
    training = record['training']
    cutoff = ind.utc(first_selection_cutoff)
    if ind.utc(training['calibration_cutoff_utc']) != cutoff:
        raise ValueError('The readout cutoff differs from original first selection')
    if type(training['extra_days']) is not int or training['extra_days'] != delay or training['season_start_years'] != [2019, 2020, 2021]:
        raise ValueError('The readout delay or historical seasons differ')
    rows = calibration_population(records, cutoff, delay)
    if [r['match_id'] for r in training['rows']] != [r['match_id'] for r in rows]:
        raise ValueError('Readout training membership differs')
    if states is None:
        states = replay_states(records, [r['forecast_cutoff_utc'] for r in rows], delay, kind)
    x, y = [], []
    for row, saved_training in zip(rows, training['rows'], strict=True):
        state = states[ind.utc(row['forecast_cutoff_utc'])]
        x.append((state.get((row['league'], row['home']), 1000.) - state.get((row['league'], row['away']), 1000.)) / 400.)
        y.append(ind._outcome(row['payload']))
        parent.close(saved_training['x'], x[-1])
        parent.close(saved_training['y'], y[-1], tolerance=0)
        if ind.utc(saved_training['forecast_cutoff_utc']) != ind.utc(row['forecast_cutoff_utc']):
            raise ValueError('Readout row explanatory clock differs')
        if any(saved_training[key] != row[key] for key in ('league', 'season', 'home', 'away', 'day')):
            raise ValueError('Readout row fixture metadata differs')
        if ind.utc(saved_training['quote_available_at_utc']) != ind.availability(row['day'], row['league'], delay)[1]:
            raise ValueError('Readout row availability differs')
        provenance = saved_training['rating_provenance']
        if ind.utc(provenance['cutoff_utc']) != ind.utc(row['forecast_cutoff_utc']) or type(provenance['extra_days']) is not int or provenance['extra_days'] != delay:
            raise ValueError('Readout row prequential state clock differs')
    parent.close(record['x'], x)
    parent.close(record['y'], y, tolerance=0)
    if record['model']['training_digest'] != digest({'x': record['x'], 'y': record['y']}):
        raise ValueError('Saved readout fitting-input digest differs')
    audit = ind.ordered_objective_audit(record['model'], x, y)
    for key in ('objective', 'gradient', 'kkt_residual'):
        parent.close(record['model']['diagnostics'][key], audit[key])
    if audit['kkt_residual'] > 1e-6 or record['model']['diagnostics']['solver_success'] is not True:
        raise ValueError('Readout optimum failed its frozen acceptance tolerance')
    return {'rows': len(rows), 'training_ids_sha256': digest([r['match_id'] for r in rows]),
            'objective': audit['objective'], 'kkt_residual': audit['kkt_residual'], 'kind': kind, 'fits': 0}


def audit_report(rows, saved, spec, phase):
    if phase not in ('selection', 'transfer'):
        raise ValueError('Invalid verification phase')
    if spec['candidates'] != [CANDIDATE] or saved['selected_candidate'] != CANDIDATE:
        raise ValueError('The sole frozen candidate cannot be reselected')
    required = ['production_default_dc_auto', 'dc365_elo50', 'dc_180', INCUMBENT]
    if phase == 'selection':
        required += ['xg_add90_selection_safeguard', 'outcome_control', 'outcome_control_shot50']
    required += [CONTROL]
    if spec[phase]['references'] != required or list(saved['comparisons']) != [CANDIDATE] or set(saved['comparisons'][CANDIDATE]) != set(required):
        raise ValueError('Every frozen reference is mandatory')
    if len(rows) != spec[phase]['rows'] or digest([r['match_id'] for r in rows]) != spec[phase]['ordered_original_match_ids_sha256']:
        raise ValueError('The exact original scored population is required')
    parent.audit_report(rows, saved, spec, phase, candidate_inventory=[CANDIDATE])


def bound(item, bindings):
    path = ROOT / item['path']
    value = sha(path)
    if value != item['sha256'] or ('bytes' in item and path.stat().st_size != item['bytes']):
        raise ValueError('Closed input changed: ' + item['path'])
    if item['path'] in bindings and bindings[item['path']] != value:
        raise ValueError('Conflicting immutable binding')
    bindings[item['path']] = value
    return path


def bind_local(path, bindings):
    path = Path(path)
    return bound({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}, bindings)


def check_resources(value, spec):
    elapsed, rss = value['elapsed_seconds'], value['peak_rss_bytes']
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or not 0 <= elapsed <= spec['resources']['maximum_stage_wall_seconds']:
        raise ValueError('Invalid stage wall-time evidence')
    if type(rss) is not int or not 0 < rss <= spec['resources']['maximum_peak_rss_bytes']:
        raise ValueError('Invalid stage memory evidence')


def required_predecessors(phase, delay):
    _delay(delay)
    if phase not in ('selection', 'transfer'):
        raise ValueError('Unknown phase')
    result = ['selection_delay1'] if phase == 'transfer' or delay == 7 else []
    if phase == 'transfer':
        result.append('selection_delay7')
        if delay == 7:
            result.append('transfer_delay1')
    return result


def reference_check(spec, parent_spec, phase, references):
    expected = read(ROOT / spec['inputs'][f'references_{phase}']['path'])
    if phase == 'selection':
        previous = read(ROOT / spec['inputs']['primary_forecasts']['path'])
        if [r['match_id'] for r in previous] != [r['match_id'] for r in expected]:
            raise ValueError('Primary parent forecast order differs')
        for row, old in zip(expected, previous, strict=True):
            for key, value in row.items():
                if key == 'probabilities':
                    if any(old[key][name] != p for name, p in value.items()):
                        raise ValueError('Original reference vector differs')
                elif old[key] != value:
                    raise ValueError('Original primary fixture metadata differs')
            for name in ('outcome_control', 'outcome_control_shot50'):
                row['probabilities'][name] = old['probabilities'][name]
    if references != expected:
        raise ValueError('Saved legacy controls or reference metadata changed')
    if len(references) != spec[phase]['rows'] or digest([r['match_id'] for r in references]) != spec[phase]['ordered_original_match_ids_sha256']:
        raise ValueError('The complete original reference population is required')
    for row in references:
        if any(k in row for k in ('label', 'outcome', 'goals', 'y')):
            raise ValueError('Reference ledger contains outcome labels')
    parent.reference_check(parent_spec, phase, references)


def audit_query(query, batch, cutoff, delay, kind, state):
    """Recompute issued ratings and readout x, without trusting saved state."""
    _kind(kind); _delay(delay)
    provenance = query['provenance']
    if ind.utc(provenance['cutoff_utc']) != ind.utc(cutoff) or type(provenance['extra_days']) is not int or provenance['extra_days'] != delay:
        raise ValueError('Rating query clock or delay differs')
    if 'kind' in provenance and provenance['kind'] != kind:
        raise ValueError('Rating process provenance differs')
    if 'K' in provenance and provenance['K'] != (175. if kind == 'odds' else 14.):
        raise ValueError('Rating step-size provenance differs')
    queried = provenance['fixtures']
    if [r['match_id'] for r in queried] != [r['match_id'] for r in batch]:
        raise ValueError('Rating query fixture order differs')
    x = []
    for fixture, saved in zip(batch, queried, strict=True):
        rh = state.get((fixture['league'], fixture['home']), 1000.)
        ra = state.get((fixture['league'], fixture['away']), 1000.)
        parent.close(saved['rating_home'], rh)
        parent.close(saved['rating_away'], ra)
        x.append((rh - ra) / 400.)
        for key in ('home_last_update_utc', 'away_last_update_utc'):
            if saved.get(key) is not None and ind.utc(saved[key]) >= ind.utc(cutoff):
                raise ValueError('A rating provenance update was not strictly earlier')
    parent.close(query['x'], x)
    return x


def verify(out, phase, delay):
    out = Path(out)
    previous_stages = required_predecessors(phase, delay)
    check_failures(out)
    reject_operational_imports()
    bindings = {}
    design = read(bind_local(out / 'design_lock.json', bindings))
    for group in ('sources', 'inputs'):
        for path, expected in design[group].items():
            bound({'path': path, 'sha256': expected}, bindings)
    spec_path = HERE / 'specification.json'
    if design['sources'].get(str(spec_path.relative_to(ROOT))) != sha(spec_path):
        raise ValueError('The fixed specification is not execution-bound')
    spec = read(spec_path)
    for item in spec['inputs'].values():
        bound(item, bindings)
        if design['inputs'].get(item['path']) != item['sha256']:
            raise ValueError('A predeclared input is missing from the execution lock')
    parent_spec = read(ROOT / spec['inputs']['parent_specification']['path'])
    inherited = read(ROOT / spec['inputs']['parent_primary_verification']['path'])
    if inherited['status'] != 'passed' or inherited['passes_all_gates'] is not False or inherited['result_sha256'] != spec['inputs']['parent_primary_result']['sha256']:
        raise ValueError('The original failed-family verification differs')
    for path, expected in inherited['bindings'].items():
        if design['inputs'].get(path) != expected:
            raise ValueError('Inherited verification graph is incomplete')
    inherited_design = read(ROOT / spec['inputs']['parent_design_lock']['path'])
    for group in ('sources', 'inputs'):
        if any(design['inputs'].get(path) != expected for path, expected in inherited_design[group].items()):
            raise ValueError('Inherited execution graph is incomplete')
    for key in ('primary_forecasts', 'primary_market_readout', 'parent_primary_issuance_lock', 'references_selection', 'references_transfer'):
        item = spec['inputs'][key]
        if inherited['bindings'].get(item['path']) != item['sha256']:
            raise ValueError('Parent verification does not bind a reused input')
    for key in ('review', 'pre_fit_tests', 'synthetic_feasibility'):
        bound(design[key], bindings)
    review, tests = (read(ROOT / design[key]['path']) for key in ('review', 'pre_fit_tests'))
    if review.get('approved_for_execution_lock') is not True or review['source_files'] != design['sources'] or tests['exit_code'] != 0 or tests['source_files'] != design['sources']:
        raise ValueError('The source review or pre-fit tests differ')
    if any(type(design[key]) is not int or design[key] != expected for key, expected in [('maximum_new_readout_fits', 3), ('strength_fits', 0), ('historical_new_fits_before_lock', 0)]) or design['new_scores_before_lock'] is not False:
        raise ValueError('Frozen fit budget differs')
    data_lock = read(bind_local(out / 'data_lock.json', bindings))
    if data_lock['design_lock_sha256'] != sha(out / 'design_lock.json') or data_lock['references_labels_attached'] is not False:
        raise ValueError('Prepared data or reference label status differs')
    if type(data_lock['new_fits']) is not int or data_lock['new_fits'] != 0 or data_lock['new_scores'] is not False:
        raise ValueError('Preparation must contain no fits or new scores')
    for item in data_lock['files'].values():
        bound(item, bindings)
    references = read(out / f'references_{phase}.json')
    reference_check(spec, parent_spec, phase, references)
    raw_records = parent.raw_archive(parent_spec, read(out / 'archive_metadata.json'))
    raw_by_id = {r['match_id']: r for r in raw_records}
    stage = f'{phase}_delay{delay}'
    result, closure, decision, attempt = (read(bind_local(out / f'{stage}{suffix}.json', bindings))
                                         for suffix in ('', '_issuance_lock', '_decision_lock', '_attempt'))
    if attempt['design_lock_sha256'] != sha(out / 'design_lock.json'):
        raise ValueError('Stage attempt used another design lock')
    if result['design_lock_sha256'] != sha(out / 'design_lock.json') or result['data_lock_sha256'] != sha(out / 'data_lock.json') or closure['data_lock_sha256'] != sha(out / 'data_lock.json'):
        raise ValueError('Result design/data binding differs')
    if result['issuance_lock_sha256'] != sha(out / f'{stage}_issuance_lock.json') or closure['labels_attached'] is not False:
        raise ValueError('Forecasts were not closed without scoring labels')
    primary = phase == 'selection' and delay == 1
    expected_fits = (1 if primary else 2) if phase == 'selection' else 0
    if type(closure['optimizer_fits']) is not int or closure['optimizer_fits'] != expected_fits:
        raise ValueError('New optimizer count differs from the fixed budget')
    if set(closure['readouts']) != set(KINDS):
        raise ValueError('Both matched readouts are required')
    for item in [closure['forecasts'], closure['states'], *closure['readouts'].values()]:
        bound(item, bindings)
    predictions = read(ROOT / closure['forecasts']['path'])
    saved_states = read(ROOT / closure['states']['path'])
    readouts = {kind: read(ROOT / item['path']) for kind, item in closure['readouts'].items()}
    if delay == 1 and closure['readouts']['odds']['sha256'] != spec['inputs']['primary_market_readout']['sha256']:
        raise ValueError('The primary market readout bytes were changed or refitted')
    if len(predictions) != spec[phase]['rows'] or closure['rows'] != len(predictions) or [r['match_id'] for r in predictions] != [r['match_id'] for r in references]:
        raise ValueError('The full original issuance ledger is required')
    selection_refs = read(out / 'references_selection.json')
    first_selection = min(ind.utc(r['forecast_cutoff_utc']) for r in selection_refs)
    calibration = calibration_population(raw_records, first_selection, delay)
    clocks = [r['forecast_cutoff_utc'] for r in references] + [r['forecast_cutoff_utc'] for r in calibration]
    states = {kind: replay_states(raw_records, clocks, delay, kind) for kind in KINDS}
    readout_audits = {kind: audit_readout(readouts[kind], raw_records, delay, kind, first_selection, states=states[kind]) for kind in KINDS}
    if readout_audits['odds']['training_ids_sha256'] != readout_audits['result']['training_ids_sha256']:
        raise ValueError('Readout populations are not matched')
    parent_primary = {r['match_id']: r for r in read(ROOT / spec['inputs']['primary_forecasts']['path'])} if primary else None
    indexed_predictions = {r['match_id']: r for r in predictions}
    issuance_clocks = sorted({ind.utc(r['forecast_cutoff_utc']) for r in references})
    if len(saved_states) != len(issuance_clocks):
        raise ValueError('Every issuance clock needs one state record')
    replayed, supported_total, maximum_error = 0, 0, 0.
    for index, (clock, saved_state) in enumerate(zip(issuance_clocks, saved_states, strict=True)):
        batch = [r for r in references if ind.utc(r['forecast_cutoff_utc']) == clock]
        state_id = f'{index:04d}'
        if saved_state['state_id'] != state_id or ind.utc(saved_state['cutoff_utc']) != clock or saved_state['batch_ids'] != [r['match_id'] for r in batch]:
            raise ValueError('State identity, clock or fixture assignment differs')
        history = support_history(raw_records, clock, delay)
        ids = [r['match_id'] for r in history]
        if saved_state['support_training_ids'] != ids or saved_state['support_training_ids_sha256'] != digest(ids):
            raise ValueError('Support training membership differs')
        teams = {(r['league'], t) for r in history for t in (r['home'], r['away'])}
        supported = [(r['league'], r['home']) in teams and (r['league'], r['away']) in teams for r in batch]
        parent.close(saved_state['supported'], supported, tolerance=0)
        if set(saved_state['ratings']) != ({'result'} if primary else set(KINDS)):
            raise ValueError('The fixed operational rating inventory differs')
        for kind, query in saved_state['ratings'].items():
            audit_query(query, batch, clock, delay, kind, states[kind][clock])
        for reference, support in zip(batch, supported, strict=True):
            saved = indexed_predictions[reference['match_id']]
            if any(k in saved for k in ('label', 'outcome', 'goals', 'y')):
                raise ValueError('Scoring labels entered the issuance ledger')
            for key, value in reference.items():
                if key == 'probabilities':
                    if any(saved[key][name] != vector for name, vector in value.items()):
                        raise ValueError('Frozen reference probability bits changed')
                elif saved[key] != value:
                    raise ValueError('Original issuance metadata changed')
            if type(saved['support']) is not bool or saved['support'] is not support or saved['elo_state_id'] != state_id:
                raise ValueError('Forecast support or state binding differs')
            if set(saved['probabilities']) != {CANDIDATE, *spec[phase]['references']}:
                raise ValueError('Issued probability inventory differs')
            incumbent = reference['probabilities'][INCUMBENT]
            for kind, name in [('odds', CANDIDATE), ('result', CONTROL)]:
                s = states[kind][clock]
                rh, ra = s.get((reference['league'], reference['home']), 1000.), s.get((reference['league'], reference['away']), 1000.)
                expected = ind.ordered_logit_predict(readouts[kind]['model'], rh, ra) if support else np.asarray(incumbent)
                actual = ind.simplex(saved['probabilities'][name])
                error = float(np.max(np.abs(actual - expected)))
                maximum_error = max(maximum_error, error)
                if error > 2e-11 or (not support and saved['probabilities'][name] != incumbent):
                    raise ValueError('Independent probability replay or exact fallback differs')
                replayed += 1
            if primary and (saved['probabilities'][CANDIDATE] != parent_primary[reference['match_id']]['probabilities'][CANDIDATE] or parent_primary[reference['match_id']]['support'] is not support):
                raise ValueError('Primary candidate vector bits or support changed')
            supported_total += support
    if phase == 'selection':
        expected_support = spec['support']['expected_selection_supported_primary' if delay == 1 else 'expected_selection_supported_sensitivity']
        if supported_total != expected_support:
            raise ValueError('Fixed supported selection population differs')
    labeled = [{**r, 'label': ind._outcome(raw_by_id[r['match_id']]['payload'])} for r in predictions]
    audit_report(labeled, result['summary'], spec, phase)
    parent.close(decision['passes_all_gates'], result['summary']['passes_all_gates'], tolerance=0)
    if decision['selected_candidate'] != CANDIDATE or decision['result_sha256'] != sha(out / f'{stage}.json') or decision['design_lock_sha256'] != sha(out / 'design_lock.json'):
        raise ValueError('Decision identity or result binding differs')
    for previous in previous_stages:
        prior = read(bind_local(out / f'{previous}_verification.json', bindings))
        prior_result_path = bind_local(out / f'{previous}.json', bindings)
        prior_result = read(prior_result_path)
        if prior['status'] != 'passed' or prior['passes_all_gates'] is not True or prior_result['summary']['passes_all_gates'] is not True or prior_result['summary']['selected_candidate'] != CANDIDATE or prior['result_sha256'] != sha(prior_result_path):
            raise ValueError('Required predecessor did not independently pass')
        for path, expected in prior['bindings'].items():
            bound({'path': path, 'sha256': expected}, bindings)
        if ind.utc(prior['verified_at_utc']) > ind.utc(attempt['started_at_utc']):
            raise ValueError('A later stage began before its predecessor verification')
    times = [design['locked_at_utc'], data_lock['closed_at_utc'], attempt['started_at_utc']]
    if phase == 'selection':
        times.extend(readouts[k]['closed_at_utc'] for k in KINDS if not (primary and k == 'odds'))
    else:
        previous = read(out / f'selection_delay{delay}_issuance_lock.json')
        if previous['readouts'] != closure['readouts']:
            raise ValueError('Transfer readout parameters were refitted or rebound')
    times += [closure['closed_at_utc'], result['completed_at_utc'], decision['closed_at_utc']]
    if list(map(ind.utc, times)) != sorted(map(ind.utc, times)):
        raise ValueError('Forecast/label/decision chronology differs')
    for value in (data_lock['resources'], closure['resources'], result['resources']):
        check_resources(value, spec)
    for path, expected in bindings.items():
        if sha(ROOT / path) != expected:
            raise ValueError('An immutable binding changed during verification')
    check_failures(out)
    receipt = {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
               'stage': stage, 'result_sha256': sha(out / f'{stage}.json'), 'bindings': bindings,
               'rows': len(predictions), 'supported': supported_total,
               'candidate_and_control_vectors_replayed': replayed, 'maximum_prediction_error': maximum_error,
               'readout_optima_checked': readout_audits, 'fits_performed': 0,
               'passes_all_gates': result['summary']['passes_all_gates'], 'promotion': False,
               'scope': 'Independent raw membership and atomic rating replay, prequential readout optima, original vector reuse, all fixed metrics and gates; no optimizer rerun or receipt-time certification.'}
    with (out / f'{stage}_verification.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('selection', 'transfer'))
    parser.add_argument('--delay', type=int, choices=(1, 7), default=1)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    stage = f'{args.phase}_delay{args.delay}'
    with (args.out / f'{stage}_verification_attempt.json').open('x') as stream:
        json.dump({'started_at_utc': datetime.now(timezone.utc).isoformat()}, stream)
    try:
        result = verify(args.out, args.phase, args.delay)
    except BaseException as error:
        with (args.out / f'{stage}_verification_failure.json').open('x') as stream:
            json.dump({'failed_at_utc': datetime.now(timezone.utc).isoformat(), 'exception_type': type(error).__name__, 'message': str(error)}, stream)
        raise
    print(json.dumps({k: value for k, value in result.items() if k != 'bindings'}), flush=True)


if __name__ == '__main__':
    main()
