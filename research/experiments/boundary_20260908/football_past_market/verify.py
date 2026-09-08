"""Independently replay the closed football experiment without fitting.

No import of the experiment's data, model or runner modules is permitted.
Suggested commit: research(football): verify past-market forecasts and optima.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

from . import independent as ind

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts/research/boundary_20260908/football_past_market'
INCUMBENT = 'shot_strength_90d_ridge0.1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(actual, expected, *, tolerance=2e-11):
    if isinstance(actual, (bool, np.bool_)) or isinstance(expected, (bool, np.bool_)):
        if type(actual) is not bool or type(expected) is not bool or actual is not expected:
            raise ValueError('Independent boolean field differs')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Independent mapping schema differs')
        for key in expected:
            close(actual[key], expected[key], tolerance=tolerance)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Independent sequence differs')
        for a, b in zip(actual, expected):
            close(a, b, tolerance=tolerance)
    elif type(expected) in (float, int) and type(actual) in (float, int):
        if not np.isfinite([actual, expected]).all() or abs(actual - expected) > tolerance:
            raise ValueError(f'Independent numeric replay differs: {actual} versus {expected}')
    elif actual != expected:
        raise ValueError(f'Independent value differs: {actual!r} versus {expected!r}')


def bound(item, bindings):
    path = ROOT / item['path']
    actual = sha(path)
    if actual != item['sha256'] or ('bytes' in item and path.stat().st_size != item['bytes']):
        raise ValueError('Closed artifact changed: ' + str(path))
    bindings[item['path']] = actual
    return path


def raw_archive(spec, metadata):
    records, expected_metadata = [], []
    for binding in spec['inputs']['cached_csv_files']:
        path = ROOT / binding['path']
        match = re.fullmatch(r'(?:transfer_)?(E0|I1|SP1)_(20\d\d)_(20\d\d)\.csv', path.name)
        if not match or int(match[3]) != int(match[2]) + 1:
            raise ValueError('Independent source filename mismatch')
        league, season = match[1], int(match[2])
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for number, payload in enumerate(csv.DictReader(stream), 2):
                if not payload.get('Date') and not payload.get('HomeTeam') and not payload.get('AwayTeam'):
                    continue
                date_string = payload['Date']
                day = datetime.strptime(date_string, '%d/%m/%Y' if len(date_string.split('/')[-1]) == 4 else '%d/%m/%y').date()
                home, away = payload['HomeTeam'], payload['AwayTeam']
                if payload['Div'] != league or not home or not away or home == away:
                    raise ValueError('Independent metadata mismatch')
                mid = f'{league}:{season}:{home}:{away}'
                a0, _ = ind.availability(day, league, 1)
                issue, _ = ind.availability(day - timedelta(days=1), league, 1)
                meta = {'match_id': mid, 'league': league, 'season': season, 'day': day.isoformat(),
                        'home': home, 'away': away, 'forecast_cutoff_utc': issue.isoformat(), 'a0_utc': a0.isoformat(),
                        'availability_by_delay': {str(delay): ind.availability(day, league, delay)[1].isoformat() for delay in (1, 7)},
                        'source_path': binding['path'], 'source_sha256': binding['sha256'],
                        'source_row_number': number, 'source_row_hash': digest(payload)}
                expected_metadata.append(meta)
                records.append({**meta, 'payload': payload})
    if metadata != expected_metadata or len(records) != spec['quotes']['verified_rows']:
        raise ValueError('Independent raw CSV metadata replay differs')
    if len({row['match_id'] for row in records}) != len(records):
        raise ValueError('Duplicate raw identities')
    return records


def elo_states(records, clocks, delay):
    """Independent one-pass event chronology; payload access follows admission."""
    events = sorted(records, key=lambda row: (ind.availability(row['day'], row['league'], delay)[1], row['match_id']))
    states, position, ratings = {}, 0, {}
    for cutoff in sorted(set(map(ind.utc, clocks))):
        while position < len(events) and ind.availability(events[position]['day'], events[position]['league'], delay)[1] < cutoff:
            clock = ind.availability(events[position]['day'], events[position]['league'], delay)[1]
            observations = []
            while position < len(events) and ind.availability(events[position]['day'], events[position]['league'], delay)[1] == clock:
                row = events[position]
                odds = ind._odds(row['payload'], row['season'])
                if odds is not None:
                    q, _ = ind.devig(odds, 'normalized')
                    observations.append({'match_id': row['match_id'], 'league': row['league'],
                                         'home': row['home'], 'away': row['away'], 'q': q})
                position += 1
            ratings = ind.elo_atomic_update(ratings, observations)
        states[cutoff] = dict(ratings)
    return states


def reference_check(spec, phase, references):
    original = read(ROOT / spec['inputs']['football_selection' if phase == 'selection' else 'football_transfer']['path'])['predictions']
    if digest([row['match_id'] for row in references]) != spec[phase]['ordered_original_match_ids_sha256']:
        raise ValueError('Original ordered population differs')
    xg = read(ROOT / spec['inputs']['xg_selection_issued']['path']) if phase == 'selection' else None
    for i, (row, previous) in enumerate(zip(references, original, strict=True)):
        for key in ('match_id', 'league', 'season', 'home', 'away', 'day', 'forecast_cutoff_utc', 'fit_cutoff_utc', 'fit_id', 'result_available_at'):
            if row[key] != previous[key]:
                raise ValueError('Reference identity or clock changed')
        for name in ('production_default_dc_auto', 'dc365_elo50', 'dc_180', INCUMBENT):
            if row['probabilities'][name] != previous['probabilities'][name]:
                raise ValueError('Original probability bits changed')
        if xg is not None:
            if any(xg[i][key] != row[key] for key in ('match_id', 'forecast_cutoff_utc', 'fit_cutoff_utc')):
                raise ValueError('Original xG identity or clock differs')
            if row['probabilities']['xg_add90_selection_safeguard'] != xg[i]['probabilities']['xg_add90']:
                raise ValueError('Original xG vector changed')


def audit_report(rows, saved, spec, phase, *, candidate_inventory=None):
    if saved['rows'] != len(rows) or saved['population_sha256'] != digest([row['match_id'] for row in rows]):
        raise ValueError('Scored population changed')
    candidates = list(saved['comparisons'])
    if candidate_inventory is None:
        candidate_inventory = spec['candidates'] if phase == 'selection' else [saved['selected_candidate']]
    if candidates != candidate_inventory:
        raise ValueError('The fixed candidate inventory differs')
    refs = spec[phase]['references']
    names = candidates + refs
    metrics = {name: ind.metrics(rows, name) for name in names}
    close(saved['metrics'], metrics)
    independent_comparisons = {}
    for candidate in candidates:
        independent_comparisons[candidate] = {}
        for ref in refs:
            independent_comparisons[candidate][ref] = {}
            for days in spec['uncertainty']['blocks_calendar_days']:
                independently = ind.paired_uncertainty(rows, candidate, ref, days, spec)
                independent_comparisons[candidate][ref][str(days)] = independently
                close(saved['comparisons'][candidate][ref][str(days)], independently)
    winner = saved['selected_candidate']
    if len(candidates) > 1 and winner != min(spec['candidates'], key=lambda name: metrics[name]['log_loss']):
        raise ValueError('Candidate selection differs')
    country = {league: {name: ind.metrics([row for row in rows if row['league'] == league], name) for name in names}
               for league in sorted({row['league'] for row in rows})}
    seasons = {f'{league}:{year}': {name: ind.metrics([row for row in rows if row['league'] == league and row['season'] == year], name) for name in names}
               for league, year in sorted({(row['league'], row['season']) for row in rows})}
    close(saved['by_country'], country)
    close(saved['by_country_season'], seasons)
    checks = {}
    for ref in refs:
        record = {'relative_nll_gain': 1 - metrics[winner]['log_loss'] / metrics[ref]['log_loss'],
                  'minimum_gain_met': metrics[winner]['log_loss'] <= (1 - spec[phase]['minimum_relative_nll_gain_each_reference']) * metrics[ref]['log_loss'],
                  'negative_28day_upper': independent_comparisons[winner][ref]['28']['percentile_95_interval'][1] < 0}
        if phase == 'transfer':
            record.update({'each_country_improves': all(v[winner]['log_loss'] < v[ref]['log_loss'] for v in country.values()),
                           'no_country_season_nll_deterioration': all(v[winner]['log_loss'] <= v[ref]['log_loss'] for v in seasons.values()),
                           'pooled_brier_nonworse': metrics[winner]['brier_sum_classes'] <= metrics[ref]['brier_sum_classes'],
                           'country_season_brier_nonworse': all(v[winner]['brier_sum_classes'] <= v[ref]['brier_sum_classes'] for v in seasons.values())})
        checks[ref] = record
    close(saved['checks'], checks)
    passed = all(v for record in checks.values() for key, v in record.items() if key != 'relative_nll_gain')
    if type(saved['passes_all_gates']) is not bool or passed != saved['passes_all_gates'] or saved['promotion'] is not False:
        raise ValueError('Advancement decision differs')
    if phase == 'transfer':
        subset = [row for row in rows if row['match_id'] != 'I1:2024:Fiorentina:Inter']
        expected = {'rows': len(subset), 'metrics': {name: ind.metrics(subset, name) for name in names},
                    'paired_28day': {ref: ind.paired_uncertainty(subset, winner, ref, 28, spec) for ref in refs}}
        close(saved['resumed_fixture_excluded'], expected)


def verify(out, phase, delay):
    if any(out.glob('*_failure.json')):
        raise ValueError('A terminal failed stage cannot be verified as an eligible result')
    for name in ('model', 'data', 'run'):
        if f'research.experiments.boundary_20260908.football_past_market.{name}' in sys.modules:
            raise ValueError('Verifier must not import fitting or operational modules')
    spec = read(HERE / 'specification.json')
    stage = f'{phase}_delay{delay}'
    bindings = {}
    design = read(out / 'design_lock.json')
    for group in ('sources', 'inputs'):
        for path, expected in design[group].items():
            bound({'path': path, 'sha256': expected}, bindings)
    for key in ('review', 'pre_fit_tests', 'synthetic_feasibility'):
        bound(design[key], bindings)
    review = read(ROOT / design['review']['path'])
    tests = read(ROOT / design['pre_fit_tests']['path'])
    if review.get('approved_for_execution_lock') is not True or review['source_files'] != design['sources']:
        raise ValueError('Independent source review differs from the execution lock')
    if tests['exit_code'] != 0 or tests['source_files'] != design['sources']:
        raise ValueError('Pre-fit test source receipt differs')
    data_lock = read(out / 'data_lock.json')
    if data_lock['design_lock_sha256'] != sha(out / 'design_lock.json'):
        raise ValueError('Prepared source lock differs')
    for item in data_lock['files'].values():
        bound(item, bindings)
    references = read(out / f'references_{phase}.json')
    reference_check(spec, phase, references)
    records = raw_archive(spec, read(out / 'archive_metadata.json'))
    by_id = {row['match_id']: row for row in records}
    for filename in ('design_lock.json', 'data_lock.json', f'{stage}.json', f'{stage}_issuance_lock.json', f'{stage}_decision_lock.json'):
        path = out / filename
        bindings[str(path.relative_to(ROOT))] = sha(path)
    result = read(out / f'{stage}.json')
    closure = read(out / f'{stage}_issuance_lock.json')
    if result['design_lock_sha256'] != sha(out / 'design_lock.json') or result['data_lock_sha256'] != sha(out / 'data_lock.json'):
        raise ValueError('Result design or preparation lock differs')
    if closure['data_lock_sha256'] != sha(out / 'data_lock.json'):
        raise ValueError('Forecast closure uses different prepared data')
    if result['issuance_lock_sha256'] != sha(out / f'{stage}_issuance_lock.json') or closure['labels_attached'] is not False:
        raise ValueError('Result is not bound to an unlabeled forecast closure')
    for key in ('forecasts', 'fit_index', 'elo_readout'):
        bound(closure[key], bindings)
    predictions = read(ROOT / closure['forecasts']['path'])
    if len(predictions) != spec[phase]['rows'] or closure['rows'] != len(predictions):
        raise ValueError('Closed population differs')
    prediction_by_id = {row['match_id']: row for row in predictions}
    if list(prediction_by_id) != [row['match_id'] for row in references]:
        raise ValueError('Closed forecast identities/order differ')
    fit_index = read(ROOT / closure['fit_index']['path'])
    if len(fit_index) != spec[phase]['unique_forecast_clocks'] or closure['optimizer_fits'] != 2 * len(fit_index) + int(phase == 'selection'):
        raise ValueError('Actual fit count differs from the fixed protocol')
    for item in fit_index:
        bound(item, bindings)
    readout = read(ROOT / closure['elo_readout']['path'])
    calibration = readout['training']['rows']
    clocks = [row['forecast_cutoff_utc'] for row in references] + [row['forecast_cutoff_utc'] for row in calibration]
    states = elo_states(records, clocks, delay)
    calibration_cutoff = ind.utc(readout['training']['calibration_cutoff_utc'])
    first_selection = min(ind.utc(row['forecast_cutoff_utc']) for row in read(out / 'references_selection.json'))
    if calibration_cutoff != first_selection:
        raise ValueError('Readout was not frozen at the original first selection cutoff')
    expected_calibration = sorted((row for row in records if row['season'] in (2019, 2020, 2021)
                                  and ind.availability(row['day'], row['league'], delay)[1] < calibration_cutoff),
                                 key=lambda row: (ind.utc(row['forecast_cutoff_utc']), row['match_id']))
    expected_calibration = [row for row in expected_calibration if ind._odds(row['payload'], row['season']) is not None]
    if [row['match_id'] for row in calibration] != [row['match_id'] for row in expected_calibration]:
        raise ValueError('Readout training population differs')
    x, y = [], []
    for row in expected_calibration:
        state = states[ind.utc(row['forecast_cutoff_utc'])]
        x.append((state.get((row['league'], row['home']), 1000.) - state.get((row['league'], row['away']), 1000.)) / 400.)
        y.append(ind._outcome(row['payload']))
    close(readout['x'], x)
    close(readout['y'], y, tolerance=0)
    audit = ind.ordered_objective_audit(readout['model'], x, y)
    for key in ('objective', 'gradient', 'kkt_residual'):
        close(readout['model']['diagnostics'][key], audit[key])
    if audit['kkt_residual'] > 1e-6 or readout['model']['diagnostics']['solver_success'] is not True:
        raise ValueError('Readout optimum failed its original tolerance')
    ordered_records = sorted(records, key=lambda row: (ind.availability(row['day'], row['league'], delay)[1], row['match_id']))
    replay_count, maximum_error, seen = 0, 0., set()
    names = list(result['summary']['comparisons'])
    for index, item in enumerate(fit_index):
        fit = read(ROOT / item['path'])
        history = ind.history_membership(ordered_records, fit['cutoff_utc'], extra_days=delay)
        if history['training_ids'] != fit['training_ids'] or digest(fit['training_ids']) != fit['training_ids_sha256']:
            raise ValueError('Independent fit membership differs')
        close(fit['weights'], history['raw_weights'])
        for kind in ('past_market', 'outcome_control'):
            payload = fit['models'][kind]
            if payload['training_ids'] != history['training_ids'] or payload['target_kind'] != kind:
                raise ValueError('Serialized training identities or targets differ')
            audit = ind.strength_objective_audit(payload, history['rows'])
            for key in ('objective', 'gradient', 'gradient_max'):
                close(payload['diagnostics'][key], audit[key])
            if audit['gradient_max'] > 1e-6 or payload['diagnostics']['solver_success'] is not True:
                raise ValueError('Strength optimum failed its original tolerance')
        teams = {(row['league'], team) for row in history['rows'] for team in (row['home'], row['away'])}
        state = states[ind.utc(fit['cutoff_utc'])]
        batch = [row for row in references if ind.utc(row['forecast_cutoff_utc']) == ind.utc(fit['cutoff_utc'])]
        if fit['batch_ids'] != [row['match_id'] for row in batch]:
            raise ValueError('Fit issued the wrong clock population')
        for i, reference in enumerate(batch):
            saved = prediction_by_id[reference['match_id']]
            if any(key in saved for key in ('label', 'outcome', 'goals', 'y')):
                raise ValueError('Issued ledger contains scoring outcomes')
            for key, value in reference.items():
                if key == 'probabilities':
                    if any(saved[key][name] != vector for name, vector in value.items()):
                        raise ValueError('Issued incumbent vector changed')
                elif saved[key] != value:
                    raise ValueError('Issued reference field changed')
            league, home, away = (reference[key] for key in ('league', 'home', 'away'))
            supported = (league, home) in teams and (league, away) in teams
            if saved['support'] != supported or fit['supported'][i] != supported or saved['past_market_fit_id'] != fit['fit_id']:
                raise ValueError('Support mask or fit assignment differs')
            incumbent = reference['probabilities'][INCUMBENT]
            rh, ra = state.get((league, home), 1000.), state.get((league, away), 1000.)
            close(fit['elo']['x'][i], (rh - ra) / 400.)
            if supported:
                past = ind.strength_predict(fit['models']['past_market'], league, home, away)
                control = ind.strength_predict(fit['models']['outcome_control'], league, home, away)
                rebuilt = {'past_market': past, 'past_market_shot50': ind.fallback_or_blend(past, incumbent, supported=True, blend=True),
                           'outcome_control': control, 'outcome_control_shot50': ind.fallback_or_blend(control, incumbent, supported=True, blend=True),
                           'elo_odds_reference': ind.ordered_logit_predict(readout['model'], rh, ra)}
            else:
                rebuilt = {name: np.asarray(incumbent) for name in [*names, 'outcome_control', 'outcome_control_shot50', 'elo_odds_reference']}
            for name in [*names, 'outcome_control', 'outcome_control_shot50', 'elo_odds_reference']:
                error = float(np.max(np.abs(np.asarray(saved['probabilities'][name]) - rebuilt[name])))
                maximum_error = max(maximum_error, error)
                if error > 2e-11 or (not supported and saved['probabilities'][name] != incumbent):
                    raise ValueError('Independent probability replay differs')
                replay_count += 1
            seen.add(reference['match_id'])
        if index % 40 == 0 or index + 1 == len(fit_index):
            print(json.dumps({'verifying': stage, 'clocks': index + 1, 'total': len(fit_index)}), flush=True)
    if seen != set(prediction_by_id):
        raise ValueError('Not every forecast was independently replayed')
    labeled = [{**row, 'label': ind._outcome(by_id[row['match_id']]['payload'])} for row in predictions]
    expected_candidates = spec['candidates'] if phase == 'selection' and delay == 1 else [read(out / 'selection_delay1.json')['summary']['selected_candidate']]
    audit_report(labeled, result['summary'], spec, phase, candidate_inventory=expected_candidates)
    decision = read(out / f'{stage}_decision_lock.json')
    close(decision['passes_all_gates'], result['summary']['passes_all_gates'], tolerance=0)
    if decision['result_sha256'] != sha(out / f'{stage}.json') or decision['passes_all_gates'] != result['summary']['passes_all_gates']:
        raise ValueError('Decision receipt differs')
    if decision['design_lock_sha256'] != sha(out / 'design_lock.json') or decision['selected_candidate'] != result['summary']['selected_candidate']:
        raise ValueError('Selected candidate or decision design differs')
    attempt = read(out / f'{stage}_attempt.json')
    milestones = [design['locked_at_utc'], data_lock['closed_at_utc'], attempt['started_at_utc']]
    if phase == 'selection':
        milestones.append(readout['closed_at_utc'])
    milestones += [closure['closed_at_utc'], result['completed_at_utc'], decision['closed_at_utc']]
    if list(map(ind.utc, milestones)) != sorted(map(ind.utc, milestones)):
        raise ValueError('Execution chronology violates the locked forecast/scoring order')
    if delay == 7 or phase == 'transfer':
        stages = ['selection_delay1'] + (['selection_delay7'] if phase == 'transfer' else [])
        for previous in stages:
            predecessor = read(out / f'{previous}_verification.json')
            if predecessor['status'] != 'passed' or predecessor['passes_all_gates'] is not True:
                raise ValueError('Predecessor did not independently pass')
            if predecessor['result_sha256'] != sha(out / f'{previous}.json'):
                raise ValueError('Predecessor changed after verification')
            if read(out / f'{previous}.json')['summary']['selected_candidate'] != decision['selected_candidate']:
                raise ValueError('Candidate was reselected after primary selection')
    for path, expected in bindings.items():
        if sha(ROOT / path) != expected:
            raise ValueError('A verified binding changed during replay')
    if any(out.glob('*_failure.json')):
        raise ValueError('A terminal failure appeared during verification')
    receipt = {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
               'stage': stage, 'result_sha256': sha(out / f'{stage}.json'), 'bindings': bindings,
               'rows': len(predictions), 'candidate_and_control_vectors_replayed': replay_count,
               'maximum_prediction_error': maximum_error, 'strength_optima_checked': 2 * len(fit_index),
               'readout_optima_checked': 1, 'fits_performed': 0,
               'passes_all_gates': result['summary']['passes_all_gates'],
               'scope': 'Independent CSV membership, clock/Elo replay, parameter optimum checks, all forecast vectors and scoring decisions; no optimizer rerun.'}
    with (out / f'{stage}_verification.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'bindings'}), flush=True)


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
        verify(args.out, args.phase, args.delay)
    except BaseException as error:
        with (args.out / f'{stage}_verification_failure.json').open('x') as stream:
            json.dump({'failed_at_utc': datetime.now(timezone.utc).isoformat(),
                       'exception_type': type(error).__name__, 'message': str(error)}, stream)
        raise


if __name__ == '__main__':
    main()
