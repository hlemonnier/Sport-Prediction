"""No-refit independent deployment-policy artifact and numerical verification.

This module intentionally imports no canonical runtime/model/data module.
Suggested commit: research(football): verify deployment score distributions independently.
"""
from __future__ import annotations

from collections import Counter
import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

from . import independent as ind

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'artifacts/research/boundary_20260908/football_deployment_policy'
METADATA = ('match_id', 'league', 'season', 'day', 'home', 'away', 'forecast_cutoff_utc',
            'fit_cutoff_utc', 'fit_id', 'result_available_at')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_operational_imports():
    forbidden = ('packages.football.mrp.',
                 'research.experiments.boundary_20260908.football_deployment_policy.run',
                 'research.experiments.boundary_20260908.football_deployment_policy.runtime',
                 'research.experiments.boundary_20260908.football_deployment_policy.data',
                 'research.experiments.boundary_20260908.football_deployment_policy.scoring')
    if any(name.startswith(forbidden) for name in sys.modules):
        raise ValueError('Independent historical verification requires no operational imports')


def check_failures(out):
    if any(Path(out).glob('*_failure.json')):
        raise ValueError('A terminal failed attempt cannot verify as passed')


def bound(record, root, bindings):
    path = Path(root) / record['path']
    actual = sha(path)
    if actual != record['sha256'] or ('bytes' in record and path.stat().st_size != record['bytes']):
        raise ValueError('Artifact byte binding differs: '+record['path'])
    if record['path'] in bindings and bindings[record['path']] != actual:
        raise ValueError('Conflicting artifact binding')
    bindings[record['path']] = actual
    return path


def close(a, b, *, atol=1e-12, context='numeric replay'):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or not np.allclose(a, b, rtol=0, atol=atol):
        raise ValueError('Independent mismatch: '+context)
    return float(np.max(np.abs(a-b))) if a.size else 0.


def legacy(clock):
    return ind.utc(clock).replace(tzinfo=None).isoformat()


def csv_metadata(path, binding, *, expected_rows=380, expected_teams=20):
    match = re.fullmatch(r'(?:transfer_)?(E0|SP1|I1)_(20\d\d)_(20\d\d)\.csv', Path(path).name)
    if match is None or int(match[3]) != int(match[2])+1:
        raise ValueError('Unexpected raw CSV filename')
    league, season = match[1], int(match[2])
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for number, payload in enumerate(csv.DictReader(stream), 2):
            if not any(payload.values()):
                continue
            if None in payload or payload.get('Div') != league:
                raise ValueError('Raw CSV league/column shape differs')
            day_value = payload['Date']
            day = datetime.strptime(day_value, '%d/%m/%Y' if len(day_value.split('/')[-1]) == 4 else '%d/%m/%y').date()
            home, away = payload['HomeTeam'].strip(), payload['AwayTeam'].strip()
            if not date(season, 7, 1) <= day <= date(season+1, 8, 31) or not home or not away or home == away:
                raise ValueError('Invalid raw metadata')
            rows.append({'match_id': f'{league}:{season}:{home}:{away}', 'league': league, 'season': season,
                         'day': day.isoformat(), 'home': home, 'away': away,
                         'forecast_cutoff_utc': legacy(ind.local_midnight(day, league)),
                         'result_available_at': legacy(ind.local_midnight(day+timedelta(days=1), league)),
                         'source_path': binding['path'], 'source_sha256': binding['sha256'],
                         'source_row_number': number, 'source_row_hash': ind.digest(payload), 'payload': payload})
    counts = Counter(team for row in rows for team in (row['home'], row['away']))
    if len(rows) != expected_rows or len({r['match_id'] for r in rows}) != len(rows) or len(counts) != expected_teams or set(counts.values()) != {2*(expected_teams-1)}:
        raise ValueError('Raw CSV is not the complete canonical season')
    return rows


def goals(row):
    payload = row['payload']
    values = []
    for name in ('FTHG', 'FTAG'):
        raw = payload[name]
        if type(raw) is not str or re.fullmatch(r'\s*\+?\d+\s*', raw) is None:
            raise ValueError('Raw score must be a nonnegative integer string')
        values.append(int(raw))
    label = 0 if values[0] > values[1] else 1 if values[0] == values[1] else 2
    if payload['FTR'] != 'HDA'[label]:
        raise ValueError('Raw score/result mismatch')
    return values, label


def match_record(row):
    """Independent dictionary equivalent of the fixed canonical input record."""
    h, a = goals(row)[0]
    result = {'match_id': row['match_id'], 'date': row['forecast_cutoff_utc'],
              'season': row['season'], 'league': 'epl' if row['league'] == 'E0' else row['league'],
              'round_number': None, 'home_team_id': row['home'], 'away_team_id': row['away'],
              'home_goals': h, 'away_goals': a, 'result_available_at': row['result_available_at']}
    for key in ('home_xg', 'away_xg', 'venue_name', 'venue_latitude', 'venue_longitude', 'timezone', 'xg_available_at'):
        result[key] = None
    return result


def partition_metadata(records):
    groups = ind.chronological_partitions(records)
    by_id = {r['match_id']: r for r in records}
    result = {'name': 'disjoint_chronological_fit_calibration_selection_test_v1',
              'evaluation_status': 'held_out' if groups['test'] else 'insufficient_history',
              'parameter_policy': 'Frozen fit-prefix parameters are used for evaluation and fixtures; completed history may update form features.',
              'availability_policy': 'explicit result publication time, otherwise following UTC day',
              'excluded_boundary_ids': groups['excluded_boundary_ids']}
    for name in ('fit', 'calibration', 'selection', 'test'):
        ids = groups[name]; values = [by_id[x] for x in ids]
        result[name] = {'sample_size': len(ids), 'match_ids': ids, 'data_sha256': ind.digest(values),
                        'population_sha256': ind.digest(ids),
                        'first_kickoff': min((r['date'] for r in values), default=None),
                        'last_kickoff': max((r['date'] for r in values), default=None),
                        'latest_result_available_at': max((r['result_available_at'] for r in values), default=None)}
    return result


def audit_model(state, history):
    teams = {team for r in history for team in (r['home_team_id'], r['away_team_id'])}
    if set(state['attack']) != teams or set(state['defense']) != teams:
        raise ValueError('Serialized model teams differ from actual training membership')
    diag = state['diagnostics']
    ids = [r['match_id'] for r in history]
    if diag.get('status') != 'fitted' or diag.get('converged') is not True or type(diag.get('fit_sample_size')) is not int or diag['fit_sample_size'] != len(history) or diag.get('half_life_days') is not None:
        raise ValueError('Invalid successful equal-weight fit diagnostics')
    if 'fit_match_ids' in diag and diag['fit_match_ids'] != ids:
        raise ValueError('Serialized optimizer fit identities differ')
    if abs(math.fsum(state['attack'].values())) > 1e-7 or abs(math.fsum(state['defense'].values())) > 1e-7 or not -.2 <= ind.finite(state['rho']) <= .2:
        raise ValueError('Fitted state centering/rho constraint violation')
    if any(abs(ind.finite(v)) > 3+1e-7 for group in ('attack', 'defense') for v in state[group].values()):
        raise ValueError('Fitted team-effect bound violation')
    unseen = '__independent_neutral__'
    while unseen in teams:
        unseen += '_'
    names = [*sorted(teams), unseen]
    for home in names:
        for away in names:
            # Same-name Cartesian feasibility is a kernel constraint, not a fixture.
            lh = math.exp(ind.finite(state['home_intercept'])+state['attack'].get(home, 0.)+state['defense'].get(away, 0.))
            la = math.exp(ind.finite(state['away_intercept'])+state['attack'].get(away, 0.)+state['defense'].get(home, 0.))
            ind.rho_support(lh, la, state['rho'])
            if min(lh, la) < .05-1e-7 or max(lh, la) > 6+1e-7:
                raise ValueError('Serialized all-pair goal rate violates bounds')


def audit_output(saved, goal_state, calibrator_state, fixture):
    expected = ind.replay_output(goal_state, calibrator_state, fixture['home'], fixture['away'])
    if saved['match_id'] != fixture['match_id'] or saved['date'] != fixture['forecast_cutoff_utc'] or saved['home_team_id'] != fixture['home'] or saved['away_team_id'] != fixture['away'] or saved['season'] != fixture['season'] or saved['league'] != ('epl' if fixture['league'] == 'E0' else fixture['league']):
        raise ValueError('Serialized fixture metadata differs')
    if saved['reconciled'] is not True or saved['dimensions'] != expected['dimensions'] or saved['mode'][:2] != expected['mode'][:2]:
        raise ValueError('Serialized joint support/mode differs')
    maximum = 0.
    for key, value in expected.items():
        if key == 'reconciled':
            continue
        maximum = max(maximum, close(saved[key], value, atol=1e-10 if key in ('expected_goals', 'hda', 'raw_hda', 'selected_hda') else 1e-12, context=key))
    return maximum


def audit_caller(policy_record):
    states = policy_record['fixtures']; rows = policy_record['canonical_rows']
    diagnostics = policy_record['diagnostics']
    if [r['match_id'] for r in rows] != [s['match_id'] for s in states] or set(diagnostics['fixture_distributions']) != {s['match_id'] for s in states}:
        raise ValueError('Actual caller fixture population differs')
    for output, row in zip(states, rows):
        saved = diagnostics['fixture_distributions'][row['match_id']]
        for key, target in [('score_probability_matrix', 'matrix'), ('outcome_probabilities', 'hda'),
                            ('expected_goals', 'expected_goals'), ('base_omitted_probability_mass', 'omitted_probability_mass'),
                            ('reconciled_tail_error_bound', 'tail_probability_bound')]:
            close(saved[key], output[target], atol=1e-10 if target in ('hda', 'expected_goals') else 1e-12, context='actual caller '+key)
        close([float(row[k]) for k in ('home_win_prob', 'draw_prob', 'away_win_prob')], output['hda'], atol=6e-13)
        close([float(row[k]) for k in ('raw_home_win_prob', 'raw_draw_prob', 'raw_away_win_prob')], output['raw_hda'], atol=6e-13)
        close([float(row[k]) for k in ('expected_home_goals', 'expected_away_goals')], output['expected_goals'], atol=5.00001e-7)
        close([float(row[k]) for k in ('dixon_base_lambda_home_goals', 'dixon_base_lambda_away_goals')], [output['lambda_home'], output['lambda_away']], atol=5.00001e-7)
        if row['predicted_scoreline'] != f"{output['mode'][0]}-{output['mode'][1]}" or row['model_used'] != 'dixon' or row['gbdt_enabled'] != 'false':
            raise ValueError('Actual caller mode/policy differs')
        close(float(row['scoreline_prob']), output['mode'][2], atol=6e-13)


def audit_block(record, descriptor, archive):
    """Replay saved parameters and both native caller records without any fit."""
    admitted = ind.history_membership(archive, descriptor['fit_cutoff_utc'], descriptor['league'])
    ids = [r['match_id'] for r in admitted]
    lineage = descriptor['saved_full_lineage']
    if ids != lineage['fit_match_ids'] or ind.digest(ids) != lineage['fit_ids_sha256']:
        raise ValueError('Independent full-history membership differs')
    records = [match_record(r) for r in admitted]
    compact = [[r['match_id'], r['day'], *goals(r)[0]] for r in admitted]
    if ind.digest(compact) != lineage['fit_content_sha256']:
        raise ValueError('Independent admitted training content differs')
    if lineage['latest_result_available_at'] != max(r['result_available_at'] for r in records) or lineage['cutoff_utc'] != descriptor['fit_cutoff_utc']:
        raise ValueError('Original full-history timing lineage differs')
    partitions = partition_metadata(records)
    if partitions != descriptor['expected_partitions']:
        raise ValueError('Independent prefix/calibration/selection/test partitions differ')
    by_id = {r['match_id']: r for r in records}
    fixtures = descriptor['fixture_metadata']
    if set(ids) & {r['match_id'] for r in fixtures}:
        raise ValueError('A forecast fixture entered fitted history')
    prefix = [by_id[x] for x in partitions['fit']['match_ids']]
    reference, candidate = record['reference'], record['candidate']
    if reference['policy'] != ind.REFERENCE or candidate['policy'] != ind.CANDIDATE:
        raise ValueError('Fixed policy identity differs')
    audit_model(reference['model_state'], prefix)
    audit_model(candidate['model_state'], records)
    if candidate['calibrator_state']['method'] != 'identity' or any(x != {'kind': 'identity'} for x in candidate['calibrator_state']['class_functions']):
        raise ValueError('Full-history policy must use identity calibration')
    if candidate['assessment_metrics_computed'] is not False:
        raise ValueError('Full-history fixture model must not emit held-out assessment metrics')
    line = candidate['lineage']
    for key, expected in [('fit_match_ids', ids), ('fit_ids_sha256', ind.digest(ids)),
                          ('match_records_sha256', ind.digest(records)), ('fit_sample_size', len(ids)),
                          ('cutoff_utc', descriptor['fit_cutoff_utc']),
                          ('latest_result_available_at', max(r['result_available_at'] for r in records)),
                          ('weight_policy', 'equal'), ('half_life_days', None), ('calibration_policy', 'off')]:
        if line[key] != expected:
            raise ValueError('Full-history fixture lineage differs: '+key)
    ref_line = reference['lineage']
    if ref_line['assessment_partitions'] != partitions or ref_line['goal_fit_ids'] != partitions['fit']['match_ids'] or ref_line['calibration_ids'] != partitions['calibration']['match_ids'] or ref_line['cutoff_utc'] != descriptor['fit_cutoff_utc']:
        raise ValueError('Disjoint assessment model lineage differs')
    if candidate['restoration_branch'] == 'validated_saved_full_history_state':
        saved = dict(descriptor['saved_full_model']); saved.setdefault('schema', 'football_dc_state_v1')
        if candidate['model_state'] != saved or candidate['new_goal_model_fits'] != 0:
            raise ValueError('Restored model is not the exact saved full-history state')
    elif candidate['restoration_branch'] != 'canonical_full_history_fit' or candidate['new_goal_model_fits'] != 1:
        raise ValueError('Unknown or inconsistent full-history reconstruction branch')
    if any(record['parity'].get(k) is not True for k in ('reference_actual_caller', 'reference_serialized_replay',
            'candidate_state_injected_caller_output', 'old_reference_hda_checked', 'old_candidate_hda_checked')):
        raise ValueError('All required caller parity proofs must be explicit true')
    counts = record['fit_counts']
    expected_counts = {'prefix_goal_fits': 1, 'full_history_goal_fits': candidate['new_goal_model_fits'],
                       'reference_calibration_policy_calls': 1, 'candidate_learned_calibration_fits': 0, 'candidate_parity_goal_fits': 0}
    if counts != expected_counts or any(type(x) is not int for x in counts.values()):
        raise ValueError('Block optimizer fit budget differs')
    maximum = 0.
    for policy, old in [(reference, 'production_default_dc_auto'), (candidate, 'dc_equal')]:
        if any(policy['diagnostics']['protocol'][key] != value for key, value in partitions.items()):
            raise ValueError('Actual caller assessment partition lineage differs')
        if [r['match_id'] for r in policy['fixtures']] != [r['match_id'] for r in fixtures]:
            raise ValueError('Block forecast membership/order differs')
        for output, fixture in zip(policy['fixtures'], fixtures):
            maximum = max(maximum, audit_output(output, policy['model_state'], policy['calibrator_state'], fixture))
            close(output['hda'], fixture['probabilities'][old], atol=1e-10, context='closed original vector parity')
        audit_caller(policy)
    return {'fixtures': len(fixtures), 'maximum_output_error': maximum, 'fit_counts': counts}


def audit_report(actual, expected, path='report'):
    """Structural replay including exact independently determined gate booleans."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('Report fields differ: '+path)
        for key in expected:
            audit_report(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError('Report array differs: '+path)
        for i, (a, b) in enumerate(zip(actual, expected)):
            audit_report(a, b, path+f'[{i}]')
    elif isinstance(expected, bool):
        if actual is not expected:
            raise ValueError('Independent gate/boolean differs: '+path)
    elif expected is None or isinstance(expected, str):
        if actual != expected:
            raise ValueError('Report semantic value differs: '+path)
    elif isinstance(expected, int):
        if type(actual) is not int or actual != expected:
            raise ValueError('Report integer differs: '+path)
    else:
        close(actual, expected, atol=1e-11, context=path)


def reference_inputs(spec, root, inputs, bindings):
    """Read raw metadata and old state ancestry independently of the new loader."""
    def load(record):
        if inputs.get(record['path']) != record['sha256']:
            raise ValueError('Pinned input is absent from the execution input lock')
        return read(bound(record, root, bindings))
    cfg = spec['inputs']
    old = load(cfg['old_evaluation']); prior = load(cfg['old_verification']); design = load(cfg['old_design_lock'])
    source_map = old['source_files']
    sm = cfg['inherited_source_map']
    if len(source_map) != sm['entries'] or ind.digest(source_map) != sm['canonical_sha256'] or any(inputs.get(p) != h for p, h in source_map.items()):
        raise ValueError('Original verified source ancestry differs')
    if prior['status'] != 'passed' or prior['evaluation_sha256'] != cfg['old_evaluation']['sha256'] or prior['source_sha256'] != ind.digest(source_map) or old['source_sha256'] != ind.digest(source_map) or old['spec_sha256'] != design['spec_sha256']:
        raise ValueError('Original evaluation lacks exact passed verification')
    manifests = [load(x) for x in cfg['acquisition_manifests']]
    acquired = {f['name']: f for m in manifests for f in m['files']}
    if len(acquired) != sum(len(m['files']) for m in manifests):
        raise ValueError('Duplicate acquisition name')
    features, csvs = {}, {}
    for league, record in cfg['feature_artifacts'].items():
        f = features[league] = load(record)
        if old['feature_artifacts_sha256'][league] != record['sha256'] or f['source_files'] != source_map or f['source_sha256'] != ind.digest(source_map) or f['spec_sha256'] != old['spec_sha256']:
            raise ValueError('Feature artifact ancestry differs')
        maps = [m for m in cfg['cached_csv_maps'] if m['artifact'] == record['path']]
        if len(maps) != 1 or len(f['input_hashes']) != maps[0]['entries'] or ind.digest(f['input_hashes']) != maps[0]['canonical_sha256']:
            raise ValueError('Frozen raw input map differs')
        for path, expected in f['input_hashes'].items():
            receipt = acquired[Path(path).name]
            if path in csvs or receipt['sha256'] != expected or inputs.get(path) != expected:
                raise ValueError('Raw CSV lacks matching acquisition/feature/execution bindings')
            csvs[path] = {'path': path, 'sha256': expected, 'bytes': receipt['bytes']}
    if len(csvs) != cfg['cached_csv_total']:
        raise ValueError('Incomplete cached raw source inventory')
    archive = []
    for item in csvs.values():
        archive.extend(csv_metadata(bound(item, root, bindings), item))
    if len({r['match_id'] for r in archive}) != len(archive) or Counter((r['league'], r['season']) for r in archive) != Counter({(l, y): 380 for l in ind.ZONES for y in range(2017, 2026)}):
        raise ValueError('Incomplete 27-season raw metadata population')
    by_id = {r['match_id']: r for r in archive}
    def fixture(row):
        result = {k: row[k] for k in METADATA}
        result['probabilities'] = {k: row['probabilities'][k] for k in ('production_default_dc_auto', 'dc_equal')}
        raw = by_id[result['match_id']]
        for key in ('match_id', 'league', 'season', 'day', 'home', 'away', 'forecast_cutoff_utc', 'result_available_at'):
            if result[key] != raw[key]:
                raise ValueError('Frozen fixture metadata differs from independent raw record')
        for p in result['probabilities'].values():
            close(p, ind.normalize(p), atol=1e-10)
            if min(p) <= 0:
                raise ValueError('Old saved probabilities are not strictly positive')
        return result
    fixtures = [fixture(r) for r in old['predictions']]
    ids = [r['match_id'] for r in fixtures]
    if len(ids) != spec['population']['rows'] or len(set(ids)) != len(ids) or ind.digest(ids) != spec['population']['ordered_original_match_ids_sha256']:
        raise ValueError('Original complete ordered fixture population differs')
    original_by_id = {r['match_id']: r for r in fixtures}
    descriptors, inventory, seen = [], [], set()
    for league, feature in features.items():
        local = [fixture(r) for r in feature['rows']]
        for row in local:
            if row['league'] != league or original_by_id[row['match_id']] != row or row['match_id'] in seen:
                raise ValueError('Feature/evaluation row attribution differs')
            seen.add(row['match_id'])
        for i, fit in enumerate(feature['fits']):
            base, pop = fit['base'], fit['production_default']['populations']
            batch = [r for r in local if r['fit_id'] == base['fit_id']]
            if not batch or min(ind.utc(r['forecast_cutoff_utc']) for r in batch) != ind.utc(base['cutoff_utc']) or any(r['fit_cutoff_utc'] != base['cutoff_utc'] for r in batch):
                raise ValueError('Block cutoff and fixture clocks differ')
            item = {'league': league, 'feature_artifact': cfg['feature_artifacts'][league]['path'], 'feature_fit_index': i,
                    'fit_id': base['fit_id'], 'fit_cutoff_utc': base['cutoff_utc'], 'full_admitted_rows': len(base['fit_match_ids']),
                    'full_admitted_ids_sha256': base['fit_ids_sha256'], 'full_admitted_content_sha256': base['fit_content_sha256'],
                    'saved_dc_equal_parameters_sha256': ind.digest(base['dixon_coles']['dc_equal']),
                    'prefix_partition_sha256': ind.digest(pop), 'forecast_rows': len(batch),
                    'ordered_forecast_ids_sha256': ind.digest([r['match_id'] for r in batch]),
                    'prefix_fit_rows': pop['fit']['sample_size'], 'calibration_rows': pop['calibration']['sample_size'],
                    'historical_selection_rows': pop['selection']['sample_size'], 'historical_test_rows': pop['test']['sample_size']}
            inventory.append(item)
            descriptors.append({'block_id': f"{league}:{base['fit_id']}", 'league': league, 'fit_id': base['fit_id'],
                'fit_cutoff_utc': base['cutoff_utc'], 'fixture_metadata': batch, 'inventory': item,
                'saved_full_model': base['dixon_coles']['dc_equal'], 'expected_partitions': pop,
                'saved_full_lineage': {k: base[k] for k in ('fit_match_ids', 'fit_ids_sha256', 'fit_content_sha256', 'cutoff_utc', 'latest_result_available_at')},
                'input_sources': feature['input_hashes']})
    order = lambda r: (r['fit_cutoff_utc'], r['league'], r['fit_id'])
    descriptors.sort(key=order); inventory.sort(key=order)
    if len(descriptors) != spec['population']['fit_blocks'] or len({b['block_id'] for b in descriptors}) != len(descriptors) or seen != set(ids) or ind.digest(inventory) != spec['population']['block_inventory_binding']['complete_derived_inventory_sha256']:
        raise ValueError('Complete independent block inventory differs')
    return archive, fixtures, descriptors, inventory


def verify(out=OUT, *, root=ROOT):
    """Verify closed historical outputs; this function does not write or fit."""
    reject_operational_imports(); check_failures(out)
    out, root, bindings = Path(out), Path(root), {}
    def local(name):
        path = out/name
        record = {'path': str(path.relative_to(root)), 'sha256': sha(path), 'bytes': path.stat().st_size}
        return read(bound(record, root, bindings))
    design = local('design_lock.json')
    if design['historical_new_fits_before_lock'] != 0 or type(design['historical_new_fits_before_lock']) is not int or design['new_scores_before_lock'] is not False or design['candidate_state_branch'] != 'reuse_all_36_saved_full_history_states':
        raise ValueError('Design is not the frozen zero-score fixed restoration branch')
    sources, snapshots = design['sources'], design['reference_sources']
    if set(sources) != set(snapshots):
        raise ValueError('Complete immutable source snapshot is required')
    lane = root/'research/experiments/boundary_20260908/football_deployment_policy'
    required = {str(p.relative_to(root)) for pattern in ('*.py', '*.json', '*.md') for p in lane.glob(pattern)}
    required.update('research/experiments/boundary_20260908/football_deployment_policy/'+name for name in ('verify.py', 'independent.py', 'specification.json'))
    if not required.issubset(sources):
        raise ValueError('Current verifier/execution source inventory is incomplete')
    for original, expected in sources.items():
        if snapshots[original]['sha256'] != expected:
            raise ValueError('Reference snapshot differs from reviewed source')
        bound(snapshots[original], root, bindings)
        if original.startswith('research/experiments/boundary_20260908/football_deployment_policy/'):
            if sha(root/original) != expected:
                raise ValueError('Independent execution/verifier source changed')
    spec_key = 'research/experiments/boundary_20260908/football_deployment_policy/specification.json'
    spec = read(bound(snapshots[spec_key], root, bindings))
    for key in ('review', 'pre_fit_tests'):
        receipt = read(bound(design[key], root, bindings))
        if receipt['source_files'] != sources:
            raise ValueError('Pre-fit review/test source binding differs')
        if key == 'review' and receipt.get('approved_for_execution_lock') is not True:
            raise ValueError('No exact-source independent pre-fit approval')
        if key == 'pre_fit_tests' and (type(receipt['exit_code']) is not int or receipt['exit_code'] != 0):
            raise ValueError('Frozen synthetic tests did not pass')
    # Canonical implementation changes after this experiment remain audit-able
    # through immutable snapshots; data bytes must still match their originals.
    for path, expected in design['inputs'].items():
        if path in snapshots:
            if snapshots[path]['sha256'] != expected:
                raise ValueError('Inherited input/source snapshot conflict')
            bound(snapshots[path], root, bindings)
        else:
            bound({'path': path, 'sha256': expected}, root, bindings)
    data = local('data_lock.json'); closure = local('forecast_closure.json'); evaluation = local('evaluation.json')
    if data['design_lock_sha256'] != sha(out/'design_lock.json') or data['new_fits'] != 0 or type(data['new_fits']) is not int or data['new_scores'] is not False:
        raise ValueError('Metadata preparation has invalid source/fitting lineage')
    if closure['design_lock_sha256'] != sha(out/'design_lock.json') or closure['data_lock_sha256'] != sha(out/'data_lock.json') or closure['labels_attached'] is not False or type(closure['rows']) is not int or closure['rows'] != spec['population']['rows']:
        raise ValueError('Invalid complete target-free forecast closure')
    for key, name in [('forecast_closure', 'forecast_closure.json'), ('scored_rows', 'scored_rows.json'), ('design_lock', 'design_lock.json')]:
        if bound(evaluation[key], root, bindings) != out/name:
            raise ValueError('Evaluation binds a different stage artifact')
    if evaluation['production_activated'] is not False:
        raise ValueError('Research result falsely reports production activation')
    attempts = {name: local(name+'_attempt.json') for name in ('freeze', 'prepare', 'forecast', 'score')}
    clocks = [attempts['freeze']['started_at_utc'], design['locked_at_utc'], attempts['prepare']['started_at_utc'],
              data['closed_at_utc'], attempts['forecast']['started_at_utc'], closure['closed_at_utc'],
              attempts['score']['started_at_utc'], evaluation['created_at_utc']]
    if any(ind.utc(a) > ind.utc(b) for a, b in zip(clocks, clocks[1:])) or any(attempts[k]['stage'] != k for k in attempts):
        raise ValueError('Source/metadata/forecast/label closure chronology differs')
    for saved in (design, data, closure, evaluation):
        resource = saved['resources']
        if not 0 <= ind.finite(resource['elapsed_seconds']) <= spec['resources']['maximum_stage_wall_seconds'] or type(resource['peak_rss_bytes']) is not int or not 0 < resource['peak_rss_bytes'] <= spec['resources']['maximum_peak_rss_bytes']:
            raise ValueError('Execution resources violated the fixed bounds')
    closed_data = {k: read(bound(value, root, bindings)) for k, value in data['files'].items()}
    issued = read(bound(closure['forecasts'], root, bindings))
    scored = read(bound(evaluation['scored_rows'], root, bindings))
    blocks = [read(bound(item, root, bindings)) for item in closure['blocks']]
    # All output files are bound before interpreting raw training/scoring values.
    archive, fixtures, descriptors, inventory = reference_inputs(spec, root, design['inputs'], bindings)
    metadata = [{k: value for k, value in r.items() if k != 'payload'} for r in sorted(archive, key=lambda r: (r['league'], r['day'], r['match_id']))]
    if closed_data != {'archive_metadata': metadata, 'fixture_metadata': fixtures, 'blocks_metadata': descriptors, 'block_inventory': inventory} or data['rows'] != len(fixtures) or data['blocks'] != len(descriptors):
        raise ValueError('Closed metadata differs from independent reconstruction')
    if len(blocks) != len(descriptors) or [b['block_id'] for b in blocks] != [b['block_id'] for b in descriptors]:
        raise ValueError('Block output population/order differs')
    outputs, maximum, fit_counts = {}, 0., {key: 0 for key in ('prefix_goal_fits', 'full_history_goal_fits', 'reference_calibration_policy_calls', 'candidate_learned_calibration_fits', 'candidate_parity_goal_fits')}
    for block, descriptor in zip(blocks, descriptors):
        if block['design_lock_sha256'] != sha(out/'design_lock.json') or any(block[k] != descriptor[k] for k in ('block_id', 'league', 'fit_id', 'fit_cutoff_utc', 'fixture_metadata', 'input_sources', 'expected_partitions')):
            raise ValueError('Closed block attribution/source differs')
        admitted = ind.history_membership(archive, descriptor['fit_cutoff_utc'], descriptor['league'])
        expected_meta = [{k: x for k, x in r.items() if k != 'payload'} for r in admitted]
        if block['history_metadata'] != expected_meta:
            raise ValueError('Block admitted raw-source metadata differs')
        expected_lineage = {**descriptor['saved_full_lineage'], 'match_records_sha256': ind.digest([match_record(r) for r in admitted])}
        if block['saved_full_lineage'] != expected_lineage:
            raise ValueError('Captured restored-source lineage differs')
        result = audit_block(block, descriptor, archive)
        maximum = max(maximum, result['maximum_output_error'])
        for key in fit_counts:
            fit_counts[key] += result['fit_counts'][key]
        for fixture, reference, candidate in zip(descriptor['fixture_metadata'], block['reference']['fixtures'], block['candidate']['fixtures']):
            if fixture['match_id'] in outputs:
                raise ValueError('Repeated forecast across blocks')
            outputs[fixture['match_id']] = {**fixture, 'block_id': descriptor['block_id'], 'outputs': {ind.REFERENCE: reference, ind.CANDIDATE: candidate}}
    expected_issued = [outputs[r['match_id']] for r in fixtures]
    if issued != expected_issued or len(issued) != len(scored) or closure['fit_counts'] != fit_counts:
        raise ValueError('Complete forecast ledger or total fit counts differ')
    expected_counts = {**{key: 0 for key in fit_counts}, 'prefix_goal_fits': len(descriptors), 'reference_calibration_policy_calls': len(descriptors)}
    if fit_counts != expected_counts:
        raise ValueError('Frozen all-state restoration/optimizer budget changed')
    raw = {r['match_id']: r for r in archive}
    reconstructed_scored = []
    for row in expected_issued:
        target, label = goals(raw[row['match_id']])
        reconstructed_scored.append({**row, 'goals': target, 'label': label})
    if scored != reconstructed_scored:
        raise ValueError('External label attachment changed the forecast or raw target')
    independent_report = ind.evaluate(reconstructed_scored, spec)
    audit_report(evaluation['report'], independent_report)
    check_failures(out)
    if any(sha(root/path) != expected for path, expected in bindings.items()):
        raise ValueError('A bound input/output changed during independent verification')
    return {'status': 'passed', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'evaluation_sha256': sha(out/'evaluation.json'), 'design_lock_sha256': sha(out/'design_lock.json'),
            'forecast_closure_sha256': sha(out/'forecast_closure.json'), 'bindings': bindings,
            'reference_source_map': sources, 'rows': len(scored), 'blocks': len(blocks),
            'raw_metadata_rows': len(archive), 'maximum_saved_output_replay_error': maximum, 'fit_counts': fit_counts,
            'independent_checks': independent_report['checks'], 'passes_all_gates': independent_report['passes_all_gates'],
            'production_activated': False, 'refits': 0,
            'scope': 'Independent raw metadata/history/partitions, saved DC/calibration/finite matrix/caller outputs, labels, all metrics and paired gates; no optimizer or operational imports.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(); out = args.out
    def save(name, value):
        with (out/name).open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    check_failures(out)
    save('verification_attempt.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'design_lock_sha256': sha(out/'design_lock.json')})
    try:
        result = verify(out)
        save('verification.json', result)
    except BaseException as exc:
        save('verification_failure.json', {'failed_at_utc': datetime.now(timezone.utc).isoformat(), 'exception_type': type(exc).__name__, 'message': str(exc), 'advancement_allowed': False})
        raise
    print(json.dumps({'status': result['status'], 'sha256': sha(out/'verification.json'), 'passes_all_gates': result['passes_all_gates']}), flush=True)


if __name__ == '__main__':
    main()
