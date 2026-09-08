"""Independent verifier contracts, using only synthetic temporary artifacts."""
from copy import deepcopy
from datetime import date, datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from . import independent as ind
from . import verify as v
from .test_independent import goal_state, identity, scoring_fixture


@pytest.mark.parametrize('nonfinite', [False, True])
def test_independent_full_report_matches_separate_operational_math(nonfinite):
    from . import scoring
    rows, spec = scoring_fixture()
    if nonfinite:
        rows[0]['goals'] = [99, 0]
    v.audit_report(scoring.evaluate(rows, spec), ind.evaluate(rows, spec))


def test_independent_same_original_block_bootstrap_semantics():
    from research.experiments.boundary_20260908.football_past_market import independent as old
    rows, spec = scoring_fixture()
    for i, row in enumerate(rows):
        row['probabilities'] = {key: row['outputs'][key]['hda'] for key in ind.POLICIES}
        row['day'] = (date(row['season'], 8, 1)+timedelta(days=17*(i % 3))).isoformat()
    for days in (1, 7, 28):
        original = old.paired_uncertainty(rows, ind.CANDIDATE, ind.REFERENCE, days, spec)
        delta = [-np.log(r['probabilities'][ind.CANDIDATE][0])+np.log(r['probabilities'][ind.REFERENCE][0]) for r in rows]
        ours = ind.paired_losses(rows, delta, days, spec)
        for key in ('difference_candidate_minus_reference', 'percentile_95_interval', 'bootstrap_fraction_difference_below_zero'):
            np.testing.assert_allclose(ours[key], original[key], atol=1e-14, rtol=0)
        assert ours['blocks_by_season'] == original['blocks_by_season']


@pytest.mark.parametrize('value', [1, 'true', False])
def test_report_cannot_smuggle_truthy_gate(value):
    rows, spec = scoring_fixture()
    expected = ind.evaluate(rows, spec); actual = deepcopy(expected)
    actual['passes_all_gates'] = value
    with pytest.raises(ValueError):
        v.audit_report(actual, expected)


def test_independent_gate_sign_is_not_waived_by_numeric_tolerance():
    rows, spec = scoring_fixture()
    expected = ind.evaluate(rows, spec); actual = deepcopy(expected)
    expected['paired']['28']['joint_score']['percentile_95_interval'][1] = 1e-15
    expected['checks']['joint_score_paired28_upper_nonpositive'] = False
    expected['passes_all_gates'] = False
    actual['paired']['28']['joint_score']['percentile_95_interval'][1] = -1e-15
    with pytest.raises(ValueError):
        v.audit_report(actual, expected)


def test_metadata_csv_native_parity_without_accessing_poisoned_scores(tmp_path):
    from . import data
    path = tmp_path/'E0_2023_2024.csv'
    path.write_text('Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,31/03/2024,A,B,POISON,POISON,POISON\nE0,01/04/2024,B,A,POISON,POISON,POISON\n')
    binding = {'path': str(path), 'sha256': v.sha(path)}
    actual = v.csv_metadata(path, binding, expected_rows=2, expected_teams=2)
    expected, _ = data.read_csv_metadata(path, binding, expected_rows=2, expected_teams=2)
    assert [{k: x for k, x in r.items() if k != 'payload'} for r in actual] == [r.metadata() for r in expected]
    assert actual[0]['result_available_at'] == '2024-03-31T23:00:00'
    with pytest.raises(ValueError):
        v.goals(actual[0])


def synthetic_block():
    archive = []
    for i in range(220):
        d = date(2023, 1, 1)+timedelta(days=i)
        archive.append({'match_id': str(i), 'league': 'E0', 'season': 2023, 'day': d.isoformat(),
                        'home': 'A', 'away': 'B', 'forecast_cutoff_utc': v.legacy(ind.local_midnight(d, 'E0')),
                        'result_available_at': v.legacy(ind.local_midnight(d+timedelta(days=1), 'E0')),
                        'payload': {'FTHG': '1', 'FTAG': '0', 'FTR': 'H'}})
    cutoff = v.legacy(ind.local_midnight(date(2023, 9, 1), 'E0'))
    history = [v.match_record(r) for r in archive]
    partitions = v.partition_metadata(history)
    ids = [r['match_id'] for r in history]
    state = goal_state(); state['diagnostics'] = {'status': 'fitted', 'converged': True, 'fit_sample_size': len(ids), 'half_life_days': None}
    fixture = {'match_id': 'fixture', 'league': 'E0', 'season': 2023, 'day': '2023-09-01',
               'home': 'A', 'away': 'B', 'forecast_cutoff_utc': cutoff, 'result_available_at': '2023-09-01T23:00:00',
               'fit_id': 'fit', 'fit_cutoff_utc': cutoff, 'probabilities': {}}
    compact = [[r['match_id'], r['day'], 1, 0] for r in archive]
    full_lineage = {'fit_match_ids': ids, 'fit_ids_sha256': ind.digest(ids), 'fit_content_sha256': ind.digest(compact),
                    'match_records_sha256': ind.digest(history), 'cutoff_utc': cutoff,
                    'latest_result_available_at': history[-1]['result_available_at']}
    descriptor = {'league': 'E0', 'fit_cutoff_utc': cutoff, 'saved_full_lineage': full_lineage,
                  'expected_partitions': partitions, 'fixture_metadata': [fixture], 'saved_full_model': deepcopy(state)}
    lineage = {**full_lineage, 'fit_sample_size': len(ids), 'weight_policy': 'equal',
               'half_life_days': None, 'calibration_policy': 'off'}
    candidate = {'policy': ind.CANDIDATE, 'model_state': deepcopy(state), 'calibrator_state': identity(),
                 'lineage': lineage, 'restoration_branch': 'validated_saved_full_history_state',
                 'new_goal_model_fits': 0, 'assessment_metrics_computed': False}
    reference = {'policy': ind.REFERENCE, 'model_state': deepcopy(state), 'calibrator_state': identity(),
                 'lineage': {'assessment_partitions': partitions, 'goal_fit_ids': partitions['fit']['match_ids'],
                             'calibration_ids': partitions['calibration']['match_ids'], 'cutoff_utc': cutoff}}
    reference['model_state']['diagnostics']['fit_sample_size'] = len(partitions['fit']['match_ids'])
    for policy, old in [(reference, 'production_default_dc_auto'), (candidate, 'dc_equal')]:
        output = ind.replay_output(policy['model_state'], identity(), 'A', 'B')
        output.update(match_id='fixture', date=cutoff, league='epl', season=2023, home_team_id='A', away_team_id='B')
        fixture['probabilities'][old] = output['hda']
        policy['fixtures'] = [output]
        canonical = {'match_id': 'fixture', 'model_used': 'dixon', 'gbdt_enabled': 'false',
                     'predicted_scoreline': f"{output['mode'][0]}-{output['mode'][1]}", 'scoreline_prob': f"{output['mode'][2]:.12f}"}
        for prefix, name in [('', 'hda'), ('raw_', 'raw_hda')]:
            canonical.update({prefix+k: f'{p:.12f}' for k, p in zip(('home_win_prob', 'draw_prob', 'away_win_prob'), output[name])})
        for k, value in zip(('expected_home_goals', 'expected_away_goals'), output['expected_goals']):
            canonical[k] = f'{value:.6f}'
        canonical.update(dixon_base_lambda_home_goals=f"{output['lambda_home']:.6f}", dixon_base_lambda_away_goals=f"{output['lambda_away']:.6f}")
        policy['canonical_rows'] = [canonical]
        policy['diagnostics'] = {'protocol': deepcopy(partitions), 'fixture_distributions': {'fixture': {
            'score_probability_matrix': output['matrix'], 'outcome_probabilities': output['hda'],
            'expected_goals': output['expected_goals'], 'base_omitted_probability_mass': output['omitted_probability_mass'],
            'reconciled_tail_error_bound': output['tail_probability_bound']}}}
    record = {'reference': reference, 'candidate': candidate,
              'parity': {k: True for k in ('reference_actual_caller', 'reference_serialized_replay', 'candidate_state_injected_caller_output', 'old_reference_hda_checked', 'old_candidate_hda_checked')},
              'fit_counts': {'prefix_goal_fits': 1, 'full_history_goal_fits': 0, 'reference_calibration_policy_calls': 1,
                             'candidate_learned_calibration_fits': 0, 'candidate_parity_goal_fits': 0}}
    return record, descriptor, archive


def test_complete_synthetic_saved_block_replays_without_fit():
    result = v.audit_block(*synthetic_block())
    assert result['fixtures'] == 1 and result['maximum_output_error'] == 0


@pytest.mark.parametrize('mutation', [lambda r, d, a: r['parity'].update(reference_actual_caller=1),
                                     lambda r, d, a: r['candidate'].update(assessment_metrics_computed=True),
                                     lambda r, d, a: r['candidate']['lineage']['fit_match_ids'].append('fixture'),
                                     lambda r, d, a: r['candidate']['fixtures'][0]['matrix'][0].__setitem__(0, .1),
                                     lambda r, d, a: r['candidate']['canonical_rows'][0].update(home_win_prob='.9'),
                                     lambda r, d, a: r['candidate']['model_state']['attack'].update(A=.2),
                                     lambda r, d, a: r['fit_counts'].update(prefix_goal_fits=True),
                                     lambda r, d, a: a[0]['payload'].update(FTHG='2'),
                                     lambda r, d, a: d['expected_partitions']['fit']['match_ids'].reverse()])
def test_material_saved_block_tampering_fails(mutation):
    record, descriptor, archive = synthetic_block(); mutation(record, descriptor, archive)
    with pytest.raises(ValueError):
        v.audit_block(record, descriptor, archive)


def test_bound_bytes_and_failure_guard(tmp_path):
    path = tmp_path/'closed.json'; path.write_text('{}')
    binding = {'path': 'closed.json', 'sha256': v.sha(path), 'bytes': 2}
    assert v.bound(binding, tmp_path, {}) == path
    path.write_text('{ }')
    with pytest.raises(ValueError):
        v.bound(binding, tmp_path, {})
    (tmp_path/'prepare_failure.json').write_text('{}')
    with pytest.raises(ValueError):
        v.check_failures(tmp_path)


def closed_stage(tmp_path, monkeypatch):
    """Full new-stage schema; only independently tested raw ancestry is stubbed."""
    root = tmp_path; out = root/'artifacts/closed'; out.mkdir(parents=True)
    lane = root/'research/experiments/boundary_20260908/football_deployment_policy'; lane.mkdir(parents=True)
    record, descriptor, archive = synthetic_block()
    fixture = descriptor['fixture_metadata'][0]
    raw_target = {k: fixture[k] for k in ('match_id', 'league', 'season', 'day', 'home', 'away', 'forecast_cutoff_utc', 'result_available_at')}
    raw_target['payload'] = {'FTHG': '1', 'FTAG': '0', 'FTR': 'H'}
    archive.append(raw_target)
    descriptor.update(block_id='E0:fit', fit_id='fit', input_sources={}, inventory={})
    fixtures, descriptors, inventory = [fixture], [descriptor], [{}]
    _, spec = scoring_fixture()
    spec['population'].update(rows=1, countries=['E0'], season_start_years=[2023], rows_each_country_season=1,
                              ordered_original_match_ids_sha256=ind.digest(['fixture']), fit_blocks=1)
    spec['resources'] = {'maximum_stage_wall_seconds': 3600, 'maximum_peak_rss_bytes': 2**31}
    def save(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, sort_keys=True, allow_nan=False))
        return {'path': str(path.relative_to(root)), 'sha256': v.sha(path), 'bytes': path.stat().st_size}
    save(lane/'specification.json', spec)
    for name in ('verify.py', 'independent.py'):
        (lane/name).write_bytes((Path(v.__file__).parent/name).read_bytes())
    sources = {str(p.relative_to(root)): v.sha(p) for p in lane.iterdir()}
    snapshots = {}
    for path, expected in sources.items():
        destination = out/'reference_sources'/path; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root/path).read_bytes())
        snapshots[path] = {'path': str(destination.relative_to(root)), 'sha256': expected, 'bytes': destination.stat().st_size}
    resource = {'elapsed_seconds': 1., 'peak_rss_bytes': 100000}
    review = save(out/'review.json', {'source_files': sources, 'approved_for_execution_lock': True})
    tests = save(out/'pre_fit_tests.json', {'source_files': sources, 'exit_code': 0})
    design = {'sources': sources, 'reference_sources': snapshots, 'inputs': {}, 'runtime': {}, 'review': review,
              'pre_fit_tests': tests, 'historical_new_fits_before_lock': 0, 'new_scores_before_lock': False,
              'candidate_state_branch': 'reuse_all_36_saved_full_history_states', 'locked_at_utc': '2026-09-08T01:00:01Z', 'resources': resource}
    design_record = save(out/'design_lock.json', design)
    metas = [{k: x for k, x in r.items() if k != 'payload'} for r in sorted(archive, key=lambda r: (r['league'], r['day'], r['match_id']))]
    prepared = {'archive_metadata': metas, 'fixture_metadata': fixtures, 'blocks_metadata': descriptors, 'block_inventory': inventory}
    data_record = save(out/'data_lock.json', {'design_lock_sha256': design_record['sha256'], 'new_fits': 0, 'new_scores': False,
        'closed_at_utc': '2026-09-08T01:00:03Z', 'rows': 1, 'blocks': 1, 'resources': resource,
        'files': {k: save(out/(k+'.json'), value) for k, value in prepared.items()}})
    record.update({k: descriptor[k] for k in ('block_id', 'league', 'fit_id', 'fit_cutoff_utc', 'fixture_metadata', 'input_sources', 'expected_partitions', 'saved_full_lineage')})
    record['design_lock_sha256'] = design_record['sha256']
    record['history_metadata'] = [{k: x for k, x in r.items() if k != 'payload'} for r in archive[:-1]]
    block_record = save(out/'blocks/00.json', record)
    issued = [{**fixture, 'block_id': 'E0:fit', 'outputs': {ind.REFERENCE: record['reference']['fixtures'][0], ind.CANDIDATE: record['candidate']['fixtures'][0]}}]
    closure_record = save(out/'forecast_closure.json', {'design_lock_sha256': design_record['sha256'],
        'data_lock_sha256': data_record['sha256'], 'labels_attached': False, 'rows': 1, 'resources': resource,
        'closed_at_utc': '2026-09-08T01:00:05Z', 'forecasts': save(out/'forecasts.json', issued),
        'blocks': [block_record], 'fit_counts': record['fit_counts']})
    scored = [{**issued[0], 'goals': [1, 0], 'label': 0}]
    evaluation = {'forecast_closure': closure_record, 'scored_rows': save(out/'scored_rows.json', scored),
                  'design_lock': design_record, 'production_activated': False, 'created_at_utc': '2026-09-08T01:00:07Z',
                  'resources': resource, 'report': ind.evaluate(scored, spec)}
    save(out/'evaluation.json', evaluation)
    for stage, second in [('freeze', 0), ('prepare', 2), ('forecast', 4), ('score', 6)]:
        save(out/(stage+'_attempt.json'), {'stage': stage, 'started_at_utc': f'2026-09-08T01:00:0{second}Z'})
    monkeypatch.setattr(v, 'reject_operational_imports', lambda: None)
    monkeypatch.setattr(v, 'reference_inputs', lambda *_: (archive, fixtures, descriptors, inventory))
    return root, out


def test_entire_synthetic_stage_closes_and_verifies_without_refitting(tmp_path, monkeypatch):
    root, out = closed_stage(tmp_path, monkeypatch)
    result = v.verify(out, root=root)
    assert result['status'] == 'passed' and result['rows'] == 1 and result['blocks'] == 1
    assert result['refits'] == 0 and result['passes_all_gates'] is False


@pytest.mark.parametrize('kind', ['label', 'closure_clock', 'surviving_failure'])
def test_full_stage_rejects_target_timing_and_terminal_failure(tmp_path, monkeypatch, kind):
    root, out = closed_stage(tmp_path, monkeypatch)
    if kind == 'label':
        rows = v.read(out/'scored_rows.json'); rows[0]['goals'] = [2, 0]
        (out/'scored_rows.json').write_text(json.dumps(rows))
    elif kind == 'closure_clock':
        attempt = v.read(out/'score_attempt.json'); attempt['started_at_utc'] = '2026-09-08T00:00:00Z'
        (out/'score_attempt.json').write_text(json.dumps(attempt))
    else:
        (out/'forecast_failure.json').write_text('{}')
    with pytest.raises(ValueError):
        v.verify(out, root=root)
