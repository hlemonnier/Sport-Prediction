"""Independent verification contracts: synthetic rows, no optimizer or archive."""
from copy import deepcopy
from datetime import date, timedelta
import json
import math

import numpy as np
import pytest

from . import verify as v


def raw(i=0, *, day='2020-01-01', league='E0', label=0, odds=True):
    goals = [(2, 0), (1, 1), (0, 2)][label]
    payload = {'AvgH': '2', 'AvgD': '3', 'AvgA': '4', 'FTR': 'HDA'[label],
               'FTHG': str(goals[0]), 'FTAG': str(goals[1])}
    if not odds:
        payload['AvgH'] = ''
    issue = v.ind.availability(date.fromisoformat(day)-timedelta(days=1), league, 1)[0]
    return {'match_id': f'{league}:2020:{i}', 'league': league, 'season': 2020,
            'home': f'h{i}', 'away': f'a{i}', 'day': day, 'payload': payload,
            'forecast_cutoff_utc': issue.isoformat()}


class Poison(dict):
    def __getitem__(self, key):
        raise AssertionError('Future numeric payload was touched')
    def get(self, key, default=None):
        raise AssertionError('Future numeric payload was touched')


def test_result_update_has_paper_k14_orientation_and_conservation():
    expected = 1/(1+10**(-80/400))
    records = [{'match_id': str(i), 'league': 'E0', 'home': f'h{i}', 'away': f'a{i}', 'outcome': i} for i in range(3)]
    ratings = v.result_atomic_update({}, records)
    for i, observed in enumerate((1., .5, 0.)):
        assert ratings[('E0', f'h{i}')] == pytest.approx(1000+14*(observed-expected), abs=1e-12)
        assert ratings[('E0', f'a{i}')] == pytest.approx(1000-14*(observed-expected), abs=1e-12)
    assert math.fsum(ratings.values()) == pytest.approx(6000, abs=1e-12)


def test_equal_clock_update_is_atomic_and_input_order_invariant():
    rows = [{'match_id': 'a', 'league': 'E0', 'home': 'a', 'away': 'b', 'outcome': 0},
            {'match_id': 'b', 'league': 'E0', 'home': 'a', 'away': 'c', 'outcome': 2}]
    got = v.result_atomic_update({}, rows)
    assert got == v.result_atomic_update({}, list(reversed(rows)))
    sequential = v.result_atomic_update(v.result_atomic_update({}, rows[:1]), rows[1:])
    assert abs(got[('E0', 'a')]-sequential[('E0', 'a')]) > .01
    assert v.result_atomic_update({('I1', 'a'): 1250}, rows)[('I1', 'a')] == 1250


@pytest.mark.parametrize('label', [True, False, 1., -1, 3, None, 'H'])
def test_invalid_result_outcomes_fail(label):
    with pytest.raises(ValueError):
        v.result_atomic_update({}, [{'match_id': 'a', 'league': 'E0', 'home': 'a', 'away': 'b', 'outcome': label}])


def test_duplicate_update_and_nonfinite_state_rejected():
    row = {'match_id': 'a', 'league': 'E0', 'home': 'a', 'away': 'b', 'outcome': 0}
    with pytest.raises(ValueError, match='unique'):
        v.result_atomic_update({}, [row, row])
    with pytest.raises(ValueError):
        v.result_atomic_update({('E0', 'a'): float('nan')}, [row])
    with pytest.raises(ValueError):
        v.result_atomic_update({('E0', 'a'): 1.7e308, ('E0', 'b'): -1.7e308}, [row])


@pytest.mark.parametrize('kind', v.KINDS)
@pytest.mark.parametrize('delay', [1, 7])
def test_strict_clock_prefix_and_future_poison(kind, delay):
    row = raw(); clock = v.ind.availability(row['day'], row['league'], delay)[1]
    before = v.replay_states([row], [clock], delay, kind)
    assert before[clock] == {}
    poisoned = {**row, 'payload': Poison()}
    assert v.replay_states([poisoned], [clock], delay, kind) == before
    assert v.replay_states([poisoned], [clock-timedelta(microseconds=1)], delay, kind)[clock-timedelta(microseconds=1)] == {}
    after = v.replay_states([row], [clock+timedelta(microseconds=1)], delay, kind)
    assert len(after[clock+timedelta(microseconds=1)]) == 2


def test_missing_quotes_skip_corrupt_goals_in_both_processes():
    row = raw(odds=False); row['payload']['FTR'] = 'BAD'; row['payload']['FTHG'] = 'poison'
    cutoff = '2020-02-01T00:00:00Z'
    assert v.replay_states([row], [cutoff], 1, 'odds')[v.ind.utc(cutoff)] == {}
    assert v.replay_states([row], [cutoff], 1, 'result')[v.ind.utc(cutoff)] == {}
    row['payload']['AvgH'] = '2'
    assert v.replay_states([row], [cutoff], 1, 'odds')[v.ind.utc(cutoff)]
    with pytest.raises(ValueError):
        v.replay_states([row], [cutoff], 1, 'result')


def test_ratings_do_not_expire_with_rolling_support():
    row = raw(day='2019-01-01');row['season']=2019
    cutoff = '2024-01-10T00:00:00Z'
    assert v.support_teams([row], cutoff, 1) == set()
    assert len(v.replay_states([row], [cutoff], 1, 'result')[v.ind.utc(cutoff)]) == 2


@pytest.mark.parametrize('league', ['E0', 'I1', 'SP1'])
@pytest.mark.parametrize('day', ['2024-03-29', '2024-10-25'])
def test_calendar_delay_and_matched_support_at_dst(league, day):
    row = raw(day=day, league=league); a0, primary = v.ind.availability(day, league, 1)
    _, slower = v.ind.availability(day, league, 7)
    assert (primary-a0).total_seconds() in (23*3600, 24*3600, 25*3600)
    assert v.support_teams([row], slower, 7) == set()
    assert len(v.support_teams([row], slower+timedelta(microseconds=1), 7)) == 2


def readout_fixture():
    rows = [raw(i, day=f'2020-02-{i+1:02}', label=i%3) for i in range(6)]
    theta = [0., -math.log(2), 2*math.log(2)]
    payload = {'kind': 'ordered_logit_v1', 'beta': theta[0], 't1': theta[1], 'gap': theta[2],
               't2': theta[1]+theta[2], 'theta': theta, 'floor': 1e-12}
    audit = v.ind.ordered_objective_audit(payload, [0.]*6, [i%3 for i in range(6)])
    payload['diagnostics'] = {**audit, 'solver_success': True}
    payload['training_digest'] = v.digest({'x': [0.]*6, 'y': [i%3 for i in range(6)]})
    training_rows = [{**r, 'x': 0., 'y': i%3,
                      'quote_available_at_utc': v.ind.availability(r['day'], r['league'], 1)[1].isoformat(),
                      'rating_provenance': {'cutoff_utc': r['forecast_cutoff_utc'], 'extra_days': 1}} for i,r in enumerate(rows)]
    record = {'training': {'calibration_cutoff_utc': '2022-08-01T00:00:00Z', 'rows': training_rows,
                           'extra_days': 1, 'season_start_years': [2019,2020,2021]},
              'x': [0.]*6, 'y': [i%3 for i in range(6)], 'model': payload}
    return record, rows


@pytest.mark.parametrize('kind', v.KINDS)
def test_prequential_readout_optimum_no_optimizer(kind):
    record, rows = readout_fixture()
    result = v.audit_readout(record, rows, 1, kind, '2022-08-01T00:00:00Z')
    assert result['rows'] == 6 and result['fits'] == 0
    assert result['objective'] == pytest.approx(math.log(3), abs=1e-14)
    assert result['kkt_residual'] < 1e-12


@pytest.mark.parametrize('tamper', ['x', 'y', 'ids', 'cutoff', 'gradient', 'success'])
def test_readout_metadata_features_and_optimum_fail_closed(tamper):
    record, rows = readout_fixture(); record = deepcopy(record)
    if tamper == 'x':record['x'][0] = 1.
    if tamper == 'y':record['y'][0] = 2
    if tamper == 'ids':record['training']['rows'].reverse()
    if tamper == 'cutoff':record['training']['calibration_cutoff_utc'] = '2023-08-01T00:00:00Z'
    if tamper == 'gradient':record['model']['diagnostics']['gradient'][0] += .01
    if tamper == 'success':record['model']['diagnostics']['solver_success'] = 1
    with pytest.raises(ValueError):
        v.audit_readout(record, rows, 1, 'result', '2022-08-01T00:00:00Z')


def test_readout_metadata_admission_never_opens_excluded_payloads():
    row = raw(); out = {**row, 'season': 2022, 'payload': Poison()}
    future = {**row, 'day': '2025-01-01', 'payload': Poison()}
    assert v.calibration_population([out, future], '2022-08-01T00:00:00Z', 1) == []


def test_terminal_failure_guard(tmp_path):
    (tmp_path/'selection_delay1_failure.json').write_text('{}')
    with pytest.raises(ValueError, match='terminal'):
        v.check_failures(tmp_path)


@pytest.mark.parametrize('delay', [True, 0, 2, 1., '1'])
def test_invalid_delay_on_empty_replay(delay):
    with pytest.raises(ValueError):
        v.replay_states([], [], delay, 'result')


@pytest.mark.parametrize('kind', v.KINDS)
@pytest.mark.parametrize('delay', [1, 7])
def test_actual_cursor_matches_independent_replay_all_prefixes(kind, delay):
    from . import elo
    from ..football_past_market import data
    records = [raw(i, day=f'2020-03-{20+i//2:02}', label=i%3) for i in range(12)]
    for i, r in enumerate(records):
        r.update(home=f't{i%3}', away=f't{(i+1)%3}')
    sources = tuple(data.RawFixture(r['match_id'], r['league'], r['season'], r['home'], r['away'],
                    date.fromisoformat(r['day']), r['payload'], 'synthetic.csv', 'a'*64, 2, 'b'*64) for r in records)
    cursor = elo.EloCursor(data.Archive(sources), delay, kind)
    clocks = sorted({v.ind.availability(r['day'], r['league'], delay)[1]+timedelta(microseconds=n) for r in records for n in (0,1)})
    states = v.replay_states(records, clocks, delay, kind)
    fixtures = [records[0], records[-1]]
    for clock in clocks:
        queried = cursor.query(fixtures, clock)
        assert set(cursor.ratings) == set(states[clock])
        for key, value in cursor.ratings.items():
            assert value == pytest.approx(states[clock][key], abs=2e-11, rel=0)
        v.audit_query(queried, fixtures, clock, delay, kind, states[clock])


def scored_report(phase='selection'):
    from . import run
    spec = deepcopy(run.read(run.SPEC));spec['uncertainty']['resamples']=32
    rows=[]
    for league in (('E0',) if phase=='selection' else ('E0','I1','SP1')):
        for year in ((2022,2023) if phase=='selection' else (2024,2025)):
            for i in range(6):
                p=[.1]*3;p[i%3]=.8
                q=[.2]*3;q[i%3]=.6
                rows.append({'match_id':f'{league}:{year}:{i}', 'league':league,'season':year,
                             'day':f'{year}-08-{i+1:02}', 'label':i%3,
                             'probabilities':{name:list(p if name==v.CANDIDATE else q) for name in [v.CANDIDATE,*spec[phase]['references']]}})
    spec[phase]['rows']=len(rows);spec[phase]['ordered_original_match_ids_sha256']=v.digest([r['match_id'] for r in rows])
    return rows,spec,run.make_report(rows,spec,phase,v.CANDIDATE)


@pytest.mark.parametrize('phase',['selection','transfer'])
def test_all_fixed_report_metrics_and_gates_match_runner(phase):
    rows,spec,report=scored_report(phase)
    v.audit_report(rows,report,spec,phase)
    assert report['passes_all_gates'] is True
    assert len(report['comparisons'][v.CANDIDATE])==(8 if phase=='selection' else 5)


@pytest.mark.parametrize('tamper',['drop_ref','change_candidate','drop_row','pass_int','promotion_int'])
def test_fixed_report_scope_cannot_be_bypassed(tamper):
    rows,spec,report=scored_report()
    if tamper=='drop_ref':
        spec['selection']['references'].remove('outcome_control_shot50')
        report['comparisons'][v.CANDIDATE].pop('outcome_control_shot50')
    if tamper=='change_candidate':report['selected_candidate']='elo_result_k14'
    if tamper=='drop_row':rows.pop()
    if tamper=='pass_int':report['passes_all_gates']=1
    if tamper=='promotion_int':report['promotion']=0
    with pytest.raises(ValueError):v.audit_report(rows,report,spec,'selection')


def test_independent_interval_sign_controls_gate(monkeypatch):
    rows,spec,report=scored_report()
    ref=spec['selection']['references'][0]
    report['comparisons'][v.CANDIDATE][ref]['28']['percentile_95_interval'][1]=-1e-12
    original=v.ind.paired_uncertainty
    def replay(rows,candidate,reference,days,spec):
        value=original(rows,candidate,reference,days,spec)
        if reference==ref and days==28:value['percentile_95_interval'][1]=1e-12
        return value
    monkeypatch.setattr(v.ind,'paired_uncertainty',replay)
    with pytest.raises(ValueError):v.audit_report(rows,report,spec,'selection')


@pytest.mark.parametrize('phase,delay,expected',[
    ('selection',1,[]),('selection',7,['selection_delay1']),
    ('transfer',1,['selection_delay1','selection_delay7']),
    ('transfer',7,['selection_delay1','selection_delay7','transfer_delay1'])])
def test_frozen_predecessor_order(phase,delay,expected):
    assert v.required_predecessors(phase,delay)==expected


def test_historical_verifier_rejects_operational_imports():
    from . import run  # Pure import only; the standalone verifier must reject it.
    assert run.CANDIDATE==v.CANDIDATE
    with pytest.raises(ValueError,match='operational'):
        v.reject_operational_imports()


def synthetic_execution(tmp_path, monkeypatch):
    """A complete closed primary artifact graph, with no fitted estimator.

    Parent CSV parsing/reference ancestry has its own frozen tests; this fixture
    substitutes those two readers to focus on the new end-to-end stage schema.
    All new verification logic and independent state/optimum/scoring run.
    """
    from . import run
    spec = deepcopy(run.read(run.SPEC));spec['uncertainty']['resamples']=32
    out=tmp_path/'out';source=tmp_path/'source';out.mkdir();source.mkdir()
    def save(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))
        return {'path':name,'sha256':v.sha(path),'bytes':path.stat().st_size}
    old_readout,training=readout_fixture();old_readout['closed_at_utc']='2026-09-08T00:00:00Z'
    uniform=[1/3]*3;refs=[];future=[]
    base_names=spec['selection']['references'][:5]
    for i in range(6):
        r=raw(i,day=f'2022-08-{i+1:02}',label=i%3)
        r.update(match_id=f'E0:2022:new{i}',season=2022,home=f'newh{i}',away=f'newa{i}')
        future.append(r)
        refs.append({k:r[k] for k in ('match_id','league','season','home','away','day','forecast_cutoff_utc')})
        refs[-1]['probabilities']={name:list(uniform) for name in base_names}
    previous=deepcopy(refs)
    for row in previous:
        row['support']=False
        row['probabilities'].update({name:list(uniform) for name in (v.CANDIDATE,'outcome_control','outcome_control_shot50')})
    prepared=deepcopy(refs)
    for row in prepared:
        row['probabilities'].update({name:list(uniform) for name in ('outcome_control','outcome_control_shot50')})
    old_readout['training']['calibration_cutoff_utc']=refs[0]['forecast_cutoff_utc']
    spec['selection']['rows']=6;spec['selection']['ordered_original_match_ids_sha256']=v.digest([r['match_id'] for r in refs])
    spec['support']['expected_selection_supported_primary']=0
    inputs={'parent_specification':save('parent/spec.json',{}),
            'parent_design_lock':save('parent/design.json',{'sources':{},'inputs':{}}),
            'parent_primary_result':save('parent/result.json',{'failed':True}),
            'parent_primary_issuance_lock':save('parent/closure.json',{'old':True}),
            'primary_forecasts':save('parent/forecasts.json',previous),
            'primary_market_readout':save('parent/readout.json',old_readout),
            'references_selection':save('parent/selection_refs.json',refs),
            'references_transfer':save('parent/transfer_refs.json',[])}
    inputs['parent_primary_verification']=save('parent/verification.json',{
        'status':'passed','passes_all_gates':False,'result_sha256':inputs['parent_primary_result']['sha256'],
        'bindings':{item['path']:item['sha256'] for item in inputs.values()}})
    spec['inputs']=inputs
    specification=save('source/specification.json',spec)
    sources={specification['path']:specification['sha256']}
    review=save('review.json',{'approved_for_execution_lock':True,'source_files':sources})
    tests=save('tests.json',{'exit_code':0,'source_files':sources})
    feasibility=save('feasibility.json',{'synthetic':True})
    design=save('out/design_lock.json',{'locked_at_utc':'2026-09-08T00:00:01Z','sources':sources,
        'inputs':{item['path']:item['sha256'] for item in inputs.values()},'review':review,'pre_fit_tests':tests,
        'synthetic_feasibility':feasibility,'maximum_new_readout_fits':3,'strength_fits':0,
        'historical_new_fits_before_lock':0,'new_scores_before_lock':False})
    resources={'elapsed_seconds':.01,'peak_rss_bytes':1000000}
    files={key:save('out/'+key+'.json',value) for key,value in [
        ('references_selection',prepared),('references_transfer',[]),('archive_metadata',[])]}
    data=save('out/data_lock.json',{'closed_at_utc':'2026-09-08T00:00:02Z',
        'design_lock_sha256':design['sha256'],'files':files,'references_labels_attached':False,
        'new_fits':0,'new_scores':False,'resources':resources})
    save('out/selection_delay1_attempt.json',{'started_at_utc':'2026-09-08T00:00:03Z','design_lock_sha256':design['sha256']})
    odds=save('out/elo_odds_readout_delay1.json',old_readout)
    result_readout=deepcopy(old_readout);result_readout['kind']='result';result_readout['closed_at_utc']='2026-09-08T00:00:04Z'
    result=save('out/elo_result_readout_delay1.json',result_readout)
    records=training+future;issued=[];states=[]
    for i,row in enumerate(prepared):
        state_id=f'{i:04d}';clock=row['forecast_cutoff_utc']
        history=v.support_history(records,clock,1);ids=[r['match_id'] for r in history]
        states.append({'state_id':state_id,'cutoff_utc':clock,'batch_ids':[row['match_id']],
            'supported':[False],'support_training_ids':ids,'support_training_ids_sha256':v.digest(ids),
            'ratings':{'result':{'x':[0.],'provenance':{'cutoff_utc':clock,'extra_days':1,'kind':'result','K':14.,
                'fixtures':[{'match_id':row['match_id'],'rating_home':1000.,'rating_away':1000.,
                             'home_last_update_utc':None,'away_last_update_utc':None}]}}}})
        new=deepcopy(row);new.update(support=False,elo_state_id=state_id)
        new['probabilities'].update({v.CANDIDATE:list(uniform),v.CONTROL:list(uniform)});issued.append(new)
    closure=save('out/selection_delay1_issuance_lock.json',{'closed_at_utc':'2026-09-08T00:00:05Z',
        'forecasts':save('out/issued.json',issued),'rows':6,'labels_attached':False,'data_lock_sha256':data['sha256'],
        'states':save('out/states.json',states),'readouts':{'odds':odds,'result':result},'optimizer_fits':1,'resources':resources})
    labeled=[{**row,'label':i%3} for i,row in enumerate(issued)]
    report=run.make_report(labeled,spec,'selection',v.CANDIDATE)
    output=save('out/selection_delay1.json',{'completed_at_utc':'2026-09-08T00:00:06Z','summary':report,
        'design_lock_sha256':design['sha256'],'data_lock_sha256':data['sha256'],'issuance_lock_sha256':closure['sha256'],'resources':resources})
    save('out/selection_delay1_decision_lock.json',{'closed_at_utc':'2026-09-08T00:00:07Z',
        'selected_candidate':v.CANDIDATE,'passes_all_gates':False,'result_sha256':output['sha256'],'design_lock_sha256':design['sha256']})
    monkeypatch.setattr(v,'ROOT',tmp_path);monkeypatch.setattr(v,'HERE',source)
    monkeypatch.setattr(v.parent,'ROOT',tmp_path)
    monkeypatch.setattr(v.parent,'raw_archive',lambda contract,metadata:records)
    monkeypatch.setattr(v.parent,'reference_check',lambda contract,phase,refs:None)
    monkeypatch.setattr(v,'reject_operational_imports',lambda:None)
    return out


def test_full_primary_artifact_verifier_end_to_end(tmp_path,monkeypatch):
    out=synthetic_execution(tmp_path,monkeypatch)
    result=v.verify(out,'selection',1)
    assert result['status']=='passed' and result['passes_all_gates'] is False
    assert result['rows']==6 and result['supported']==0
    assert result['candidate_and_control_vectors_replayed']==12
    assert result['fits_performed']==0 and result['maximum_prediction_error']==0
    assert (out/'selection_delay1_verification.json').exists()


@pytest.mark.parametrize('target',['attempt','data','candidate','late_failure'])
def test_full_stage_guard_rejects_tampering(tmp_path,monkeypatch,target):
    out=synthetic_execution(tmp_path,monkeypatch)
    if target=='late_failure':
        original=v.audit_report
        def audit(*args):
            original(*args);(out/'concurrent_failure.json').write_text('{}')
        monkeypatch.setattr(v,'audit_report',audit)
    else:
        path=out/({'attempt':'selection_delay1_attempt.json','data':'data_lock.json','candidate':'issued.json'}[target])
        payload=json.loads(path.read_text())
        if target=='attempt':payload['design_lock_sha256']='0'*64
        if target=='data':payload['new_scores']=True
        if target=='candidate':payload[0]['probabilities'][v.CANDIDATE]=[.6,.2,.2]
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):v.verify(out,'selection',1)
    assert not (out/'selection_delay1_verification.json').exists()
