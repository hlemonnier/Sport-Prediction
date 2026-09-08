from copy import deepcopy
import json
from pathlib import Path

import pytest

from research.experiments.boundary_20260908.sector_forecast.execution_v2 import data
from research.experiments.boundary_20260908.sector_forecast import baselines
from research.experiments.boundary_20260908.sector_pilot import run as pilot_run


def line(second,payload):
    ms=round(second*1000);h,rest=divmod(ms,3600000);m,rest=divmod(rest,60000);s,ms=divmod(rest,1000)
    return f'{h:02}:{m:02}:{s:02}.{ms:03}'+json.dumps(payload)+'\n'


def patch(second,driver='1',**body):return line(second,{'Lines':{driver:body}})


def make_paths(tmp_path,timing=None):
    if timing is None:
        timing=patch(1,NumberOfLaps=0,InPit=False,Retired=False,Stopped=False)
        for i in range(5):
            timing+=patch(11+100*i,Sectors={'0':{'Value':'30'}})
            timing+=patch(41+100*i,Sectors={'1':{'Value':'30'}})
            timing+=patch(71+100*i,NumberOfLaps=i+1,Sectors={'2':{'Value':'30'}},LastLapTime={'Value':'1:30'})
    bodies={'TimingData':timing,'SessionStatus':line(0,{'Status':'Started'}),
            'TrackStatus':line(0,{'Status':'1'}),'TimingAppData':line(0,{'Lines':{}})}
    result={}
    for kind,body in bodies.items():result[kind]=tmp_path/(kind+'.jsonStream');result[kind].write_text(body)
    return result


@pytest.fixture(autouse=True)
def fixed_test_encoder(monkeypatch):
    def encode(checkpoint,history,ctx,result):
        assert history and all(r['driver']==checkpoint['driver'] and r['available_ms']<checkpoint['checkpoint_ms'] for r in history)
        assert all(r['observed_valid_completed'] is True for r in history)
        assert set(result['points'])==set(baselines.REFERENCE_NAMES)|{baselines.GAUSSIAN_NAME}
        return {'current_sector':checkpoint['sector_seconds'],'history_last_lap':history[-1]['full_lap_seconds'],'optional':float('nan')}
    monkeypatch.setattr(data,'_feature_encoder',lambda:(('current_sector','history_last_lap','optional'),encode))


def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=True)


def test_every_checkpoint_retained_and_all_models_share_minimum_history_gate(tmp_path):
    records,diagnostics=data.build_event(make_paths(tmp_path),202201)
    assert len(records)==10 and [r['status'] for r in records]==['insufficient_history']*6+['issued']*4
    assert diagnostics['stream_input_valid'] and diagnostics['canonical_csv_read'] is False
    assert len({r['issuance_id'] for r in records})==10
    for row in records:
        assert 'target' not in row and 'y' not in row
        if row['status']=='issued':
            assert len(row['points'])==5 and row['history_last_available_ms']<row['checkpoint_ms']
            assert tuple(row['features'])==('current_sector','history_last_lap','optional')
        else:assert row['points']=={} and row['features']=={}


def test_target_free_build_never_reads_or_calls_a_target_loader(tmp_path,monkeypatch):
    p=make_paths(tmp_path);original_open=Path.open
    def guarded(path,*args,**kwargs):
        if path.suffix=='.csv':raise AssertionError('Target CSV access during feature construction')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    monkeypatch.setattr(pilot_run,'target_index',lambda *_:(_ for _ in ()).throw(AssertionError('Target loader called')))
    records,_=data.build_event(p,202201);assert any(r['status']=='issued' for r in records)


def test_future_valid_packet_and_control_poisoning_preserves_all_earlier_records(tmp_path):
    p=make_paths(tmp_path);before,_=data.build_event(p,202201)
    with p['TimingData'].open('a') as f:f.write(patch(600,Retired=True,InPit=True,PitOut=True)+patch(650,Sectors={'0':{'Value':'999'}}))
    with p['TrackStatus'].open('a') as f:f.write(line(580,{'Status':'7'}))
    with p['SessionStatus'].open('a') as f:f.write(line(590,{'Status':'Finished'}))
    after,_=data.build_event(p,202201);assert canonical(before)==canonical(after[:len(before)])


def synthetic_completion(identity,clock,driver='1',valid=True):
    return {'event_key':202201,'driver':driver,'completion_id':identity,'available_ms':clock,'packet_sequence':0,
            'observed_valid_completed':valid,'sectors_seconds':[30.,30.,30.],'full_lap_seconds':90.,'local_epoch':999}


def test_equal_time_and_other_driver_completions_never_enter_own_history(tmp_path,monkeypatch):
    timing=patch(1,NumberOfLaps=0,InPit=False)+patch(40,Sectors={'0':{'Value':'30'}})+patch(60,Sectors={'1':{'Value':'30'}})
    def completed(_paths,_event,stats):
        yield synthetic_completion('a',10000);yield synthetic_completion('b',20000)
        yield synthetic_completion('peer',30000,'2');yield synthetic_completion('invalid',30001,valid=False)
        yield synthetic_completion('same_clock',40000);stats['stream_input_valid']=True
    monkeypatch.setattr(data.completed,'iter_completed',completed)
    records,_=data.build_event(make_paths(tmp_path,timing),202201)
    assert records[0]['history_count']==2 and records[0]['status']=='insufficient_history'
    assert records[1]['history_count']==3 and records[1]['status']=='issued'
    assert records[1]['history_last_available_ms']==40000


def test_complete_stream_validation_must_finish_before_records_return(tmp_path,monkeypatch):
    consumed=[]
    def completed(_paths,_event,stats):
        yield synthetic_completion('a',10000);consumed.append(True);stats['stream_input_valid']=False
    monkeypatch.setattr(data.completed,'iter_completed',completed)
    with pytest.raises(ValueError,match='Full raw stream'):data.build_event(make_paths(tmp_path),202201)
    assert consumed==[True]


def test_unplaceable_future_stream_gap_rejects_archive_without_publishing_prefix(tmp_path):
    p=make_paths(tmp_path)
    with p['TrackStatus'].open('a') as f:f.write('unplaceable timestamp\n')
    with pytest.raises(ValueError,match='Full raw stream'):data.build_event(p,202201)


@pytest.mark.parametrize('change',['missing','duplicate','wrong_counter','wrong_attribution'])
def test_missing_duplicate_or_inconsistent_context_fails_closed(tmp_path,monkeypatch,change):
    original=data.context.iter_contexts
    def changed(paths,event,stats):
        rows=list(original(paths,event,stats))
        if change=='missing':rows.pop()
        elif change=='duplicate':rows.append(deepcopy(rows[0]))
        elif change=='wrong_counter':rows[0]['counter_seen']=999
        else:rows[0]['epoch_attribution_flags']=['future repair']
        yield from rows
    monkeypatch.setattr(data.context,'iter_contexts',changed)
    with pytest.raises(ValueError,match='context|Context'):data.build_event(make_paths(tmp_path),202201)


def test_known_retired_or_stopped_is_an_asof_gate_without_deleting_rows(tmp_path):
    p=make_paths(tmp_path);body=p['TimingData'].read_text();needle=patch(311,Sectors={'0':{'Value':'30'}})
    p['TimingData'].write_text(body.replace(needle,patch(305,Stopped=True)+needle))
    records,_=data.build_event(p,202201);assert len(records)==10
    assert records[6]['status']=='retired_or_stopped' and records[6]['history_count']==3
    assert 'known_retired_or_stopped_at_issuance' in records[6]['status_reasons']
    assert records[5]['status']=='insufficient_history'


@pytest.mark.parametrize('bad',['schema','infinity'])
def test_feature_schema_or_infinity_cannot_silently_drop_one_model(tmp_path,monkeypatch,bad):
    names=('x',)
    monkeypatch.setattr(data,'_feature_encoder',lambda:(names,lambda *_:{'wrong':1.} if bad=='schema' else {'x':float('inf')}))
    with pytest.raises(ValueError,match='schema|infinity|numeric'):data.build_event(make_paths(tmp_path),202201)


def test_repeated_unchanged_s1_in_s2_packet_uses_immutable_prior_support(tmp_path):
    p=make_paths(tmp_path);body=p['TimingData'].read_text()
    body=body.replace(patch(341,Sectors={'1':{'Value':'30'}}),patch(341,Sectors={'0':{'Value':'30'},'1':{'Value':'30'}}))
    p['TimingData'].write_text(body);rows,_=data.build_event(p,202201)
    repeat=[r for r in rows if r['checkpoint_ms']==341000 and r['sector']==1][0]
    s2=[r for r in rows if r['checkpoint_ms']==341000 and r['sector']==2][0]
    assert repeat['status']=='pilot_excluded' and s2['status']=='issued'
    source=s2['source_provenance']['s1_prefix']
    assert source['prior_support']['available_ms']==311000
    assert source['value_source']['available_ms']==341000
    assert source['same_packet_value_update'] and not source['value_changed_from_prior_support']


def correction_stream(tmp_path,*,reverse_keys=False,prior_revision=False,s3='23.993',no_prior=False):
    """Three complete histories, then the real Hungary packet-value structure."""
    paths=make_paths(tmp_path)
    warmup=paths['TimingData'].read_text().split(patch(311,Sectors={'0':{'Value':'30'}}))[0]
    before=warmup
    if not no_prior:
        before+=patch(340.999 if not prior_revision else 311,Sectors={'0':{'Value':'30.526'}})
    if prior_revision:
        before+=patch(340.999,Sectors={'0':{'PreviousValue':'30.526','Value':'30.486'}})
    sectors={'0':{'PreviousValue':'30.526','Value':'30.486'},
             '1':{'Value':'31.233'},'2':{'Value':s3}}
    if reverse_keys:sectors=dict(reversed(list(sectors.items())))
    after=before+patch(341,Sectors=sectors)
    paths['TimingData'].write_text(after)
    return paths,before,after


@pytest.mark.parametrize('reverse_keys',[False,True])
def test_atomic_s1_correction_uses_current_value_without_backdating_or_revising_prior_forecast(tmp_path,monkeypatch,reverse_keys):
    from research.experiments.boundary_20260908.sector_forecast.execution_v2.features import FEATURES,encode
    monkeypatch.setattr(data,'_feature_encoder',lambda:(tuple(FEATURES),encode))
    paths,before,after=correction_stream(tmp_path,reverse_keys=reverse_keys)
    paths['TimingData'].write_text(before);old,_=data.build_event(paths,202201)
    paths['TimingData'].write_text(after);rows,_=data.build_event(paths,202201)
    assert canonical(old)==canonical(rows[:len(old)])
    revision,s2=rows[-2:]
    assert revision['sector']==1 and revision['status']=='pilot_excluded'
    assert s2['sector']==2 and s2['status']=='issued' and s2['history_count']==3
    assert len(s2['points'])==5 and len(s2['features'])==75
    assert s2['points']['sector_last_template']==pytest.approx(30.486+31.233+30.)
    scale=s2['points']['sector_pace_scaled_median5']
    assert s2['features']['s1_over_b4']*scale==pytest.approx(30.486)
    assert s2['features']['s2_over_b4']*scale==pytest.approx(31.233)
    source=s2['source_provenance']['s1_prefix']
    assert source['prior_support']['value']==30.526
    assert source['prior_support']['available_ms']==340999
    assert source['prior_support']['packet_sequence']<s2['packet_sequence']
    assert source['value_source']=={'value':30.486,'available_ms':341000,'sequence':s2['packet_sequence']}
    assert source['same_packet_value_update'] and source['value_changed_from_prior_support']


def test_prior_packet_s1_revision_then_current_repeat_keeps_both_actual_source_clocks(tmp_path):
    paths,_,_=correction_stream(tmp_path,prior_revision=True)
    rows,_=data.build_event(paths,202201);s2=rows[-1]
    assert s2['status']=='issued'
    assert s2['points']['sector_last_template']==pytest.approx(91.719)
    source=s2['source_provenance']['s1_prefix']
    assert source['prior_support']['value']==30.526
    assert source['prior_support']['available_ms']==311000
    assert source['value_source']['available_ms']==341000
    assert source['same_packet_value_update'] and source['value_changed_from_prior_support']


@pytest.mark.parametrize('regressed',[False,True])
def test_s2_keeps_pilot_eligibility_after_an_excluded_intermediate_revision(tmp_path,regressed):
    paths=make_paths(tmp_path);body=paths['TimingData'].read_text()
    first=patch(311,Sectors={'0':{'Value':'30'}})
    # Ordinary revisions are excluded checkpoints but preserve the pilot's S2
    # prerequisite. A regressed revision contaminates the whole pilot epoch,
    # so the later S2 must remain excluded too. Neither rule changes in v2.
    revision_time=310 if regressed else 312
    body=body.replace(first,first+patch(revision_time,Sectors={'0':{'Value':'30.486'}}))
    paths['TimingData'].write_text(body);rows,_=data.build_event(paths,202201)
    revision=next(r for r in rows if r['recorded_ms']==revision_time*1000)
    s2=next(r for r in rows if r['checkpoint_ms']==341000 and r['sector']==2)
    assert revision['status']=='pilot_excluded'
    if regressed:
        assert 'raw_timestamp_regression' in revision['status_reasons']
        assert not s2['pilot_candidate_checkpoint'] and s2['status']=='pilot_excluded'
        assert 'raw_timestamp_regression' in s2['status_reasons']
        assert s2['points']=={} and s2['features']=={}
        return
    assert s2['pilot_candidate_checkpoint'] and s2['status']=='issued'
    assert s2['points']['sector_last_template']==pytest.approx(90.486)
    source=s2['source_provenance']['s1_prefix']
    assert source['prior_support']['value']==30.
    assert source['prior_support']['packet_sequence']<source['value_source']['sequence']<s2['packet_sequence']
    assert source['value_source']['value']==30.486
    assert not source['same_packet_value_update']


def test_atomic_revision_is_independent_of_sector_key_order_and_current_s3(tmp_path,monkeypatch):
    from research.experiments.boundary_20260908.sector_forecast.execution_v2.features import FEATURES,encode
    monkeypatch.setattr(data,'_feature_encoder',lambda:(tuple(FEATURES),encode))
    first=tmp_path/'first';first.mkdir();second=tmp_path/'second';second.mkdir()
    p1,_,_=correction_stream(first)
    p2,_,_=correction_stream(second,reverse_keys=True,s3='999999999')
    a,_=data.build_event(p1,202201);b,_=data.build_event(p2,202201)
    assert canonical(a)==canonical(b)


def test_same_packet_first_s1_s2_without_prior_support_remains_explicitly_excluded(tmp_path):
    paths,_,_=correction_stream(tmp_path,no_prior=True)
    rows,_=data.build_event(paths,202201)
    assert rows[-2]['sector']==1 and rows[-2]['status']=='issued'
    assert rows[-1]['sector']==2 and rows[-1]['status']=='pilot_excluded'
    assert 'no_strictly_prior_s1_in_epoch' in rows[-1]['status_reasons']
    assert rows[-1]['points']=={} and rows[-1]['features']=={}


@pytest.mark.parametrize('bad',['missing','equal_packet','different_epoch','different_driver','different_event',
                               'ambiguous','future_support_clock','future_value_clock','future_value_packet'])
def test_prefix_rejects_missing_wrong_epoch_or_future_support(tmp_path,bad):
    paths,_,_=correction_stream(tmp_path);raw=[]
    data.ledger.build(paths,202201,raw.append)
    s2=deepcopy(raw[-1]);prior=deepcopy(raw[-3])
    if bad=='missing':prior=None
    elif bad=='equal_packet':prior['packet_sequence']=s2['packet_sequence']
    elif bad=='different_epoch':prior['local_epoch']+=1
    elif bad=='different_driver':prior['driver']='different'
    elif bad=='different_event':prior['event_key']=202202
    elif bad=='ambiguous':prior['ambiguity_reasons']=['counter_quarantine']
    elif bad=='future_support_clock':prior['checkpoint_ms']=s2['checkpoint_ms']+1
    elif bad=='future_value_clock':s2['received_driver_fields']['Sector1']['available_ms']=s2['checkpoint_ms']+1
    else:s2['received_driver_fields']['Sector1']['sequence']=s2['packet_sequence']+1
    with pytest.raises(ValueError,match='prior S1|S1 value source'):data._prefix(s2,prior)


def test_published_parsers_baselines_and_actual_fixed_encoder_integrate_on_synthetic_raw(tmp_path,monkeypatch):
    from research.experiments.boundary_20260908.sector_forecast.execution_v2.features import FEATURES,encode
    monkeypatch.setattr(data,'_feature_encoder',lambda:(tuple(FEATURES),encode))
    rows,diagnostics=data.build_event(make_paths(tmp_path),202201)
    issued=[r for r in rows if r['status']=='issued']
    assert len(issued)==4 and len(FEATURES)==75
    assert all(tuple(r['features'])==tuple(FEATURES) and len(r['points'])==5 for r in issued)
    assert issued[0]['features']['stage']==1. and issued[1]['features']['stage']==2.
    assert issued[0]['features']['history_count']==issued[1]['features']['history_count']==3.
    assert diagnostics['target_free'] and not diagnostics['terminal']


def target_free_row(ordinal,sector,clock):
    return {'event_key':202201,'ledger_id':ordinal,'driver':'1','sector':sector,'packet_sequence':ordinal,
            'checkpoint_ms':clock,'issuance_id':f'202201:1:{ordinal}:S{sector}','candidate_checkpoint':True,
            'status':'issued','points':{'fixed':90.},'features':{'prefix':30.}}


def test_targets_attached_to_copies_and_s1_s2_share_one_weight_identity(tmp_path):
    csv=tmp_path/'completed.csv';csv.write_text('Time,DriverNumber,LapNumber,LapTime,IsAccurate,PitInTime,PitOutTime,TrackStatus\n100,1,5,90,True,,,1\n')
    records=[target_free_row(0,1,20000),target_free_row(1,2,60000),target_free_row(2,1,100000)]
    before=deepcopy(records);labeled=data.attach_targets(records,csv,True)
    assert records==before and len(labeled)==3
    assert labeled[0]['target_id']==labeled[1]['target_id']=='202201:1:csvrow:0'
    assert labeled[0]['target_group_id']==labeled[1]['target_group_id']
    assert labeled[0]['y']==labeled[1]['y']==90.
    assert labeled[2]['target_id'] is None and labeled[2]['y'] is None
    assert labeled[2]['outcome_status']=='unmatched_in_terminal_recorded_archive'
    assert all(r['features']==o['features'] and r['points']==o['points'] for r,o in zip(labeled,records))
    assert data.attach_targets(records,csv,False)[-1]['outcome_status']=='unresolved_in_incomplete_archive'


def test_target_attachment_rejects_ambiguous_terminal_or_duplicate_issuance(tmp_path):
    with pytest.raises(ValueError,match='explicit boolean'):data.attach_targets([],tmp_path/'never-read.csv','False')
    row=target_free_row(0,1,1000)
    with pytest.raises(ValueError,match='Duplicate issuance'):data.attach_targets([row,row],tmp_path/'never-read.csv',True)
