from copy import deepcopy
import json
from pathlib import Path

import pytest

from research.experiments.boundary_20260908.sector_forecast.execution import data
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


def test_published_parsers_baselines_and_actual_fixed_encoder_integrate_on_synthetic_raw(tmp_path,monkeypatch):
    from research.experiments.boundary_20260908.sector_forecast.execution.features import FEATURES,encode
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
