import json
from collections import Counter
import pytest
from research.experiments.boundary_20260908.sector_pilot import ledger


def line(second,payload):
    ms=round(second*1000);h,rest=divmod(ms,3600000);m,rest=divmod(rest,60000);s,ms=divmod(rest,1000)
    return f'{h:02}:{m:02}:{s:02}.{ms:03}'+json.dumps(payload)+'\n'


def patch(second,**body):return line(second,{'Lines':{'1':body}})


def setup(tmp_path,timing,app=None,status=None):
    bodies={'TimingData':timing,'TimingAppData':app or line(0,{'Lines':{'1':{'Stints':{'0':{'Compound':'SOFT','New':'true','StartLaps':0}}}}}),
            'SessionStatus':status or line(0,{'Status':'Started'}),'TrackStatus':line(0,{'Status':'1'})}
    paths={}
    for kind,body in bodies.items():
        paths[kind]=tmp_path/(kind+'.jsonStream');paths[kind].write_text(body)
    return paths


def run(paths):
    rows=[];stats=ledger.build(paths,202201,rows.append);return rows,stats


def prefix():return patch(1,NumberOfLaps=0,InPit=False)+patch(10,Sectors={'0':{'Value':'30.0'}})+patch(40,Sectors={'1':{'Value':'30.1'}})


def test_causal_prefix_unchanged_by_future_packets_and_tyre_updates(tmp_path):
    paths=setup(tmp_path,prefix());before,_=run(paths)
    with paths['TimingData'].open('a') as f:f.write(patch(70,NumberOfLaps=1,LastLapTime={'Value':'99'})+patch(100,Sectors={'0':{'Value':'31'}}))
    with paths['TimingAppData'].open('a') as f:f.write(line(80,{'Lines':{'1':{'Stints':{'1':{'Compound':'HARD','TotalLaps':30}}}}}))
    after,_=run(paths);assert before==after[:2]
    assert [x['candidate_checkpoint'] for x in before]==[True,True]
    assert before[0]['received_tyre_candidate']['fields']['Compound']['value']=='SOFT'


def test_revision_repeat_preserved_without_overwriting_first_issuance(tmp_path):
    text=patch(1,NumberOfLaps=0,InPit=False)+patch(10,Sectors={'0':{'Value':'30'}})+patch(20,Sectors={'0':{'Value':'31'}})+patch(21,Sectors={'0':{'Value':'31'}})
    rows,stats=run(setup(tmp_path,text));assert [r['update_kind'] for r in rows]==['first','revision','repeat']
    assert rows[0]['sector_seconds']==30 and rows[0]['received_driver_fields']['Sector1']['value']==30
    assert stats['ledger_rows']==3 and sum(r['candidate_checkpoint'] for r in rows)==1


def test_counter_collision_and_reversal_quarantine(tmp_path):
    text=prefix()+patch(70,NumberOfLaps=1,Sectors={'0':{'Value':'31'}})+patch(90,NumberOfLaps=0)+patch(100,Sectors={'0':{'Value':'32'}})+patch(120,NumberOfLaps=1)+patch(130,Sectors={'0':{'Value':'33'}})+patch(150,NumberOfLaps=2)+patch(160,Sectors={'0':{'Value':'34'}})
    rows,_=run(setup(tmp_path,text));assert 'counter_sector_same_packet' in rows[2]['ambiguity_reasons']
    assert 'counter_quarantine' in rows[3]['ambiguity_reasons'] and 'counter_quarantine' in rows[4]['ambiguity_reasons']
    assert rows[-1]['candidate_checkpoint']


def test_clear_new_cycle_without_counter_remains_explicitly_ambiguous(tmp_path):
    text=prefix()+patch(60,Sectors={'0':{'Value':''},'1':{'Value':''}})+patch(100,Sectors={'0':{'Value':'32'}})
    rows,stats=run(setup(tmp_path,text));assert rows[-1]['update_kind']=='first'
    assert 'reset_cycle_without_counter_advance' in rows[-1]['ambiguity_reasons'];assert not rows[-1]['candidate_checkpoint'];assert stats['ledger_rows']==3


def test_equal_time_auxiliary_is_not_visible(tmp_path):
    paths=setup(tmp_path,prefix(),status=line(10,{'Status':'Started'}),app=line(10,{'Lines':{'1':{'Stints':{'0':{'Compound':'SOFT'}}}}}))
    rows,_=run(paths);assert rows[0]['session_status'] is None and rows[0]['received_tyre_candidate']['fields']=={}
    assert rows[1]['session_status']=='Started' and rows[1]['received_tyre_candidate']['fields']['Compound']['available_ms']==10000


def test_nonmonotone_raw_packet_is_not_backdated_or_silently_discarded(tmp_path):
    rows,stats=run(setup(tmp_path,prefix()+patch(35,Sectors={'0':{'Value':'29'}})))
    assert rows[-1]['recorded_ms']==35000 and rows[-1]['checkpoint_ms']==40000
    assert 'raw_timestamp_regression' in rows[-1]['ambiguity_reasons'];assert stats['timing']['timestamp_regressions']==1


def test_unknown_counter_and_unstarted_updates_are_retained(tmp_path):
    text=patch(1,InPit=True,Sectors={'0':{'Value':'28'}})+patch(2,Sectors={'1':{'Value':'29'}})
    rows,stats=run(setup(tmp_path,text,status=line(0,{'Status':'Inactive'})))
    assert len(rows)==2 and all(not r['candidate_checkpoint'] for r in rows)
    assert all('unknown_counter' in r['ambiguity_reasons'] for r in rows)
    assert stats['timing']['candidate_checkpoints']==0


def test_invalid_future_auxiliary_payload_or_timestamp_cannot_change_prefix(tmp_path):
    paths=setup(tmp_path,prefix());before,_=run(paths)
    with paths['TimingAppData'].open('a') as f:f.write('00:01:40.000{invalid}\n')
    after,_=run(paths);assert before==after
    with paths['TimingAppData'].open('a') as f:f.write('bad timestamp and payload\n')
    after,_=run(paths);assert before==after


def test_sector_snapshot_atomic_and_key_order_invariant(tmp_path):
    initial=patch(1,NumberOfLaps=0,InPit=False)
    paths=setup(tmp_path,initial+patch(10,Sectors={'0':{'Value':'30'},'1':{'Value':'31'}}));a,_=run(paths)
    paths['TimingData'].write_text(initial+patch(10,Sectors={'1':{'Value':'31'},'0':{'Value':'30'}}));b,_=run(paths)
    assert a==b and a[0]['received_driver_fields']==a[1]['received_driver_fields']
    assert 'no_strictly_prior_s1_in_epoch' in a[1]['ambiguity_reasons']


def test_explicit_clear_erases_latest_field_and_invalidates_s1_support(tmp_path):
    text=patch(1,NumberOfLaps=0,InPit=False)+patch(10,Sectors={'0':{'Value':'30'}})+patch(20,Sectors={'0':{'Value':''}})+patch(40,Sectors={'1':{'Value':'31'}})
    rows,_=run(setup(tmp_path,text));assert rows[-1]['received_driver_fields']['Sector1']['value'] is None
    assert not rows[-1]['raw_field_presence']['Sector1'] and not rows[-1]['candidate_checkpoint']


def test_ambiguous_s1_cannot_certify_following_s2(tmp_path):
    text=patch(1,NumberOfLaps=0,InPit=False)+patch(70,NumberOfLaps=1,Sectors={'0':{'Value':'30'}})+patch(110,Sectors={'1':{'Value':'31'}})
    rows,_=run(setup(tmp_path,text));assert 'prior_s1_attribution_ambiguous' in rows[-1]['ambiguity_reasons']
    assert not rows[-1]['candidate_checkpoint']


def test_outcome_diagnostic_retains_unmatched_and_excludes_equal_clock():
    from research.experiments.boundary_20260908.sector_pilot.run import resolve_target
    row={'event_key':1,'ledger_id':7,'driver':'1','checkpoint_ms':10000,'candidate_checkpoint':True}
    index={'1':[{'recorded_time_seconds':10},{'recorded_time_seconds':11}]}
    assert resolve_target(row,index,True)['target']['recorded_time_seconds']==11
    result=resolve_target(row,{},True);assert result['ledger_id']==7 and result['status']=='unmatched_in_terminal_recorded_archive'
    assert resolve_target(row,{},False)['status']=='unresolved_in_incomplete_archive'


def test_late_auxiliary_started_cannot_silently_revive_finished_status(tmp_path):
    timing=patch(1,NumberOfLaps=0,InPit=False)+patch(40,Sectors={'0':{'Value':'30'}})
    status=line(0,{'Status':'Started'})+line(20,{'Status':'Finished'})+line(15,{'Status':'Started'})
    rows,_=run(setup(tmp_path,timing,status=status));assert not rows[0]['candidate_checkpoint']
    assert rows[0]['session_status']=='Started' and rows[0]['session_status_source']['timestamp_regressed']
    assert 'auxiliary_latest_value_timestamp_regressed' in rows[0]['exclusion_reasons']


def test_repeated_s1_same_packet_as_s2_preserves_real_prior_support(tmp_path):
    timing=patch(1,NumberOfLaps=0,InPit=False)+patch(10,Sectors={'0':{'Value':'30'}})+patch(40,Sectors={'0':{'Value':'30'},'1':{'Value':'31'}})
    rows,_=run(setup(tmp_path,timing));assert rows[-1]['candidate_checkpoint']
    assert rows[-2]['update_kind']=='repeat' and rows[-1]['ambiguity_reasons']==[]
