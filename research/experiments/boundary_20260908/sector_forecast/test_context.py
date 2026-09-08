import json
from pathlib import Path

import pytest

from research.experiments.boundary_20260908.sector_forecast.context import iter_contexts
from research.experiments.boundary_20260908.sector_pilot import ledger


def line(second,payload):
    ms=round(second*1000);h,rest=divmod(ms,3600000);m,rest=divmod(rest,60000);s,ms=divmod(rest,1000)
    return f'{h:02}:{m:02}:{s:02}.{ms:03}'+json.dumps(payload)+'\n'


def patch(second,driver='1',**body):return line(second,{'Lines':{driver:body}})


def paths(tmp_path,timing,track=None,status=None):
    values={'TimingData':timing,'TrackStatus':track or line(0,{'Status':'1'}),
            'SessionStatus':status or line(0,{'Status':'Started'}),'TimingAppData':line(0,{'Lines':{}})}
    result={}
    for kind,value in values.items():
        result[kind]=tmp_path/(kind+'.jsonStream');result[kind].write_text(value)
    return result


def prefix():return patch(1,NumberOfLaps=0,InPit=False)+patch(10,Sectors={'0':{'Value':'30'}})+patch(40,Sectors={'1':{'Value':'31'}})


def run(p):return list(iter_contexts(p,202201))


def test_fresh_pit_out_contaminates_but_stale_true_does_not_carry_forever(tmp_path):
    timing=patch(1,NumberOfLaps=0,InPit=False,PitOut=True)+patch(10,Sectors={'0':{'Value':'30'}})+patch(40,Sectors={'1':{'Value':'31'}})
    timing+=patch(70,NumberOfLaps=1)+patch(100,Sectors={'0':{'Value':'30'}})+patch(110,PitOut=True)+patch(130,Sectors={'1':{'Value':'31'}})
    rows=run(paths(tmp_path,timing))
    assert [r['known_pit_contamination'] for r in rows]==[True,True,False,True]
    assert rows[2]['last_received_pit_out_value'] is True and not rows[2]['fresh_pit_out_true_in_epoch']
    assert rows[3]['pit_contamination_sources']['PitOut_fresh_true']['available_ms']==110000


def test_current_in_pit_carries_into_new_epoch_but_atomic_exit_does_not(tmp_path):
    timing=patch(1,NumberOfLaps=0,InPit=True)+patch(10,Sectors={'0':{'Value':'30'}})+patch(70,NumberOfLaps=1)+patch(100,Sectors={'0':{'Value':'31'}})
    timing+=patch(140,NumberOfLaps=2,InPit=False)+patch(170,Sectors={'0':{'Value':'32'}})
    rows=run(paths(tmp_path,timing));assert [r['known_pit_contamination'] for r in rows]==[True,True,False]
    assert rows[1]['pit_contamination_sources']['InPit']['active_at_epoch_start']


def test_brief_control_contamination_between_driver_updates_is_retained(tmp_path):
    timing=prefix()+patch(70,NumberOfLaps=1)+patch(100,Sectors={'0':{'Value':'32'}})
    track=line(0,{'Status':'1'})+line(20,{'Status':'4'})+line(21,{'Status':'1'})
    rows=run(paths(tmp_path,timing,track=track))
    assert [r['known_neutralization_contamination'] for r in rows]==[False,True,False]
    assert rows[1]['strictly_prior_control_snapshot']['TrackStatus']['value']=='1'
    assert rows[1]['neutralization_contamination_sources']['4']['available_ms']==20000


def test_transitions_apply_to_peer_epochs_without_peer_packet(tmp_path):
    timing=patch(1,'1',NumberOfLaps=0,InPit=False)+patch(1,'2',NumberOfLaps=0,InPit=False)
    timing+=patch(10,'1',Sectors={'0':{'Value':'30'}})+patch(25,'1',Position='1')+patch(40,'2',Sectors={'0':{'Value':'30'}})
    track=line(0,{'Status':'1'})+line(20,{'Status':'6'})+line(21,{'Status':'1'})
    rows=run(paths(tmp_path,timing,track=track));assert rows[-1]['driver']=='2' and rows[-1]['known_neutralization_contamination']


def test_equal_clock_controls_excluded_then_available_at_later_checkpoint(tmp_path):
    track=line(0,{'Status':'1'})+line(10,{'Status':'4'})
    rows=run(paths(tmp_path,prefix(),track=track));assert not rows[0]['known_asof_contamination'] and rows[1]['known_asof_contamination']
    assert rows[0]['strictly_prior_control_snapshot']['TrackStatus']['available_ms']==0


def test_active_neutralization_carries_across_counter_boundary(tmp_path):
    rows=run(paths(tmp_path,prefix(),track=line(0,{'Status':'6'})))
    assert all(r['known_neutralization_contamination'] for r in rows)
    assert rows[0]['neutralization_contamination_sources']['6']['active_at_epoch_start']


def test_unknown_coverage_persists_until_observed_new_epoch(tmp_path):
    timing=prefix()+patch(70,NumberOfLaps=1)+patch(100,Sectors={'0':{'Value':'30'}})
    rows=run(paths(tmp_path,timing,track=line(20,{'Status':'1'})))
    assert [r['unknown_coverage'] for r in rows]==[True,True,False]
    assert not any(r['known_asof_contamination'] for r in rows)


def test_atomic_patch_key_order_and_sector_order_have_identical_context(tmp_path):
    initial=patch(1,NumberOfLaps=0,InPit=False)
    p=paths(tmp_path,initial+patch(10,Sectors={'0':{'Value':'30'},'1':{'Value':'31'}},PitOut=True,InPit=True));a=run(p)
    p['TimingData'].write_text(initial+patch(10,InPit=True,PitOut=True,Sectors={'1':{'Value':'31'},'0':{'Value':'30'}}));b=run(p)
    assert a==b and all(r['known_pit_contamination'] for r in a)
    assert a[0]['strictly_prior_control_snapshot']==a[1]['strictly_prior_control_snapshot']


def test_future_poison_appends_cannot_change_any_saved_context(tmp_path):
    p=paths(tmp_path,prefix());before=run(p)
    with p['TimingData'].open('a') as f:f.write(patch(100,NumberOfLaps=1,InPit=True,PitOut=True,Sectors={'0':{'Value':'99'}}))
    with p['TrackStatus'].open('a') as f:f.write(line(90,{'Status':'7'})+'00:02:00.000{invalid}\n'+'invalid future timestamp\n')
    with p['SessionStatus'].open('a') as f:f.write(line(80,{'Status':'Finished'}))
    after=run(p);assert before==after[:len(before)]


def test_regressed_control_is_marked_unknown_and_never_backdated(tmp_path):
    track=line(0,{'Status':'1'})+line(20,{'Status':'4'})+line(15,{'Status':'1'})
    rows=run(paths(tmp_path,prefix(),track=track));assert not rows[0]['unknown_coverage']
    assert 'TrackStatus_timestamp_regression' in rows[1]['unknown_flags']
    assert rows[1]['strictly_prior_control_snapshot']['TrackStatus']['available_ms']==20000


def test_unknown_pit_and_unrecognized_numeric_track_are_not_declared_clean(tmp_path):
    timing=patch(1,NumberOfLaps=0)+patch(10,Sectors={'0':{'Value':'30'}})
    rows=run(paths(tmp_path,timing,track=line(0,{'Status':'99'})))
    assert rows[0]['unknown_coverage'] and not rows[0]['known_asof_contamination']
    assert set(rows[0]['unknown_flags'])=={'InPit_coverage_unknown','TrackStatus_unrecognized_value'}


def test_all_updates_and_pilot_epoch_counter_semantics_retained(tmp_path):
    timing=prefix()+patch(41,Sectors={'1':{'Value':'32'}})+patch(42,Sectors={'1':{'Value':'32'}})
    timing+=patch(60,Sectors={'0':{'Value':''},'1':{'Value':''}})+patch(70,Sectors={'0':{'Value':'32'}})
    timing+=patch(100,NumberOfLaps=3)+patch(110,Sectors={'0':{'Value':'33'}})+patch(140,NumberOfLaps=2)+patch(150,Sectors={'0':{'Value':'34'}})
    timing+=patch(180,NumberOfLaps=3)+patch(190,Sectors={'0':{'Value':'35'}})+patch(220,NumberOfLaps=4)+patch(230,Sectors={'0':{'Value':'36'}})
    p=paths(tmp_path,timing);context=run(p);pilot=[];ledger.build(p,202201,pilot.append)
    fields=['event_key','driver','packet_sequence','sector','checkpoint_ms','counter_seen','local_epoch','update_kind']
    assert [[r[k] for k in fields] for r in context]==[[r[k] for k in fields] for r in pilot]
    assert [r['epoch_attribution_flags'] for r in context]==[r['ambiguity_reasons'] for r in pilot]


@pytest.mark.parametrize('event',[202201,202301])
def test_every_preserved_pilot_key_counter_epoch_matches_without_csv(event):
    root=Path(__file__).resolve().parents[4];manifest=root/'artifacts/research/boundary_20260908/sector_pilot/network_retry/acquisition.json'
    if not manifest.exists():pytest.skip('Optional exact pilot raw inputs are not part of a source-only checkout')
    acquired=json.loads(manifest.read_text());p={r['stream']:root/r['decoded_body_path'] for r in acquired['streams'] if r['event_key']==event}
    p['TrackStatus']=root/f'data/f1/boundary_20260908/control/{event}_TrackStatus.jsonStream'
    if not all(path.exists() for path in p.values()):pytest.skip('Optional exact pilot raw bodies absent')
    context=list(iter_contexts(p,event));pilot=[];ledger.build(p,event,pilot.append)
    fields=['event_key','driver','packet_sequence','sector','checkpoint_ms','counter_seen','local_epoch','update_kind']
    assert [[r[k] for k in fields] for r in context]==[[r[k] for k in fields] for r in pilot]
    assert [r['epoch_attribution_flags'] for r in context]==[r['ambiguity_reasons'] for r in pilot]
    assert len(context)=={202201:2224,202301:2088}[event]
