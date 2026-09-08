"""Synthetic parser and chronology tests; no historical payload access."""
import json
from pathlib import Path

import pytest

from research.experiments.boundary_20260908.rival_gap_pilot_v2 import pilot as p


@pytest.mark.parametrize('value,category,number',[
    ('+ 1.250','seconds',1.25),(0,'seconds',0.),('1:02.500','seconds',62.5),
    ('1:02:03.5','seconds',3723.5),('+1 LAP','lap_deficit',1),('2 LAPS','lap_deficit',2),
    ('LAP 57','leader_lap_counter',57),('LEADER','leader',None),
    (None,'clear_null',None),('  ','clear_blank',None),('--','unavailable',None),
    ({'Value':'+0.4','Catching':True},'seconds',.4),({'Catching':False},'metadata_only',None),
    (True,'invalid',None),(-1,'invalid',None),('NaN','invalid',None),
    (float('inf'),'invalid',None),('1:61','invalid',None),('1.5:00','invalid',None),
    ('+0 LAP','invalid',None),('banana','invalid',None),([],'invalid',None),
    ('1e999999999:00','invalid',None),('++1:02','invalid',None)])
def test_field_categories_do_not_coerce_unknowns_or_laps_into_seconds(value,category,number):
    result=p.parse_value(value);assert result['category']==category and result['value']==number
    assert result['updates'] is (category!='metadata_only')


@pytest.mark.parametrize('value,category',[(1,'rank'),('20','rank'),('2.0','rank'),(0,'invalid'),(21,'invalid'),('1.2','invalid'),(False,'invalid'),('', 'clear_blank')])
def test_position_is_rank_not_physical_location(value,category):
    assert p.parse_value(value,position=True)['category']==category


def line(ms,patches):
    seconds,part=divmod(ms,1000)
    return f'00:{seconds//60:02}:{seconds%60:02}.{part:03}'+json.dumps({'Lines':patches})+'\n'


def stream(tmp_path,rows,name='raw'):
    path=tmp_path/name;path.write_text(''.join(rows));return path


def follower(gap='0.5',position='2'):
    return {'Position':position,'GapToLeader':'5.0','IntervalToPositionAhead':{'Value':gap}}


def test_equal_clock_packets_are_all_excluded_then_atomically_visible(tmp_path):
    path=stream(tmp_path,[line(1000,{'1':follower()}),line(1000,{'1':{'IntervalToPositionAhead':{'Value':'0.4'}}})])
    cursor=p.GapCursor(path)
    before=cursor.query('1',cutoff_ns=1_000_000_000)
    assert before['fields']['Position']['category']=='never_observed' and before['processed_packet_count']==0
    after=cursor.query('1',cutoff_ns=1_000_000_001)
    assert after['fields']['IntervalToPositionAhead']['value']==.4
    assert after['fields']['IntervalToPositionAhead']['age_seconds']==1e-9
    assert after['race_order_gap_ready'] and after['processed_packet_count']==2


@pytest.mark.parametrize('suffix',[line(90000,{'1':follower('100','3')}),'00:01:30.000{invalid json}\n'])
def test_timestamp_valid_future_append_cannot_change_earlier_snapshot(tmp_path,suffix):
    prefix=line(1000,{'1':follower()})
    a=stream(tmp_path,[prefix],name='prefix');b=stream(tmp_path,[prefix,suffix],name='extended')
    left=p.GapCursor(a).query('1',cutoff_ns=2_000_000_000)
    right=p.GapCursor(b).query('1',cutoff_ns=2_000_000_000)
    assert left==right


def test_missing_update_and_metadata_only_do_not_refresh_gap_age(tmp_path):
    path=stream(tmp_path,[line(1000,{'1':follower()}),line(8000,{'1':{'IntervalToPositionAhead':{'Catching':True}}}),
                          line(9000,{'1':{'Position':'2'}})])
    snapshot=p.GapCursor(path).query('1',cutoff_ns=12_000_000_000)
    assert snapshot['fields']['IntervalToPositionAhead']['age_seconds']==11
    assert snapshot['fields']['IntervalToPositionAhead']['sequence']==0
    assert snapshot['race_order_gap_ready'] is False
    assert snapshot['readiness_by_max_age_seconds']['30'] is True


@pytest.mark.parametrize('clear',[None,'',{'Value':None},{'Value':' '}])
def test_explicit_clear_replaces_previous_value_and_records_true_clock(tmp_path,clear):
    path=stream(tmp_path,[line(1000,{'1':follower()}),line(3000,{'1':{'IntervalToPositionAhead':clear}})])
    got=p.GapCursor(path).query('1',cutoff_ns=4_000_000_000)
    field=got['fields']['IntervalToPositionAhead']
    assert field['category'] in {'clear_null','clear_blank'} and field['value'] is None
    assert field['age_seconds']==1 and field['sequence']==1 and got['race_order_gap_ready'] is False


def test_rank_change_invalidates_previous_rival_interval_until_update(tmp_path):
    path=stream(tmp_path,[line(1000,{'1':follower()}),line(3000,{'1':{'Position':'3'}}),
        line(5000,{'1':{'IntervalToPositionAhead':{'Value':'1.5'}}})])
    cursor=p.GapCursor(path);got=cursor.query('1',cutoff_ns=4_000_000_000)
    assert got['flags']['interval_predates_last_rank_change'] and not got['race_order_gap_ready']
    got=cursor.query('1',cutoff_ns=6_000_000_000)
    assert not got['flags']['interval_predates_last_rank_change'] and got['race_order_gap_ready']


def test_same_packet_rank_swap_and_gap_updates_are_driver_and_key_order_invariant(tmp_path):
    initial={'1':follower(position='2'),'2':follower(position='3')}
    patch={'1':follower('1.2','3'),'2':follower('2.3','2')}
    reverse={driver:dict(reversed(list(value.items()))) for driver,value in reversed(list(patch.items()))}
    a=stream(tmp_path,[line(1000,initial),line(3000,patch)],name='a')
    b=stream(tmp_path,[line(1000,initial),line(3000,reverse)],name='b')
    aa=p.GapCursor(a).query('1',cutoff_ns=4_000_000_000)
    bb=p.GapCursor(b).query('1',cutoff_ns=4_000_000_000)
    assert aa==bb and aa['race_order_gap_ready']


def test_duplicate_latest_reported_rank_is_flagged_without_guessing_ahead_driver(tmp_path):
    path=stream(tmp_path,[line(1000,{'1':follower(),'2':follower()})])
    value=p.GapCursor(path).query('1',cutoff_ns=2_000_000_000)
    assert value['flags']['duplicate_reported_rank'] and not value['race_order_gap_ready']
    assert 'ahead_driver' not in value


@pytest.mark.parametrize('gap,interval,consistent',[('LEADER','',True),('LAP 4',None,True),('0','',True),('1.2','',False),('+1 LAP','',False),('LEADER','.5',False)])
def test_leader_gap_and_rank_categories_are_explicit(tmp_path,gap,interval,consistent):
    path=stream(tmp_path,[line(1000,{'1':{'Position':'1','GapToLeader':gap,'IntervalToPositionAhead':{'Value':interval}}})])
    value=p.GapCursor(path).query('1',cutoff_ns=2_000_000_000)
    assert value['race_order_gap_ready'] is consistent
    assert value['flags']['leader_rank_category_inconsistent'] is not consistent


def test_nonleader_with_leader_marker_is_inconsistent(tmp_path):
    patch=follower();patch['GapToLeader']='LEADER'
    value=p.GapCursor(stream(tmp_path,[line(1000,{'1':patch})])).query('1',cutoff_ns=2_000_000_000)
    assert value['flags']['leader_rank_category_inconsistent'] and not value['race_order_gap_ready']


def test_regressed_gap_uses_current_cumulative_clock_and_is_ambiguous_until_update(tmp_path):
    path=stream(tmp_path,[line(1000,{'1':follower()}),line(5000,{}),line(3000,{'1':{'IntervalToPositionAhead':{'Value':'.2'}}}),
                         line(8000,{'1':{'IntervalToPositionAhead':{'Value':'.3'}}})])
    cursor=p.GapCursor(path);value=cursor.query('1',cutoff_ns=6_000_000_000)
    field=value['fields']['IntervalToPositionAhead']
    assert field['recorded_ms']==3000 and field['available_ms']==5000 and field['age_seconds']==1
    assert value['flags']['latest_field_timestamp_regressed'] and not value['race_order_gap_ready']
    assert cursor.query('1',cutoff_ns=9_000_000_000)['race_order_gap_ready']


def test_processed_parse_gap_is_explicit_and_never_silently_filled(tmp_path):
    path=stream(tmp_path,[line(1000,{'1':follower()}),'00:00:03.000{bad json}\n',line(4000,{'1':follower('.1')})])
    cursor=p.GapCursor(path)
    assert cursor.query('1',cutoff_ns=2_000_000_000)['race_order_gap_ready']
    after=cursor.query('1',cutoff_ns=5_000_000_000)
    assert after['flags']['source_parse_gap_or_unknown_clock'] and not after['race_order_gap_ready']
    assert cursor.diagnostics()['source_ambiguous']
    cursor.close()


def test_full_inventory_independent_of_query_lags(tmp_path):
    path=stream(tmp_path,[line(1000,{'1':follower()}),line(3000,{'1':{'GapToLeader':'+1 LAP'}}),line(8000,{'1':{'GapToLeader':''}})])
    a=p.GapCursor(path);b=p.GapCursor(path)
    a.query('1',cutoff_ns=2_000_000_000);a.query('1',cutoff_ns=5_000_000_000)
    b.query('1',cutoff_ns=4_000_000_000)
    before_a=a.diagnostics();before_b=b.diagnostics()
    first=p.inventory_stream(path);second=p.inventory_stream(path)
    assert first==second
    assert a.diagnostics()==before_a and b.diagnostics()==before_b
    a.close();b.close()


@pytest.mark.parametrize('suffix',[b'\xff\n',b'{invalid json}\n',b'x'*10000+b'\n'])
def test_future_payload_bytes_and_size_are_not_read_or_decoded(tmp_path,monkeypatch,suffix):
    monkeypatch.setattr(p,'MAX_PAYLOAD_BYTES',1024)
    prefix=line(1000,{'1':follower()}).encode()
    a=tmp_path/'prefix';a.write_bytes(prefix)
    b=tmp_path/'extended';b.write_bytes(prefix+b'00:01:30.000'+suffix)
    cursor=p.GapCursor(b)
    assert cursor.query('1',cutoff_ns=2_000_000_000)==p.GapCursor(a).query('1',cutoff_ns=2_000_000_000)
    assert cursor.header_cursor.file.tell()==len(prefix)+12
    admitted=cursor.query('1',cutoff_ns=91_000_000_000)
    assert admitted['flags']['source_parse_gap_or_unknown_clock']
    assert admitted['processed_packet_count']==2
    cursor.close()


@pytest.mark.parametrize('header',[b'00:60:00.000',b'00:00:60.000',b'0:00:00.000',
    b'-1:00:00.000',b'00:00:01,000',b'00:00:01.00x',b'00:00:01.00\xff',b'bad\n'])
def test_strict_invalid_header_is_unplaceable_and_stops_forecasting(tmp_path,header):
    prefix=line(1000,{'1':follower()}).encode()
    path=tmp_path/'raw';path.write_bytes(prefix+header+b'{}\n'+line(3000,{'1':follower('.1')}).encode())
    cursor=p.GapCursor(path)
    before=cursor.query('1',cutoff_ns=1_000_000_000)
    assert before['processed_packet_count']==0 and not before['flags']['source_parse_gap_or_unknown_clock']
    reached=cursor.query('1',cutoff_ns=2_000_000_000)
    assert reached['processed_packet_count']==1 and reached['flags']['source_parse_gap_or_unknown_clock']
    boundary=reached['unplaceable_timestamp_boundary']
    assert boundary['recorded_ms'] is None and boundary['available_ms'] is None and boundary['sequence']==1
    later=cursor.query('1',cutoff_ns=9_000_000_000)
    assert later['processed_packet_count']==1 and later['fields']['IntervalToPositionAhead']['value']==.5
    assert later['unplaceable_timestamp_boundary']==boundary
    cursor.close()


def test_future_valid_header_shields_later_unknown_header_from_earlier_snapshot(tmp_path):
    prefix=line(1000,{'1':follower()}).encode()
    path=tmp_path/'raw';path.write_bytes(prefix+line(90000,{}).encode()+b'badheaderxxx{}\n')
    cursor=p.GapCursor(path)
    early=cursor.query('1',cutoff_ns=2_000_000_000)
    assert early['race_order_gap_ready'] and not early['flags']['source_parse_gap_or_unknown_clock']
    assert cursor.header_cursor.file.tell()==len(prefix)+12
    reached=cursor.query('1',cutoff_ns=91_000_000_000)
    assert reached['flags']['source_parse_gap_or_unknown_clock'] and reached['processed_packet_count']==2
    cursor.close()


def test_full_inventory_continues_after_unknown_header_without_changing_forecasts(tmp_path):
    path=tmp_path/'raw';path.write_bytes(line(1000,{'1':follower()}).encode()+b'badheaderxxx{}\n'+
        line(3000,{'1':{'GapToLeader':'+1 LAP'}}).encode()+b'00:00:04.000\xff\n')
    cursor=p.GapCursor(path);snapshot=cursor.query('1',cutoff_ns=5_000_000_000)
    inventory=p.inventory_stream(path)
    assert inventory['counts']['packets']==4 and inventory['counts']['invalid_timestamp_headers']==1
    assert inventory['payload_errors']=={'UnicodeDecodeError':1}
    assert inventory['raw_update_inventory']['GapToLeader/string/string/lap_deficit']==1
    assert cursor.query('1',cutoff_ns=5_000_000_000)==snapshot
    cursor.close()


def metadata(instant):
    return {'event_key':202201,'driver_id':'1','issued_after_lap_number':4,'issued_at_timestamp':instant/1e9,
        'issued_at_ns':instant,'issuance_id':f'202201/1/4/{instant}'}


def test_original_issuances_are_retained_without_reading_targets(tmp_path):
    path=tmp_path/'issues.jsonl';rows=[metadata(1_000_000_000),metadata(2_000_000_000)]
    path.write_text('\n'.join(json.dumps({**row,'unused_past_feature':7}) for row in rows)+'\n')
    assert p.issuance_metadata(path,202201,2)==rows
    with pytest.raises(ValueError,match='population changed'):p.issuance_metadata(path,202201,1)
    path.write_text(json.dumps({**rows[0],'outcome_status':'unmatched'})+'\n')
    with pytest.raises(ValueError,match='Target fields'):p.issuance_metadata(path,202201,1)


def test_no_historical_execution_before_design_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'source_files',lambda:pytest.fail('historical work before lock'))
    with pytest.raises(FileNotFoundError):p.run(tmp_path)
    assert not list(tmp_path.iterdir())


def test_spec_limits_two_races_and_preserves_no_model_scope():
    spec=p.read(p.HERE/'specification.json');p.validate_spec(spec)
    assert spec['expected_rows_both_lags']==3454 and 'model fitting' in spec['forbidden']
    source=Path(p.__file__).read_text()
    assert '.fit(' not in source and 'read_csv(' not in source and 'requests.' not in source


def test_one_leading_bom_preserves_header_clocks_packet_sequences_and_inventory(tmp_path):
    payload=(line(1000,{'1':follower()})+line(3000,{'1':follower('.3')})).encode()
    ordinary=tmp_path/'ordinary';ordinary.write_bytes(payload)
    marked=tmp_path/'marked';marked.write_bytes(b'\xef\xbb\xbf'+payload)
    a=p.GapCursor(ordinary);b=p.GapCursor(marked)
    for cutoff in (1_000_000_000,2_000_000_000,4_000_000_000):
        assert a.query('1',cutoff_ns=cutoff)==b.query('1',cutoff_ns=cutoff)
    assert p.inventory_stream(ordinary)==p.inventory_stream(marked)
    a.close();b.close()


@pytest.mark.parametrize('leading_bom',[b'',b'\xef\xbb\xbf'])
def test_bom_handling_does_not_read_or_decode_first_future_payload(tmp_path,leading_bom):
    path=tmp_path/'raw';path.write_bytes(leading_bom+b'00:01:30.000\xff\n')
    cursor=p.GapCursor(path)
    before=cursor.query('1',cutoff_ns=2_000_000_000)
    assert before['processed_packet_count']==0 and not before['flags']['source_parse_gap_or_unknown_clock']
    assert cursor.header_cursor.file.tell()==len(leading_bom)+12
    after=cursor.query('1',cutoff_ns=91_000_000_000)
    assert after['processed_packet_count']==1 and after['flags']['source_parse_gap_or_unknown_clock']
    cursor.close()


@pytest.mark.parametrize('prefix',[b'\xef\xbb\xbf',b'',b'\n'])
def test_midstream_bom_is_not_removed(tmp_path,prefix):
    path=tmp_path/'raw';path.write_bytes(prefix+line(1000,{'1':follower()}).encode()+
        b'\xef\xbb\xbf'+line(3000,{'1':follower('.1')}).encode())
    cursor=p.GapCursor(path);before=cursor.query('1',cutoff_ns=1_000_000_000)
    assert before['processed_packet_count']==0 and not before['flags']['source_parse_gap_or_unknown_clock']
    after=cursor.query('1',cutoff_ns=4_000_000_000)
    assert after['processed_packet_count']==1 and after['flags']['source_parse_gap_or_unknown_clock']
    assert after['fields']['IntervalToPositionAhead']['value']==.5
    assert after['unplaceable_timestamp_boundary']['available_ms'] is None
    cursor.close()


@pytest.mark.parametrize('prefix',[b'\xef\xbb\xbf\xef\xbb\xbf',b'\n\xef\xbb\xbf'])
def test_repeated_or_noninitial_bom_stops_without_guessed_availability(tmp_path,prefix):
    path=tmp_path/'raw';path.write_bytes(prefix+line(1000,{'1':follower()}).encode())
    cursor=p.GapCursor(path);value=cursor.query('1',cutoff_ns=2_000_000_000)
    assert value['processed_packet_count']==0 and value['flags']['source_parse_gap_or_unknown_clock']
    assert value['unplaceable_timestamp_boundary']['available_ms'] is None
    cursor.close()


def test_v2_only_adds_bom_handling_to_executable_parser_source():
    original=p.ROOT/'research/experiments/boundary_20260908/rival_gap_pilot/pilot.py'
    patch="""            # Exactly one encoding marker is allowed at absolute byte zero.
            # Complete only the timestamp prefix, never the future payload.
            if sequence==0 and prefix.startswith(b'\\xef\\xbb\\xbf'):
                prefix=prefix[3:]
                if not prefix.endswith(b'\\n'):prefix+=self.file.readline(12-len(prefix))
"""
    modified=Path(p.__file__).read_text()
    assert patch in modified
    restored=modified.replace(patch,'').replace("OUT=ROOT/'artifacts/research/boundary_20260908/rival_gap_pilot_v2'", "OUT=ROOT/'artifacts/research/boundary_20260908/rival_gap_pilot'")
    assert restored==original.read_text()
