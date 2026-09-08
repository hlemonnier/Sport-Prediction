"""Synthetic gap math and prefix tests; no historical construction or fitting.

Suggested commit: test(f1-live): verify causal rival-gap encoding and resets
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from . import features as f
from research.experiments.boundary_20260908.rival_gap_pilot_v2 import pilot


def packet(ms, lines):
    seconds, fraction = divmod(ms, 1000)
    return (f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}.{fraction:03}'
            +json.dumps({'Lines': lines})+'\n').encode()


def follower(interval='1.5', rank='2', gap='10'):
    return {'Position': rank, 'GapToLeader': gap, 'IntervalToPositionAhead': {'Value': interval}}


def stream(tmp_path, packets, name='raw'):
    path = tmp_path/name; path.write_bytes(b''.join(packets)); return path


def values(snapshot): return dict(zip(f.FEATURE_NAMES, snapshot.gap_values))


def assert_pilot_parity(path, queries):
    with f.GapFeatureCursor(path) as actual:
        expected = pilot.GapCursor(path)
        try:
            for driver, cutoff in queries:
                result = actual.query(driver, cutoff_ns=cutoff); original = expected.query(driver, cutoff_ns=cutoff)
                assert result.gap_supported is original['race_order_gap_ready']
                assert result.provenance['pilot_snapshot'] == original
                assert result.gap_values.shape == (45,) and not np.isinf(result.gap_values).any()
                assert np.isfinite(result.gap_values[12:]).all()
        finally: expected.close()


def test_exact_specification_order_and_both_frozen_parser_bindings():
    spec = json.loads((Path(f.__file__).parent/'specification.json').read_text())
    assert list(f.FEATURE_NAMES) == spec['features']['feature_order']
    assert list(f.CONTENT_NAMES) == spec['features']['content_names']
    assert list(f.QUALITY_NAMES) == spec['features']['quality_names']
    assert f.CONTENT_INDICES == tuple(range(12)) and f.QUALITY_INDICES == tuple(range(12,45))
    assert len(set(f.FEATURE_NAMES)) == 45 and len(f.dependency_bindings()) == 2
    assert '/rival_gap_pilot/pilot.py' in next(iter(f.dependency_bindings()))


def test_source_drift_refused_before_raw_file_open(monkeypatch):
    monkeypatch.setattr(pilot, 'sha', lambda path:'changed')
    with pytest.raises(ValueError, match='Frozen gap parser changed'): f.GapFeatureCursor('does-not-exist')


def test_empty_encoding_is_nan_content_with_explicit_quality_and_strict_json(tmp_path):
    path = stream(tmp_path, [])
    with f.GapFeatureCursor(path) as cursor: snapshot = cursor.query('1', cutoff_ns=-1)
    got = values(snapshot)
    assert np.isnan(snapshot.gap_values[:12]).all() and not snapshot.gap_supported
    for name in ('position', 'leader_gap', 'interval'):
        assert got[name+'_observed'] == got[name+'_valid'] == got[name+'_fresh'] == 0
        assert got[name+'_age_seconds'] == 180
    assert got['leader_gap_category_other'] == got['interval_category_other'] == 1
    assert json.loads(json.dumps(snapshot.as_dict(), allow_nan=False))['gap_values'][:12] == [None]*12


def test_static_units_strict_thresholds_clipping_and_validity(tmp_path):
    path = stream(tmp_path, [packet(1000, {'1': follower('2', '20', '250')})])
    with f.GapFeatureCursor(path) as cursor: result = cursor.query('1', cutoff_ns=2*f.NS)
    got = values(result)
    assert [got[x] for x in f.CONTENT_NAMES[:8]] == [20,180,2,0,0,0,0,1]
    assert [got[x] for x in ('position_observed','position_valid','position_fresh','position_age_seconds')] == [1,1,1,1]
    assert result.gap_supported and got['history_count'] == 1
    assert math.isnan(got['interval_change_5s']) and got['interval_range_30s'] == 0


@pytest.mark.parametrize('raw', [None, '', '--', 'banana', True, float('nan'), float('inf')])
def test_clear_invalid_values_are_never_fabricated_zero(raw, tmp_path):
    path = stream(tmp_path, [packet(1000, {'1': follower(raw)})])
    with f.GapFeatureCursor(path) as cursor: snapshot = cursor.query('1', cutoff_ns=2*f.NS)
    got = values(snapshot)
    assert math.isnan(got['interval_seconds']) and math.isnan(got['interval_under_5s'])
    assert got['interval_observed'] == got['interval_fresh'] == 1 and got['interval_valid'] == 0
    assert got['interval_category_other'] == 1 and not snapshot.gap_supported
    json.dumps(snapshot.as_dict(), allow_nan=False)


@pytest.mark.parametrize('marker', ['LEADER', 'LAP 57', '0'])
def test_consistent_leader_has_zero_leader_gap_but_no_fake_interval(marker, tmp_path):
    path = stream(tmp_path, [packet(1000, {'1': follower(None, '1', marker)})])
    with f.GapFeatureCursor(path) as cursor: snapshot = cursor.query('1', cutoff_ns=2*f.NS)
    got = values(snapshot)
    assert snapshot.gap_supported and got['is_leader'] == 1 and got['leader_gap_seconds'] == got['leader_lap_deficit'] == 0
    assert math.isnan(got['interval_seconds']) and got['history_count'] == 0


def test_inconsistent_leader_marker_does_not_certify_zero(tmp_path):
    path = stream(tmp_path, [packet(1000, {'1': follower('.5', '1', 'LEADER')})])
    with f.GapFeatureCursor(path) as cursor: snapshot = cursor.query('1', cutoff_ns=2*f.NS)
    got = values(snapshot)
    assert got['quality_leader_rank_category_inconsistent'] == 1 and not snapshot.gap_supported
    assert math.isnan(got['leader_gap_seconds']) and math.isnan(got['leader_lap_deficit'])


def test_lap_deficit_is_not_seconds_or_a_leader_lap_counter(tmp_path):
    path = stream(tmp_path, [packet(1000, {'1': follower('1 LAP', '5', '40 LAPS')})])
    with f.GapFeatureCursor(path) as cursor: snapshot = cursor.query('1', cutoff_ns=2*f.NS)
    got = values(snapshot)
    assert got['leader_lap_deficit'] == 20 and math.isnan(got['leader_gap_seconds']) and math.isnan(got['interval_seconds'])
    assert got['leader_gap_category_lap_deficit'] == got['interval_category_lap_deficit'] == 1
    assert got['interval_valid'] == 1 and not snapshot.gap_supported


def test_exact_ns_strict_packet_admission_and_ten_second_freshness(tmp_path):
    path = stream(tmp_path, [packet(1000, {'1': follower()})])
    with f.GapFeatureCursor(path) as cursor:
        assert math.isnan(values(cursor.query('1',cutoff_ns=f.NS))['rank'])
        assert cursor.query('1',cutoff_ns=f.NS+1).gap_supported
        assert cursor.query('1',cutoff_ns=11*f.NS).gap_supported
        later = cursor.query('1',cutoff_ns=11*f.NS+1)
    got = values(later)
    assert not later.gap_supported and math.isnan(got['interval_seconds']) and math.isnan(got['leader_gap_seconds'])
    assert got['rank'] == 2 and got['is_leader'] == 0 and got['position_fresh'] == 0
    assert got['history_count'] == 1 and got['history_delta5_available'] == 0
    assert np.isnan(later.gap_values[8:12]).all()


@pytest.mark.parametrize('cutoff', [True, 1., np.int64(1)])
def test_cutoff_requires_exact_python_integer(cutoff, tmp_path):
    with f.GapFeatureCursor(stream(tmp_path, [])) as cursor:
        with pytest.raises(ValueError): cursor.query('1', cutoff_ns=cutoff)


def test_global_cutoff_is_monotone_and_driver_type_explicit(tmp_path):
    with f.GapFeatureCursor(stream(tmp_path, [])) as cursor:
        cursor.query('1',cutoff_ns=10)
        with pytest.raises(ValueError): cursor.query('2',cutoff_ns=9)
        with pytest.raises(ValueError): cursor.query(1,cutoff_ns=10)


def test_deltas_reference_newest_observation_not_query_time_and_ols_units(tmp_path):
    path=stream(tmp_path,[packet(0,{'1':follower('20')}),packet(10000,{'1':{'IntervalToPositionAhead':'10'}}),
                          packet(20000,{'1':{'IntervalToPositionAhead':'3'}})])
    with f.GapFeatureCursor(path) as cursor: got=values(cursor.query('1',cutoff_ns=29*f.NS))
    assert got['interval_change_5s'] == -7 and got['interval_change_15s'] == -17
    assert got['interval_trend_30s'] == pytest.approx(-.85) and got['interval_range_30s'] == 17
    assert got['history_delta5_available'] == got['history_delta15_available'] == got['history_trend_available'] == 1


def test_history_uses_raw_seconds_before_final_feature_clipping(tmp_path):
    path=stream(tmp_path,[packet(t*1000,{'1':follower(str(y))}) for t,y in [(0,0),(10,100),(20,200)]])
    with f.GapFeatureCursor(path) as cursor: got=values(cursor.query('1',cutoff_ns=21*f.NS))
    assert got['interval_seconds'] == 60 and got['interval_change_5s'] == got['interval_change_15s'] == 30
    assert got['interval_trend_30s'] == 2 and got['interval_range_30s'] == 60


def test_thirty_second_lower_boundary_is_inclusive_without_rounding(tmp_path):
    path=stream(tmp_path,[packet(t*1000,{'1':follower(str(t))}) for t in (0,5,15,25)])
    with f.GapFeatureCursor(path) as cursor:
        first=cursor.query('1',cutoff_ns=30*f.NS); second=cursor.query('1',cutoff_ns=30*f.NS+1)
    assert values(first)['history_count'] == 4 and values(first)['interval_range_30s'] == 25
    assert values(second)['history_count'] == 3 and values(second)['interval_range_30s'] == 20
    assert first.provenance['retained_interval_observations'][0]['available_ns'] == 0


def test_same_availability_revisions_collapse_to_latest_physical_sequence(tmp_path):
    path=stream(tmp_path,[packet(0,{'1':follower('4')}),packet(10000,{'1':{'IntervalToPositionAhead':'3'}}),
                          packet(10000,{'1':{'IntervalToPositionAhead':'2'}})])
    with f.GapFeatureCursor(path) as cursor: snapshot=cursor.query('1',cutoff_ns=10*f.NS+1)
    history=snapshot.provenance['retained_interval_observations']
    assert len(history) == 2 and history[-1]['sequence'] == 2 and history[-1]['seconds'] == 2
    assert values(snapshot)['interval_change_5s'] == -2 and values(snapshot)['history_trend_available'] == 0


@pytest.mark.parametrize('rank', ['3', None, '', 'invalid'])
def test_actual_rank_change_including_clear_invalid_resets_history(rank, tmp_path):
    path=stream(tmp_path,[packet(0,{'1':follower('4')}),packet(6000,{'1':{'IntervalToPositionAhead':'3'}}),
                          packet(8000,{'1':{'Position':rank}})])
    with f.GapFeatureCursor(path) as cursor: snapshot=cursor.query('1',cutoff_ns=9*f.NS)
    assert values(snapshot)['history_count'] == 0 and not snapshot.gap_supported
    assert snapshot.provenance['last_history_reset']['reasons'] == ['position_changed']


def test_same_rank_and_metadata_only_updates_do_not_refresh_or_reset_history(tmp_path):
    path=stream(tmp_path,[packet(0,{'1':follower('4')}),packet(6000,{'1':{'IntervalToPositionAhead':'3'}}),
        packet(8000,{'1':{'Position':'2.0','IntervalToPositionAhead':{'Catching':True}}})])
    with f.GapFeatureCursor(path) as cursor: snapshot=cursor.query('1',cutoff_ns=9*f.NS)
    got=values(snapshot)
    assert got['history_count'] == 2 and got['interval_age_seconds'] == 3 and snapshot.gap_supported
    assert snapshot.provenance['last_history_reset']['sequence'] == 0


@pytest.mark.parametrize('clear', [None,'','--','LEADER','LAP 4','1 LAP','invalid',True])
def test_explicit_non_numeric_interval_resets_and_clean_recovery_starts_new_history(clear, tmp_path):
    path=stream(tmp_path,[packet(0,{'1':follower('4')}),packet(6000,{'1':{'IntervalToPositionAhead':'3'}}),
        packet(8000,{'1':{'IntervalToPositionAhead':clear}}),packet(9000,{'1':{'IntervalToPositionAhead':'1'}})])
    with f.GapFeatureCursor(path) as cursor:
        reset=cursor.query('1',cutoff_ns=8*f.NS+1); recovered=cursor.query('1',cutoff_ns=10*f.NS)
    assert values(reset)['history_count'] == 0
    assert values(recovered)['history_count'] == 1 and math.isnan(values(recovered)['interval_change_5s'])


def test_regressed_numeric_update_resets_without_admission_then_recovers(tmp_path):
    path=stream(tmp_path,[packet(0,{'1':follower('4')}),packet(6000,{'1':{'IntervalToPositionAhead':'3'}}),
        packet(10000,{}),packet(8000,{'1':{'IntervalToPositionAhead':'2'}}),
        packet(12000,{'1':{'IntervalToPositionAhead':'1'}})])
    with f.GapFeatureCursor(path) as cursor:
        reset=cursor.query('1',cutoff_ns=11*f.NS); recovered=cursor.query('1',cutoff_ns=13*f.NS)
    assert values(reset)['history_count'] == 0 and not reset.gap_supported
    assert reset.provenance['last_history_reset']['reasons'] == ['interval_timestamp_regressed']
    assert recovered.gap_supported and values(recovered)['history_count'] == 1


def test_same_packet_rank_and_interval_apply_atomically_regardless_key_order(tmp_path):
    initial=packet(0,{'1':follower('4'),'2':follower('5','3')})
    changes={'1':follower('2','3'),'2':follower('1','2')}
    reverse={d:dict(reversed(list(p.items()))) for d,p in reversed(list(changes.items()))}
    a=stream(tmp_path,[initial,packet(5000,changes)],'a'); b=stream(tmp_path,[initial,packet(5000,reverse)],'b')
    with f.GapFeatureCursor(a) as x, f.GapFeatureCursor(b) as y:
        left=x.query('1',cutoff_ns=6*f.NS); right=y.query('1',cutoff_ns=6*f.NS)
    assert left.as_dict() == right.as_dict() and left.gap_supported
    assert values(left)['history_count'] == 1


@pytest.mark.parametrize('rank',['1',None,'bad'])
def test_no_history_for_leader_or_invalid_rank(rank,tmp_path):
    path=stream(tmp_path,[packet(t*1000,{'1':follower(str(t),rank)}) for t in (0,10,20)])
    with f.GapFeatureCursor(path) as cursor: snapshot=cursor.query('1',cutoff_ns=21*f.NS)
    assert values(snapshot)['history_count'] == 0 and np.isnan(snapshot.gap_values[8:12]).all()


def test_history_identity_requires_sequence_availability_and_raw_value():
    history=[{'sequence':1,'available_ns':10*f.NS,'seconds':2.}]
    field={'sequence':1,'available_ms':10000,'value':2.,'category':'seconds','age_seconds':1.,'source_regressed':False}
    assert f.history_content(history,field,2)[0][3] == 0
    for key,value in [('sequence',2),('available_ms',10001),('value',3.)]:
        changed={**field,key:value}; content,flags=f.history_content(history,changed,2)
        assert all(math.isnan(v) for v in content) and flags == [False]*3


def test_history_count_clips_only_diagnostic_not_actual_history(tmp_path):
    path=stream(tmp_path,[packet(i*100,{'1':follower(str(i/10))}) for i in range(200)])
    with f.GapFeatureCursor(path) as cursor: snapshot=cursor.query('1',cutoff_ns=20*f.NS)
    got=values(snapshot)
    assert got['history_count'] == 128 and len(snapshot.provenance['retained_interval_observations']) == 200
    assert got['history_span_seconds'] == pytest.approx(19.9) and got['interval_trend_30s'] == pytest.approx(1.)


@pytest.mark.parametrize('series', [[sys.float_info.max]*3,[0,sys.float_info.max/2,sys.float_info.max],
                                  [sys.float_info.max,sys.float_info.max/2,0],[5e-324]*3,[0,5e-324,1e-323]])
def test_extreme_finite_observations_never_overflow_content(series,tmp_path):
    path=stream(tmp_path,[packet(t*1000,{'1':follower(y)}) for t,y in zip((0,10,20),series)])
    with f.GapFeatureCursor(path) as cursor: snapshot=cursor.query('1',cutoff_ns=21*f.NS)
    assert np.isfinite(snapshot.gap_values[8:12]).all() and not np.isinf(snapshot.gap_values).any()
    json.dumps(snapshot.as_dict(),allow_nan=False)
    if len(set(series)) == 1: assert values(snapshot)['interval_trend_30s'] == 0


@pytest.mark.parametrize('suffix',[b'\xff\n',b'{invalid}\n',b'x'*2000+b'\n'])
def test_future_payload_poisoning_is_not_read_or_included_in_provenance(suffix,tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'MAX_PAYLOAD_BYTES',1024)
    prefix=packet(1000,{'1':follower()})
    a=stream(tmp_path,[prefix],'a'); b=stream(tmp_path,[prefix,b'00:01:30.000'+suffix],'b')
    with f.GapFeatureCursor(a) as first, f.GapFeatureCursor(b) as second:
        left=first.query('1',cutoff_ns=2*f.NS); right=second.query('1',cutoff_ns=2*f.NS)
        assert left.as_dict() == right.as_dict()
        assert second.header_cursor.file.tell() == len(prefix)+12


def test_other_raw_fields_do_not_supply_targets_or_features(tmp_path):
    ordinary=follower(); poisoned={**ordinary,'LastLapTime':{'Value':'10000'},'Sectors':{'0':{'Value':'999'}},
                                 'LapTime':'future target poison','Utc':'2099-01-01','target':0}
    a=stream(tmp_path,[packet(1000,{'1':ordinary})],'a');b=stream(tmp_path,[packet(1000,{'1':poisoned})],'b')
    with f.GapFeatureCursor(a) as x,f.GapFeatureCursor(b) as y:
        assert x.query('1',cutoff_ns=2*f.NS).as_dict() == y.query('1',cutoff_ns=2*f.NS).as_dict()


def test_snapshot_mutation_cannot_change_future_state(tmp_path):
    path=stream(tmp_path,[packet(1000,{'1':follower()})])
    with f.GapFeatureCursor(path) as cursor:
        before=cursor.query('1',cutoff_ns=2*f.NS); expected=before.as_dict()
        with pytest.raises(ValueError): before.gap_values[0]=100
        before.provenance['retained_interval_observations'][0]['seconds']=999
        assert cursor.query('1',cutoff_ns=2*f.NS).as_dict() == expected


def test_bom_duplicate_rank_unknown_boundary_and_revisions_retain_exact_pilot_support(tmp_path):
    content=[b'\xef\xbb\xbf'+packet(0,{'1':follower(),'2':follower()}),
        packet(2000,{'2':{'Position':'3'}}),packet(5000,{'1':{'Position':'3'}}),
        packet(6000,{'1':{'IntervalToPositionAhead':'2'},'2':{'Position':'2'}}),
        packet(9000,{}),packet(8000,{'1':{'IntervalToPositionAhead':'1'}}),
        packet(10000,{'1':{'IntervalToPositionAhead':'0.5'}}),b'invalidclock{}\n']
    path=stream(tmp_path,content)
    assert_pilot_parity(path,[('1',ns) for ns in (0,1,3*f.NS,5*f.NS+1,7*f.NS,9*f.NS+1,11*f.NS,40*f.NS)])
