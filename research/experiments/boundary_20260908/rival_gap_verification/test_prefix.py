"""Independent history arithmetic checks, using constructed packet histories."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from .prefix import PrefixReplay, history_content


def stream(tmp_path, updates):
    path = tmp_path/'timing.stream'
    lines = []
    for seconds, patch in updates:
        lines.append(f'00:00:{seconds:06.3f}'+json.dumps({'Lines':{'2':patch}})+'\n')
    path.write_bytes(b'\xef\xbb\xbf'+''.join(lines).encode())
    return path


def update(value, **fields):
    return {'IntervalToPositionAhead':{'Value':value}, **fields}


def test_independent_decimal_slope_and_physical_time_deltas(tmp_path):
    path = stream(tmp_path, [(0,update('3',Position='2',GapToLeader='5')),
                             (5,update('4')),(15,update('2')),(25,update('1'))])
    cursor = PrefixReplay(path)
    try:
        result = cursor.reconstruct('2',cutoff_ns=26_000_000_000)
    finally:
        cursor.close()
    values = result['values']
    assert result['supported'] is True
    assert math.isnan(values[1])  # Leader gap is stale; interval is fresh.
    assert values[2] == 1 and values[5:8] == [0,1,1]
    np.testing.assert_allclose(values[8:12],[-1,-3,-37.5/368.75,3],rtol=0,atol=1e-14)
    assert values[-5:] == [1,1,1,4,25]


def test_equal_cutoff_payload_excluded_and_future_invalid_utf8_untouched(tmp_path):
    path = stream(tmp_path,[(0,update('3',Position='2',GapToLeader='5')),(5,update('4'))])
    with path.open('ab') as f:
        f.write(b'00:00:15.000\xff\n')
    cursor = PrefixReplay(path)
    try:
        result = cursor.reconstruct('2',cutoff_ns=15_000_000_000)
        assert result['supported'] is True and result['values'][2] == 4
        assert result['values'][8] == 1
        after = cursor.reconstruct('2',cutoff_ns=15_000_000_001)
        assert after['supported'] is False
    finally:
        cursor.close()


def test_same_clock_has_only_last_physical_update(tmp_path):
    path = stream(tmp_path,[(0,update('3',Position='2')),(5,update('4')),(5,update('5'))])
    cursor = PrefixReplay(path)
    try:
        result = cursor.reconstruct('2',cutoff_ns=6_000_000_000)
    finally:
        cursor.close()
    assert result['history'] == [(0,0,3.0),(2,5000,5.0)]
    assert result['values'][8] == 2 and result['values'][-2] == 2


@pytest.mark.parametrize('barrier',[{'Position':'3'},update(''),update(None),update('1 LAP')])
def test_rank_or_explicit_nonnumeric_update_resets_history(tmp_path,barrier):
    path = stream(tmp_path,[(0,update('3',Position='2')),(5,update('4')),(10,barrier),(15,update('2'))])
    cursor = PrefixReplay(path)
    try:
        result = cursor.reconstruct('2',cutoff_ns=16_000_000_000)
    finally:
        cursor.close()
    assert len(result['history']) == 1 and math.isnan(result['values'][8])


def test_metadata_does_not_refresh_or_reset_history(tmp_path):
    path = stream(tmp_path,[(0,update('3',Position='2')),(5,update('4')),
                             (10,{'IntervalToPositionAhead':{'Catching':True}})])
    cursor = PrefixReplay(path)
    try:
        result = cursor.reconstruct('2',cutoff_ns=11_000_000_000)
    finally:
        cursor.close()
    assert result['history'][-1] == (1,5000,4.0) and result['values'][8] == 1


def test_regressed_numeric_update_resets_without_append(tmp_path):
    path = stream(tmp_path,[(0,update('3',Position='2')),(10,update('4')),(8,update('2')),(20,update('1'))])
    cursor = PrefixReplay(path)
    try:
        result = cursor.reconstruct('2',cutoff_ns=21_000_000_000)
    finally:
        cursor.close()
    assert result['history'] == [(3,20000,1.0)]


def test_decimal_extreme_finite_values_and_stale_history():
    h = [(0,0,0.0),(1,10000,1e308),(2,20000,1e308)]
    current = dict(category='seconds',source_regressed=False,age_seconds=1,
                   sequence=2,available_ms=20000,value=1e308)
    values,quality,kept = history_content(h,current,2,21_000_000_000)
    assert values == [0,30,2,60] and quality == [1,1,1,3,20]
    current['age_seconds'] = 11
    values,quality,kept = history_content(h,current,2,31_000_000_000)
    assert all(math.isnan(v) for v in values)
    assert quality == [0,0,0,2,10]
