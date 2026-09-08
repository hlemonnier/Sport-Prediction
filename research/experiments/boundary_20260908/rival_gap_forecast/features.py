"""Causal race-order gap magnitudes and explicitly reset interval histories.

No outcome, canonical lap index, source UTC, future payload or inferred rival.
Suggested commit: research(f1-live): encode causal rival-gap magnitude and dynamics
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from research.experiments.boundary_20260908.rival_gap_pilot_v2 import pilot

ROOT = Path(__file__).resolve().parents[4]
CONTENT_NAMES = ('rank', 'leader_gap_seconds', 'interval_seconds', 'leader_lap_deficit',
    'is_leader', 'interval_under_1s', 'interval_under_2s', 'interval_under_5s',
    'interval_change_5s', 'interval_change_15s', 'interval_trend_30s', 'interval_range_30s')
QUALITY_NAMES = ('position_observed', 'position_valid', 'position_fresh', 'position_age_seconds',
    'leader_gap_observed', 'leader_gap_valid', 'leader_gap_fresh', 'leader_gap_age_seconds',
    'interval_observed', 'interval_valid', 'interval_fresh', 'interval_age_seconds',
    'leader_gap_category_seconds', 'leader_gap_category_leader', 'leader_gap_category_lap_deficit', 'leader_gap_category_other',
    'interval_category_seconds', 'interval_category_leader', 'interval_category_lap_deficit', 'interval_category_other',
    'quality_unknown_or_invalid_rank', 'quality_duplicate_reported_rank', 'quality_interval_predates_last_rank_change',
    'quality_leader_rank_category_inconsistent', 'quality_latest_field_timestamp_regressed',
    'quality_source_parse_gap_or_unknown_clock', 'quality_missing_clear_invalid_or_stale_gap', 'quality_supported',
    'history_delta5_available', 'history_delta15_available', 'history_trend_available', 'history_count', 'history_span_seconds')
FEATURE_NAMES = CONTENT_NAMES + QUALITY_NAMES
CONTENT_INDICES = tuple(range(12))
QUALITY_INDICES = tuple(range(12, 45))
NS = 1_000_000_000
WINDOW_NS = 30*NS
LEADERS = frozenset(('leader', 'leader_lap_counter'))
GAP_CATEGORIES = LEADERS | {'seconds', 'lap_deficit'}
PILOT_FLAGS = tuple(name.removeprefix('quality_') for name in QUALITY_NAMES[20:27])
_DEPENDENCIES = {
    'research/experiments/boundary_20260908/rival_gap_pilot/pilot.py':
        'b45b9b745ac2ed9e5797ed20861bbe360e649c074d95be1c966dfd84bcd0f6f7',
    'research/experiments/boundary_20260908/rival_gap_pilot_v2/pilot.py':
        'a05834853b2c588fccd851e91cb300044c46b6b50a49f8a560e261fced59fb34'}


def dependency_bindings():
    """Bind both preserved parser versions, although only v2 is executed."""
    for path, expected in _DEPENDENCIES.items():
        if pilot.sha(ROOT/path) != expected: raise ValueError('Frozen gap parser changed: '+path)
    return dict(_DEPENDENCIES)


@dataclass(frozen=True)
class GapFeatureSnapshot:
    gap_values: np.ndarray
    gap_supported: bool
    gap_cutoff_ns: int
    provenance: dict

    def as_dict(self):
        return {'gap_values': [None if math.isnan(value) else float(value) for value in self.gap_values],
            'gap_supported': self.gap_supported, 'gap_cutoff_ns': self.gap_cutoff_ns,
            'provenance': deepcopy(self.provenance)}


def fresh(field):
    age = field['age_seconds']
    return age is not None and age <= 10 and not field.get('source_regressed', False)


def _quality(field, position=False):
    observed = 'sequence' in field
    valid = field['category'] == 'rank' if position else field['category'] in GAP_CATEGORIES
    age = min(max(field['age_seconds'], 0.), 180.) if observed else 180.
    return [float(observed), float(valid), float(fresh(field)), float(age)]


def _category(field):
    value = field['category']
    bits = [value == 'seconds', value in LEADERS, value == 'lap_deficit']
    return [float(x) for x in (*bits, not any(bits))]


def _slope(history):
    """OLS seconds/second, centered in time and scaled before summation."""
    end = history[-1]['available_ns']
    x = [(row['available_ns']-end)/NS for row in history]
    y = [row['seconds'] for row in history]
    scale = max(y)
    if scale == 0: return 0.
    unit = [value/scale for value in y]
    center_x, center_y = math.fsum(x)/len(x), math.fsum(unit)/len(unit)
    dx = [value-center_x for value in x]
    denominator = math.fsum(value*value for value in dx)
    if denominator <= 0: raise ValueError('OLS requires distinct observation times')
    slope = math.fsum(a*(b-center_y) for a,b in zip(dx,unit))/denominator
    if abs(slope) > 2./scale: return math.copysign(2., slope)
    return min(max(slope*scale, -2.), 2.)


def history_content(history, interval, rank):
    """No interpolation or history-only issuance gate; exact current identity."""
    output = [math.nan]*4; flags = [False]*3
    matches = bool(history and rank is not None and rank > 1 and fresh(interval)
        and interval['category'] == 'seconds'
        and interval.get('sequence') == history[-1]['sequence']
        and interval.get('available_ms', -1)*1_000_000 == history[-1]['available_ns']
        and interval['value'] == history[-1]['seconds'])
    if not matches: return output, flags
    newest = history[-1]
    for i, seconds in enumerate((5, 15)):
        previous = next((row for row in reversed(history)
                         if row['available_ns'] <= newest['available_ns']-seconds*NS), None)
        if previous is not None:
            output[i] = min(max(newest['seconds']-previous['seconds'], -30.), 30.); flags[i] = True
    if len(history) >= 3 and newest['available_ns']-history[0]['available_ns'] >= 15*NS:
        output[2] = _slope(history); flags[2] = True
    output[3] = min(max(row['seconds'] for row in history)-min(row['seconds'] for row in history), 60.)
    return output, flags


class GapFeatureCursor(pilot.GapCursor):
    """Frozen atomic packet parser plus own-driver histories at its exact clock."""
    def __init__(self, path):
        dependency_bindings()
        super().__init__(path)
        self.histories = {}; self.history_resets = {}

    def _prune(self, driver, cutoff_ns):
        history = self.histories.setdefault(driver, deque())
        while history and history[0]['available_ns'] < cutoff_ns-WINDOW_NS: history.popleft()
        return history

    def _reset(self, driver, record, reasons):
        self.histories.setdefault(driver, deque()).clear()
        previous = self.history_resets.get(driver, {})
        self.history_resets[driver] = {'sequence': record['sequence'], 'available_ns': record['available_ms']*1_000_000,
            'recorded_ms': record['recorded_ms'], 'reasons': reasons, 'count': previous.get('count', 0)+1}

    def _apply(self, record):
        lines = record['payload'].get('Lines', {}) if not record['error'] else {}
        before = {}
        if isinstance(lines, dict):
            for driver, patch in lines.items():
                if isinstance(driver, str) and isinstance(patch, dict):
                    field = self.states.get(driver, {}).get('fields', {}).get('Position', {})
                    before[driver] = (field.get('category'), field.get('value'))
        super()._apply(record)
        # The complete original packet has already been applied to every driver.
        for driver, old_rank in before.items():
            fields = self.states[driver]['fields']; position = fields.get('Position', {})
            interval = fields.get('IntervalToPositionAhead', {})
            reasons = []
            if old_rank != (position.get('category'), position.get('value')): reasons.append('position_changed')
            updated = interval.get('sequence') == record['sequence']
            if updated and interval['category'] != 'seconds': reasons.append('interval_non_numeric')
            if updated and interval['category'] == 'seconds' and record['regressed']: reasons.append('interval_timestamp_regressed')
            if reasons: self._reset(driver, record, reasons)
            if (not updated or interval['category'] != 'seconds' or record['regressed']
                    or position.get('category') != 'rank' or position['value'] <= 1): continue
            now = record['available_ms']*1_000_000
            history = self._prune(driver, now+1)
            observation = {'sequence': record['sequence'], 'available_ns': now,
                'recorded_ms': record['recorded_ms'], 'seconds': float(interval['value']), 'source_regressed': False}
            if history and history[-1]['available_ns'] == now: history[-1] = observation
            else: history.append(observation)

    def query(self, driver, *, cutoff_ns):
        if not isinstance(driver, str) or not driver: raise ValueError('Driver must be a nonempty string')
        raw = super().query(driver, cutoff_ns=cutoff_ns)
        history = list(self._prune(driver, cutoff_ns))
        fields = raw['fields']; position, gap, interval = (fields[name] for name in pilot.FIELDS)
        rank = position['value'] if position['category'] == 'rank' else None
        leader = rank == 1 and not raw['flags']['leader_rank_category_inconsistent']
        numeric_gap = fresh(gap) and gap['category'] == 'seconds'
        numeric_interval = fresh(interval) and interval['category'] == 'seconds'
        leader_marker = fresh(gap) and gap['category'] in LEADERS and leader
        gap_seconds = min(gap['value'], 180.) if numeric_gap else 0. if leader_marker else math.nan
        lap_deficit = (float(min(gap['value'], 20)) if fresh(gap) and gap['category'] == 'lap_deficit'
                       else 0. if numeric_gap or leader_marker else math.nan)
        content = [float(rank) if rank is not None else math.nan, gap_seconds,
            min(interval['value'], 60.) if numeric_interval else math.nan, lap_deficit,
            float(rank == 1) if rank is not None else math.nan,
            *[float(interval['value'] < limit) if numeric_interval else math.nan for limit in (1, 2, 5)]]
        dynamics, flags = history_content(history, interval, rank)
        content += dynamics
        quality = (_quality(position, True)+_quality(gap)+_quality(interval)+_category(gap)+_category(interval)
            +[float(raw['flags'][name]) for name in PILOT_FLAGS]+[float(raw['race_order_gap_ready'])]
            +[float(value) for value in flags]+[float(min(len(history), 128)),
                (history[-1]['available_ns']-history[0]['available_ns'])/NS if history else 0.])
        values = np.asarray(content+quality, dtype=np.float64)
        if values.shape != (45,) or np.isinf(values).any() or not np.isfinite(values[12:]).all():
            raise AssertionError('Invalid fixed gap encoding')
        values.setflags(write=False)
        return GapFeatureSnapshot(values, raw['race_order_gap_ready'], cutoff_ns,
            {'driver_id': driver, 'clock': 'physical_archive_cummax_milliseconds_not_certified_client_receipt',
             'pilot_snapshot': deepcopy(raw), 'retained_interval_observations': deepcopy(history),
             'last_history_reset': deepcopy(self.history_resets.get(driver)),
             'history_rule': 'cutoff_minus30s_inclusive_to_cutoff_exclusive;latest_sequence_per_availability;follower_nonregressed_only'})

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
