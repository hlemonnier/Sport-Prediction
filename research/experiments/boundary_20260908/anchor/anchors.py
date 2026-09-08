"""Unclipped observed-history anchors with an explicit causal reset boundary."""
from collections import deque
import numpy as np
import pandas as pd

KEYS = ['event_key', 'driver_id', 'issued_after_lap_number', 'issued_at_timestamp']
EXTRA_FEATURES = [f'a_{feature}{window}' for window in [3, 5] for feature in ['gap', 'count', 'mad']]


def numeric(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError): return np.nan


def driver_key(value):
    x = numeric(value)
    return str(int(x)) if np.isfinite(x) and x.is_integer() else str(value).strip()


def available_pit(value, timestamp):
    x = numeric(value)
    return bool(np.isfinite(x) and x <= timestamp)


def observed_anchors(raw, event_key):
    work = raw.copy()
    work['_driver'] = work.DriverNumber.map(driver_key)
    work['Time'] = pd.to_numeric(work.Time, errors='raise')
    if not np.isfinite(work.Time).all(): raise ValueError('finite completed-lap clocks required')
    work = work.sort_values(['Time', '_driver', 'LapNumber'], kind='stable')
    states, records = {}, []
    for row in work.to_dict('records'):
        d = row['_driver']; time = float(row['Time']); lap = int(row['LapNumber'])
        s = states.setdefault(d, {'stint': np.nan, 'compound': '', 'generation': 0,
                                   'history': deque(maxlen=5), 'last_lap': 0, 'time': -np.inf})
        if lap <= s['last_lap'] or time <= s['time']: raise ValueError('driver chronology is not strictly increasing')
        stint = numeric(row.get('Stint'))
        compound = str(row.get('Compound', '')).strip().upper()
        known_compound = compound not in {'', 'UNKNOWN', 'NONE', 'NAN'}
        pit_out = available_pit(row.get('PitOutTime'), time)
        pit_in = available_pit(row.get('PitInTime'), time)
        reset = (s['generation'] == 0 or pit_out
                 or (known_compound and s['compound'] and compound != s['compound'])
                 or (np.isfinite(stint) and np.isfinite(s['stint']) and stint != s['stint']))
        if reset:
            s['history'].clear(); s['generation'] += 1
            if not np.isfinite(stint): s['stint'] = np.nan
        if np.isfinite(stint): s['stint'] = stint
        if known_compound: s['compound'] = compound
        s['last_lap'], s['time'] = lap, time
        y = numeric(row.get('LapTime'))
        eligible = (np.isfinite(y) and y > 0 and str(row.get('IsAccurate', False)).strip().lower() in {'true', '1', '1.0'}
                    and not (pit_in or pit_out) and not any(c in str(row.get('TrackStatus', '')) for c in '4567'))
        if not eligible: continue
        s['history'].append({'seconds': y, 'time': time})
        record = dict(event_key=int(event_key), driver_id=d, issued_after_lap_number=lap,
                      issued_at_timestamp=time, a_observed_seconds=y, a_generation=s['generation'],
                      a_evidence_max_timestamp=time)
        for window in [3, 5]:
            recent = list(s['history'])[-window:]
            values = np.array([x['seconds'] for x in recent]); median = float(np.median(values))
            record.update({f'a_median{window}_seconds': median, f'a_gap{window}': median - y,
                           f'a_count{window}': len(values), f'a_mad{window}': float(np.median(np.abs(values - median))),
                           f'a_evidence_min_timestamp{window}': recent[0]['time']})
        records.append(record)
    return pd.DataFrame(records)


def points(frame, residual, baseline, config):
    window = config['window']; anchor = frame[f'a_median{window}_seconds'].to_numpy()
    candidate = anchor + np.asarray(residual)
    valid = np.isfinite(candidate) & (candidate > 0)
    gate = np.ones(len(frame), dtype=bool)
    if config['policy'] == 'gate':
        gate = (frame[f'a_count{window}'].to_numpy() >= 3) & (np.abs(frame[f'a_gap{window}'].to_numpy()) >= 1)
    elif config['policy'] != 'full': raise ValueError('unknown anchor policy')
    predicted = np.where(gate & valid, candidate, np.asarray(baseline))
    assert np.array_equal(predicted[~gate], np.asarray(baseline)[~gate])
    return predicted, {'gate_rows': int(gate.sum()), 'invalid_point_fallbacks': int((~valid & gate).sum())}
