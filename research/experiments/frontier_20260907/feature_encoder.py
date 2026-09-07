"""Inference-only extraction copied from the frozen experimental encoder.

Only changes: require the observed schema, return issuances without future
matching, and support prefixes with zero issuances. Model features are identical.
"""
from collections import deque
import numpy as np
import pandas as pd
from research.experiments.frontier_20260907.live_frontier import (base, finite, KEYS, FEATURE_PREFIX, SECTORS, SPEEDS)

REQUIRED = ['DriverNumber','LapNumber','Time','LapTime','IsAccurate','Stint','Compound','TyreLife',
            'PitInTime','PitOutTime','TrackStatus',*SECTORS,*SPEEDS,'Position','FreshTyre']

def validate_schema(raw):
    missing=sorted(set(REQUIRED)-set(raw))
    if missing:
        raise ValueError('Observed-input schema is incomplete: '+', '.join(missing))
    for column in ['Time','LapTime','PitInTime','PitOutTime',*SECTORS,*SPEEDS,'Position','TyreLife']:
        if not pd.api.types.is_numeric_dtype(raw[column]):
            raise TypeError(column+' must have explicit numeric seconds or native speed/position units')

def observed_features(raw, event_key):
    validate_schema(raw)
    issued, _ = base.stream_event(raw, event_key)
    if issued.empty:
        return issued
    work = raw.copy()
    work['Time'] = pd.to_numeric(work.Time)
    work['_driver'] = work.DriverNumber.map(base.driver_key)
    work = work.sort_values(['Time', '_driver', 'LapNumber'], kind='mergesort')
    states, extra = {}, []
    issuance_keys = set(zip(issued.driver_id, issued.issued_after_lap_number, issued.issued_at_timestamp))
    for timestamp, batch in work.groupby('Time', sort=False):
        peers = {d: s['peer'] for d, s in states.items() if s['peer'] is not None}
        for row in batch.to_dict('records'):
            d, lap = row['_driver'], int(row['LapNumber'])
            s = states.setdefault(d, {'stint': np.nan, 'compound': '', 'history': deque(maxlen=12),
                                      'ewma': {}, 'peer': None, 'generation': 0})
            stint, compound = base.numeric(row.get('Stint')), str(row.get('Compound', 'UNKNOWN')).upper()
            pit_out = np.isfinite(base.numeric(row.get('PitOutTime')))
            pit_in = np.isfinite(base.numeric(row.get('PitInTime')))
            reset = s['generation'] == 0 or pit_out or (s['compound'] and compound != s['compound']) or (
                np.isfinite(stint) and np.isfinite(s['stint']) and stint != s['stint'])
            if reset:
                s['history'].clear(); s['ewma'].clear(); s['peer'] = None; s['generation'] += 1
                if not np.isfinite(stint): s['stint'] = np.nan
            if np.isfinite(stint): s['stint'] = stint
            s['compound'] = compound
            y = base.numeric(row.get('LapTime'))
            eligible = np.isfinite(y) and y > 0 and base.truth(row.get('IsAccurate', False)) and not (pit_in or pit_out) and not any(c in str(row.get('TrackStatus', '')) for c in '4567')
            if not eligible: continue
            previous = s['history'][-1] if s['history'] else None
            delta = (y - previous['y']) / (lap - previous['lap']) if previous else 0.
            s['peer'] = {'time': timestamp, 'lap': lap, 'y': y, 'delta': float(np.clip(delta, -5, 5))}
            item = {'lap': lap, 'y': y, **{k: base.numeric(row.get(k)) for k in SECTORS + SPEEDS}}
            s['history'].append(item)
            for alpha in [.25, .5, .75]:
                old = s['ewma'].get(alpha, y)
                s['ewma'][alpha] = y if abs(y-old) > 2 else old+alpha*(y-old)
            if (d, lap, timestamp) not in issuance_keys: continue
            h = list(s['history']); yy = np.array([i['y'] for i in h]); ll = np.array([i['lap'] for i in h])
            features = {'observed_lap': lap, 'position': finite(row.get('Position'), 11), 'lap_duration': y,
                        'fresh_tyre': float(base.truth(row.get('FreshTyre'))), 'history_count': len(h)}
            for k in range(1, 7):
                features[f'lag_{k}_gap'] = float(np.clip(yy[-k-1]-y, -15, 15)) if len(yy) > k else 0.
                features[f'lag_{k}_lap_distance'] = float(lap-ll[-k-1]) if len(yy) > k else 0.
            experts = {}
            for k in [2, 3, 5, 8]:
                v, l = yy[-k:], ll[-k:]
                med = float(np.median(v)); avg = float(v.mean())
                slope = float(np.dot(l-l.mean(), v-v.mean())/np.square(l-l.mean()).sum()) if len(v) >= 2 else 0.
                slope = float(np.clip(slope, -1, 1))
                features.update({f'mean_{k}_gap': float(np.clip(avg-y, -15, 15)), f'median_{k}_gap': float(np.clip(med-y, -15, 15)),
                    f'mad_{k}': float(np.median(np.abs(v-med))), f'trend_{k}': slope,
                    f'range_{k}': float(np.clip(v.max()-v.min(), 0, 20))})
                experts[f'mean_{k}'] = y+float(np.clip(avg-y, -3, 3))
                experts[f'median_{k}'] = y+float(np.clip(med-y, -3, 3))
                experts[f'trend_{k}'] = y+float(np.clip(avg+slope*(lap+1-l.mean())-y, -3, 3))
            for alpha, value in s['ewma'].items():
                features[f'reset_ewma_{alpha}_gap'] = value-y
                experts[f'reset_ewma_{alpha}'] = value
            for col in SECTORS+SPEEDS:
                current = finite(row.get(col))
                vals = np.array([v[col] for v in h[-5:]])
                vals = vals[np.isfinite(vals)]
                features[col] = current
                features[col+'_missing'] = float(not np.isfinite(base.numeric(row.get(col))))
                features[col+'_median_gap'] = float(np.clip(np.median(vals)-current, -30, 30)) if len(vals) else 0.
            for c in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']:
                features['compound_'+c] = float(compound == c)
            others = [p for driver, p in peers.items() if driver != d and 0 < timestamp-p['time'] <= 180 and abs(lap-p['lap']) <= 1]
            features['peer_count'] = len(others)
            features['peer_delta_mean'] = float(np.mean([p['delta'] for p in others])) if others else 0.
            features['peer_delta_median'] = float(np.median([p['delta'] for p in others])) if others else 0.
            features['peer_delta_mad'] = float(np.median(np.abs(np.array([p['delta'] for p in others])-features['peer_delta_median']))) if others else 0.
            features['relative_field_pace'] = float(np.clip(y-np.median([p['y'] for p in others]), -15, 15)) if others else 0.
            extra.append(dict(event_key=event_key, driver_id=d, issued_after_lap_number=lap, issued_at_timestamp=float(timestamp),
                **{FEATURE_PREFIX+k: float(v) for k, v in features.items()}, **{'expert_'+k: float(v) for k,v in experts.items()}))
    extra = pd.DataFrame(extra)
    enriched = issued.merge(extra, on=KEYS, validate='one_to_one')
    assert len(enriched) == len(issued)
    return enriched
