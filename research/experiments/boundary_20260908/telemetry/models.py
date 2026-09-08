"""Frozen telemetry augmentation: three 2022 fits, unchanged frontier objective.

The runner owns source/data locks and closes forecasts before attaching outcomes.
This module never loads historical inputs, target tables, or serialized models.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.experiments.frontier_20260907 import live_frontier as original


BASE_FEATURES = (
    'level_gap', 'own_last_delta', 'own_robust_trend', 'common_increment',
    'common_support', 'log_stint_clean_count', 'tyre_age', 'clean_lap_gap',
    'wet_compound', 'x_observed_lap', 'x_position', 'x_lap_duration',
    'x_fresh_tyre', 'x_history_count', 'x_lag_1_gap', 'x_lag_1_lap_distance',
    'x_lag_2_gap', 'x_lag_2_lap_distance', 'x_lag_3_gap', 'x_lag_3_lap_distance',
    'x_lag_4_gap', 'x_lag_4_lap_distance', 'x_lag_5_gap', 'x_lag_5_lap_distance',
    'x_lag_6_gap', 'x_lag_6_lap_distance', 'x_mean_2_gap', 'x_median_2_gap',
    'x_mad_2', 'x_trend_2', 'x_range_2', 'x_mean_3_gap', 'x_median_3_gap',
    'x_mad_3', 'x_trend_3', 'x_range_3', 'x_mean_5_gap', 'x_median_5_gap',
    'x_mad_5', 'x_trend_5', 'x_range_5', 'x_mean_8_gap', 'x_median_8_gap',
    'x_mad_8', 'x_trend_8', 'x_range_8', 'x_reset_ewma_0.25_gap',
    'x_reset_ewma_0.5_gap', 'x_reset_ewma_0.75_gap', 'x_Sector1Time',
    'x_Sector1Time_missing', 'x_Sector1Time_median_gap', 'x_Sector2Time',
    'x_Sector2Time_missing', 'x_Sector2Time_median_gap', 'x_Sector3Time',
    'x_Sector3Time_missing', 'x_Sector3Time_median_gap', 'x_SpeedI1',
    'x_SpeedI1_missing', 'x_SpeedI1_median_gap', 'x_SpeedI2', 'x_SpeedI2_missing',
    'x_SpeedI2_median_gap', 'x_SpeedFL', 'x_SpeedFL_missing',
    'x_SpeedFL_median_gap', 'x_SpeedST', 'x_SpeedST_missing',
    'x_SpeedST_median_gap', 'x_compound_SOFT', 'x_compound_MEDIUM',
    'x_compound_HARD', 'x_compound_INTERMEDIATE', 'x_compound_WET',
    'x_peer_count', 'x_peer_delta_mean', 'x_peer_delta_median',
    'x_peer_delta_mad', 'x_relative_field_pace',
)
MODEL_NAMES = ('base_hgb', 'telemetry_hgb', 'quality_hgb')
CONFIG = {'kind': 'hgb', 'leaves': 15, 'iterations': 150}
EXPECTED_SELECTION_BASE_MAE = 0.512080723333514
KEYS = tuple(original.KEYS)


def layout():
    """Use the independently frozen encoder order, never alphabetical sorting."""
    from .features import FEATURE_NAMES, CONTENT_INDICES, QUALITY_INDICES
    names = tuple(FEATURE_NAMES)
    content, quality = tuple(CONTENT_INDICES), tuple(QUALITY_INDICES)
    if (len(names) != 90 or len(set(names)) != 90 or len(content) != 27
            or len(quality) != 63 or sorted(content + quality) != list(range(90))):
        raise ValueError('expected exactly 27 content and 63 quality coordinates')
    return names, content, quality


def support_mask(values, n):
    items = list(values)
    if len(items) != n or not all(isinstance(x, (bool, np.bool_)) for x in items):
        raise ValueError('support must contain one explicit boolean per issuance')
    return np.asarray(items, dtype=bool)


def validate_issuances(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('a nonempty issuance DataFrame is required')
    required = [*KEYS, 'issuance_id', 'forecast_naive_seconds', *BASE_FEATURES]
    if any(c not in frame for c in required):
        raise ValueError('missing canonical issuance metadata or base features')
    if not frame.index.is_unique or frame[list(KEYS)].isna().any().any():
        raise ValueError('issuance index and canonical keys must be complete')
    if frame.duplicated(list(KEYS)).any() or frame.issuance_id.duplicated().any():
        raise ValueError('duplicate issuances cannot acquire extra training weight')
    if not frame.issuance_id.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError('issuance IDs must be nonempty strings')
    keys = frame[['event_key', 'issued_after_lap_number', 'issued_at_timestamp']].to_numpy(float)
    if not np.isfinite(keys).all() or (keys[:, :2] <= 0).any():
        raise ValueError('canonical numeric keys must be finite and positive')
    if not np.equal(keys[:, :2], np.floor(keys[:, :2])).all():
        raise ValueError('event keys and lap numbers must be integers')
    anchor = frame.forecast_naive_seconds.to_numpy(float)
    base = frame[list(BASE_FEATURES)].to_numpy(float)
    if not np.isfinite(anchor).all() or (anchor <= 0).any() or not np.isfinite(base).all():
        raise ValueError('original base features and positive anchor must be finite')


def matrices(frame, *, latency_seconds):
    """Build identical 170-column layouts; only control content is zeroed."""
    if isinstance(latency_seconds, bool) or latency_seconds not in (0, 2):
        raise ValueError('only frozen 2-second primary or 0-second sensitivity is allowed')
    validate_issuances(frame)
    required = ['issued_at_ns', 'telemetry_cutoff_ns', 'telemetry_supported', 'telemetry_values']
    if any(c not in frame for c in required):
        raise ValueError('missing telemetry values, support or exact cutoff clocks')
    for issued, cutoff in zip(frame.issued_at_ns, frame.telemetry_cutoff_ns):
        if (not isinstance(issued, (int, np.integer)) or isinstance(issued, (bool, np.bool_))
                or not isinstance(cutoff, (int, np.integer)) or isinstance(cutoff, (bool, np.bool_))
                or int(issued) - int(cutoff) != latency_seconds * 1_000_000_000):
            raise ValueError('telemetry cutoff must use exact frozen nanosecond lag')
    supported = support_mask(frame.telemetry_supported, len(frame))
    names, content, quality = layout()
    values = np.asarray(frame.telemetry_values.tolist(), dtype=float)
    if values.shape != (len(frame), 90) or np.isinf(values).any():
        raise ValueError('telemetry must have 90 numeric finite-or-NaN coordinates')
    columns = ['telemetry__' + n for n in names]
    if set(columns) & set(BASE_FEATURES):
        raise ValueError('telemetry column collision')
    base = frame[list(BASE_FEATURES)].copy()
    actual = pd.concat([base, pd.DataFrame(values, index=frame.index, columns=columns)], axis=1)
    control = actual.copy()
    control.loc[:, [columns[i] for i in content]] = 0.0
    # No content-dependent missingness is retained in the control's 27 slots.
    # The encoder's declared 63 quality coordinates are retained verbatim.
    return base, actual, control, supported


def _training_frame(frame, x):
    out = x.copy()
    for col in ['event_key', 'lap_time_seconds', 'forecast_naive_seconds']:
        out[col] = frame[col].to_numpy()
    return out


def fit_models(frame):
    """Fit all matched 2022 rows once, using primary 2-second inputs only.

    This function deliberately does not support a training-mask parameter.
    Unsupported telemetry rows remain in all three identical populations.
    """
    base, augmented, control, supported = matrices(frame, latency_seconds=2)
    if ('year' not in frame or not frame.year.eq(2022).all()
            or not frame.event_key.floordiv(100).eq(2022).all()):
        raise ValueError('all three models must be fitted only on 2022')
    if 'outcome_status' not in frame or not frame.outcome_status.eq('matched').all():
        raise ValueError('training requires the original matched target population')
    if 'lap_time_seconds' not in frame:
        raise ValueError('training targets are required after data closure')
    y = frame.lap_time_seconds.to_numpy(float)
    if not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('training lap targets must be positive and finite')
    fitted = {}
    with threadpool_limits(limits=1):
        for name, x in zip(MODEL_NAMES, (base, augmented, control)):
            # Calls the exact frozen objective, event balancing and HGB settings:
            # residual train clip +/-5 seconds, predict correction clip +/-3.
            fitted[name] = original.fit_model(_training_frame(frame, x), dict(CONFIG), list(x.columns))
    ids = frame.issuance_id.tolist()
    return {
        'models': fitted,
        'base_features': list(BASE_FEATURES),
        'telemetry_features': list(layout()[0]),
        'fit_summary': {
            'fit_year': 2022, 'fit_latency_seconds': 2,
            'rows_per_fit': len(frame), 'fits': 3, 'augmented_fits': 2,
            'supported_training_rows': int(supported.sum()),
            'unsupported_training_rows_retained': int((~supported).sum()),
            'events': sorted(map(int, frame.event_key.unique())),
            'issuance_ids_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
            'weighting': 'unchanged frontier event-balanced weights on all matched 2022 rows',
            'configuration': dict(CONFIG),
        },
    }


def predict_models(bundle, frame, *, latency_seconds):
    """Issue all rows, including unmatched outcomes; never read their labels."""
    base, augmented, control, supported = matrices(frame, latency_seconds=latency_seconds)
    if (bundle.get('base_features') != list(BASE_FEATURES)
            or bundle.get('telemetry_features') != list(layout()[0])
            or set(bundle.get('models', {})) != set(MODEL_NAMES)
            or bundle.get('fit_summary', {}).get('fit_year') != 2022
            or bundle.get('fit_summary', {}).get('fit_latency_seconds') != 2):
        raise ValueError('only the matching 2022 primary-fit bundle is allowed')
    with threadpool_limits(limits=1):
        base['forecast_naive_seconds'] = frame.forecast_naive_seconds.to_numpy()
        baseline = np.asarray(original.predict_model(base, bundle['models']['base_hgb']), dtype=float)
        predictions = {'base_hgb': baseline}
        for name, x in [('telemetry_hgb', augmented), ('quality_hgb', control)]:
            values = baseline.copy()
            if supported.any():
                available = x.loc[supported].copy()
                available['forecast_naive_seconds'] = frame.loc[supported, 'forecast_naive_seconds'].to_numpy()
                values[supported] = original.predict_model(available, bundle['models'][name])
            # Assignment happens only on support; fallback retains baseline bits.
            predictions[name] = values
    if any(v.shape != (len(frame),) or not np.isfinite(v).all() or (v <= 0).any()
           for v in predictions.values()):
        raise ValueError('every issuance must receive three positive finite forecasts')
    return predictions
