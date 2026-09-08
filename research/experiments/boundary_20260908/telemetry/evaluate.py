"""Full-population telemetry comparisons with the frozen frontier uncertainty.

Run only after forecast/source closure. Outcome-unmatched issuances remain in the
coverage ledger and receive forecasts, but cannot contribute invented MAE labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.frontier_20260907 import live_frontier as original
from .models import BASE_FEATURES, EXPECTED_SELECTION_BASE_MAE, KEYS, MODEL_NAMES, support_mask


SELECTION_EVENTS = tuple(range(202301, 202323))
REFERENCES = ('base_hgb', 'quality_hgb')
TARGET_FIELDS = ('target_id', 'target_at_ns', 'target_lap_number',
                 'target_timestamp', 'lap_time_seconds', 'target_same_stint')


def _predictions(predictions, n):
    if set(predictions) != set(MODEL_NAMES):
        raise ValueError('all three frozen forecasts are mandatory')
    result = {}
    for name in MODEL_NAMES:
        value = np.asarray(predictions[name], dtype=float)
        if value.shape != (n,) or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError('positive finite predictions are required for every issuance')
        result[name] = value
    return result


def _keyed(frame):
    if any(k not in frame for k in KEYS) or frame[list(KEYS)].isna().any().any():
        raise ValueError('complete canonical issuance keys are required')
    if frame.duplicated(list(KEYS)).any():
        raise ValueError('duplicate issuance keys would change original weights')
    return frame.set_index(list(KEYS), verify_integrity=True)


def _assert_original_values(actual, expected, columns, *, population):
    if any(col not in expected or col not in actual for col in columns):
        raise ValueError(f'original {population} metadata, anchor and features are required')
    for column in columns:
        if not pd.Series(actual[column].to_numpy(), dtype=object).equals(
                pd.Series(expected[column].to_numpy(), dtype=object)):
            raise ValueError(f'original {population} value changed: {column}')


def validate_population(frame, predictions, expected_matched, *, expected_issuances, expected_events):
    """Check BOTH original inventories, including every unmatched issuance.

    ``expected_issuances`` must come from the frozen original encoder's issuance
    output, not from a subset of the telemetry assembly being evaluated.
    ``expected_matched`` independently fixes the subset whose outcomes can score.
    """
    if frame.empty or expected_matched.empty or expected_issuances.empty:
        raise ValueError('nonempty full issuance and original matched populations are required')
    full, original_full = _keyed(frame), _keyed(expected_issuances)
    if len(full) != len(original_full) or set(full.index) != set(original_full.index):
        raise ValueError('full issuance population must equal the original inventory, including unmatched rows')
    original_full = original_full.reindex(full.index)
    columns = ['forecast_naive_seconds', *BASE_FEATURES]
    columns += [c for c in ['issuance_id', 'issued_at_ns'] if c in original_full]
    _assert_original_values(full, original_full, columns, population='issuance')
    if ('issuance_id' not in frame or frame.issuance_id.duplicated().any()
            or not frame.issuance_id.map(lambda x: isinstance(x, str) and bool(x)).all()):
        raise ValueError('unique nonempty issuance IDs are required')
    if 'outcome_status' not in frame or not frame.outcome_status.isin(['matched', 'unmatched']).all():
        raise ValueError('every issuance needs an explicit matched/unmatched outcome status')
    if any(c not in frame for c in TARGET_FIELDS):
        raise ValueError('all outcome fields must be present, including explicit unmatched nulls')
    event_values = frame.event_key.to_numpy(float)
    if not np.isfinite(event_values).all() or not np.equal(event_values, np.floor(event_values)).all():
        raise ValueError('event keys must be finite integers')
    expected_events = tuple(sorted(expected_events))
    if tuple(sorted(map(int, frame.event_key.unique()))) != expected_events:
        raise ValueError('all declared events must remain in the issuance population')
    if tuple(sorted(map(int, expected_matched.event_key.unique()))) != expected_events:
        raise ValueError('the original matched reference must cover every declared event')
    if tuple(sorted(map(int, expected_issuances.event_key.unique()))) != expected_events:
        raise ValueError('the original issuance inventory must cover every declared event')
    predicted = _predictions(predictions, len(frame))
    supported = support_mask(frame.telemetry_supported, len(frame))
    for name in ['telemetry_hgb', 'quality_hgb']:
        if not np.array_equal(predicted[name][~supported].view(np.uint64),
                              predicted['base_hgb'][~supported].view(np.uint64)):
            raise ValueError('unsupported issuances must use the bitwise exact base forecast')
    matched = frame.outcome_status.eq('matched').to_numpy()
    for column in TARGET_FIELDS:
        if frame.loc[~matched, column].notna().any():
            raise ValueError('unmatched target fields must remain null')
        if frame.loc[matched, column].isna().any():
            raise ValueError('matched target fields cannot be null')
    eligible = frame.loc[matched]
    numeric = eligible[['target_lap_number', 'target_timestamp', 'lap_time_seconds']].to_numpy(float)
    if (not np.isfinite(numeric).all() or (numeric[:, [0, 2]] <= 0).any()
            or not (numeric[:, 1] > eligible.issued_at_timestamp.to_numpy(float)).all()):
        raise ValueError('matched targets must be positive and strictly after issuance')
    if not eligible.target_id.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError('matched target IDs must be nonempty strings')
    target_identity = ['event_key', 'driver_id', 'target_lap_number', 'target_timestamp',
                       'target_at_ns', 'lap_time_seconds']
    if (eligible.groupby('target_id')[target_identity].nunique(dropna=False) > 1).any().any():
        raise ValueError('one target ID cannot refer to inconsistent targets')
    actual, expected = _keyed(eligible), _keyed(expected_matched)
    if len(actual) != len(expected) or set(actual.index) != set(expected.index):
        raise ValueError('matched telemetry cohort must equal the original cohort exactly')
    # Preserve original target, anchor, and old feature values; no reweighting or
    # filtering based on telemetry support, missingness, targets or model scores.
    expected = expected.reindex(actual.index)
    columns = ['target_lap_number', 'target_timestamp', 'lap_time_seconds',
               'target_same_stint', 'forecast_naive_seconds', *BASE_FEATURES]
    _assert_original_values(actual, expected, columns, population='matched')
    coverage = {
        'original_full_issuance_inventory_verified': True,
        'original_matched_inventory_verified': True,
        'rows_all': len(frame), 'rows_matched': int(matched.sum()),
        'rows_unmatched_retained': int((~matched).sum()),
        'supported_rows_all': int(supported.sum()),
        'fallback_rows_all': int((~supported).sum()),
        'supported_rows_matched': int((supported & matched).sum()),
        'fallback_rows_matched': int((~supported & matched).sum()),
        'per_event': [],
    }
    for event in expected_events:
        mask = frame.event_key.eq(event).to_numpy()
        coverage['per_event'].append({
            'event_key': int(event), 'rows_all': int(mask.sum()),
            'rows_matched': int((mask & matched).sum()),
            'rows_unmatched_retained': int((mask & ~matched).sum()),
            'supported_rows_all': int((mask & supported).sum()),
            'fallback_rows_all': int((mask & ~supported).sum()),
            'supported_rows_matched': int((mask & matched & supported).sum()),
        })
    return matched, predicted, coverage


def comparison(frame, candidate, reference):
    """Exactly original sorted-event weights, RNG seed and draw order."""
    y = frame.lap_time_seconds.to_numpy(float)
    candidate, reference = np.asarray(candidate, float), np.asarray(reference, float)
    if (not len(frame) or candidate.shape != (len(frame),) or reference.shape != (len(frame),)
            or not np.isfinite(y).all() or not np.isfinite(candidate).all()
            or not np.isfinite(reference).all()):
        raise ValueError('comparison requires aligned finite matched targets and forecasts')
    # The frozen function uses seed20260907, 20,000 event draws, then 20,000
    # circular length-three event-block draws, followed by all-event LOO means.
    with np.errstate(divide='ignore', invalid='ignore'):
        result = original.diagnostics(frame, candidate, reference=reference)
    if result['baseline_mae'] == 0.0:
        # Relative improvement over a perfect reference is undefined. Keep
        # finite absolute losses/intervals and fail the relative-gain gate.
        result['relative_reduction'] = None
    return result


def _checks(value, minimum_gain):
    return {
        'minimum_relative_gain': (value['relative_reduction'] is not None
                                  and value['relative_reduction'] >= minimum_gain),
        'event_ci_upper_negative': value['event_ci95'][1] < 0,
        'block3_ci_upper_negative': value['block3_ci95'][1] < 0,
        'all_leave_one_event_out_negative': (
            value['loo_max_delta'] is not None and value['loo_max_delta'] < 0),
    }


def _lag_parity(primary, sensitivity, primary_predictions, sensitivity_predictions):
    fields = ['issuance_id', *KEYS, *TARGET_FIELDS, 'forecast_naive_seconds', *BASE_FEATURES]
    if len(primary) != len(sensitivity):
        raise ValueError('0-second sensitivity cannot change the issuance population')
    for column in fields:
        a, b = primary[column].to_numpy(), sensitivity[column].to_numpy()
        # pandas handles strings and explicit None while preserving ordered IDs.
        if not pd.Series(a, dtype=object).equals(pd.Series(b, dtype=object)):
            raise ValueError(f'0-second sensitivity changed population or target: {column}')
    if not np.array_equal(primary_predictions['base_hgb'].view(np.uint64),
                          sensitivity_predictions['base_hgb'].view(np.uint64)):
        raise ValueError('base80 predictions must be identical across telemetry lags')


def evaluate_selection(primary, primary_predictions, sensitivity, sensitivity_predictions,
                       *, expected_matched, expected_issuances):
    """The only selectable mechanism is telemetry content, versus both references."""
    pm, pp, pc = validate_population(primary, primary_predictions, expected_matched,
                                     expected_issuances=expected_issuances, expected_events=SELECTION_EVENTS)
    sm, sp, sc = validate_population(sensitivity, sensitivity_predictions, expected_matched,
                                     expected_issuances=expected_issuances, expected_events=SELECTION_EVENTS)
    _lag_parity(primary, sensitivity, pp, sp)
    main_frame, zero_frame = primary.loc[pm], sensitivity.loc[sm]
    metrics = {}
    for lag, frame, mask, forecasts in [('2', main_frame, pm, pp), ('0', zero_frame, sm, sp)]:
        values = {}
        for name in MODEL_NAMES:
            errors = np.abs(forecasts[name][mask] - frame.lap_time_seconds.to_numpy(float))
            event_errors = pd.DataFrame({'event_key': frame.event_key.to_numpy(), 'error': errors}).groupby('event_key').error.mean()
            values[name] = {'rows': len(frame), 'events': len(event_errors),
                            'event_mae_seconds': float(event_errors.mean()),
                            'row_mae_seconds': float(errors.mean())}
        metrics[lag] = values
    baseline_value = metrics['2']['base_hgb']['event_mae_seconds']
    if abs(baseline_value - EXPECTED_SELECTION_BASE_MAE) > 1e-12:
        raise ValueError('2022-only base HGB does not reproduce frozen 2023 selection MAE')
    comparisons = {
        '2': {name: comparison(main_frame, pp['telemetry_hgb'][pm], pp[name][pm]) for name in REFERENCES},
        '0': {name: comparison(zero_frame, sp['telemetry_hgb'][sm], sp[name][sm]) for name in REFERENCES},
    }
    checks = {'2': {name: _checks(comparisons['2'][name], .01) for name in REFERENCES},
              '0': {name: {'positive_relative_gain': (comparisons['0'][name]['relative_reduction'] is not None
                                                     and comparisons['0'][name]['relative_reduction'] > 0)}
                    for name in REFERENCES}}
    advances = all(check for lag in checks.values() for ref in lag.values() for check in ref.values())
    return {
        'status': 'retrospective_telemetry_research_no_promotion',
        'primary_latency_seconds': 2, 'sensitivity_latency_seconds': 0,
        'sensitivity_refitted': False, 'model_names': list(MODEL_NAMES),
        'expected_events': list(SELECTION_EVENTS), 'coverage': {'2': pc, '0': sc},
        'metrics': metrics, 'comparisons': comparisons, 'gate_checks': checks,
        'baseline_reproduction': {'expected_event_mae_seconds': EXPECTED_SELECTION_BASE_MAE,
                                  'actual_event_mae_seconds': baseline_value,
                                  'absolute_tolerance_seconds': 1e-12},
        'advances_to_later_evaluation': bool(advances),
        'decision': 'advance_to_frozen_later_evaluation' if advances else 'stop_no_new_transfer_or_promotion',
        'uncertainty': {'seed': 20260907, 'resamples_each': 20000, 'circular_event_block_length': 3,
                        'interval': 'percentile95', 'sign': 'candidate minus reference',
                        'limit': 'Conditional historical event resampling; not a correction for previous research selection.'},
    }
