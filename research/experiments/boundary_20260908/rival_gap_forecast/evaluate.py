"""Fixed gap-only advancement against both base and quality references.

The naming adapter reuses the frozen telemetry population checks, event weights,
20,000-draw event/block3 intervals and LOO calculation without changing their
mathematics. The source/data bindings and forecast-before-label closure belong
to the runner, not this array-level evaluator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.telemetry import evaluate as original
from . import models

MODEL_NAMES = models.MODEL_NAMES
CANDIDATE = 'gap_hgb'
REFERENCES = ('base_hgb', 'quality_hgb')
SELECTION_EVENTS = tuple(range(202301, 202323))
SELECTION_ROWS = 20_432
SELECTION_MATCHED = 20_007
SELECTION_UNMATCHED = 425


def _predictions(frame, predictions, reference):
    if set(predictions) != set(MODEL_NAMES):
        raise ValueError('All three gap/base/quality forecasts are mandatory')
    baseline = models.reference_values(frame, reference)
    support = models.old_models.support_mask(frame.gap_supported, len(frame))
    result = {}
    for name in MODEL_NAMES:
        value = predictions[name]
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64
                or value.shape != (len(frame),) or not np.isfinite(value).all() or (value <= 0).any()):
            raise ValueError('Every forecast must be a finite positive float64 vector')
        if name == 'base_hgb' and not np.array_equal(value.view(np.uint64), baseline.view(np.uint64)):
            raise ValueError('Closed base reference forecast bits changed')
        if name != 'base_hgb' and not np.array_equal(value[~support].view(np.uint64), baseline[~support].view(np.uint64)):
            raise ValueError('Unsupported gap forecasts must retain exact saved base bits')
        result['telemetry_hgb' if name == 'gap_hgb' else name] = value
    return result


def _population(frame, expected_matched, expected_issuances, *, latency_seconds):
    if (len(frame) != SELECTION_ROWS or len(expected_issuances) != SELECTION_ROWS
            or len(expected_matched) != SELECTION_MATCHED
            or 'outcome_status' not in frame
            or int(frame.outcome_status.eq('matched').sum()) != SELECTION_MATCHED
            or int(frame.outcome_status.eq('unmatched').sum()) != SELECTION_UNMATCHED):
        raise ValueError('Selection requires all 20432 issuances, 20007 matched and 425 unmatched rows')
    models.validate_original_issuances(frame, expected_issuances)
    models.matrices(frame, latency_seconds=latency_seconds)
    # Both present label tables must bind derived identities/clocks, not merely
    # the numeric lap outcome; the old evaluator also checks their consistency.
    matched = frame.loc[frame.outcome_status.eq('matched')]
    actual, expected = original._keyed(matched), original._keyed(expected_matched)
    if len(actual) != len(expected) or set(actual.index) != set(expected.index):
        raise ValueError('Original matched selection identities changed')
    columns = [c for c in ('target_id', 'target_at_ns') if c in expected]
    original._assert_original_values(actual, expected.reindex(actual.index), columns, population='matched')
    adapted = frame.copy(deep=False)
    adapted['telemetry_supported'] = frame.gap_supported.to_numpy()
    return adapted


def contributions(frame, candidate, reference):
    """Driver contributions sum to the event-balanced, not row-balanced, delta."""
    y = frame.lap_time_seconds.to_numpy(float)
    if (not len(frame) or np.asarray(candidate).shape != y.shape or np.asarray(reference).shape != y.shape
            or not np.isfinite(y).all() or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError('Contribution diagnostics require aligned finite matched rows')
    delta = np.abs(candidate-y)-np.abs(reference-y)
    counts = frame.groupby('event_key').event_key.transform('size').to_numpy(float)
    events = frame.event_key.nunique()
    values = pd.DataFrame({'event_key': frame.event_key.to_numpy(),
        'driver_id': frame.driver_id.astype(str).to_numpy(), 'delta': delta,
        'contribution': delta/(events*counts)})
    pairs = [{'event_key': int(event), 'driver_id': driver, 'rows': len(group),
              'row_mae_delta_seconds': float(group.delta.mean()),
              'contribution_to_event_mae_delta_seconds': float(group.contribution.sum())}
             for (event, driver), group in values.groupby(['event_key', 'driver_id'], sort=True)]
    return {'event_balanced_delta_seconds': float(values.contribution.sum()),
        'row_mae_delta_seconds': float(values.delta.mean()),
        'weight_definition': '1/(number of events * matched rows within the row event)',
        'per_event': [{'event_key': int(event), 'rows': len(group),
                      'mae_delta_seconds': float(group.delta.mean()),
                      'contribution_to_event_mae_delta_seconds': float(group.contribution.sum())}
                     for event, group in values.groupby('event_key', sort=True)],
        'per_driver': [{'driver_id': driver, 'rows': len(group),
                       'contribution_to_event_mae_delta_seconds': float(group.contribution.sum())}
                      for driver, group in values.groupby('driver_id', sort=True)],
        'largest_event_driver_contributions': sorted(pairs, key=lambda row: (
            -abs(row['contribution_to_event_mae_delta_seconds']), row['event_key'], row['driver_id']))[:20]}


def _rename(value):
    if isinstance(value, dict):
        return {('gap_hgb' if k == 'telemetry_hgb' else k): _rename(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename(v) for v in value]
    return 'gap_hgb' if isinstance(value, str) and value == 'telemetry_hgb' else value


def evaluate_selection(primary, primary_predictions, sensitivity, sensitivity_predictions, *,
                       expected_matched, expected_issuances, primary_references, sensitivity_references):
    """Only gap_hgb may advance; both fixed references and both lags are mandatory."""
    main = _population(primary, expected_matched, expected_issuances, latency_seconds=2)
    zero = _population(sensitivity, expected_matched, expected_issuances, latency_seconds=0)
    pp = _predictions(primary, primary_predictions, primary_references)
    sp = _predictions(sensitivity, sensitivity_predictions, sensitivity_references)
    report = _rename(original.evaluate_selection(main, pp, zero, sp,
        expected_matched=expected_matched, expected_issuances=expected_issuances))
    report.update(status='retrospective_rival_gap_research_no_promotion', promotion=False,
        candidate_eligible_for_advancement=CANDIDATE,
        reference_source_sha256={'2': primary_references.source_sha256, '0': sensitivity_references.source_sha256},
        baseline_parameter_scope='Saved 2022-only HGB; not the deployed 2022-2023 parameter asset',
        evaluation_adapter='Exact telemetry evaluator with gap_hgb renamed to telemetry_hgb and gap_supported renamed to telemetry_supported')
    report['contributing_events_and_drivers'] = {}
    for lag, frame, predictions in [('2', primary, primary_predictions), ('0', sensitivity, sensitivity_predictions)]:
        mask = frame.outcome_status.eq('matched').to_numpy()
        scored = frame.loc[mask]
        report['contributing_events_and_drivers'][lag] = {
            reference: contributions(scored, predictions[CANDIDATE][mask], predictions[reference][mask])
            for reference in REFERENCES}
        for name in MODEL_NAMES:
            errors = np.abs(predictions[name][mask]-scored.lap_time_seconds.to_numpy(float))
            event_rows = pd.DataFrame({'event_key': scored.event_key.to_numpy(), 'error': errors})
            report['metrics'][lag][name]['per_event'] = [
                {'event_key': int(event), 'rows': len(group), 'mae_seconds': float(group.error.mean())}
                for event, group in event_rows.groupby('event_key', sort=True)]
    return report
