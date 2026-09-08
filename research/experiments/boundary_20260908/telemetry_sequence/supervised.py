"""Three fixed HGBs on frozen16-vectors; no representation tuning or raw I/O.

The runner owns immutable feature/encoder/reference closures, resource limits,
and withholding external2023 labels until every prediction has closed.
Suggested commit: research(f1-live): compare frozen sequence embeddings fairly
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.experiments.boundary_20260908.telemetry import models as old_models
from research.experiments.boundary_20260908.telemetry import evaluate as old_evaluate

CONTROLS = ('ordered', 'permuted', 'random')
NEW_MODEL_NAMES = tuple(control+'_hgb' for control in CONTROLS)
OLD_MODEL_NAMES = tuple(old_models.MODEL_NAMES)
MODEL_NAMES = (*OLD_MODEL_NAMES, *NEW_MODEL_NAMES)
REFERENCES = (*OLD_MODEL_NAMES, 'permuted_hgb', 'random_hgb')
CANDIDATE = 'ordered_hgb'
EMBEDDING_FEATURES = tuple(f'sequence__embedding_{i:02d}' for i in range(16))
TRAIN_ROWS = 18_363
TRAIN_EVENTS = tuple(range(202201, 202223))
CONFIG = {'kind': 'hgb', 'leaves': 15, 'iterations': 150}


@dataclass(frozen=True)
class EmbeddingTable:
    issuance_ids: tuple[str, ...]
    cutoff_ns: tuple[int, ...]
    values: np.ndarray
    empty_context: np.ndarray
    encoder_sha256: str
    control: str


@dataclass(frozen=True)
class ReferenceForecasts:
    issuance_ids: tuple[str, ...]
    cutoff_ns: tuple[int, ...]
    predictions: dict[str, np.ndarray]
    source_sha256: str


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('A lowercase SHA256 binding is required')
    return value


def _alignment(frame, ids, cutoff):
    if not isinstance(ids, tuple) or ids != tuple(frame.issuance_id):
        raise ValueError('Embedding/reference issuance IDs must match exactly in order')
    if (not isinstance(cutoff, tuple) or len(cutoff) != len(frame)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in cutoff)
            or cutoff != tuple(frame.telemetry_cutoff_ns)):
        raise ValueError('Embedding/reference cutoffs must match exact issuance clocks')


def _embedding(table, frame, supported, control):
    if not isinstance(table, EmbeddingTable) or table.control != control:
        raise ValueError('Embedding control identity differs from its fixed branch')
    _sha(table.encoder_sha256);_alignment(frame, table.issuance_ids, table.cutoff_ns)
    values = table.values
    if (not isinstance(values, np.ndarray) or values.dtype != np.float32
            or values.shape != (len(frame), 16) or not np.isfinite(values).all()):
        raise ValueError('Frozen embeddings must be finite float32[N,16]')
    empty = old_models.support_mask(table.empty_context, len(frame))
    if (empty & supported).any():
        raise ValueError('An empty context cannot claim supported telemetry')
    if np.any(values[empty] != 0):
        raise ValueError('Empty contexts require exact zero embeddings')
    norm = np.linalg.norm(values[~empty].astype(np.float64), axis=1)
    if np.any(np.abs(norm-1.) > 2e-6):
        raise ValueError('Every nonempty embedding must have unit norm')
    return values


def reference_values(frame, reference):
    if not isinstance(reference, ReferenceForecasts):
        raise TypeError('Closed ReferenceForecasts are required')
    _sha(reference.source_sha256);_alignment(frame, reference.issuance_ids, reference.cutoff_ns)
    if set(reference.predictions) != set(OLD_MODEL_NAMES):
        raise ValueError('All three closed old references are mandatory')
    result = {}
    for name in OLD_MODEL_NAMES:
        values = reference.predictions[name]
        if (not isinstance(values, np.ndarray) or values.dtype != np.float64
                or values.shape != (len(frame),) or not np.isfinite(values).all() or (values <= 0).any()):
            raise ValueError('Old forecasts must retain positive finite float64 values')
        result[name] = values.copy()
    return result


def _inputs(frame, tables, *, latency_seconds):
    _, augmented, _, supported = old_models.matrices(frame, latency_seconds=latency_seconds)
    if not np.isfinite(augmented.to_numpy()).all():
        raise ValueError('Closed telemetry90 predictors must be finite')
    if set(tables) != set(CONTROLS):
        raise ValueError('Exactly ordered, permuted and random embedding tables are required')
    values = {control: _embedding(tables[control], frame, supported, control) for control in CONTROLS}
    # Construct one186-column matrix at a time in fit/predict; avoid retaining
    # three full copies in addition to the immutable materialized source frame.
    return augmented, values, supported


def _matrix(augmented, values):
    return pd.concat([augmented, pd.DataFrame(values, index=augmented.index, columns=EMBEDDING_FEATURES)], axis=1)


def _training_population(frame, expected):
    if (len(frame) != TRAIN_ROWS or len(expected) != TRAIN_ROWS
            or 'year' not in frame or not frame.year.eq(2022).all()
            or tuple(sorted(map(int, frame.event_key.unique()))) != TRAIN_EVENTS
            or not frame.outcome_status.eq('matched').all()):
        raise ValueError('Training requires all18363 original matched2022 rows and all22 events')
    actual, original = old_evaluate._keyed(frame), old_evaluate._keyed(expected)
    if not actual.index.equals(original.index):
        raise ValueError('Original matched training identities/order changed')
    columns = [*old_models.BASE_FEATURES, 'forecast_naive_seconds', 'lap_time_seconds',
               'target_lap_number', 'target_timestamp', 'target_same_stint']
    old_evaluate._assert_original_values(actual, original, columns, population='training')
    target = frame.lap_time_seconds.to_numpy(float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError('Original training outcomes must be finite and positive')


def fit_models(frame, tables_by_control, *, expected_matched):
    """Exactly3 fits. Both supplied frames come from separately closed old inputs."""
    _training_population(frame, expected_matched)
    augmented, values, supported = _inputs(frame, tables_by_control, latency_seconds=2)
    fitted = {}
    with threadpool_limits(limits=1):
        for control, name in zip(CONTROLS, NEW_MODEL_NAMES, strict=True):
            x = _matrix(augmented, values[control])
            trained = old_models._training_frame(frame, x)
            fitted[name] = old_models.original.fit_model(trained, dict(CONFIG), list(x.columns))
            del x, trained
    return {'models': fitted, 'base_features': list(old_models.BASE_FEATURES),
        'telemetry_features': list(old_models.layout()[0]), 'embedding_features': list(EMBEDDING_FEATURES),
        'encoder_sha256': {control: tables_by_control[control].encoder_sha256 for control in CONTROLS},
        'fit_summary': {'fit_year': 2022, 'fit_latency_seconds': 2, 'rows_per_fit': len(frame),
            'events': list(TRAIN_EVENTS), 'fits': 3, 'representation_fits': 0, 'old_reference_refits': 0,
            'configuration': dict(CONFIG), 'supported_training_rows': int(supported.sum()),
            'unsupported_training_rows_retained': int((~supported).sum()),
            'issuance_ids_sha256': hashlib.sha256(json.dumps(frame.issuance_id.tolist(), separators=(',', ':')).encode()).hexdigest(),
            'weighting': 'unchanged event-balanced weights on the complete original matched2022 population'}}


def _bundle(bundle, tables):
    summary = bundle.get('fit_summary', {})
    if (set(bundle.get('models', {})) != set(NEW_MODEL_NAMES)
            or bundle.get('base_features') != list(old_models.BASE_FEATURES)
            or bundle.get('telemetry_features') != list(old_models.layout()[0])
            or bundle.get('embedding_features') != list(EMBEDDING_FEATURES)
            or summary.get('fit_year') != 2022 or summary.get('fit_latency_seconds') != 2
            or summary.get('rows_per_fit') != TRAIN_ROWS or summary.get('events') != list(TRAIN_EVENTS)
            or summary.get('fits') != 3 or summary.get('configuration') != CONFIG
            or bundle.get('encoder_sha256') != {control: tables[control].encoder_sha256 for control in CONTROLS}):
        raise ValueError('Prediction bundle differs from its frozen training/encoder contract')


def predict_models(bundle, frame, tables_by_control, *, latency_seconds, saved_references):
    """All6 points, including every unmatched issuance; no outcome reads."""
    augmented, embeddings, supported = _inputs(frame, tables_by_control, latency_seconds=latency_seconds)
    _bundle(bundle, tables_by_control)
    result = reference_values(frame, saved_references)
    # Also check original reference fallbacks on the new full issuance frame.
    for name in ('telemetry_hgb', 'quality_hgb'):
        if not np.array_equal(result[name][~supported].view(np.uint64), result['base_hgb'][~supported].view(np.uint64)):
            raise ValueError('Old unsupported reference points differ from the saved base')
    with threadpool_limits(limits=1):
        for control, name in zip(CONTROLS, NEW_MODEL_NAMES, strict=True):
            prediction = result['base_hgb'].copy()
            if supported.any():
                x = _matrix(augmented.loc[supported], embeddings[control][supported])
                x['forecast_naive_seconds'] = frame.loc[supported, 'forecast_naive_seconds'].to_numpy()
                prediction[supported] = old_models.original.predict_model(x, bundle['models'][name])
                del x
            if not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise ValueError('Every new issuance forecast must be positive and finite')
            result[name] = prediction
    return result


def _predictions(frame, predictions, references):
    if set(predictions) != set(MODEL_NAMES):
        raise ValueError('All six candidate/reference forecasts are mandatory')
    expected = reference_values(frame, references)
    result = {}
    supported = old_models.support_mask(frame.telemetry_supported, len(frame))
    for name in MODEL_NAMES:
        value = np.asarray(predictions[name])
        if value.dtype != np.float64 or value.shape != (len(frame),) or not np.isfinite(value).all() or (value <= 0).any():
            raise ValueError('All six forecasts must be finite positive float64 vectors')
        if name in OLD_MODEL_NAMES and not np.array_equal(value.view(np.uint64), expected[name].view(np.uint64)):
            raise ValueError('A closed old reference forecast was changed')
        if name in NEW_MODEL_NAMES and not np.array_equal(value[~supported].view(np.uint64), expected['base_hgb'][~supported].view(np.uint64)):
            raise ValueError('Every unsupported new forecast must retain exact saved base bits')
        result[name] = value
    return result


def _metrics(frame, predictions):
    y = frame.lap_time_seconds.to_numpy(float);metrics = {}
    for name in MODEL_NAMES:
        errors = np.abs(predictions[name]-y)
        per_event = pd.DataFrame({'event_key': frame.event_key.to_numpy(), 'error': errors}).groupby('event_key').error.mean()
        metrics[name] = {'rows': len(frame), 'events': len(per_event),
            'event_mae_seconds': float(per_event.mean()), 'row_mae_seconds': float(errors.mean())}
    return metrics


def contributions(frame, candidate, reference):
    """Additive contributions use1/(events * matched rows in that event)."""
    y = frame.lap_time_seconds.to_numpy(float)
    if (len(frame) == 0 or np.asarray(candidate).shape != y.shape or np.asarray(reference).shape != y.shape
            or not np.isfinite(y).all() or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError('Contribution diagnostics require aligned finite matched rows')
    delta = np.abs(candidate-y)-np.abs(reference-y)
    counts = frame.groupby('event_key').event_key.transform('size').to_numpy(float)
    events = frame.event_key.nunique()
    value = pd.DataFrame({'event_key': frame.event_key.to_numpy(), 'driver_id': frame.driver_id.astype(str).to_numpy(),
        'delta': delta, 'contribution': delta/(events*counts)})
    rows = []
    for (event, driver), group in value.groupby(['event_key', 'driver_id'], sort=True):
        rows.append({'event_key': int(event), 'driver_id': driver, 'rows': len(group),
            'row_mae_delta_seconds': float(group.delta.mean()),
            'contribution_to_event_mae_delta_seconds': float(group.contribution.sum())})
    largest = sorted(rows, key=lambda r: (-abs(r['contribution_to_event_mae_delta_seconds']), r['event_key'], r['driver_id']))[:20]
    return {'event_balanced_delta_seconds': float(value.contribution.sum()),
        'row_mae_delta_seconds': float(value.delta.mean()),
        'weight_definition': '1/(number of events * matched rows within the row event)',
        'per_event': [{'event_key': int(event), 'rows': len(group), 'mae_delta_seconds': float(group.delta.mean()),
            'contribution_to_event_mae_delta_seconds': float(group.contribution.sum())}
            for event, group in value.groupby('event_key', sort=True)],
        'per_driver': [{'driver_id': driver, 'rows': len(group),
            'contribution_to_event_mae_delta_seconds': float(group.contribution.sum())}
            for driver, group in value.groupby('driver_id', sort=True)],
        'largest_event_driver_contributions': largest}


def evaluate_selection(primary, primary_predictions, sensitivity, sensitivity_predictions, *,
                       expected_matched, expected_issuances, primary_references, sensitivity_references):
    """Ordered alone can advance, and it must beat all5 fixed comparators."""
    pp = _predictions(primary, primary_predictions, primary_references)
    sp = _predictions(sensitivity, sensitivity_predictions, sensitivity_references)
    old_primary = {name: pp[name] for name in OLD_MODEL_NAMES}
    old_sensitivity = {name: sp[name] for name in OLD_MODEL_NAMES}
    pm, _, pc = old_evaluate.validate_population(primary, old_primary, expected_matched,
        expected_issuances=expected_issuances, expected_events=old_evaluate.SELECTION_EVENTS)
    sm, _, sc = old_evaluate.validate_population(sensitivity, old_sensitivity, expected_matched,
        expected_issuances=expected_issuances, expected_events=old_evaluate.SELECTION_EVENTS)
    old_evaluate._lag_parity(primary, sensitivity, old_primary, old_sensitivity)
    metrics, comparisons, checks, diagnostic = {}, {}, {}, {}
    for lag, frame, mask, predicted in (('2', primary.loc[pm], pm, pp), ('0', sensitivity.loc[sm], sm, sp)):
        points = {name: predicted[name][mask] for name in MODEL_NAMES}
        metrics[lag] = _metrics(frame, points)
        comparisons[lag] = {name: old_evaluate.comparison(frame, points[CANDIDATE], points[name]) for name in REFERENCES}
        checks[lag] = {name: old_evaluate._checks(comparisons[lag][name], .01) if lag == '2' else {
            'positive_relative_gain': (comparisons[lag][name]['relative_reduction'] is not None
                                       and comparisons[lag][name]['relative_reduction'] > 0)} for name in REFERENCES}
        diagnostic[lag] = {name: contributions(frame, points[CANDIDATE], points[name]) for name in REFERENCES}
    base = metrics['2']['base_hgb']['event_mae_seconds']
    if abs(base-old_models.EXPECTED_SELECTION_BASE_MAE) > 1e-12:
        raise ValueError('Closed old baseline does not reproduce its frozen selection MAE')
    advances = bool(all(flag for lag in checks.values() for ref in lag.values() for flag in ref.values()))
    return {'status': 'retrospective_sequence_research_no_promotion', 'model_names': list(MODEL_NAMES),
        'candidate_eligible_for_advancement': CANDIDATE, 'references': list(REFERENCES),
        'coverage': {'2': pc, '0': sc}, 'metrics': metrics, 'comparisons': comparisons,
        'gate_checks': checks, 'contributing_events_and_drivers': diagnostic,
        'primary_latency_seconds': 2, 'sensitivity_latency_seconds': 0, 'sensitivity_refitted': False,
        'old_references_preserved_bitwise': True, 'advances_to_later_evaluation': advances,
        'decision': 'advance_to_frozen_later_evaluation' if advances else 'stop_no_new_transfer_or_promotion',
        'promotion': False, 'baseline_reproduction': {'expected_event_mae_seconds': old_models.EXPECTED_SELECTION_BASE_MAE,
            'actual_event_mae_seconds': base, 'absolute_tolerance_seconds': 1e-12},
        'uncertainty': {'seed': 20260907, 'resamples_each': 20000, 'circular_event_block_length': 3,
            'interval': 'percentile95', 'sign': 'candidate minus reference',
            'limit': 'Historical event resampling does not remove previous adaptive research exposure.'}}
