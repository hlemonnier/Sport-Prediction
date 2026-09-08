"""Two fixed gap fits with a reused, never refitted, 2022 base forecaster.

This module performs no raw/label/model-file I/O. The runner must bind the
supplied original inventories and saved model/forecast objects to closed files.
Suggested commit: research(f1-live): test causal rival-gap magnitude and dynamics
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

BASE_FEATURES = tuple(old_models.BASE_FEATURES)
KEYS = tuple(old_models.KEYS)
MODEL_NAMES = ('base_hgb', 'gap_hgb', 'quality_hgb')
NEW_MODEL_NAMES = ('gap_hgb', 'quality_hgb')
CONFIG = {'kind': 'hgb', 'leaves': 15, 'iterations': 150}
TRAIN_ROWS = 18_363
TRAIN_EVENTS = tuple(range(202201, 202223))
BASE_PARAMETERS = dict(loss='absolute_error', learning_rate=.06, max_iter=150,
                       max_leaf_nodes=15, min_samples_leaf=80,
                       l2_regularization=10, early_stopping=False, random_state=20260907)


@dataclass(frozen=True)
class SavedBaseForecasts:
    issuance_ids: tuple[str, ...]
    cutoff_ns: tuple[int, ...]
    predictions: np.ndarray
    source_sha256: str


def _sha(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('A lowercase SHA256 binding is required')
    return value


def layout():
    from .features import FEATURE_NAMES, CONTENT_INDICES, QUALITY_INDICES
    names = tuple(FEATURE_NAMES)
    content, quality = tuple(CONTENT_INDICES), tuple(QUALITY_INDICES)
    if (len(names) != 45 or len(set(names)) != 45
            or content != tuple(range(12)) or quality != tuple(range(12, 45))
            or names[39] != 'quality_supported'):
        raise ValueError('The frozen ordered 12 content / 33 quality layout is required')
    return names, content, quality


def _latency(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (0, 2):
        raise ValueError('Only integer 2-second primary or 0-second sensitivity is allowed')


def matrices(frame, *, latency_seconds):
    """Return base80, gap125, quality125 and the unchanged pilot support mask."""
    _latency(latency_seconds)
    old_models.validate_issuances(frame)
    if any(c not in frame for c in ('issued_at_ns', 'gap_cutoff_ns', 'gap_supported', 'gap_values')):
        raise ValueError('Missing gap values, exact clocks or explicit support')
    for issued, cutoff in zip(frame.issued_at_ns, frame.gap_cutoff_ns, strict=True):
        if (isinstance(issued, (bool, np.bool_)) or not isinstance(issued, (int, np.integer))
                or isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, (int, np.integer))
                or int(issued)-int(cutoff) != int(latency_seconds)*1_000_000_000):
            raise ValueError('Gap cutoff must retain the exact integer nanosecond lag')
    support = old_models.support_mask(frame.gap_supported, len(frame))
    names, content, quality = layout()
    values = np.asarray(frame.gap_values.tolist(), dtype=np.float64)
    if (values.shape != (len(frame), 45) or np.isinf(values).any()
            or not np.isfinite(values[:, quality]).all()):
        raise ValueError('Gap content may be NaN; all 33 quality coordinates must be finite')
    if not np.array_equal(values[:, names.index('quality_supported')], support.astype(float)):
        raise ValueError('Quality support coordinate must equal the exact pilot support bit')
    columns = ['gap__'+name for name in names]
    if set(columns) & set(BASE_FEATURES):
        raise ValueError('Gap/base feature collision')
    base = frame[list(BASE_FEATURES)].copy()
    augmented = pd.concat([base, pd.DataFrame(values, index=frame.index, columns=columns)], axis=1)
    control = augmented.copy()
    control.loc[:, [columns[i] for i in content]] = 0.
    return base, augmented, control, support


def validate_original_issuances(frame, expected):
    """An independently supplied original inventory fixes IDs, order and inputs."""
    old_models.validate_issuances(frame)
    old_models.validate_issuances(expected)
    actual, original = old_evaluate._keyed(frame), old_evaluate._keyed(expected)
    if len(actual) != len(original) or not actual.index.equals(original.index):
        raise ValueError('Original issuance identities/order changed')
    old_evaluate._assert_original_values(actual, original,
        ['issuance_id', 'issued_at_ns', 'forecast_naive_seconds', *BASE_FEATURES], population='issuance')


def _training_population(frame, expected):
    validate_original_issuances(frame, expected)
    if (len(frame) != TRAIN_ROWS or len(expected) != TRAIN_ROWS
            or 'year' not in frame or not frame.year.eq(2022).all()
            or tuple(sorted(map(int, frame.event_key.unique()))) != TRAIN_EVENTS
            or 'outcome_status' not in frame or not frame.outcome_status.eq('matched').all()):
        raise ValueError('Training requires all 18363 original matched 2022 rows and all 22 events')
    actual, original = old_evaluate._keyed(frame), old_evaluate._keyed(expected)
    columns = ['lap_time_seconds', 'target_lap_number', 'target_timestamp', 'target_same_stint']
    # The old original matched table need not contain the later ledger's derived
    # target ID/nanosecond columns. If supplied, bind those too.
    columns += [c for c in ('target_id', 'target_at_ns') if c in original]
    old_evaluate._assert_original_values(actual, original, columns, population='training')
    target = frame.lap_time_seconds.to_numpy(float)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError('Training outcomes must be finite positive lap times')


def _saved_base(model, source_sha256):
    _sha(source_sha256)
    if (not isinstance(model, dict) or model.get('kind') != 'hgb'
            or model.get('features') != list(BASE_FEATURES)):
        raise ValueError('The saved 2022 base must use the exact original 80-column HGB layout')
    estimator = model.get('model')
    if not callable(getattr(estimator, 'predict', None)) or not callable(getattr(estimator, 'get_params', None)):
        raise ValueError('A saved base HGB estimator is required')
    parameters = estimator.get_params()
    if any(parameters.get(k) != value for k, value in BASE_PARAMETERS.items()):
        raise ValueError('The saved base configuration differs from the frozen 2022 HGB')
    if getattr(estimator, 'n_features_in_', 80) != 80:
        raise ValueError('The saved base estimator width differs from its feature declaration')
    return model


def fit_models(frame, *, expected_matched, saved_base_model, saved_base_model_sha256):
    """Exactly two fits on every original matched 2022 row, supported or not."""
    _training_population(frame, expected_matched)
    saved = _saved_base(saved_base_model, saved_base_model_sha256)
    _, augmented, control, support = matrices(frame, latency_seconds=2)
    fitted = {'base_hgb': saved}
    with threadpool_limits(limits=1):
        for name, x in zip(NEW_MODEL_NAMES, (augmented, control), strict=True):
            fitted[name] = old_models.original.fit_model(
                old_models._training_frame(frame, x), dict(CONFIG), list(x.columns))
    return {'models': fitted, 'base_features': list(BASE_FEATURES),
        'gap_features': list(layout()[0]), 'saved_base_model_sha256': saved_base_model_sha256,
        'fit_summary': {'fit_year': 2022, 'fit_latency_seconds': 2, 'rows_per_fit': len(frame),
            'events': list(TRAIN_EVENTS), 'fits': 2, 'base_refits': 0,
            'configuration': dict(CONFIG), 'supported_training_rows': int(support.sum()),
            'unsupported_training_rows_retained': int((~support).sum()),
            'issuance_ids_sha256': hashlib.sha256(json.dumps(frame.issuance_id.tolist(), separators=(',', ':')).encode()).hexdigest(),
            'weighting': 'N/(E*n_event) on all original matched 2022 rows',
            'baseline': 'Saved 2022-only parameters; not the deployed 2022-2023 parameter asset'}}


def _bundle(bundle):
    if not isinstance(bundle, dict):
        raise ValueError('A frozen gap model bundle is required')
    summary = bundle.get('fit_summary', {})
    if (set(bundle.get('models', {})) != set(MODEL_NAMES)
            or bundle.get('base_features') != list(BASE_FEATURES)
            or bundle.get('gap_features') != list(layout()[0])
            or summary.get('fit_year') != 2022 or summary.get('fit_latency_seconds') != 2
            or summary.get('rows_per_fit') != TRAIN_ROWS or summary.get('events') != list(TRAIN_EVENTS)
            or summary.get('fits') != 2 or summary.get('base_refits') != 0
            or summary.get('configuration') != CONFIG):
        raise ValueError('Prediction bundle differs from the fixed two-fit contract')
    _saved_base(bundle['models']['base_hgb'], bundle.get('saved_base_model_sha256'))
    expected_features = [*BASE_FEATURES, *('gap__'+n for n in layout()[0])]
    for name in NEW_MODEL_NAMES:
        model = bundle['models'][name]
        if model.get('kind') != 'hgb' or model.get('features') != expected_features:
            raise ValueError('Challenger feature order or model identity changed')


def reference_values(frame, reference):
    if not isinstance(reference, SavedBaseForecasts):
        raise TypeError('Closed SavedBaseForecasts are required')
    _sha(reference.source_sha256)
    if (not isinstance(reference.issuance_ids, tuple)
            or reference.issuance_ids != tuple(frame.issuance_id)):
        raise ValueError('Saved base issuance IDs must match exactly in order')
    if (not isinstance(reference.cutoff_ns, tuple) or len(reference.cutoff_ns) != len(frame)
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in reference.cutoff_ns)
            or reference.cutoff_ns != tuple(frame.gap_cutoff_ns)):
        raise ValueError('Saved base cutoffs must match exact current issuance clocks')
    values = reference.predictions
    if (not isinstance(values, np.ndarray) or values.dtype != np.float64
            or values.shape != (len(frame),) or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError('Saved base forecasts must retain positive finite float64 values')
    return values.copy()


def _target_free(frame):
    forbidden = [c for c in frame if c.startswith('target_') or c in ('lap_time_seconds', 'outcome_status', 'LapTime')]
    if forbidden:
        raise ValueError('Prediction inputs must be target-free: '+', '.join(forbidden))


def predict_models(bundle, frame, *, latency_seconds, expected_issuances, saved_base_forecasts=None):
    """Forecast every supplied issuance; unsupported points retain base bits.

    2023 always requires the closed old reference table. Optional base estimator
    inference is restricted to 2022 and cannot silently regenerate 2023 points.
    """
    _target_free(frame)
    _target_free(expected_issuances)
    validate_original_issuances(frame, expected_issuances)
    base, augmented, control, support = matrices(frame, latency_seconds=latency_seconds)
    _bundle(bundle)
    years = set((frame.event_key.to_numpy(dtype=np.int64)//100).tolist())
    if not years <= {2022, 2023}:
        raise ValueError('This discovery bundle only accepts the declared 2022-2023 scope')
    if saved_base_forecasts is None and years != {2022}:
        raise ValueError('2023 inference requires the exact closed base forecasts')
    with threadpool_limits(limits=1):
        if saved_base_forecasts is None:
            base['forecast_naive_seconds'] = frame.forecast_naive_seconds.to_numpy()
            baseline = np.asarray(old_models.original.predict_model(base, bundle['models']['base_hgb']), dtype=np.float64)
        else:
            baseline = reference_values(frame, saved_base_forecasts)
        result = {'base_hgb': baseline}
        for name, x in zip(NEW_MODEL_NAMES, (augmented, control), strict=True):
            prediction = baseline.copy()
            if support.any():
                available = x.loc[support].copy()
                available['forecast_naive_seconds'] = frame.loc[support, 'forecast_naive_seconds'].to_numpy()
                prediction[support] = old_models.original.predict_model(available, bundle['models'][name])
            result[name] = prediction
    if any(v.shape != (len(frame),) or not np.isfinite(v).all() or (v <= 0).any() for v in result.values()):
        raise ValueError('Every issuance requires three finite positive forecasts')
    return result
