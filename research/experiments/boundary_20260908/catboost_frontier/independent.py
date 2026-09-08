"""Independent numeric CatBoost JSON inference and bound event uncertainty.

This module never imports CatBoost and never fits a model. It supports only
scalar MAE regression with numeric features, symmetric trees and NaN-Min.
Statistical resampling explicitly reuses the hash-bound earlier independent
verifier; it is independent of the new experiment's fitting/evaluation code,
not a newly independent implementation of that inherited bootstrap routine.
Suggested commit: research(f1-live): independently replay CatBoost JSON forecasts
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

STATISTICS_PATH = Path(__file__).resolve().parents[1] / 'telemetry_verification_v2/verify.py'
STATISTICS_SHA256 = '631115bdc2834f0c458e94cd18833f13d8c6bb0d9a190e76f7593d17da44c7b6'


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name+' must be an exact nonnegative integer')
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name+' must be a finite JSON number')
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(name+' must be finite') from exc
    if not np.isfinite(result):
        raise ValueError(name+' must be finite')
    return result


def _metadata(payload):
    info = payload.get('model_info', {})
    if not isinstance(info, dict):
        raise ValueError('model_info must be an object')
    params = info.get('params', {})
    if not isinstance(params, dict):
        raise ValueError('model_info.params must be an object')
    loss = params.get('loss_function', {})
    if not isinstance(loss, dict) or loss.get('type', 'MAE') != 'MAE':
        raise ValueError('Only scalar MAE regression metadata is supported')
    learner = params.get('tree_learner_options', {})
    if not isinstance(learner, dict) or learner.get('grow_policy', 'SymmetricTree') != 'SymmetricTree':
        raise ValueError('Only SymmetricTree metadata is supported')
    data = params.get('data_processing_options', {})
    if not isinstance(data, dict):
        raise ValueError('Invalid data_processing_options')
    bins = data.get('float_features_binarization', {})
    if not isinstance(bins, dict) or bins.get('nan_mode', 'Min') != 'Min':
        raise ValueError('Only NaN-Min metadata is supported')
    flat = params.get('flat_params', {})
    if not isinstance(flat, dict):
        raise ValueError('Invalid flat_params')
    for key, expected in (('loss_function', 'MAE'), ('grow_policy', 'SymmetricTree'), ('nan_mode', 'Min')):
        if key in flat and flat[key] != expected:
            raise ValueError('Unsupported flat parameter: '+key)


def json_predict(payload, x):
    """Return raw scalar corrections from exported JSON, before task clipping.

    CatBoost numeric inputs are float32 even when supplied as float64. The
    j-th split of an oblivious tree sets bit j of its leaf index when the
    rounded feature is strictly greater than its border. NaNs take false.
    The final prediction is scale * sum(tree leaves) + scalar bias.
    """
    if not isinstance(payload, dict):
        raise ValueError('Model JSON must be an object')
    allowed = {'features_info', 'model_info', 'oblivious_trees', 'scale_and_bias'}
    if set(payload)-allowed:
        raise ValueError('Unsupported model sections: '+str(sorted(set(payload)-allowed)))
    _metadata(payload)
    feature_info = payload.get('features_info')
    if not isinstance(feature_info, dict) or set(feature_info)-{'float_features'}:
        raise ValueError('Only numeric float_features are supported')
    features = feature_info.get('float_features')
    if not isinstance(features, list) or not features:
        raise ValueError('Nonempty numeric feature metadata is required')
    columns, borders, flat_indices = {}, {}, set()
    for feature in features:
        if not isinstance(feature, dict):
            raise ValueError('Invalid float feature')
        index = _integer(feature.get('feature_index'), 'feature_index')
        flat = _integer(feature.get('flat_feature_index'), 'flat_feature_index')
        if index in columns or flat in flat_indices:
            raise ValueError('Duplicate feature index')
        if feature.get('nan_value_treatment') not in ('AsFalse', 'AsIs'):
            raise ValueError('Only NaN-Min/AsFalse or unused-NaN/AsIs features are supported')
        if type(feature.get('has_nans')) is not bool:
            raise ValueError('has_nans must be a boolean')
        if feature['has_nans'] and feature['nan_value_treatment'] != 'AsFalse':
            raise ValueError('Observed NaNs must use AsFalse')
        raw_borders = feature.get('borders')
        if not isinstance(raw_borders, list):
            raise ValueError('Feature borders must be a list')
        values = [_number(v, 'feature border') for v in raw_borders]
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError('Feature borders must be strictly increasing')
        columns[index], borders[index] = flat, values
        flat_indices.add(flat)
    count = len(features)
    if set(columns) != set(range(count)) or flat_indices != set(range(count)):
        raise ValueError('Numeric feature indices must cover every input column')
    array = np.asarray(x)
    if array.ndim != 2 or array.shape[1] != count or array.dtype.kind not in 'fiu':
        raise ValueError('x must be a numeric rows-by-features matrix matching metadata')
    if np.isinf(array).any():
        raise ValueError('Infinite features are unsupported')
    with np.errstate(over='ignore', invalid='ignore'):
        values = array.astype(np.float32)
    if np.isinf(values).any():
        raise ValueError('Features must remain finite or NaN after float32 conversion')
    trees = payload.get('oblivious_trees')
    if not isinstance(trees, list) or not trees:
        raise ValueError('Nonempty symmetric trees are required')
    total = np.zeros(len(values), dtype=np.float64)
    for tree in trees:
        if not isinstance(tree, dict) or set(tree)-{'leaf_values', 'leaf_weights', 'splits'}:
            raise ValueError('Unsupported tree structure')
        splits = tree.get('splits')
        # A depth-zero constant tree is exported with a null split list.
        if splits is None:
            splits = []
        if not isinstance(splits, list) or len(splits) > 16:
            raise ValueError('Expected a symmetric tree of depth at most16')
        leaf_values = tree.get('leaf_values')
        if not isinstance(leaf_values, list) or len(leaf_values) != 1 << len(splits):
            raise ValueError('Scalar leaf count must equal 2**tree depth')
        leaves = np.array([_number(v, 'leaf value') for v in leaf_values], dtype=np.float64)
        if 'leaf_weights' in tree:
            weights = tree['leaf_weights']
            if not isinstance(weights, list) or len(weights) != len(leaves):
                raise ValueError('Leaf weights must match scalar leaf count')
            if any(_number(w, 'leaf weight') < 0 for w in weights):
                raise ValueError('Leaf weights cannot be negative')
        indices = np.zeros(len(values), dtype=np.uint32)
        for bit, split in enumerate(splits):
            if not isinstance(split, dict) or set(split) != {'border', 'float_feature_index', 'split_index', 'split_type'}:
                raise ValueError('Unsupported split metadata')
            if split['split_type'] != 'FloatFeature':
                raise ValueError('Only FloatFeature splits are supported')
            feature_index = _integer(split['float_feature_index'], 'float_feature_index')
            split_index = _integer(split['split_index'], 'split_index')
            if feature_index not in columns:
                raise ValueError('Split refers to an absent feature')
            border = _number(split['border'], 'split border')
            if border not in borders[feature_index]:
                raise ValueError('Split border is absent from feature metadata')
            expected_split = sum(len(borders[j]) for j in range(feature_index)) + borders[feature_index].index(border)
            if split_index != expected_split:
                raise ValueError('Global split index disagrees with numeric border inventory')
            # NaN > finite is False, implementing the frozen nan_mode=Min.
            indices |= (values[:, columns[feature_index]].astype(np.float64) > border).astype(np.uint32) << bit
        with np.errstate(over='ignore', invalid='ignore'):
            total += leaves[indices]
        if not np.isfinite(total).all():
            raise ValueError('Nonfinite raw tree sum')
    scale_bias = payload.get('scale_and_bias')
    if not isinstance(scale_bias, list) or len(scale_bias) != 2 or not isinstance(scale_bias[1], list) or len(scale_bias[1]) != 1:
        raise ValueError('Scalar scale_and_bias must be [scale,[bias]]')
    scale, bias = _number(scale_bias[0], 'scale'), _number(scale_bias[1][0], 'bias')
    with np.errstate(over='ignore', invalid='ignore'):
        result = total * scale + bias
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite scaled prediction')
    return result


def _statistics():
    if hashlib.sha256(STATISTICS_PATH.read_bytes()).hexdigest() != STATISTICS_SHA256:
        raise ValueError('Inherited independent statistics source hash changed')
    spec = importlib.util.spec_from_file_location('_catboost_bound_independent_statistics', STATISTICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison(events, y, candidate, reference):
    """Equal-event MAE with the frozen20k paired/event/block3/LOO routine.

    Six-digit event IDs define chronological order for the circular three-event
    block bootstrap. At least two events are required; rows never masquerade
    as independent event replicates. Absolute errors are computed on full,
    untrimmed targets supplied by the separately validated population join.
    """
    raw = np.asarray(events)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError('A nonempty one-dimensional event vector is required')
    normalized = []
    for item in raw.tolist():
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError('Event IDs must be canonical six-digit integers or strings')
        text = str(item)
        if len(text) != 6 or not text.isascii() or not text.isdecimal() or text[:4] == '0000' or not 1 <= int(text[4:]) <= 99:
            raise ValueError('Event IDs must be canonical six-digit year/round values')
        normalized.append(int(text))
    events = np.asarray(normalized, dtype=np.int64)
    if len(np.unique(events)) < 2:
        raise ValueError('At least two events are required for leave-one-event-out uncertainty')
    vectors = []
    for name, vector in (('target', y), ('candidate', candidate), ('reference', reference)):
        array = np.asarray(vector)
        if array.dtype.kind not in 'fiu' or array.shape != events.shape:
            raise ValueError(name+' must be a numeric vector aligned to events')
        array = array.astype(np.float64)
        if not np.isfinite(array).all():
            raise ValueError(name+' must be finite')
        vectors.append(array)
    y, candidate, reference = vectors
    with np.errstate(over='ignore', invalid='ignore'):
        errors = np.stack((np.abs(candidate-y), np.abs(reference-y)))
    if not np.isfinite(errors).all():
        raise ValueError('Absolute errors must remain finite')
    result = _statistics().independent_comparison(events, y, candidate, reference)
    result['candidate_row_mae'] = float(errors[0].mean())
    result['baseline_row_mae'] = float(errors[1].mean())
    return result
