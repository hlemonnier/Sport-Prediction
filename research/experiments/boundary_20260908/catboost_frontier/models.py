"""Fixed Plain/Ordered CatBoost comparison on the original 80 live inputs.

No data/model-file I/O or backend import occurs at module import. The runner
binds original inventories, the isolated backend and native serialized models.
Suggested commit: research(f1-live): compare fixed CatBoost boosting schemes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.experiments.boundary_20260908.telemetry import models as inherited

BASE_FEATURES = tuple(inherited.BASE_FEATURES)
KEYS = tuple(inherited.KEYS)
MODEL_NAMES = ('plain', 'ordered')
TRAIN_ROWS = 18_363
TRAIN_EVENTS = tuple(range(202201, 202223))
BACKEND_VERSION = '1.2.10'
COMMON_PARAMETERS = dict(loss_function='MAE', iterations=1000, depth=6,
    learning_rate=.03, l2_leaf_reg=10, grow_policy='SymmetricTree', task_type='CPU',
    thread_count=1, use_best_model=False, leaf_estimation_method='Exact',
    leaf_estimation_iterations=1, boost_from_average=True, bootstrap_type='No',
    random_strength=0, border_count=254, nan_mode='Min', has_time=False,
    random_seed=20260908, allow_writing_files=False, verbose=False)
TARGET_COLUMNS = ('lap_time_seconds', 'target_lap_number', 'target_timestamp', 'target_same_stint')


def parameters(name):
    if name not in MODEL_NAMES:
        raise ValueError('Only the frozen Plain and Ordered candidates are allowed')
    return {**COMMON_PARAMETERS, 'boosting_type': name.title()}


def _exact_ns(values, name):
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values):
        raise ValueError(name+' must contain exact integer nanoseconds')


def validate_original_issuances(frame, expected):
    """Independently supplied original inventory fixes the complete row order."""
    for data in (frame, expected):
        if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
            raise ValueError('A DataFrame with unique column names is required')
        inherited.validate_issuances(data)
        if 'issued_at_ns' not in data:
            raise ValueError('Exact original issuance clock is required')
        _exact_ns(data.issued_at_ns, 'issued_at_ns')
        if not data.driver_id.map(lambda v:isinstance(v,str) and bool(v)).all():
            raise ValueError('Driver keys must be nonempty strings')
    if len(frame) != len(expected):
        raise ValueError('Original issuance population changed')
    for column in (*KEYS, 'issuance_id', 'issued_at_ns'):
        if frame[column].tolist() != expected[column].tolist():
            raise ValueError('Original issuance identity/order changed: '+column)
    columns = ['forecast_naive_seconds', *BASE_FEATURES]
    left = np.ascontiguousarray(frame[columns].to_numpy(dtype=np.float64))
    right = np.ascontiguousarray(expected[columns].to_numpy(dtype=np.float64))
    if not np.array_equal(left.view(np.uint64),right.view(np.uint64)):
        raise ValueError('Original 80 features or naive anchor changed')


def _training_population(frame, expected):
    validate_original_issuances(frame, expected)
    if (len(frame) != TRAIN_ROWS or tuple(sorted(map(int,frame.event_key.unique()))) != TRAIN_EVENTS
            or 'year' not in frame or not frame.year.eq(2022).all()
            or 'outcome_status' not in frame or not frame.outcome_status.eq('matched').all()):
        raise ValueError('Training requires all 18363 matched 2022 rows and all 22 events')
    for data in (frame,expected):
        if any(name not in data for name in TARGET_COLUMNS):
            raise ValueError('Original matched target columns are required')
        values = data[['lap_time_seconds','target_lap_number','target_timestamp']].to_numpy(float)
        if (not np.isfinite(values).all() or (values[:,:2] <= 0).any()
                or not np.equal(values[:,1],np.floor(values[:,1])).all()
                or not (values[:,1] > data.issued_after_lap_number.to_numpy(float)).all()
                or not (values[:,2] > data.issued_at_timestamp.to_numpy(float)).all()):
            raise ValueError('Original targets must be finite, positive and strictly after issuance')
        if not all(isinstance(v,(bool,np.bool_)) for v in data.target_same_stint):
            raise ValueError('target_same_stint must contain explicit booleans')
    for column in TARGET_COLUMNS:
        if frame[column].tolist() != expected[column].tolist():
            raise ValueError('Original training target changed: '+column)
    for column in ('target_id','target_at_ns'):
        if column in expected:
            if column not in frame or frame[column].tolist() != expected[column].tolist():
                raise ValueError('Original training target identity changed: '+column)
            if column == 'target_at_ns':
                _exact_ns(frame[column],column)
                if not all(t > i for t,i in zip(frame[column],frame.issued_at_ns,strict=True)):
                    raise ValueError('Exact target clocks must follow issuance')


def event_weights(frame):
    """N/(E*n_event); each event has total weight N/E, global mean one."""
    events=frame.event_key
    counts=events.map(events.value_counts()).to_numpy(dtype=np.float64)
    if len(counts)==0 or not np.isfinite(counts).all() or (counts<=0).any():
        raise ValueError('Nonempty event membership is required')
    return len(frame)/(events.nunique()*counts)


def _backend():
    # The runner must prepend only its hash-bound isolated 1.2.10 runtime.
    import catboost
    if catboost.__version__ != BACKEND_VERSION:
        raise ValueError('Only the pinned CatBoost '+BACKEND_VERSION+' backend is allowed')
    return catboost.CatBoostRegressor


def fit_candidates(frame, expected_matched, backend_class=None):
    """Exactly two MAE fits; no eval_set, early stopping or population masks."""
    _training_population(frame,expected_matched)
    backend = _backend() if backend_class is None else backend_class
    if not callable(backend):
        raise TypeError('A CatBoost-compatible estimator class is required')
    x=frame[list(BASE_FEATURES)].copy()
    y=np.clip(frame.lap_time_seconds.to_numpy(float)-frame.forecast_naive_seconds.to_numpy(float),-5.,5.)
    weights=event_weights(frame)
    fitted={}
    with threadpool_limits(limits=1):
        for name in MODEL_NAMES:
            estimator=backend(**parameters(name))
            # New copies isolate each candidate even if a backend mutates fit arguments.
            estimator.fit(x.copy(deep=True),y.copy(),sample_weight=weights.copy())
            fitted[name]=estimator
    return fitted


def _target_free(frame):
    if not isinstance(frame,pd.DataFrame):
        raise ValueError('A target-free original issuance DataFrame is required')
    forbidden=[name for name in frame if name.startswith('target_')
               or name in ('lap_time_seconds','outcome_status','LapTime')]
    if forbidden:
        raise ValueError('Prediction inputs must be target-free: '+', '.join(forbidden))


def predict_candidates(models, frame, expected_issuances):
    """Predict every original issuance, including those without later targets."""
    _target_free(frame);_target_free(expected_issuances)
    validate_original_issuances(frame,expected_issuances)
    if not isinstance(models,dict) or set(models)!=set(MODEL_NAMES):
        raise ValueError('Exactly both frozen CatBoost estimators are required')
    x=frame[list(BASE_FEATURES)].copy()
    anchor=frame.forecast_naive_seconds.to_numpy(dtype=np.float64)
    predictions={}
    with threadpool_limits(limits=1):
        for name in MODEL_NAMES:
            model=models[name]
            if not callable(getattr(model,'predict',None)):
                raise ValueError('Each serialized candidate must be a predictive estimator')
            if tuple(getattr(model,'feature_names_',())) != BASE_FEATURES:
                raise ValueError('Serialized candidate must preserve the ordered original 80 feature names')
            if getattr(model,'tree_count_',None) != COMMON_PARAMETERS['iterations']:
                raise ValueError('Serialized candidate must contain all 1000 fixed iterations')
            delta=np.asarray(model.predict(x.copy(deep=True),thread_count=1),dtype=np.float64)
            if delta.shape!=(len(frame),) or not np.isfinite(delta).all():
                raise ValueError('One finite correction is required per original issuance')
            point=anchor+np.clip(delta,-3.,3.)
            if not np.isfinite(point).all() or (point<=0).any():
                raise ValueError('All original issuances require positive finite predictions')
            predictions[name]=point
    return predictions
