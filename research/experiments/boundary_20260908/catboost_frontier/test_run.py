"""Synthetic runner boundaries; no historical rows, backend or fitted assets.

Suggested commit: test(f1-live): guard CatBoost source closure and forecast chronology
"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from . import models, run
from .test_models import frame, issued


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


@pytest.fixture
def graph(tmp_path, monkeypatch):
    """Complete small file graph with original population *metadata*, no data."""
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    parent = tmp_path/'parent'
    monkeypatch.setattr(run, 'PARENT', parent)
    put(parent/'design_lock.json', {'sources': {}, 'synthetic': True})
    original = {'years': {}}
    label_records = {}
    for year, total, matched in ((2022, 18788, 18363), (2023, 20432, 20007)):
        references, labels = [], []
        for i in range(22):
            event = year*100+i+1
            path = put(parent/f'original/{event}.jsonl', {'synthetic': True})
            references.append({**run.record(path), 'event_key': event, 'rows': total-21 if i == 0 else 1})
            path = put(parent/f'labels_{year}/{event}.jsonl', {'synthetic': True})
            labels.append({**run.record(path), 'event_key': event})
        original['years'][str(year)] = {'original_references': references}
        path = put(parent/f'labels_{year}/label_closure.json', {
            'all_issuances': total, 'matched': matched, 'events': labels})
        label_records[year] = run.record(path)
    put(parent/'data_lock.json', original)
    base = put(parent/'models.pkl', {'synthetic_asset_not_pickle': True})
    put(parent/'fit_lock.json', {'models': run.record(base), 'training_labels': label_records[2022]})
    forecasts = put(parent/'forecasts.jsonl', {'synthetic_forecast_marker': True})
    put(parent/'selection_issuance_lock.json', {'forecasts': {'2': run.record(forecasts)}})
    runtime_file = put(tmp_path/'runtime/site/backend.py', {'synthetic': True})
    wheel = put(tmp_path/'runtime/backend.whl', {'synthetic_not_wheel': True})
    feasibility = put(tmp_path/'runtime/feasibility.json', {'synthetic': True})
    runtime = put(tmp_path/'runtime/manifest.json', {
        'isolated_site': 'runtime/site',
        'files': {run.record(runtime_file)['path']: run.sha(runtime_file)},
        'wheels': {run.record(wheel)['path']: run.sha(wheel)},
        'synthetic_feasibility': run.record(feasibility)})
    verified = {str(path.relative_to(tmp_path)): {'sha256': run.sha(path), 'bytes': path.stat().st_size}
                for path in parent.rglob('*') if path.is_file()}
    receipt = put(parent/'verification/result_v2.json', {'status': 'passed', 'bindings': verified})
    spec = {'inputs': {'data_lock_sha256': run.sha(parent/'data_lock.json'),
        'fit_lock_sha256': run.sha(parent/'fit_lock.json'),
        'issuance_lock_sha256': run.sha(parent/'selection_issuance_lock.json'),
        'parent_verification_sha256': run.sha(receipt), 'runtime_manifest': run.record(runtime)}}
    return SimpleNamespace(root=tmp_path, parent=parent, spec=spec, verified=verified,
                           runtime_file=runtime_file, wheel=wheel)


def test_parent_graph_reads_only_closed_metadata_and_hashes(graph, monkeypatch):
    original_read = run.read
    def metadata_only(path):
        assert Path(path).suffix not in ('.jsonl', '.pkl', '.whl')
        return original_read(path)
    monkeypatch.setattr(run, 'read', metadata_only)
    parent, bindings = run.parent_graph(graph.spec)
    assert parent['years']['2022']['rows'] == 18788
    assert parent['years']['2023']['matched'] == 20007
    for path in graph.verified:
        assert path in bindings
    assert str(graph.runtime_file.relative_to(graph.root)) in bindings
    assert str(graph.wheel.relative_to(graph.root)) in bindings


@pytest.mark.parametrize('what', ['original', 'label', 'runtime', 'wheel'])
def test_parent_graph_rejects_consumed_file_hash_drift(graph, what):
    path = {'original': graph.parent/'original/202201.jsonl',
            'label': graph.parent/'labels_2023/202301.jsonl',
            'runtime': graph.runtime_file, 'wheel': graph.wheel}[what]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        run.parent_graph(graph.spec)


@pytest.mark.parametrize('year', [2022, 2023])
def test_rewritten_label_and_closure_cannot_replace_verified_original(graph, year):
    """Updating the inner receipt must not re-bless changed target bytes."""
    payload = graph.parent/f'labels_{year}/{year}01.jsonl'
    put(payload, {'synthetic': 'changed target'})
    closure_path = graph.parent/f'labels_{year}/label_closure.json'
    closure = run.read(closure_path)
    closure['events'][0].update(run.record(payload))
    put(closure_path, closure)
    with pytest.raises(ValueError):
        run.parent_graph(graph.spec)


def test_original_year_refuses_target_columns_before_model_use(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    path = tmp_path/'original.jsonl'
    run.data._write_rows(path, [{'issuance_id': 'synthetic', 'lap_time_seconds': 90.}])
    lock = {'parent': {'years': {'2023': {'original_references': [{**run.record(path), 'rows': 1}]}}}}
    with pytest.raises(ValueError, match='Target entered'):
        run.original_year(lock, 2023)


@pytest.mark.parametrize('fault', ['order', 'clock', 'lag', 'count', 'nonfinite'])
def test_saved_baseline_requires_original_identity_order_and_finite_values(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    original = issued(frame(3)).assign(year=2023)
    rows = [{**{key: row[key] for key in (*run.data.KEYS, 'issuance_id', 'issued_at_ns')},
             'lag_seconds': 2, 'predictions': {'base_hgb': 90.}} for row in original.to_dict('records')]
    if fault == 'order': rows.reverse()
    if fault == 'clock': rows[0]['issued_at_ns'] += 1
    if fault == 'lag': rows[0]['lag_seconds'] = 0
    if fault == 'count': rows.pop()
    if fault == 'nonfinite': rows[0]['predictions']['base_hgb'] = None
    path = tmp_path/'saved.jsonl'
    run.data._write_rows(path, rows)
    with pytest.raises((ValueError, TypeError)):
        run.reference_predictions({'parent': {'reference_forecasts': run.record(path)}}, original)


def test_attempt_failure_and_retry_are_exclusive(tmp_path):
    with pytest.raises(RuntimeError, match='synthetic crash'):
        with run.attempt(tmp_path, 'selection'):
            raise RuntimeError('synthetic crash')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    entered = False
    with pytest.raises(FileExistsError):
        with run.attempt(tmp_path, 'selection'):
            entered = True
    assert not entered
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_failed_execution_is_rejected_before_design_or_inputs_read(tmp_path, monkeypatch):
    put(tmp_path/'selection_failure.json', {'error': 'synthetic'})
    monkeypatch.setattr(run, 'read', lambda _: pytest.fail('A failed execution read further inputs'))
    with pytest.raises(ValueError, match='Failed execution'):
        run.verify_design(tmp_path)


def selection_frame():
    """Unequal event sizes expose accidental row-weighted selection."""
    events = np.array([202301]*(20007-21)+list(range(202302, 202323))+[202301]*425)
    matched = np.arange(20432) < 20007
    result = pd.DataFrame({'event_key': events, 'outcome_status': np.where(matched, 'matched', 'unmatched')})
    for column in run.parent_run.TARGET_COLUMNS:
        if column != 'outcome_status': result[column] = pd.Series([None]*20432, dtype=object)
    result.loc[matched, 'lap_time_seconds'] = 90.
    for column in ('target_lap_number', 'target_timestamp', 'target_at_ns'): result.loc[matched, column] = 100
    result.loc[matched, 'target_id'] = ['synthetic-'+str(i) for i in range(20007)]
    result.loc[matched, 'target_same_stint'] = True
    error = np.where(events == 202301, 1., 2.)
    predictions = {'base_hgb': 90.+error, 'plain': 90.+.98*error, 'ordered': 90.+.985*error}
    return result, predictions


@pytest.fixture
def comparisons(monkeypatch):
    base = {'relative_reduction': .02, 'event_ci95': [-.03, -.01],
            'block3_ci95': [-.04, -.005], 'loo_max_delta': -.009}
    monkeypatch.setattr(run.evaluate, 'comparison', lambda *args: copy.deepcopy(base))
    return base


def test_selection_uses_event_means_retains_unmatched_and_all_candidates(comparisons):
    frame, predictions = selection_frame()
    report = run.score(frame, predictions)
    assert report['metrics']['base_hgb']['event_mae_seconds'] == pytest.approx(43/22)
    assert report['metrics']['base_hgb']['row_mae_seconds'] != pytest.approx(43/22)
    assert report['winner'] == 'plain' and report['advances_to_later_evaluation'] is True
    assert set(report['metrics']) == {'plain', 'ordered', 'base_hgb'}
    assert report['rows_all'] == 20432 and report['rows_matched'] == 20007
    assert report['unmatched_retained'] == 425 and report['promotion'] is False
    predictions['ordered'] = predictions['plain'].copy()
    assert run.score(frame, predictions)['winner'] == 'plain'


@pytest.mark.parametrize('key,value', [('relative_reduction', .009999),
    ('event_ci95', [-.1, 0.]), ('block3_ci95', [-.1, 0.]), ('loo_max_delta', 0.)])
def test_each_frozen_gate_can_stop_later_evaluation(comparisons, key, value):
    comparisons[key] = value
    report = run.score(*selection_frame())
    assert report['advances_to_later_evaluation'] is False


@pytest.mark.parametrize('fault', ['drop', 'event', 'status', 'unmatched_target', 'prediction'])
def test_score_rejects_population_loss_or_invented_unmatched_target(comparisons, fault):
    frame, predictions = selection_frame()
    if fault == 'drop': frame = frame.iloc[:-1]
    if fault == 'event': frame.loc[frame.event_key.eq(202322), 'event_key'] = 202321
    if fault == 'status': frame.loc[0, 'outcome_status'] = 'unknown'
    if fault == 'unmatched_target': frame.loc[20431, 'lap_time_seconds'] = 90.
    if fault == 'prediction': predictions['plain'][-1] = np.nan
    with pytest.raises(ValueError): run.score(frame, predictions)


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """Real runner serialization, model wrapper and exclusive files; fake backend."""
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(models, 'TRAIN_ROWS', 8)
    monkeypatch.setattr(models, 'TRAIN_EVENTS', (202201, 202202))
    out = tmp_path/'execution'
    put(out/'design_lock.json', {'synthetic': True})
    runtime = put(tmp_path/'runtime.json', {'isolated_site': 'isolated'})
    labels = {year: {'path': f'labels_{year}.json', 'sha256': str(year)} for year in (2022, 2023)}
    lock = {'parent': {'runtime_manifest': run.record(runtime), 'years': {
        str(year): {'labels': labels[year]} for year in (2022, 2023)}}}
    spec = {'resources': {'maximum_peak_rss_bytes': 2**50}}
    monkeypatch.setattr(run, 'verify_design', lambda _: (lock, spec))
    monkeypatch.setattr(run, 'progress', lambda **kw: None)
    train = frame()
    query = issued(frame(5, (202301, 202302))).assign(year=2023)
    originals = {2022: issued(train), 2023: query}
    monkeypatch.setattr(run, 'original_year', lambda _, year: originals[year].copy(deep=True))
    events = []
    def join(original, label):
        year = next(year for year in labels if labels[year] == label)
        events.append('labels_'+str(year))
        if year == 2022: return train.copy(deep=True)
        assert (out/'forecast_lock.json').is_file()
        closed = run.read(out/'forecast_lock.json')
        assert closed['external_selection_labels_read'] is False
        forecasts = run.data._read_rows(run.check(closed['forecasts']))
        assert len(forecasts) == len(query)
        assert all(set(r['predictions']) == {'base_hgb', 'plain', 'ordered'} for r in forecasts)
        return original.assign(outcome_status=['matched']*4+['unmatched'])
    monkeypatch.setattr(run.parent_run, 'join_labels', join)
    class Backend:
        def __init__(self, **parameters): self.parameters = parameters
        def get_params(self): return self.parameters
        def get_all_params(self): return self.parameters
        def fit(self, x, y, **kwargs):
            events.append('fit_'+self.parameters['boosting_type'])
            assert not (out/'forecast_lock.json').exists() and 'labels_2023' not in events
            assert set(kwargs) == {'sample_weight'} and len(x) == 8
            self.feature_names_ = list(x.columns); self.tree_count_ = 1000
            return self
        def save_model(self, path, format):
            with Path(path).open('x') as f: json.dump(self.parameters, f)
        def load_model(self, path):
            self.parameters = json.loads(Path(path).read_text())
            self.feature_names_ = list(models.BASE_FEATURES); self.tree_count_ = 1000
        def predict(self, x, *, thread_count):
            events.append('predict_'+self.parameters['boosting_type'])
            assert 'labels_2023' not in events and len(x) == 5 and thread_count == 1
            return np.full(len(x), .1)
    fake = SimpleNamespace(__version__='1.2.10', __file__=str(tmp_path/'isolated/catboost/__init__.py'),
                           CatBoostRegressor=Backend)
    monkeypatch.setitem(sys.modules, 'catboost', fake)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    baseline = np.array([90., np.nextafter(90., np.inf), 91., 92., 93.])
    monkeypatch.setattr(run, 'reference_predictions', lambda *_: baseline.copy())
    def score(frame, predictions):
        events.append('score')
        assert frame.outcome_status.tolist() == ['matched']*4+['unmatched']
        np.testing.assert_array_equal(predictions['base_hgb'].view(np.uint64), baseline.view(np.uint64))
        return {'winner': 'plain', 'advances_to_later_evaluation': False, 'promotion': False}
    monkeypatch.setattr(run, 'score', score)
    return SimpleNamespace(out=out, events=events, fake=fake, original=query, originals=originals)


def test_selection_fits_twice_and_closes_all_forecasts_before_external_labels(stage):
    run.select(stage.out)
    assert stage.events == ['labels_2022', 'fit_Plain', 'fit_Ordered',
        'predict_Plain', 'predict_Ordered', 'labels_2023', 'score']
    fit = run.read(stage.out/'fit_lock.json')
    assert fit['fits'] == 2 and fit['base_refits'] == 0 and fit['rows_per_fit'] == 8
    assert len(fit['fit_receipts']) == 2 and fit['external_selection_labels_read'] is False
    closed = run.read(stage.out/'selection_lock.json')
    assert closed['advances_to_later_evaluation'] is False and closed['promotion'] is False
    before = {p.name: p.read_bytes() for p in stage.out.iterdir()}
    with pytest.raises(FileExistsError): run.select(stage.out)
    assert before == {p.name: p.read_bytes() for p in stage.out.iterdir()}
    assert stage.events.count('fit_Plain') == stage.events.count('fit_Ordered') == 1


def test_forecast_write_failure_never_opens_selection_labels(stage, monkeypatch):
    def fail(*args): raise OSError('synthetic disk failure')
    monkeypatch.setattr(run.data, '_write_rows', fail)
    with pytest.raises(OSError, match='disk failure'): run.select(stage.out)
    assert 'labels_2023' not in stage.events and 'score' not in stage.events
    assert (stage.out/'selection_failure.json').exists()
    assert not (stage.out/'forecast_lock.json').exists()
    assert not (stage.out/'selection_lock.json').exists()


@pytest.mark.parametrize('fault', ['version', 'location'])
def test_wrong_backend_fails_before_any_label_or_fit(stage, fault):
    if fault == 'version': stage.fake.__version__ = 'different'
    else: stage.fake.__file__ = '/not-the-isolated-runtime/catboost.py'
    with pytest.raises(ValueError, match='isolated CatBoost'): run.select(stage.out)
    assert stage.events == []


@pytest.mark.parametrize('changed', ['sources', 'inputs'])
def test_freeze_rejects_drift_during_tests(tmp_path, monkeypatch, changed):
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    here = tmp_path/'source'; monkeypatch.setattr(run, 'HERE', here)
    source = put(here/'example.py', {'synthetic': 'source'})
    spec_path = put(here/'specification.json', {'models': {'parameters': models.COMMON_PARAMETERS, 'names': list(run.NAMES)}})
    source_map = lambda: {run.record(p)['path']: run.sha(p) for p in (source, spec_path)}
    review = put(tmp_path/'review.json', {'approved_for_execution_lock': True, 'source_files': source_map()})
    consumed = put(tmp_path/'consumed.json', {'synthetic': 'input'})
    monkeypatch.setattr(run, 'sources', source_map)
    monkeypatch.setattr(run, 'parent_graph', lambda _: ({}, {run.record(consumed)['path']: run.sha(consumed)}))
    def fake_tests(*args, **kwargs):
        path = source if changed == 'sources' else consumed
        path.write_bytes(path.read_bytes()+b' ')
        return SimpleNamespace(returncode=0, stdout='synthetic tests passed', stderr='')
    monkeypatch.setattr(run.subprocess, 'run', fake_tests)
    out = tmp_path/'out'
    with pytest.raises(ValueError): run.freeze(out, review)
    assert (out/'pre_fit_tests.json').exists() and (out/'freeze_failure.json').exists()
    assert not (out/'design_lock.json').exists()
