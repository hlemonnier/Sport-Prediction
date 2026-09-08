"""Fixed two-model full-population CatBoost screen.

Suggested commit: research(f1-live): test symmetric and ordered boosting on the original horizon
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.experiments.boundary_20260908.telemetry import data, evaluate, run as parent_run

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/research/boundary_20260908/catboost_frontier'
PARENT = ROOT/'artifacts/research/boundary_20260908/telemetry/execution'
NAMES = ('plain', 'ordered')


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return data._read_json(path)
def write(path, value): return data._write_json(path, value)
def sha(path): return data.sha(path)
def record(path): return {'path': str(Path(path).resolve().relative_to(ROOT)), 'sha256': sha(path)}
def progress(**value): print(json.dumps(value, allow_nan=False), flush=True)


def check(item):
    path = ROOT/item['path']
    if sha(path) != item['sha256']: raise ValueError('Changed binding: '+str(path))
    return path


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<f8').tobytes()).hexdigest()


def sources():
    inherited = read(PARENT/'design_lock.json')['sources']
    paths = set(HERE.glob('*.py')) | set(HERE.glob('*.json')) | set(HERE.glob('*.md'))
    for path, value in inherited.items():
        if sha(ROOT/path) != value: raise ValueError('Inherited source changed: '+path)
        paths.add(ROOT/path)
    spec = read(HERE/'specification.json')
    p = ROOT/spec['verification']['statistical_implementation']
    if sha(p) != spec['verification']['statistical_source_sha256']: raise ValueError('Statistical source changed')
    paths.add(p)
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def parent_graph(spec):
    bindings = {}
    def add(path, expected=None):
        path = Path(path).resolve(); value = sha(path)
        if expected is not None and value != expected: raise ValueError('Parent input changed')
        bindings[str(path.relative_to(ROOT))] = value
        return record(path)
    parent = {}
    for name, key in [('data_lock.json','data_lock_sha256'), ('fit_lock.json','fit_lock_sha256'),
                      ('selection_issuance_lock.json','issuance_lock_sha256')]:
        add(PARENT/name, spec['inputs'][key])
    verification_path = PARENT/'verification/result_v2.json'
    add(verification_path, spec['inputs']['parent_verification_sha256'])
    verified = read(verification_path)
    if verified['status'] != 'passed': raise ValueError('Parent verification did not pass')
    add(PARENT/'design_lock.json')
    original = read(PARENT/'data_lock.json'); fitted = read(PARENT/'fit_lock.json')
    issued = read(PARENT/'selection_issuance_lock.json')
    parent['years'] = {}
    for year, expected, matched in [(2022,18788,18363),(2023,20432,20007)]:
        refs = original['years'][str(year)]['original_references']
        if [r['event_key'] for r in refs] != list(range(year*100+1,year*100+23)) or sum(r['rows'] for r in refs) != expected:
            raise ValueError('Original event or issuance population changed')
        for item in refs: add(check(item), item['sha256'])
        label_path = PARENT/f'labels_{year}/label_closure.json'; label = read(label_path)
        if year == 2022 and check(fitted['training_labels']) != label_path:
            raise ValueError('Training labels differ from the original fit closure')
        if label['all_issuances'] != expected or label['matched'] != matched: raise ValueError('Original label population changed')
        for item in label['events']: add(check(item), item['sha256'])
        parent['years'][str(year)] = {'original_references':refs, 'labels':add(label_path), 'rows':expected, 'matched':matched}
    parent['base_models'] = add(check(fitted['models']), fitted['models']['sha256'])
    parent['reference_forecasts'] = issued['forecasts']['2']
    add(check(parent['reference_forecasts']), parent['reference_forecasts']['sha256'])
    for path,value in bindings.items():
        if ROOT/path == verification_path: continue
        prior = verified['bindings'].get(path)
        if prior is None or value != prior['sha256'] or (ROOT/path).stat().st_size != prior['bytes']:
            raise ValueError('Original input differs from the pinned independent verification: '+path)
    parent['runtime_manifest'] = spec['inputs']['runtime_manifest']
    runtime_path = check(parent['runtime_manifest']); add(runtime_path)
    runtime = read(runtime_path)
    for table in ('files','wheels'):
        for path, value in runtime[table].items(): add(ROOT/path, value)
    add(check(runtime['synthetic_feasibility']))
    return parent, bindings


def resource_check(spec):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin': peak *= 1024
    if peak > spec['resources']['maximum_peak_rss_bytes']: raise MemoryError('Declared peak RSS exceeded')
    return peak


@contextmanager
def attempt(out, stage):
    write(Path(out)/(stage+'_attempt.json'), {'started_at_utc':now(), 'stage':stage})
    try: yield
    except Exception as exc:
        write(Path(out)/(stage+'_failure.json'), {'failed_at_utc':now(), 'error':str(exc), 'traceback':traceback.format_exc()})
        raise


def freeze(out, review_path):
    from . import models
    out = Path(out)
    with attempt(out, 'freeze'):
        spec = read(HERE/'specification.json'); before = sources(); review = read(review_path)
        if review.get('approved_for_execution_lock') is not True or review['source_files'] != before:
            raise ValueError('Exact-source independent approval required')
        if spec['models']['parameters'] != models.COMMON_PARAMETERS or tuple(spec['models']['names']) != NAMES:
            raise ValueError('Specified model settings differ from implementation')
        parent, bindings = parent_graph(spec)
        command = [sys.executable,'-m','pytest','-q','--import-mode=importlib','-p','no:cacheprovider',str(HERE)]
        tested = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        write(out/'pre_fit_tests.json', {'command':command,'exit_code':tested.returncode,'stdout':tested.stdout,
            'stderr':tested.stderr,'completed_at_utc':now(),'source_files':before})
        if tested.returncode or sources() != before: raise ValueError('Tests failed or source changed during freeze')
        for path,value in bindings.items():
            if sha(ROOT/path) != value: raise ValueError('Input changed during freeze')
        write(out/'design_lock.json', {'closed_at_utc':now(),'sources':before,'inputs':bindings,'parent':parent,
            'specification':record(HERE/'specification.json'),'review':record(review_path),'pre_fit_tests':record(out/'pre_fit_tests.json'),
            'historical_candidate_fits_before_lock':0,'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}})
        resource_check(spec)
        progress(stage='frozen',design=record(out/'design_lock.json'),input_bindings=len(bindings))


def verify_design(out):
    out = Path(out)
    if any(out.glob('*_failure.json')): raise ValueError('Failed execution cannot advance')
    lock = read(out/'design_lock.json')
    if sources() != lock['sources']: raise ValueError('Frozen source changed')
    for path,value in lock['inputs'].items():
        if sha(ROOT/path) != value: raise ValueError('Frozen input changed: '+path)
    for key in ('specification','review','pre_fit_tests'): check(lock[key])
    return lock,read(check(lock['specification']))


def original_year(lock, year):
    rows = []
    for item in lock['parent']['years'][str(year)]['original_references']:
        values = data._read_rows(check(item))
        if len(values) != item['rows']: raise ValueError('Original ledger count changed')
        rows.extend(values)
    if any(set(row)&set(parent_run.TARGET_COLUMNS) for row in rows): raise ValueError('Target entered original feature rows')
    return pd.DataFrame(rows)


def reference_predictions(lock, original):
    rows = data._read_rows(check(lock['parent']['reference_forecasts']))
    if len(rows) != len(original): raise ValueError('Baseline population changed')
    for saved,row in zip(rows,original.to_dict('records')):
        if any(saved[k] != row[k] for k in (*data.KEYS,'issuance_id','issued_at_ns')) or saved['lag_seconds'] != 2:
            raise ValueError('Baseline identity or issuance changed')
    result = np.array([r['predictions']['base_hgb'] for r in rows],dtype=np.float64)
    if not np.isfinite(result).all() or (result <= 0).any(): raise ValueError('Invalid saved baseline')
    return result


def score(frame, predictions):
    if len(frame) != 20432 or tuple(sorted(frame.event_key.unique())) != tuple(range(202301,202323)):
        raise ValueError('Full selection population changed')
    if not frame.outcome_status.isin(['matched','unmatched']).all(): raise ValueError('Unknown target status')
    matched = frame.outcome_status.eq('matched').to_numpy()
    if matched.sum() != 20007: raise ValueError('Matched population changed')
    for col in parent_run.TARGET_COLUMNS:
        if col != 'outcome_status' and frame.loc[~matched,col].notna().any(): raise ValueError('Unmatched targets must stay null')
    scored = frame.loc[matched]
    metrics = {}
    for name,p in predictions.items():
        if len(p) != len(frame) or not np.isfinite(p).all() or (p <= 0).any(): raise ValueError('All forecasts must be finite and positive')
        errors = np.abs(p[matched]-scored.lap_time_seconds.to_numpy(float))
        per = pd.DataFrame({'event_key':scored.event_key.to_numpy(),'error':errors}).groupby('event_key').error.mean()
        metrics[name] = {'event_mae_seconds':float(per.mean()),'row_mae_seconds':float(errors.mean()),'rows':20007,'events':22}
    comparisons = {name:evaluate.comparison(scored,predictions[name][matched],predictions['base_hgb'][matched]) for name in NAMES}
    winner = min(NAMES,key=lambda name:metrics[name]['event_mae_seconds'])
    checks = {name:evaluate._checks(comparisons[name],.01) for name in NAMES}
    return {'metrics':metrics,'comparisons':comparisons,'winner':winner,'checks':checks,
        'advances_to_later_evaluation':all(checks[winner].values()),'promotion':False,
        'rows_all':20432,'rows_matched':20007,'unmatched_retained':425}


def select(out):
    from . import models
    out = Path(out); lock,spec = verify_design(out)
    with attempt(out,'selection'):
        runtime = read(check(lock['parent']['runtime_manifest']))
        sys.path.insert(0,str(ROOT/runtime['isolated_site']))
        import catboost
        if catboost.__version__ != '1.2.10' or not Path(catboost.__file__).resolve().is_relative_to(ROOT/runtime['isolated_site']):
            raise ValueError('Wrong isolated CatBoost runtime')
        train_original = original_year(lock,2022)
        labeled = parent_run.join_labels(train_original,lock['parent']['years']['2022']['labels'])
        training = labeled.loc[labeled.outcome_status.eq('matched')].copy()
        fit_receipts = []
        class ObservedCatBoost(catboost.CatBoostRegressor):
            def fit(self, *args, **kwargs):
                start = now(); timer = time.perf_counter()
                progress(stage='fit_started',boosting_type=self.get_params()['boosting_type'])
                result = super().fit(*args,**kwargs)
                fit_receipts.append({'boosting_type':self.get_params()['boosting_type'],'started_at_utc':start,
                    'completed_at_utc':now(),'seconds':time.perf_counter()-timer,'peak_rss_bytes':resource_check(spec)})
                return result
        with threadpool_limits(limits=1): fitted = models.fit_candidates(training,training.copy(deep=True),backend_class=ObservedCatBoost)
        assets = {}
        for name,model in fitted.items():
            native = out/f'{name}.cbm'; exported = out/f'{name}.json'
            if native.exists() or exported.exists(): raise FileExistsError('Model outputs are exclusive')
            model.save_model(str(native),format='cbm');model.save_model(str(exported),format='json')
            assets[name] = {'native':record(native),'json':record(exported),'parameters':model.get_all_params(),'trees':model.tree_count_}
        write(out/'fit_lock.json',{'closed_at_utc':now(),'design_lock':record(out/'design_lock.json'),'models':assets,
            'training_labels':lock['parent']['years']['2022']['labels'],'fits':2,'base_refits':0,'rows_per_fit':len(training),
            'training_ids_sha256':data.digest(training.issuance_id.tolist()),
            'training_matrix_sha256':array_hash(training[list(models.BASE_FEATURES)].to_numpy(float)),
            'weights_sha256':array_hash(models.event_weights(training)),
            'target_sha256':array_hash(np.clip(training.lap_time_seconds.to_numpy(float)-training.forecast_naive_seconds.to_numpy(float),-5,5)),
            'fit_receipts':fit_receipts,'external_selection_labels_read':False})
        reloaded = {}
        for name,asset in assets.items():
            model = catboost.CatBoostRegressor();model.load_model(str(check(asset['native'])));reloaded[name] = model
        original = original_year(lock,2023)
        predictions = models.predict_candidates(reloaded,original,original.copy(deep=True))
        predictions['base_hgb'] = reference_predictions(lock,original)
        rows = [{**{k:r[k] for k in (*data.KEYS,'issuance_id','issued_at_ns')},'original_row_sha256':data.digest(r),
                 'predictions':{name:float(p[i]) for name,p in predictions.items()}} for i,r in enumerate(original.to_dict('records'))]
        forecast_path = out/'selection_forecasts.jsonl';data._write_rows(forecast_path,rows)
        verify_design(out);resource_check(spec)
        write(out/'forecast_lock.json',{'closed_at_utc':now(),'design_lock':record(out/'design_lock.json'),
            'fit_lock':record(out/'fit_lock.json'),'forecasts':record(forecast_path),'rows':len(rows),'external_selection_labels_read':False})
        progress(stage='all_forecasts_closed',rows=len(rows))
        # Only now are the already exposed external 2023 label rows attached.
        selection = parent_run.join_labels(original,lock['parent']['years']['2023']['labels'])
        summary = score(selection,predictions)
        verify_design(out);resource_check(spec)
        write(out/'selection.json',{'completed_at_utc':now(),'summary':summary,'design_lock':record(out/'design_lock.json'),
            'fit_lock':record(out/'fit_lock.json'),'forecast_lock':record(out/'forecast_lock.json'),
            'selection_labels':lock['parent']['years']['2023']['labels']})
        write(out/'selection_lock.json',{'closed_at_utc':now(),'selection':record(out/'selection.json'),
            'winner':summary['winner'],'advances_to_later_evaluation':summary['advances_to_later_evaluation'],'promotion':False})
        resource_check(spec)
        progress(stage='selection_complete',selection=record(out/'selection.json'),winner=summary['winner'],advances=summary['advances_to_later_evaluation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','select']);parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--review',type=Path);args = parser.parse_args()
    if args.stage == 'freeze':
        if args.review is None: parser.error('--review is required')
        freeze(args.out,args.review)
    else: select(args.out)


if __name__ == '__main__': main()
