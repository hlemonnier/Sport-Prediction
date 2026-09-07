"""Bounded causal race-order performance research; never changes runtime policy."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = Path(__file__).with_name('specification.json')
FEATURES = json.loads(SPEC_PATH.read_text())['features']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranks(values, ids):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(set(ids)) != len(ids):
        raise ValueError('rank inputs must be finite with unique identities')
    order = np.lexsort((np.asarray(ids, dtype=str), values))
    out = np.empty(len(order), dtype=float)
    out[order] = np.arange(1, len(order) + 1)
    return out


@dataclass
class Event:
    key: str
    year: int
    round_number: int
    circuit: str
    weekend_format: str
    ids: list[str]
    teams: list[str]
    qualifying: np.ndarray
    target: np.ndarray
    classified_finish: np.ndarray


def load_events():
    events, manifest, excluded = [], {}, []
    for folder in sorted((ROOT/'data/f1/raw/weekends').glob('20*/round_*')):
        year = int(folder.parent.name)
        if not 2022 <= year <= 2026:
            continue
        number = int(folder.name.split('_')[1])
        key = f'{year}:{number:02d}'
        qp = sorted(p for p in folder.glob('*_qualifying_results.csv') if re.fullmatch(r'\d{2}_qualifying_results\.csv', p.name))
        rp = sorted(p for p in folder.glob('*_race_results.csv') if re.fullmatch(r'\d{2}_race_results\.csv', p.name))
        mp = folder/'weekend_metadata.json'
        for path in [*qp, *rp, mp]:
            if path.exists(): manifest[str(path.relative_to(ROOT))] = digest(path)
        try:
            if len(qp) != 1 or len(rp) != 1:
                raise ValueError('exactly one canonical Q/Race classification required')
            q, r = pd.read_csv(qp[0]), pd.read_csv(rp[0])
            for frame in [q, r]:
                if frame['Abbreviation'].isna().any() or frame['Abbreviation'].duplicated().any():
                    raise ValueError('missing or duplicate driver identities')
            if set(q.Abbreviation.astype(str)) != set(r.Abbreviation.astype(str)):
                raise ValueError('qualifying and race rosters differ')
            q = q.sort_values('Abbreviation').reset_index(drop=True)
            r = r.set_index(r.Abbreviation.astype(str)).loc[q.Abbreviation.astype(str)]
            ids = q.Abbreviation.astype(str).tolist()
            qp_values = pd.to_numeric(q.Position, errors='coerce').to_numpy(float)
            # Source order is not used to resolve target ties or missing target positions.
            target = pd.to_numeric(r.Position, errors='coerce').to_numpy(float)
            if not np.isfinite(qp_values).all() or (qp_values < 1).any() or len(set(qp_values)) != len(q):
                raise ValueError('incomplete qualifying numeric positions')
            if sorted(target.tolist()) != list(range(1, len(q)+1)):
                raise ValueError('race classification is not a complete permutation')
            if q.TeamId.isna().any():
                raise ValueError('missing qualifying team identity')
            status = r.Status.fillna('').astype(str).str.lower().str.strip()
            finish = (status.eq('finished') | status.str.match(r'^\+\d+\s+laps?$')).to_numpy()
            meta = json.loads(mp.read_text())
            fmt = 'sprint' if any(s.get('session_type') in {'sprint','sprint_qualifying','sprint_shootout'} or s.get('session_name') == 'Sprint' for s in meta['sessions']) else 'standard'
            events.append(Event(key, year, number, folder.name.split('_',2)[2], fmt, ids,
                                q.TeamId.astype(str).tolist(), ranks(qp_values, ids), target, finish))
        except (ValueError, KeyError) as exc:
            excluded.append({'event': key, 'reason': str(exc)})
    return events, manifest, excluded


def features_for(event, history, index):
    n = len(event.ids)
    q = (event.qualifying-1)/(n-1)
    rows = []
    for i, (driver, team) in enumerate(zip(event.ids, event.teams)):
        same_team = np.array([t == team for t in event.teams])
        dh = [h for h in history if h['year'] == event.year and h['driver'] == driver]
        th = [h for h in history if h['year'] == event.year and h['team'] == team]
        ch = [h for h in history if h['circuit'] == event.circuit]
        def form(hist):
            if not hist: return (0., .5, .15, 0.)
            w = np.array([2**(-(index-h['index'])/8.) for h in hist])
            y = np.array([h['target'] for h in hist])
            movement = np.array([h['q']-h['target'] for h in hist])
            finish = np.array([h['finished'] for h in hist], dtype=bool)
            wp = w*finish
            gain = float(np.dot(wp, movement)/(wp.sum()+5.))
            finish_form = float((np.dot(wp,y)+5*.5)/(wp.sum()+5.))
            failure = float((np.dot(w,~finish)+5*.15)/(w.sum()+5.))
            support = float(w.sum()/(w.sum()+5.))
            return gain, finish_form, failure, support
        d, t = form(dh), form(th)
        circuit_movement = float(np.mean([abs(h['target']-h['q']) for h in ch])) if ch else .15
        rows.append([q[i], q[same_team].mean(), q[i]-q[same_team].mean(), d[0],t[0],d[1],t[1],d[2],t[2],d[3],t[3],circuit_movement])
    result = np.asarray(rows, dtype=float)
    assert result.shape == (n,len(FEATURES)) and np.isfinite(result).all()
    return result


def build_panels(events):
    history, panels = [], []
    for index, event in enumerate(events):
        # All features are constructed before any current race label is added.
        x = features_for(event, history, index)
        panels.append(x)
        n = len(event.ids)
        for i in range(n):
            history.append({'index':index, 'year':event.year, 'driver':event.ids[i], 'team':event.teams[i],
                            'circuit':event.circuit, 'q':(event.qualifying[i]-1)/(n-1),
                            'target':(event.target[i]-1)/(n-1), 'finished':bool(event.classified_finish[i])})
    return panels


def candidates(spec):
    out = {'baseline': ('baseline', 0.)}
    for a in spec['candidates']['shrunken_movement']['weights']:
        out[f'shrunken_movement_{a:g}'] = ('shrunken_movement', a)
    for a in spec['candidates']['ridge_movement']['alpha']:
        out[f'ridge_movement_{a:g}'] = ('ridge_movement', a)
    for a in spec['candidates']['huber_movement']['alpha']:
        out[f'huber_movement_{a:g}'] = ('huber_movement', a)
    for a in spec['candidates']['boosted_absolute_movement']['l2_regularization']:
        out[f'boosted_absolute_movement_{a:g}'] = ('boosted_absolute_movement', a)
    return out


def predict(kind, parameter, train_x, train_y, weights, current, ids):
    q = current[:,0]
    if kind == 'baseline': return ranks(q, ids)
    if kind == 'shrunken_movement':
        return ranks(q-parameter*(.4*current[:,3]+.6*current[:,4]), ids)
    scaler = StandardScaler().fit(train_x, sample_weight=weights)
    x, z = scaler.transform(train_x), scaler.transform(current)
    if kind == 'ridge_movement': model = Ridge(alpha=parameter)
    elif kind == 'huber_movement': model = HuberRegressor(alpha=parameter, epsilon=1.35,max_iter=2000,tol=1e-7)
    else:
        model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=100, max_leaf_nodes=4,
            min_samples_leaf=30, learning_rate=.05, l2_regularization=parameter, random_state=20260907,
            early_stopping=False)
    model.fit(x, train_y, sample_weight=weights)
    return ranks(q+model.predict(z), ids)


def event_score(event, pred):
    return {'event':event.key,'year':event.year,'format':event.weekend_format,'drivers':len(event.ids),
            'mae':float(np.mean(np.abs(pred-event.target))),
            'kendall':float(kendalltau(pred,event.target).statistic),
            'winner_hit':int(np.argmin(pred)==np.argmin(event.target)),
            'top3_overlap':float(len(set(np.where(pred<=3)[0]) & set(np.where(event.target<=3)[0]))/3)}


def summarize(records, baseline, seed=20260907):
    if not records: return None
    assert [r['event'] for r in records] == [r['event'] for r in baseline]
    d = np.array([r['mae']-b['mae'] for r,b in zip(records,baseline)])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0,len(d),(20000,len(d)))].mean(axis=1)
    loo = [(d.sum()-v)/(len(d)-1) for v in d] if len(d)>1 else [float(d[0])]
    return {'events':len(records),'drivers':sum(r['drivers'] for r in records),
            'mae':float(np.mean([r['mae'] for r in records])),
            'baseline_mae':float(np.mean([r['mae'] for r in baseline])),
            'paired_delta':float(d.mean()), 'paired_event_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'events_won':int((d<0).sum()),'events_tied':int((d==0).sum()),'events_lost':int((d>0).sum()),
            'loo_delta_min':float(min(loo)),'loo_delta_max':float(max(loo)),
            'kendall':float(np.mean([r['kendall'] for r in records])),
            'winner_hit_rate':float(np.mean([r['winner_hit'] for r in records])),
            'top3_overlap':float(np.mean([r['top3_overlap'] for r in records]))}


def run(output):
    spec = json.loads(SPEC_PATH.read_text())
    events, manifest, exclusions = load_events()
    panels = build_panels(events)
    configurations = candidates(spec)
    scores = {k:[] for k in configurations}
    predictions, fitting_warnings = [], []
    selection, preferred, active = None, None, list(configurations)
    for index,event in enumerate(events):
        if event.year<2023: continue
        if event.year>=2024 and selection is None:
            validation = {k:float(np.mean([r['mae'] for r in v if r['year']==2023])) for k,v in scores.items()}
            selection = {}
            for kind in spec['candidates']:
                available = [k for k,(family,_) in configurations.items() if family==kind]
                selection[kind] = min(available,key=lambda k:(validation[k],k))
            preferred = min(['baseline',*selection.values()],key=lambda k:(validation[k],k))
            active = ['baseline',*selection.values()]
            lock = {'specification_sha256':digest(SPEC_PATH),'selection_year':2023,'validation_mae':validation,
                    'selected_by_family':selection,'preferred_mechanism':preferred,
                    'held_forward_metrics_not_yet_computed':True}
            (output/'selection_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
        tx = np.vstack(panels[:index])
        ty = np.concatenate([(e.target-e.qualifying)/(len(e.ids)-1) for e in events[:index]])
        weights = np.concatenate([np.full(len(e.ids),1/len(e.ids)) for e in events[:index]])
        for name in active:
            family,parameter = configurations[name]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pred = predict(family,parameter,tx,ty,weights,panels[index],event.ids)
            fitting_warnings.extend({'event':event.key,'candidate':name,'message':str(w.message)} for w in caught)
            scores[name].append(event_score(event,pred))
            predictions.extend({'event':event.key,'year':event.year,'driver_id':driver,'candidate':name,
                                'prediction':int(pred[i]),'baseline':int(event.qualifying[i]),'actual':int(event.target[i])}
                               for i,driver in enumerate(event.ids))
        print(event.key, 'completed',len(active),'candidates',flush=True)
    summaries = {}
    for block,years in [('selection_2023',[2023]),('held_forward_2024_2025',[2024,2025]),('2024',[2024]),('2025',[2025]),('exposed_2026',[2026])]:
        summaries[block] = {}
        base = [r for r in scores['baseline'] if r['year'] in years]
        for name,records in scores.items():
            selected = [r for r in records if r['year'] in years]
            if selected: summaries[block][name] = summarize(selected,base)
    predictions_path=output/'predictions.csv'
    pd.DataFrame(predictions).to_csv(predictions_path,index=False)
    result={'schema_version':'f1_race_performance_cycle_v1','specification':spec,
            'specification_sha256':digest(SPEC_PATH),'baseline_commit':spec['baseline_commit'],
            'runtime':{'python':platform.python_version(),'numpy':np.__version__},
            'information_horizon':spec['information_horizon'],'empirical_role':'historical_retrospective_research',
            'promotion_eligible':False,'production_changed':False,
            'inventory':{'included_events':len(events),'included_by_year':{str(y):sum(e.year==y for e in events) for y in range(2022,2027)},'excluded':exclusions},
            'all_candidate_configurations':configurations,'selected_by_family':selection,'preferred_mechanism':preferred,
            'summary':summaries,'event_metrics':scores,'fit_warnings':fitting_warnings,
            'input_manifest':manifest,
            'implementation_manifest':{str(Path(__file__).relative_to(ROOT)):digest(__file__),str(SPEC_PATH.relative_to(ROOT)):digest(SPEC_PATH)},
            'predictions_path':str(predictions_path.relative_to(ROOT)),'predictions_sha256':digest(predictions_path),
            'selection_lock_sha256':digest(output/'selection_lock.json')}
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'selected':selection,'preferred':preferred,'transfer':summaries['held_forward_2024_2025'],'exposed_2026':summaries['exposed_2026']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/research/performance_20260907/race/v1')
    args=parser.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False);run(args.output)
