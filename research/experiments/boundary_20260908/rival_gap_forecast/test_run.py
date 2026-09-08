"""Synthetic orchestration tests. No historical raw payloads or labels are read."""
from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest

from research.experiments.boundary_20260908.rival_gap_forecast import run as r
from research.experiments.boundary_20260908.rival_gap_forecast import models,features,evaluate


def issue(event=202201,driver='1',second=4,lap=1):
    ns=second*10**9
    return {**{k:float('nan') for k in r.old_data.BASE_FEATURES},'event_key':event,'driver_id':driver,
        'issued_after_lap_number':lap,'issued_at_timestamp':float(second),'issued_at_ns':ns,
        'issuance_id':f'{event}/{driver}/{lap}/{ns}','year':event//100,'forecast_naive_seconds':90.}


def reference(tmp_path,rows):
    path=tmp_path/'issued.jsonl';r.old_data._write_rows(path,rows)
    return {**r.record(path),'event_key':rows[0]['event_key'],'rows':len(rows)}


def packet(ms,patch):
    sec,part=divmod(ms,1000)
    return f'00:{sec//60:02}:{sec%60:02}.{part:03}'.encode()+json.dumps({'Lines':patch}).encode()+b'\n'


def test_prepare_preserves_every_original_row_and_closes_both_lags_before_inventory(tmp_path,monkeypatch):
    rows=[issue(second=4),issue(driver='2',second=5)]
    original=reference(tmp_path,rows);path=tmp_path/'raw';path.write_bytes(b'\xef\xbb\xbf'+packet(1000,{
        '1':{'Position':'2','GapToLeader':'2.5','IntervalToPositionAhead':{'Value':'.5'}}}))
    source={'event_key':202201,'stream':r.record(path)};out=tmp_path/'outputs'
    inventory=r.pilot.inventory_stream
    def checked_inventory(p):
        for lag in (2,0):assert (out/f'features_2022/202201_lag{lag}.jsonl').exists()
        return inventory(p)
    monkeypatch.setattr(r.pilot,'inventory_stream',checked_inventory)
    entry=r.prepare_event(out,original,source)
    for lag in (2,0):
        saved=r.old_data._read_rows(r.check(entry['ledgers'][str(lag)]))
        assert len(saved)==2 and [v['issuance_id'] for v in saved]==[v['issuance_id'] for v in rows]
        assert [v['gap_supported'] for v in saved]==[True,False]
        for old,new in zip(rows,saved):
            assert r.old_data.digest({k:v for k,v in new.items() if k not in r.ADDED})==r.old_data.digest(old)
            assert new['gap_cutoff_ns']==old['issued_at_ns']-lag*10**9
            assert len(new['gap_values'])==45 and any(v is None for v in new['gap_values'])
            assert not set(r.TARGET_COLUMNS).intersection(new)


def test_future_payload_poison_cannot_change_prepared_prefix(tmp_path):
    rows=[issue(second=4),issue(driver='2',second=5)];original=reference(tmp_path,rows)
    prefix=packet(1000,{'1':{'Position':'2','GapToLeader':'2','IntervalToPositionAhead':{'Value':'.5'}}})
    a=tmp_path/'a';a.write_bytes(prefix)
    b=tmp_path/'b';b.write_bytes(prefix+b'00:01:30.000\xff\n')
    left=r.prepare_event(tmp_path/'left',original,{'event_key':202201,'stream':r.record(a)})
    right=r.prepare_event(tmp_path/'right',original,{'event_key':202201,'stream':r.record(b)})
    for lag in ('2','0'):
        assert r.check(left['ledgers'][lag]).read_bytes()==r.check(right['ledgers'][lag]).read_bytes()
    assert right['full_stream_inventory']['payload_errors']=={'UnicodeDecodeError':1}


@pytest.mark.parametrize('change',[{'lap_time_seconds':91.},{'outcome_status':'unmatched'},
    {'telemetry_values':[0]*90},{'gap_values':[0]*45},{'issuance_id':'wrong'},{'issued_at_ns':4.0}])
def test_original_reference_rejects_targets_or_changed_metadata(tmp_path,change):
    item=reference(tmp_path,[{**issue(),**change}])
    with pytest.raises((ValueError,TypeError)):r.original_rows(item)


def test_original_reference_rejects_duplicate_or_reversed_issuances(tmp_path):
    item=reference(tmp_path,[issue(),issue()])
    with pytest.raises(ValueError,match='population'):r.original_rows(item)
    item=reference(tmp_path/'reversed',[issue(second=5),issue(second=4)])
    with pytest.raises(ValueError,match='chronological'):r.original_rows(item)


@pytest.mark.parametrize('stage',[r.prepare,r.select])
def test_historical_stages_cannot_run_before_design_lock(tmp_path,monkeypatch,stage):
    monkeypatch.setattr(r,'sources',lambda:pytest.fail('Source work preceded lock read'))
    with pytest.raises(FileNotFoundError):stage(tmp_path)
    assert not list(tmp_path.iterdir())


def test_failure_receipt_is_preserved_and_prevents_advancement(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'resource_check',lambda:0)
    with pytest.raises(RuntimeError,match='synthetic'):
        with r.attempt(tmp_path,'prepare'):raise RuntimeError('synthetic failure')
    failed=(tmp_path/'prepare_failure.json').read_bytes()
    with pytest.raises(FileExistsError):
        with r.attempt(tmp_path,'prepare'):pass
    assert (tmp_path/'prepare_failure.json').read_bytes()==failed
    r.save(tmp_path/'design_lock.json',{})
    with pytest.raises(ValueError,match='failed execution'):r.verify_design(tmp_path)


def test_late_resource_failure_cannot_be_accepted_as_completed_stage(tmp_path,monkeypatch):
    calls=[]
    def resource():
        calls.append(1)
        if len(calls)>1:raise MemoryError('synthetic RSS')
    monkeypatch.setattr(r,'resource_check',resource)
    with pytest.raises(MemoryError):
        with r.attempt(tmp_path,'prepare'):r.save(tmp_path/'data_lock.json',{'synthetic':True})
    r.save(tmp_path/'design_lock.json',{})
    with pytest.raises(ValueError,match='failed execution'):r.verify_design(tmp_path)


def gap_frame(event=202301):
    rows=[]
    for driver in ('1','2'):
        row=issue(event,driver)
        rows.append({**row,'gap_values':[None]*12+[0.]*33,'gap_supported':driver=='1','gap_cutoff_ns':row['issued_at_ns']-2*10**9,
            'gap_lag_seconds':2,'gap_provenance':{},'original_row_sha256':r.old_data.digest(row)})
    return pd.DataFrame(rows)


def test_saved_base_reuse_preserves_bits_and_requires_exact_full_id_order(tmp_path):
    frame=gap_frame();base=np.array([np.nextafter(90.,91.),91.],dtype=np.float64)
    rows=[{**{k:v[k] for k in (*r.old_data.KEYS,'issuance_id','issued_at_ns')},'lag_seconds':2,
        'predictions':{'base_hgb':float(base[i])}} for i,v in enumerate(frame.to_dict('records'))]
    path=tmp_path/'forecasts.jsonl';r.old_data._write_rows(path,rows);item={**r.record(path),'rows':2}
    got=r.load_saved_base(frame,item,2)
    assert np.array_equal(base.view(np.uint64),got.predictions.view(np.uint64))
    with pytest.raises(ValueError,match='identity'):r.load_saved_base(frame.iloc[::-1],item,2)
    with pytest.raises(ValueError,match='latency'):r.load_saved_base(frame,item,0)


def test_forecast_closure_retains_unsupported_and_rejects_labels(tmp_path):
    frame=gap_frame();predictions={name:np.array([90.,91.]) for name in r.MODEL_NAMES}
    rows=r.forecast_rows(frame,predictions,2);assert len(rows)==2 and not rows[1]['gap_supported']
    path=tmp_path/'new.jsonl';r.old_data._write_rows(path,rows)
    got=r.load_forecasts(frame,r.record(path),2)
    for name in predictions:assert np.array_equal(predictions[name].view(np.uint64),got[name].view(np.uint64))
    changed=frame.copy();changed['outcome_status']='unmatched'
    with pytest.raises(ValueError,match='labels'):r.forecast_rows(changed,predictions,2)
    changed=frame.copy();changed.at[0,'forecast_naive_seconds']=92
    with pytest.raises(ValueError,match='binding'):r.load_forecasts(changed,r.record(path),2)


def test_synthetic_selection_saves_both_full_forecasts_before_any_2023_label_read(tmp_path,monkeypatch):
    r.save(tmp_path/'design_lock.json',{'synthetic':True})
    r.save(tmp_path/'data_lock.json',{'design_lock':r.record(tmp_path/'design_lock.json'),'external_labels_read':False,'model_fits':0})
    model_path=tmp_path/'old.pkl'
    with model_path.open('wb') as f:pickle.dump({'models':{'base_hgb':{'synthetic_old_base':True}}},f)
    parent={'models':r.record(model_path),'reference_forecasts':{'2':{'synthetic':2},'0':{'synthetic':0}},
        'years':{'2022':{'labels':{'year':2022}},'2023':{'labels':{'year':2023},'original_references':[]}}}
    lock={'parent':parent};spec={'discovery':{'fit_matched_rows':1}};calls=[]
    monkeypatch.setattr(r,'verify_design',lambda out:(lock,spec));monkeypatch.setattr(r,'resource_check',lambda:0)
    def load(_,year,lag):
        frame=gap_frame(year*100+1);frame['gap_cutoff_ns']=frame.issued_at_ns-lag*10**9;frame['gap_lag_seconds']=lag;return frame
    monkeypatch.setattr(r,'load_year',load)
    monkeypatch.setattr(r,'reference_year',lambda _,year:pd.DataFrame([issue(year*100+1,d) for d in ('1','2')]))
    def labels(frame,item):
        calls.append(('labels',item['year']))
        if item['year']==2023:
            assert (tmp_path/'selection_issuance_lock.json').exists()
            for lag in (2,0):assert len(r.old_data._read_rows(tmp_path/f'selection_lag{lag}_forecasts.jsonl'))==2
        result=frame.copy();result['outcome_status']=['matched','unmatched'];result['lap_time_seconds']=[91.,None];return result
    monkeypatch.setattr(r.old_run,'join_labels',labels)
    def fit(frame,**kwargs):
        calls.append(('fit',len(frame)));assert len(frame)==1 and kwargs['saved_base_model']=={'synthetic_old_base':True}
        assert kwargs['saved_base_model_sha256']==parent['models']['sha256']
        return {'models':{'synthetic':True},'fit_summary':{'new_fits':2}}
    monkeypatch.setattr(models,'fit_models',fit)
    def base(frame,item,lag):
        return models.SavedBaseForecasts(tuple(frame.issuance_id),tuple(frame.gap_cutoff_ns),np.array([90.,90.]),'a'*64)
    monkeypatch.setattr(r,'load_saved_base',base)
    def predict(bundle,frame,**kwargs):
        calls.append(('predict',kwargs['latency_seconds']))
        assert not any(k in frame for k in r.TARGET_COLUMNS)
        return {name:np.array([90.,90.]) for name in r.MODEL_NAMES}
    monkeypatch.setattr(models,'predict_models',predict)
    def score(primary,pp,sensitivity,sp,**kwargs):
        calls.append(('score',len(primary)));assert len(primary)==len(sensitivity)==2
        assert len(kwargs['expected_matched'])==1 and len(kwargs['expected_issuances'])==2
        return {'advances_to_later_evaluation':False}
    monkeypatch.setattr(evaluate,'evaluate_selection',score)
    r.select(tmp_path)
    first_2023=calls.index(('labels',2023))
    assert calls.index(('predict',2))<first_2023 and calls.index(('predict',0))<first_2023
    assert r.read(tmp_path/'selection_lock.json')['advances_to_later_evaluation'] is False
    assert r.read(tmp_path/'selection_issuance_lock.json')['external_selection_labels_read'] is False


def test_no_extra_execution_stages_or_external_acquisition_routes():
    source=Path(r.__file__).read_text()
    assert "choices=('freeze','prepare','select')" in source
    assert 'requests.' not in source and 'read_csv(' not in source and 'Session.load' not in source


def metadata_graph(tmp_path,monkeypatch):
    """Closed metadata with deliberately unreadable label/model/raw bodies."""
    parent=tmp_path/'parent';parent.mkdir();written=[]
    monkeypatch.setattr(r,'ROOT',tmp_path);monkeypatch.setattr(r,'PARENT',parent)
    def body(name):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'not parsed in metadata preflight')
        written.append(path);return r.record(path)
    def meta(name,value):
        path=tmp_path/name;r.save(path,value);written.append(path);return r.record(path)
    design=meta('parent/design_lock.json',{'sources':{}});years={};labels_by_year={}
    for year,total,matched in ((2022,18788,18363),(2023,20432,20007)):
        refs=[];labels=[]
        for i,event in enumerate(range(year*100+1,year*100+23)):
            n=total-21 if i==0 else 1;m=matched-21 if i==0 else 1
            refs.append({**body(f'parent/original_references/{event}_issued.jsonl'),'event_key':event,'rows':n})
            labels.append({**body(f'parent/labels_{year}/{event}_labels.jsonl'),'event_key':event,'rows':n,'matched':m,'unmatched':n-m})
        feature=meta(f'parent/features_{year}/feature_closure.json',{'metadata_only':True})
        validation=meta(f'parent/input_validation_{year}.json',{'status':'PASS'})
        label=meta(f'parent/labels_{year}/label_closure.json',{'feature_closure':feature,'all_issuances':total,'events':labels})
        labels_by_year[year]=label;years[str(year)]={'original_references':refs,'feature_closure':feature,'input_validation':validation}
    data=meta('parent/data_lock.json',{'design_lock':design,'external_labels_attached':False,'model_fits':0,
        'original_issuances_per_lag':39220,'years':years})
    old_models=body('parent/models.pkl')
    fit=meta('parent/fit_lock.json',{'data_lock':data,'models':old_models,'training_labels':labels_by_year[2022],
        'external_selection_labels_attached':False,'fit_summary':{'fit_year':2022,'fit_latency_seconds':2,'rows_per_fit':18363}})
    forecasts={str(lag):{**body(f'parent/selection_lag{lag}_forecasts.jsonl'),'rows':20432} for lag in (0,2)}
    meta('parent/selection_issuance_lock.json',{'fit_lock':fit,'data_lock':data,'selection_labels_attached':False,'forecasts':forecasts})
    selection=body('parent/selection.json')
    meta('parent/selection_lock.json',{'selection':selection})
    verified=meta('parent/verification/result_v2.json',{'status':'passed','selection_sha256':selection['sha256'],
        'bindings':{str(p.relative_to(tmp_path)):r.record(p) for p in written}})
    streams=[];sessions=[]
    for i,event in enumerate(r.EVENTS):
        raw=body(f'raw/{event}.jsonStream');session=f'named/{event}/Race/'
        stream={'event_key':event,'stream':'TimingData','session_path':session,'status':'downloaded',
            'decoded_body_path':raw['path'],'decoded_body_sha256':raw['sha256'],'decoded_body_bytes':(tmp_path/raw['path']).stat().st_size}
        if i==0:stream.update(status='reused_verified_pilot',source_status='downloaded')
        streams.append(stream);sessions.append({'event_key':event,'session_path':session})
    acquisition=meta('acquisition.json',{'status':'complete_all_stream_jobs_recorded','models_fitted':0,
        'predictive_scores_computed':0,'sessions':sessions,'streams':streams})
    pilot_result=meta('pilot_result.json',{'status':'closed_target_free_feasibility_not_model_validation'})
    spec={'inputs':{'parent_data_lock_sha256':data['sha256'],'parent_selection_sha256':selection['sha256'],
        'parent_verification_sha256':verified['sha256'],'timing_acquisition':acquisition,'pilot_result':pilot_result}}
    return spec


def test_exact_parent_metadata_schema_closes_without_parsing_raw_labels_models_or_old_predictions(tmp_path,monkeypatch):
    spec=metadata_graph(tmp_path,monkeypatch)
    monkeypatch.setattr(r.old_data,'_read_rows',lambda _:pytest.fail('Row payload read during metadata preflight'))
    graph,parent=r.parent_graph(spec)
    assert len(graph)==150 and len(parent['streams'])==44
    assert parent['years']['2022']['matched_rows']==18363 and parent['years']['2023']['rows']==20432


@pytest.mark.parametrize('change',['wrong_status','missing_stream','duplicate_stream','wrong_race','unavailable'])
def test_metadata_preflight_rejects_acquisition_scope_or_schema_changes(tmp_path,monkeypatch,change):
    spec=metadata_graph(tmp_path,monkeypatch);path=r.check(spec['inputs']['timing_acquisition']);value=r.read(path)
    if change=='wrong_status':value['status']='all_discovery_attempts_recorded'
    elif change=='missing_stream':value['streams'].pop()
    elif change=='duplicate_stream':value['streams'][-1]=deepcopy(value['streams'][0])
    elif change=='wrong_race':value['streams'][0]['session_path']='wrong/Race/'
    else:value['streams'][0]['status']='unavailable'
    path.write_text(json.dumps(value));spec['inputs']['timing_acquisition']=r.record(path)
    with pytest.raises(ValueError):r.parent_graph(spec)


def test_freeze_preflight_error_has_immutable_failure_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'resource_check',lambda:0)
    monkeypatch.setattr(r,'_freeze',lambda *_:(_ for _ in ()).throw(ValueError('synthetic review failure')))
    with pytest.raises(ValueError,match='synthetic review'):r.freeze(tmp_path,tmp_path/'absent_review.json')
    assert r.read(tmp_path/'freeze_failure.json')['error']=='synthetic review failure'
    before=(tmp_path/'freeze_failure.json').read_bytes()
    with pytest.raises(FileExistsError):r.freeze(tmp_path,tmp_path/'absent_review.json')
    assert (tmp_path/'freeze_failure.json').read_bytes()==before
