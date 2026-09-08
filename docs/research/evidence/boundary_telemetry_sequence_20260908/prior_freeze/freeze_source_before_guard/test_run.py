"""Synthetic orchestration regressions; never construct a historical corpus."""
import base64
import copy
import json
from pathlib import Path
import types
import zlib

import numpy as np
import pandas as pd
import pytest
import torch

from research.experiments.boundary_20260908.telemetry_sequence import run as r
from research.experiments.boundary_20260908.telemetry_sequence import corpus, supervised


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_every_historical_stage_requires_lock_before_work(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'sources',lambda:pytest.fail('dependency read before lock'))
    for stage in (r.build_corpus,r.pretrain,r.embed,r.select):
        with pytest.raises(FileNotFoundError):stage(tmp_path)
    assert not list(tmp_path.iterdir())


def test_spec_accepts_exact_settings_without_reading_historical_payload(monkeypatch):
    spec=r.read(r.HERE/'specification.json')
    calls=[];monkeypatch.setattr(r,'check',lambda item:calls.append(item) or r.ROOT/item['path'])
    r.validate_spec(spec)
    assert len(calls)==len(spec['parent_bindings'])+len(spec['frozen_components'])


@pytest.mark.parametrize('section,key,value',[
    ('ssl','steps',799),('ssl','selection_year_tokens_used_for_training',True),
    ('resources','pretraining_shared_seconds',7200),('resources','max_resident_bytes',6*1024**3),
    ('selection_gate','minimum_relative_event_mae_reduction_each_reference',.001),
    ('selection_gate','references',['base_hgb']),('supervised','zero_lag_refit',True),
    ('discovery','original_issuances',38370),('tokens','primary_lag_seconds',0)])
def test_spec_cannot_silently_relax_settings_or_population(monkeypatch,section,key,value):
    spec=r.read(r.HERE/'specification.json');spec[section][key]=value
    monkeypatch.setattr(r,'check',lambda item:r.ROOT/item['path'])
    with pytest.raises(ValueError,match='specification mismatch'):r.validate_spec(spec)


def test_attempt_is_immutable_and_failure_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'resource_check',lambda:0)
    with pytest.raises(RuntimeError,match='planted'):
        with r.attempt(tmp_path,'example'):raise RuntimeError('planted')
    assert r.read(tmp_path/'example_failure.json')['error']=='planted'
    old=(tmp_path/'example_attempt.json').read_bytes()
    with pytest.raises(FileExistsError):
        with r.attempt(tmp_path,'example'):pytest.fail('reused attempt')
    assert (tmp_path/'example_attempt.json').read_bytes()==old


def test_resource_breach_cannot_close_success(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'resource_check',lambda:(_ for _ in ()).throw(corpus.ResourceLimitError('budget')))
    with pytest.raises(corpus.ResourceLimitError):
        with r.attempt(tmp_path,'example'):pytest.fail('budget bypassed')
    assert r.read(tmp_path/'example_failure.json')['type']=='ResourceLimitError'


def packet(ms,channels):
    payload=json.dumps({'Entries':[{'Cars':{'1':{'Channels':channels}}}]}).encode()
    c=zlib.compressobj(wbits=-zlib.MAX_WBITS)
    body=json.dumps(base64.b64encode(c.compress(payload)+c.flush()).decode()).encode()
    seconds,part=divmod(ms,1000)
    return f'00:{seconds//60:02}:{seconds%60:02}.{part:03}'.encode()+body+b'\n'


def row(*,instant=31_000_000_000,lag=0,support=True):
    return {'event_key':202201,'year':2022,'driver_id':'1','issued_after_lap_number':4,
        'issued_at_timestamp':instant/1e9,'issued_at_ns':instant,
        'issuance_id':f'202201/1/4/{instant}','telemetry_cutoff_ns':instant-lag*10**9,
        'telemetry_lag_seconds':lag,'telemetry_supported':support,'forecast_naive_seconds':90.}


@pytest.mark.parametrize('lag',[0,2])
def test_embedding_future_payload_poisoning_preserves_every_coordinate(tmp_path,monkeypatch,lag):
    monkeypatch.setattr(r,'resource_check',lambda:0)
    channels={'0':9000,'2':200,'3':7,'4':80,'5':0,'45':12}
    body=b''.join(packet(t,channels) for t in [1000,6000,11000,16000,21000,26000])
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(body);b.write_bytes(body+b'00:01:10.000"poisoned-future-base64"\n')
    model=r.encoder.new_model(frozen=True);models={key:model for key in r.CONTROLS}
    def stream(path):return {'event_key':202201,'status':'downloaded_unparsed',
        'decoded_body_path':str(path),'decoded_body_sha256':r.old_data.sha(path)}
    first=r.embed_event([row(lag=lag)],stream(a),models,lag)
    second=r.embed_event([row(lag=lag)],stream(b),models,lag)
    for name in first:np.testing.assert_array_equal(first[name],second[name])
    assert first['support'].tolist()==[True] and first['empty_context'].tolist()==[False]
    for name in r.CONTROLS:
        assert first[name].dtype==np.float32 and first[name].shape==(1,16)
        assert np.linalg.norm(first[name][0])==pytest.approx(1,abs=1e-6)


def test_embedding_empty_unavailable_is_retained_and_zero(monkeypatch):
    monkeypatch.setattr(r,'resource_check',lambda:0)
    model=r.encoder.new_model(frozen=True);models={key:model for key in r.CONTROLS}
    value=r.embed_event([row(support=False)],{'event_key':202201,'status':'unavailable'},models,0)
    assert len(value['issuance_ids'])==1 and value['empty_context'][0]
    for key in r.CONTROLS:np.testing.assert_array_equal(value[key],np.zeros((1,16),np.float32))


def test_embedding_rejects_old_support_disagreement_and_targets(monkeypatch):
    monkeypatch.setattr(r,'resource_check',lambda:0)
    with pytest.raises(ValueError,match='support differs'):
        r.embed_event([row()],{'event_key':202201,'status':'unavailable'},{},0)
    with pytest.raises(ValueError,match='Targets entered'):
        r.embed_event([{**row(),'outcome_status':'unmatched'}],{}, {},0)
    with pytest.raises(ValueError,match='cutoff/lag'):
        r.embed_event([row(lag=2)],{'event_key':202201,'status':'unavailable'},{},0)


def prepare_pretrain(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'verify_design',lambda out:({},{}));monkeypatch.setattr(r,'resource_check',lambda:0)
    for name in ('design_lock.json','corpus.json','schedule.json'):r.save(tmp_path/name,{})
    r.save(tmp_path/'corpus_lock.json',{'design_lock':r.record(tmp_path/'design_lock.json'),
        'source_years':[2022],'corpus':r.record(tmp_path/'corpus.json'),'schedule':r.record(tmp_path/'schedule.json')})
    class Provider:
        def __init__(self,*args):self.closed=False
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True
        def __call__(self,step):return step
    providers=[]
    def provider(*args):
        obj=Provider(*args);providers.append(obj);return obj
    monkeypatch.setattr(corpus,'BatchProvider',provider)
    return providers


def test_pretraining_uses_exact_initial_state_same_provider_and_one_deadline(tmp_path,monkeypatch):
    providers=prepare_pretrain(tmp_path,monkeypatch);calls=[]
    def fit(provider,*,control,deadline_monotonic):
        calls.append((provider,control,deadline_monotonic));model=r.encoder.new_model(frozen=True)
        return model,{'initial_state_sha256':r.encoder.state_digest(model),'steps':800,'batch_size':32,'elapsed_seconds':0}
    monkeypatch.setattr(r.encoder,'fit_cpc',fit);r.pretrain(tmp_path)
    assert [c[1] for c in calls]==['ordered','permuted'] and calls[0][0] is calls[1][0]
    assert calls[0][2]==calls[1][2] and providers[0].closed
    lock=r.read(tmp_path/'pretrain_lock.json');assert lock['fits']==2
    for control in ('ordered','permuted'):
        trace=r.load_record(lock['encoders'][control]['training'])
        assert trace['initial_state_sha256']==lock['encoders']['random']['state_sha256']
        assert trace['schedule']==r.record(tmp_path/'schedule.json')
    assert set(r.load_encoders(lock))==set(r.CONTROLS)


@pytest.mark.parametrize('mode',['deadline','initial_state','short_training'])
def test_pretraining_failure_preserves_attempt_without_success_lock(tmp_path,monkeypatch,mode):
    providers=prepare_pretrain(tmp_path,monkeypatch)
    def fit(*args,**kwargs):
        if mode=='deadline':raise TimeoutError('deadline')
        model=r.encoder.new_model(frozen=True)
        return model,{'initial_state_sha256':'0'*64 if mode=='initial_state' else r.encoder.state_digest(model),
            'steps':799 if mode=='short_training' else 800,'batch_size':32,'elapsed_seconds':0}
    monkeypatch.setattr(r.encoder,'fit_cpc',fit)
    with pytest.raises((ValueError,TimeoutError)):r.pretrain(tmp_path)
    assert providers[0].closed and (tmp_path/'encoders/random.pt').is_file()
    assert (tmp_path/'pretrain_failure.json').is_file() and not (tmp_path/'pretrain_lock.json').exists()


def test_closed_old_reference_vectors_reused_without_refitting(tmp_path):
    frame=pd.DataFrame([row()]);pred={name:np.array([90.+i*.01]) for i,name in enumerate(supervised.OLD_MODEL_NAMES)}
    path=tmp_path/'forecasts.jsonl';r.old_data._write_rows(path,r.old_run.forecast_rows(frame,pred,0))
    ref=r.references(frame,r.record(path),0)
    for name in pred:np.testing.assert_array_equal(ref.predictions[name].view(np.uint64),pred[name].view(np.uint64))
    altered=frame.copy();altered['forecast_naive_seconds']=89.
    with pytest.raises(ValueError,match='binding differs'):r.references(altered,r.record(path),0)


def test_saved_forecast_writer_preserves_unsupported_unmatched_inventory():
    frame=pd.DataFrame([row(support=False),row(instant=36_000_000_000)])
    predictions={name:np.array([90.,91.]) for name in supervised.MODEL_NAMES}
    saved=r.forecast_rows(frame,predictions,0)
    assert len(saved)==2 and saved[0]['telemetry_supported'] is False
    assert set(saved[0]['predictions'])==set(supervised.MODEL_NAMES)
    with pytest.raises(ValueError,match='labels precede'):r.forecast_rows(frame.assign(outcome_status='unmatched'),predictions,0)
    with pytest.raises(ValueError,match='Every old and new'):r.forecast_rows(frame,{'ordered_hgb':np.ones(2)},0)


def test_runner_exposes_no_later_evaluation_command():
    source=Path(r.__file__).read_text()
    assert "choices=('freeze','corpus','pretrain','embed','select')" in source
    assert 'Session.load(' not in source and 'read_csv(' not in source


def test_corpus_stage_selects_only_2022_then_closes_schedule(tmp_path,monkeypatch):
    r.save(tmp_path/'design_lock.json',{})
    streams=[{'event_key':key} for key in r.EVENTS]
    monkeypatch.setattr(r,'verify_design',lambda out:({'parent':{'streams':streams}},{}))
    monkeypatch.setattr(r,'resource_check',lambda:0);calls=[]
    def build(events,out,**kwargs):
        assert [v['event_key'] for v in events]==list(range(202201,202223))
        assert not Path(out).exists();calls.append('corpus')
        assert kwargs['design_sha256']==r.old_data.sha(tmp_path/'design_lock.json')
        p=Path(out)/'corpus.json';r.save(p,{});return p
    def schedule(path,out):
        assert path==tmp_path/'corpus/corpus.json' and not Path(out).exists()
        calls.append('schedule');p=Path(out)/'schedule.json';r.save(p,{});return p
    monkeypatch.setattr(corpus,'build_corpus',build);monkeypatch.setattr(corpus,'build_schedule',schedule)
    r.build_corpus(tmp_path);assert calls==['corpus','schedule']
    lock=r.read(tmp_path/'corpus_lock.json');assert lock['source_years']==[2022] and lock['encoder_fits']==0


def stage_fixture(tmp_path,monkeypatch,*,failed_lag=None):
    calls=[];monkeypatch.setattr(r,'resource_check',lambda:0)
    r.save(tmp_path/'design_lock.json',{});r.save(tmp_path/'pretrain_lock.json',{})
    r.save(tmp_path/'embedding_lock.json',{'design_lock':r.record(tmp_path/'design_lock.json'),
        'external_labels_read':False,'supervised_fits':0,'pretrain_lock':r.record(tmp_path/'pretrain_lock.json')})
    parent={'years':{str(year):{'labels':{'synthetic_year':year}} for year in r.YEARS},
            'reference_forecasts':{str(lag):{'synthetic_lag':lag} for lag in r.LAGS}}
    monkeypatch.setattr(r,'verify_design',lambda out:({'parent':parent},{}))
    def frame(year,lag):
        data=[]
        for i in range(2):
            value=row(instant=(31+i*5)*10**9,lag=lag,support=i==0)
            value.update(year=year,event_key=year*100+1,issuance_id=f'{year}/1/{i}')
            data.append(value)
        return pd.DataFrame(data)
    def embed(lock,closed,year,lag):
        value=frame(year,lag)
        return value,{control:supervised.EmbeddingTable(tuple(value.issuance_id),tuple(value.telemetry_cutoff_ns),
            np.ones((2,16),np.float32)/4,np.zeros(2,bool),'a'*64,control) for control in r.CONTROLS}
    monkeypatch.setattr(r,'load_embedding_year',embed)
    monkeypatch.setattr(r.old_run,'load_year',lambda lock,year,lag:frame(year,lag))
    monkeypatch.setattr(r.old_run,'reference_year',lambda lock,year:frame(year,2))
    def labels(value,receipt):
        year=receipt['synthetic_year'];calls.append('labels_'+str(year))
        if year==2023:
            assert (tmp_path/'selection_issuance_lock.json').is_file()
            closed=r.read(tmp_path/'selection_issuance_lock.json')
            assert closed['external_selection_labels_read'] is False
            assert set(closed['forecasts'])=={'0','2'}
            for item in closed['forecasts'].values():assert len(r.old_data._read_rows(r.check(item)))==2
        answer=value.copy();answer['outcome_status']=['matched','unmatched'];answer['lap_time_seconds']=[90.,None]
        return answer
    monkeypatch.setattr(r.old_run,'join_labels',labels)
    def fit(value,tables,*,expected_matched):
        calls.append('fit');assert len(value)==len(expected_matched)==1
        assert all(len(t.issuance_ids)==1 for t in tables.values())
        return {'fit_summary':{'fits':3}}
    monkeypatch.setattr(supervised,'fit_models',fit)
    monkeypatch.setattr(r,'references',lambda value,item,lag:{'lag':lag})
    def predict(bundle,value,tables,*,latency_seconds,saved_references):
        calls.append('predict_'+str(latency_seconds))
        assert 'outcome_status' not in value and len(value)==2
        assert not (tmp_path/'selection_issuance_lock.json').exists()
        if failed_lag==latency_seconds:raise ValueError('planted prediction failure')
        return {name:np.array([90.,91.]) for name in supervised.MODEL_NAMES}
    monkeypatch.setattr(supervised,'predict_models',predict)
    def evaluate(primary,pp,sensitivity,sp,**kwargs):
        calls.append('evaluate')
        assert len(primary)==len(sensitivity)==len(kwargs['expected_issuances'])==2
        assert len(kwargs['expected_matched'])==1
        assert primary.outcome_status.tolist()==['matched','unmatched']
        return {'advances_to_later_evaluation':False}
    monkeypatch.setattr(supervised,'evaluate_selection',evaluate)
    return calls


def test_selection_closes_both_lags_before_any_external_selection_label(tmp_path,monkeypatch):
    calls=stage_fixture(tmp_path,monkeypatch);r.select(tmp_path)
    assert calls.index('fit')<calls.index('predict_2')<calls.index('predict_0')<calls.index('labels_2023')<calls.index('evaluate')
    decision=r.read(tmp_path/'selection_lock.json')
    assert decision['promotion'] is False and decision['advances_to_later_evaluation'] is False
    assert r.read(tmp_path/'fit_lock.json')['external_selection_labels_read'] is False


def test_second_lag_failure_never_reads_external_labels_or_scores(tmp_path,monkeypatch):
    calls=stage_fixture(tmp_path,monkeypatch,failed_lag=0)
    with pytest.raises(ValueError,match='planted prediction'):r.select(tmp_path)
    assert 'labels_2023' not in calls and 'evaluate' not in calls
    assert (tmp_path/'selection_lag2_forecasts.jsonl').is_file()
    assert (tmp_path/'selection_failure.json').is_file()
    assert not (tmp_path/'selection_issuance_lock.json').exists()
    assert not (tmp_path/'selection.json').exists()
