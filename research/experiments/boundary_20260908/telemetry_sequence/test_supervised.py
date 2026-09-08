"""Synthetic incremental contracts; old objective/population suites are reused."""
from dataclasses import replace
import copy

import numpy as np
import pandas as pd
import pytest

from . import supervised as s
from research.experiments.boundary_20260908.telemetry.test_models import frame as old_frame
from research.experiments.boundary_20260908.telemetry.test_evaluate import population as old_population


def tables(frame):
    result={}
    for i,control in enumerate(s.CONTROLS):
        values=np.zeros((len(frame),16),dtype=np.float32);values[:,i]=1
        result[control]=s.EmbeddingTable(tuple(frame.issuance_id),tuple(frame.telemetry_cutoff_ns),
            values,np.zeros(len(frame),dtype=bool),str(i+1)*64,control)
    return result


def references(frame, points=None):
    if points is None:
        base=np.full(len(frame),np.nextafter(90.,91.),dtype=np.float64)
        support=frame.telemetry_supported.to_numpy(bool)
        points={'base_hgb':base,'telemetry_hgb':np.where(support,90.7,base),'quality_hgb':np.where(support,90.9,base)}
    return s.ReferenceForecasts(tuple(frame.issuance_id),tuple(frame.telemetry_cutoff_ns),
        {name:value.copy() for name,value in points.items()},'a'*64)


@pytest.fixture(scope='module')
def full_training():
    f=old_frame(s.TRAIN_ROWS)
    f['event_key']=np.array(s.TRAIN_EVENTS)[np.arange(len(f))%22]
    f['target_lap_number']=f.issued_after_lap_number+1
    f['target_timestamp']=f.issued_at_timestamp+90
    f['target_same_stint']=True
    return f


@pytest.fixture
def recording(monkeypatch):
    calls=[]
    class Estimator:
        def __init__(self,**config):self.config=config;self.number=len(calls);calls.append(self)
        def fit(self,x,y,sample_weight):
            self.shape=x.shape;self.columns=list(x.columns)
            self.first=x.iloc[0].to_numpy().copy();self.weights=sample_weight.copy();self.y=y.copy()
        def predict(self,x):
            self.predicted_rows=len(x)
            return np.full(len(x),[-20.,20.,2.][self.number])
    monkeypatch.setattr(s.old_models.original,'HistGradientBoostingRegressor',Estimator)
    return calls


def test_three186_column_fits_keep_full_original_population_and_embedding_identity(full_training,recording):
    f=full_training;embedding=tables(f);bundle=s.fit_models(f,embedding,expected_matched=f.copy())
    assert len(recording)==3 and all(c.shape==(18363,186) for c in recording)
    assert bundle['fit_summary']['events']==list(s.TRAIN_EVENTS)
    assert bundle['fit_summary']['old_reference_refits']==bundle['fit_summary']['representation_fits']==0
    count=f.groupby('event_key').event_key.transform('size').to_numpy()
    for i,c in enumerate(recording):
        assert c.columns[-16:]==list(s.EMBEDDING_FEATURES)
        np.testing.assert_array_equal(c.first[-16:],embedding[s.CONTROLS[i]].values[0])
        np.testing.assert_allclose(c.weights,len(f)/(22*count),rtol=1e-15,atol=0)
        np.testing.assert_array_equal(c.y,np.clip(f.lap_time_seconds-f.forecast_naive_seconds,-5,5))
        assert c.config['max_iter']==150 and c.config['max_leaf_nodes']==15 and c.config['loss']=='absolute_error'


def test_predict_never_reads_labels_preserves_old_reference_bits_and_exact_new_fallback(full_training,recording):
    bundle=s.fit_models(full_training,tables(full_training),expected_matched=full_training.copy())
    issued=old_frame(8).drop(columns=['lap_time_seconds','outcome_status','year'])
    emb=tables(issued);ref=references(issued);before=copy.deepcopy(ref.predictions)
    result=s.predict_models(bundle,issued,emb,latency_seconds=2,saved_references=ref)
    assert set(result)==set(s.MODEL_NAMES)
    support=issued.telemetry_supported.to_numpy()
    for name in s.OLD_MODEL_NAMES:
        np.testing.assert_array_equal(result[name].view(np.uint64),before[name].view(np.uint64))
        assert result[name] is not ref.predictions[name]
    for name in s.NEW_MODEL_NAMES:
        np.testing.assert_array_equal(result[name][~support].view(np.uint64),before['base_hgb'][~support].view(np.uint64))
    np.testing.assert_array_equal(result['ordered_hgb'][support],87.)
    np.testing.assert_array_equal(result['permuted_hgb'][support],93.)
    np.testing.assert_array_equal(result['random_hgb'][support],92.)
    assert [x.predicted_rows for x in recording]==[4,4,4]
    for name in s.OLD_MODEL_NAMES:np.testing.assert_array_equal(ref.predictions[name],before[name])
    zero=issued.copy();zero['telemetry_cutoff_ns']=zero.issued_at_ns
    zr=references(zero,ref.predictions)
    repeated=s.predict_models(bundle,zero,tables(zero),latency_seconds=0,saved_references=zr)
    for name in s.MODEL_NAMES:np.testing.assert_array_equal(result[name],repeated[name])
    assert len(recording)==3


@pytest.mark.parametrize('fault',['shape','nan','infinity','not_unit','float64','row_order','cutoff','control','hash','empty_nonzero','empty_supported','missing_control'])
def test_embedding_alignment_values_and_encoder_bindings_cannot_be_bypassed(fault):
    f=old_frame(8);emb=tables(f);t=emb['ordered']
    if fault=='shape':t=replace(t,values=t.values[:,:15])
    elif fault=='nan':t.values[0,0]=np.nan
    elif fault=='infinity':t.values[0,0]=np.inf
    elif fault=='not_unit':t.values[0,0]=.5
    elif fault=='float64':t=replace(t,values=t.values.astype(np.float64))
    elif fault=='row_order':t=replace(t,issuance_ids=t.issuance_ids[::-1])
    elif fault=='cutoff':t=replace(t,cutoff_ns=tuple(v+1 for v in t.cutoff_ns))
    elif fault=='control':t=replace(t,control='random')
    elif fault=='hash':t=replace(t,encoder_sha256='not_a_hash')
    elif fault=='empty_nonzero':t.empty_context[0]=True
    elif fault=='empty_supported':t.empty_context[1]=True;t.values[1]=0
    else:del emb['random']
    emb['ordered']=t
    with pytest.raises(ValueError):s._inputs(f,emb,latency_seconds=2)


def test_zero_embeddings_are_allowed_only_for_explicit_empty_unsupported_contexts():
    f=old_frame(8);emb=tables(f)
    emb['ordered'].values[0]=0;emb['ordered'].empty_context[0]=True
    s._inputs(f,emb,latency_seconds=2)
    emb['ordered'].empty_context[0]=False
    with pytest.raises(ValueError,match='unit norm'):s._inputs(f,emb,latency_seconds=2)


@pytest.mark.parametrize('fault',['rows','missing_event','wrong_year','unmatched','target','order'])
def test_original_complete_training_cohort_required_before_any_fit(full_training,monkeypatch,fault):
    monkeypatch.setattr(s.old_models.original,'fit_model',lambda *a,**k:pytest.fail('Invalid cohort reached fit'))
    f=full_training.copy()
    if fault=='rows':f=f.iloc[:-1]
    elif fault=='missing_event':f.loc[f.event_key==202222,'event_key']=202221
    elif fault=='wrong_year':f['year']=2023
    elif fault=='unmatched':f.loc[0,'outcome_status']='unmatched'
    elif fault=='target':f.loc[0,'lap_time_seconds']+=.1
    else:f=f.iloc[::-1]
    with pytest.raises(ValueError):s.fit_models(f,tables(f),expected_matched=full_training)


def test_changed_encoder_asset_cannot_be_substituted_at_prediction(full_training,recording):
    bundle=s.fit_models(full_training,tables(full_training),expected_matched=full_training)
    issued=old_frame(8);emb=tables(issued)
    emb['ordered']=replace(emb['ordered'],encoder_sha256='f'*64)
    with pytest.raises(ValueError,match='encoder contract'):
        s.predict_models(bundle,issued,emb,latency_seconds=2,saved_references=references(issued))


def evaluation_inputs():
    primary,pold,zero,zold,matched,original=old_population()
    refs=references(primary,pold);zrefs=references(zero,zold)
    p={k:v.copy() for k,v in pold.items()};zp={k:v.copy() for k,v in zold.items()}
    mask=primary.telemetry_supported.to_numpy()
    for name,point in zip(s.NEW_MODEL_NAMES,[90.1,90.3,90.4]):
        p[name]=np.where(mask,point,pold['base_hgb']);zp[name]=p[name].copy()
    return primary,p,zero,zp,matched,original,refs,zrefs


def evaluate(values):
    f,p,z,zp,matched,original,ref,zref=values
    return s.evaluate_selection(f,p,z,zp,expected_matched=matched,expected_issuances=original,
        primary_references=ref,sensitivity_references=zref)


def test_all_six_models_full_original_rows_and_five_comparators_are_reported():
    report=evaluate(evaluation_inputs())
    assert report['candidate_eligible_for_advancement']=='ordered_hgb'
    assert report['advances_to_later_evaluation'] is True and report['promotion'] is False
    assert report['coverage']['2']['rows_all']==66 and report['coverage']['2']['rows_unmatched_retained']==22
    for lag in ('2','0'):
        assert set(report['metrics'][lag])==set(s.MODEL_NAMES)
        assert set(report['comparisons'][lag])==set(s.REFERENCES)
        assert set(report['gate_checks'][lag])==set(s.REFERENCES)
        assert set(report['contributing_events_and_drivers'][lag])==set(s.REFERENCES)


@pytest.mark.parametrize('reference',s.REFERENCES)
@pytest.mark.parametrize('field,bad',[('relative_reduction',.00999),('event_ci95',[-.1,0]),('block3_ci95',[-.1,0]),('loo_max_delta',0)])
def test_ordered_cannot_bypass_any_primary_gate_for_any_of_five_references(monkeypatch,reference,field,bad):
    values=evaluation_inputs();counter=0
    def comparison(frame,candidate,baseline):
        nonlocal counter
        name=s.REFERENCES[counter%5];lag=counter//5;counter+=1
        result=dict(relative_reduction=.02,event_ci95=[-.2,-.1],block3_ci95=[-.2,-.1],loo_max_delta=-.1)
        if name==reference and lag==0:result[field]=bad
        return result
    monkeypatch.setattr(s.old_evaluate,'comparison',comparison)
    report=evaluate(values)
    assert counter==10 and report['advances_to_later_evaluation'] is False


@pytest.mark.parametrize('reference',s.REFERENCES)
def test_zero_lag_must_improve_every_fixed_reference(monkeypatch,reference):
    values=evaluation_inputs();counter=0
    def comparison(frame,candidate,baseline):
        nonlocal counter
        name=s.REFERENCES[counter%5];lag=counter//5;counter+=1
        return dict(relative_reduction=0 if name==reference and lag==1 else .02,
            event_ci95=[-.2,-.1],block3_ci95=[-.2,-.1],loo_max_delta=-.1)
    monkeypatch.setattr(s.old_evaluate,'comparison',comparison)
    assert evaluate(values)['advances_to_later_evaluation'] is False


@pytest.mark.parametrize('name',s.MODEL_NAMES)
def test_no_model_or_reference_may_be_omitted(name):
    values=evaluation_inputs();del values[1][name]
    with pytest.raises(ValueError,match='six'):evaluate(values)


@pytest.mark.parametrize('reference',s.OLD_MODEL_NAMES)
def test_old_reference_modification_is_rejected_bitwise_including_unmatched(reference):
    values=evaluation_inputs();values[1][reference][-1]=np.nextafter(values[1][reference][-1],np.inf)
    with pytest.raises(ValueError,match='closed old reference'):evaluate(values)


def test_new_unsupported_forecast_cannot_change_even_an_unmatched_row():
    values=evaluation_inputs();values[1]['ordered_hgb'][-1]+=1e-8
    with pytest.raises(ValueError,match='unsupported'):evaluate(values)


def test_reused_old_population_checks_reject_jointly_dropped_unmatched_row():
    f,p,z,zp,matched,original,ref,zref=evaluation_inputs()
    f=f.iloc[:-1].copy();z=z.iloc[:-1].copy()
    p={k:v[:-1].copy() for k,v in p.items()};zp={k:v[:-1].copy() for k,v in zp.items()}
    ref=references(f,{k:p[k] for k in s.OLD_MODEL_NAMES});zref=references(z,{k:zp[k] for k in s.OLD_MODEL_NAMES})
    with pytest.raises(ValueError,match='full issuance population'):evaluate((f,p,z,zp,matched,original,ref,zref))


def test_control_cannot_be_selected_even_when_it_is_better_than_ordered():
    values=evaluation_inputs();mask=values[0].telemetry_supported.to_numpy()
    for p in (values[1],values[3]):p['permuted_hgb'][mask]=90.
    report=evaluate(values)
    assert report['candidate_eligible_for_advancement']=='ordered_hgb'
    assert report['advances_to_later_evaluation'] is False


def test_driver_contributions_sum_to_event_weighted_delta_not_row_weighted_delta():
    f=pd.DataFrame({'event_key':[1]+[2]*9,'driver_id':['A']+['B']*9,'lap_time_seconds':[90.]*10})
    candidate=np.array([98.]+[91.]*9);reference=np.array([100.]+[91.]*9)
    r=s.contributions(f,candidate,reference)
    assert r['event_balanced_delta_seconds']==-1 and r['row_mae_delta_seconds']==-.2
    assert sum(row['contribution_to_event_mae_delta_seconds'] for row in r['per_driver'])==-1
    assert sum(row['contribution_to_event_mae_delta_seconds'] for row in r['per_event'])==-1
    assert r['largest_event_driver_contributions'][0]['driver_id']=='A'
