"""Synthetic saved-schema, mathematical and rejection tests; no historical I/O.

Suggested commit: test(f1-live): exercise independent CPC evidence replay
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from . import verify as v


def record(path, **extra):
    return {'path': str(path), 'sha256': v.old.sha(path), 'bytes': path.stat().st_size, **extra}


def binfile(path, value, dtype):
    array = np.asarray(value, dtype=dtype); array.tofile(path)
    return record(path, dtype=array.dtype.str, shape=list(array.shape))


def test_closed_selection_guard_precedes_any_payload_access(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('Payload access before closed selection')
    monkeypatch.setattr(v.old, 'read', forbidden)
    monkeypatch.setattr(v.np, 'load', forbidden)
    monkeypatch.setattr(v.pickle, 'load', forbidden)
    with pytest.raises(ValueError, match='Closed selection_lock'):
        v.verify(tmp_path, design_sha256='d'*64, selection_sha256='s'*64)


def test_lock_hash_guard_precedes_other_files(tmp_path, monkeypatch):
    (tmp_path/'selection_lock.json').write_text(json.dumps({'selection': {'sha256': 'wrong'}, 'design_lock': {'sha256': 'd'*64}}))
    monkeypatch.setattr(v.old.Bindings, 'load', lambda *args: pytest.fail('Other file accessed before lock hash guard'))
    with pytest.raises(AssertionError): v.verify(tmp_path, design_sha256='d'*64, selection_sha256='s'*64)


def test_terminal_failure_marker_rejected_even_with_selection_lock(tmp_path, monkeypatch):
    (tmp_path/'selection_lock.json').write_text('{}')
    (tmp_path/'selection_failure.json').write_text('{}')
    monkeypatch.setattr(v.old, 'read', lambda *args: pytest.fail('Payload read after terminal failure'))
    with pytest.raises(ValueError, match='Failed execution'):
        v.verify(tmp_path, design_sha256='d'*64, selection_sha256='s'*64)


def test_output_is_exclusive_before_replay(tmp_path, monkeypatch):
    path = tmp_path/'result.json'; path.write_text('immutable')
    monkeypatch.setattr(v, 'verify', lambda *args, **kwargs: pytest.fail('Unexpected replay'))
    monkeypatch.setattr('sys.argv', ['verify', '--execution', str(tmp_path), '--out', str(path),
                                  '--design-sha256', 'd'*64, '--selection-sha256', 's'*64])
    with pytest.raises(FileExistsError): v.main()
    assert path.read_text() == 'immutable'


@pytest.mark.parametrize('times', [np.arange(160, dtype=np.int64)*5_000_000_000,
    np.array([0]*20+list(range(0, 1_000_000_000_000, 5_000_000_000)), dtype=np.int64),
    np.array([0, 1], dtype=np.int64), np.array([], dtype=np.int64)])
def test_endpoint_geometry_against_brute_physical_prefix(times):
    expected = []
    for i in range(max(0, len(times)-32)):
        own = [t for t in times[:i+1] if int(times[i])+1-30_000_000_000 <= t < int(times[i])+1]
        negative = [j for j, t in enumerate(times) if abs(int(t)-int(times[i])) >= 180_000_000_000
                    and j not in (i+4, i+16, i+32)]
        if len(own) >= 6 and own[-1]-own[0] >= 24_000_000_000 and len(negative) >= 31: expected.append(i)
    assert v.eligibility(times).tolist() == expected


def test_all_positive_ids_excluded_even_after_large_gap():
    times = np.arange(160, dtype=np.int64)*200_000_000_000
    pool = v.negative_pool(times, 10)
    assert not np.isin([14, 26, 42], pool).any()
    assert 10 not in pool and len(pool) == 156


@pytest.fixture(scope='module')
def scheduled(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic_corpus')
    times = np.arange(160, dtype=np.int64)*5_000_000_000
    # Independently enumerate eligible IDs for these regular5-second packets:
    # endpoint5 has six points spanning25s; all up to127 retain32 positives.
    valid = np.arange(5, 128, dtype=np.int64)
    assert np.array_equal(v.eligibility(times), valid)
    shared = {
        'availability_ns': binfile(root/'times.bin', times, '<i8'),
        'packet_sequence': binfile(root/'seq.bin', np.arange(1, 161), '<i8'),
        'previous_availability_ns': binfile(root/'previous.bin', np.r_[-1, times[:-1]], '<i8'),
        'values': binfile(root/'values.bin', np.zeros((160, 34)), '<f4'),
        'eligible_endpoints': binfile(root/'eligible.bin', valid, '<i8')}
    corpus = {'status': 'closed', 'years': [2022], 'lap_labels_read': False, 'model_fits': 0,
        'events': [{'event_key': event, 'drivers': [{'driver_id': '1', 'packets': 160,
            'eligible_endpoints': len(valid), 'arrays': shared}],
            'raw_validation': {'complete': True, 'stream_input_valid': True}} for event in v.TRAIN_EVENTS]}
    arrays = {'stream_index': np.empty((800,32), np.int32), 'endpoint': np.empty((800,32), np.int64),
        'positives': np.empty((800,32,3), np.int64), 'negatives': np.empty((800,32,31), np.int64)}
    rng = np.random.default_rng(20260908)
    for step in range(800):
        for row in range(32):
            event = int(rng.integers(22)); rng.integers(1)
            i = int(valid[rng.integers(len(valid))]); pos = np.array([i+4,i+16,i+32])
            pool = np.array([j for j in range(len(times)) if abs(j-i)*5 >= 180 and j not in pos])
            arrays['stream_index'][step,row] = event; arrays['endpoint'][step,row] = i
            arrays['positives'][step,row] = pos; arrays['negatives'][step,row] = rng.choice(pool,31,replace=False)
    counts = np.bincount(arrays['stream_index'].ravel(), minlength=22)
    schedule = {'status': 'closed', 'seed': 20260908, 'rng': 'PCG64', 'steps': 800, 'batch_size': 32,
        'horizons': [4,16,32], 'negatives': 31, 'model_fits_before_schedule_close': 0,
        'arrays': {k: binfile(root/(k+'.bin'), value, value.dtype) for k,value in arrays.items()},
        'driver_draw_counts': [{'event_key': e, 'driver_id': '1', 'draws': int(n)} for e,n in zip(v.TRAIN_EVENTS,counts)],
        'event_draw_counts': [{'event_key': e, 'draws': int(n)} for e,n in zip(v.TRAIN_EVENTS,counts)]}
    return root, corpus, schedule


def test_all_25600_frozen_schedule_draws(scheduled):
    root, corpus, schedule = scheduled
    result = v.verify_schedule(corpus, schedule, v.old.Bindings(root))
    assert result['scheduled_rows'] == 25600 and result['negative_identities'] == 793600


@pytest.mark.parametrize('field', ['stream_index', 'endpoint', 'positives', 'negatives'])
def test_schedule_mutation_rejected(scheduled, tmp_path, field):
    root, corpus, original = scheduled; schedule = deepcopy(original)
    entry = schedule['arrays'][field]
    values = np.fromfile(entry['path'], dtype=entry['dtype']).reshape(entry['shape']).copy()
    values.flat[0] += 1
    # Shared fixture directory is the binding root; fresh names preserve originals.
    path = root/(tmp_path.name+'_'+field+'.bin')
    schedule['arrays'][field] = binfile(path, values, values.dtype)
    with pytest.raises(AssertionError): v.verify_schedule(corpus, schedule, v.old.Bindings(root))


def feature_row(i=0, supported=True):
    return {'event_key':202301, 'year':2023, 'driver_id':'1', 'issued_after_lap_number':i+3,
        'issued_at_timestamp':float(100+i), 'issued_at_ns':(100+i)*1_000_000_000,
        'issuance_id':f'202301/1/{i+3}/{(100+i)*1_000_000_000}',
        'telemetry_cutoff_ns':(98+i)*1_000_000_000, 'telemetry_supported':supported,
        'forecast_naive_seconds':90., 'telemetry_values':[.5]*90,
        **{f'f{k}':float(k)/100 for k in range(80)}}


def npz(path, rows, **changes):
    n = len(rows); vec = np.zeros((n,16), dtype=np.float32); vec[:,0] = 1
    value = {'issuance_ids':np.array([r['issuance_id'] for r in rows]),
        'cutoff_ns':np.array([r['telemetry_cutoff_ns'] for r in rows],dtype=np.int64),
        'empty_context':np.zeros(n,bool), 'support':np.array([r['telemetry_supported'] for r in rows],bool),
        'original_feature_row_sha256':np.array([v.old.digest(r) for r in rows]),
        'context_sha256':np.array(['c'*64]*n), **{c:vec.copy() for c in v.CONTROLS}, **changes}
    np.savez(path, **value)
    return {**record(path), 'rows':n}


def test_exact_runner_npz_schema(tmp_path):
    rows = [feature_row(0), feature_row(1,False)]
    data = v.embedding_arrays(npz(tmp_path/'a.npz',rows), rows, v.old.Bindings(tmp_path))
    assert set(data) == v.NPZ_FIELDS and data['ordered'].dtype == np.float32


@pytest.mark.parametrize('change', ['clock','support','features','missing_field','empty_nonzero'])
def test_saved_embedding_corruption_rejected(tmp_path, change):
    rows = [feature_row()]
    changes = {'clock':{'cutoff_ns':np.array([1],np.int64)}, 'support':{'support':np.array([False])},
        'features':{'original_feature_row_sha256':np.array(['d'*64])},
        'missing_field':{'extra':np.array([1])}, 'empty_nonzero':{'empty_context':np.array([True])}}[change]
    with pytest.raises(AssertionError):
        v.embedding_arrays(npz(tmp_path/'bad.npz',rows,**changes),rows,v.old.Bindings(tmp_path))


def labels(rows):
    output=[]
    for i,row in enumerate(rows):
        output.append({**{k:row[k] for k in (*v.old.KEYS,'issuance_id','year','issued_at_ns')},
            'outcome_status':'matched' if i==0 else 'unmatched',
            'target_id':'202301/1/4' if i==0 else None,
            'target_at_ns':row['issued_at_ns']+90_000_000_000 if i==0 else None,
            'target_lap_number':4 if i==0 else None,
            'target_timestamp':row['issued_at_timestamp']+90 if i==0 else None,
            'lap_time_seconds':90. if i==0 else None, 'target_same_stint':True if i==0 else None})
    return output


def test_manual_closed_label_join_retains_unmatched_and_order():
    rows=[feature_row(0),feature_row(1)]; targets=labels(rows)
    result=v.attach_labels(rows,targets[::-1])
    assert result.outcome_status.tolist()==['matched','unmatched']
    assert result.issuance_id.tolist()==[r['issuance_id'] for r in rows]
    assert 'outcome_status' not in rows[0]


@pytest.mark.parametrize('change',['drop','clock','future_target','target_in_features'])
def test_label_identity_and_temporal_mutations_rejected(change):
    rows=[feature_row(0),feature_row(1)]; targets=labels(rows)
    if change=='drop': targets.pop()
    if change=='clock': targets[0]['issued_at_ns']+=1
    if change=='future_target': targets[0]['target_at_ns']=rows[0]['issued_at_ns']
    if change=='target_in_features': rows[0]['lap_time_seconds']=0
    with pytest.raises(AssertionError): v.attach_labels(rows,targets)


class FixedEstimator:
    def __init__(self,column): self.column=column
    def predict(self,frame): return frame.iloc[:,self.column].to_numpy()*10


def test_direct_estimator_replay_uses186_columns_clipping_and_exact_fallback():
    frame=pd.DataFrame([feature_row(0),feature_row(1,False)])
    base=[f'f{k}' for k in range(80)]; extra=[f't{k}' for k in range(90)]
    embeds=[f'sequence__embedding_{i:02d}' for i in range(16)]
    old_bundle={'base_features':base,'telemetry_features':extra,
        'models':{name:{'model':FixedEstimator(0 if name=='base_hgb' else 80)} for name in v.OLD_NAMES}}
    new={'base_features':base,'telemetry_features':extra,'embedding_features':embeds,
        'models':{name:{'model':FixedEstimator(170),'features':base+['telemetry__'+x for x in extra]+embeds,'kind':'hgb'} for name in v.NEW_NAMES}}
    tables={c:np.ones((2,16),np.float32) for c in v.CONTROLS}
    before=v.old.digest(frame.to_dict('records'))
    replay=v.direct_replay(new,old_bundle,frame,tables)
    assert set(replay)==set(v.NAMES) and replay['base_hgb'].tolist()==[90,90]
    for name in v.NEW_NAMES: assert replay[name].tolist()==[93,90]
    assert replay['telemetry_hgb'].tolist()==[93,90] and replay['quality_hgb'].tolist()==[90,90]
    assert before==v.old.digest(frame.to_dict('records'))


def test_bits_detect_negative_zero_and_one_ulp():
    v.bits([1.],[1.],'same')
    for a,b in [([0.],[-0.]),([1.],[np.nextafter(1.,2.)])]:
        with pytest.raises(AssertionError): v.bits(a,b,'different')


def comparisons():
    point={'relative_reduction':.02,'event_ci95':[-.03,-.01], 'block3_ci95':[-.03,-.001], 'loo_max_delta':-.005}
    return {lag:{name:deepcopy(point) for name in v.REFERENCES} for lag in ('2','0')}


@pytest.mark.parametrize('reference', v.REFERENCES)
def test_every_reference_gate_is_decisive(reference):
    values=comparisons(); assert v.gates(values)[1]
    values['2'][reference]['block3_ci95'][1]=0
    assert not v.gates(values)[1]
    values=comparisons(); values['0'][reference]['relative_reduction']=0
    assert not v.gates(values)[1]


def test_perfect_reference_does_not_create_nonfinite_or_pass():
    values=comparisons(); values['2']['base_hgb']['relative_reduction']=None
    checks,passed=v.gates(values)
    assert not passed and checks['2']['base_hgb']['minimum_relative_gain'] is False
    json.dumps(checks,allow_nan=False)


def test_independent_metric_denominators_with_unequal_event_sizes():
    events=np.array([202301,202302,202302,202302]); y=np.zeros(4)
    result=v.old.independent_comparison(events,y,np.array([1.,3,3,3]),np.array([2.,4,4,4]))
    assert result['candidate_mae']==2 and result['baseline_mae']==3 and result['delta']==-1
    assert result['event_ci95']==[-1,-1] and result['block3_ci95']==[-1,-1] and result['loo_max_delta']==-1


def test_protocol_bytes_are_unchanged():
    assert v.old.sha(Path(v.__file__).parent/'protocol.json')==v.PROTOCOL_SHA


def test_synthetic_raw_prefix_replay_all_controls(tmp_path):
    import torch
    from research.experiments.boundary_20260908.telemetry_sequence import encoder, tokens
    from research.experiments.boundary_20260908.telemetry_sequence.test_tokens import packet
    torch.set_num_threads(1)
    path=tmp_path/'raw'; path.write_bytes(b''.join(packet(i*5000,speed=100+i*8) for i in range(22)))
    model=encoder.new_model(frozen=True)
    parameters={c:{k:x.detach().numpy() for k,x in model.state_dict().items()} for c in v.CONTROLS}
    rows=[feature_row(i) for i in range(3)]; snapshots=[]
    with tokens.TokenCursor(path) as cursor:
        for row,second in zip(rows,(20,50,100)):
            row['issued_at_ns']=(second+2)*1_000_000_000
            row['telemetry_cutoff_ns']=second*1_000_000_000
            snapshot=cursor.query('1',cutoff_ns=row['telemetry_cutoff_ns'])
            snapshots.append(snapshot); row['telemetry_supported']=snapshot.supported
    keys=tuple(tokens.permutation_key(r['event_key'],r['driver_id'],s.provenance['last_admitted_driver_packet_sequence'])
               for r,s in zip(rows,snapshots))
    contexts=np.stack([s.values for s in snapshots])
    arrays={'cutoff_ns':np.array([r['telemetry_cutoff_ns'] for r in rows]),
        'support':np.array([s.supported for s in snapshots]), 'empty_context':np.zeros(3,bool),
        'context_sha256':np.array([v.old.digest(s.values) for s in snapshots]),
        **{c:encoder.embeddings(model,contexts,context_keys=keys,control=c) for c in v.CONTROLS}}
    entry={'event_key':202301,'lag_seconds':2,'raw_source':{'status':'available','decoded_body_path':str(path)}}
    result=v.replay_neural(entry,rows,arrays,parameters)
    assert len(result)==3 and all(set(r['maximum_absolute_forward_error'])==set(v.CONTROLS) for r in result)
    assert result[-1]['reconstructed_admitted_sequences']==snapshots[-1].source_sequences[snapshots[-1].source_sequences>=0].tolist()
    arrays['context_sha256'][0]='bad'
    with pytest.raises(AssertionError): v.replay_neural(entry,rows,arrays,parameters)


def test_saved_initial_states_and_exact800_trace_bindings(tmp_path):
    import torch
    from research.experiments.boundary_20260908.telemetry_sequence import encoder
    torch.set_num_threads(1); model=encoder.new_model(frozen=True)
    state=model.state_dict(); digest=v.state_digest({k:x.detach().numpy() for k,x in state.items()})
    corpus=tmp_path/'corpus.json'; corpus.write_text('{}')
    schedule=tmp_path/'schedule.json'; schedule.write_text('{}')
    corpus_lock={'corpus':record(corpus),'schedule':record(schedule)}
    trained={'initial_state_sha256':digest,'fits':2,'historical_lap_labels_read':False,
        'checkpoint_policy':'final_fixed_step_only','shared_deadline_seconds':3600,'encoders':{}}
    for control in v.CONTROLS:
        path=tmp_path/(control+'.pt'); torch.save(state,path)
        item={**record(path),'state_sha256':digest}; trained['encoders'][control]=item
        if control!='random':
            trace={'control':control,'steps':800,'batch_size':32,'seed':20260908,
                'initial_state_sha256':digest,'final_state_sha256':digest,'parameters':15048,
                'learning_rate':.001,'gradient_limit':1.,'elapsed_seconds':10.,
                'checkpoint_policy':'final_fixed_step_only','losses':[1.]*800,
                'gradient_norm_before_clip':[0.]*800,**corpus_lock}
            output=tmp_path/(control+'_training.json'); output.write_text(json.dumps(trace))
            item['training']=record(output)
    _,checks=v.verify_encoders(trained,corpus_lock,v.old.Bindings(tmp_path))
    assert checks['random_is_exact_initial_state'] and checks['recorded_steps_per_encoder']==800
    item=trained['encoders']['ordered']['training']; path=Path(item['path'])
    trace=json.loads(path.read_text()); trace['losses'].pop(); path.write_text(json.dumps(trace))
    trained['encoders']['ordered']['training']=record(path)
    with pytest.raises(AssertionError): v.verify_encoders(trained,corpus_lock,v.old.Bindings(tmp_path))
