"""Synthetic architecture, CPC mathematics and control tests; no raw inputs."""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from . import encoder as e


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tokens(shape, seed=1):
    gen=torch.Generator().manual_seed(seed)
    value=torch.rand(shape, generator=gen);value[..., -1]=0
    return value


def batch(size=2, *, seed=1):
    return e.CPCBatch(tokens((size,128,34), seed), tokens((size,3,34), seed+1),
                      tokens((size,31,34), seed+2), tuple(f'synthetic/event/driver/{i}' for i in range(size)))


def test_architecture_and_full_parameter_budget_are_exact():
    model=e.new_model()
    assert e.parameter_count(model)==15048 and e.parameter_count(model)<25000
    assert len(model.layers)==7
    for i,layer in enumerate(model.layers):
        assert layer.conv.in_channels==(34 if i==0 else 24)
        assert layer.conv.out_channels==24 and layer.conv.kernel_size==(3,)
        assert layer.conv.dilation==(2**i,) and layer.conv.padding==(0,)
        assert layer.left_padding==2*(2**i) and layer.norm.normalized_shape==(24,)
    assert model.projection.in_features==24 and model.projection.out_features==16
    assert isinstance(model.target_encoder, nn.Linear)
    assert (model.target_encoder.in_features, model.target_encoder.out_features)==(34,16)
    assert model.heads.shape==(3,16,16)
    assert not any(isinstance(v,(nn.Dropout,nn.BatchNorm1d)) for v in model.modules())
    assert e.HORIZONS==(4,16,32) and e.TRAIN_STEPS==800 and e.BATCH_SIZE==32


def test_fixed_initialization_preserves_global_rng_and_random_control_identity():
    torch.manual_seed(99);before=torch.random.get_rng_state().clone()
    primary=e.new_model();permuted=e.new_model();random=e.new_model(frozen=True)
    assert torch.equal(before,torch.random.get_rng_state())
    assert e.state_digest(primary)==e.state_digest(permuted)==e.state_digest(random)
    assert all(p.requires_grad for p in primary.parameters())
    assert not any(p.requires_grad for p in random.parameters()) and not random.training


@pytest.mark.parametrize('endpoint', [0,1,3,16,63,95,126])
def test_future_token_changes_cannot_change_any_earlier_state(endpoint):
    model=e.new_model();x=tokens((2,128,34));changed=x.clone()
    changed[:,endpoint+1:,:-1]=5-changed[:,endpoint+1:,:-1]
    with torch.no_grad():
        original=model(x);other=model(changed)
    torch.testing.assert_close(original[:,:endpoint+1],other[:,:endpoint+1],rtol=0,atol=0)


@pytest.mark.parametrize('endpoint', [0,16,64,100])
def test_prefix_replay_and_future_gradients_are_causal(endpoint):
    model=e.new_model();x=tokens((1,128,34)).requires_grad_()
    all_states=model(x);prefix=model(x[:,:endpoint+1])
    # Different convolution lengths may choose different arithmetic kernels;
    # exact zero future derivatives test causality independently of roundoff.
    torch.testing.assert_close(all_states[:,:endpoint+1],prefix,rtol=2e-6,atol=2e-6)
    gradient=torch.autograd.grad(all_states[0,endpoint].square().sum(),x)[0]
    assert torch.count_nonzero(gradient[:,endpoint+1:])==0
    assert torch.count_nonzero(gradient[:,:endpoint+1])>0


def test_layer_normalization_and_representation_do_not_mix_batch_members():
    model=e.new_model();x=tokens((3,128,34))
    first=model.representation(x[:1]);full=model.representation(x)
    torch.testing.assert_close(first,full[:1],rtol=2e-6,atol=2e-6)
    x[1:,:,:-1]=5
    torch.testing.assert_close(full[:1],model.representation(x)[:1],rtol=0,atol=0)


def test_newest_endpoint_not_temporal_pooling_and_normalized_output():
    model=e.new_model();x=tokens((2,128,34))
    raw=model(x);expected=torch.nn.functional.normalize(raw[:,-1],dim=-1)
    torch.testing.assert_close(model.representation(x),expected,rtol=0,atol=0)
    torch.testing.assert_close(expected.norm(dim=-1),torch.ones(2),rtol=1e-6,atol=1e-6)
    assert not torch.allclose(expected,torch.nn.functional.normalize(raw.mean(1),dim=-1))


def test_empty_left_padded_context_is_exact_zero_embedding():
    x=torch.zeros((2,128,34));x[:,:,-1]=1
    z=e.embeddings(e.new_model(frozen=True),x,context_keys=('a','b'),control='random')
    assert z.dtype==np.float32 and z.shape==(2,16) and np.count_nonzero(z)==0


def test_pointwise_future_target_encoder_has_no_temporal_or_batch_context():
    model=e.new_model();x=tokens((2,3,34));changed=x.clone();changed[:,1:,:-1]=5
    first=model.target_encoder(x);other=model.target_encoder(changed)
    torch.testing.assert_close(first[:,0],other[:,0],rtol=0,atol=0)


def test_logits_match_independent_bilinear_positive_plus31_negative_formula():
    model=e.new_model();b=batch()
    actual=e.cpc_logits(model,b)
    c=torch.nn.functional.normalize(model(b.contexts)[:,-1],dim=-1).detach().double()
    pos=torch.nn.functional.normalize(model.target_encoder(b.positives),dim=-1).detach().double()
    neg=torch.nn.functional.normalize(model.target_encoder(b.negatives),dim=-1).detach().double()
    heads=model.heads.detach().double();expected=torch.empty((2,3,32),dtype=torch.float64)
    for i in range(2):
        for k in range(3):
            for j in range(32): expected[i,k,j]=(c[i]@heads[k]@(pos[i,k] if j==0 else neg[i,j-1]))/.1
    torch.testing.assert_close(actual.double(),expected,rtol=2e-6,atol=2e-6)
    wanted=(torch.logsumexp(expected,dim=-1)-expected[...,0]).mean()
    torch.testing.assert_close(e.cpc_loss(model,b).double(),wanted,rtol=1e-6,atol=1e-6)


def test_uniform_scores_give_log32_objective_not_log_batch_size():
    model=e.new_model()
    with torch.no_grad(): model.heads.zero_()
    assert float(e.cpc_loss(model,batch(size=3)).detach())==pytest.approx(math.log(32),abs=5e-7)


def test_objective_derivative_matches_softmax_minus_positive_indicator():
    model=e.new_model();logits=e.cpc_logits(model,batch())
    loss=(torch.logsumexp(logits,-1)-logits[...,0]).mean()
    derivative=torch.autograd.grad(loss,logits)[0]
    expected=logits.detach().softmax(-1);expected[...,0]-=1;expected/=6
    torch.testing.assert_close(derivative,expected,rtol=2e-6,atol=2e-8)


def test_cpc_backpropagates_finite_nonzero_gradients_to_context_target_and_all_heads():
    model=e.new_model();loss=e.cpc_loss(model,batch());loss.backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert model.layers[0].conv.weight.grad.norm()>0
    assert model.target_encoder.weight.grad.norm()>0
    assert all(head.norm()>0 for head in model.heads.grad)


def test_step_clips_gradients_and_uses_frozen_adam_settings():
    model=e.new_model();optimizer=e.new_optimizer(model)
    assert optimizer.defaults['lr']==.001 and optimizer.defaults['weight_decay']==0
    before=e.state_digest(model);report=e.train_step(model,optimizer,batch())
    assert math.isfinite(report['loss']) and report['gradient_norm_before_clip']>0
    norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    assert norm<=1.000001 and e.state_digest(model)!=before


def test_same_initialization_schedule_and_steps_are_bitwise_reproducible():
    a=e.new_model();b=e.new_model();oa=e.new_optimizer(a);ob=e.new_optimizer(b)
    for i in range(3):
        value=batch(seed=20+i)
        assert e.train_step(a,oa,value)==e.train_step(b,ob,value)
    assert e.state_digest(a)==e.state_digest(b)


def test_permuted_training_and_inference_use_one_keyed_prefix_transform():
    from .tokens import permute_context
    b=batch();x=b.contexts.clone();x[:,:10,:]=0;x[:,:10,-1]=1
    transformed=e.transformed_contexts(x,b.context_keys,control='permuted')
    for i,key in enumerate(b.context_keys):
        np.testing.assert_array_equal(transformed[i].numpy(),permute_context(x[i].numpy(),key=key,seed=e.SEED))
    assert not torch.equal(transformed,x)
    torch.testing.assert_close(transformed[:,-1],x[:,-1],rtol=0,atol=0)
    torch.testing.assert_close(transformed[...,32:],x[...,32:],rtol=0,atol=0)
    model=e.new_model();before=e.state_digest(model)
    inferred=e.embeddings(model,x,context_keys=b.context_keys,control='permuted')
    torch.testing.assert_close(torch.tensor(inferred),model.representation(transformed),rtol=0,atol=0)
    assert e.state_digest(model)==before
    actual=e.cpc_logits(model,replace(b,contexts=x),control='permuted')
    expected=e.cpc_logits(model,replace(b,contexts=transformed),control='ordered')
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_prefix_key_does_not_enter_ordered_or_random_numeric_representation():
    x=tokens((2,128,34));model=e.new_model(frozen=True)
    a=e.embeddings(model,x,context_keys=('event1/driver1/packet1','event2/driver2/packet2'))
    b=e.embeddings(model,x,context_keys=('unrelated','metadata'),control='random')
    np.testing.assert_array_equal(a,b)


def test_embeddings_preserve_readonly_numpy_input_and_model_training_state():
    x=tokens((2,128,34)).numpy();before=x.copy();x.setflags(write=False)
    model=e.new_model();initial=e.state_digest(model)
    z=e.embeddings(model,x,context_keys=('a','b'))
    np.testing.assert_array_equal(x,before)
    assert model.training and e.state_digest(model)==initial and np.isfinite(z).all()


@pytest.mark.parametrize('fault', ['nan','infinity','negative','too_large','fractional_padding','nonzero_padding','right_padding','double','short','wrong_positive','wrong_negative','padded_positive','padded_negative','padded_endpoint','missing_keys'])
def test_invalid_batches_fail_before_scoring(fault):
    b=batch()
    if fault in ('nan','infinity','negative','too_large'): b.contexts[0,0,0]={'nan':float('nan'),'infinity':float('inf'),'negative':-1.,'too_large':5.1}[fault]
    elif fault=='fractional_padding': b.contexts[0,0,-1]=.5
    elif fault=='nonzero_padding': b.contexts[0,0,-1]=1
    elif fault=='right_padding': b.contexts[0,-1]=0;b.contexts[0,-1,-1]=1
    elif fault=='double': b=replace(b,contexts=b.contexts.double())
    elif fault=='short': b=replace(b,contexts=b.contexts[:,:127])
    elif fault=='wrong_positive': b=replace(b,positives=b.positives[:,:2])
    elif fault=='wrong_negative': b=replace(b,negatives=b.negatives[:,:30])
    elif fault=='padded_positive': b.positives[0,0]=0;b.positives[0,0,-1]=1
    elif fault=='padded_negative': b.negatives[0,0]=0;b.negatives[0,0,-1]=1
    elif fault=='padded_endpoint': b.contexts[0]=0;b.contexts[0,:,-1]=1
    else: b=replace(b,context_keys=('', 'b'))
    with pytest.raises((ValueError,TypeError)): e.cpc_loss(e.new_model(),b)


def test_random_is_frozen_only_and_never_a_third_pretraining_run():
    with pytest.raises(ValueError,match='never pretrained'): e.cpc_loss(e.new_model(),batch(),control='random')
    with pytest.raises(ValueError,match='Exactly'): e.fit_cpc(lambda _:batch(),control='random',deadline_monotonic=e.time.monotonic()+100)


def test_fit_uses_exact800_identical_schedule_indices_and_initialization_for_both_controls(monkeypatch):
    called=[];value=batch(32)
    monkeypatch.setattr(e,'train_step',lambda model,optimizer,batch,control: {'loss':1.,'gradient_norm_before_clip':.5})
    def schedule(index): called.append(index);return value
    a,ra=e.fit_cpc(schedule,control='ordered',deadline_monotonic=e.time.monotonic()+100)
    b,rb=e.fit_cpc(schedule,control='permuted',deadline_monotonic=e.time.monotonic()+100)
    assert called==list(range(800))*2
    assert ra['initial_state_sha256']==rb['initial_state_sha256']==e.state_digest(e.new_model())
    assert ra['steps']==rb['steps']==800 and len(ra['losses'])==800
    assert not a.training and not b.training and not any(p.requires_grad for p in a.parameters())


def test_training_batch32_is_required_before_update(monkeypatch):
    monkeypatch.setattr(e,'train_step',lambda *a,**kw:pytest.fail('Wrong batch reached optimizer'))
    with pytest.raises(ValueError,match='batch32'):
        e.fit_cpc(lambda _:batch(2),control='ordered',deadline_monotonic=e.time.monotonic()+10)


def test_shared_deadline_overrun_rejects_incomplete_checkpoint(monkeypatch):
    clock=iter([0.,0.,0.,11.]);monkeypatch.setattr(e,'time',SimpleNamespace(monotonic=lambda:next(clock)))
    calls=[]
    def update(*args,**kwargs): calls.append(True);return {'loss':1.,'gradient_norm_before_clip':.5}
    monkeypatch.setattr(e,'train_step',update)
    with pytest.raises(TimeoutError,match='after step'):
        e.fit_cpc(lambda _:batch(32),control='ordered',deadline_monotonic=10.)
    assert len(calls)==1


@pytest.mark.parametrize('deadline', [float('nan'),float('inf'),-1,True])
def test_invalid_training_deadline_is_rejected(deadline):
    with pytest.raises(ValueError,match='deadline'):
        e.fit_cpc(lambda _:batch(32),control='ordered',deadline_monotonic=deadline)


def test_bounded_microbenchmark_is_synthetic_and_counts_actual_parameters():
    result=e.synthetic_benchmark(steps=1)
    assert result['synthetic_only'] is True and result['parameters']==15048
    assert result['context_tensor_bytes']==557056 and result['cpu_threads']==1
    assert 0<result['seconds_per_step']<60


@pytest.mark.parametrize('steps',[0,21,True,1.5])
def test_benchmark_cannot_expand_into_unbounded_training(steps):
    with pytest.raises(ValueError):e.synthetic_benchmark(steps=steps)
