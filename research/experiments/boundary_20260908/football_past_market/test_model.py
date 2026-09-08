"""Synthetic convexity, probability, identification and optimization regressions."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit, softmax

from . import model as m


def sample_rows(n_each=30,teams_each=(4,4,4),seed=71):
    rng=np.random.default_rng(seed);result=[]
    for li,(league,n_teams) in enumerate(zip(m.LEAGUES,teams_each)):
        strengths=rng.normal(size=n_teams)*.6;draws=rng.normal(size=n_teams)*.15
        for i in range(n_each):
            h,a=rng.choice(n_teams,2,replace=False)
            c=.4+strengths[h]-strengths[a];d=-.5+draws[h]+draws[a]
            q=softmax([c/2,d,-c/2])
            result.append(dict(league=league,home='team'+str(h),away='team'+str(a),
                weight=float(rng.uniform(.13,1)),q=q.tolist(),outcome=int(rng.choice(3,p=q)),
                match_id=f'{league}:{i}',a0_utc='2020-01-02T00:00:00+00:00'))
    return result


def finite_gradient(function,x,step=1e-6):
    return np.array([(function(x+np.eye(len(x))[j]*step)[0]-function(x-np.eye(len(x))[j]*step)[0])/(2*step) for j in range(len(x))])


def finite_hessian(function,x,step=2e-5):
    return np.column_stack([(function(x+np.eye(len(x))[j]*step)[1]-function(x-np.eye(len(x))[j]*step)[1])/(2*step) for j in range(len(x))])


@pytest.mark.parametrize('n',[2,3,20,32])
def test_centered_orthonormal_contrasts_and_penalty_norm(n):
    b=m.contrast(n);z=np.linspace(-2,3,n-1)
    np.testing.assert_allclose(b.T@b,np.eye(n-1),rtol=0,atol=1e-14)
    np.testing.assert_allclose(b.sum(0),0,rtol=0,atol=1e-14)
    assert np.sum((b@z)**2)==pytest.approx(np.sum(z*z),rel=1e-14)


@pytest.mark.parametrize('target_kind',m.TARGET_KINDS)
def test_strength_gradient_and_strong_convexity(target_kind):
    data=m.prepare_strength(sample_rows(),target_kind)
    theta=data.prior+np.linspace(-.6,.7,len(data.prior))
    function=lambda x:m.strength_objective(x,data)
    value,gradient=function(theta)
    np.testing.assert_allclose(gradient,finite_gradient(function,theta),rtol=2e-7,atol=4e-10)
    hessian=finite_hessian(function,theta)
    np.testing.assert_allclose(hessian,hessian.T,rtol=0,atol=2e-10)
    assert np.linalg.eigvalsh(hessian).min()>=m.PENALTY-1e-9
    assert np.isfinite(value)


def test_prior_is_exact_neutral_identity_with_nonuniform_weights():
    rows=sample_rows()
    for row in rows:row['q']=[.375,.25,.375]
    fitted=m.fit_strength(rows,'past_market')
    np.testing.assert_allclose(fitted['theta'],fitted['diagnostics']['prior'],rtol=0,atol=1e-14)
    fixtures=[{k:r[k] for k in ('league','home','away')} for r in rows]
    np.testing.assert_allclose(m.predict_strength(fitted,fixtures),np.tile([.375,.25,.375],(len(rows),1)),atol=1e-15)
    assert fitted['diagnostics']['gradient_max']<1e-14
    assert len(fitted['theta'])==2*sum(len(b['teams']) for b in fitted['leagues'].values())


def test_league_mass_and_common_weight_scaling_leave_objective_unchanged():
    rows=sample_rows();other=copy.deepcopy(rows)
    for row in other:row['weight']*=dict(E0=12.,I1=.5,SP1=3.)[row['league']]
    a=m.prepare_strength(rows,'past_market');b=m.prepare_strength(other,'past_market')
    np.testing.assert_allclose(a.weights,b.weights,rtol=3e-16,atol=1e-18)
    for li in range(3):assert a.weights[a.league_index==li].sum()==pytest.approx(1/3)
    theta=a.prior+np.linspace(-1,1,len(a.prior))
    for left,right in zip(m.strength_objective(theta,a),m.strength_objective(theta,b)):
        np.testing.assert_allclose(left,right,rtol=0,atol=1e-15)


def test_soft_onehot_control_equivalence_and_unused_payload_is_unread():
    rows=sample_rows()
    for row in rows:row['q']=np.eye(3)[row['outcome']].tolist()
    a=m.fit_strength(rows,'past_market');b=m.fit_strength(rows,'outcome_control')
    np.testing.assert_array_equal(a['theta'],b['theta'])
    poisoned=copy.deepcopy(rows)
    for row in poisoned:row['outcome']=object();row['unknown_future_quote']=object()
    c=m.fit_strength(poisoned,'past_market')
    np.testing.assert_array_equal(a['theta'],c['theta']);assert a['training_digest']==c['training_digest']
    for row in rows:row['q']=object()
    d=m.fit_strength(rows,'outcome_control')
    np.testing.assert_array_equal(b['theta'],d['theta'])


def test_strength_fit_serialization_constraints_and_input_immutability():
    rows=sample_rows();before=copy.deepcopy(rows)
    model=m.fit_strength(rows,'past_market');restored=json.loads(json.dumps(model,allow_nan=False))
    assert rows==before and model['diagnostics']['solver_success'] is True
    assert model['diagnostics']['gradient_max']<=1e-6
    for block in model['leagues'].values():
        assert abs(sum(block['s']))<1e-12 and abs(sum(block['v']))<1e-12
    fixtures=[{k:r[k] for k in ('league','home','away')} for r in rows]
    a=m.predict_strength(model,fixtures);b=m.predict_strength(restored,fixtures)
    np.testing.assert_array_equal(a,b);np.testing.assert_allclose(a.sum(1),1,atol=2e-16)
    assert np.all(a>0) and m.predict_strength(model,[]).shape==(0,3)
    with pytest.raises(ValueError):m.predict_strength(model,[dict(league='E0',home='unseen',away='team0')])
    restored['leagues']['E0']['s'][0]+=.01
    with pytest.raises(ValueError):m.predict_strength(restored,fixtures)


def test_each_league_owns_distinct_same_named_team_effects():
    data=m.prepare_strength(sample_rows(),'past_market')
    a=data.prior.copy();a[0]=.9
    objective,gradient=m.strength_objective(a,data)
    assert np.isfinite(objective) and np.isfinite(gradient).all()
    assert set(data.teams)==set(m.LEAGUES)
    starts=[data.slices[l][1] for l in m.LEAGUES]
    assert starts==[0,4,8]


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(weight=0),lambda r:r.update(weight=True),lambda r:r.update(weight=np.inf),
    lambda r:r.update(q=[.5,.5,.5]),lambda r:r.update(q=[True,0,0]),lambda r:r.update(q=[np.nan,0,1]),
    lambda r:r.update(league='UNKNOWN'),lambda r:r.update(home=r['away']),lambda r:r.update(home=' '),
])
def test_invalid_admitted_strength_rows_rejected(mutate):
    rows=sample_rows();mutate(rows[0])
    with pytest.raises(ValueError):m.fit_strength(rows,'past_market')


def test_missing_league_duplicate_ids_and_invalid_outcomes_rejected():
    rows=sample_rows()
    with pytest.raises(ValueError):m.fit_strength(rows[:30],'past_market')
    with pytest.raises(ValueError):m.fit_strength(rows+rows[:1],'past_market')
    rows[0]['outcome']='H'
    with pytest.raises(ValueError):m.fit_strength(rows,'outcome_control')
    with pytest.raises(ValueError):m.fit_strength(rows,'another_kind')


def test_optimizer_failure_is_not_replaced_by_another_fit(monkeypatch):
    calls=[]
    def fail(function,initial,*args,**kwargs):
        calls.append((initial.copy(),kwargs));return SimpleNamespace(x=initial,success=False,message='synthetic failure')
    monkeypatch.setattr(m,'minimize',fail)
    with pytest.raises(ValueError,match='Strength fit failed'):m.fit_strength(sample_rows(),'past_market')
    assert len(calls)==1 and calls[0][1]['method']=='L-BFGS-B'
    assert calls[0][1]['options']=={'ftol':1e-13,'gtol':1e-9,'maxiter':2000}


@pytest.mark.parametrize('theta',[[.7,-.4,1.1],[0.,-1.,.2],[2.,.8,3.]])
def test_ordered_logit_exact_probabilities_gradient_and_convexity(theta):
    theta=np.array(theta);x=np.linspace(-2,2,30);y=np.tile([0,1,2],10)
    logp,_,_,_=m._ordered(theta,x)
    beta,t1,gap=theta;a=t1-beta*x;b=a+gap
    expected=np.column_stack((expit(-b),expit(b)-expit(a),expit(a)))
    np.testing.assert_allclose(np.exp(logp),expected,rtol=1e-14,atol=1e-15)
    function=lambda t:m.ordered_objective(t,x,y)
    # Interior finite differences; the beta=0 example checks one-sided boundary below.
    if theta[0]>0:
        np.testing.assert_allclose(function(theta)[1],finite_gradient(function,theta),rtol=2e-7,atol=4e-10)
        h=finite_hessian(function,theta)
        np.testing.assert_allclose(h,h.T,atol=2e-8)
        assert np.linalg.eigvalsh(h).min()>=-1e-9
    else:
        plus=theta.copy();plus[0]+=1e-6
        numeric=(function(plus)[0]-function(theta)[0])/1e-6
        assert numeric==pytest.approx(function(theta)[1][0],abs=1e-6)


@pytest.mark.parametrize('theta',[[1.,0.,1e-6],[1.,1000.,1.],[1.,-1000.,1.],[0.,0.,1000.]])
def test_extreme_ordered_logits_are_stable_without_density_clipping(theta):
    x=np.array([-1e4,-1000.,0.,1000.,1e4]);y=np.array([0,1,2,1,0])
    value,gradient=m.ordered_objective(theta,x,y)
    logp,*_=m._ordered(theta,x)
    assert np.isfinite(logp).all() and np.isfinite(value) and np.isfinite(gradient).all()
    p=m._probabilities(logp);assert np.all(p>0)
    np.testing.assert_allclose(p.sum(1),1,atol=2e-16)
    # The draw log probability remains finite even where direct sigmoid subtraction is zero.
    assert np.isfinite(logp[:,1]).all()


def test_readout_fit_serialization_and_unweighted_objective():
    rng=np.random.default_rng(101);x=rng.normal(size=600)
    truth=np.array([1.2,-.6,1.]);logp,*_=m._ordered(truth,x)
    y=np.array([rng.choice(3,p=p) for p in np.exp(logp)])
    before=x.copy();fitted=m.fit_ordered_logit(x,y)
    assert fitted['beta']>=0 and fitted['gap']>=1e-6
    assert fitted['diagnostics']['kkt_residual']<=1e-6
    exact=m.ordered_objective(fitted['theta'],x,y)[0]
    assert fitted['diagnostics']['objective']==exact
    p=m.predict_ordered_logit(fitted,x)
    q=m.predict_ordered_logit(json.loads(json.dumps(fitted,allow_nan=False)),x)
    np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(x,before)
    assert m.readout_kkt([0,0,1e-6],[1,0,2])==0
    assert m.readout_kkt([0,0,1e-6],[-1,0,-2])==2


def test_readout_beta_boundary_neutral_fixture_counts():
    # Exact equal class counts at every x make beta=0 the convex optimum.
    x=np.repeat(np.array([-2.,-1.,0.,1.,2.]),3);y=np.tile([0,1,2],5)
    fitted=m.fit_ordered_logit(x,y)
    assert fitted['beta']<1e-7
    np.testing.assert_allclose(m.predict_ordered_logit(fitted,x),np.full((15,3),1/3),atol=1e-7)


@pytest.mark.parametrize('x,y',[
    ([1,2,3],[0,0,2]),([1,2],[0,1,2]),([1,np.nan,3],[0,1,2]),([True,False,True],[0,1,2]),
    ([1,2,3],[True,1,2]),([1,2,3],['H','D','A']),
])
def test_readout_invalid_training_inputs_fail(x,y):
    with pytest.raises(ValueError):m.fit_ordered_logit(x,y)


def test_readout_failure_no_silent_new_solver(monkeypatch):
    def fail(function,initial,*args,**kwargs):
        assert initial.tolist()==[1.,-.5,1.]
        assert kwargs['method']=='SLSQP' and kwargs['bounds']==[(0.,None),(None,None),(1e-6,None)]
        assert kwargs['options']=={'ftol':1e-12,'maxiter':2000}
        return SimpleNamespace(x=initial,success=False,message='synthetic failure')
    monkeypatch.setattr(m,'minimize',fail)
    with pytest.raises(ValueError,match='Readout failed'):m.fit_ordered_logit([1.,2.,3.],[0,1,2])
