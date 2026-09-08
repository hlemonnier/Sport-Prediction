"""Convex centered team strengths and a constrained ordered-logit readout.

Only already admitted numeric rows enter this module. The caller owns clocks,
quote availability, source hashes, support and exact incumbent fallbacks.
Suggested commit: research(football): fit centered past-market strengths
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from numbers import Real

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits

LEAGUES = ('E0', 'I1', 'SP1')
PENALTY = .002
FLOOR = 1e-12
PRIOR_D = -0.40546510810816444
MIN_GAP = 1e-6
TARGET_KINDS = ('past_market', 'outcome_control')


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def contrast(n):
    """Deterministic Helmert columns: B.T B=I and ones.T B=0."""
    if type(n) is not int or n < 2:
        raise ValueError('At least two observed teams are required per league')
    b=np.zeros((n,n-1))
    for j in range(n-1):
        denominator=np.sqrt((j+1)*(j+2))
        b[:j+1,j]=1/denominator;b[j+1,j]=-(j+1)/denominator
    return b


def _number(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):
        raise ValueError(name+' must be finite numeric')
    result=float(value)
    if not np.isfinite(result):raise ValueError(name+' exceeds finite float64 range')
    return result


def _team(value):
    if not isinstance(value,str) or not value or value != value.strip():
        raise ValueError('Team names must be nonempty canonical strings')
    return value


def _outcomes(values):
    if not all(isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_)) and 0 <= v <= 2 for v in values):
        raise ValueError('Outcomes must be integer H=0, D=1, A=2')
    return np.asarray(values,dtype=np.int64)


@dataclass
class StrengthData:
    teams: dict
    bases: dict
    slices: dict
    league_index: np.ndarray
    home_index: np.ndarray
    away_index: np.ndarray
    weights: np.ndarray
    targets: np.ndarray
    prior: np.ndarray
    normalization: dict
    training_digest: str
    training_ids: list


def prepare_strength(rows,target_kind):
    if target_kind not in TARGET_KINDS or not isinstance(rows,list) or not rows:
        raise ValueError('A declared target kind and nonempty admitted row list are required')
    teams={league:set() for league in LEAGUES};clean=[]
    for row in rows:
        if not isinstance(row,dict) or row.get('league') not in LEAGUES:
            raise ValueError('Every row requires a fixed supported league')
        league=row['league'];home=_team(row['home']);away=_team(row['away'])
        if home==away:raise ValueError('A team cannot play itself')
        weight=_number(row['weight'],'weight')
        if weight<=0:raise ValueError('Admitted weights must be strictly positive')
        if target_kind=='past_market':
            q=row['q']
            if not isinstance(q,(list,tuple,np.ndarray)) or len(q)!=3:
                raise ValueError('Expected an H/D/A soft-target vector')
            target=np.array([_number(v,'q') for v in q])
            if (target<0).any() or (target>1).any() or not np.isclose(target.sum(),1.,rtol=0,atol=1e-12):
                raise ValueError('Soft targets must be on the probability simplex')
        else:
            index=_outcomes([row['outcome']])[0];target=np.eye(3)[index]
        item={'league':league,'home':home,'away':away,'weight':weight,'target':target.tolist()}
        for key in ('match_id','a0_utc'):
            if key in row:
                if not isinstance(row[key],str) or not row[key]:raise ValueError('Invalid '+key)
                item[key]=row[key]
        clean.append(item);teams[league].update((home,away))
    ids=[r['match_id'] for r in clean if 'match_id' in r]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate admitted match IDs')
    teams={league:sorted(names) for league,names in teams.items()}
    bases={league:contrast(len(teams[league])) for league in LEAGUES}
    slices={};team_maps={};free_start=team_start=0;prior=[]
    for league in LEAGUES:
        n=len(teams[league]);width=2*n
        slices[league]=(free_start,team_start,n)
        team_maps[league]={name:team_start+i for i,name in enumerate(teams[league])}
        prior.extend([0.,PRIOR_D]+[0.]*(width-2));free_start+=width;team_start+=n
    league_index=np.array([LEAGUES.index(r['league']) for r in clean],dtype=np.int64)
    weights=np.array([r['weight'] for r in clean]);normalization={}
    for i,league in enumerate(LEAGUES):
        mask=league_index==i
        # Scaling before summation prevents overflow without changing normalized weights.
        scale=float(weights[mask].max());scaled=weights[mask]/scale;total=float(scaled.sum())
        raw_sum=scale*total
        if not np.isfinite(raw_sum):raise ValueError('League weight sum is not finite')
        weights[mask]=scaled/(3*total)
        normalization[league]={'rows':int(mask.sum()),'raw_weight_sum':raw_sum,
            'normalization_scale':scale,'scaled_weight_sum':total,'objective_weight_sum':float(weights[mask].sum())}
    return StrengthData(teams,bases,slices,league_index,
        np.array([team_maps[r['league']][r['home']] for r in clean]),
        np.array([team_maps[r['league']][r['away']] for r in clean]),weights,
        np.array([r['target'] for r in clean]),np.array(prior),normalization,
        _digest(clean),[r.get('match_id') for r in clean])


def strength_objective(theta,data):
    theta=np.asarray(theta,dtype=float)
    if theta.shape!=data.prior.shape or not np.isfinite(theta).all():raise ValueError('Invalid free coordinates')
    n_teams=sum(len(v) for v in data.teams.values());s=np.zeros(n_teams);v=np.zeros(n_teams)
    h=np.zeros(3);d=np.zeros(3)
    for li,league in enumerate(LEAGUES):
        start,offset,n=data.slices[league];b=data.bases[league]
        h[li],d[li]=theta[start:start+2]
        s[offset:offset+n]=b@theta[start+2:start+n+1]
        v[offset:offset+n]=b@theta[start+n+1:start+2*n]
    contrast_value=h[data.league_index]+s[data.home_index]-s[data.away_index]
    draw=d[data.league_index]+v[data.home_index]+v[data.away_index]
    logits=np.column_stack((contrast_value/2,draw,-contrast_value/2))
    if not np.isfinite(logits).all():raise ValueError('Nonfinite logits')
    logp=logits-logsumexp(logits,axis=1,keepdims=True)
    displacement=theta-data.prior
    objective=-np.sum(data.weights[:,None]*data.targets*logp)+PENALTY/2*(displacement@displacement)
    residual=(np.exp(logp)*data.targets.sum(axis=1,keepdims=True)-data.targets)*data.weights[:,None]
    gcontrast=(residual[:,0]-residual[:,2])/2;gdraw=residual[:,1]
    gh=np.bincount(data.league_index,weights=gcontrast,minlength=3)
    gd=np.bincount(data.league_index,weights=gdraw,minlength=3)
    gs=np.bincount(data.home_index,weights=gcontrast,minlength=n_teams)-np.bincount(data.away_index,weights=gcontrast,minlength=n_teams)
    gv=np.bincount(data.home_index,weights=gdraw,minlength=n_teams)+np.bincount(data.away_index,weights=gdraw,minlength=n_teams)
    gradient=PENALTY*displacement
    for li,league in enumerate(LEAGUES):
        start,offset,n=data.slices[league];b=data.bases[league]
        gradient[start:start+2]+=gh[li],gd[li]
        gradient[start+2:start+n+1]+=b.T@gs[offset:offset+n]
        gradient[start+n+1:start+2*n]+=b.T@gv[offset:offset+n]
    return float(objective),gradient


def fit_strength(rows,target_kind):
    data=prepare_strength(rows,target_kind)
    with threadpool_limits(limits=1):
        result=minimize(strength_objective,data.prior.copy(),args=(data,),jac=True,method='L-BFGS-B',
            options={'ftol':1e-13,'gtol':1e-9,'maxiter':2000})
    objective,gradient=strength_objective(result.x,data);maximum=float(np.abs(gradient).max())
    if not result.success or not np.isfinite(objective) or maximum>1e-6:
        raise ValueError(f'Strength fit failed: {result.message}; gradient={maximum}')
    leagues={}
    for league in LEAGUES:
        start,_,n=data.slices[league];b=data.bases[league]
        sf=result.x[start+2:start+n+1];vf=result.x[start+n+1:start+2*n]
        leagues[league]={'teams':data.teams[league],'h':float(result.x[start]),'d':float(result.x[start+1]),
            's':(b@sf).tolist(),'v':(b@vf).tolist(),'s_free':sf.tolist(),'v_free':vf.tolist()}
    return {'kind':'past_market_strength_v1','target_kind':target_kind,'league_order':list(LEAGUES),
        'leagues':leagues,'theta':result.x.tolist(),'penalty':PENALTY,'floor':FLOOR,
        'training_digest':data.training_digest,'training_ids':data.training_ids,'normalization':data.normalization,
        'diagnostics':{'rows':len(rows),'objective':objective,'gradient_max':maximum,'gradient':gradient.tolist(),
            'iterations':int(result.nit),'function_evaluations':int(result.nfev),'solver':'L-BFGS-B',
            'solver_success':bool(result.success),'message':str(result.message),'prior':data.prior.tolist()}}


def _probabilities(logp):
    p=np.maximum(np.exp(logp),FLOOR);p/=p.sum(axis=1,keepdims=True)
    if not np.isfinite(p).all() or (p<=0).any():raise ValueError('Invalid predictive probabilities')
    return p


def predict_strength(model,fixtures):
    if (model.get('kind')!='past_market_strength_v1' or model.get('target_kind') not in TARGET_KINDS
            or model.get('league_order')!=list(LEAGUES) or model.get('penalty')!=PENALTY or model.get('floor')!=FLOOR):
        raise ValueError('Unknown strength model contract')
    tables={};theta=[]
    for league in LEAGUES:
        block=model['leagues'][league];teams=block['teams'];n=len(teams);b=contrast(n)
        if teams!=sorted(set(teams)) or any(_team(t)!=t for t in teams):raise ValueError('Invalid team roster')
        s=np.asarray(block['s'],float);v=np.asarray(block['v'],float)
        sf=np.asarray(block['s_free'],float);vf=np.asarray(block['v_free'],float)
        h=_number(block['h'],'h');d=_number(block['d'],'d')
        if (s.shape!=(n,) or v.shape!=(n,) or sf.shape!=(n-1,) or vf.shape!=(n-1,)
                or not np.isfinite(np.r_[s,v,sf,vf]).all()
                or not np.allclose(s,b@sf,rtol=0,atol=1e-10) or not np.allclose(v,b@vf,rtol=0,atol=1e-10)):
            raise ValueError('Centered actual effects differ from free coordinates')
        tables[league]=(block,{t:i for i,t in enumerate(teams)},s,v,h,d)
        theta.extend([h,d,*sf,*vf])
    if not np.array_equal(np.array(theta),np.asarray(model['theta'],float)):
        raise ValueError('Serialized free coordinates are inconsistent')
    logits=[]
    for fixture in fixtures:
        league=fixture['league'];home=_team(fixture['home']);away=_team(fixture['away'])
        if league not in tables or home==away:raise ValueError('Invalid fixture')
        _,index,s,v,h,d=tables[league]
        if home not in index or away not in index:raise ValueError('Unsupported team requires caller incumbent fallback')
        i,j=index[home],index[away];a=(h+s[i]-s[j])/2
        logits.append([a,d+v[i]+v[j],-a])
    if not logits:return np.empty((0,3))
    values=np.array(logits)
    if not np.isfinite(values).all():raise ValueError('Nonfinite fixture logits')
    return _probabilities(values-logsumexp(values,axis=1,keepdims=True))


def _ordered(theta,x):
    beta,t1,gap=np.asarray(theta,dtype=float)
    if not np.isfinite([beta,t1,gap]).all() or beta<0 or gap<MIN_GAP:raise ValueError('Invalid ordered-logit constraints')
    a=t1-beta*x;b=a+gap
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite ordered logits')
    logp=np.column_stack((-np.logaddexp(0.,b),
        -np.logaddexp(0.,-b)-np.logaddexp(0.,a)+np.log(-np.expm1(-gap)),
        -np.logaddexp(0.,-a)))
    return logp,a,b,gap


def ordered_objective(theta,x,y):
    logp,a,b,gap=_ordered(theta,x)
    da=np.column_stack((-expit(b),expit(-b)-expit(a),expit(-a)))
    dg=np.column_stack((-expit(b),expit(-b)+np.exp(-gap)/(-np.expm1(-gap)),np.zeros(len(x))))
    indices=np.arange(len(x));gradient_a=da[indices,y];gradient_g=dg[indices,y]
    return float(-logp[indices,y].mean()),np.array([np.mean(x*gradient_a),-gradient_a.mean(),-gradient_g.mean()])


def _x(values):
    raw=np.asarray(values)
    if raw.ndim!=1 or not len(raw) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise ValueError('Expected a nonempty finite numeric rating-difference vector')
    return raw.astype(float)


def readout_kkt(theta,gradient):
    lower=np.array([0.,-np.inf,MIN_GAP]);theta=np.asarray(theta);gradient=np.asarray(gradient)
    return float(np.max(np.abs(theta-np.maximum(theta-gradient,lower))))


def fit_ordered_logit(x,y):
    x=_x(x);y=_outcomes(list(y))
    if y.shape!=x.shape or set(y.tolist())!={0,1,2}:raise ValueError('Readout needs aligned labels and all three classes')
    initial=np.array([1.,-.5,1.])
    with threadpool_limits(limits=1):
        result=minimize(ordered_objective,initial,args=(x,y),jac=True,method='SLSQP',
            bounds=[(0.,None),(None,None),(MIN_GAP,None)],options={'ftol':1e-12,'maxiter':2000})
    value,gradient=ordered_objective(result.x,x,y);kkt=readout_kkt(result.x,gradient)
    if not result.success or not np.isfinite(value) or kkt>1e-6:raise ValueError(f'Readout failed: {result.message}; KKT={kkt}')
    beta,t1,gap=map(float,result.x)
    return {'kind':'ordered_logit_v1','beta':beta,'t1':t1,'t2':t1+gap,'gap':gap,'theta':result.x.tolist(),'floor':FLOOR,
        'training_digest':_digest({'x':x.tolist(),'y':y.tolist()}),
        'diagnostics':{'rows':len(x),'objective':value,'gradient':gradient.tolist(),'kkt_residual':kkt,
            'iterations':int(result.nit),'function_evaluations':int(result.nfev),'solver':'SLSQP',
            'solver_success':bool(result.success),'message':str(result.message),'initial':[1.,-.5,1.]}}


def predict_ordered_logit(model,x):
    x=_x(x)
    if (model.get('kind')!='ordered_logit_v1' or model.get('floor')!=FLOOR
            or model['theta']!=[model['beta'],model['t1'],model['gap']]
            or model['t2']!=model['t1']+model['gap']):raise ValueError('Invalid readout serialization')
    logp,_,_,_=_ordered(model['theta'],x)
    return _probabilities(logp)
