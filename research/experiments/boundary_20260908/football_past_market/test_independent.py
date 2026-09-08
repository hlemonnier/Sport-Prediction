"""Synthetic-only independent probability, timing and statistic regressions."""
from copy import deepcopy
from dataclasses import make_dataclass
from datetime import date, datetime, timedelta, timezone
import importlib
import math

import numpy as np
import pytest

from research.experiments.boundary_20260908.football_past_market import independent as m


def strength():
    return {'kind': 'past_market_strength_v1', 'league_order': list(m.LEAGUES),
            'leagues': {league: {'teams': ['A', 'B', 'C'], 'h': 0., 'd': math.log(2/3),
                's': [0., 0., 0.], 'v': [0., 0., 0.], 's_free': [0., 0.], 'v_free': [0., 0.]}
                for league in m.LEAGUES}}


def readout(beta=1., low=-.5, high=.5):
    return {'kind': 'ordered_logit_v1', 'beta': beta, 't1': low, 't2': high,
            'gap': high-low, 'theta': [beta, low, high-low]}


def raw(day='2022-01-01', league='E0', mid=None, **kw):
    result = {'match_id': mid or f'{league}:{day}:A:B', 'league': league, 'season': 2021,
        'home': 'A', 'away': 'B', 'day': date.fromisoformat(day),
        'payload': {'AvgH': '2.0', 'AvgD': '3.0', 'AvgA': '4.0', 'FTR': 'H', 'FTHG': '2', 'FTAG': '1'}}
    result.update(kw)
    return result


def scored():
    random = np.random.default_rng(27)
    rows = []
    for league in m.LEAGUES:
        for season in (2022, 2023):
            for j, day in enumerate((0, 0, 2, 8, 27, 28, 29, 66, 99)):
                rows.append({'match_id': f'{league}:{season}:{j}', 'league': league, 'season': season,
                    'day': (date(season, 8, 1)+timedelta(days=day)).isoformat(), 'label': j%3,
                    'probabilities': {'candidate': random.dirichlet([2., 3., 2.]).tolist(),
                                      'reference': random.dirichlet([2., 3., 2.]).tolist()}})
    # Stratum/block insertion order is deliberately not chronological.
    return [rows[i] for i in random.permutation(len(rows))]


def assert_nested_close(a, b):
    if isinstance(b, dict):
        assert set(a) == set(b)
        for key in b:
            assert_nested_close(a[key], b[key])
    elif isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_close(x, y)
    elif isinstance(b, float):
        assert a == pytest.approx(b, rel=0, abs=2e-13)
    else:
        assert a == b


def test_strength_prior_and_actual_effect_orientation():
    payload = strength()
    np.testing.assert_allclose(m.strength_predict(payload, 'E0', 'A', 'B'), [.375, .25, .375], rtol=0, atol=1e-16)
    payload['leagues']['E0']['s'] = [1., -1., 0.]
    home = m.strength_predict(payload, 'E0', 'A', 'B')
    away = m.strength_predict(payload, 'E0', 'B', 'A')
    np.testing.assert_allclose(home, away[::-1], rtol=0, atol=1e-16)
    assert home[0] > .375 and home[2] < .375
    payload['leagues']['E0']['v'] = [.5, .5, -1.]
    assert m.strength_predict(payload, 'E0', 'A', 'B')[1] > home[1]


def test_free_coordinates_cannot_override_actual_parameter_replay():
    payload = strength()
    expected = m.strength_predict(payload, 'E0', 'A', 'B')
    payload['leagues']['E0']['s_free'] = [999., -999.]
    payload['leagues']['E0']['v_free'] = [999., -999.]
    np.testing.assert_array_equal(m.strength_predict(payload, 'E0', 'A', 'B'), expected)


def test_strength_extreme_finite_logits_and_output_floor():
    payload = strength()
    payload['leagues']['E0']['s'] = [1000., -1000., 0.]
    p = m.strength_predict(payload, 'E0', 'A', 'B')
    assert np.isfinite(p).all() and (p > 0).all()
    assert p[0] > 1.-3e-12 and sum(p) == pytest.approx(1.)


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(kind='other'),
    lambda p: p.update(league_order=['SP1', 'I1', 'E0']),
    lambda p: p['leagues'].pop('SP1'),
    lambda p: p['leagues']['E0'].update(teams=['B', 'A', 'C']),
    lambda p: p['leagues']['E0'].update(teams=['A', 'A', 'C']),
    lambda p: p['leagues']['E0'].update(s=[1., 0., 0.]),
    lambda p: p['leagues']['E0'].update(v=[0., 0.]),
    lambda p: p['leagues']['E0'].update(h=np.inf),
    lambda p: p['leagues']['E0'].update(d=True),
])
def test_invalid_strength_payload_fails(mutation):
    payload = strength()
    mutation(payload)
    with pytest.raises(ValueError):
        m.strength_predict(payload, 'E0', 'A', 'B')


def test_unknown_team_needs_explicit_fallback():
    with pytest.raises(ValueError, match='Unknown teams'):
        m.strength_predict(strength(), 'E0', 'A', 'future-team')


@pytest.mark.parametrize('difference', [-1000000., -1200., 0., 1200., 1000000.])
def test_ordinal_distribution_symmetry_and_order(difference):
    p = m.ordered_logit_predict(readout(), 1000.+difference, 1000.)
    q = m.ordered_logit_predict(readout(), 1000.-difference, 1000.)
    np.testing.assert_allclose(p, q[::-1], rtol=0, atol=2e-16)
    assert np.isfinite(p).all() and min(p) > 0 and sum(p) == pytest.approx(1.)
    if difference > 0:
        assert p[0] > p[2]


def test_ordinal_rare_draw_avoids_sigmoid_subtraction_cancellation():
    p = m.ordered_logit_predict(readout(low=25., high=25.1), 1000., 1000.)
    expected_draw = math.exp(-25.)*(-math.expm1(-.1))/((1.+math.exp(-25.))*(1.+math.exp(-25.1)))
    assert p[1] == pytest.approx(expected_draw, rel=1e-12, abs=0)


def test_zero_slope_ordinal_is_rating_invariant():
    np.testing.assert_array_equal(m.ordered_logit_predict(readout(beta=0), 1., 10000.),
                                  m.ordered_logit_predict(readout(beta=0), 10000., 1.))


@pytest.mark.parametrize('changes', [{'beta': -1.}, {'gap': 0.}, {'t2': -.5}, {'t1': np.nan},
    {'beta': True}, {'gap': 2.}, {'kind': 'other'}])
def test_invalid_readout_rejected(changes):
    payload = readout(); payload.update(changes)
    with pytest.raises(ValueError):
        m.ordered_logit_predict(payload, 1000., 1000.)


def test_fallback_preserves_exact_bits_without_reading_prediction():
    p = np.array([.3, .25, .45000000000000007])
    for blend in (False, True):
        result = m.fallback_or_blend(object(), p, supported=False, blend=blend)
        np.testing.assert_array_equal(result.view(np.uint64), p.view(np.uint64))
    new = np.array([.5, .25, .25])
    np.testing.assert_array_equal(m.fallback_or_blend(new, p, supported=True, blend=True), .5*new+.5*p)


@pytest.mark.parametrize('flag', [0, 1, 'false', None, np.bool_(True)])
def test_fallback_support_is_strict_boolean(flag):
    with pytest.raises(ValueError):
        m.fallback_or_blend([.3, .3, .4], [.3, .3, .4], supported=flag)


@pytest.mark.parametrize('league', m.LEAGUES)
@pytest.mark.parametrize('day,extra,hours', [('2023-03-25',1,23), ('2023-10-28',1,25),
    ('2023-03-19',7,167), ('2023-10-22',7,169)])
def test_calendar_delays_cross_dst_exactly(league, day, extra, hours):
    a0, available = m.availability(day, league, extra)
    assert (available-a0).total_seconds() == hours*3600
    assert a0.tzinfo == available.tzinfo == timezone.utc


def test_calendar_lags_are_not_primary_plus_seven():
    a0, primary = m.availability('2023-01-01', 'E0', 1)
    _, sensitivity = m.availability('2023-01-01', 'E0', 7)
    assert (sensitivity-primary).days == 6 and (sensitivity-a0).days == 7


def test_strict_eligibility_and_naive_utc():
    _, clock = m.availability('2023-01-01', 'E0', 1)
    assert not m.admitted('2023-01-01', 'E0', clock, 1)
    assert m.admitted('2023-01-01', 'E0', clock+timedelta(microseconds=1), 1)
    assert m.utc(clock.replace(tzinfo=None)) == clock


def test_rolling_local_calendar_left_endpoint():
    cutoff = datetime(2023, 7, 1, 22, tzinfo=timezone.utc)  # RomeJul2, LondonJul1.
    day = date(2023, 7, 2)-timedelta(days=1095)
    assert m.admitted(day, 'I1', cutoff, 1)
    assert not m.admitted(day-timedelta(days=1), 'I1', cutoff, 1)


class Poison(dict):
    def get(self, *args, **kwargs):
        raise AssertionError('Future or skipped payload was inspected')


def test_future_equalclock_outofwindow_and_missing_quote_poison_invariance():
    prior = raw()
    cutoff = datetime(2022, 1, 5, tzinfo=timezone.utc)
    future = raw('2022-01-04', payload=Poison())
    same = raw('2022-01-03', payload=Poison())
    old = raw('2018-01-01', payload=Poison())
    expected = m.history_membership([prior], cutoff, extra_days=1)
    actual = m.history_membership([prior, future, same, old], cutoff, extra_days=1)
    assert actual == expected
    missing = raw('2022-01-02', payload={'AvgH': '', 'FTR': object(), 'FTHG': object(), 'FTAG': object()})
    result = m.history_membership([prior, missing], cutoff, extra_days=1)
    assert result['training_ids'] == expected['training_ids']
    assert result['missing_quote_ids'] == [missing['match_id']]


def test_future_dataclass_payload_property_is_not_fetched():
    class Future:
        league='E0'; day=date(2022, 1, 10)
        @property
        def payload(self):
            raise AssertionError('payload fetched')
    result=m.history_membership([Future()], '2022-01-05T00:00:00', extra_days=1)
    assert result['training_ids']==[]


def test_history_matches_rawfixture_attributes_and_reconstructs_weights():
    rows = [raw(league=league, mid=league+':one') for league in m.LEAGUES]
    rows += [raw('2022-01-02', league='E0', mid='E0:two')]
    result = m.history_membership(rows, '2022-01-05T00:00:00', extra_days=1)
    expected = sorted(rows, key=lambda row:(m.availability(row['day'],row['league'],1)[1], row['match_id']))
    assert result['training_ids'] == [row['match_id'] for row in expected]
    assert result['missing_leagues'] == []
    assert math.fsum(result['objective_weights']) == pytest.approx(1.)
    for league in m.LEAGUES:
        assert sum(w for row,w in zip(result['rows'],result['objective_weights']) if row['league']==league) == pytest.approx(1/3)
    e0 = result['training_ids'].index('E0:one')
    assert result['raw_weights'][e0] == 2.**(-3./365.)
    Record = make_dataclass('Record', [(k,object) for k in rows[0]])
    assert m.history_membership([Record(**r) for r in rows], '2022-01-05T00:00:00', extra_days=1) == result


def test_quote_only_elo_history_never_reads_outcomes_and_has_no_rolling_reset():
    old = raw('2017-08-01', season=2017, payload={'BbAvH':'2','BbAvD':'3','BbAvA':'4', 'FTR':object()})
    result = m.history_membership([old], '2023-01-01T00:00:00', extra_days=1, windowed=False, include_outcomes=False)
    assert result['training_ids']==[old['match_id']] and 'outcome' not in result['rows'][0]
    assert m.history_membership([old], '2023-01-01T00:00:00', extra_days=1)['training_ids']==[]
    assert m.elo_replay([old], '2023-01-01T00:00:00', extra_days=1)[('E0','A')] != 1000.


@pytest.mark.parametrize('key,value', [('FTR','X'),('FTR','A'),('FTHG','2.5'),('FTAG','-1'),('FTHG',True),('FTAG','nan')])
def test_admitted_outcome_corruption_fails(key,value):
    row=raw();row['payload'][key]=value
    with pytest.raises(ValueError):
        m.history_membership([row], '2022-01-05', extra_days=1)


def test_duplicate_admitted_ids_and_undeclared_lag_fail():
    with pytest.raises(ValueError,match='unique'):
        m.history_membership([raw(),raw()], '2022-01-05', extra_days=1)
    for lag in (0,2,8,True):
        with pytest.raises(ValueError):m.history_membership([], '2022-01-05', extra_days=lag)


@pytest.mark.parametrize('odds', [[2.,3.,4.], [3.,3.,3.], [10.,20.,30.], [1.001,100.,10000.], [1e100,1e200,1e300], [1.+2**-52]*3])
@pytest.mark.parametrize('method',['normalized','power'])
def test_independent_devig_matches_frozen_kernel(odds,method):
    from research.experiments.boundary_20260908.football_information.model import devig
    actual,power=m.devig(odds,method);expected,k=devig(odds,method)
    np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-13)
    if power is not None:assert power==pytest.approx(k,rel=2e-11)


@pytest.mark.parametrize('odds', [[True,3.,4.], [1.,3.,4.], [np.nan,3.,4.], [np.inf,3.,4.], [2.,3.]])
def test_invalid_decimal_odds(odds):
    with pytest.raises(ValueError):m.devig(odds)


def test_odds_cache_is_bounded_and_cannot_be_mutated_by_callers():
    m._cached_devig.cache_clear()
    a,k=m.devig([2.,3.,4.]); original=a.copy(); a[:]=0
    b,k2=m.devig([2.,3.,4.])
    np.testing.assert_array_equal(b,original)
    assert k==k2 and m._cached_devig.cache_info().hits==1
    assert m._cached_devig.cache_info().maxsize==32768


def test_atomic_elo_equation_is_not_sequential_and_conserves_sum():
    before={('E0','A'):1100.,('E0','B'):900.,('E0','C'):1000.}
    rows=[{'match_id':'one','league':'E0','home':'A','away':'B','q':[.5,.25,.25]},
          {'match_id':'two','league':'E0','home':'A','away':'C','q':[.4,.3,.3]}]
    actual=m.elo_atomic_update(before,rows)
    d1=175.*(.625-1/(1+10**(-280/400)))
    d2=175.*(.55-1/(1+10**(-180/400)))
    assert actual[('E0','A')]==pytest.approx(1100.+d1+d2)
    assert math.fsum(actual.values())==pytest.approx(math.fsum(before.values()))
    assert actual==m.elo_atomic_update(before,rows[::-1])
    assert actual != m.elo_atomic_update(m.elo_atomic_update(before,rows[:1]),rows[1:])
    assert before[('E0','A')]==1100.


def test_elo_new_team_neutral_and_league_names_isolated():
    row={'league':'I1','home':'A','away':'B','q':[.5,.25,.25]}
    after=m.elo_atomic_update({('E0','A'):1500.},[row])
    assert after[('E0','A')]==1500.
    assert after[('I1','A')]+after[('I1','B')]==pytest.approx(2000.)


def test_metrics_match_frozen_helper_without_normalizing_incumbents():
    from research.experiments.performance_20260907.football import benchmark
    rows=scored();before=deepcopy(rows)
    assert_nested_close(m.metrics(rows,'candidate'),benchmark.metrics(rows,'candidate'))
    assert rows==before


@pytest.mark.parametrize('days',[1,7,28])
def test_independent_block_resampling_matches_inherited_calendar_semantics(days):
    from research.experiments.performance_20260907.football import benchmark
    rows=scored(); spec={'uncertainty':{'resamples':4000,'seed':20260908}}
    adjusted=[{**r,'season':f"{r['league']}:{r['season']}",
               'probabilities':{**r['probabilities'],'dc_equal':r['probabilities']['reference']}} for r in rows]
    expected=benchmark.paired_uncertainty(adjusted,'candidate',spec,days)
    expected['difference_candidate_minus_reference']=expected.pop('difference_candidate_minus_dc_equal')
    expected['reference']='reference'
    assert_nested_close(m.paired_uncertainty(rows,'candidate','reference',days,spec),expected)


def test_metrics_manual_class_sum_and_top_label_bin():
    rows=[{'label':0,'probabilities':{'m':[.5,.3,.2]}}, {'label':2,'probabilities':{'m':[.2,.3,.5]}}]
    result=m.metrics(rows,'m')
    assert result['log_loss']==pytest.approx(math.log(2.))
    assert result['brier_sum_classes']==pytest.approx(.38)
    assert result['accuracy']==1 and result['top_label_ece_10_bins']==.5
    assert result['calibration_bins']==[{'lower':.5,'upper':.6,'n':2,'mean_confidence':.5,'accuracy':1.}]


@pytest.mark.parametrize('fault',['label_bool','label_float','label_bad','nan','zero','negative','not_simplex','empty'])
def test_metrics_reject_invalid_inputs(fault):
    rows=[{'label':0,'probabilities':{'m':[.5,.3,.2]}}]
    if fault=='empty':rows=[]
    elif fault.startswith('label'):rows[0]['label']={'label_bool':True,'label_float':0.,'label_bad':3}[fault]
    else:rows[0]['probabilities']['m']={'nan':[np.nan,.3,.2],'zero':[0.,.5,.5],
        'negative':[-.1,.5,.6],'not_simplex':[.5,.5,.5]}[fault]
    with pytest.raises(ValueError):m.metrics(rows,'m')


def numerical_payload(target_kind):
    from research.experiments.boundary_20260908.football_past_market import model
    rows=[]
    for league in m.LEAGUES:
        for i,(home,away) in enumerate((('A','B'),('A','C'),('B','C'),('B','A'),('C','A'),('C','B'))):
            rows.append({'match_id':f'{league}:{i}','league':league,'home':home,'away':away,
                         'weight':.3+.1*i,'q':[.43,.27,.30],'outcome':i%3})
    data=model.prepare_strength(rows,target_kind)
    theta=data.prior+np.random.default_rng(77).normal(size=len(data.prior))*.25
    leagues={}
    for league in m.LEAGUES:
        start,_,n=data.slices[league]; sf=theta[start+2:start+n+1]; vf=theta[start+n+1:start+2*n]
        leagues[league]={'teams':data.teams[league], 'h':float(theta[start]),'d':float(theta[start+1]),
            's':(data.bases[league]@sf).tolist(),'v':(data.bases[league]@vf).tolist(),
            's_free':sf.tolist(),'v_free':vf.tolist()}
    payload={'kind':'past_market_strength_v1','target_kind':target_kind,'league_order':list(m.LEAGUES),
             'leagues':leagues,'theta':theta.tolist(),'penalty':.002,'floor':1e-12}
    return model,rows,data,theta,payload


@pytest.mark.parametrize('kind',['past_market','outcome_control'])
def test_independent_strength_forecast_objective_gradient_match_kernel_without_fit(kind):
    model,rows,data,theta,payload=numerical_payload(kind)
    actual=m.strength_objective_audit(payload,rows)
    value,gradient=model.strength_objective(theta,data)
    assert actual['objective']==pytest.approx(value,rel=0,abs=2e-14)
    np.testing.assert_allclose(actual['gradient'],gradient,rtol=0,atol=3e-16)
    fixtures=[{'league':league,'home':'A','away':'B'} for league in m.LEAGUES]
    expected=model.predict_strength(payload,fixtures)
    for row,p in zip(fixtures,expected):
        np.testing.assert_allclose(m.strength_predict(payload,**row),p,rtol=0,atol=2e-16)
    # Objective derivative independently agrees with perturbations, without any optimizer.
    eps=1e-6
    numerical=[]
    for k in range(len(theta)):
        a=theta.copy();b=theta.copy();a[k]+=eps;b[k]-=eps
        numerical.append((model.strength_objective(a,data)[0]-model.strength_objective(b,data)[0])/(2*eps))
    np.testing.assert_allclose(actual['gradient'],numerical,rtol=0,atol=4e-10)


@pytest.mark.parametrize('fault',['roster','free','theta','weight','target','penalty'])
def test_optimum_audit_rejects_parameter_or_admitted_history_drift(fault):
    _,rows,_,_,payload=numerical_payload('past_market')
    if fault=='roster':rows[0]['home']='future-team'
    elif fault=='free':payload['leagues']['E0']['s_free'][0]+=1
    elif fault=='theta':payload['theta'][0]+=1
    elif fault=='weight':rows[0]['weight']=0
    elif fault=='target':rows[0]['q']=[.5,.5,.5]
    else:payload['penalty']=.01
    with pytest.raises(ValueError):m.strength_objective_audit(payload,rows)


@pytest.mark.parametrize('theta',[[1.,-.5,1.],[0.,-.2,.5],[.1,-5.,1e-6],[2.,25.,.1],[2.,-25.,.1]])
def test_independent_ordered_objective_kkt_and_predictions_match_kernel(theta):
    from research.experiments.boundary_20260908.football_past_market import model
    x=np.array([-3.,-1.,0.,.2,1.,3.]); y=np.array([2,1,0,1,2,0])
    payload=readout(theta[0],theta[1],theta[1]+theta[2]);payload.update(gap=theta[2],theta=theta,floor=1e-12)
    actual=m.ordered_objective_audit(payload,x,y)
    value,gradient=model.ordered_objective(np.array(theta),x,y)
    assert actual['objective']==pytest.approx(value,rel=0,abs=2e-14)
    np.testing.assert_allclose(actual['gradient'],gradient,rtol=2e-15,atol=3e-14)
    assert actual['kkt_residual']==pytest.approx(model.readout_kkt(theta,gradient),rel=2e-15,abs=3e-14)
    expected=model.predict_ordered_logit(payload,x)
    actualp=[m.ordered_logit_predict(payload,1000.+value*400.,1000.) for value in x]
    np.testing.assert_allclose(actualp,expected,rtol=0,atol=5e-16)


@pytest.mark.parametrize('delay',[1,7])
def test_independent_membership_weights_elo_and_calendar_match_actual_data_kernel(delay):
    from research.experiments.boundary_20260908.football_past_market import data
    raws=[raw('2022-01-01',league=league,mid=league+':1') for league in m.LEAGUES]
    raws += [raw('2022-01-02',mid='E0:2',home='B',away='C'),raw('2022-01-20',mid='future',payload=Poison())]
    objects=[data.RawFixture(**row,source_path='synthetic.csv',source_sha256='a'*64,
             source_row_number=i+2,source_row_hash='b'*64) for i,row in enumerate(raws)]
    archive=data.Archive(tuple(objects));cutoff='2022-01-15T00:00:00+00:00'
    expected,_=data.training_rows(archive,cutoff,delay)
    actual=m.history_membership(objects,cutoff,extra_days=delay)
    assert actual['training_ids']==[r['match_id'] for r in expected]
    for a,b in zip(actual['rows'],expected):
        for key in ('match_id','league','season','home','away','day','outcome','weight','a0_utc','quote_available_at_utc'):
            assert a[key]==b[key]
        np.testing.assert_allclose(a['q'],b['q'],rtol=0,atol=3e-13)
        np.testing.assert_allclose(a['q_normalized'],b['q_normalized'],rtol=0,atol=2e-16)
    fixtures=[{'match_id':'query','league':'E0','home':'A','away':'C'}]
    state=data.EloCursor(archive,delay).query(fixtures,cutoff)
    independent=m.elo_replay(objects,cutoff,extra_days=delay)
    assert (independent[('E0','A')]-independent[('E0','C')])/400 == pytest.approx(state['x'][0],rel=0,abs=2e-15)
