"""Regression tests for causal timing, score probability and evaluation defects.

Synthetic rows exercise invariants only; they are not empirical football evidence.
"""
from __future__ import annotations
from dataclasses import replace, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import math
import json
import numpy as np
import pytest

from mrp.config import PredictionConfig
from mrp.data import MatchRecord, FixtureRecord, LocalFootballData, select_training_matches, match_available_at, _parse_match_row
from mrp.joint import fit_dixon_coles, dixon_coles_tau, validate_rho
from mrp.protocol import chronological_populations
from mrp.score_distribution import build_score_distribution
from mrp.prediction import run_prediction
from mrp.training import (train_gradient_boosting_benchmark, fit_probability_calibrator_from_rows,
                          build_history_from_matches, fixture_feature_vector, most_likely_scoreline,
                          _multiclass_log_loss, normalize_probabilities)


def match(i: int, *, date: datetime | None = None, home: str = 'a', away: str = 'b', goals: tuple[int,int] = (2,1)) -> MatchRecord:
    return MatchRecord(str(i),date,2024,'epl',None,home,away,*goals,None,None)


def fixture(date: datetime | None = datetime(2025,1,1)) -> FixtureRecord:
    return FixtureRecord('fixture',date,2025,'epl',1,'a','b')


def synthetic_matches(n: int, seed: int = 42) -> list[MatchRecord]:
    rng = np.random.default_rng(seed)
    result = []
    for i in range(n):
        home,away = rng.choice(8,2,replace=False)
        result.append(match(i,date=datetime(2024,1,1)+timedelta(days=i),home=str(home),away=str(away),
                            goals=(int(rng.poisson(1.5)),int(rng.poisson(1.2)))))
    return result


def write_dataset(root: Path, matches: list[MatchRecord]) -> None:
    root.mkdir(parents=True,exist_ok=True)
    fields=['match_id','date','season','league','round','home_team_id','away_team_id','home_goals','away_goals']
    with (root/'matches.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for m in matches:
            writer.writerow(dict(match_id=m.match_id,date=m.date.isoformat() if m.date else '',season=2024,league='epl',round='',
                home_team_id=m.home_team_id,away_team_id=m.away_team_id,home_goals=m.home_goals,away_goals=m.away_goals))
    with (root/'fixtures.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerow(dict(match_id='target',date='2025-07-01T12:00:00',season=2025,league='epl',round=1,
                             home_team_id='0',away_team_id='1'))


def test_cutoff_rejects_future_only_and_undated_history() -> None:
    f=fixture(datetime(2024,8,1))
    candidates=[match(0,date=datetime(2024,8,2)),match(1)]
    selected,_=select_training_matches(LocalFootballData(None,{},candidates,[f]),PredictionConfig('epl',2025,1,'match_result'),[f])
    assert selected == []
    selected,_=select_training_matches(LocalFootballData(None,{},candidates,[f]),PredictionConfig('epl',2025,1,'match_result'),[fixture(None)])
    assert selected == []


def test_results_require_actual_availability_not_just_earlier_kickoff() -> None:
    known=match(0,date=datetime(2024,1,1,10))
    delayed=replace(known,match_id='delayed',result_available_at=datetime(2024,1,3))
    same_day=fixture(datetime(2024,1,1,20))
    next_day=fixture(datetime(2024,1,2))
    config=PredictionConfig('epl',2025,1,'match_result')
    assert select_training_matches(LocalFootballData(None,{},[known],[same_day]),config,[same_day])[0] == []
    assert select_training_matches(LocalFootballData(None,{},[known,delayed],[next_day]),config,[next_day])[0] == [known]
    with pytest.raises(ValueError,match='follow kickoff'):
        match_available_at(replace(known,result_available_at=known.date))


def test_form_features_do_not_read_unavailable_results_or_backfilled_xg() -> None:
    m=replace(match(0,date=datetime(2024,1,1,10)),home_xg=100.,away_xg=0.,xg_available_at=datetime(2024,1,4))
    assert build_history_from_matches([m],as_of=datetime(2024,1,1,20)) == {}
    history=build_history_from_matches([m],as_of=datetime(2024,1,2))
    assert history['a'][0]['xg_diff'] == 1.0
    assert build_history_from_matches([m],as_of=datetime(2024,1,5))['a'][0]['xg_diff'] == 100.


def test_joint_optimizer_is_feasible_on_previously_negative_support_case() -> None:
    rows=[match(i,home=str(i%2),away=str((i+1)%2),goals=(6,6)) for i in range(10)]
    model=fit_dixon_coles(rows,[])
    assert model.diagnostics['converged'] is True
    assert model.diagnostics['kkt_stationarity_inf_norm'] <= 1e-5
    assert model.rho == pytest.approx(0.,abs=1e-10)
    for h in ('0','1','unseen'):
        for a in ('0','1','other_unseen'):
            lh,la=model.expected_goals(h,a)
            for gh,ga in ((0,0),(0,1),(1,0),(1,1)):
                assert dixon_coles_tau(gh,ga,lh,la,model.rho) >= 0


def test_joint_solver_reports_verified_convergence_and_improves_initial_objective() -> None:
    model=fit_dixon_coles(synthetic_matches(180),[])
    d=model.diagnostics
    assert d['objective'] == 'joint_dixon_coles_penalized_negative_log_likelihood'
    assert d['kkt_stationarity_inf_norm'] <= d['kkt_tolerance']
    assert d['constraint_violation_max'] < 1e-7
    assert d['negative_log_likelihood_per_weight'] < d['initial_negative_log_likelihood_per_weight'] - .01


def test_invalid_rho_is_rejected_instead_of_clipping_negative_mass() -> None:
    with pytest.raises(ValueError,match='support'):
        build_score_distribution(3,2,.2)
    with pytest.raises(ValueError,match='support'):
        validate_rho(6,6,-.2)


@pytest.mark.parametrize('rates',[(1.35,1.05),(3.,2.),(6.,6.)])
def test_scoreline_probability_matches_unconditional_poisson_to_tail_bound(rates: tuple[float,float]) -> None:
    h,a,p=most_likely_scoreline(*rates,0.)
    expected=math.exp(-sum(rates))*rates[0]**h/math.factorial(h)*rates[1]**a/math.factorial(a)
    assert p == pytest.approx(expected,rel=2e-12)
    joint=build_score_distribution(*rates,0.)
    assert joint.omitted_probability_mass <= 1e-12
    assert sum(joint.outcome_probabilities) == pytest.approx(1.,abs=1e-12)
    assert joint.expected_goals == pytest.approx(rates,abs=1e-10)


def test_selected_one_x_two_expected_goals_and_scoreline_share_one_distribution() -> None:
    joint=build_score_distribution(3,2,-.1).reconcile((.1,.8,.1))
    assert joint.outcome_probabilities == pytest.approx((.1,.8,.1),abs=1e-12)
    h,a,p=joint.most_likely_scoreline
    assert p == joint.matrix[h][a]
    assert joint.expected_goals[0] == pytest.approx(sum(h*p for h,row in enumerate(joint.matrix) for p in row))


def test_gbdt_holdout_scales_with_data_and_can_be_calibrated() -> None:
    rows=synthetic_matches(380)
    model=train_gradient_boosting_benchmark(rows,[])
    assert model is not None
    assert len(model.validation_labels) == 76
    notes=[]
    calibrator=fit_probability_calibrator_from_rows(model.validation_probabilities,model.validation_labels,notes,policy='isotonic')
    assert calibrator.method == 'isotonic'
    assert not any('echantillon insuffisant' in note for note in notes)


def test_populations_are_disjoint_in_time_and_content_fingerprinted() -> None:
    rows=synthetic_matches(200)
    p=chronological_populations(rows)
    assert [len(getattr(p,k)) for k in ('fit','calibration','selection','test')] == [100,30,30,40]
    groups=[p.fit,p.calibration,p.selection,p.test]
    assert sum(map(len,groups)) == len({m.match_id for group in groups for m in group})
    for left,right in zip(groups,groups[1:]):
        assert max(match_available_at(m) for m in left) <= min(m.date for m in right)
    modified=list(rows);modified[-1]=replace(modified[-1],home_goals=8)
    assert chronological_populations(modified).metadata()['test']['data_sha256'] != p.metadata()['test']['data_sha256']
    assert chronological_populations(modified).metadata()['fit']['data_sha256'] == p.metadata()['fit']['data_sha256']


def test_whole_matchdays_and_late_results_do_not_cross_stage_boundaries() -> None:
    rows=[replace(m,date=datetime(2024,1,1)+timedelta(days=i//4)) for i,m in enumerate(synthetic_matches(240))]
    p=chronological_populations(rows)
    groups=[p.fit,p.calibration,p.selection,p.test]
    for left,right in zip(groups,groups[1:]):
        assert max(m.date.date() for m in left) < min(m.date.date() for m in right)
    late=replace(p.fit[-1],result_available_at=p.calibration[1].date+timedelta(days=2))
    rows=[late if m.match_id==late.match_id else m for m in rows]
    adjusted=chronological_populations(rows)
    assert late.match_id in adjusted.excluded_ids


def test_full_pipeline_keeps_outer_targets_out_of_model_and_policy_selection(tmp_path: Path) -> None:
    rows=synthetic_matches(200)
    write_dataset(tmp_path/'before',rows)
    config=PredictionConfig('epl',2025,1,'match_result',data_source=str(tmp_path/'before'),football_model='hybrid')
    before=run_prediction(config)
    changed=list(rows);changed[-1]=replace(changed[-1],home_goals=20,away_goals=0)
    write_dataset(tmp_path/'after',changed)
    after=run_prediction(replace(config,data_source=str(tmp_path/'after')))
    assert before.diagnostics['goal_model_fit'] == after.diagnostics['goal_model_fit']
    assert before.diagnostics['hybrid_weight_w'] == after.diagnostics['hybrid_weight_w']
    assert before.diagnostics['evaluation_rows'][0]['selected_probabilities'] == after.diagnostics['evaluation_rows'][0]['selected_probabilities']
    hashes={m['population_sha256'] for m in before.diagnostics['models'].values()}
    assert len(hashes)==1
    assert all(m['sample_size']==40 for m in before.diagnostics['models'].values())
    matrix=before.diagnostics['fixture_distributions']['target']['score_probability_matrix']
    row=before.rows[0]
    sums=[sum(p for h,r in enumerate(matrix) for a,p in enumerate(r) if predicate(h,a))
          for predicate in (lambda h,a:h>a,lambda h,a:h==a,lambda h,a:h<a)]
    assert [float(row[k]) for k in ('home_win_prob','draw_prob','away_win_prob')] == pytest.approx(sums,abs=1e-11)
    assert float(row['expected_home_goals']) == pytest.approx(sum(h*p for h,r in enumerate(matrix) for p in r),abs=1e-6)
    assert before.diagnostics['predictive_edge_established'] is False
    json.dumps(asdict(before),allow_nan=False)


def test_insufficient_history_never_emits_resubstitution_as_validation(tmp_path: Path) -> None:
    write_dataset(tmp_path,synthetic_matches(40))
    result=run_prediction(PredictionConfig('epl',2025,1,'match_result',data_source=str(tmp_path),football_model='hybrid'))
    assert result.diagnostics['models'] == {}
    assert result.diagnostics['protocol']['metrics_status'] == 'insufficient_history'
    assert result.diagnostics['model_used']=='dixon'


def test_undated_history_produces_explicit_prior_only_status(tmp_path: Path) -> None:
    write_dataset(tmp_path,[replace(m,date=None) for m in synthetic_matches(40)])
    result=run_prediction(PredictionConfig('epl',2025,1,'scoreline',data_source=str(tmp_path)))
    assert result.rows[0]['forecast_status']=='prior_only'
    assert result.diagnostics['goal_model_fit']['fit_sample_size']==0
    assert result.diagnostics['models']=={}


def test_wrong_requested_round_produces_no_unrelated_forecasts(tmp_path: Path) -> None:
    write_dataset(tmp_path,synthetic_matches(40))
    result=run_prediction(PredictionConfig('epl',2025,99,'scoreline',data_source=str(tmp_path)))
    assert result.rows==[]
    assert result.diagnostics['status']=='unavailable'


def test_temporal_weighting_is_explicit_and_requires_dates() -> None:
    rows=synthetic_matches(100)
    model=fit_dixon_coles(rows,[],half_life_days=60,reference_time=datetime(2024,5,1))
    assert model.diagnostics['half_life_days']==60
    assert model.diagnostics['effective_sample_size'] < len(rows)
    with pytest.raises(ValueError,match='requires dates'):
        fit_dixon_coles([match(0)],[],half_life_days=60)


@pytest.mark.parametrize('score',['1.5','-1','inf'])
def test_malformed_final_scores_are_rejected(score: str) -> None:
    with pytest.raises(ValueError,match='nonnegative integers'):
        _parse_match_row({'home':'a','away':'b','home_goals':score,'away_goals':'0'},'m',0)


def test_probability_inputs_and_alignment_fail_closed() -> None:
    with pytest.raises(ValueError,match='equal lengths'):
        _multiclass_log_loss([0,1],[[.5,.3,.2]])
    with pytest.raises(ValueError,match='finite'):
        normalize_probabilities((float('nan'),.5,.5))


def test_tail_bound_remains_small_when_an_improbable_region_is_upweighted() -> None:
    joint=build_score_distribution(.05,6.,0.).reconcile((.95,.025,.025))
    assert joint.tail_probability_bound <= 1e-12
    assert joint.outcome_probabilities == pytest.approx((.95,.025,.025),abs=1e-12)


def test_calibration_cutoff_compares_actual_instants_across_offsets() -> None:
    from mrp.joint import DixonColesModel
    from mrp.training import fit_probability_calibrator_with_policy
    model=DixonColesModel({}, {}, math.log(1.5),math.log(1.2),0.,
        {"fit_match_ids":["fit"],"latest_fit_result_available_at":"2024-01-02T10:00:00+02:00"})
    after=[match(i,date=datetime(2024,1,2,8,30,tzinfo=timezone.utc)) for i in range(30)]
    fit_probability_calibrator_with_policy(after,model,[],policy='auto')
    before=[replace(m,date=datetime(2024,1,2,9,tzinfo=timezone(timedelta(hours=2)))) for m in after]
    with pytest.raises(ValueError,match='follow availability'):
        fit_probability_calibrator_with_policy(before,model,[],policy='auto')
    assert before[0].date == datetime(2024,1,2,7)
