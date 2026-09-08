"""Synthetic state, chronology and complete-distribution contracts."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
import math

import numpy as np
import pytest

from packages.football.mrp import deployment as d, training
from packages.football.mrp.data import MatchRecord, FixtureRecord
from packages.football.mrp.joint import fit_dixon_coles


def synthetic_history(n=240):
    rng = np.random.default_rng(514)
    rows = []
    for i in range(n):
        home, away = rng.choice(6,2,replace=False)
        date = datetime(2021,1,1) + timedelta(days=i*2)
        rows.append(MatchRecord(f"h{i:04}",date,2021,"epl",None,f"t{home}",f"t{away}",
            int(rng.poisson(1.5)),int(rng.poisson(1.1)),None,None))
    return rows


@pytest.fixture(scope="module")
def fitted():
    rows=synthetic_history(); cutoff=rows[-1].date+timedelta(days=3)
    fixtures=[FixtureRecord("fixture-a",cutoff,2022,"epl",None,"t0","t1"),
              FixtureRecord("fixture-b",cutoff+timedelta(days=1),2022,"epl",None,"unseen","t2")]
    model=fit_dixon_coles(rows,[])
    state=d.serialize_dc(model)
    lineage=d.validate_inputs(rows,fixtures,cutoff)
    lineage["goal_state_sha256"]=d.canonical_sha256({k:v for k,v in state.items() if k!="schema"})
    return rows,fixtures,cutoff,state,lineage


def test_restored_full_history_requires_no_fit_and_preserves_input(fitted,monkeypatch):
    rows,fixtures,cutoff,state,lineage=fitted
    before=d.canonical_sha256([state,lineage,[asdict(r) for r in rows]])
    monkeypatch.setattr(d,"fit_dixon_coles",lambda *a,**kw:pytest.fail("restoration fitted a model"))
    result=d.full_history_forecast(rows,fixtures,cutoff=cutoff,restored_model=state,restored_lineage=lineage)
    assert result["new_goal_model_fits"]==0 and result["assessment_metrics_computed"] is False
    assert result["lineage"]["fit_match_ids"]==[m.match_id for m in rows]
    assert d.canonical_sha256([state,lineage,[asdict(r) for r in rows]])==before
    json.dumps(result,allow_nan=False)
    for output in result["fixtures"]:
        matrix=np.asarray(output["matrix"])
        assert output["dimensions"]==list(matrix.shape)
        assert np.all(matrix>=0) and abs(matrix.sum()-1)<1e-12
        assert np.allclose(output["hda"],[np.tril(matrix,-1).sum(),np.trace(matrix),np.triu(matrix,1).sum()],atol=1e-14)
        assert np.allclose(output["expected_goals"],[(np.arange(matrix.shape[0])[:,None]*matrix).sum(),(np.arange(matrix.shape[1])[None,:]*matrix).sum()],atol=1e-13)
        assert output["tail_probability_bound"]<=1e-12


def test_fresh_branch_fits_exact_admitted_population_once(fitted,monkeypatch):
    rows,fixtures,cutoff,state,_=fitted; calls=[]
    def fit(matches,notes,**kwargs):
        calls.append(([m.match_id for m in matches],kwargs));return d.restore_dc(state)
    monkeypatch.setattr(d,"fit_dixon_coles",fit)
    result=d.full_history_forecast(rows,fixtures,cutoff=cutoff)
    assert calls==[([m.match_id for m in rows],{"half_life_days":None,"reference_time":cutoff})]
    assert result["new_goal_model_fits"]==1


@pytest.mark.parametrize("bad_status",[False,1,None,"true"])
def test_unsuccessful_fresh_fit_cannot_emit_forecasts(fitted,monkeypatch,bad_status):
    rows,fixtures,cutoff,state,_=fitted; model=d.restore_dc(state)
    model.diagnostics["converged"]=bad_status
    monkeypatch.setattr(d,"fit_dixon_coles",lambda *a,**kw:model)
    monkeypatch.setattr(d,"joint_output",lambda *a,**kw:pytest.fail("failed fit issued a forecast"))
    with pytest.raises(RuntimeError,match="successful convergence"):
        d.full_history_forecast(rows,fixtures,cutoff=cutoff)


def test_empty_history_returns_explicit_canonical_prior(fitted):
    _,fixtures,cutoff,_,_=fitted
    result=d.full_history_forecast([],fixtures,cutoff=cutoff)
    assert result["model_state"]["diagnostics"]=={"status":"prior_only","converged":None,"fit_sample_size":0}
    assert result["lineage"]["fit_match_ids"]==[]
    assert all(o["lambda_home"]==pytest.approx(1.35) and o["lambda_away"]==pytest.approx(1.05) for o in result["fixtures"])


@pytest.mark.parametrize("field",["cutoff_utc","fit_match_ids","fit_ids_sha256","match_records_sha256","latest_result_available_at","goal_state_sha256"])
def test_rejects_misbound_restoration_before_fit(fitted,field,monkeypatch):
    rows,fixtures,cutoff,state,lineage=fitted;bad=deepcopy(lineage);bad[field]="wrong"
    monkeypatch.setattr(d,"fit_dixon_coles",lambda *a,**kw:pytest.fail("invalid restoration changed branch"))
    with pytest.raises(ValueError,match="Restored"):
        d.full_history_forecast(rows,fixtures,cutoff=cutoff,restored_model=state,restored_lineage=bad)


@pytest.mark.parametrize("edit",["future_date","equal_date","unavailable","duplicate","reversed","goals_bool","goals_negative","self_match"])
def test_history_admission_is_strict(fitted,edit):
    rows,fixtures,cutoff,_,_=fitted; rows=list(rows)
    if edit=="future_date":rows[-1]=replace(rows[-1],date=cutoff+timedelta(days=1),home_goals=object())
    elif edit=="equal_date":rows[-1]=replace(rows[-1],date=cutoff,home_goals=object())
    elif edit=="unavailable":rows[-1]=replace(rows[-1],result_available_at=cutoff+timedelta(microseconds=1),home_goals=object())
    elif edit=="duplicate":rows.append(rows[-1])
    elif edit=="reversed":rows.reverse()
    elif edit=="goals_bool":rows[-1]=replace(rows[-1],home_goals=True)
    elif edit=="goals_negative":rows[-1]=replace(rows[-1],away_goals=-1)
    elif edit=="self_match":rows[-1]=replace(rows[-1],away_team_id=rows[-1].home_team_id)
    with pytest.raises(ValueError):d.validate_inputs(rows,fixtures,cutoff)


def test_current_fixture_values_never_enter_output_or_support(fitted):
    rows,fixtures,cutoff,state,lineage=fitted
    poisoned=[MatchRecord(f.match_id,f.date,f.season,f.league,f.round_number,f.home_team_id,f.away_team_id,
        object(),object(),object(),object()) for f in fixtures]
    a=d.full_history_forecast(rows,fixtures,cutoff=cutoff,restored_model=state,restored_lineage=lineage)
    b=d.full_history_forecast(rows,poisoned,cutoff=cutoff,restored_model=state,restored_lineage=lineage)
    assert a==b


def test_availability_equality_and_utc_offset_preserve_contract(fitted):
    rows,fixtures,cutoff,_,_=fitted
    modified=[*rows[:-1],replace(rows[-1],result_available_at=cutoff)]
    a=d.validate_inputs(modified,fixtures,cutoff)
    b=d.validate_inputs(modified,fixtures,cutoff.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=2))))
    assert a==b


@pytest.mark.parametrize("policy",["off","platt","isotonic"])
def test_native_calibrator_serialization_exact_without_refit(policy,monkeypatch):
    rng=np.random.default_rng(51); p=rng.dirichlet([2.,2.,2.],300); labels=[int(rng.choice(3,p=x)) for x in p]
    calibrator=training.fit_probability_calibrator_from_rows(p.tolist(),labels,[],policy=policy)
    assert calibrator.method=={"off":"identity","platt":"platt","isotonic":"isotonic"}[policy]
    state=json.loads(json.dumps(d.serialize_calibrator(calibrator),allow_nan=False))
    monkeypatch.setattr(training.LogisticRegression,"fit",lambda *a,**kw:pytest.fail("restore fits Platt"))
    monkeypatch.setattr(training.IsotonicRegression,"fit",lambda *a,**kw:pytest.fail("restore fits isotonic"))
    restored=d.restore_calibrator(state)
    for value in [*p.tolist(),[0,0,1],[1,0,0],[1e-300,.1,.9],[1,1,1]]:
        assert restored.apply(tuple(value))==calibrator.apply(tuple(value))
    assert d.serialize_calibrator(restored)==state


def test_classwise_identity_fallback_is_serialized():
    rng=np.random.default_rng(15);p=rng.dirichlet([2,2,2],100)
    c=training.fit_probability_calibrator_from_rows(p.tolist(),[i%2 for i in range(100)],[],policy="platt")
    s=d.serialize_calibrator(c)
    assert [item["kind"] for item in s["class_functions"]]==["platt","platt","identity"]
    assert d.restore_calibrator(s).apply((.4,.3,.3))==c.apply((.4,.3,.3))


def test_unrecognized_calibration_function_fails_closed():
    c=training.ProbabilityCalibrator("identity",[lambda v:v]*3)
    with pytest.raises(ValueError,match="Unrecognized"):d.serialize_calibrator(c)


@pytest.mark.parametrize("value",[float("nan"),float("inf"),True])
def test_restored_coefficients_cannot_be_nonfinite_or_bool(fitted,value):
    state=deepcopy(fitted[3]);state["rho"]=value
    with pytest.raises(ValueError):d.restore_dc(state)


def test_saved_joint_kernel_is_not_inferred_from_hda(fitted):
    model=d.restore_dc(fitted[3]); fixture=fitted[1][0]
    identity=training.ProbabilityCalibrator("identity",[training._identity]*3)
    a=d.joint_output(model,fixture,identity)
    changed=deepcopy(model);changed.home_intercept+=.01
    b=d.joint_output(changed,fixture,identity)
    assert a["matrix"]!=b["matrix"] and a["lambda_home"]!=b["lambda_home"]
    assert "goals" not in a and "target" not in a
