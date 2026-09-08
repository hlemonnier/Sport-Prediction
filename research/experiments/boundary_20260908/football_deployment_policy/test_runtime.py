"""Synthetic unchanged-caller differential and fit-budget tests."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from packages.football.mrp import deployment as d, prediction
from packages.football.mrp.protocol import chronological_populations
from packages.football.tests.test_deployment import fitted, synthetic_history
from . import runtime


def test_actual_caller_budget_complete_outputs_and_restoration(fitted,monkeypatch):
    rows,fixtures,cutoff,state,lineage=fitted
    monkeypatch.setattr(d,"fit_dixon_coles",lambda *a,**kw:pytest.fail("saved full model refitted"))
    expected=chronological_populations(rows).metadata()
    result=runtime.execute_block(rows,fixtures,state,cutoff,expected,saved_full_lineage=lineage)
    assert result["fit_counts"]=={"prefix_goal_fits":1,"full_history_goal_fits":0,
        "reference_calibration_policy_calls":1,"candidate_learned_calibration_fits":0,"candidate_parity_goal_fits":0}
    assert result["reference"]["diagnostics"]["protocol"]["metrics_status"]=="disabled"
    assert result["reference"]["lineage"]["goal_fit_ids"]==expected["fit"]["match_ids"]
    assert result["candidate"]["lineage"]["fit_match_ids"]==[m.match_id for m in rows]
    assert len(result["reference"]["fixtures"])==len(result["candidate"]["fixtures"])==len(fixtures)
    assert result["parity"]["reference_actual_caller"] is True


def test_preserved_assessment_entrypoint_ignores_future_default_policy(fitted,monkeypatch):
    rows,fixtures,cutoff,state,lineage=fitted
    old_caller=getattr(prediction,"run_assessment_prediction",prediction.run_prediction)
    monkeypatch.delattr(prediction,"run_assessment_prediction",raising=False)
    monkeypatch.setattr(prediction,"run_prediction",old_caller)
    expected=chronological_populations(rows).metadata()
    before=runtime.execute_block(rows,fixtures,state,cutoff,expected,saved_full_lineage=lineage)
    assert before["reference"]["canonical_caller_entrypoint"].endswith(".run_prediction")
    def new_default(config):
        pytest.fail("Frozen reference reached the future full-history default")
    monkeypatch.setattr(prediction,"run_assessment_prediction",old_caller,raising=False)
    monkeypatch.setattr(prediction,"run_prediction",new_default)
    after=runtime.execute_block(rows,fixtures,state,cutoff,expected,saved_full_lineage=lineage)
    assert before["fit_counts"]==after["fit_counts"]
    assert after["fit_counts"]["prefix_goal_fits"]==1
    assert after["fit_counts"]["candidate_parity_goal_fits"]==0
    for policy in ("reference","candidate"):
        assert after[policy]["canonical_caller_entrypoint"].endswith(".run_assessment_prediction")
        for field in ("fixtures","model_state","calibrator_state","canonical_rows","diagnostics","lineage"):
            assert before[policy][field]==after[policy][field]


def test_invalid_explicit_assessment_entrypoint_never_falls_through(fitted,monkeypatch):
    rows,fixtures,_,_,_=fitted
    monkeypatch.setattr(prediction,"run_assessment_prediction",None,raising=False)
    monkeypatch.setattr(prediction,"run_prediction",lambda cfg:pytest.fail("Invalid reference silently reached default"))
    with pytest.raises(TypeError,match="assessment entrypoint"):
        runtime._run_caller(rows,fixtures,expected_partitions=chronological_populations(rows).metadata(),calibration_policy="auto")


def test_gbdt_suppression_and_shadow_metrics_leave_selected_dixon_unchanged(fitted):
    rows,fixtures,cutoff,state,lineage=fitted; expected=chronological_populations(rows).metadata()
    original_result,captures=runtime._run_caller(rows,fixtures,expected_partitions=expected,
        calibration_policy="auto",shadow_eval=True,suppress_unused_gbdt=False)
    # This is the real canonical GBDT path, not a stub returning None.
    assert "gbdt_raw" in original_result.diagnostics["models"]
    suppressed,suppressed_captures=runtime._run_caller(rows,fixtures,expected_partitions=expected,
        calibration_policy="auto",shadow_eval=True,suppress_unused_gbdt=True,
        model_override=captures["model"])
    assert original_result.rows==suppressed.rows
    assert original_result.diagnostics["fixture_distributions"]==suppressed.diagnostics["fixture_distributions"]
    for key in ("selected_model","dixon_coles_raw","dixon_coles_calibrated","baseline_frequency"):
        assert original_result.diagnostics["models"][key]==suppressed.diagnostics["models"][key]
    before=deepcopy(suppressed.diagnostics)
    candidate=d.full_history_forecast(rows,fixtures,cutoff=cutoff,restored_model=state,restored_lineage=lineage)
    assert suppressed.diagnostics==before
    assert {r["match_id"] for r in suppressed.diagnostics["evaluation_rows"]}==set(expected["test"]["match_ids"])
    assert set(expected["test"]["match_ids"]).isdisjoint(captures["model"].diagnostics["fit_match_ids"])
    assert candidate["lineage"]["fit_match_ids"]!=captures["model"].diagnostics["fit_match_ids"]
    assert suppressed_captures["goal_kernel_fits"]==0


def test_no_partition_rewrite_or_changed_cutoff_before_any_fit(fitted,monkeypatch):
    rows,fixtures,cutoff,state,lineage=fitted;part=chronological_populations(rows).metadata()
    monkeypatch.setattr(prediction,"fit_dixon_coles",lambda *a,**kw:pytest.fail("invalid block fitted"))
    with pytest.raises(ValueError,match="first-fixture"):
        runtime.execute_block(rows,fixtures,state,cutoff-timedelta(hours=1),part,saved_full_lineage=lineage)
    part=deepcopy(part);part["fit"]["match_ids"].reverse()
    with pytest.raises(ValueError,match="partitions"):
        runtime.execute_block(rows,fixtures,state,cutoff,part,saved_full_lineage=lineage)


def test_failed_fit_restores_every_canonical_seam_and_lock(fitted,monkeypatch):
    rows,fixtures,_,_,_=fitted;expected=chronological_populations(rows).metadata()
    def failure(*args,**kwargs):raise RuntimeError("synthetic solver failure")
    monkeypatch.setattr(prediction,"fit_dixon_coles",failure)
    names=("load_local_football_data","select_target_fixtures","select_training_matches",
        "fit_dixon_coles","fit_probability_calibrator_with_policy","fit_probability_calibrator_from_rows","train_gradient_boosting_model")
    before={name:getattr(prediction,name) for name in names}
    with pytest.raises(RuntimeError,match="synthetic solver failure"):
        runtime._run_caller(rows,fixtures,expected_partitions=expected,calibration_policy="auto")
    assert before=={name:getattr(prediction,name) for name in names}
    assert not runtime._CALLER_LOCK.locked()


def test_reference_nonconvergence_aborts_before_calibration(fitted,monkeypatch):
    rows,fixtures,_,state,_=fitted;expected=chronological_populations(rows).metadata()
    failed=d.restore_dc(state)
    failed.diagnostics.update(fit_sample_size=expected["fit"]["sample_size"],converged=False)
    monkeypatch.setattr(prediction,"fit_dixon_coles",lambda *a,**kw:failed)
    monkeypatch.setattr(prediction,"fit_probability_calibrator_with_policy",lambda *a,**kw:pytest.fail("failed reference reached calibration"))
    with pytest.raises(RuntimeError,match="successful convergence"):
        runtime._run_caller(rows,fixtures,expected_partitions=expected,calibration_policy="auto")
    assert not runtime._CALLER_LOCK.locked()


@pytest.mark.parametrize("flag",["false",0,1,None])
def test_caller_boolean_flags_are_not_truthy_strings(fitted,flag):
    rows,fixtures,_,_,_=fitted
    with pytest.raises(TypeError):runtime._run_caller(rows,fixtures,expected_partitions=chronological_populations(rows).metadata(),
        calibration_policy="auto",shadow_eval=flag)


def test_exact_old_vector_parity_is_checked_not_substituted(fitted):
    rows,fixtures,cutoff,state,lineage=fitted;part=chronological_populations(rows).metadata()
    with pytest.raises(ValueError,match="old dc_equal HDA"):
        runtime.execute_block(rows,fixtures,state,cutoff,part,saved_full_lineage=lineage,
            expected_candidate_probabilities=[[.1,.2,.7]]*len(fixtures))


def test_actual_io_filters_are_validated_not_bypassed(fitted):
    rows,fixtures,_,_,_=fitted;part=chronological_populations(rows).metadata()
    altered=[*fixtures[:-1],replace(fixtures[-1],season=2023)]
    with pytest.raises(ValueError,match="canonical league/season/round"):
        runtime._run_caller(rows,altered,expected_partitions=part,calibration_policy="auto")


def test_replayed_complete_hda_matches_source_matrix(fitted):
    rows,fixtures,_,_,_=fitted
    result,capture=runtime._run_caller(rows,fixtures,expected_partitions=chronological_populations(rows).metadata(),calibration_policy="auto")
    states=[d.joint_output(capture["model"],f,capture["calibrator"]) for f in fixtures]
    runtime._check_caller_outputs(states,result)
    corrupted=deepcopy(states);corrupted[0]["matrix"][0][0]+=.00001
    with pytest.raises(ValueError,match="score_probability_matrix"):
        runtime._check_caller_outputs(corrupted,result)
