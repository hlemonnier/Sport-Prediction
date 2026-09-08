"""Synthetic integration proof for the full-history automatic Dixon default.

These tests exercise the production entrypoint and actual legacy CLI parser.
The API routing test executes its existing command builder, without starting a
server. No provider data or historical research outcomes are loaded.

Suggested commit: test(football): verify full-history default integration.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from packages.football.mrp import deployment, prediction
from packages.football.mrp.config import PredictionConfig
from packages.football.mrp.data import FixtureRecord, LocalFootballData, MatchRecord
from packages.football.mrp.protocol import chronological_populations
from packages.football.tests.test_deployment import synthetic_history

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "research/projects/Football/Match Result Prediction/Python/run_experiment.py"


@pytest.fixture(scope="module")
def population():
    history = synthetic_history()
    cutoff = history[-1].date + timedelta(days=3)
    fixtures = [FixtureRecord("new-fixture-a",cutoff,2022,"epl",None,"t0","t1"),
                FixtureRecord("new-fixture-b",cutoff+timedelta(days=1),2022,"epl",None,"t2","t3")]
    return history, fixtures, cutoff


def install_dataset(monkeypatch, population, *, module=prediction, scored_fixtures=False):
    history, fixtures, cutoff = population
    if scored_fixtures:
        fixtures = [MatchRecord(f.match_id,f.date,f.season,f.league,f.round_number,
            f.home_team_id,f.away_team_id,object(),object(),object(),object()) for f in fixtures]
    dataset = LocalFootballData(None,{},list(history),list(fixtures))
    monkeypatch.setattr(module,"load_local_football_data",lambda config:(dataset,[]))
    return dataset


def config(**changes):
    return PredictionConfig(league="epl",season=2022,round_number=1,mode="match_result",**changes)


def assert_helper_outputs(actual, expected):
    assert [r["match_id"] for r in actual.rows]==[r["match_id"] for r in expected["fixtures"]]
    for row, state in zip(actual.rows,expected["fixtures"],strict=True):
        emitted=actual.diagnostics["fixture_distributions"][row["match_id"]]
        np.testing.assert_allclose(emitted["score_probability_matrix"],state["matrix"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(emitted["outcome_probabilities"],state["hda"],atol=1e-10,rtol=0)
        np.testing.assert_allclose(emitted["expected_goals"],state["expected_goals"],atol=1e-10,rtol=0)
        np.testing.assert_allclose([float(row[k]) for k in ("home_win_prob","draw_prob","away_win_prob")],state["hda"],atol=6e-13,rtol=0)
        assert row["predicted_scoreline"]==f"{state['mode'][0]}-{state['mode'][1]}"
        assert float(row["scoreline_prob"])==pytest.approx(state["mode"][2],abs=6e-13,rel=0)
        assert row["calibration_method_effective"]=="dixon:identity"


def test_automatic_default_uses_full_history_and_complete_joint_outputs(population,monkeypatch):
    history, fixtures, cutoff=population
    install_dataset(monkeypatch,population)
    result=prediction.run_prediction(config(shadow_eval=False))
    expected=deployment.full_history_forecast(history,fixtures,cutoff=cutoff)
    assert_helper_outputs(result,expected)
    assert result.diagnostics["fixture_policy"]=="dc_full_admitted_equal_off"
    assert result.diagnostics["fixture_calibration_method_effective"]=="dixon:identity"
    assert result.diagnostics["forecast_training_sample_size"]==len(history)
    assert result.diagnostics["fixture_model"]["lineage"]["fit_match_ids"]==[r.match_id for r in history]
    assert result.diagnostics["fixture_model"]["calibrator_state"]["method"]=="identity"
    assert result.diagnostics["training_sample_size"]==len(chronological_populations(history).fit)


def test_assessment_metrics_and_heldout_ids_remain_on_original_prefix(population,monkeypatch):
    install_dataset(monkeypatch,population)
    reference=prediction.run_assessment_prediction(config())
    actual=prediction.run_prediction(config())
    assert actual.diagnostics["models"]==reference.diagnostics["models"]
    assert actual.diagnostics["evaluation_rows"]==reference.diagnostics["evaluation_rows"]
    assessment=actual.diagnostics["assessment_model"]
    assert assessment["protocol"]==reference.diagnostics["protocol"]
    assert assessment["goal_model_fit"]==reference.diagnostics["goal_model_fit"]
    assert assessment["calibration_method_effective"]==reference.diagnostics["calibration_method_effective"]
    evaluation_ids={r["match_id"] for r in actual.diagnostics["evaluation_rows"]}
    prefix_ids=set(assessment["goal_model_fit"]["fit_match_ids"])
    full_ids=set(actual.diagnostics["fixture_model"]["lineage"]["fit_match_ids"])
    assert evaluation_ids and evaluation_ids.isdisjoint(prefix_ids)
    assert evaluation_ids<=full_ids and full_ids>prefix_ids
    assert actual.diagnostics["protocol"]["parameter_policy"]!=reference.diagnostics["protocol"]["parameter_policy"]
    for key,value in reference.diagnostics["protocol"].items():
        if key!="parameter_policy":
            assert actual.diagnostics["protocol"][key]==value


@pytest.mark.parametrize("history_league,fixture_league",[
    (None,"epl"),
    ("epl",None),
    ("EPL","ePl"),
    (None,None),
])
def test_selected_league_metadata_is_normalized_on_fixture_copies_only(
        population,monkeypatch,history_league,fixture_league):
    history,fixtures,cutoff=population
    history=[replace(row,league=history_league) for row in history]
    fixtures=[replace(row,league=fixture_league) for row in fixtures]
    before=deepcopy(([asdict(row) for row in history],[asdict(row) for row in fixtures]))
    install_dataset(monkeypatch,(history,fixtures,cutoff))
    reference=prediction.run_assessment_prediction(config())
    actual=prediction.run_prediction(config())
    normalized_history=[replace(row,league="epl") for row in history]
    normalized_fixtures=[replace(row,league="epl") for row in fixtures]
    expected=deployment.full_history_forecast(normalized_history,normalized_fixtures,cutoff=cutoff)
    assert_helper_outputs(actual,expected)
    assert actual.diagnostics["fixture_model"]["lineage"]==expected["lineage"]
    assert actual.diagnostics["assessment_model"]["protocol"]==reference.diagnostics["protocol"]
    assert actual.diagnostics["assessment_model"]["goal_model_fit"]==reference.diagnostics["goal_model_fit"]
    assert actual.diagnostics["models"]==reference.diagnostics["models"]
    assert actual.diagnostics["evaluation_rows"]==reference.diagnostics["evaluation_rows"]
    assert before==([asdict(row) for row in history],[asdict(row) for row in fixtures])
    for original,normalized in zip(history,normalized_history,strict=True):
        assert {k:v for k,v in asdict(original).items() if k!="league"}=={k:v for k,v in asdict(normalized).items() if k!="league"}


@pytest.mark.parametrize("options",[
    {"football_calibration":"off"},
    {"football_calibration":"platt"},
    {"football_calibration":"isotonic"},
    {"goal_strength_half_life_days":365.0},
    {"football_model":"gbdt"},
    {"football_model":"hybrid"},
])
def test_explicit_nondefault_policies_preserve_original_outputs(population,monkeypatch,options):
    install_dataset(monkeypatch,population)
    cfg=config(**options)
    expected=prediction.run_assessment_prediction(cfg)
    actual=prediction.run_prediction(cfg)
    assert actual==expected
    if options.get("football_model") in {"gbdt","hybrid"}:
        assert actual.diagnostics["model_used"]==options["football_model"]
        assert actual.diagnostics["gbdt_enabled"] is True


def test_future_fixture_values_do_not_change_default_forecasts(population,monkeypatch):
    install_dataset(monkeypatch,population)
    before=prediction.run_prediction(config(shadow_eval=False))
    install_dataset(monkeypatch,population,scored_fixtures=True)
    after=prediction.run_prediction(config(shadow_eval=False))
    assert before==after


def test_future_unavailable_results_cannot_enter_full_fixture_model(population,monkeypatch):
    history, fixtures, cutoff=population
    install_dataset(monkeypatch,population)
    before=prediction.run_prediction(config(shadow_eval=False))
    future=replace(history[-1],match_id="future-unavailable",date=cutoff+timedelta(days=5),home_goals=object(),away_goals=object())
    # The canonical selector may test whether scores exist; it must not interpret
    # those numeric values before excluding their future clock.
    dataset=install_dataset(monkeypatch,([*history,future],fixtures,cutoff))
    after=prediction.run_prediction(config(shadow_eval=False))
    assert before.rows==after.rows
    assert before.diagnostics==after.diagnostics
    assert dataset.matches[-1] is future
    assert future.match_id not in after.diagnostics["fixture_model"]["lineage"]["fit_match_ids"]


def load_cli(monkeypatch):
    monkeypatch.syspath_prepend(str(CLI.parent))
    spec=importlib.util.spec_from_file_location("_football_default_deployment_cli",CLI)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual_module=sys.modules[module.run_prediction.__module__]
    assert Path(actual_module.__file__).resolve()==ROOT/"packages/football/mrp/prediction.py"
    return module,actual_module


def execute_cli(monkeypatch,population,tmp_path,argv):
    cli,actual_module=load_cli(monkeypatch)
    install_dataset(monkeypatch,population,module=actual_module)
    output=tmp_path/"prediction.json"
    cli.main([*argv,"--output-format","json","--output-path",str(output),"--quiet"])
    payload=json.loads(output.read_text())
    assert payload["config"]["football_model"]=="dixon"
    assert payload["config"]["football_calibration"]=="auto"
    assert payload["config"]["goal_strength_half_life_days"] is None
    assert payload["diagnostics"]["fixture_policy"]=="dc_full_admitted_equal_off"
    expected=deployment.full_history_forecast(population[0],population[1],cutoff=population[2])
    for row, state in zip(payload["rows"],expected["fixtures"],strict=True):
        assert row["match_id"]==state["match_id"]
        np.testing.assert_allclose(payload["diagnostics"]["fixture_distributions"][row["match_id"]]["score_probability_matrix"],state["matrix"],atol=1e-12,rtol=0)
    return payload


def test_actual_cli_defaults_reach_full_history_policy(population,monkeypatch,tmp_path):
    execute_cli(monkeypatch,population,tmp_path,["--mode","match_result","--league","epl","--season","2022","--round","1"])


def test_actual_api_command_defaults_reach_same_cli_and_policy(population,monkeypatch,tmp_path):
    node=shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the actual API command builder; direct CLI path is tested separately.")
    api=(ROOT/"apps/api/src/index.js").read_text()
    # Evaluate the exact existing helper/command-builder function definitions,
    # without executing the API server, database setup or network listeners.
    source=api[api.index("function getString("):api.index("async function executePythonRun(")]
    request={"source":source,"entrypoint":str(CLI),"python":sys.executable,
        "params":{"mode":"match_result","league":"epl","season":2022,"round_number":1},
        "output":str(tmp_path/"api-unused-output.json")}
    javascript="""
const fs = require('fs');
const request = JSON.parse(fs.readFileSync(0,'utf8'));
const AppError = {badRequest: x => new Error(x), internal: x => new Error(x)};
const make = new Function('fs','AppError','PYTHON_COMMAND',request.source+'; return buildCommand;');
const command = make(fs,AppError,request.python)({kind:'Football',pythonEntrypoint:request.entrypoint},request.params,request.output);
process.stdout.write(JSON.stringify(command));
"""
    response=subprocess.run([node,"-e",javascript],input=json.dumps(request),text=True,capture_output=True,check=True,timeout=20)
    executable,args=json.loads(response.stdout)
    assert executable==sys.executable and args[0]==str(CLI)
    assert args[args.index("--football_model")+1]=="dixon"
    assert args[args.index("--football_calibration")+1]=="auto"
    # Preserve all API-constructed options; only redirect its output path to the
    # synthetic harness location, avoiding a subprocess with unpatched data IO.
    args=args[1:]
    for option in ("--output-format","--output-path"):
        at=args.index(option);del args[at:at+2]
    args.remove("--quiet")
    execute_cli(monkeypatch,population,tmp_path,args)
