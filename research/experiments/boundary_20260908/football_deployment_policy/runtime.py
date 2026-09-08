"""Research-only, in-memory adapter around the unchanged canonical caller.

No filesystem IO, provider lookup, target attachment, scoring or activation is
performed here. Patch seams are explicit and restored even when a fit fails.
The adapter is intentionally serial: it temporarily wraps module-level calls.
If production later exposes a named legacy assessment entrypoint, that explicit
reference is used instead of the potentially changed production default. The
resolved entrypoint is recorded; source locks still govern historical replay.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from threading import Lock
from unittest.mock import patch
from typing import Any

import numpy as np

from packages.football.mrp import prediction
from packages.football.mrp.config import PredictionConfig
from packages.football.mrp.data import LocalFootballData, _normalize_timestamp
from packages.football.mrp.deployment import (
    canonical_sha256, fixture_view, full_history_forecast, joint_output,
    restore_calibrator, restore_dc, serialize_calibrator, serialize_dc, validate_inputs,
    validate_fit_status,
)
from packages.football.mrp.protocol import chronological_populations

_CALLER_LOCK = Lock()


def _equal(actual, expected, *, tolerance, context):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.allclose(a,b,atol=tolerance,rtol=0):
        raise ValueError(f"Canonical caller parity failed: {context}")


def _check_caller_outputs(states, result):
    if [s["match_id"] for s in states] != [r["match_id"] for r in result.rows]:
        raise ValueError("Canonical caller changed fixture order/population.")
    if set(result.diagnostics["fixture_distributions"]) != {s["match_id"] for s in states}:
        raise ValueError("Canonical caller omitted or added a complete distribution.")
    for state, row in zip(states, result.rows):
        actual = result.diagnostics["fixture_distributions"][state["match_id"]]
        for k, target, tolerance in (
            ("score_probability_matrix", "matrix", 1e-12),
            ("outcome_probabilities", "hda", 1e-10),
            ("expected_goals", "expected_goals", 1e-10),
            ("base_omitted_probability_mass", "omitted_probability_mass", 1e-15),
            ("reconciled_tail_error_bound", "tail_probability_bound", 1e-15),
        ):
            _equal(actual[k], state[target], tolerance=tolerance, context=f"{state['match_id']} {k}")
        _equal([float(row[k]) for k in ("home_win_prob","draw_prob","away_win_prob")], state["hda"],
            tolerance=6e-13, context="12-decimal HDA")
        _equal([float(row[k]) for k in ("raw_home_win_prob","raw_draw_prob","raw_away_win_prob")], state["raw_hda"],
            tolerance=6e-13, context="12-decimal raw HDA")
        _equal([float(row[k]) for k in ("dixon_base_lambda_home_goals","dixon_base_lambda_away_goals")],
            [state["lambda_home"],state["lambda_away"]], tolerance=5.00001e-7, context="rendered base rates")
        _equal([float(row[k]) for k in ("expected_home_goals","expected_away_goals")], state["expected_goals"],
            tolerance=5.00001e-7, context="rendered expected goals")
        if row["predicted_scoreline"] != f"{state['mode'][0]}-{state['mode'][1]}":
            raise ValueError("Canonical mode score differs.")
        _equal(float(row["scoreline_prob"]), state["mode"][2], tolerance=6e-13, context="rendered mode probability")
        if row["model_used"] != "dixon" or row["gbdt_enabled"] != "false":
            raise ValueError("The selected canonical policy is no longer Dixon.")


def _run_caller(history, fixtures, *, expected_partitions, calibration_policy,
                shadow_eval=False, suppress_unused_gbdt=True, model_override=None):
    if type(shadow_eval) is not bool or type(suppress_unused_gbdt) is not bool:
        raise TypeError("Caller mode switches must be actual booleans.")
    entrypoint_name = "run_assessment_prediction" if hasattr(prediction,"run_assessment_prediction") else "run_prediction"
    entrypoint = getattr(prediction,entrypoint_name)
    if not callable(entrypoint):
        raise TypeError("The explicit canonical assessment entrypoint must be callable.")
    leagues, seasons, rounds = ({getattr(f, key) for f in fixtures} for key in ("league", "season", "round_number"))
    if len(leagues) != 1 or len(seasons) != 1 or len(rounds) != 1:
        raise ValueError("Frozen fixture block is incompatible with canonical league/season/round selection.")
    config = PredictionConfig(league=fixtures[0].league, season=fixtures[0].season,
        round_number=fixtures[0].round_number or 1, mode="matchday",
        football_model="dixon", football_calibration=calibration_policy,
        goal_strength_half_life_days=None, shadow_eval=shadow_eval, weather_enabled=False)
    dataset = LocalFootballData(None, {}, list(history), list(fixtures))
    captures: dict[str, Any] = {"goal_fit_calls":0, "goal_kernel_fits":0, "dixon_calibration_calls":0,
        "dixon_learned_calibration_calls":0, "gbdt_suppressed":suppress_unused_gbdt,
        "state_injected":model_override is not None,
        "canonical_entrypoint":f"packages.football.mrp.prediction.{entrypoint_name}"}
    original_fit = prediction.fit_dixon_coles
    original_cal = prediction.fit_probability_calibrator_with_policy
    original_rows = prediction.fit_probability_calibrator_from_rows
    original_select_history = prediction.select_training_matches
    original_select_fixtures = prediction.select_target_fixtures
    def select_history(data, cfg, selected):
        rows, notes = original_select_history(data,cfg,selected)
        if canonical_sha256(rows_as_dict(rows)) != canonical_sha256(rows_as_dict(history)):
            raise ValueError("Canonical history selection changed the frozen block population.")
        return rows, notes
    def select_fixtures(data,cfg):
        rows, notes = original_select_fixtures(data,cfg)
        if rows != fixtures:
            raise ValueError("Canonical fixture selection changed the frozen block population/order.")
        return rows, notes
    def fit(rows, notes, **kwargs):
        captures["goal_fit_calls"] += 1
        if captures["goal_fit_calls"] != 1:
            raise RuntimeError("The canonical caller requested duplicate goal fits.")
        if [m.match_id for m in rows] != expected_partitions["fit"]["match_ids"]:
            raise ValueError("Actual canonical prefix fit identities changed.")
        if kwargs.get("half_life_days") is not None:
            raise ValueError("Canonical goal weighting changed.")
        if model_override is None:
            captures["goal_kernel_fits"] += 1
            model = original_fit(rows,notes,**kwargs)
            validate_fit_status(model, rows)
        else:
            model = model_override
        captures["model"] = model
        return model
    def calibrate(rows, model, notes, **kwargs):
        captures["dixon_calibration_calls"] += 1
        captures["dixon_learned_calibration_calls"] += int(kwargs.get("policy") != "off")
        if [m.match_id for m in rows] != expected_partitions["calibration"]["match_ids"] or kwargs.get("policy") != calibration_policy:
            raise ValueError("Actual canonical calibration population/policy changed.")
        result = original_cal(rows,model,notes,**kwargs)
        captures["calibrator"] = result
        return result
    def from_rows(*args,**kwargs):
        result = original_rows(*args,**kwargs)
        if kwargs.get("context") == "dixon_no_heldout_calibration":
            captures["dixon_calibration_calls"] += 1
            captures["calibrator"] = result
        return result
    if not _CALLER_LOCK.acquire(blocking=False):
        raise RuntimeError("Concurrent canonical adapters are forbidden.")
    try:
        with ExitStack() as stack:
            bindings = {"load_local_football_data":lambda cfg:(dataset, ["Frozen block in-memory IO adapter; provider loading disabled."]),
                "select_target_fixtures":select_fixtures,"select_training_matches":select_history,
                "fit_dixon_coles":fit,"fit_probability_calibrator_with_policy":calibrate,
                "fit_probability_calibrator_from_rows":from_rows}
            if suppress_unused_gbdt:
                bindings["train_gradient_boosting_model"] = lambda rows,notes:None
            for name, replacement in bindings.items():
                stack.enter_context(patch.object(prediction,name,replacement))
            result = entrypoint(config)
    finally:
        _CALLER_LOCK.release()
    if captures["goal_fit_calls"] != 1 or captures["dixon_calibration_calls"] != 1:
        raise RuntimeError("Canonical goal/calibration call budget changed.")
    observed = {k:result.diagnostics["protocol"][k] for k in expected_partitions}
    if observed != expected_partitions:
        raise ValueError("Canonical assessment partition metadata changed.")
    return result, captures


def rows_as_dict(rows):
    from dataclasses import asdict
    return [asdict(r) for r in rows]


def execute_block(history, fixtures, saved_full_model, cutoff, expected_partitions, *,
                  saved_full_lineage=None, expected_reference_probabilities=None,
                  expected_candidate_probabilities=None, shadow_eval=False,
                  suppress_unused_gbdt=True):
    """Run one frozen block without attaching fixture scores or changing policy.

    A supplied saved model selects restoration; None selects one full-history
    fit. This is a caller decision, never a score-dependent fallback.
    """
    if type(shadow_eval) is not bool or type(suppress_unused_gbdt) is not bool:
        raise TypeError("Caller mode switches must be actual booleans.")
    cutoff = _normalize_timestamp(datetime.fromisoformat(cutoff) if isinstance(cutoff,str) else cutoff)
    fixtures = [fixture_view(f) for f in fixtures]
    full_lineage = validate_inputs(history,fixtures,cutoff)
    if cutoff != min(f.date for f in fixtures):
        raise ValueError("Block cutoff must equal the canonical first-fixture cutoff.")
    if chronological_populations(history).metadata() != expected_partitions:
        raise ValueError("Frozen assessment partitions no longer match admitted history.")
    # Validate/reuse or fit exactly once; failure cannot trigger a different branch.
    candidate = full_history_forecast(history,fixtures,cutoff=cutoff,
        restored_model=saved_full_model,restored_lineage=saved_full_lineage)
    result, capture = _run_caller(history,fixtures,expected_partitions=expected_partitions,
        calibration_policy="auto",shadow_eval=shadow_eval,suppress_unused_gbdt=suppress_unused_gbdt)
    model, calibrator = capture.pop("model"), capture.pop("calibrator")
    reference_states = [joint_output(model,f,calibrator) for f in fixtures]
    _check_caller_outputs(reference_states,result)
    model_state, cal_state = serialize_dc(model), serialize_calibrator(calibrator)
    replay_cal = restore_calibrator(cal_state)
    for f, state in zip(fixtures,reference_states):
        replay = joint_output(restore_dc(model_state),f,replay_cal)
        _equal(replay["matrix"],state["matrix"],tolerance=1e-12,context="serialized reference replay")
    # The unchanged caller still chooses a prefix. Injecting the full-history
    # state at that seam proves only complete fixture-output parity, not that
    # the existing production policy already fits full history.
    candidate_result, candidate_capture = _run_caller(history,fixtures,
        expected_partitions=expected_partitions,calibration_policy="off",shadow_eval=False,
        suppress_unused_gbdt=suppress_unused_gbdt,model_override=restore_dc(candidate["model_state"]))
    _check_caller_outputs(candidate["fixtures"],candidate_result)
    candidate["canonical_rows"] = deepcopy(candidate_result.rows)
    candidate["diagnostics"] = deepcopy(candidate_result.diagnostics)
    candidate["canonical_caller_entrypoint"] = candidate_capture["canonical_entrypoint"]
    candidate["canonical_replay_role"] = "state-injected fixture-output parity only; full-history fit lineage is stored separately"
    for states, expected, name in ((reference_states,expected_reference_probabilities,"old reference HDA"),
                                  (candidate["fixtures"],expected_candidate_probabilities,"old dc_equal HDA")):
        if expected is not None:
            _equal([s["hda"] for s in states],expected,tolerance=1e-10,context=name)
    reference = {"policy":"dc_frozen_prefix_auto", "model_state":model_state,
        "calibrator_state":cal_state, "fixtures":reference_states,
        "canonical_caller_entrypoint":capture["canonical_entrypoint"],
        "canonical_rows":deepcopy(result.rows), "diagnostics":deepcopy(result.diagnostics),
        "lineage":{"purpose":"frozen_prefix_assessment_and_reference_fixtures",
            "cutoff_utc":cutoff.isoformat(),"full_admitted_history":full_lineage,
            "assessment_partitions":deepcopy(expected_partitions),
            "goal_fit_ids":list(expected_partitions["fit"]["match_ids"]),
            "calibration_ids":list(expected_partitions["calibration"]["match_ids"])} }
    return {"schema":"football_deployment_block_v1","reference":reference,"candidate":candidate,
        "parity":{"reference_actual_caller":True,"reference_serialized_replay":True,
            "candidate_state_injected_caller_output":True,
            "old_reference_hda_checked":expected_reference_probabilities is not None,
            "old_candidate_hda_checked":expected_candidate_probabilities is not None,
            "state_injection_limit":"Candidate caller replay changes no fit population; it substitutes an independently full-history-bound model only to verify emitted fixture calculations.",
            "gbdt_suppressed":suppress_unused_gbdt},
        "fit_counts":{"prefix_goal_fits":capture["goal_kernel_fits"],
            "full_history_goal_fits":candidate["new_goal_model_fits"],
            "reference_calibration_policy_calls":capture["dixon_calibration_calls"],
            "candidate_learned_calibration_fits":candidate_capture["dixon_learned_calibration_calls"],
            "candidate_parity_goal_fits":candidate_capture["goal_kernel_fits"]}}
