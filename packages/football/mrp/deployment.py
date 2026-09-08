"""Explicit full-admitted-history Dixon forecasts; no default-policy activation.

The caller owns admission/window selection. This module rejects an inadmissible
row instead of silently changing that population. Historical assessment models
are not refitted or relabelled by this independent fixture-model helper.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime
import hashlib
import json
import math
from typing import Any

from .data import FixtureRecord, MatchRecord, _normalize_timestamp, match_available_at
from .joint import DixonColesModel, SUPPORT_EPSILON, fit_dixon_coles, dixon_coles_tau
from .score_distribution import build_score_distribution
from . import training
from .utils import record_sort_key


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=lambda x: x.isoformat()).encode()).hexdigest()


def fixture_view(record: FixtureRecord | MatchRecord) -> FixtureRecord:
    """Copy only issue metadata, including when a caller supplies a scored row."""
    return FixtureRecord(**{f.name: getattr(record, f.name) for f in fields(FixtureRecord)})


def validate_inputs(history: list[MatchRecord], fixtures: list[FixtureRecord],
                    cutoff: datetime) -> dict[str, Any]:
    cutoff = _normalize_timestamp(cutoff)
    if cutoff is None:
        raise ValueError("A dated forecast cutoff is required.")
    # Finish clock/identity admission before accessing any numeric result.
    if len({m.match_id for m in history}) != len(history):
        raise ValueError("Historical match IDs must be unique.")
    for m in history:
        if m.date is None or m.date >= cutoff:
            raise ValueError("Every admitted history date must precede cutoff.")
        available = match_available_at(m)
        if available is None or available > cutoff:
            raise ValueError("Every admitted result must be available by cutoff.")
    if [m.match_id for m in history] != [m.match_id for m in sorted(history, key=record_sort_key)]:
        raise ValueError("Admitted history must retain canonical chronological order.")
    if not fixtures or len({f.match_id for f in fixtures}) != len(fixtures):
        raise ValueError("A nonempty unique fixture population is required.")
    if {f.match_id for f in fixtures} & {m.match_id for m in history}:
        raise ValueError("Forecast fixtures cannot also be fitted history.")
    if any(f.date is None or f.date < cutoff for f in fixtures):
        raise ValueError("Fixture dates must be at or after forecast cutoff.")
    leagues = {f.league for f in fixtures}
    if len(leagues) != 1 or any(m.league not in leagues for m in history):
        raise ValueError("A forecast block must contain one identical history/fixture league.")
    for m in history:
        if m.home_team_id == m.away_team_id:
            raise ValueError("A match must contain distinct teams.")
        if any(type(v) is not int or v < 0 for v in (m.home_goals, m.away_goals)):
            raise ValueError("Every admitted result must have nonnegative integer goals.")
    if any(f.home_team_id == f.away_team_id for f in fixtures):
        raise ValueError("A fixture must contain distinct teams.")
    ids = [m.match_id for m in history]
    return {"cutoff_utc": cutoff.isoformat(), "fit_match_ids": ids,
        "fit_ids_sha256": canonical_sha256(ids), "fit_sample_size": len(ids),
        "match_records_sha256": canonical_sha256([asdict(m) for m in history]),
        "latest_result_available_at": max((match_available_at(m).isoformat() for m in history), default=None),
        "weight_policy": "equal", "half_life_days": None,
        "calibration_policy": "off", "purpose": "fixture_forecast_full_admitted_history"}


def serialize_dc(model: DixonColesModel) -> dict[str, Any]:
    state = {"schema": "football_dc_state_v1", **asdict(model)}
    canonical_sha256(state)  # reject non-standard/nonfinite JSON state
    return deepcopy(state)


def validate_fit_status(model: DixonColesModel, history: list[MatchRecord]) -> None:
    """Reject unsuccessful fresh fits; an empty population has an explicit prior."""
    diagnostics = model.diagnostics
    if type(diagnostics.get("fit_sample_size")) is not int or diagnostics["fit_sample_size"] != len(history):
        raise RuntimeError("Goal fit returned an inconsistent fitted population size.")
    if history:
        if diagnostics.get("status") != "fitted" or diagnostics.get("converged") is not True:
            raise RuntimeError("Goal fit did not return explicit successful convergence.")
        if diagnostics.get("fit_match_ids") != [m.match_id for m in history]:
            raise RuntimeError("Goal fit returned different fitted match identities.")
        if diagnostics.get("half_life_days") is not None:
            raise RuntimeError("Goal fit returned a different weighting policy.")
    elif diagnostics.get("status") != "prior_only" or diagnostics.get("converged") is not None:
        raise RuntimeError("Empty history requires the explicit canonical prior-only state.")


def restore_dc(state: dict[str, Any]) -> DixonColesModel:
    """Accept the exact old dc_equal parameter map or this module's state."""
    if state.get("schema", "football_dc_state_v1") != "football_dc_state_v1":
        raise ValueError("Unknown goal-model schema.")
    required = {"attack", "defense", "home_intercept", "away_intercept", "rho", "diagnostics"}
    if set(state) - {"schema"} != required:
        raise ValueError("Unexpected or missing goal-model fields.")
    canonical_sha256(state)
    model = DixonColesModel(**deepcopy({k: state[k] for k in required}))
    if set(model.attack) != set(model.defense):
        raise ValueError("Attack/defense team populations differ.")
    values = [*model.attack.values(), *model.defense.values(), model.home_intercept, model.away_intercept, model.rho]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Goal-model coefficients must be finite numbers.")
    if not -.2 <= model.rho <= .2:
        raise ValueError("Restored rho violates the canonical fit bounds.")
    if any(abs(v) > 3 + 1e-7 for v in [*model.attack.values(), *model.defense.values()]):
        raise ValueError("Restored team coefficient violates fit bounds.")
    if max(abs(math.fsum(model.attack.values())), abs(math.fsum(model.defense.values()))) > 1e-7:
        raise ValueError("Restored effects are not centered.")
    neutral = "__deployment_unseen_neutral__"
    while neutral in model.attack:
        neutral += "_"
    for h in [*model.attack, neutral]:
        for a in [*model.attack, neutral]:
            lh, la = model.expected_goals(h, a)
            if min(lh, la) < .05 - 1e-7 or max(lh, la) > 6 + 1e-7:
                raise ValueError("Restored all-pair rates violate canonical bounds.")
            if min(dixon_coles_tau(x,y,lh,la,model.rho) for x,y in ((0,0),(0,1),(1,0),(1,1))) < SUPPORT_EPSILON - 1e-7:
                raise ValueError("Restored low-score support violates fit constraints.")
    return model


def serialize_calibrator(calibrator: training.ProbabilityCalibrator) -> dict[str, Any]:
    """Capture actual fitted class transformers, including identity classes."""
    if calibrator.method not in {"identity", "isotonic", "platt"} or len(calibrator.per_class_functions) != 3:
        raise ValueError("Unsupported calibrator.")
    functions = []
    for fn in calibrator.per_class_functions:
        if fn is training._identity:
            functions.append({"kind": "identity"})
            continue
        defaults = getattr(fn, "__defaults__", None)
        if not defaults or len(defaults) != 1:
            raise ValueError("Unrecognized canonical calibration closure.")
        estimator = defaults[0]
        if type(estimator) is training.IsotonicRegression and calibrator.method == "isotonic":
            functions.append({"kind": "isotonic", "params": estimator.get_params(deep=False),
                "X_thresholds_": estimator.X_thresholds_.tolist(), "y_thresholds_": estimator.y_thresholds_.tolist(),
                "X_min_": float(estimator.X_min_), "X_max_": float(estimator.X_max_),
                "increasing_": bool(estimator.increasing_),
                "n_features_in_": getattr(estimator, "n_features_in_", None)})
        elif type(estimator) is training.LogisticRegression and calibrator.method == "platt":
            functions.append({"kind": "platt", "params": estimator.get_params(deep=False),
                "coef_": estimator.coef_.tolist(), "intercept_": estimator.intercept_.tolist(),
                "classes_": estimator.classes_.tolist(), "n_features_in_": int(estimator.n_features_in_),
                "n_iter_": estimator.n_iter_.tolist()})
        else:
            raise ValueError("Calibration closure does not match its declared method.")
    state = {"schema": "football_calibrator_state_v1", "method": calibrator.method,
        "class_functions": functions, "normalization": "canonical normalize before and after class functions",
        "nonidentity_class_floor": 1e-12, "platt_input_clip": [1e-12, 1-1e-12]}
    canonical_sha256(state)
    return deepcopy(state)


def restore_calibrator(state: dict[str, Any]) -> training.ProbabilityCalibrator:
    """Restore native transformer state without optimization or synthetic fit."""
    import numpy as np
    if state.get("schema") != "football_calibrator_state_v1" or state.get("method") not in {"identity", "isotonic", "platt"}:
        raise ValueError("Unknown calibrator state.")
    if len(state.get("class_functions", [])) != 3:
        raise ValueError("Three class functions are required.")
    if state.get("nonidentity_class_floor") != 1e-12 or state.get("platt_input_clip") != [1e-12, 1-1e-12]:
        raise ValueError("Calibration numerical semantics changed.")
    canonical_sha256(state)
    functions = []
    for item in state["class_functions"]:
        kind = item.get("kind")
        if kind == "identity":
            functions.append(training._identity)
        elif kind == "isotonic" and state["method"] == "isotonic":
            model = training.IsotonicRegression(**item["params"])
            x, y = np.asarray(item["X_thresholds_"], dtype=float), np.asarray(item["y_thresholds_"], dtype=float)
            if x.ndim != 1 or x.size < 1 or y.shape != x.shape or np.any(np.diff(x) <= 0) or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
                raise ValueError("Invalid isotonic threshold arrays.")
            if item["X_min_"] != float(x[0]) or item["X_max_"] != float(x[-1]) or item["params"].get("out_of_bounds") != "clip":
                raise ValueError("Isotonic bounds differ from canonical fitted state.")
            model.X_thresholds_, model.y_thresholds_ = x, y
            model.X_min_, model.X_max_ = item["X_min_"], item["X_max_"]
            model.increasing_ = item["increasing_"]
            if item["n_features_in_"] is not None:
                model.n_features_in_ = item["n_features_in_"]
            model._build_f(x, y)
            def iso(value, transformer=model):
                return float(transformer.predict([value])[0])
            functions.append(iso)
        elif kind == "platt" and state["method"] == "platt":
            model = training.LogisticRegression(**item["params"])
            if item["classes_"] != [0,1] or item["n_features_in_"] != 1:
                raise ValueError("Platt model must be the canonical binary one-feature fit.")
            model.coef_ = np.asarray(item["coef_"], dtype=float)
            model.intercept_ = np.asarray(item["intercept_"], dtype=float)
            if model.coef_.shape != (1,1) or model.intercept_.shape != (1,):
                raise ValueError("Invalid Platt coefficient shape.")
            model.classes_ = np.asarray(item["classes_"], dtype=int)
            model.n_features_in_, model.n_iter_ = 1, np.asarray(item["n_iter_"], dtype=int)
            def platt(value, transformer=model):
                bounded = min(1-1e-12, max(1e-12, value))
                return float(transformer.predict_proba([[math.log(bounded/(1-bounded))]])[0][1])
            functions.append(platt)
        else:
            raise ValueError("Unknown or method-inconsistent class transformer.")
    return training.ProbabilityCalibrator(state["method"], functions)


def joint_output(model: DixonColesModel, fixture: FixtureRecord,
                 calibrator: training.ProbabilityCalibrator) -> dict[str, Any]:
    lh, la = model.expected_goals(fixture.home_team_id, fixture.away_team_id)
    raw = build_score_distribution(lh, la, model.rho)
    selected = calibrator.apply(raw.outcome_probabilities)
    joint = raw.reconcile(selected)
    return {"match_id": fixture.match_id, "date": fixture.date.isoformat(),
        "league": fixture.league, "season": fixture.season,
        "home_team_id": fixture.home_team_id, "away_team_id": fixture.away_team_id,
        "lambda_home": lh, "lambda_away": la, "rho": model.rho,
        "raw_hda": list(raw.outcome_probabilities), "selected_hda": list(selected),
        "hda": list(joint.outcome_probabilities), "matrix": [list(r) for r in joint.matrix],
        "dimensions": [len(joint.matrix), len(joint.matrix[0])],
        "expected_goals": list(joint.expected_goals), "mode": list(joint.most_likely_scoreline),
        "omitted_probability_mass": joint.omitted_probability_mass,
        "raw_omitted_probability_mass": raw.omitted_probability_mass,
        "tail_probability_bound": joint.tail_probability_bound,
        "reconciled": joint.reconciled, "source_parameters": list(joint.source_parameters),
        "reconciliation": "canonical conditional-score-preserving selected-HDA reconciliation"}


def full_history_forecast(history: list[MatchRecord], fixtures: list[FixtureRecord], *,
                          cutoff: datetime, restored_model: dict[str, Any] | DixonColesModel | None = None,
                          restored_lineage: dict[str, Any] | None = None) -> dict[str, Any]:
    fixtures = [fixture_view(f) for f in fixtures]
    lineage = validate_inputs(history, fixtures, cutoff)
    notes: list[str] = []
    if restored_model is None:
        if restored_lineage is not None:
            raise ValueError("Restored lineage without a restored model is ambiguous.")
        model = fit_dixon_coles(history, notes, half_life_days=None, reference_time=_normalize_timestamp(cutoff))
        validate_fit_status(model, history)
        branch, fits = "canonical_full_history_fit", 1
    else:
        if not isinstance(restored_lineage, dict):
            raise ValueError("Restored parameters require bound full-history lineage.")
        for key in ("cutoff_utc", "fit_match_ids", "fit_ids_sha256", "match_records_sha256", "latest_result_available_at"):
            if restored_lineage.get(key) != lineage[key]:
                raise ValueError(f"Restored full-history lineage mismatch: {key}")
        state = serialize_dc(restored_model) if isinstance(restored_model, DixonColesModel) else restored_model
        if restored_lineage.get("goal_state_sha256") != canonical_sha256({k:v for k,v in state.items() if k != "schema"}):
            raise ValueError("Restored coefficients do not match the bound goal-state digest.")
        model = restore_dc(state)
        diagnostics = model.diagnostics
        if history and (diagnostics.get("status") != "fitted" or diagnostics.get("converged") is not True):
            raise ValueError("Restored full model has no successful canonical fit.")
        if diagnostics.get("fit_sample_size") != len(history) or diagnostics.get("half_life_days") is not None:
            raise ValueError("Restored fit count or equal-weight policy differs.")
        if history and diagnostics.get("latest_fit_result_available_at") != lineage["latest_result_available_at"]:
            raise ValueError("Restored diagnostic result-availability cutoff differs.")
        if diagnostics.get("fit_match_ids", lineage["fit_match_ids"]) != lineage["fit_match_ids"]:
            raise ValueError("Restored diagnostic fit IDs differ.")
        teams = {m.home_team_id for m in history} | {m.away_team_id for m in history}
        if set(model.attack) != teams:
            raise ValueError("Restored model teams differ from full admitted history.")
        lineage["restored_source_lineage"] = deepcopy(restored_lineage)
        branch, fits = "validated_saved_full_history_state", 0
    identity = training.ProbabilityCalibrator("identity", [training._identity]*3)
    return {"policy": "dc_full_admitted_equal_off", "model_state": serialize_dc(model),
        "calibrator_state": serialize_calibrator(identity), "lineage": lineage,
        "restoration_branch": branch, "new_goal_model_fits": fits, "notes": notes,
        "fixtures": [joint_output(model, f, identity) for f in fixtures],
        "assessment_metrics_computed": False}
