#!/usr/bin/env python3
"""Frozen causal replay audit for the 2026 live-strategy learning surface.

This runner does not claim that historical actions are optimal.  It builds the
causal lap-to-lap replay dataset, reports partial-label behavior-cloning
diagnostics against a trivial prior-event baseline, and fails closed for
offline-Q or propensity OPE when their evidence contracts are not satisfied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.live_race.action_space import (
    ACTION_STAY_OUT,
    LEGAL_ACTION_MASK_SCHEMA_VERSION,
)
from packages.f1.models.live_race.environment import (
    LEAKAGE_CONTRACT_VERSION,
    REWARD_SEMANTICS,
    TRANSITION_FINGERPRINT_VERSION,
    StrategyTransition,
    build_replay_transitions,
)
from packages.f1.models.live_race.replay_buffer import (
    REPLAY_RECORD_SCHEMA_VERSION,
    ReplayBufferRecord,
)
from packages.f1.models.live_race.rl.behavior_cloning import (
    TrivialLegalActionBaseline,
    evaluate_behavior_cloning,
    fit_behavior_cloning,
)
from packages.f1.models.live_race.rl.replay_buffer import (
    REPLAY_DATASET_SCHEMA_VERSION,
    RLReplayDataset,
    StrategyActionIndex,
    build_rl_replay_dataset,
)
from packages.f1.models.live_race.sources import _standardize_laps
from packages.sports_core.paths import find_repo_root


SCHEMA_VERSION = "f1_live_strategy_replay_audit_v1"
YEAR = 2026
ROUNDS = tuple(range(1, 10))
ROUND_PATTERN = re.compile(r"^round_(\d{2})_")
PIT_LANE_SIGNAL = re.compile(r"\bPIT LANE ENTRY (OPEN|CLOSED)\b", re.IGNORECASE)
DEFAULT_OUTPUT = Path(
    "artifacts/backtests/f1/live_strategy/"
    "live_strategy_replay_audit_v2_20260714.json"
)

# The audit binds every module that directly defines ingestion, state,
# transition, action-index, replay, and BC semantics.  Configs are separately
# bound below so neither code nor policy settings can drift unnoticed.
IMPLEMENTATION_DEPENDENCIES: tuple[str, ...] = (
    "packages/__init__.py",
    "packages/f1/__init__.py",
    "packages/f1/data/__init__.py",
    "packages/f1/data/schemas/__init__.py",
    "packages/f1/data/schemas/session.py",
    "packages/f1/data/utils.py",
    "packages/f1/models/__init__.py",
    "packages/f1/models/live_race/__init__.py",
    "packages/f1/models/live_race/action_space.py",
    "packages/f1/models/live_race/environment.py",
    "packages/f1/models/live_race/replay_buffer.py",
    "packages/f1/models/live_race/state.py",
    "packages/f1/models/live_race/sources.py",
    "packages/f1/models/live_race/rl/__init__.py",
    "packages/f1/models/live_race/rl/behavior_cloning.py",
    "packages/f1/models/live_race/rl/replay_buffer.py",
    "packages/sports_core/__init__.py",
    "packages/sports_core/paths.py",
)
CONFIGURATION_DEPENDENCIES: tuple[str, ...] = (
    "configs/f1/profiles/live_race.yaml",
    "configs/f1/profiles/live_strategy.yaml",
    "configs/f1/profiles/live_strategy_rl.yaml",
)

LOCKED_COUNTS: Mapping[str, object] = {
    "rounds": 9,
    "transitions": 9506,
    "behavior_cloning_rows": 9506,
    "offline_q_rows": 0,
    "propensity_ope_rows": 0,
    "elapsed_time_positive_rows": 9506,
    "reward_components_fully_observed_rows": 9489,
    "reward_components_incomplete_rows": 17,
    "nonzero_position_gain_rows": 1424,
    "states_with_multiple_used_compounds": 5939,
    "compatible_action_key_count": 8,
    "exact_action_key_count": 0,
    "supported_action_family_count": 4,
    "action_family_counts": {
        "stay_out": 9191,
        "pit_now:HARD": 163,
        "pit_now:SOFT": 93,
        "pit_now:MEDIUM": 59,
    },
}


def _repo_root() -> Path:
    return find_repo_root(__file__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validated_generated_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("generated-at must be a timezone-aware ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("generated-at must use an explicit UTC timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _round_number(path: Path) -> int:
    match = ROUND_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"not a round directory: {path}")
    return int(match.group(1))


def _single_file(round_dir: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(round_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"{round_dir} must contain exactly one {label}; found {len(matches)}"
        )
    return matches[0]


def _input_pairs(
    root: Path,
    *,
    weekends_dir: Path,
) -> list[tuple[int, Path, Path]]:
    year_dir = (weekends_dir / str(YEAR)).resolve()
    round_dirs = sorted(
        (path for path in year_dir.glob("round_*") if path.is_dir()),
        key=_round_number,
    )
    by_round = {_round_number(path): path for path in round_dirs}
    if tuple(sorted(by_round)) != ROUNDS:
        raise ValueError(
            f"frozen replay requires exactly rounds {ROUNDS}; observed {tuple(sorted(by_round))}"
        )
    pairs: list[tuple[int, Path, Path]] = []
    for round_number in ROUNDS:
        round_dir = by_round[round_number]
        laps = _single_file(round_dir, "*_race_laps.csv", label="race-laps CSV")
        messages = _single_file(
            round_dir,
            "*_race_race_control_messages.csv",
            label="race-control CSV",
        )
        for path in (laps, messages):
            resolved = path.resolve()
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"input escapes repository root: {resolved}")
        pairs.append((round_number, laps.resolve(), messages.resolve()))
    return pairs


def _manifest(paths: Iterable[Path], *, root: Path) -> dict[str, str]:
    return {
        str(path.resolve().relative_to(root)): _sha256(path.resolve())
        for path in sorted(set(paths))
    }


def _input_manifest(
    pairs: Sequence[tuple[int, Path, Path]],
    *,
    root: Path,
) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for round_number, laps, messages in pairs:
        for role, path in (("race_laps", laps), ("race_control", messages)):
            entries[str(path.relative_to(root))] = {
                "sha256": _sha256(path),
                "bytes": int(path.stat().st_size),
                "year": YEAR,
                "round": int(round_number),
                "role": role,
            }
    return dict(sorted(entries.items()))


def _assert_hash_manifest_current(
    manifest: Mapping[str, str],
    *,
    root: Path,
    label: str,
) -> None:
    changed = [
        relative
        for relative, expected in manifest.items()
        if _sha256(root / relative) != expected
    ]
    if changed:
        raise RuntimeError(f"{label} changed during replay audit: {sorted(changed)}")


def apply_causal_pit_lane_status(
    frame: pd.DataFrame,
    messages: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach only strictly prior PIT LANE ENTRY OPEN/CLOSED messages.

    A message stamped with race-control lap ``L`` is not available to a state
    at the end of lap ``L``.  It becomes eligible only when ``L < state_lap``.
    Pit-exit messages describe a different rule and are deliberately ignored.
    """

    if "lap_number" not in frame.columns:
        raise ValueError("standardized replay frame requires lap_number")
    output = frame.copy()
    output["pit_lane_open"] = False
    output["pit_lane_open_known"] = False
    output["pit_lane_status_source_lap"] = np.nan
    output["pit_lane_status_source_message"] = None

    if messages.empty or "Message" not in messages.columns or "Lap" not in messages.columns:
        return output, {
            "pit_lane_entry_signal_rows": 0,
            "state_rows_with_known_pit_lane_status": 0,
            "state_rows_with_closed_pit_lane": 0,
            "strictly_prior_source_lap_contract": True,
        }

    work = messages.copy().reset_index(drop=False).rename(columns={"index": "_row_order"})
    work["_message"] = work["Message"].fillna("").astype(str).str.strip()
    work["_status"] = work["_message"].str.extract(PIT_LANE_SIGNAL, expand=False)
    work["_control_lap"] = pd.to_numeric(work["Lap"], errors="coerce")
    work["_control_time"] = pd.to_datetime(work.get("Time"), errors="coerce")
    signals = work.loc[work["_status"].notna() & work["_control_lap"].notna()].copy()
    signals["_status"] = signals["_status"].str.upper()
    signals = signals.sort_values(
        ["_control_lap", "_control_time", "_row_order"],
        kind="mergesort",
        na_position="first",
    )

    known_rows = 0
    closed_rows = 0
    lags: list[int] = []
    for state_lap in sorted(pd.to_numeric(output["lap_number"], errors="coerce").dropna().unique()):
        eligible = signals.loc[signals["_control_lap"].lt(float(state_lap))]
        if eligible.empty:
            continue
        signal = eligible.iloc[-1]
        rows = pd.to_numeric(output["lap_number"], errors="coerce").eq(float(state_lap))
        is_open = str(signal["_status"]).upper() == "OPEN"
        output.loc[rows, "pit_lane_open"] = bool(is_open)
        output.loc[rows, "pit_lane_open_known"] = True
        output.loc[rows, "pit_lane_status_source_lap"] = float(signal["_control_lap"])
        output.loc[rows, "pit_lane_status_source_message"] = str(signal["_message"])
        row_count = int(rows.sum())
        known_rows += row_count
        closed_rows += int((not is_open) * row_count)
        lags.extend([int(float(state_lap) - float(signal["_control_lap"]))] * row_count)

    known_mask = output["pit_lane_open_known"].astype(bool)
    if known_mask.any():
        state_laps = pd.to_numeric(output.loc[known_mask, "lap_number"], errors="raise")
        source_laps = pd.to_numeric(
            output.loc[known_mask, "pit_lane_status_source_lap"], errors="raise"
        )
        if not bool(source_laps.lt(state_laps).all()):
            raise RuntimeError("pit-lane message leaked from the same or a future lap")

    return output, {
        "pit_lane_entry_signal_rows": int(len(signals)),
        "state_rows_with_known_pit_lane_status": int(known_rows),
        "state_rows_with_closed_pit_lane": int(closed_rows),
        "minimum_source_lap_lag": int(min(lags)) if lags else None,
        "maximum_source_lap_lag": int(max(lags)) if lags else None,
        "strictly_prior_source_lap_contract": True,
        "ignored_pit_exit_messages": int(
            work["_message"].str.contains("PIT EXIT", case=False, regex=False).sum()
        ),
    }


def _action_family(transition: StrategyTransition) -> str:
    action = transition.action_t
    if action.action_type == ACTION_STAY_OUT:
        return ACTION_STAY_OUT
    return f"{action.action_type}:{action.compound}"


def _reward_diagnostics(
    transitions: Sequence[StrategyTransition],
) -> dict[str, object]:
    statuses = Counter(
        str(transition.metadata.get("reward_observation_status", "missing"))
        for transition in transitions
    )
    blockers: Counter[str] = Counter()
    for transition in transitions:
        blockers.update(
            str(value)
            for value in transition.metadata.get("reward_observation_blockers", ())
        )
    return {
        "semantics": REWARD_SEMANTICS,
        "elapsed_time_positive_rows": int(
            sum(
                float(transition.reward_t.components.get("race_time_delta_seconds", 0.0)) > 0.0
                for transition in transitions
            )
        ),
        "reward_components_fully_observed_rows": int(
            statuses.get("observed_required_components", 0)
        ),
        "reward_components_incomplete_rows": int(statuses.get("incomplete", 0)),
        "reward_observation_status_counts": dict(sorted(statuses.items())),
        "reward_observation_blocker_counts": dict(sorted(blockers.items())),
        "nonzero_position_gain_rows": int(
            sum(
                abs(float(transition.reward_t.components.get("position_gain", 0.0))) > 0.0
                for transition in transitions
            )
        ),
    }


def _dataset_summary(
    dataset: RLReplayDataset,
    transitions: Sequence[StrategyTransition],
) -> dict[str, object]:
    diagnostics = dataset.diagnostics()
    support = diagnostics["action_support"]
    bc_support = support["behavior_cloning"]
    offline_q_ineligible_reasons = Counter(
        str(reason)
        for example in dataset.examples
        for reason in example.ood_reasons
    )
    propensity_ope_ineligible_reasons = Counter(
        str(reason)
        for example in dataset.examples
        for reason in example.propensity_ope_ineligible_reasons
    )
    current_mask_input_blockers: Counter[str] = Counter()
    next_mask_input_blockers: Counter[str] = Counter()
    for transition in transitions:
        current_evidence = transition.metadata.get(
            "legal_action_mask_input_evidence",
            {},
        )
        if isinstance(current_evidence, Mapping):
            current_mask_input_blockers.update(
                str(value) for value in current_evidence.get("blockers", ())
            )
        if transition.done:
            continue
        next_evidence = transition.metadata.get(
            "next_legal_action_mask_input_evidence",
            {},
        )
        if isinstance(next_evidence, Mapping):
            next_mask_input_blockers.update(
                str(value) for value in next_evidence.get("blockers", ())
            )
    return {
        "transitions": int(len(transitions)),
        "behavior_cloning_rows": int(diagnostics["behavior_cloning_rows"]),
        "offline_q_rows": int(diagnostics["offline_q_rows"]),
        "propensity_ope_rows": int(diagnostics["propensity_ope_rows"]),
        "feature_count": int(diagnostics["feature_count"]),
        "action_count": int(diagnostics["action_count"]),
        "behavior_cloning_label_kinds": diagnostics["behavior_cloning_label_kinds"],
        "behavior_cloning_action_support": bc_support,
        "offline_q_action_support": support["offline_q"],
        "propensity_ope_action_support": support["propensity_ope"],
        "offline_q_ineligible_reason_counts": dict(
            sorted(offline_q_ineligible_reasons.items())
        ),
        "propensity_ope_ineligible_reason_counts": dict(
            sorted(propensity_ope_ineligible_reasons.items())
        ),
        "current_legal_mask_input_blocker_counts": dict(
            sorted(current_mask_input_blockers.items())
        ),
        "nonterminal_next_legal_mask_input_blocker_counts": dict(
            sorted(next_mask_input_blockers.items())
        ),
        "strategy_training_readiness_gate_pass": bool(
            diagnostics["strategy_training_readiness_gate_pass"]
        ),
        "action_family_counts": dict(
            sorted(Counter(_action_family(item) for item in transitions).items())
        ),
        "states_with_multiple_used_compounds": int(
            sum(len(tuple(item.state_t.used_compounds or ())) > 1 for item in transitions)
        ),
        "reward": _reward_diagnostics(transitions),
    }


def _mean_metric(rows: Sequence[dict[str, object]], path: Sequence[str]) -> float | None:
    values: list[float] = []
    for row in rows:
        value: object = row
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _rolling_behavior_cloning_diagnostics(
    records_by_round: Mapping[int, Sequence[ReplayBufferRecord]],
    *,
    action_index: StrategyActionIndex,
) -> dict[str, object]:
    prior_records: list[ReplayBufferRecord] = []
    event_rows: list[dict[str, object]] = []
    for round_number in ROUNDS:
        current_records = list(records_by_round[round_number])
        if not prior_records:
            event_rows.append(
                {
                    "round": int(round_number),
                    "role": "warmup_no_prior_event",
                    "train_rows": 0,
                    "test_rows": int(len(current_records)),
                    "evaluation": None,
                }
            )
            prior_records.extend(current_records)
            continue
        train = build_rl_replay_dataset(prior_records, action_index=action_index)
        test = build_rl_replay_dataset(current_records, action_index=action_index)
        policy = fit_behavior_cloning(train)
        baseline = TrivialLegalActionBaseline.fit(train)
        evaluation = evaluate_behavior_cloning(policy, test, baseline=baseline)
        event_rows.append(
            {
                "round": int(round_number),
                "role": "rolling_origin_audit",
                "train_rounds": list(range(1, round_number)),
                "train_rows": int(train.rows),
                "test_rows": int(test.rows),
                "evaluation": evaluation,
            }
        )
        prior_records.extend(current_records)

    audited = [row for row in event_rows if row["evaluation"] is not None]
    metrics = (
        "action_selection_accuracy",
        "action_type_accuracy",
        "pit_decision_accuracy",
        "pit_decision_precision",
        "pit_decision_recall",
        "pit_decision_f1",
        "pit_timing_lap_mae",
    )
    return {
        "role": "partial_label_warm_start_diagnostic_only",
        "selection_protocol": "strictly_prior_same_season_events_only",
        "historical_action_accuracy_is_not_policy_value": True,
        "offline_q_or_ope_value_claim": False,
        "promotion_gate_pass": False,
        "events": event_rows,
        "event_mean_candidate_metrics": {
            metric: _mean_metric(audited, ("evaluation", "metrics", metric))
            for metric in metrics
        },
        "event_mean_trivial_baseline_metrics": {
            metric: _mean_metric(
                audited,
                ("evaluation", "trivial_baseline", "metrics", metric),
            )
            for metric in metrics
        },
    }


def _locked_observed_counts(summary: Mapping[str, object]) -> dict[str, object]:
    reward = summary["reward"]
    bc_support = summary["behavior_cloning_action_support"]
    if not isinstance(reward, Mapping) or not isinstance(bc_support, Mapping):
        raise TypeError("invalid replay summary structure")
    return {
        "rounds": int(summary["rounds"]),
        "transitions": int(summary["transitions"]),
        "behavior_cloning_rows": int(summary["behavior_cloning_rows"]),
        "offline_q_rows": int(summary["offline_q_rows"]),
        "propensity_ope_rows": int(summary["propensity_ope_rows"]),
        "elapsed_time_positive_rows": int(reward["elapsed_time_positive_rows"]),
        "reward_components_fully_observed_rows": int(
            reward["reward_components_fully_observed_rows"]
        ),
        "reward_components_incomplete_rows": int(
            reward["reward_components_incomplete_rows"]
        ),
        "nonzero_position_gain_rows": int(reward["nonzero_position_gain_rows"]),
        "states_with_multiple_used_compounds": int(
            summary["states_with_multiple_used_compounds"]
        ),
        "compatible_action_key_count": int(bc_support["compatible_action_key_count"]),
        "exact_action_key_count": int(bc_support["exact_action_key_count"]),
        "supported_action_family_count": int(
            bc_support["supported_action_family_count"]
        ),
        "action_family_counts": summary["action_family_counts"],
    }


def _assert_locked_counts(summary: Mapping[str, object]) -> None:
    observed = _locked_observed_counts(summary)
    if observed != dict(LOCKED_COUNTS):
        raise RuntimeError(
            "frozen replay counts changed; refusing to publish: "
            + json.dumps({"expected": LOCKED_COUNTS, "observed": observed}, sort_keys=True)
        )


def run_audit(
    *,
    root: Path | None = None,
    weekends_dir: Path | None = None,
    generated_at: str = "2026-07-14T00:00:00Z",
    validate_locked_counts: bool = True,
) -> dict[str, object]:
    root = (root or _repo_root()).resolve()
    weekends_dir = (weekends_dir or (root / "data/f1/raw/weekends")).resolve()
    pairs = _input_pairs(root, weekends_dir=weekends_dir)

    implementation_paths = [Path(__file__).resolve()]
    implementation_paths.extend((root / path).resolve() for path in IMPLEMENTATION_DEPENDENCIES)
    configuration_paths = [(root / path).resolve() for path in CONFIGURATION_DEPENDENCIES]
    missing = [path for path in (*implementation_paths, *configuration_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"replay audit dependency missing: {missing}")
    implementation_manifest = _manifest(implementation_paths, root=root)
    configuration_manifest = _manifest(configuration_paths, root=root)
    input_manifest = _input_manifest(pairs, root=root)

    action_index = StrategyActionIndex.from_action_space()
    all_transitions: list[StrategyTransition] = []
    records_by_round: dict[int, list[ReplayBufferRecord]] = {}
    round_rows: list[dict[str, object]] = []
    for round_number, laps_path, control_path in pairs:
        raw_laps = pd.read_csv(laps_path)
        raw_control = pd.read_csv(control_path)
        frame = _standardize_laps(
            raw_laps,
            event_key=(YEAR * 100) + round_number,
            source_used="local_frozen_replay",
            session_name="Race",
        )
        frame, control_diagnostics = apply_causal_pit_lane_status(frame, raw_control)
        transitions = build_replay_transitions(frame)
        records = [
            ReplayBufferRecord.from_transition(
                transition,
                source="observed_race_laps",
                split_key=f"{YEAR}-R{round_number:02d}",
                metadata={"year": YEAR, "round": round_number},
            )
            for transition in transitions
        ]
        dataset = build_rl_replay_dataset(records, action_index=action_index)
        records_by_round[round_number] = records
        all_transitions.extend(transitions)
        event_summary = _dataset_summary(dataset, transitions)
        round_rows.append(
            {
                "year": YEAR,
                "round": int(round_number),
                "event_key": (YEAR * 100) + round_number,
                "input_rows": int(len(frame)),
                "pit_lane_control": control_diagnostics,
                **event_summary,
            }
        )

    all_records = [record for round_number in ROUNDS for record in records_by_round[round_number]]
    dataset = build_rl_replay_dataset(all_records, action_index=action_index)
    aggregate = {
        "rounds": int(len(round_rows)),
        **_dataset_summary(dataset, all_transitions),
    }
    if validate_locked_counts:
        _assert_locked_counts(aggregate)

    protocol = {
        "year": YEAR,
        "rounds": list(ROUNDS),
        "independent_evaluation_unit": "race_event",
        "state_cutoff": "end_of_completed_lap_t",
        "transition_target": "next_running_lap_or_first_post_pit_running_lap",
        "pit_stop_transition": "semi_markov_single_decision_interval",
        "pit_lane_message_contract": (
            "PIT_LANE_ENTRY_OPEN_OR_CLOSED_applies_only_when_control_lap_is_strictly_less_than_state_lap"
        ),
        "pit_exit_message_contract": (
            "pit_exit_messages_define_pit_exit_not_pit_lane_entry_and_are_ignored"
        ),
        "behavior_cloning_selection": "rolling_origin_strictly_prior_events",
        "offline_q_selection_or_fit": False,
        "propensity_ope_evaluation": False,
        "constraint_legality_is_authoritative": True,
        "operational_safety_fallback_is_separate_and_nonlegal": True,
        "operational_safety_fallback_excluded_from_bc_q_ope_and_policy_selection": True,
        "mandatory_compound_change_enforced_at_last_feasible_stop": True,
        "reason_no_offline_q_or_ope": (
            "no_rows_have_exact_observed_pace_mode_plus_certified_current_and_"
            "nonterminal_next_state_legal_masks; "
            "logged behavior propensities are also absent"
        ),
    }
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _validated_generated_at(generated_at),
        "git_head": _git_head(root),
        "protocol": protocol,
        "protocol_sha256": _canonical_json_sha256(protocol),
        "contracts": {
            "state": LEAKAGE_CONTRACT_VERSION,
            "transition_fingerprint": TRANSITION_FINGERPRINT_VERSION,
            "replay_record": REPLAY_RECORD_SCHEMA_VERSION,
            "rl_replay_dataset": REPLAY_DATASET_SCHEMA_VERSION,
            "legal_action_mask": LEGAL_ACTION_MASK_SCHEMA_VERSION,
            "legality_semantics": (
                "constraint_legal_mask_remains_all_false_when_no_compliant_action_exists"
            ),
            "operational_fallback_semantics": (
                "separately_tagged_nonlegal_safety_noop_never_enters_learning_or_policy_selection"
            ),
            "reward": REWARD_SEMANTICS,
        },
        "implementation_manifest": implementation_manifest,
        "implementation_manifest_sha256": _canonical_json_sha256(
            implementation_manifest
        ),
        "configuration_manifest": configuration_manifest,
        "configuration_manifest_sha256": _canonical_json_sha256(
            configuration_manifest
        ),
        "input_manifest": input_manifest,
        "input_manifest_sha256": _canonical_json_sha256(input_manifest),
        "input_counts": {
            "race_lap_files": 9,
            "race_control_files": 9,
            "total_files": 18,
        },
        "events": round_rows,
        "aggregate": aggregate,
        "behavior_cloning": _rolling_behavior_cloning_diagnostics(
            records_by_round,
            action_index=action_index,
        ),
        "decision": {
            "deterministic_fallback_retained": True,
            "behavior_cloning_trainable_as_warm_start": bool(
                aggregate["behavior_cloning_rows"]
            ),
            "behavior_cloning_promoted_as_strategy_optimizer": False,
            "offline_q_trainable": bool(aggregate["offline_q_rows"]),
            "propensity_ope_available": bool(aggregate["propensity_ope_rows"]),
            "rl_policy_promoted": False,
            "strategy_training_readiness_gate_pass": False,
        },
        "locked_count_assertions": dict(LOCKED_COUNTS),
    }

    _assert_hash_manifest_current(
        implementation_manifest,
        root=root,
        label="implementation manifest",
    )
    _assert_hash_manifest_current(
        configuration_manifest,
        root=root,
        label="configuration manifest",
    )
    _assert_hash_manifest_current(
        {path: str(entry["sha256"]) for path, entry in input_manifest.items()},
        root=root,
        label="input manifest",
    )
    return payload


def write_artifact_fail_closed(
    payload: Mapping[str, object],
    *,
    output: Path,
    root: Path,
) -> None:
    implementation = payload.get("implementation_manifest")
    configuration = payload.get("configuration_manifest")
    inputs = payload.get("input_manifest")
    if not isinstance(implementation, Mapping) or not isinstance(configuration, Mapping):
        raise TypeError("artifact manifests are missing")
    if not isinstance(inputs, Mapping):
        raise TypeError("artifact input manifest is missing")
    implementation_hashes = {str(path): str(value) for path, value in implementation.items()}
    configuration_hashes = {str(path): str(value) for path, value in configuration.items()}
    input_hashes = {
        str(path): str(value["sha256"])
        for path, value in inputs.items()
        if isinstance(value, Mapping)
    }
    for label, manifest in (
        ("implementation manifest", implementation_hashes),
        ("configuration manifest", configuration_hashes),
        ("input manifest", input_hashes),
    ):
        _assert_hash_manifest_current(manifest, root=root, label=label)

    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open("x", encoding="utf-8") as handle:
            created = True
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        for label, manifest in (
            ("implementation manifest", implementation_hashes),
            ("configuration manifest", configuration_hashes),
            ("input manifest", input_hashes),
        ):
            _assert_hash_manifest_current(manifest, root=root, label=label)
    except Exception:
        if created:
            output.unlink(missing_ok=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weekends-dir", type=Path, default=Path("data/f1/raw/weekends"))
    parser.add_argument("--generated-at", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = _repo_root()
    weekends_dir = args.weekends_dir
    if not weekends_dir.is_absolute():
        weekends_dir = root / weekends_dir
    output = args.output
    if not output.is_absolute():
        output = root / output
    payload = run_audit(
        root=root,
        weekends_dir=weekends_dir,
        generated_at=_validated_generated_at(args.generated_at),
        validate_locked_counts=True,
    )
    write_artifact_fail_closed(payload, output=output, root=root)
    print(
        json.dumps(
            {
                "output": str(output.relative_to(root)),
                "schema_version": payload["schema_version"],
                "transitions": payload["aggregate"]["transitions"],
                "behavior_cloning_rows": payload["aggregate"]["behavior_cloning_rows"],
                "offline_q_rows": payload["aggregate"]["offline_q_rows"],
                "propensity_ope_rows": payload["aggregate"]["propensity_ope_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
