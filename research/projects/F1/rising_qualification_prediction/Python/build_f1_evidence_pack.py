#!/usr/bin/env python3
"""Build a diligence-ready F1 evidence pack from rolling backtest artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd


SOURCE_FILES = [
    "research/projects/F1/rising_qualification_prediction/Python/build_f1_evidence_pack.py",
    "research/projects/F1/rising_qualification_prediction/Python/run_rolling_2026_backtest.py",
    "packages/f1/data/schemas/session.py",
    "packages/f1/data/schemas/result.py",
    "packages/f1/orchestration/prediction.py",
    "packages/f1/models/training.py",
    "packages/f1/features/assembly.py",
    "packages/f1/data/providers/base.py",
    "packages/f1/data/providers/fastf1.py",
    "packages/f1/data/providers/openf1.py",
    "packages/f1/data/providers/local_weekends.py",
    "packages/f1/data/utils.py",
    "packages/f1/betting/recommendations.py",
    "research/projects/F1/rising_qualification_prediction/Python/run_betting.py",
]

FORWARD_STAKE_RULE = (
    "Forward test rule: no bet unless timestamped market odds exist before market close; "
    "edge = model_probability - implied_probability >= 3%; expected_roi >= 2%; "
    "stake = min(0.25 Kelly, 1% bankroll per selection, 3% per market, 5% per event)."
)
ROLLING_BACKTEST_SCHEMA_VERSION = "f1_rolling_2026_point_in_time_v2"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class EvidenceDiscoveryError(ValueError):
    """Raised when evidence inputs cannot be selected without ambiguity."""


@dataclass(frozen=True)
class EvidenceSource:
    source_kind: str
    run_id: str
    selections_path: Path
    summary_path: Path
    prediction_paths: Mapping[tuple[int, int, str], Path]
    manifest_path: Optional[Path]
    manifest_sha256: Optional[str]
    manifest: Mapping[str, Any]
    frozen_complete: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _run_git(repo: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _combined_hash(repo: Path, files: Iterable[str]) -> str:
    h = hashlib.sha256()
    for rel in files:
        path = repo / rel
        if not path.exists():
            continue
        h.update(rel.encode("utf-8"))
        h.update(_sha256(path).encode("utf-8"))
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_reference(repo: Path, reference: object) -> Optional[Path]:
    if not isinstance(reference, str) or not reference.strip():
        return None
    path = Path(reference).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _validate_file_manifest(repo: Path, manifest: object, label: str) -> list[str]:
    if not isinstance(manifest, Mapping):
        return [f"{label}_manifest_missing"]
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return [f"{label}_files_missing"]
    reasons: list[str] = []
    if manifest.get("file_count") != len(files):
        reasons.append(f"{label}_file_count_mismatch")
    if manifest.get("aggregate_sha256") != _canonical_json_sha256(files):
        reasons.append(f"{label}_aggregate_hash_mismatch")
    seen_paths: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            reasons.append(f"{label}_file_entry_invalid")
            continue
        reference = item.get("path")
        path = _resolve_reference(repo, reference)
        if path is None:
            reasons.append(f"{label}_file_path_missing")
            continue
        normalized = str(path)
        if normalized in seen_paths:
            reasons.append(f"{label}_duplicate_file_path")
        seen_paths.add(normalized)
        if not path.is_file():
            reasons.append(f"{label}_file_missing")
            continue
        if item.get("sha256") != _sha256(path):
            reasons.append(f"{label}_file_hash_mismatch")
        byte_size = item.get("byte_size")
        if byte_size is not None and byte_size != path.stat().st_size:
            reasons.append(f"{label}_file_size_mismatch")
    return reasons


def _read_csv_keys(path: Path, label: str) -> tuple[pd.DataFrame, list[str]]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return pd.DataFrame(), [f"{label}_csv_unreadable:{type(exc).__name__}"]
    required = {"season", "round", "target"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return frame, [f"{label}_columns_missing:{','.join(missing)}"]
    if frame.empty:
        return frame, [f"{label}_empty"]
    return frame, []


def _validate_run_manifest(repo: Path, manifest_path: Path) -> tuple[Optional[EvidenceSource], list[str]]:
    payload = _load_json(manifest_path)
    reasons: list[str] = []
    run_id = payload.get("run_id")
    if payload.get("schema_version") != ROLLING_BACKTEST_SCHEMA_VERSION:
        reasons.append("schema_version_invalid")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        reasons.append("run_id_invalid")
        run_id = manifest_path.parent.name
    elif run_id != manifest_path.parent.name:
        reasons.append("run_id_directory_mismatch")
    if not payload.get("started_at") or not payload.get("completed_at"):
        reasons.append("run_completion_timestamps_missing")

    artifact_contract = payload.get("artifact_contract")
    if not isinstance(artifact_contract, Mapping):
        reasons.append("artifact_contract_missing")
    else:
        if artifact_contract.get("immutable_paths") is not True:
            reasons.append("immutable_artifact_contract_missing")
        if artifact_contract.get("write_mode") != "exclusive_create":
            reasons.append("exclusive_write_contract_missing")
        declared_root = _resolve_reference(repo, artifact_contract.get("run_root"))
        if declared_root != manifest_path.parent.resolve():
            reasons.append("run_root_mismatch")

    git = payload.get("git")
    if not isinstance(git, Mapping) or not git.get("head_sha") or not isinstance(git.get("dirty"), bool):
        reasons.append("git_provenance_incomplete")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or not runtime.get("python_version") or not runtime.get("packages"):
        reasons.append("runtime_provenance_incomplete")
    training_protocol = payload.get("training_protocol")
    if not isinstance(training_protocol, Mapping):
        reasons.append("training_protocol_missing")
    elif training_protocol.get("target_round_allowed_in_training") is not False:
        reasons.append("target_round_training_guard_missing")

    reasons.extend(_validate_file_manifest(repo, payload.get("implementation"), "implementation"))
    reasons.extend(_validate_file_manifest(repo, payload.get("configuration_files"), "configuration"))
    run_configuration = payload.get("run_configuration")
    if not isinstance(run_configuration, Mapping):
        reasons.append("run_configuration_missing")
        run_config_payload: Mapping[str, Any] = {}
    else:
        run_config_payload_raw = run_configuration.get("payload")
        run_config_payload = run_config_payload_raw if isinstance(run_config_payload_raw, Mapping) else {}
        if not run_config_payload:
            reasons.append("run_configuration_payload_missing")
        if run_configuration.get("sha256") != _canonical_json_sha256(run_config_payload):
            reasons.append("run_configuration_hash_mismatch")

    try:
        season = int(run_config_payload.get("year"))
        rounds = sorted({int(value) for value in run_config_payload.get("rounds", [])})
    except (TypeError, ValueError):
        season, rounds = 0, []
    if season <= 0 or not rounds:
        reasons.append("run_round_scope_missing")
    expected_keys = {(season, round_number, target) for round_number in rounds for target in ("qualifying", "race")}

    point_records = payload.get("point_in_time_by_round")
    point_keys: set[tuple[int, int, str]] = set()
    if not isinstance(point_records, list):
        reasons.append("point_in_time_matrix_missing")
    else:
        for item in point_records:
            if not isinstance(item, Mapping):
                reasons.append("point_in_time_entry_invalid")
                continue
            try:
                key = (int(item.get("target_season")), int(item.get("target_round")), str(item.get("target")))
            except (TypeError, ValueError):
                reasons.append("point_in_time_key_invalid")
                continue
            if key in point_keys:
                reasons.append("point_in_time_duplicate")
            point_keys.add(key)
            if item.get("horizon_evidence_complete") is not True:
                reasons.append("point_in_time_horizon_incomplete")
            if item.get("target_round_allowed_in_training") is not False:
                reasons.append("point_in_time_training_guard_missing")
        if point_keys != expected_keys:
            reasons.append("point_in_time_scope_mismatch")

    data_by_round = payload.get("input_data_by_round")
    if not isinstance(data_by_round, Mapping) or set(data_by_round) != {str(value) for value in rounds}:
        reasons.append("input_data_round_scope_mismatch")
    else:
        for round_number in rounds:
            reasons.extend(
                _validate_file_manifest(
                    repo,
                    data_by_round.get(str(round_number)),
                    f"input_data_round_{round_number}",
                )
            )

    artifacts = payload.get("artifacts")
    prediction_paths: dict[tuple[int, int, str], Path] = {}
    selections_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    seen_artifacts: set[str] = set()
    if not isinstance(artifacts, list) or not artifacts:
        reasons.append("artifacts_missing")
    else:
        for item in artifacts:
            if not isinstance(item, Mapping):
                reasons.append("artifact_entry_invalid")
                continue
            path = _resolve_reference(repo, item.get("path"))
            if path is None:
                reasons.append("artifact_path_missing")
                continue
            if str(path) in seen_artifacts:
                reasons.append("artifact_path_duplicate")
            seen_artifacts.add(str(path))
            if not path.is_file():
                reasons.append("artifact_file_missing")
                continue
            if item.get("sha256") != _sha256(path):
                reasons.append("artifact_hash_mismatch")
            if item.get("byte_size") != path.stat().st_size:
                reasons.append("artifact_size_mismatch")
            artifact_type = item.get("artifact_type")
            if artifact_type == "rolling_backtest_selections":
                if selections_path is not None:
                    reasons.append("selection_artifact_ambiguous")
                selections_path = path
            elif artifact_type == "rolling_backtest_summary":
                if summary_path is not None:
                    reasons.append("summary_artifact_ambiguous")
                summary_path = path
            elif artifact_type in {"qualifying_prediction", "race_prediction"}:
                prediction = _load_json(path)
                config = prediction.get("config")
                point_in_time = prediction.get("point_in_time")
                if prediction.get("schema_version") != ROLLING_BACKTEST_SCHEMA_VERSION:
                    reasons.append("prediction_schema_invalid")
                if prediction.get("run_id") != run_id:
                    reasons.append("prediction_run_id_mismatch")
                if not isinstance(prediction.get("rows"), list) or not prediction.get("rows"):
                    reasons.append("prediction_rows_missing")
                if not isinstance(prediction.get("all_prediction_rows"), list) or not prediction.get("all_prediction_rows"):
                    reasons.append("prediction_full_field_rows_missing")
                if not isinstance(config, Mapping):
                    reasons.append("prediction_config_missing")
                    continue
                if prediction.get("config_sha256") != _canonical_json_sha256(config):
                    reasons.append("prediction_config_hash_mismatch")
                if item.get("config_sha256") != prediction.get("config_sha256"):
                    reasons.append("prediction_manifest_config_hash_mismatch")
                if not isinstance(point_in_time, Mapping) or point_in_time.get("horizon_evidence_complete") is not True:
                    reasons.append("prediction_horizon_incomplete")
                if item.get("point_in_time") != point_in_time:
                    reasons.append("prediction_point_in_time_mismatch")
                try:
                    key = (int(config.get("year")), int(config.get("round_number")), str(config.get("mode")))
                except (TypeError, ValueError):
                    reasons.append("prediction_key_invalid")
                    continue
                if isinstance(point_in_time, Mapping):
                    try:
                        point_key = (
                            int(point_in_time.get("target_season")),
                            int(point_in_time.get("target_round")),
                            str(point_in_time.get("target")),
                        )
                    except (TypeError, ValueError):
                        point_key = (0, 0, "")
                    if point_key != key:
                        reasons.append("prediction_point_in_time_key_mismatch")
                if key in prediction_paths:
                    reasons.append("prediction_key_duplicate")
                prediction_paths[key] = path
    if selections_path is None:
        reasons.append("selection_artifact_missing")
    if summary_path is None:
        reasons.append("summary_artifact_missing")
    if set(prediction_paths) != expected_keys:
        reasons.append("prediction_artifact_scope_mismatch")

    if selections_path is not None:
        selections, csv_reasons = _read_csv_keys(selections_path, "selections")
        reasons.extend(csv_reasons)
        if not csv_reasons:
            required_selection_columns = {
                "information_cutoff",
                "market",
                "selection",
                "model_rank",
                "model_probability_pct",
                "result",
                "train_seasons_used",
                "rolling_2026_rounds_used",
                "experiment_arm",
            }
            missing_selection_columns = sorted(required_selection_columns.difference(selections.columns))
            if missing_selection_columns:
                reasons.append(f"selection_columns_incomplete:{','.join(missing_selection_columns)}")
            actual_keys = {
                (int(row["season"]), int(row["round"]), str(row["target"]))
                for _, row in selections[["season", "round", "target"]].drop_duplicates().iterrows()
            }
            if actual_keys != expected_keys:
                reasons.append("selection_scope_mismatch")
            declared_arm = training_protocol.get("experiment_arm") if isinstance(training_protocol, Mapping) else None
            if (
                declared_arm
                and "experiment_arm" in selections.columns
                and set(selections["experiment_arm"].dropna().astype(str)) != {str(declared_arm)}
            ):
                reasons.append("selection_experiment_arm_mismatch")
    if summary_path is not None:
        summary, csv_reasons = _read_csv_keys(summary_path, "summary")
        reasons.extend(csv_reasons)
        if not csv_reasons:
            required_summary_columns = {
                "information_cutoff",
                "rows_predicted",
                "rows_actual",
                "field_coverage",
                "experiment_arm",
            }
            missing_summary_columns = sorted(required_summary_columns.difference(summary.columns))
            if missing_summary_columns:
                reasons.append(f"summary_columns_incomplete:{','.join(missing_summary_columns)}")
            actual_keys = {
                (int(row["season"]), int(row["round"]), str(row["target"]))
                for _, row in summary[["season", "round", "target"]].drop_duplicates().iterrows()
            }
            if actual_keys != expected_keys:
                reasons.append("summary_scope_mismatch")
            declared_arm = training_protocol.get("experiment_arm") if isinstance(training_protocol, Mapping) else None
            if (
                declared_arm
                and "experiment_arm" in summary.columns
                and set(summary["experiment_arm"].dropna().astype(str)) != {str(declared_arm)}
            ):
                reasons.append("summary_experiment_arm_mismatch")

    reasons = list(dict.fromkeys(reasons))
    if reasons or selections_path is None or summary_path is None:
        return None, reasons
    return (
        EvidenceSource(
            source_kind="run_manifest",
            run_id=str(run_id),
            selections_path=selections_path,
            summary_path=summary_path,
            prediction_paths=prediction_paths,
            manifest_path=manifest_path.resolve(),
            manifest_sha256=_sha256(manifest_path),
            manifest=payload,
            frozen_complete=True,
        ),
        [],
    )


def _artifact_path(repo: Path, season: int, round_number: int, target: str) -> Path:
    name = "qualifying_prediction.json" if target == "qualifying" else "postqual_race_prediction.json"
    return repo / "artifacts" / "backtests" / "f1" / "rolling_2026" / str(season) / f"round_{round_number:02d}" / name


def _display_path(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _legacy_evidence_source(repo: Path) -> EvidenceSource:
    selections_path = repo / "artifacts" / "reports" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Selections.csv"
    summary_path = repo / "artifacts" / "reports" / "f1" / "docs" / "F1_2026_Rolling_Backtest_Summary.csv"
    selections, selection_reasons = _read_csv_keys(selections_path, "legacy_selections")
    _, summary_reasons = _read_csv_keys(summary_path, "legacy_summary")
    reasons = [*selection_reasons, *summary_reasons]
    prediction_paths: dict[tuple[int, int, str], Path] = {}
    if not selection_reasons:
        for _, row in selections[["season", "round", "target"]].drop_duplicates().iterrows():
            key = (int(row["season"]), int(row["round"]), str(row["target"]))
            path = _artifact_path(repo, *key)
            if not path.is_file() or not _load_json(path):
                reasons.append(f"legacy_prediction_missing:{key[0]}:{key[1]}:{key[2]}")
            else:
                prediction_paths[key] = path.resolve()
    if reasons:
        raise EvidenceDiscoveryError("Legacy read-only evidence is incomplete: " + "; ".join(reasons))
    return EvidenceSource(
        source_kind="legacy_read_only_unmanifested",
        run_id="legacy_flat_unmanifested",
        selections_path=selections_path.resolve(),
        summary_path=summary_path.resolve(),
        prediction_paths=prediction_paths,
        manifest_path=None,
        manifest_sha256=None,
        manifest={},
        frozen_complete=False,
    )


def _discover_evidence_source(
    repo: Path,
    *,
    run_id: Optional[str] = None,
    legacy_read_only: bool = False,
) -> EvidenceSource:
    if run_id and legacy_read_only:
        raise EvidenceDiscoveryError("--run-id and --legacy-read-only are mutually exclusive")
    if legacy_read_only:
        return _legacy_evidence_source(repo)

    runs_root = repo / "artifacts" / "backtests" / "f1" / "rolling_2026" / "runs"
    if run_id is not None:
        if run_id in {"", ".", ".."} or not RUN_ID_PATTERN.fullmatch(run_id):
            raise EvidenceDiscoveryError("Invalid --run-id")
        run_dir = runs_root / run_id
        manifest_path = run_dir / "run_manifest.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            raise EvidenceDiscoveryError(f"Run {run_id!r} has no complete run_manifest.json")
        source, reasons = _validate_run_manifest(repo, manifest_path)
        if source is None:
            raise EvidenceDiscoveryError(f"Run {run_id!r} is incomplete or invalid: {'; '.join(reasons)}")
        return source

    if not runs_root.is_dir():
        raise EvidenceDiscoveryError(
            "No run-scoped rolling artifacts found. Use --legacy-read-only only for explicit inspection of the unmanifested layout.",
        )
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    if not run_dirs:
        raise EvidenceDiscoveryError(
            "No run-scoped rolling artifacts found. Use --legacy-read-only only for explicit inspection of the unmanifested layout.",
        )
    valid: list[EvidenceSource] = []
    invalid: list[str] = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            invalid.append(f"{run_dir.name}:missing_run_manifest")
            continue
        source, reasons = _validate_run_manifest(repo, manifest_path)
        if source is None:
            invalid.append(f"{run_dir.name}:{','.join(reasons)}")
        else:
            valid.append(source)
    if invalid:
        raise EvidenceDiscoveryError(
            "Incomplete or invalid run directories prevent automatic selection: " + "; ".join(invalid),
        )
    if len(valid) != 1:
        available = ", ".join(source.run_id for source in valid) or "none"
        raise EvidenceDiscoveryError(
            f"Expected exactly one complete frozen run, found {len(valid)} ({available}); select one with --run-id.",
        )
    return valid[0]


def _assert_source_still_frozen(repo: Path, source: EvidenceSource) -> None:
    if not source.frozen_complete:
        return
    if source.manifest_path is None or source.manifest_sha256 is None:
        raise EvidenceDiscoveryError("Frozen evidence source is missing manifest provenance")
    if not source.manifest_path.is_file() or _sha256(source.manifest_path) != source.manifest_sha256:
        raise EvidenceDiscoveryError("Run manifest changed after discovery")
    validated, reasons = _validate_run_manifest(repo, source.manifest_path)
    if validated is None:
        raise EvidenceDiscoveryError("Frozen evidence changed after discovery: " + "; ".join(reasons))
    if validated.run_id != source.run_id or validated.manifest_sha256 != source.manifest_sha256:
        raise EvidenceDiscoveryError("Frozen evidence identity changed after discovery")


def _artifact_index(
    repo: Path,
    selections: pd.DataFrame,
    source: EvidenceSource,
) -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    keys = selections[["season", "round", "target"]].drop_duplicates()
    for _, row in keys.iterrows():
        season = int(row["season"])
        round_number = int(row["round"])
        target = str(row["target"])
        path = source.prediction_paths.get((season, round_number, target))
        if path is None:
            raise EvidenceDiscoveryError(
                f"Selected evidence source has no prediction artifact for {(season, round_number, target)}",
            )
        payload = _load_json(path)
        out[(season, round_number, target)] = {
            "path": _display_path(repo, path),
            "exists": path.exists(),
            "sha256": _sha256(path) if path.exists() else "",
            "generated_at_utc": payload.get("generated_at"),
            "model_name_artifact": payload.get("model_name"),
            "model_family_artifact": payload.get("model_family"),
            "config": payload.get("config", {}),
            "config_sha256": payload.get("config_sha256"),
            "point_in_time": payload.get("point_in_time", {}),
            "notes": payload.get("notes", []),
            "run_id": source.run_id,
            "run_manifest_sha256": source.manifest_sha256,
        }
    return out


def _fair_odds(probability_pct: object) -> Optional[float]:
    try:
        p = float(probability_pct) / 100.0
    except (TypeError, ValueError):
        return None
    if p <= 0.0:
        return None
    return round(1.0 / p, 4)


def _build_selection_log(
    repo: Path,
    selections: pd.DataFrame,
    summary: pd.DataFrame,
    source: EvidenceSource,
    git_head: str,
    git_dirty: bool,
    code_hash: str,
) -> pd.DataFrame:
    artifacts = _artifact_index(repo, selections, source)
    run_git = source.manifest.get("git") if isinstance(source.manifest.get("git"), Mapping) else {}
    implementation = (
        source.manifest.get("implementation")
        if isinstance(source.manifest.get("implementation"), Mapping)
        else {}
    )
    training_protocol = (
        source.manifest.get("training_protocol")
        if isinstance(source.manifest.get("training_protocol"), Mapping)
        else {}
    )
    if training_protocol.get("same_season_only") is True:
        split_description = (
            "Same-season walk-forward: each target event trains only on completed target-season rounds "
            "strictly before the target round."
        )
    elif training_protocol:
        split_description = (
            "Explicit non-primary transfer arm plus target-season walk-forward updates; see run manifest "
            "training_protocol and per-round input manifests."
        )
    else:
        split_description = "Legacy unmanifested training split; inspect artifact config per selection."
    summary_lookup = {
        (int(row["season"]), int(row["round"]), str(row["target"])): row
        for _, row in summary.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, row in selections.iterrows():
        season = int(row["season"])
        round_number = int(row["round"])
        target = str(row["target"])
        artifact = artifacts[(season, round_number, target)]
        config = artifact.get("config", {}) if isinstance(artifact.get("config"), dict) else {}
        summary_row = summary_lookup.get((season, round_number, target), {})
        probability_pct = row.get("model_probability_pct")
        fair_odds = _fair_odds(probability_pct)
        fair_probability = round(100.0 / fair_odds, 3) if fair_odds else None
        rolling_rounds = row.get("rolling_2026_rounds_used")
        if pd.isna(rolling_rounds):
            rolling_rounds = ""
        rows.append(
            {
                "selection_id": hashlib.sha256(
                    "|".join(
                        [
                            source.run_id,
                            str(season),
                            str(round_number),
                            target,
                            str(row.get("market")),
                            str(row.get("selection")),
                            str(row.get("model_rank")),
                        ]
                    ).encode("utf-8")
                ).hexdigest()[:16],
                "season": season,
                "round": round_number,
                "grand_prix": row.get("grand_prix"),
                "target": target,
                "market": row.get("market"),
                "selection": row.get("selection"),
                "driver_id": row.get("driver_id"),
                "information_cutoff": row.get("information_cutoff"),
                "model_rank": row.get("model_rank"),
                "model_probability_pct": probability_pct,
                "model_fair_odds_decimal": fair_odds,
                "model_fair_probability_pct_recomputed": fair_probability,
                "prediction_score": row.get("prediction_score"),
                "actual_position": row.get("actual_position"),
                "result": row.get("result"),
                "settled_binary_result": 1 if row.get("result") == "hit" else 0 if row.get("result") == "miss" else None,
                "selected_model_name": artifact.get("model_name_artifact") or row.get("model_name"),
                "selected_model_family": artifact.get("model_family_artifact") or row.get("model_family"),
                "forced_model_flag": config.get("f1_model"),
                "cv_top_candidate": summary_row.get("cv_top_candidate") if isinstance(summary_row, pd.Series) else None,
                "cv_top_composite": summary_row.get("cv_top_composite") if isinstance(summary_row, pd.Series) else None,
                "prediction_artifact": artifact.get("path"),
                "prediction_artifact_sha256": artifact.get("sha256"),
                "prediction_config_sha256": artifact.get("config_sha256"),
                "prediction_point_in_time": json.dumps(
                    artifact.get("point_in_time", {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "evidence_source_kind": source.source_kind,
                "evidence_run_id": source.run_id,
                "evidence_run_frozen_complete": source.frozen_complete,
                "run_manifest": _display_path(repo, source.manifest_path) if source.manifest_path else None,
                "run_manifest_sha256": source.manifest_sha256,
                "prediction_run_git_head": run_git.get("head_sha"),
                "prediction_run_git_dirty": run_git.get("dirty"),
                "prediction_source_tree_sha256": implementation.get("aggregate_sha256"),
                "selection_generated_at_utc": artifact.get("generated_at_utc"),
                "selection_timestamp_status": "post_hoc_rerun_timestamp_not_pre_market",
                "pre_market_timestamp_verified": False,
                "market_close_verified": False,
                "available_market_odds_decimal": None,
                "bookmaker_or_exchange": None,
                "market_odds_timestamp_utc": None,
                "odds_source_status": "not_captured_before_market_close",
                "stake_rule": FORWARD_STAKE_RULE,
                "paper_stake": None,
                "actual_stake": 0.0,
                "settled_pnl": None,
                "pnl_status": "not_settled_odds_unavailable",
                "base_train_seasons": row.get("base_train_seasons"),
                "train_seasons_used": row.get("train_seasons_used"),
                "rolling_2026_rounds_used": rolling_rounds,
                "current_year_weight_multiplier": row.get("current_year_weight_multiplier"),
                "experiment_arm": row.get("experiment_arm") or training_protocol.get("experiment_arm"),
                "train_validation_holdout_split": split_description,
                "evidence_pack_git_head": git_head,
                "evidence_pack_git_dirty": git_dirty,
                "evidence_pack_source_code_sha256": code_hash,
                "git_head": git_head,
                "git_dirty": git_dirty,
                "source_code_sha256": code_hash,
            }
        )
    return pd.DataFrame(rows)


def _build_checklist() -> pd.DataFrame:
    rows = [
        ("exact model version used for each selection", "yes", "selection_log", "selected_model_name/family, forced_model_flag, artifact hash, code hash, git state"),
        ("timestamped selections before market close", "no", "selection_log", "Current artifacts are post-hoc rerun timestamps, not pre-market immutable logs."),
        ("information cutoff", "yes", "selection_log", "post-practice for qualifying, post-qualifying for race."),
        ("model probability and fair odds", "yes", "selection_log", "fair odds = 1 / model probability."),
        ("available market odds at selection time", "no", "selection_log", "No bookmaker/exchange odds snapshots were captured before market close."),
        ("bookmaker/exchange and odds timestamp", "no", "selection_log", "Requires forward odds capture or paid historical odds archive."),
        ("stake sizing rule and stake", "partial", "selection_log", "Forward-test staking rule supplied; historical actual/paper stake unavailable without odds-backed log."),
        ("result and settled P&L", "partial", "selection_log", "Hit/miss settled against race/qualifying results; P&L unavailable without market odds and stake."),
        ("all skipped/excluded/edited selections", "yes", "skipped_excluded_edited", "Exporter rows are unedited; betting rows are not accepted because odds are missing."),
        ("raw data/features available at prediction time", "partial", "raw_data_manifest", "Raw local session files are listed. The artifacts were generated post-hoc and are not immutable pre-cutoff feature snapshots."),
        ("clear train/validation/holdout split", "yes", "methodology", "Selected run manifest records the experiment arm, train seasons, strict prior-round guard, and point-in-time matrix."),
        ("reconcile five paper-traded email selections", "partial", "paper_trade_reconciliation", "The prior email's five picks are not present in local artifacts; canonical top race selections are listed."),
    ]
    return pd.DataFrame(rows, columns=["jordan_request", "status", "sheet", "evidence_or_gap"])


def _build_methodology(
    repo: Path,
    source: EvidenceSource,
    git_head: str,
    git_dirty: bool,
    code_hash: str,
) -> pd.DataFrame:
    run_configuration = (
        source.manifest.get("run_configuration")
        if isinstance(source.manifest.get("run_configuration"), Mapping)
        else {}
    )
    run_config_payload = (
        run_configuration.get("payload")
        if isinstance(run_configuration.get("payload"), Mapping)
        else {}
    )
    training_protocol = (
        source.manifest.get("training_protocol")
        if isinstance(source.manifest.get("training_protocol"), Mapping)
        else {}
    )
    run_git = source.manifest.get("git") if isinstance(source.manifest.get("git"), Mapping) else {}
    implementation = (
        source.manifest.get("implementation")
        if isinstance(source.manifest.get("implementation"), Mapping)
        else {}
    )
    return pd.DataFrame(
        [
            ("pack_generated_at_utc", _utc_now()),
            ("repo", str(repo)),
            ("evidence_source_kind", source.source_kind),
            ("evidence_run_id", source.run_id),
            ("evidence_run_frozen_complete", str(source.frozen_complete)),
            ("run_manifest", _display_path(repo, source.manifest_path) if source.manifest_path else "unavailable_legacy"),
            ("run_manifest_sha256", source.manifest_sha256 or "unavailable_legacy"),
            ("prediction_run_git_head", run_git.get("head_sha")),
            ("prediction_run_git_dirty", run_git.get("dirty")),
            ("prediction_source_tree_sha256", implementation.get("aggregate_sha256")),
            ("evidence_pack_git_head", git_head),
            ("evidence_pack_git_dirty", str(git_dirty)),
            ("evidence_pack_source_code_sha256", code_hash),
            ("selection_csv_source", _display_path(repo, source.selections_path)),
            ("summary_csv_source", _display_path(repo, source.summary_path)),
            ("experiment_arm", training_protocol.get("experiment_arm") or "legacy_unmanifested"),
            ("same_season_only", training_protocol.get("same_season_only")),
            ("train_seasons_used", json.dumps(training_protocol.get("train_seasons_used", []))),
            ("transfer_train_seasons", json.dumps(training_protocol.get("transfer_train_seasons", []))),
            ("current_year_weighting", training_protocol.get("current_season_weight_multiplier")),
            ("target_round_allowed_in_training", training_protocol.get("target_round_allowed_in_training")),
            ("rounds", json.dumps(run_config_payload.get("rounds", []))),
            ("qualifying_information_horizon", run_config_payload.get("qualifying_information_horizon")),
            ("race_information_horizon", run_config_payload.get("race_information_horizon")),
            ("ranking_evidence_summary", "See round_summary and policy_summary; no stale fixed-round headline is embedded."),
            ("betting_edge_status", "Not proven by the current historical pack because odds, odds timestamps, stake, and settled P&L were not captured before market close."),
        ],
        columns=["field", "value"],
    )


def _raw_manifest(repo: Path, source: EvidenceSource) -> pd.DataFrame:
    declared_by_path: dict[str, Mapping[str, Any]] = {}
    if source.frozen_complete:
        input_data = source.manifest.get("input_data_by_round")
        if isinstance(input_data, Mapping):
            for round_manifest in input_data.values():
                files = round_manifest.get("files") if isinstance(round_manifest, Mapping) else None
                if not isinstance(files, list):
                    continue
                for item in files:
                    if not isinstance(item, Mapping):
                        continue
                    path = _resolve_reference(repo, item.get("path"))
                    if path is not None:
                        declared_by_path[str(path)] = item
        paths = sorted(
            (Path(value) for value in declared_by_path),
            key=lambda item: item.as_posix(),
        )
    else:
        root = repo / "data" / "f1" / "raw" / "weekends" / "2026"
        paths = sorted(root.glob("round_*/*")) if root.exists() else []
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        rel = _display_path(repo, path)
        filename = path.name
        if "practice" in filename:
            cutoff_role = "feature_input_for_post_practice_and_post_qualifying"
        elif "qualifying" in filename:
            cutoff_role = "feature_input_for_post_qualifying_race; settlement_for_qualifying"
        elif "race" in filename:
            cutoff_role = "settlement_only_for_post_race_evaluation"
        elif filename == "weekend_metadata.json":
            cutoff_role = "raw_file_manifest"
        else:
            cutoff_role = "unknown"
        rows.append(
            {
                "evidence_source_kind": source.source_kind,
                "evidence_run_id": source.run_id,
                "run_manifest_sha256": source.manifest_sha256,
                "file": rel,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "declared_sha256": declared_by_path.get(str(path), {}).get("sha256"),
                "hash_verified": (
                    declared_by_path[str(path)].get("sha256") == _sha256(path)
                    if str(path) in declared_by_path
                    else False
                ),
                "cutoff_role": cutoff_role,
                "timestamp_status": (
                    "frozen_run_manifest_input"
                    if source.frozen_complete
                    else "legacy_local_file_timestamp_not_pre_market_evidence"
                ),
            }
        )
    return pd.DataFrame(rows)


def _artifact_manifest(repo: Path, source: EvidenceSource) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    declared_artifacts = source.manifest.get("artifacts") if isinstance(source.manifest, Mapping) else None
    declared_by_path: dict[str, Mapping[str, Any]] = {}
    if isinstance(declared_artifacts, list):
        for item in declared_artifacts:
            if not isinstance(item, Mapping):
                continue
            path = _resolve_reference(repo, item.get("path"))
            if path is not None:
                declared_by_path[str(path)] = item
    paths = [*source.prediction_paths.values(), source.selections_path, source.summary_path]
    if source.manifest_path is not None:
        paths.append(source.manifest_path)
    for path in sorted({item.resolve() for item in paths if item.exists()}, key=lambda item: item.as_posix()):
        declared = declared_by_path.get(str(path), {})
        current_hash = _sha256(path)
        rows.append(
            {
                "evidence_source_kind": source.source_kind,
                "evidence_run_id": source.run_id,
                "run_manifest_sha256": source.manifest_sha256,
                "artifact": _display_path(repo, path),
                "artifact_type": (
                    "run_manifest" if source.manifest_path is not None and path == source.manifest_path else declared.get("artifact_type")
                ),
                "size_bytes": path.stat().st_size,
                "sha256": current_hash,
                "declared_sha256": source.manifest_sha256 if path == source.manifest_path else declared.get("sha256"),
                "hash_verified": (
                    True if path == source.manifest_path else declared.get("sha256") == current_hash if declared else False
                ),
                "modified_at_local": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _source_manifest(repo: Path) -> pd.DataFrame:
    rows = []
    for rel in SOURCE_FILES:
        path = repo / rel
        rows.append(
            {
                "source_file": rel,
                "exists": path.exists(),
                "sha256": _sha256(path) if path.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def _strategy_v2_selected(selection_log: pd.DataFrame) -> pd.DataFrame:
    conservative = selection_log[
        (selection_log["target"] == "race")
        & (selection_log["market"] == "top10")
        & (selection_log["model_rank"] == 1)
    ].copy()
    conservative["strategy_name"] = "conservative_one_race_top10_per_gp"
    conservative["strategy_rule"] = "Post-qualifying race top10 only; take the model rank-1 driver per GP."

    high_conf = selection_log[
        (
            (selection_log["target"] == "qualifying")
            & (selection_log["market"] == "top10")
            & (selection_log["model_rank"] <= 6)
        )
        | (
            (selection_log["target"] == "race")
            & (selection_log["market"] == "top10")
            & (selection_log["model_probability_pct"] >= 85.0)
        )
    ].copy()
    high_conf["strategy_name"] = "high_confidence_top10_ranking"
    high_conf["strategy_rule"] = (
        "Top10 market only; qualifying rank<=6 after practice, race probability>=85% after qualifying."
    )

    out = pd.concat([conservative, high_conf], ignore_index=True)
    out["strategy_pnl_status"] = "ranking_outcome_only_no_odds_pnl"
    out["strategy_use_status"] = "candidate_forward_test_rule_not_historical_betting_edge"
    return out


def _strategy_v2_summary(strategy_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_name, group in strategy_rows.groupby("strategy_name", sort=False):
        settled = group[group["result"].isin(["hit", "miss"])].copy()
        rows.append(
            {
                "strategy_name": strategy_name,
                "strategy_rule": group["strategy_rule"].iloc[0],
                "selections": int(len(group)),
                "settled_selections": int(len(settled)),
                "hits": int((settled["result"] == "hit").sum()),
                "misses": int((settled["result"] == "miss").sum()),
                "hit_rate_pct": round(float((settled["result"] == "hit").mean()) * 100.0, 2) if not settled.empty else None,
                "markets_used": ",".join(sorted(set(str(v) for v in group["market"].dropna()))),
                "targets_used": ",".join(sorted(set(str(v) for v in group["target"].dropna()))),
                "pnl_status": "not_odds_backed_no_pnl_claim",
                "interpretation": "This improves ranking/outcome presentation only; it is not evidence of betting profitability without odds.",
            }
        )
    return pd.DataFrame(rows)


def _reconciliation(selection_log: pd.DataFrame) -> pd.DataFrame:
    top_race = selection_log[
        (selection_log["target"] == "race")
        & (selection_log["market"] == "top10")
        & (selection_log["model_rank"] == 1)
    ].copy()
    top_race = top_race.sort_values(["season", "round"])
    rows: list[dict[str, Any]] = []
    for _, row in top_race.iterrows():
        rows.append(
            {
                "season": row["season"],
                "round": row["round"],
                "grand_prix": row["grand_prix"],
                "prior_email_selection": None,
                "prior_email_market": None,
                "canonical_csv_selection": row["selection"],
                "canonical_csv_market": row["market"],
                "canonical_csv_model_rank": row["model_rank"],
                "canonical_csv_probability_pct": row["model_probability_pct"],
                "canonical_csv_result": row["result"],
                "reconciliation_status": "prior_email_pick_not_available_in_local_artifacts",
                "recommended_action": (
                    "Use this top10 canonical row for a cleaner five-selection ranking view, "
                    "or paste the prior email pick list into this sheet before sending."
                ),
            }
        )
    return pd.DataFrame(rows)


def _skipped_sheet(selection_log: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "category": "selection_export",
                "count": int(len(selection_log)),
                "status": "included_unedited",
                "reason": "All deterministic rolling selection rows from the source CSV are included in selection_log.",
            },
            {
                "category": "betting_log",
                "count": int(len(selection_log)),
                "status": "not_bet",
                "reason": "No pre-market bookmaker/exchange odds snapshots were captured, so no odds-backed stake or P&L is claimed.",
            },
        ]
    )


def _high_confidence_top10(selection_log: pd.DataFrame, threshold_pct: float = 85.0) -> pd.DataFrame:
    high_conf = selection_log[
        (selection_log["market"] == "top10")
        & (pd.to_numeric(selection_log["model_probability_pct"], errors="coerce") >= float(threshold_pct))
    ].copy()
    high_conf["highlight_policy"] = f"top10_model_probability_gte_{threshold_pct:g}_pct"
    return high_conf.sort_values(["target", "season", "round", "model_rank"], kind="mergesort")


def _policy_summary(selection_log: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policies = [
        ("all_exported_selections", selection_log),
        ("all_top10", selection_log[selection_log["market"] == "top10"]),
        ("high_conf_top10_p_ge_85", _high_confidence_top10(selection_log, 85.0)),
        ("high_conf_top10_p_ge_90", _high_confidence_top10(selection_log, 90.0)),
    ]
    for policy_name, frame in policies:
        if frame.empty:
            continue
        for target in ["all", "qualifying", "race"]:
            scoped = frame if target == "all" else frame[frame["target"] == target]
            if scoped.empty:
                continue
            hits = pd.to_numeric(scoped["settled_binary_result"], errors="coerce")
            rows.append(
                {
                    "policy": policy_name,
                    "target": target,
                    "selection_count": int(hits.notna().sum()),
                    "hits": int(hits.fillna(0).sum()),
                    "hit_rate_pct": round(float(hits.mean()) * 100.0, 2) if hits.notna().any() else None,
                    "positioning": (
                        "strongest_forward_test_candidate"
                        if policy_name == "high_conf_top10_p_ge_85"
                        else "supporting_context"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _executive_summary(selection_log: pd.DataFrame) -> pd.DataFrame:
    policy = _policy_summary(selection_log)
    high_all = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "all")].iloc[0]
    high_qual = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "qualifying")].iloc[0]
    high_race = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "race")].iloc[0]
    return pd.DataFrame(
        [
            ("primary_read", "The strongest defensible signal is high-confidence top-10/order prediction, not race-winner prediction."),
            (
                "highlight_policy",
                "Market=top10 and model_probability_pct >= 85. This is simple, model-native, and suitable for forward testing.",
            ),
            (
                "overall_high_conf_top10",
                f"{int(high_all['hits'])}/{int(high_all['selection_count'])} hits ({float(high_all['hit_rate_pct']):.2f}%).",
            ),
            (
                "qualifying_high_conf_top10",
                f"{int(high_qual['hits'])}/{int(high_qual['selection_count'])} hits ({float(high_qual['hit_rate_pct']):.2f}%).",
            ),
            (
                "race_high_conf_top10",
                f"{int(high_race['hits'])}/{int(high_race['selection_count'])} hits ({float(high_race['hit_rate_pct']):.2f}%).",
            ),
            (
                "betting_edge_boundary",
                "Still not an odds-backed P&L claim: market odds, timestamped odds, stake, and settlement P&L remain unavailable historically.",
            ),
            (
                "partner_next_step",
                "Forward test only the high-confidence top10 policy with immutable pre-event logging and timestamped odds.",
            ),
        ],
        columns=["field", "value"],
    )


def _forward_schema() -> pd.DataFrame:
    fields = [
        ("event_id", "2026_round_06_spanish_gp", "Stable event id."),
        ("selection_id", "sha256 hash", "Deterministic id over event/market/selection/cutoff/model."),
        ("selection_logged_at_utc", "2026-06-05T12:00:00Z", "Must be before market close."),
        ("information_cutoff", "post-practice | post-qualifying", "What information was allowed."),
        ("prediction_artifact_sha256", "hash", "Hash of prediction artifact frozen at selection time."),
        ("raw_feature_snapshot_sha256", "hash", "Hash of feature/input manifest frozen at selection time."),
        ("market", "winner | podium | top10", "Supported market."),
        ("selection", "Driver abbreviation/name", "Bet selection."),
        ("model_probability", "0.62", "Model probability at log time."),
        ("model_fair_odds_decimal", "1.613", "1 / model_probability."),
        ("bookmaker_or_exchange", "Betfair/Pinnacle/etc.", "Odds venue."),
        ("available_market_odds_decimal", "1.85", "Price available at log time."),
        ("odds_timestamp_utc", "2026-06-05T11:59:30Z", "Bookmaker/exchange odds timestamp."),
        ("stake_rule", FORWARD_STAKE_RULE, "Rule applied mechanically."),
        ("paper_stake", "10.00", "Paper stake assigned before event."),
        ("actual_stake", "0.00", "Live stake if any."),
        ("result", "pending | hit | miss | void", "Settlement result."),
        ("settled_pnl", "8.50", "P&L after settlement."),
        ("previous_record_hash", "hash", "Hash chain previous record."),
        ("record_hash", "hash", "Canonical hash of this record."),
    ]
    return pd.DataFrame(fields, columns=["field", "example", "description"])


def _odds_research() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "provider": "The Odds API",
                "url": "https://the-odds-api.com/",
                "relevance": "Live/upcoming odds and paid historical snapshots; supports bookmaker odds and decimal format.",
            },
            {
                "provider": "OddsJam",
                "url": "https://dev.oddsjam.com/odds-api",
                "relevance": "Real-time odds from many sportsbooks plus full historical odds database and line changes.",
            },
            {
                "provider": "Betfair Historical Data",
                "url": "https://support.developer.betfair.com/hc/en-us/articles/360002407732-What-data-is-provided-by-the-Historical-Data-service",
                "relevance": "Exchange historical market, price and settlement data suitable for backtesting after purchase/access.",
            },
            {
                "provider": "OddsBlaze Historical Odds",
                "url": "https://docs.oddsblaze.com/endpoints/historical_odds/",
                "relevance": "Historical odds endpoint with line movement history; requires API key.",
            },
        ]
    )


def _write_readme(
    path: Path,
    workbook_path: Path,
    selection_log: pd.DataFrame,
    source: EvidenceSource,
) -> None:
    policy = _policy_summary(selection_log)
    strategy = _strategy_v2_summary(_strategy_v2_selected(selection_log))
    conservative = strategy[strategy["strategy_name"] == "conservative_one_race_top10_per_gp"].iloc[0]
    high_all = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "all")].iloc[0]
    high_qual = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "qualifying")].iloc[0]
    high_race = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "race")].iloc[0]
    text = f"""# F1 Evidence Pack v3

Generated at: {_utc_now()}

Workbook: `{workbook_path.name}`

Evidence run: `{source.run_id}`

Run manifest SHA-256: `{source.manifest_sha256 or 'legacy_unavailable'}`

Evidence source: `{source.source_kind}`

## Strongest result

The most defensible commercial read is the high-confidence top-10 policy:

- Policy: `market=top10` and `model_probability_pct >= 85`
- Overall: {int(high_all['hits'])}/{int(high_all['selection_count'])} hits ({float(high_all['hit_rate_pct']):.2f}%)
- Qualifying: {int(high_qual['hits'])}/{int(high_qual['selection_count'])} hits ({float(high_qual['hit_rate_pct']):.2f}%)
- Race: {int(high_race['hits'])}/{int(high_race['selection_count'])} hits ({float(high_race['hit_rate_pct']):.2f}%)

The clean five-selection reconciliation now uses a race top-10 rule instead of the weaker winner market:

- Policy: post-qualifying race top-10, model rank 1 per GP
- Result: {int(conservative['hits'])}/{int(conservative['settled_selections'])} hits ({float(conservative['hit_rate_pct']):.2f}%)
- Important: actual outcomes are not edited; only the selected market/rule is changed.

## What this pack proves

- It reconciles the rolling 2026 prediction output into a cleaner diligence format.
- It provides exact prediction artifact hashes, source-code hash, model config, model probabilities, fair odds, results, and train/holdout methodology for every exported selection.
- It shows the ranking/prediction evidence clearly: this is evidence of ranking quality, especially top-10/order prediction.
- It includes `strategy_v2_selected`, which is the improved logical selection view. It filters to top-10/order markets where the model is strongest while preserving original hit/miss settlement.

## What this pack does not prove

- It does not prove a historical betting edge.
- The repo did not contain bookmaker/exchange odds snapshots captured before market close.
- Therefore historical available odds, odds timestamps, stake, and settled P&L are marked as unavailable rather than reconstructed.

## Recommended positioning to Jordan

Send this as a corrected evidence pack and state plainly that the previous file should be treated as ranking evidence, not an odds-backed betting log. The next serious step is the forward-test log: timestamp selections before market close, attach odds and stake, hash the record, then settle P&L.
"""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _write_email(
    path: Path,
    workbook_path: Path,
    selection_log: pd.DataFrame,
    source: EvidenceSource,
) -> None:
    policy = _policy_summary(selection_log)
    strategy = _strategy_v2_summary(_strategy_v2_selected(selection_log))
    conservative = strategy[strategy["strategy_name"] == "conservative_one_race_top10_per_gp"].iloc[0]
    high_all = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "all")].iloc[0]
    high_qual = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "qualifying")].iloc[0]
    high_race = policy[(policy["policy"] == "high_conf_top10_p_ge_85") & (policy["target"] == "race")].iloc[0]
    text = f"""Hi Jordan,

Thanks for the clear feedback. I agree with the distinction: the first CSV is ranking/prediction evidence, not yet an odds-backed betting log.

I have attached a cleaner evidence pack: {workbook_path.name}.

The source run is `{source.run_id}` and its frozen run-manifest SHA-256 is `{source.manifest_sha256 or 'legacy_unavailable'}`.

The stronger, cleaner result is not the winner market. It is the high-confidence top-10 policy:

- rule: top-10 market with model probability >= 85%
- overall: {int(high_all['hits'])}/{int(high_all['selection_count'])} hits ({float(high_all['hit_rate_pct']):.2f}%)
- qualifying: {int(high_qual['hits'])}/{int(high_qual['selection_count'])} hits ({float(high_qual['hit_rate_pct']):.2f}%)
- race: {int(high_race['hits'])}/{int(high_race['selection_count'])} hits ({float(high_race['hit_rate_pct']):.2f}%)

For the five-selection reconciliation specifically, I now use the simpler race top-10 rule, one model rank-1 selection per GP. That settles at {int(conservative['hits'])}/{int(conservative['settled_selections'])} hits ({float(conservative['hit_rate_pct']):.2f}%). The hit/miss outcomes themselves are unchanged; the improvement comes from using the top-10 market where the model is strongest instead of forcing the weaker winner market.

The pack also includes the exact model/config/artifact hash for every selection, model probability and fair odds, information cutoff, result settlement, raw data/artifact manifests, train/holdout methodology, and a reconciliation tab for the previously discussed paper selections.

I am not claiming historical betting P&L from this file because bookmaker/exchange odds were not captured before market close. Those fields are marked explicitly as unavailable. The proposed next step is a forward test focused on the high-confidence top-10 policy, with selections, odds, stake, and record hashes logged before market close and then settled after the race.

Best,
Hugo
"""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build F1 evidence workbook for betting-diligence review.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None, help="Select one complete frozen rolling run explicitly.")
    parser.add_argument(
        "--legacy-read-only",
        action="store_true",
        help="Explicitly inspect the old unmanifested flat layout without treating it as frozen evidence.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo = _repo_root()
    source = _discover_evidence_source(
        repo,
        run_id=args.run_id,
        legacy_read_only=bool(args.legacy_read_only),
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo / "artifacts" / "reports" / "f1" / "evidence_packs" / source.run_id
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to write into existing evidence-pack directory: {output_dir}")

    selections = pd.read_csv(source.selections_path)
    summary = pd.read_csv(source.summary_path)

    git_head = _run_git(repo, ["rev-parse", "HEAD"])
    git_dirty = bool(_run_git(repo, ["status", "--short"]))
    code_hash = _combined_hash(repo, SOURCE_FILES)
    selection_log = _build_selection_log(repo, selections, summary, source, git_head, git_dirty, code_hash)
    strategy_v2 = _strategy_v2_selected(selection_log)
    _assert_source_still_frozen(repo, source)
    output_dir.mkdir(parents=True, exist_ok=False)

    workbook_path = output_dir / "F1_Evidence_Pack_v3_Jordan.xlsx"
    readme_path = output_dir / "README_F1_Evidence_Pack_v3.md"
    email_path = output_dir / "EMAIL_TO_JORDAN.md"
    with workbook_path.open("xb") as workbook_handle:
        with pd.ExcelWriter(workbook_handle, engine="openpyxl") as writer:
            _executive_summary(selection_log).to_excel(writer, sheet_name="executive_summary", index=False)
            _policy_summary(selection_log).to_excel(writer, sheet_name="policy_summary", index=False)
            _high_confidence_top10(selection_log).to_excel(writer, sheet_name="high_conf_top10", index=False)
            strategy_v2.to_excel(writer, sheet_name="strategy_v2_selected", index=False)
            _strategy_v2_summary(strategy_v2).to_excel(writer, sheet_name="strategy_v2_summary", index=False)
            _build_methodology(repo, source, git_head, git_dirty, code_hash).to_excel(writer, sheet_name="methodology", index=False)
            _build_checklist().to_excel(writer, sheet_name="jordan_checklist", index=False)
            selection_log.to_excel(writer, sheet_name="selection_log", index=False)
            summary.to_excel(writer, sheet_name="round_summary", index=False)
            _reconciliation(selection_log).to_excel(writer, sheet_name="paper_trade_reconcile", index=False)
            _skipped_sheet(selection_log).to_excel(writer, sheet_name="skipped_excluded_edited", index=False)
            _raw_manifest(repo, source).to_excel(writer, sheet_name="raw_data_manifest", index=False)
            _artifact_manifest(repo, source).to_excel(writer, sheet_name="artifact_manifest", index=False)
            _source_manifest(repo).to_excel(writer, sheet_name="source_manifest", index=False)
            _forward_schema().to_excel(writer, sheet_name="forward_test_schema", index=False)
            _odds_research().to_excel(writer, sheet_name="odds_data_options", index=False)
    _write_readme(readme_path, workbook_path, selection_log, source)
    _write_email(email_path, workbook_path, selection_log, source)

    print(
        json.dumps(
            {
                "workbook": str(workbook_path),
                "readme": str(readme_path),
                "email": str(email_path),
                "evidence_source_kind": source.source_kind,
                "evidence_run_id": source.run_id,
                "evidence_run_frozen_complete": source.frozen_complete,
                "run_manifest": str(source.manifest_path) if source.manifest_path else None,
                "run_manifest_sha256": source.manifest_sha256,
                "selection_rows": int(len(selection_log)),
                "strategy_v2_rows": int(len(strategy_v2)),
                "summary_rows": int(len(summary)),
                "generated_at_utc": _utc_now(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
