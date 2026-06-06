#!/usr/bin/env python3
"""Canonical experiment runner for Rising Qualification Prediction (F1)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from rqp import PredictionConfig, run_prediction
from rqp.evaluation import evaluate_prediction_rows
from rqp.providers import LocalWeekendProvider
from rqp.runtime import parse_compare_families, parse_json_object, parse_train_seasons

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SPORT = "F1"
PROJECT_NAME = "Rising Qualification Prediction"
ENTRYPOINT = "run_experiment.py"
SCHEMA_VERSION = "1.0"
PROBABILITY_AUDIT_SCHEMA_VERSION = "pl_gumbel_probability_audit_v2"
REQUIRED_PROBABILITY_AUDIT_FIELDS = {
    "schema_version",
    "probability_layer",
    "same_probability_layer_as_production",
    "samples",
    "event_total_audit",
    "metrics",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return bool(default)


def _as_on_off(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "on":
            return True
        if text == "off":
            return False
    return _as_bool(value, default=default)


def _as_list_str(value: Any, default: list[str]) -> list[str]:
    if isinstance(value, list):
        values = [str(v).strip() for v in value if str(v).strip()]
        return values or list(default)
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
        return values or list(default)
    return list(default)


def _resolve_project_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _load_profile(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".toml":
        if tomllib is None:
            raise SystemExit("TOML parsing unavailable on this Python version.")
        payload = tomllib.loads(text)
    elif suffix in {".yaml", ".yml"}:
        payload = None
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text)
        except Exception:
            try:
                payload = json.loads(text)
            except Exception as exc:
                raise SystemExit(
                    "YAML parsing failed. Install pyyaml or use JSON-compatible YAML/TOML."
                ) from exc
    else:
        raise SystemExit("Unsupported profile format. Use .yaml/.yml/.toml.")
    if not isinstance(payload, dict):
        raise SystemExit("Invalid profile: expected object at root.")
    return payload


def _resolve_profile_value(profile: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    part = profile.get(section, {})
    if not isinstance(part, dict):
        return default
    return part.get(key, default)


def _make_envelope(
    *,
    version: Optional[str],
    workflow: str,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    notes: list[str],
    model_name: Optional[str],
    model_family: Optional[str],
    device_used: Optional[str],
    dl_available: Optional[bool],
    candidate_leaderboard: Optional[list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "sport": SPORT,
        "project": PROJECT_NAME,
        "entrypoint": ENTRYPOINT,
        "workflow": workflow,
        "config": config,
        "rows": rows,
        "notes": notes,
        "model_name": model_name,
        "model_family": model_family,
        "device_used": device_used,
        "dl_available": dl_available,
        "candidate_leaderboard": candidate_leaderboard if candidate_leaderboard is not None else [],
        "generated_at": _utc_now(),
    }


def _build_shadow_section(
    candidate_leaderboard: list[dict[str, Any]],
    selected_model_name: Optional[str],
) -> dict[str, Any]:
    selected_row = next(
        (row for row in candidate_leaderboard if row.get("name") == selected_model_name),
        None,
    )
    baseline_row = next(
        (row for row in candidate_leaderboard if str(row.get("family")) == "baseline"),
        None,
    )
    shadow: dict[str, Any] = {"enabled": True}
    if selected_row is not None:
        shadow["selected"] = selected_row
    if baseline_row is not None:
        shadow["baseline_reference"] = baseline_row
    if selected_row is not None and baseline_row is not None:
        selected_mae = selected_row.get("mae")
        baseline_mae = baseline_row.get("mae")
        selected_composite = selected_row.get("composite")
        baseline_composite = baseline_row.get("composite")
        if isinstance(selected_mae, (int, float)) and isinstance(baseline_mae, (int, float)):
            shadow["mae_delta_vs_baseline"] = float(baseline_mae - selected_mae)
        if isinstance(selected_composite, (int, float)) and isinstance(baseline_composite, (int, float)):
            shadow["composite_delta_vs_baseline"] = float(selected_composite - baseline_composite)
    return shadow


def _assert_probability_audit_schema(payload: dict[str, Any]) -> None:
    audit = payload.get("probability_audit")
    if not isinstance(audit, dict) or not audit:
        return
    if str(audit.get("source") or "").strip().lower() != "walk_forward_oof" and not bool(audit.get("available", False)):
        return
    missing = sorted(REQUIRED_PROBABILITY_AUDIT_FIELDS - set(audit))
    if missing:
        raise RuntimeError(f"stale_probability_audit_schema: {missing}")
    if str(audit.get("schema_version") or "") != PROBABILITY_AUDIT_SCHEMA_VERSION:
        raise RuntimeError(f"stale_probability_audit_schema_version: {audit.get('schema_version', 'missing')}")


def _prediction_payload(config: PredictionConfig, *, workflow: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = run_prediction(config)
    rows: list[dict[str, Any]]
    if result.table.empty:
        rows = []
    else:
        rows = json.loads(result.table.to_json(orient="records"))

    payload = _make_envelope(
        version=result.version,
        workflow=workflow,
        config=asdict(config),
        rows=rows,
        notes=list(result.notes),
        model_name=result.model_name,
        model_family=result.model_family,
        device_used=result.device_used,
        dl_available=result.dl_available,
        candidate_leaderboard=list(result.candidate_leaderboard) if config.shadow_eval else [],
    )
    if config.shadow_eval:
        payload["shadow"] = _build_shadow_section(
            candidate_leaderboard=list(result.candidate_leaderboard),
            selected_model_name=result.model_name,
        )
    if isinstance(result.extras, dict) and result.extras:
        payload.update(result.extras)
    _assert_probability_audit_schema(payload)
    return payload, rows


def _profile_base_config(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    defaults = profile.get("defaults", {})
    training = profile.get("training", {})
    experiments = profile.get("experiments", {})
    dl = profile.get("dl", {})
    f1 = profile.get("f1", {})
    f1_listwise = f1.get("listwise", {}) if isinstance(f1, dict) else {}
    f1_live = f1.get("live", {}) if isinstance(f1, dict) else {}
    if not isinstance(defaults, dict):
        defaults = {}
    if not isinstance(training, dict):
        training = {}
    if not isinstance(experiments, dict):
        experiments = {}
    if not isinstance(dl, dict):
        dl = {}
    if not isinstance(f1, dict):
        f1 = {}
    if not isinstance(f1_listwise, dict):
        f1_listwise = {}
    if not isinstance(f1_live, dict):
        f1_live = {}

    year = _as_int(args.year if args.year is not None else defaults.get("year"), 2025)
    round_number = _as_int(args.round_number if args.round_number is not None else defaults.get("round_number"), 1)
    source = str(args.source or defaults.get("source") or "local").strip().lower()
    include_standings = _as_bool(defaults.get("include_standings", False), False)
    train_seasons = training.get("train_seasons", [year - 3, year - 2, year - 1])
    if not isinstance(train_seasons, list):
        train_seasons = [year - 3, year - 2, year - 1]
    train_seasons = sorted({int(v) for v in train_seasons})
    shadow_eval_raw: Any = args.shadow_eval if getattr(args, "shadow_eval", None) is not None else profile.get("shadow_eval", True)
    return {
        "year": year,
        "round_number": round_number,
        "source": source,
        "include_standings": include_standings,
        "train_seasons": train_seasons,
        "cache_dir": _resolve_project_path(args.cache_dir if args.cache_dir is not None else defaults.get("cache_dir")),
        "weekends_dir": _resolve_project_path(
            args.weekends_dir if args.weekends_dir is not None else defaults.get("weekends_dir", "data/f1/raw/weekends")
        ),
        "enable_dl_candidates": _as_bool(experiments.get("enable_dl_candidates", False), False),
        "compare_families": _as_list_str(experiments.get("compare_families", ["ml"]), ["ml"]),
        "dl_device": str(dl.get("device", "auto")).strip().lower(),
        "dl_arch": str(dl.get("arch", "mlp_tabular_v1")).strip(),
        "dl_hyperparams": dl.get("hyperparams", {}) if isinstance(dl.get("hyperparams", {}), dict) else {},
        "dl_seed": _as_int(dl.get("seed", 42), 42),
        "f1_model": str(
            args.f1_model
            if getattr(args, "f1_model", None) is not None
            else f1.get("model", "auto"),
        ).strip().lower(),
        "f1_listwise": str(
            args.f1_listwise
            if getattr(args, "f1_listwise", None) is not None
            else f1_listwise.get("method", "pl_gumbel"),
        ).strip().lower(),
        "f1_pl_samples": _as_int(
            getattr(args, "f1_pl_samples", None)
            if getattr(args, "f1_pl_samples", None) is not None
            else f1_listwise.get("samples", 2000),
            2000,
        ),
        "f1_pl_temperature": _as_float(
            getattr(args, "f1_pl_temperature", None)
            if getattr(args, "f1_pl_temperature", None) is not None
            else f1_listwise.get("temperature", 1.0),
            1.0,
        ),
        "f1_listwise_seed": _as_int(
            getattr(args, "f1_listwise_seed", None)
            if getattr(args, "f1_listwise_seed", None) is not None
            else f1_listwise.get("seed", 42),
            42,
        ),
        "disable_circuit_features": _as_bool(
            getattr(args, "disable_circuit_features", None)
            if getattr(args, "disable_circuit_features", None) is not None
            else experiments.get("disable_circuit_features", True),
            True,
        ),
        "shadow_eval": _as_on_off(shadow_eval_raw, True),
        "f1_mode": str(
            args.f1_mode
            if getattr(args, "f1_mode", None) is not None
            else f1.get("mode", "offline"),
        ).strip().lower(),
        "f1_live_source": str(
            args.f1_live_source
            if getattr(args, "f1_live_source", None) is not None
            else f1_live.get("source", "auto"),
        ).strip().lower(),
        "f1_live_model": str(
            args.f1_live_model
            if getattr(args, "f1_live_model", None) is not None
            else f1_live.get("model", "ssm_v1"),
        ).strip().lower(),
        "f1_live_horizon_laps": _as_int(
            getattr(args, "f1_live_horizon_laps", None)
            if getattr(args, "f1_live_horizon_laps", None) is not None
            else f1_live.get("horizon_laps", 10),
            10,
        ),
        "f1_live_seed": _as_int(
            getattr(args, "f1_live_seed", None)
            if getattr(args, "f1_live_seed", None) is not None
            else f1_live.get("seed", 42),
            42,
        ),
        "f1_live_cache_dir": _resolve_project_path(
            getattr(args, "f1_live_cache_dir", None)
            if getattr(args, "f1_live_cache_dir", None) is not None
            else f1_live.get("cache_dir")
        ),
        "f1_live_replay_path": _resolve_project_path(
            getattr(args, "f1_live_replay_path", None)
            if getattr(args, "f1_live_replay_path", None) is not None
            else f1_live.get("replay_path")
        ),
        "f1_live_replay_cutoff_lap": (
            _as_int(getattr(args, "f1_live_replay_cutoff_lap", None), 0)
            if getattr(args, "f1_live_replay_cutoff_lap", None) is not None
            else (
                _as_int(f1_live.get("replay_cutoff_lap"), 0)
                if f1_live.get("replay_cutoff_lap") is not None
                else None
            )
        ),
    }


def _build_prediction_config(
    *,
    base: dict[str, Any],
    mode: str,
    round_number: int,
    disable_runsim_features: bool,
    enable_dl_candidates: bool,
    compare_families: list[str],
    disable_circuit_features: bool = True,
) -> PredictionConfig:
    return PredictionConfig(
        source=str(base["source"]),
        mode=str(mode),
        year=int(base["year"]),
        round_number=int(round_number),
        train_seasons=list(base["train_seasons"]),
        include_standings=bool(base["include_standings"]),
        cache_dir=base.get("cache_dir"),
        meeting_name=None,
        country_name=None,
        weekends_dir=base.get("weekends_dir"),
        enable_dl_candidates=bool(enable_dl_candidates),
        compare_families=list(compare_families),
        dl_device=str(base.get("dl_device", "auto")),
        dl_arch=str(base.get("dl_arch", "mlp_tabular_v1")),
        dl_hyperparams=dict(base.get("dl_hyperparams", {})),
        dl_seed=int(base.get("dl_seed", 42)),
        disable_runsim_features=bool(disable_runsim_features),
        disable_circuit_features=bool(disable_circuit_features),
        f1_model=str(base.get("f1_model", "auto")),
        f1_listwise=str(base.get("f1_listwise", "pl_gumbel")),
        f1_pl_samples=int(base.get("f1_pl_samples", 2000)),
        f1_pl_temperature=float(base.get("f1_pl_temperature", 1.0)),
        f1_listwise_seed=int(base.get("f1_listwise_seed", 42)),
        shadow_eval=bool(base.get("shadow_eval", True)),
        f1_mode=str(base.get("f1_mode", "offline")),
        f1_live_source=str(base.get("f1_live_source", "auto")),
        f1_live_model=str(base.get("f1_live_model", "ssm_v1")),
        f1_live_horizon_laps=int(base.get("f1_live_horizon_laps", 10)),
        f1_live_seed=int(base.get("f1_live_seed", 42)),
        f1_live_cache_dir=base.get("f1_live_cache_dir"),
        f1_live_replay_path=base.get("f1_live_replay_path"),
        f1_live_replay_cutoff_lap=base.get("f1_live_replay_cutoff_lap"),
    )


def _round_blocks(round_start: int, round_end: int, block_size: int) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    start = int(round_start)
    while start <= int(round_end):
        end = min(int(round_end), start + int(block_size) - 1)
        blocks.append((start, end))
        start = end + 1
    return blocks


def _summarize_round_metrics(rows: list[dict[str, Any]], block_size: int) -> dict[str, Any]:
    mae_values = [float(r["mae"]) for r in rows if r.get("mae") is not None]
    hit_values = [float(r["top10_hit"]) for r in rows if r.get("top10_hit") is not None]
    summary = {
        "rounds": len(rows),
        "mae_avg": float(sum(mae_values) / len(mae_values)) if mae_values else None,
        "top10_hit_avg": float(sum(hit_values) / len(hit_values)) if hit_values else None,
    }
    if not rows:
        summary["blocks"] = []
        return summary
    min_round = min(int(r["round"]) for r in rows)
    max_round = max(int(r["round"]) for r in rows)
    block_rows: list[dict[str, Any]] = []
    for start, end in _round_blocks(min_round, max_round, block_size):
        chunk = [r for r in rows if start <= int(r["round"]) <= end]
        if not chunk:
            continue
        chunk_mae = [float(r["mae"]) for r in chunk if r.get("mae") is not None]
        chunk_hit = [float(r["top10_hit"]) for r in chunk if r.get("top10_hit") is not None]
        block_rows.append(
            {
                "start_round": start,
                "end_round": end,
                "mae_avg": float(sum(chunk_mae) / len(chunk_mae)) if chunk_mae else None,
                "top10_hit_avg": float(sum(chunk_hit) / len(chunk_hit)) if chunk_hit else None,
            }
        )
    summary["blocks"] = block_rows
    return summary


def _best_variant(variants: dict[str, Any], prefix: str) -> Optional[dict[str, Any]]:
    candidates = []
    for name, payload in variants.items():
        if not name.startswith(prefix):
            continue
        summary = payload.get("summary", {})
        mae = summary.get("mae_avg")
        top10 = summary.get("top10_hit_avg")
        if mae is None or top10 is None:
            continue
        candidates.append((name, float(mae), float(top10), payload))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[1], -item[2]))
    return candidates[0][3]


def _mode_gate_for(mode: str, gates: dict[str, Any]) -> dict[str, Any]:
    per_mode = gates.get(mode, {})
    if isinstance(per_mode, dict):
        merged = dict(gates)
        merged.update(per_mode)
        return merged
    return dict(gates)


def _evaluate_gate(mode: str, mode_results: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    gate_cfg = _mode_gate_for(mode, gates)
    min_mae_gain = float(gate_cfg.get("min_mae_gain", 0.05 if mode == "qualifying" else 0.03))
    min_top10_gain = float(gate_cfg.get("min_top10_gain", 0.005))
    block_stability = _as_bool(gate_cfg.get("block_stability_enabled", True), True)

    best_ml = _best_variant(mode_results, "ML_")
    best_dl = _best_variant(mode_results, "DL_")
    if best_ml is None or best_dl is None:
        return {
            "available": False,
            "reason": "missing_ml_or_dl_variant",
        }

    ml_summary = best_ml["summary"]
    dl_summary = best_dl["summary"]
    mae_gain = float(ml_summary["mae_avg"] - dl_summary["mae_avg"])
    top10_gain = float(dl_summary["top10_hit_avg"] - ml_summary["top10_hit_avg"])

    block_non_negative = 0
    block_total = 0
    if block_stability:
        ml_blocks = ml_summary.get("blocks", [])
        dl_blocks = dl_summary.get("blocks", [])
        for ml_block, dl_block in zip(ml_blocks, dl_blocks):
            if ml_block.get("mae_avg") is None or dl_block.get("mae_avg") is None:
                continue
            if ml_block.get("top10_hit_avg") is None or dl_block.get("top10_hit_avg") is None:
                continue
            block_total += 1
            block_mae_gain = float(ml_block["mae_avg"] - dl_block["mae_avg"])
            block_top10_gain = float(dl_block["top10_hit_avg"] - ml_block["top10_hit_avg"])
            if block_mae_gain >= 0.0 and block_top10_gain >= 0.0:
                block_non_negative += 1

    stability_ok = True
    if block_stability and block_total > 0:
        stability_ok = block_non_negative >= 3

    passed = (mae_gain >= min_mae_gain) and (top10_gain >= min_top10_gain) and stability_ok
    return {
        "available": True,
        "passed": bool(passed),
        "best_ml_variant": best_ml.get("variant"),
        "best_dl_variant": best_dl.get("variant"),
        "mae_gain": mae_gain,
        "top10_gain": top10_gain,
        "thresholds": {
            "min_mae_gain": min_mae_gain,
            "min_top10_gain": min_top10_gain,
            "block_stability_enabled": block_stability,
        },
        "block_stability": {
            "non_negative_blocks": block_non_negative,
            "total_blocks": block_total,
        },
    }


def _run_single_prediction(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    base = _profile_base_config(profile, args)
    mode = str(args.mode or _resolve_profile_value(profile, "defaults", "mode", "qualifying"))
    config = _build_prediction_config(
        base=base,
        mode=mode,
        round_number=int(base["round_number"]),
        disable_runsim_features=False,
        enable_dl_candidates=bool(base["enable_dl_candidates"]),
        compare_families=list(base["compare_families"]),
        disable_circuit_features=bool(base.get("disable_circuit_features", True)),
    )
    payload, _ = _prediction_payload(config, workflow="single_prediction")
    payload["profile_name"] = profile.get("name", "unknown")
    payload["mission"] = profile.get("mission", "unknown")
    return payload


def _run_weekend_phase(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    base = _profile_base_config(profile, args)
    defaults = profile.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    phase = str(args.phase or defaults.get("phase", "pre-qualifying"))
    script = Path(__file__).resolve().parent / "run_live_weekend_pipeline.py"
    output_dir = _resolve_project_path(
        args.output_dir or _resolve_profile_value(profile, "outputs", "base_dir", "outputs/f1/live")
    )
    if output_dir is None:
        raise SystemExit("Invalid output_dir for weekend workflow.")

    out_path = Path(output_dir) / f"profile_weekend_{phase.replace('-', '_')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--phase",
        phase,
        "--source",
        str(base["source"]),
        "--year",
        str(base["year"]),
        "--round",
        str(base["round_number"]),
        "--train-seasons",
        ",".join(str(y) for y in base["train_seasons"]),
        "--weekends-dir",
        str(base["weekends_dir"]),
        "--output-dir",
        str(output_dir),
        "--output-format",
        "json",
        "--output-path",
        str(out_path),
        "--quiet",
    ]
    if bool(base["include_standings"]):
        cmd.append("--include-standings")
    if bool(base["enable_dl_candidates"]):
        cmd.append("--enable-dl-candidates")
    cmd.extend(["--compare-families", ",".join(base["compare_families"])])
    cmd.extend(["--dl-device", str(base["dl_device"])])
    cmd.extend(["--dl-arch", str(base["dl_arch"])])
    cmd.extend(["--dl-hyperparams", json.dumps(base["dl_hyperparams"])])
    cmd.extend(["--dl-seed", str(base["dl_seed"])])
    cmd.extend(["--f1-mode", str(base["f1_mode"])])
    cmd.extend(["--f1-live-source", str(base["f1_live_source"])])
    cmd.extend(["--f1-live-model", str(base["f1_live_model"])])
    cmd.extend(["--f1-live-horizon-laps", str(base["f1_live_horizon_laps"])])
    cmd.extend(["--f1-live-seed", str(base["f1_live_seed"])])
    if base.get("cache_dir"):
        cmd.extend(["--cache-dir", str(base["cache_dir"])])
    if base.get("f1_live_cache_dir"):
        cmd.extend(["--f1-live-cache-dir", str(base["f1_live_cache_dir"])])
    if base.get("f1_live_replay_path"):
        cmd.extend(["--f1-live-replay-path", str(base["f1_live_replay_path"])])
    if base.get("f1_live_replay_cutoff_lap") is not None:
        cmd.extend(["--f1-live-replay-cutoff-lap", str(base["f1_live_replay_cutoff_lap"])])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"run_live_weekend_pipeline failed: {proc.stderr.strip() or proc.stdout.strip()}")

    nested_payload = json.loads(out_path.read_text(encoding="utf-8"))
    rows = nested_payload.get("rows") if isinstance(nested_payload.get("rows"), list) else []
    notes = nested_payload.get("notes") if isinstance(nested_payload.get("notes"), list) else []

    payload = _make_envelope(
        version=nested_payload.get("version"),
        workflow="weekend_phase",
        config={
            "source": str(base["source"]),
            "year": int(base["year"]),
            "round_number": int(base["round_number"]),
            "phase": phase,
            "train_seasons": list(base["train_seasons"]),
            "weekends_dir": str(base["weekends_dir"]),
            "cache_dir": base.get("cache_dir"),
            "include_standings": bool(base["include_standings"]),
            "enable_dl_candidates": bool(base["enable_dl_candidates"]),
            "compare_families": list(base["compare_families"]),
            "dl_device": str(base["dl_device"]),
            "dl_arch": str(base["dl_arch"]),
            "dl_hyperparams": dict(base["dl_hyperparams"]),
            "dl_seed": int(base["dl_seed"]),
            "f1_mode": str(base["f1_mode"]),
            "f1_live_source": str(base["f1_live_source"]),
            "f1_live_model": str(base["f1_live_model"]),
            "f1_live_horizon_laps": int(base["f1_live_horizon_laps"]),
            "f1_live_seed": int(base["f1_live_seed"]),
            "f1_live_cache_dir": base.get("f1_live_cache_dir"),
            "f1_live_replay_path": base.get("f1_live_replay_path"),
            "f1_live_replay_cutoff_lap": base.get("f1_live_replay_cutoff_lap"),
            "output_dir": str(output_dir),
        },
        rows=rows,
        notes=notes,
        model_name=nested_payload.get("model_name"),
        model_family=nested_payload.get("model_family"),
        device_used=nested_payload.get("device_used"),
        dl_available=nested_payload.get("dl_available"),
        candidate_leaderboard=nested_payload.get("candidate_leaderboard"),
    )
    payload["profile_name"] = profile.get("name", "unknown")
    payload["mission"] = profile.get("mission", "unknown")
    payload["payload"] = nested_payload
    return payload


def _run_backtest_ablation_compare(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    base = _profile_base_config(profile, args)
    training = profile.get("training", {})
    evaluation = profile.get("evaluation", {})
    outputs = profile.get("outputs", {})
    experiments = profile.get("experiments", {})
    if not isinstance(training, dict):
        training = {}
    if not isinstance(evaluation, dict):
        evaluation = {}
    if not isinstance(outputs, dict):
        outputs = {}
    if not isinstance(experiments, dict):
        experiments = {}

    holdout_year = _as_int(evaluation.get("holdout_year", base["year"]), base["year"])
    holdout_strict = _as_bool(training.get("holdout_strict", True), True)
    if holdout_strict and int(holdout_year) in {int(y) for y in base["train_seasons"]}:
        raise SystemExit(
            f"Holdout strict violated: holdout year {holdout_year} found in train_seasons={base['train_seasons']}"
        )
    base["year"] = int(holdout_year)

    if str(base["source"]) != "local":
        raise SystemExit("backtest_ablation_compare requires source=local (no API).")

    round_start = _as_int(evaluation.get("round_start", 6), 6)
    round_end = _as_int(evaluation.get("round_end", 24), 24)
    if round_end < round_start:
        raise SystemExit("Invalid round range for backtest.")

    modes = _as_list_str(evaluation.get("modes", ["qualifying", "race"]), ["qualifying", "race"])
    block_size = _as_int(evaluation.get("block_size_rounds", 5), 5)
    gates = evaluation.get("gates", {})
    if not isinstance(gates, dict):
        gates = {}

    enable_dl = _as_bool(experiments.get("enable_dl_candidates", base["enable_dl_candidates"]), False)
    compare_families = _as_list_str(experiments.get("compare_families", base["compare_families"]), ["ml"])
    dl_requested = enable_dl and ("dl" in {f.lower() for f in compare_families})

    output_root_value = _resolve_project_path(args.output_dir or outputs.get("base_dir") or "outputs/f1/preseason/holdout_2025")
    if output_root_value is None:
        raise SystemExit("Invalid output_dir for backtest workflow.")

    output_root = Path(output_root_value)
    run_dir = output_root / f"backtest_ablation_compare_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    provider = LocalWeekendProvider(weekends_dir=base["weekends_dir"])
    variant_specs = [
        {"variant": "ML_ALL", "enable_dl": False, "disable_runsim": False, "families": ["ml"]},
        {"variant": "ML_NO_RUNSIM", "enable_dl": False, "disable_runsim": True, "families": ["ml"]},
    ]
    if dl_requested:
        variant_specs.extend(
            [
                {"variant": "DL_ALL", "enable_dl": True, "disable_runsim": False, "families": ["ml", "dl"]},
                {"variant": "DL_NO_RUNSIM", "enable_dl": True, "disable_runsim": True, "families": ["ml", "dl"]},
            ]
        )

    versions: set[str] = set()
    results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_key = str(mode).strip().lower()
        mode_payload: dict[str, Any] = {}
        for spec in variant_specs:
            round_metrics: list[dict[str, Any]] = []
            for rnd in range(round_start, round_end + 1):
                config = _build_prediction_config(
                    base=base,
                    mode=mode_key,
                    round_number=rnd,
                    disable_runsim_features=bool(spec["disable_runsim"]),
                    enable_dl_candidates=bool(spec["enable_dl"]),
                    compare_families=list(spec["families"]),
                    disable_circuit_features=bool(spec.get("disable_circuit", True)),
                )
                prediction, rows = _prediction_payload(config, workflow="backtest_ablation_compare")
                version_value = prediction.get("version")
                if isinstance(version_value, str) and version_value:
                    versions.add(version_value)

                actual = (
                    provider.get_qualifying_results(base["year"], rnd)
                    if mode_key == "qualifying"
                    else provider.get_race_results(base["year"], rnd)
                )
                eval_rows = prediction.get("all_prediction_rows") or rows
                evaluation_row = evaluate_prediction_rows(
                    predicted_rows=eval_rows,
                    actual_results=actual,
                    actual_position_col="position",
                )
                mae = evaluation_row.get("field_mae") if evaluation_row.get("available") else None
                top10_hit = evaluation_row["top10_hit"] if evaluation_row.get("available") else None
                row_payload = {
                    "round": int(rnd),
                    "mae": mae,
                    "mae_valid": evaluation_row.get("mae_valid"),
                    "mae_on_common": evaluation_row.get("mae_on_common"),
                    "field_mae_penalized": evaluation_row.get("field_mae_penalized"),
                    "field_coverage": evaluation_row.get("field_coverage"),
                    "evaluation_reason": evaluation_row.get("evaluation_reason"),
                    "top10_hit": top10_hit,
                    "model_name": prediction["model_name"],
                    "model_family": prediction["model_family"],
                    "device_used": prediction["device_used"],
                    "dl_available": prediction["dl_available"],
                    "rows_common": evaluation_row.get("rows_common"),
                    "rows_actual": evaluation_row.get("rows_actual"),
                    "rows_evaluated": len(eval_rows),
                }
                round_metrics.append(row_payload)

                artifact = {
                    "variant": spec["variant"],
                    "mode": mode_key,
                    "round": int(rnd),
                    "prediction": prediction,
                    "evaluation": evaluation_row,
                    "generated_at": _utc_now(),
                }
                artifact_path = run_dir / f"{mode_key}_{spec['variant'].lower()}_r{rnd}.json"
                artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

            summary = _summarize_round_metrics(round_metrics, block_size=block_size)
            mode_payload[spec["variant"]] = {
                "variant": spec["variant"],
                "summary": summary,
                "rounds": round_metrics,
            }
        results[mode_key] = mode_payload

    acceptance = {
        mode: _evaluate_gate(mode, mode_results, gates)
        for mode, mode_results in results.items()
    }

    version: Optional[str]
    if len(versions) == 1:
        version = next(iter(versions))
    elif versions:
        version = "mixed"
    else:
        version = None

    payload = _make_envelope(
        version=version,
        workflow="backtest_ablation_compare",
        config={
            "year": int(base["year"]),
            "round_range": [int(round_start), int(round_end)],
            "train_seasons": list(base["train_seasons"]),
            "holdout_strict": bool(holdout_strict),
            "source": str(base["source"]),
            "weekends_dir": str(base["weekends_dir"]),
            "output_dir": str(run_dir),
            "modes": list(modes),
            "block_size_rounds": int(block_size),
            "enable_dl_candidates": bool(enable_dl),
            "compare_families": list(compare_families),
        },
        rows=[],
        notes=[],
        model_name=None,
        model_family=None,
        device_used=None,
        dl_available=None,
        candidate_leaderboard=[],
    )
    payload["profile_name"] = profile.get("name", "unknown")
    payload["mission"] = profile.get("mission", "unknown")
    payload["year"] = int(base["year"])
    payload["round_range"] = [int(round_start), int(round_end)]
    payload["train_seasons"] = list(base["train_seasons"])
    payload["holdout_strict"] = bool(holdout_strict)
    payload["source"] = str(base["source"])
    payload["weekends_dir"] = str(base["weekends_dir"])
    payload["output_dir"] = str(run_dir)
    payload["results"] = results
    payload["acceptance"] = acceptance

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_parser(runner: str) -> argparse.ArgumentParser:
    runner_name = str(runner).strip().lower()
    if runner_name == "profile":
        parser = argparse.ArgumentParser(description="Run F1 experiment workflows from YAML/TOML profiles.")
        parser.add_argument("--profile", required=True)
        parser.add_argument("--workflow", default=None)
        parser.add_argument("--mode", default=None)
        parser.add_argument("--phase", default=None)
        parser.add_argument("--source", default=None)
        parser.add_argument("--year", type=int, default=None)
        parser.add_argument("--round", dest="round_number", type=int, default=None)
        parser.add_argument("--weekends-dir", default=None)
        parser.add_argument("--cache-dir", default=None)
        parser.add_argument("--output-dir", default=None)
        parser.add_argument("--f1_model", choices=["auto", "baseline", "xgb_rank", "eb_rank", "lgbm_rank"], default=None)
        parser.add_argument("--f1_listwise", choices=["off", "pl_gumbel"], default=None)
        parser.add_argument("--f1_pl_samples", type=int, default=None)
        parser.add_argument("--f1_pl_temperature", type=float, default=None)
        parser.add_argument("--f1_listwise_seed", type=int, default=None)
        parser.add_argument("--f1_mode", choices=["offline", "live"], default=None)
        parser.add_argument("--f1_live_source", choices=["auto", "local", "fastf1"], default=None)
        parser.add_argument("--f1_live_model", choices=["ssm_v1"], default=None)
        parser.add_argument("--f1_live_horizon_laps", type=int, default=None)
        parser.add_argument("--f1_live_seed", type=int, default=None)
        parser.add_argument("--f1_live_cache_dir", default=None)
        parser.add_argument("--f1_live_replay_path", default=None)
        parser.add_argument("--f1_live_replay_cutoff_lap", type=int, default=None)
        parser.add_argument("--shadow_eval", choices=["on", "off"], default=None)
        parser.add_argument("--output-format", choices=["text", "json"], default="json")
        parser.add_argument("--output-path", default=None)
        parser.add_argument("--quiet", action="store_true")
        return parser

    if runner_name == "prediction":
        parser = argparse.ArgumentParser(
            description="Rising Qualification Prediction (FastF1 / OpenF1 / local offline)"
        )
        parser.add_argument("--mode", choices=["qualifying", "race"], required=True)
        parser.add_argument("--source", choices=["fastf1", "openf1", "local"], required=True)
        parser.add_argument("--year", type=int, required=True)
        parser.add_argument("--round", dest="round_number", type=int, required=True)
        parser.add_argument("--train-seasons", default="auto")
        parser.add_argument(
            "--train-policy",
            choices=["strict_transfer", "rolling", "frozen_preseason", "legacy_auto"],
            default="legacy_auto",
            help="Policy used only when --train-seasons=auto.",
        )
        parser.add_argument("--include-standings", action="store_true")
        parser.add_argument("--cache-dir", default=None)
        parser.add_argument("--weekends-dir", default="data/f1/raw/weekends")
        parser.add_argument("--meeting-name", default=None)
        parser.add_argument("--country-name", default=None)
        parser.add_argument("--enable-dl-candidates", action="store_true")
        parser.add_argument("--compare-families", default="ml")
        parser.add_argument("--dl-device", choices=["auto", "cpu", "cuda"], default="auto")
        parser.add_argument("--dl-arch", default="mlp_tabular_v1")
        parser.add_argument("--dl-hyperparams", default="{}")
        parser.add_argument("--dl-seed", type=int, default=42)
        parser.add_argument("--disable-runsim-features", action="store_true")
        parser.add_argument("--disable-circuit-features", dest="disable_circuit_features", action="store_true", default=None)
        parser.add_argument("--enable-circuit-features", dest="disable_circuit_features", action="store_false")
        parser.add_argument("--f1_model", choices=["auto", "baseline", "xgb_rank", "eb_rank", "lgbm_rank"], default="auto")
        parser.add_argument("--f1_listwise", choices=["off", "pl_gumbel"], default="pl_gumbel")
        parser.add_argument("--f1_pl_samples", type=int, default=2000)
        parser.add_argument("--f1_pl_temperature", type=float, default=1.0)
        parser.add_argument("--f1_listwise_seed", type=int, default=42)
        parser.add_argument("--f1_mode", choices=["offline", "live"], default="offline")
        parser.add_argument("--f1_live_source", choices=["auto", "local", "fastf1"], default="auto")
        parser.add_argument("--f1_live_model", choices=["ssm_v1"], default="ssm_v1")
        parser.add_argument("--f1_live_horizon_laps", type=int, default=10)
        parser.add_argument("--f1_live_seed", type=int, default=42)
        parser.add_argument("--f1_live_cache_dir", default=None)
        parser.add_argument("--f1_live_replay_path", default=None)
        parser.add_argument("--f1_live_replay_cutoff_lap", type=int, default=None)
        parser.add_argument("--shadow_eval", choices=["on", "off"], default="on")
        parser.add_argument("--output-format", choices=["text", "json"], default="text")
        parser.add_argument("--output-path", default=None)
        parser.add_argument("--quiet", action="store_true")
        return parser

    raise SystemExit(f"Unsupported runner for parser: {runner}")


def parse_args(runner: str, argv: Sequence[str]) -> argparse.Namespace:
    parser = build_parser(runner)
    return parser.parse_args(list(argv))


def _run_profile_cli(argv: Sequence[str]) -> None:
    args = parse_args("profile", argv)

    profile_path = Path(args.profile).expanduser()
    if not profile_path.exists() and not profile_path.is_absolute():
        local_candidate = Path(__file__).resolve().parent / profile_path
        if local_candidate.exists():
            profile_path = local_candidate
    if not profile_path.exists():
        raise SystemExit(f"Profile not found: {profile_path}")

    profile = _load_profile(profile_path)
    workflow = str(args.workflow or profile.get("workflow", "")).strip().lower()
    if not workflow:
        raise SystemExit("Profile workflow is missing.")

    if workflow == "single_prediction":
        payload = _run_single_prediction(profile, args)
    elif workflow == "weekend_phase":
        payload = _run_weekend_phase(profile, args)
    elif workflow == "backtest_ablation_compare":
        payload = _run_backtest_ablation_compare(profile, args)
    else:
        raise SystemExit(f"Unsupported workflow: {workflow}")

    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.quiet:
        return
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=" * 72)
    print(f"Profile: {profile.get('name', profile_path.name)}")
    print(f"Workflow: {workflow}")
    print("=" * 72)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_prediction_cli(argv: Sequence[str]) -> None:
    args = parse_args("prediction", argv)

    dl_hyperparams = parse_json_object(args.dl_hyperparams, "--dl-hyperparams")
    config = PredictionConfig(
        source=args.source,
        mode=args.mode,
        year=args.year,
        round_number=args.round_number,
        train_seasons=parse_train_seasons(args.train_seasons, args.year, args.train_policy),
        include_standings=args.include_standings,
        cache_dir=args.cache_dir,
        meeting_name=args.meeting_name,
        country_name=args.country_name,
        weekends_dir=args.weekends_dir,
        enable_dl_candidates=args.enable_dl_candidates,
        compare_families=parse_compare_families(args.compare_families),
        dl_device=args.dl_device,
        dl_arch=args.dl_arch,
        dl_hyperparams=dl_hyperparams,
        dl_seed=args.dl_seed,
        disable_runsim_features=args.disable_runsim_features,
        disable_circuit_features=True if args.disable_circuit_features is None else bool(args.disable_circuit_features),
        f1_model=args.f1_model,
        f1_listwise=args.f1_listwise,
        f1_pl_samples=args.f1_pl_samples,
        f1_pl_temperature=args.f1_pl_temperature,
        f1_listwise_seed=args.f1_listwise_seed,
        shadow_eval=_as_on_off(args.shadow_eval, True),
        f1_mode=args.f1_mode,
        f1_live_source=args.f1_live_source,
        f1_live_model=args.f1_live_model,
        f1_live_horizon_laps=args.f1_live_horizon_laps,
        f1_live_seed=args.f1_live_seed,
        f1_live_cache_dir=args.f1_live_cache_dir,
        f1_live_replay_path=args.f1_live_replay_path,
        f1_live_replay_cutoff_lap=args.f1_live_replay_cutoff_lap,
    )

    result = run_prediction(config)
    rows: list[dict[str, Any]]
    if result.table.empty:
        rows = []
    else:
        rows = json.loads(result.table.to_json(orient="records"))
    payload = _make_envelope(
        version=result.version,
        workflow="single_prediction",
        config=asdict(config),
        rows=rows,
        notes=list(result.notes),
        model_name=result.model_name,
        model_family=result.model_family,
        device_used=result.device_used,
        dl_available=result.dl_available,
        candidate_leaderboard=list(result.candidate_leaderboard) if config.shadow_eval else [],
    )
    if config.shadow_eval:
        payload["shadow"] = _build_shadow_section(
            candidate_leaderboard=list(result.candidate_leaderboard),
            selected_model_name=result.model_name,
        )
    if isinstance(result.extras, dict) and result.extras:
        payload.update(result.extras)

    if args.output_format == "json":
        if args.output_path:
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.quiet:
        return

    print("=" * 72)
    print(
        f"Mode: {config.mode} | Source: {config.source} | Year: {config.year} | Round: {config.round_number}"
    )
    print(f"Model version: {payload['version']}")
    print(f"Model selected: {payload['model_name']} [{payload['model_family']}]")
    if payload.get("device_used"):
        print(f"Device: {payload['device_used']}")
    print("=" * 72)
    if not rows:
        print("Aucune prediction disponible.")
    else:
        print(result.table.to_string(index=False))
    if payload.get("notes"):
        print("\nNotes:")
        for note in payload["notes"]:
            print(f"- {note}")


def _pick_runner(argv: list[str], default_runner: Optional[str]) -> tuple[str, list[str]]:
    if argv and argv[0] in {"profile", "prediction"}:
        return argv[0], argv[1:]

    if default_runner in {"profile", "prediction"}:
        return str(default_runner), argv

    profile_markers = {"--profile", "--workflow", "--phase"}
    if any(marker in argv for marker in profile_markers):
        return "profile", argv

    prediction_markers = {"--mode", "--source", "--year", "--round"}
    if all(marker in argv for marker in prediction_markers):
        return "prediction", argv

    return "profile", argv


def main(argv: Optional[Sequence[str]] = None, *, default_runner: Optional[str] = None) -> None:
    args = list(argv) if argv is not None else list(sys.argv[1:])
    runner, runner_args = _pick_runner(args, default_runner)

    if runner == "profile":
        _run_profile_cli(runner_args)
        return
    if runner == "prediction":
        _run_prediction_cli(runner_args)
        return
    raise SystemExit(f"Unsupported runner: {runner}")


if __name__ == "__main__":
    main()
