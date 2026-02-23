#!/usr/bin/env python3
"""Profile-driven runner for preseason/live F1 workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from rqp import PredictionConfig, run_prediction
from rqp.providers import LocalWeekendProvider

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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


def _normalize_driver_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    return " ".join(text.split())


def _actual_name_column(frame: pd.DataFrame) -> Optional[str]:
    for col in ["driver_name", "driver_id", "Abbreviation", "Driver"]:
        if col in frame.columns:
            return col
    return None


def _evaluate_prediction_rows(
    predicted_rows: list[dict[str, Any]],
    actual_results: pd.DataFrame,
    actual_position_col: str,
) -> dict[str, Any]:
    if not predicted_rows:
        return {"available": False, "reason": "prediction_rows_unavailable"}
    if actual_results is None or actual_results.empty:
        return {"available": False, "reason": "actual_results_unavailable"}

    pred = pd.DataFrame(predicted_rows).copy()
    if pred.empty or "driver_name" not in pred.columns:
        return {"available": False, "reason": "prediction_driver_name_unavailable"}
    pred["driver_key"] = pred["driver_name"].map(_normalize_driver_key)
    pred = pred[pred["driver_key"] != ""]
    if pred.empty:
        return {"available": False, "reason": "prediction_driver_key_unavailable"}
    if "rank" in pred.columns:
        pred["pred_rank"] = pd.to_numeric(pred["rank"], errors="coerce")
    else:
        pred["pred_rank"] = pd.Series(range(1, len(pred) + 1), index=pred.index, dtype=float)
    pred = pred.dropna(subset=["pred_rank"])
    pred["pred_rank"] = pred["pred_rank"].astype(float)

    actual = actual_results.copy()
    if actual_position_col not in actual.columns:
        return {"available": False, "reason": "actual_position_unavailable"}
    name_col = _actual_name_column(actual)
    if name_col is None:
        return {"available": False, "reason": "actual_driver_name_unavailable"}
    actual["driver_key"] = actual[name_col].map(_normalize_driver_key)
    actual["actual_rank"] = pd.to_numeric(actual[actual_position_col], errors="coerce")
    actual = actual[(actual["driver_key"] != "") & actual["actual_rank"].notna()]
    if actual.empty:
        return {"available": False, "reason": "actual_clean_unavailable"}

    pred_unique = pred.sort_values("pred_rank", kind="mergesort").drop_duplicates(subset=["driver_key"], keep="first")
    actual_unique = actual.sort_values("actual_rank", kind="mergesort").drop_duplicates(
        subset=["driver_key"],
        keep="first",
    )
    merged = pred_unique.merge(actual_unique[["driver_key", "actual_rank"]], on="driver_key", how="inner")
    merged = merged.dropna(subset=["pred_rank", "actual_rank"])

    mae = float((merged["pred_rank"] - merged["actual_rank"]).abs().mean()) if not merged.empty else None
    predicted_top10 = set(pred_unique.sort_values("pred_rank").head(10)["driver_key"].tolist())
    actual_top10 = set(actual_unique[actual_unique["actual_rank"] <= 10]["driver_key"].tolist())
    top10_hit = None
    if actual_top10:
        top10_hit = float(len(predicted_top10.intersection(actual_top10)) / float(min(10, len(actual_top10))))

    return {
        "available": True,
        "rows_predicted": int(len(pred_unique)),
        "rows_actual": int(len(actual_unique)),
        "rows_common": int(len(merged)),
        "mae_on_common": mae,
        "top10_hit": top10_hit,
    }


def _prediction_payload(config: PredictionConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = run_prediction(config)
    rows: list[dict[str, Any]]
    if result.table.empty:
        rows = []
    else:
        rows = json.loads(result.table.to_json(orient="records"))
    payload = {
        "version": result.version,
        "config": {
            "source": config.source,
            "mode": config.mode,
            "year": config.year,
            "round_number": config.round_number,
            "train_seasons": config.train_seasons,
            "include_standings": config.include_standings,
            "enable_dl_candidates": config.enable_dl_candidates,
            "compare_families": config.compare_families,
            "disable_runsim_features": config.disable_runsim_features,
        },
        "rows": rows,
        "notes": result.notes,
        "model_name": result.model_name,
        "model_family": result.model_family,
        "device_used": result.device_used,
        "dl_available": result.dl_available,
        "candidate_leaderboard": result.candidate_leaderboard,
    }
    return payload, rows


def _profile_base_config(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    defaults = profile.get("defaults", {})
    training = profile.get("training", {})
    experiments = profile.get("experiments", {})
    dl = profile.get("dl", {})
    if not isinstance(defaults, dict):
        defaults = {}
    if not isinstance(training, dict):
        training = {}
    if not isinstance(experiments, dict):
        experiments = {}
    if not isinstance(dl, dict):
        dl = {}

    year = _as_int(args.year if args.year is not None else defaults.get("year"), 2025)
    round_number = _as_int(args.round_number if args.round_number is not None else defaults.get("round_number"), 1)
    source = str(args.source or defaults.get("source") or "local").strip().lower()
    include_standings = _as_bool(defaults.get("include_standings", False), False)
    train_seasons = training.get("train_seasons", [year - 3, year - 2, year - 1])
    if not isinstance(train_seasons, list):
        train_seasons = [year - 3, year - 2, year - 1]
    train_seasons = sorted({int(v) for v in train_seasons})
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
    }


def _build_prediction_config(
    *,
    base: dict[str, Any],
    mode: str,
    round_number: int,
    disable_runsim_features: bool,
    enable_dl_candidates: bool,
    compare_families: list[str],
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
    )
    prediction, _ = _prediction_payload(config)
    payload = {
        "profile_name": profile.get("name", "unknown"),
        "mission": profile.get("mission", "unknown"),
        "workflow": "single_prediction",
        "generated_at": _utc_now(),
    }
    payload.update(prediction)
    return payload


def _run_weekend_phase(profile: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    base = _profile_base_config(profile, args)
    defaults = profile.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    phase = str(args.phase or defaults.get("phase", "pre-qualifying"))
    script = Path(__file__).resolve().parent / "run_live_weekend_pipeline.py"
    output_dir = _resolve_project_path(
        args.output_dir or _resolve_profile_value(profile, "outputs", "base_dir", "data/f1/live/2026")
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
    if base.get("cache_dir"):
        cmd.extend(["--cache-dir", str(base["cache_dir"])])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"run_live_weekend_pipeline failed: {proc.stderr.strip() or proc.stdout.strip()}")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    return {
        "profile_name": profile.get("name", "unknown"),
        "mission": profile.get("mission", "unknown"),
        "workflow": "weekend_phase",
        "generated_at": _utc_now(),
        "payload": payload,
    }


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

    output_root_value = _resolve_project_path(args.output_dir or outputs.get("base_dir") or "data/f1/preseason/holdout_2025")
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
                )
                prediction, rows = _prediction_payload(config)
                actual = (
                    provider.get_qualifying_results(base["year"], rnd)
                    if mode_key == "qualifying"
                    else provider.get_race_results(base["year"], rnd)
                )
                evaluation_row = _evaluate_prediction_rows(
                    predicted_rows=rows,
                    actual_results=actual,
                    actual_position_col="position",
                )
                mae = evaluation_row["mae_on_common"] if evaluation_row.get("available") else None
                top10_hit = evaluation_row["top10_hit"] if evaluation_row.get("available") else None
                row_payload = {
                    "round": int(rnd),
                    "mae": mae,
                    "top10_hit": top10_hit,
                    "model_name": prediction["model_name"],
                    "model_family": prediction["model_family"],
                    "device_used": prediction["device_used"],
                    "dl_available": prediction["dl_available"],
                    "rows_common": evaluation_row.get("rows_common"),
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

    payload = {
        "profile_name": profile.get("name", "unknown"),
        "mission": profile.get("mission", "unknown"),
        "workflow": "backtest_ablation_compare",
        "year": int(base["year"]),
        "round_range": [int(round_start), int(round_end)],
        "train_seasons": list(base["train_seasons"]),
        "holdout_strict": bool(holdout_strict),
        "source": str(base["source"]),
        "weekends_dir": str(base["weekends_dir"]),
        "output_dir": str(run_dir),
        "results": results,
        "acceptance": acceptance,
        "generated_at": _utc_now(),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run F1 workflows from YAML/TOML profiles.")
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
    parser.add_argument("--output-format", choices=["text", "json"], default="json")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
