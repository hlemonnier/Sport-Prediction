#!/usr/bin/env python3
"""Walk-forward survival-aware Race Final Position challenger backtest."""

from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from packages.f1.data.providers import LocalWeekendProvider
from packages.f1.domain.starting_grid import RacePredictionHorizon
from packages.f1.models.pre_race.evaluate import evaluate_terminal_status_probabilities
from packages.f1.models.pre_race.joint import SurvivalAwareRaceModel
from packages.f1.models.pre_race.ranking import BradleyTerryOrderRanker, ConditionalOrderConfig
from packages.f1.models.pre_race.status import TerminalStatus, reason_code_terminal_status
from packages.f1.orchestration.model_runtime import f1_model_runtime_doctor
from packages.f1.orchestration.non_live_validation import EventError, evaluate_race_promotion


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(weekends_dir: Path, year: int, round_number: int) -> tuple[dict[str, Any], Path]:
    matches = sorted((weekends_dir / str(year)).glob(f"round_{round_number:02d}_*"))
    if not matches:
        raise FileNotFoundError(f"missing local weekend {year} round {round_number}")
    path = matches[0] / "weekend_metadata.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def _event_as_of(metadata: dict[str, Any]) -> str:
    raw = metadata.get("scheduled_event_date")
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(parsed):
        year = int(metadata["year"])
        round_number = int(metadata["round_number"])
        parsed = pd.Timestamp(year=year, month=1, day=1, tz="UTC") + timedelta(
            days=round_number * 7
        )
    return parsed.isoformat().replace("+00:00", "Z")


def _normalized_circuit(metadata: dict[str, Any]) -> str:
    text = str(metadata.get("event_name") or "unknown").lower()
    return "_".join(part for part in "".join(char if char.isalnum() else " " for char in text).split() if part)


def _rolling_group_mean(
    history: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
    prior_strength: float,
) -> dict[str, float]:
    if history.empty or key_column not in history.columns or value_column not in history.columns:
        return {}
    values = pd.to_numeric(history[value_column], errors="coerce")
    global_mean = float(values.mean()) if values.notna().any() else 0.0
    output: dict[str, float] = {}
    for key, indexes in history.groupby(key_column, dropna=False).groups.items():
        group = values.loc[indexes].dropna()
        if group.empty:
            continue
        output[str(key)] = float(
            (group.sum() + prior_strength * global_mean) / (len(group) + prior_strength)
        )
    return output


def _rolling_rate(
    history: pd.DataFrame,
    *,
    key_column: str,
    mask: pd.Series,
    alpha: float = 1.0,
    beta: float = 4.0,
) -> dict[str, float]:
    if history.empty or key_column not in history.columns:
        return {}
    labels = mask.reindex(history.index).fillna(False).astype(float)
    output: dict[str, float] = {}
    for key, indexes in history.groupby(key_column, dropna=False).groups.items():
        output[str(key)] = float((labels.loc[indexes].sum() + alpha) / (len(indexes) + alpha + beta))
    return output


def _pre_race_red_flag_count(root: Path, metadata: dict[str, Any]) -> int:
    qualifying_order = min(
        (
            int(session.get("session_order", 999))
            for session in metadata.get("sessions", [])
            if str(session.get("session_type", "")).lower() == "qualifying"
        ),
        default=999,
    )
    messages: list[str] = []
    for session in metadata.get("sessions", []):
        if int(session.get("session_order", 999)) > qualifying_order:
            continue
        reference = session.get("race_control_messages_path")
        if not reference:
            continue
        path = Path(str(reference))
        path = path if path.is_absolute() else root / path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for column in ("Flag", "Status", "Message"):
            if column in frame.columns:
                messages.extend(frame[column].fillna("").astype(str).str.lower().tolist())
    return int(sum("red" in value and "flag" in value for value in set(messages)))


def _pre_race_wet_evidence(root: Path, metadata: dict[str, Any]) -> float:
    qualifying_order = min(
        (
            int(session.get("session_order", 999))
            for session in metadata.get("sessions", [])
            if str(session.get("session_type", "")).lower() == "qualifying"
        ),
        default=999,
    )
    evidence: list[float] = []
    for session in metadata.get("sessions", []):
        if int(session.get("session_order", 999)) > qualifying_order:
            continue
        reference = session.get("weather_path")
        if not reference:
            continue
        path = Path(str(reference))
        path = path if path.is_absolute() else root / path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "Rainfall" in frame.columns:
            rainfall = pd.to_numeric(frame["Rainfall"], errors="coerce")
            if rainfall.notna().any():
                evidence.append(float(rainfall.fillna(0.0).gt(0.0).mean()))
    return float(max(evidence, default=0.0))


def _team_column(frame: pd.DataFrame) -> str | None:
    return next((column for column in ("team_name", "fp_team_name", "fp1_team_name") if column in frame.columns), None)


def _build_event_rows(
    *,
    root: Path,
    provider: LocalWeekendProvider,
    weekends_dir: Path,
    year: int,
    round_number: int,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    metadata, metadata_path = _metadata(weekends_dir, year, round_number)
    qualifying = provider.get_qualifying_results(year, round_number)
    race = provider.get_race_results(year, round_number)
    practice = provider.get_fp_features(year, round_number, prediction_target="race")
    if qualifying.empty or race.empty:
        raise ValueError(f"{year} round {round_number} has incomplete Q/Race classifications")

    q = qualifying[[column for column in ("driver_id", "position", "team_name") if column in qualifying]].copy()
    q = q.rename(columns={"position": "qualy_position"})
    r = race[
        [
            column
            for column in (
                "driver_id",
                "position",
                "team_name",
                "race_status_raw",
                "race_status_evidence_complete",
                "retirement_fraction",
            )
            if column in race
        ]
    ].copy()
    r = r.rename(columns={"position": "finish_position", "team_name": "race_team_name"})
    frame = q.merge(r, on="driver_id", how="outer", validate="one_to_one")
    if not practice.empty:
        frame = frame.merge(practice, on="driver_id", how="left", validate="one_to_one")
    frame["team_name"] = frame.get("team_name", pd.Series(index=frame.index, dtype=object)).where(
        frame.get("team_name", pd.Series(index=frame.index, dtype=object)).notna(),
        frame.get("race_team_name", pd.Series(index=frame.index, dtype=object)),
    )
    frame["team_name"] = frame["team_name"].fillna("unknown_team").astype(str)
    event_key = int(year) * 100 + int(round_number)
    frame["event_key"] = event_key
    frame["event_as_of"] = _event_as_of(metadata)
    frame["feature_as_of"] = _event_as_of(metadata)
    frame["circuit_id"] = _normalized_circuit(metadata)
    frame["grid_position"] = pd.to_numeric(frame["qualy_position"], errors="coerce")
    frame["grid_status"] = "grid"
    frame["grid_starter_eligible"] = 1.0
    frame["grid_pit_lane_start"] = False
    frame["race_information_horizon"] = RacePredictionHorizon.POST_QUALIFYING_PRE_GRID.value
    race_only = frame["qualy_position"].isna() & frame["finish_position"].notna()
    if race_only.any():
        drivers = sorted(frame.loc[race_only, "driver_id"].astype(str).tolist())
        raise ValueError(
            "post-Qualifying proxy cannot score Race-only entrants without a causal grid anchor: "
            f"{drivers}"
        )
    qualifying_only = frame["race_status_raw"].isna() & frame["qualy_position"].notna()
    if qualifying_only.any():
        # A Qualifying entrant absent from the authoritative Race classification
        # is retained as a target-population DNS/withdrawal at the tail.  This
        # target repair is never exposed to the inference feature frame.
        frame.loc[qualifying_only, "race_status_raw"] = "Did not start"
        frame.loc[qualifying_only, "race_status_evidence_complete"] = True
        used = {
            int(value)
            for value in pd.to_numeric(frame["finish_position"], errors="coerce").dropna()
        }
        next_position = 1
        for index in frame.index[qualifying_only]:
            while next_position in used:
                next_position += 1
            frame.loc[index, "finish_position"] = next_position
            frame.loc[index, "retirement_fraction"] = 0.0
            used.add(next_position)
    frame["terminal_status"] = frame["race_status_raw"].map(reason_code_terminal_status)
    if frame["terminal_status"].isna().any():
        unresolved = sorted(frame.loc[frame["terminal_status"].isna(), "race_status_raw"].astype(str).unique())
        raise ValueError(f"unresolved terminal status labels: {unresolved}")
    frame["terminal_status"] = frame["terminal_status"].map(lambda value: value.value)

    places_gained = pd.to_numeric(history.get("grid_position"), errors="coerce") - pd.to_numeric(
        history.get("finish_position"), errors="coerce"
    ) if not history.empty else pd.Series(dtype=float)
    prior = history.copy()
    prior["places_gained"] = places_gained
    team_strength = _rolling_group_mean(
        prior, key_column="team_name", value_column="places_gained", prior_strength=8.0
    )
    driver_strength = _rolling_group_mean(
        prior, key_column="driver_id", value_column="places_gained", prior_strength=5.0
    )
    terminal_mask = (
        prior.get("terminal_status", pd.Series(index=prior.index, dtype=object)).astype(str)
        != TerminalStatus.CLASSIFIED_FINISH.value
    )
    mechanical_mask = prior.get(
        "terminal_status", pd.Series(index=prior.index, dtype=object)
    ).astype(str).eq(TerminalStatus.MECHANICAL_POWER_UNIT.value)
    incident_mask = prior.get(
        "terminal_status", pd.Series(index=prior.index, dtype=object)
    ).astype(str).eq(TerminalStatus.COLLISION_INCIDENT.value)
    team_mechanical = _rolling_rate(
        prior, key_column="team_name", mask=mechanical_mask
    )
    driver_incident = _rolling_rate(
        prior, key_column="driver_id", mask=incident_mask
    )
    circuit_dnf = _rolling_rate(prior, key_column="circuit_id", mask=terminal_mask)

    frame["race_team_strength_score"] = frame["team_name"].map(team_strength).fillna(0.0)
    frame["race_driver_strength_score"] = frame["driver_id"].astype(str).map(driver_strength).fillna(0.0)
    frame["race_team_mechanical_rate"] = frame["team_name"].map(team_mechanical)
    frame["race_driver_incident_rate"] = frame["driver_id"].astype(str).map(driver_incident)
    frame["race_circuit_dnf_rate"] = frame["circuit_id"].map(circuit_dnf)
    frame["race_weekend_stoppage_count"] = float(_pre_race_red_flag_count(root, metadata))
    frame["race_wet_probability"] = _pre_race_wet_evidence(root, metadata)

    if "fp_race_sim_delta" in frame.columns:
        frame["race_long_run_pace_delta"] = pd.to_numeric(frame["fp_race_sim_delta"], errors="coerce")
        frame["race_teammate_long_run_delta"] = frame["race_long_run_pace_delta"] - frame.groupby(
            "team_name", dropna=False
        )["race_long_run_pace_delta"].transform("median")
    if "fp_race_sim_evidence_share" in frame.columns:
        frame["race_long_run_evidence_share"] = pd.to_numeric(
            frame["fp_race_sim_evidence_share"], errors="coerce"
        )
    if "fp_race_sim_laps" in frame.columns:
        frame["race_longest_clean_stint_laps"] = pd.to_numeric(
            frame["fp_race_sim_laps"], errors="coerce"
        )
    if "fp_race_sim_raw_degradation_mad" in frame.columns:
        frame["race_long_run_uncertainty"] = pd.to_numeric(
            frame["fp_race_sim_raw_degradation_mad"], errors="coerce"
        )
    if "sprint_race_sim_delta" in frame.columns:
        frame["race_sprint_pace_delta"] = pd.to_numeric(
            frame["sprint_race_sim_delta"], errors="coerce"
        )
    if "track_finish_order_mobility" not in frame.columns:
        same_circuit = prior.loc[prior.get("circuit_id", pd.Series(index=prior.index)).eq(frame["circuit_id"].iloc[0])]
        if same_circuit.empty:
            mobility = 0.5
        else:
            mobility = float(
                (
                    pd.to_numeric(same_circuit["finish_position"], errors="coerce")
                    - pd.to_numeric(same_circuit["grid_position"], errors="coerce")
                ).abs().mean()
                / max(1.0, len(frame))
            )
        frame["track_finish_order_mobility"] = float(np.clip(mobility, 0.0, 1.0))

    input_paths = [metadata_path]
    for session in metadata.get("sessions", []):
        for key in ("laps_path", "results_path", "weather_path", "race_control_messages_path"):
            reference = session.get(key)
            if not reference:
                continue
            path = Path(str(reference))
            path = path if path.is_absolute() else root / path
            if path.exists():
                input_paths.append(path)
    info = {
        "event_key": event_key,
        "year": int(year),
        "round": int(round_number),
        "event_name": str(metadata.get("event_name") or f"Round {round_number}"),
        "event_format": str(metadata.get("event_format") or "unknown"),
        "field_size": int(len(frame)),
        "information_horizon": RacePredictionHorizon.POST_QUALIFYING_PRE_GRID.value,
        "final_grid_snapshot_used": False,
    }
    return frame, info, input_paths


def _binary_metrics(actual: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(actual, dtype=float)
    return (
        float(np.mean(np.square(p - y))),
        float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
    )


def _mean(events: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(event[key]) for event in events]))


def run(
    *,
    weekends_dir: Path,
    years: Sequence[int],
    evaluation_years: Sequence[int],
    simulations: int,
    plackett_luce_temperature: float,
    order_residual_weight: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    root = _root()
    provider = LocalWeekendProvider(str(weekends_dir))
    history = pd.DataFrame()
    event_frames: dict[int, pd.DataFrame] = {}
    event_info: dict[int, dict[str, Any]] = {}
    inputs: set[Path] = set()
    for year in sorted(set(int(value) for value in years)):
        for item in provider.list_rounds(year):
            round_number = int(item["round_number"])
            frame, info, event_inputs = _build_event_rows(
                root=root,
                provider=provider,
                weekends_dir=weekends_dir,
                year=year,
                round_number=round_number,
                history=history,
            )
            event_frames[int(info["event_key"])] = frame
            event_info[int(info["event_key"])] = info
            inputs.update(event_inputs)
            history = pd.concat([history, frame], ignore_index=True)

    evaluation_set = set(int(value) for value in evaluation_years)
    events: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for event_key in sorted(event_frames):
        if event_key // 100 not in evaluation_set:
            continue
        prior = history.loc[pd.to_numeric(history["event_key"], errors="coerce").lt(event_key)].copy()
        current = event_frames[event_key].copy()
        if prior["event_key"].nunique() < 4:
            continue
        model = SurvivalAwareRaceModel(
            order_model=BradleyTerryOrderRanker(
                ConditionalOrderConfig(residual_weight=float(order_residual_weight))
            )
        ).fit(
            prior,
            cutoff=current["event_as_of"].iloc[0],
        )
        forecast = model.predict_joint(
            current.drop(
                columns=[
                    "finish_position",
                    "terminal_status",
                    "race_status_raw",
                    "race_status_evidence_complete",
                    "retirement_fraction",
                    "laps_completed",
                ],
                errors="ignore",
            ),
            horizon=RacePredictionHorizon.POST_QUALIFYING_PRE_GRID,
            prediction_as_of=current["event_as_of"].iloc[0],
            simulations=int(simulations),
            seed=int(seed) + event_key,
            plackett_luce_temperature=float(plackett_luce_temperature),
        )
        scored = current[
            ["driver_id", "grid_position", "finish_position", "terminal_status"]
        ].merge(
            forecast.point_classification[
                ["driver_id", "predicted_position", "predicted_terminal_status", "expected_position"]
            ],
            on="driver_id",
            validate="one_to_one",
        ).merge(
            forecast.status_probabilities,
            on="driver_id",
            validate="one_to_one",
        ).merge(
            forecast.position_probabilities,
            on="driver_id",
            validate="one_to_one",
        )
        actual_position = pd.to_numeric(scored["finish_position"], errors="coerce")
        baseline_position = pd.to_numeric(scored["grid_position"], errors="coerce")
        candidate_position = pd.to_numeric(scored["predicted_position"], errors="coerce")
        actual_terminal = scored["terminal_status"].ne(TerminalStatus.CLASSIFIED_FINISH.value).astype(float)
        candidate_terminal = pd.to_numeric(scored["p_terminal"], errors="coerce")
        historical_terminal = prior["terminal_status"].astype(str).ne(
            TerminalStatus.CLASSIFIED_FINISH.value
        ).astype(float)
        baseline_terminal_probability = float(
            (historical_terminal.sum() + 2.0) / (len(historical_terminal) + 10.0)
        )
        baseline_probability = np.full(len(scored), baseline_terminal_probability, dtype=float)
        baseline_brier, baseline_log_loss = _binary_metrics(
            actual_terminal.to_numpy(dtype=float), baseline_probability
        )
        candidate_brier, candidate_log_loss = _binary_metrics(
            actual_terminal.to_numpy(dtype=float), candidate_terminal.to_numpy(dtype=float)
        )
        status_evaluation = evaluate_terminal_status_probabilities(
            scored[["driver_id", "terminal_status"]],
            forecast.status_probabilities,
        )
        event = {
            **event_info[event_key],
            "training_events": int(prior["event_key"].nunique()),
            "baseline_mae": float((baseline_position - actual_position).abs().mean()),
            "candidate_mae": float((candidate_position - actual_position).abs().mean()),
            "baseline_kendall": float(baseline_position.corr(actual_position, method="kendall")),
            "candidate_kendall": float(candidate_position.corr(actual_position, method="kendall")),
            "baseline_status_brier": baseline_brier,
            "candidate_status_brier": candidate_brier,
            "baseline_status_log_loss": baseline_log_loss,
            "candidate_status_log_loss": candidate_log_loss,
            "candidate_status_multiclass_brier": status_evaluation["multiclass_brier"],
            "candidate_status_multiclass_log_loss": status_evaluation["multiclass_log_loss"],
            "candidate_reason_recall": status_evaluation["reason_recall"],
            "candidate_terminal_calibration": status_evaluation["terminal_calibration"],
            "actual_terminal_rate": float(actual_terminal.mean()),
            "baseline_terminal_probability": baseline_terminal_probability,
            "candidate_mean_terminal_probability": float(candidate_terminal.mean()),
            "legal_permutation": sorted(candidate_position.astype(int).tolist()) == list(range(1, len(scored) + 1)),
            "entrant_coverage": float(len(scored) / len(current)),
        }
        event["delta_candidate_minus_baseline"] = event["candidate_mae"] - event["baseline_mae"]
        events.append(event)
        for row in scored.to_dict(orient="records"):
            prediction_rows.append({**event_info[event_key], **row})

    if not events:
        raise ValueError("no Race evaluation events were scored")
    audit_year = max(evaluation_set)
    audit_events = [event for event in events if int(event["year"]) == audit_year]
    if len(audit_events) < 2:
        raise ValueError(f"audit year {audit_year} has fewer than two scored events")
    promotion = evaluate_race_promotion(
        [
            EventError(
                str(event["event_key"]),
                float(event["baseline_mae"]),
                float(event["candidate_mae"]),
                "sprint" if "sprint" in str(event["event_format"]).lower() else "standard",
            )
            for event in audit_events
        ],
        baseline_kendall=_mean(audit_events, "baseline_kendall"),
        candidate_kendall=_mean(audit_events, "candidate_kendall"),
        baseline_status_brier=_mean(audit_events, "baseline_status_brier"),
        candidate_status_brier=_mean(audit_events, "candidate_status_brier"),
        baseline_status_log_loss=_mean(audit_events, "baseline_status_log_loss"),
        candidate_status_log_loss=_mean(audit_events, "candidate_status_log_loss"),
        entrant_coverage=min(float(event["entrant_coverage"]) for event in audit_events),
        all_classifications_legal=all(bool(event["legal_permutation"]) for event in audit_events),
        bootstrap_samples=int(bootstrap_samples),
        seed=int(seed),
    )
    implementation_paths = [
        Path(__file__).resolve(),
        root / "packages/f1/models/pre_race/joint.py",
        root / "packages/f1/models/pre_race/ranking.py",
        root / "packages/f1/models/pre_race/survival.py",
        root / "packages/f1/models/pre_race/status.py",
        root / "packages/f1/features/race.py",
        root / "packages/f1/orchestration/non_live_validation.py",
    ]
    return {
        "schema_version": "f1_race_survival_order_event_block_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": "race_final_position",
        "target": "official_terminal_race_classification_and_status",
        "protocol": {
            "training": "strictly_earlier_complete_events",
            "years_loaded": sorted(set(int(value) for value in years)),
            "evaluation_years": sorted(evaluation_set),
            "promotion_audit_year": audit_year,
            "horizon": RacePredictionHorizon.POST_QUALIFYING_PRE_GRID.value,
            "baseline_order": "grand_prix_qualifying_order_proxy",
            "baseline_status": "causal_beta_smoothed_rolling_terminal_rate",
            "final_grid_claimed": False,
            "simulations": int(simulations),
            "plackett_luce_temperature": float(plackett_luce_temperature),
            "order_residual_weight": float(order_residual_weight),
        },
        "aggregate": {
            "events": len(events),
            "baseline_mean_mae": _mean(events, "baseline_mae"),
            "candidate_mean_mae": _mean(events, "candidate_mae"),
            "baseline_mean_kendall": _mean(events, "baseline_kendall"),
            "candidate_mean_kendall": _mean(events, "candidate_kendall"),
            "baseline_status_brier": _mean(events, "baseline_status_brier"),
            "candidate_status_brier": _mean(events, "candidate_status_brier"),
            "baseline_status_log_loss": _mean(events, "baseline_status_log_loss"),
            "candidate_status_log_loss": _mean(events, "candidate_status_log_loss"),
            "by_year": {
                str(year): {
                    "events": len([event for event in events if int(event["year"]) == year]),
                    "baseline_mean_mae": _mean(
                        [event for event in events if int(event["year"]) == year], "baseline_mae"
                    ),
                    "candidate_mean_mae": _mean(
                        [event for event in events if int(event["year"]) == year], "candidate_mae"
                    ),
                }
                for year in sorted({int(event["year"]) for event in events})
            },
        },
        "promotion": promotion.to_payload(),
        "runtime": f1_model_runtime_doctor(),
        "events": events,
        "predictions": prediction_rows,
        "input_manifest": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in sorted(inputs)
        ],
        "implementation_manifest": [
            {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
            for path in implementation_paths
        ],
    }


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekends-dir", type=Path, default=_root() / "data/f1/raw/weekends")
    parser.add_argument("--years", type=_csv_ints, default=(2022, 2023, 2024, 2025, 2026))
    parser.add_argument("--evaluation-years", type=_csv_ints, default=(2025, 2026))
    parser.add_argument("--simulations", type=int, default=2_000)
    parser.add_argument("--plackett-luce-temperature", type=float, default=0.25)
    parser.add_argument(
        "--order-residual-weight",
        type=float,
        default=0.45,
        help="fixed on 2025 transfer validation before the 2026 audit",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--output",
        type=Path,
        default=_root() / "artifacts/backtests/f1/race_final_position/survival_order_v1.json",
    )
    args = parser.parse_args()
    payload = run(
        weekends_dir=args.weekends_dir.expanduser().resolve(),
        years=args.years,
        evaluation_years=args.evaluation_years,
        simulations=args.simulations,
        plackett_luce_temperature=args.plackett_luce_temperature,
        order_residual_weight=args.order_residual_weight,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = args.output.expanduser()
    if not output.is_absolute():
        output = _root() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": payload["aggregate"], "promotion": payload["promotion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Suggested commit name: feat(f1-race): add survival-aware order walk-forward evidence
