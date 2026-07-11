"""Locked-replay calibration for live-race filter and Monte Carlo priors."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from packages.f1.models.live_race.state import (
    FilterConfig,
    build_event_lap_baseline,
    parse_track_status,
)


CALIBRATION_ARTIFACT_VERSION = "live_race_locked_replay_calibration_v2_component_scope"
REGIMES: tuple[str, ...] = ("green", "yellow", "sc_vsc")


@dataclass(frozen=True)
class MonteCarloPriorConfig:
    """Regime-transition and pit-loss priors used by race rollouts."""

    green_to_sc_vsc: float = 0.035
    green_to_yellow: float = 0.050
    yellow_to_sc_vsc: float = 0.220
    yellow_to_green: float = 0.250
    sc_vsc_to_green: float = 0.380
    sc_vsc_to_yellow: float = 0.140
    pit_loss_green_mean: float = 21.0
    pit_loss_yellow_mean: float = 15.5
    pit_loss_sc_vsc_mean: float = 11.0
    pit_loss_green_std: float = 1.4
    pit_loss_yellow_std: float = 1.4
    pit_loss_sc_vsc_std: float = 1.4
    calibration_mode: str = "hand_prior"
    calibration_source_id: str | None = None
    transition_rows: int = 0
    pit_rows: int = 0
    calibration_version: str = CALIBRATION_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        transition_pairs = (
            (self.green_to_sc_vsc, self.green_to_yellow),
            (self.yellow_to_sc_vsc, self.yellow_to_green),
            (self.sc_vsc_to_green, self.sc_vsc_to_yellow),
        )
        for left, right in transition_pairs:
            if not 0.0 <= float(left) <= 1.0 or not 0.0 <= float(right) <= 1.0:
                raise ValueError("regime transition probabilities must be between zero and one")
            if float(left) + float(right) > 1.0:
                raise ValueError("outgoing regime transition probabilities cannot exceed one")
        for value in (
            self.pit_loss_green_mean,
            self.pit_loss_yellow_mean,
            self.pit_loss_sc_vsc_mean,
            self.pit_loss_green_std,
            self.pit_loss_yellow_std,
            self.pit_loss_sc_vsc_std,
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError("pit-loss means and standard deviations must be positive and finite")
        mode = str(self.calibration_mode or "").strip().lower()
        if mode not in {"hand_prior", "locked_replay"}:
            raise ValueError("calibration_mode must be hand_prior or locked_replay")
        object.__setattr__(self, "calibration_mode", mode)
        if mode == "locked_replay":
            if not str(self.calibration_source_id or "").strip():
                raise ValueError("locked_replay MC calibration requires calibration_source_id")
            if int(self.transition_rows) <= 0 or int(self.pit_rows) <= 0:
                raise ValueError("locked_replay MC calibration requires transition and pit rows")

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_mode == "locked_replay"

    @property
    def promotion_ready(self) -> bool:
        return self.is_calibrated

    def transition_probabilities(self, regime: str) -> dict[str, float]:
        current = str(regime or "green").lower()
        if current == "sc_vsc":
            return {
                "green": float(self.sc_vsc_to_green),
                "yellow": float(self.sc_vsc_to_yellow),
                "sc_vsc": float(1.0 - self.sc_vsc_to_green - self.sc_vsc_to_yellow),
            }
        if current == "yellow":
            return {
                "green": float(self.yellow_to_green),
                "yellow": float(1.0 - self.yellow_to_green - self.yellow_to_sc_vsc),
                "sc_vsc": float(self.yellow_to_sc_vsc),
            }
        return {
            "green": float(1.0 - self.green_to_sc_vsc - self.green_to_yellow),
            "yellow": float(self.green_to_yellow),
            "sc_vsc": float(self.green_to_sc_vsc),
        }

    def pit_loss_parameters(self, regime: str) -> tuple[float, float]:
        current = str(regime or "green").lower()
        if current == "sc_vsc":
            return float(self.pit_loss_sc_vsc_mean), float(self.pit_loss_sc_vsc_std)
        if current == "yellow":
            return float(self.pit_loss_yellow_mean), float(self.pit_loss_yellow_std)
        return float(self.pit_loss_green_mean), float(self.pit_loss_green_std)

    def diagnostics(self) -> dict[str, object]:
        return {
            "calibration_mode": self.calibration_mode,
            "calibration_source_id": self.calibration_source_id,
            "transition_rows": int(self.transition_rows),
            "pit_rows": int(self.pit_rows),
            "calibration_version": self.calibration_version,
            "is_calibrated": self.is_calibrated,
            "promotion_ready": self.promotion_ready,
            "uses_hand_tuned_priors": not self.is_calibrated,
        }


@dataclass(frozen=True)
class LiveRaceCalibrationBundle:
    filter_config: FilterConfig
    monte_carlo_priors: MonteCarloPriorConfig
    source_id: str
    source_rows: int
    artifact_version: str = CALIBRATION_ARTIFACT_VERSION

    @property
    def prior_calibration_ready(self) -> bool:
        return bool(self.filter_config.promotion_ready and self.monte_carlo_priors.promotion_ready)

    @property
    def promotion_ready(self) -> bool:
        """Model-level promotion is never implied by a prior-only artifact."""

        return False

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "calibration_mode": "locked_replay" if self.prior_calibration_ready else "hand_prior",
            "source_id": self.source_id,
            "source_rows": int(self.source_rows),
            "prior_calibration_ready": self.prior_calibration_ready,
            "model_promotion_ready": False,
            "promotion_ready": False,
            "promotion_scope": "prior_components_only",
            "promotion_blockers": [
                "locked_model_replay_and_comparator_evidence_required",
                "heuristic_strategy_template_probabilities_not_calibrated",
            ],
            "uses_hand_tuned_priors": not self.prior_calibration_ready,
            "filter_config": asdict(self.filter_config),
            "monte_carlo_priors": asdict(self.monte_carlo_priors),
        }


def _required_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} requires columns: {tuple(missing)}")


def _regime(value: object) -> str:
    flags = parse_track_status(value)
    if flags.is_sc_vsc or flags.is_red:
        return "sc_vsc"
    if flags.is_yellow:
        return "yellow"
    return "green"


def fit_filter_config_from_locked_replay(
    laps: pd.DataFrame,
    *,
    source_id: str,
    min_clean_rows: int = 30,
) -> FilterConfig:
    """Estimate phi/Q/R from clean, ordered, locked historical replay rows."""

    if not isinstance(laps, pd.DataFrame) or laps.empty:
        raise ValueError("locked replay laps must be a non-empty DataFrame")
    _required_columns(
        laps,
        ("event_key", "driver_id", "lap_number", "lap_time_seconds", "track_status"),
        "filter calibration",
    )
    if not str(source_id or "").strip():
        raise ValueError("source_id is required for locked replay calibration")
    frame = laps.copy()
    frame["lap_number"] = pd.to_numeric(frame["lap_number"], errors="coerce")
    frame["lap_time_seconds"] = pd.to_numeric(frame["lap_time_seconds"], errors="coerce")
    accurate = frame.get("is_accurate", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    box = frame.get("is_box_lap", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    green = frame["track_status"].map(lambda value: _regime(value) == "green")
    clean = frame[
        accurate
        & ~box
        & green
        & frame["lap_number"].notna()
        & frame["lap_time_seconds"].notna()
        & (frame["lap_time_seconds"] > 0.0)
    ].copy()
    if len(clean) < int(min_clean_rows):
        raise ValueError(f"insufficient clean locked replay rows: {len(clean)} < {int(min_clean_rows)}")

    clean["_baseline"] = np.nan
    for _, event in clean.groupby("event_key", sort=False):
        baseline = build_event_lap_baseline(event, min_clean_obs_per_lap=2)
        clean.loc[event.index, "_baseline"] = event["lap_number"].map(baseline.value_at)
    clean["_residual"] = clean["lap_time_seconds"] - clean["_baseline"]
    group_columns = ["event_key", "driver_id"]
    if "stint_id" in clean.columns:
        group_columns.append("stint_id")

    deg_previous: list[float] = []
    deg_current: list[float] = []
    pace_innovations: list[float] = []
    observation_residuals: list[float] = []
    for _, group in clean.groupby(group_columns, sort=False, dropna=False):
        ordered = group.sort_values("lap_number", kind="mergesort")
        residual = pd.to_numeric(ordered["_residual"], errors="coerce").dropna().to_numpy(dtype=float)
        if residual.size < 4:
            continue
        differences = np.diff(residual)
        deg_previous.extend(differences[:-1].tolist())
        deg_current.extend(differences[1:].tolist())
        pace_innovations.extend((differences[1:] - differences[:-1]).tolist())
        x = np.arange(residual.size, dtype=float)
        slope, intercept = np.polyfit(x, residual, deg=1)
        observation_residuals.extend((residual - (intercept + slope * x)).tolist())

    previous = np.asarray(deg_previous, dtype=float)
    current = np.asarray(deg_current, dtype=float)
    valid = np.isfinite(previous) & np.isfinite(current)
    previous, current = previous[valid], current[valid]
    if previous.size < max(10, int(min_clean_rows // 4)):
        raise ValueError("insufficient sequential locked replay rows for filter dynamics calibration")
    denominator = float(np.dot(previous, previous))
    phi = float(np.dot(previous, current) / denominator) if denominator > 1e-9 else 0.0
    phi = float(np.clip(phi, 0.05, 0.98))
    q_deg = float(np.std(current - phi * previous, ddof=1))
    q_pace = float(np.std(np.asarray(pace_innovations, dtype=float), ddof=1))
    obs = np.asarray(observation_residuals, dtype=float)
    obs = obs[np.isfinite(obs)]
    if obs.size < 10:
        raise ValueError("insufficient locked replay residuals for observation variance calibration")
    r_obs = float(np.var(obs, ddof=1))
    return FilterConfig(
        phi=phi,
        q_pace=float(np.clip(q_pace, 0.005, 2.0)),
        q_deg=float(np.clip(q_deg, 0.002, 0.5)),
        r_obs=float(np.clip(r_obs, 0.0025, 25.0)),
        calibration_mode="locked_replay",
        calibration_source_id=str(source_id),
        calibration_rows=int(len(clean)),
    )


def fit_monte_carlo_priors_from_locked_replay(
    laps: pd.DataFrame,
    *,
    source_id: str,
    min_transitions_per_regime: int = 5,
    min_pit_rows_per_regime: int = 3,
) -> MonteCarloPriorConfig:
    """Estimate regime transitions and regime-specific pit losses."""

    if not isinstance(laps, pd.DataFrame) or laps.empty:
        raise ValueError("locked replay laps must be a non-empty DataFrame")
    _required_columns(laps, ("event_key", "lap_number", "track_status"), "MC prior calibration")
    if not str(source_id or "").strip():
        raise ValueError("source_id is required for locked replay calibration")
    frame = laps.copy()
    frame["lap_number"] = pd.to_numeric(frame["lap_number"], errors="coerce")
    lap_status = (
        frame.dropna(subset=["lap_number"])
        .sort_values(["event_key", "lap_number"], kind="mergesort")
        .drop_duplicates(["event_key", "lap_number"], keep="first")
    )
    counts = {regime: {target: 0 for target in REGIMES} for regime in REGIMES}
    for _, event in lap_status.groupby("event_key", sort=False):
        regimes = [_regime(value) for value in event["track_status"].tolist()]
        laps_ordered = event["lap_number"].to_numpy(dtype=float)
        for idx in range(len(regimes) - 1):
            if int(laps_ordered[idx + 1]) != int(laps_ordered[idx]) + 1:
                continue
            counts[regimes[idx]][regimes[idx + 1]] += 1
    totals = {regime: int(sum(counts[regime].values())) for regime in REGIMES}
    insufficient = {regime: total for regime, total in totals.items() if total < int(min_transitions_per_regime)}
    if insufficient:
        raise ValueError(f"insufficient regime transition evidence: {insufficient}")

    pit_column = next(
        (column for column in ("observed_pit_loss_seconds", "pit_loss_seconds") if column in frame.columns),
        None,
    )
    if pit_column is None:
        raise ValueError("MC prior calibration requires observed_pit_loss_seconds or pit_loss_seconds")
    frame["_pit_loss"] = pd.to_numeric(frame[pit_column], errors="coerce")
    frame["_regime"] = frame["track_status"].map(_regime)
    pit_stats: dict[str, tuple[float, float, int]] = {}
    for regime in REGIMES:
        values = frame.loc[frame["_regime"] == regime, "_pit_loss"].dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size < int(min_pit_rows_per_regime):
            raise ValueError(
                f"insufficient {regime} pit-loss evidence: {int(values.size)} < {int(min_pit_rows_per_regime)}"
            )
        pit_stats[regime] = (
            float(np.mean(values)),
            float(max(np.std(values, ddof=1), 0.1)),
            int(values.size),
        )

    def probabilities(source: str) -> dict[str, float]:
        alpha = 1.0
        denominator = float(totals[source] + alpha * len(REGIMES))
        return {target: float((counts[source][target] + alpha) / denominator) for target in REGIMES}

    green = probabilities("green")
    yellow = probabilities("yellow")
    sc_vsc = probabilities("sc_vsc")
    return MonteCarloPriorConfig(
        green_to_sc_vsc=green["sc_vsc"],
        green_to_yellow=green["yellow"],
        yellow_to_sc_vsc=yellow["sc_vsc"],
        yellow_to_green=yellow["green"],
        sc_vsc_to_green=sc_vsc["green"],
        sc_vsc_to_yellow=sc_vsc["yellow"],
        pit_loss_green_mean=pit_stats["green"][0],
        pit_loss_yellow_mean=pit_stats["yellow"][0],
        pit_loss_sc_vsc_mean=pit_stats["sc_vsc"][0],
        pit_loss_green_std=pit_stats["green"][1],
        pit_loss_yellow_std=pit_stats["yellow"][1],
        pit_loss_sc_vsc_std=pit_stats["sc_vsc"][1],
        calibration_mode="locked_replay",
        calibration_source_id=str(source_id),
        transition_rows=int(sum(totals.values())),
        pit_rows=int(sum(item[2] for item in pit_stats.values())),
    )


def fit_live_race_calibration_from_locked_replay(
    laps: pd.DataFrame,
    *,
    source_id: str,
    min_clean_filter_rows: int = 30,
    min_transitions_per_regime: int = 5,
    min_pit_rows_per_regime: int = 3,
) -> LiveRaceCalibrationBundle:
    filter_config = fit_filter_config_from_locked_replay(
        laps,
        source_id=source_id,
        min_clean_rows=min_clean_filter_rows,
    )
    priors = fit_monte_carlo_priors_from_locked_replay(
        laps,
        source_id=source_id,
        min_transitions_per_regime=min_transitions_per_regime,
        min_pit_rows_per_regime=min_pit_rows_per_regime,
    )
    return LiveRaceCalibrationBundle(
        filter_config=filter_config,
        monte_carlo_priors=priors,
        source_id=str(source_id),
        source_rows=int(len(laps)),
    )


def write_live_race_calibration(bundle: LiveRaceCalibrationBundle, path: str | Path) -> Path:
    if not bundle.prior_calibration_ready:
        raise ValueError("only fully locked-replay-calibrated bundles may be written")
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_live_race_calibration(path: str | Path) -> LiveRaceCalibrationBundle:
    input_path = Path(path).expanduser()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("artifact_version") != CALIBRATION_ARTIFACT_VERSION:
        raise ValueError("unsupported live-race calibration artifact")
    filter_payload = payload.get("filter_config")
    priors_payload = payload.get("monte_carlo_priors")
    if not isinstance(filter_payload, Mapping) or not isinstance(priors_payload, Mapping):
        raise ValueError("live-race calibration artifact is missing filter or MC priors")
    bundle = LiveRaceCalibrationBundle(
        filter_config=FilterConfig(**dict(filter_payload)),
        monte_carlo_priors=MonteCarloPriorConfig(**dict(priors_payload)),
        source_id=str(payload.get("source_id") or ""),
        source_rows=int(payload.get("source_rows") or 0),
        artifact_version=str(payload.get("artifact_version")),
    )
    if (
        not bundle.prior_calibration_ready
        or payload.get("prior_calibration_ready") is not True
        or payload.get("model_promotion_ready") is not False
        or payload.get("promotion_ready") is not False
    ):
        raise ValueError("live-race calibration artifact is not valid for prior-component calibration")
    return bundle


__all__ = [
    "CALIBRATION_ARTIFACT_VERSION",
    "LiveRaceCalibrationBundle",
    "MonteCarloPriorConfig",
    "fit_filter_config_from_locked_replay",
    "fit_live_race_calibration_from_locked_replay",
    "fit_monte_carlo_priors_from_locked_replay",
    "load_live_race_calibration",
    "write_live_race_calibration",
]
