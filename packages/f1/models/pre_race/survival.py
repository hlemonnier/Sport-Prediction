"""Causal partial-pooled terminal-status hazard for pre-Race forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from packages.f1.features.race import engineer_survival_aware_race_features
from packages.f1.models.pre_race.status import (
    TERMINAL_STATUSES,
    TerminalStatus,
    reason_code_terminal_status,
)


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return pd.Timestamp(parsed)


def _clip_probability(value: float, epsilon: float = 1e-8) -> float:
    return float(np.clip(float(value), epsilon, 1.0 - epsilon))


def _logit(value: float) -> float:
    p = _clip_probability(value)
    return float(np.log(p / (1.0 - p)))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0))))


@dataclass(frozen=True)
class TerminalHazardConfig:
    """Regularization and recency contract for the inspectable hazard."""

    prior_strength: float = 12.0
    recency_half_life_days: float = 365.0
    minimum_probability: float = 1e-5
    team_weight: float = 0.35
    driver_weight: float = 0.20
    power_unit_weight: float = 0.20
    circuit_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.prior_strength <= 0.0:
            raise ValueError("prior_strength must be positive")
        if self.recency_half_life_days <= 0.0:
            raise ValueError("recency_half_life_days must be positive")
        if not 0.0 < self.minimum_probability < 0.1:
            raise ValueError("minimum_probability must be in (0, 0.1)")


@dataclass(frozen=True)
class _Posterior:
    probabilities: np.ndarray
    support: float


class PartialPooledTerminalHazard:
    """Reason-coded beta/Dirichlet hazard with causal recency weighting.

    ``fit`` consumes complete historical event blocks.  Supplying ``cutoff``
    enforces a strict walk-forward boundary: rows at or after the forecast
    cutoff are excluded.  Team, driver, power-unit and circuit posteriors shrink
    to a global Dirichlet distribution rather than overfitting rare histories.
    """

    backend = "partial_pooled_beta_dirichlet_v1"

    def __init__(self, config: TerminalHazardConfig | None = None) -> None:
        self.config = config or TerminalHazardConfig()
        self._fitted = False
        self._global = np.full(len(TERMINAL_STATUSES), 1.0 / len(TERMINAL_STATUSES))
        self._group_posteriors: dict[str, dict[str, _Posterior]] = {}
        self._retirement_beta: dict[TerminalStatus, tuple[float, float]] = {}
        self.training_max_as_of: str | None = None
        self.training_rows = 0
        self.status_column = "terminal_status"

    @property
    def status_labels(self) -> tuple[str, ...]:
        return tuple(status.value for status in TERMINAL_STATUSES)

    @property
    def model_card(self) -> dict[str, object]:
        if not self._fitted:
            raise RuntimeError("terminal hazard must be fitted before inspection")
        return {
            "backend": self.backend,
            "training_rows": self.training_rows,
            "training_max_as_of": self.training_max_as_of,
            "global_status_probabilities": {
                status.value: float(self._global[index])
                for index, status in enumerate(TERMINAL_STATUSES)
            },
            "partial_pool_group_counts": {
                family: len(posteriors)
                for family, posteriors in self._group_posteriors.items()
            },
            "prior_strength": self.config.prior_strength,
            "recency_half_life_days": self.config.recency_half_life_days,
        }

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        status_col: str = "terminal_status",
        event_as_of_col: str = "event_as_of",
        cutoff: object | None = None,
        retirement_fraction_col: str = "retirement_fraction",
    ) -> "PartialPooledTerminalHazard":
        if frame.empty:
            raise ValueError("terminal hazard requires non-empty historical rows")
        rows = frame.copy()
        self.status_column = status_col
        if status_col not in rows.columns:
            raise ValueError(f"missing terminal status column: {status_col}")

        event_times: pd.Series | None = None
        if event_as_of_col in rows.columns:
            event_times = pd.to_datetime(rows[event_as_of_col], errors="coerce", utc=True)
            if event_times.isna().any():
                raise ValueError(f"{event_as_of_col} contains invalid timestamps")
        if cutoff is not None:
            if event_times is None:
                raise ValueError("cutoff requires an event_as_of column for causal filtering")
            cutoff_time = _utc_timestamp(cutoff, "cutoff")
            rows = rows.loc[event_times < cutoff_time].copy()
            event_times = event_times.loc[rows.index]
            if rows.empty:
                raise ValueError("no terminal-history rows precede the causal cutoff")

        encoded = rows[status_col].map(reason_code_terminal_status)
        valid = encoded.notna()
        if not valid.all():
            examples = sorted(rows.loc[~valid, status_col].astype(str).unique().tolist())[:5]
            raise ValueError(
                "terminal hazard target contains unrecognized reason codes; failed closed: "
                f"{examples}"
            )
        rows = rows.loc[valid].copy()
        encoded = encoded.loc[valid]
        if event_times is not None:
            event_times = event_times.loc[valid]
        if rows.empty:
            raise ValueError("terminal status history contains no recognized reason codes")

        if event_times is None:
            weights = pd.Series(1.0, index=rows.index, dtype=float)
            self.training_max_as_of = None
        else:
            reference = event_times.max()
            age_days = (reference - event_times).dt.total_seconds() / 86400.0
            weights = np.power(0.5, age_days / self.config.recency_half_life_days)
            self.training_max_as_of = reference.isoformat().replace("+00:00", "Z")

        labels = {status: index for index, status in enumerate(TERMINAL_STATUSES)}
        counts = np.ones(len(TERMINAL_STATUSES), dtype=float)
        for index, status in encoded.items():
            counts[labels[status]] += float(weights.loc[index])
        self._global = counts / counts.sum()

        group_columns = {
            "team": ("team_name", "constructor_name", "team_id"),
            "driver": ("driver_id",),
            "power_unit": ("power_unit", "power_unit_manufacturer", "engine_manufacturer"),
            "circuit": ("circuit_id", "circuit_name", "track_id"),
        }
        self._group_posteriors = {}
        for family, candidates in group_columns.items():
            column = next((candidate for candidate in candidates if candidate in rows.columns), None)
            if column is None:
                self._group_posteriors[family] = {}
                continue
            family_posteriors: dict[str, _Posterior] = {}
            normalized = rows[column].astype("string").str.strip()
            for value in sorted(normalized.dropna().unique().tolist()):
                mask = normalized.eq(value).fillna(False)
                group_counts = np.zeros(len(TERMINAL_STATUSES), dtype=float)
                for index, status in encoded.loc[mask].items():
                    group_counts[labels[status]] += float(weights.loc[index])
                support = float(group_counts.sum())
                posterior = (
                    group_counts + (self.config.prior_strength * self._global)
                ) / (support + self.config.prior_strength)
                family_posteriors[str(value)] = _Posterior(posterior, support)
            self._group_posteriors[family] = family_posteriors

        fractions = pd.to_numeric(rows.get(retirement_fraction_col), errors="coerce")
        if not isinstance(fractions, pd.Series):
            fractions = pd.Series(np.nan, index=rows.index, dtype=float)
        fractions = fractions.clip(0.0, 1.0)
        self._retirement_beta = {}
        for status in TERMINAL_STATUSES:
            status_mask = encoded.eq(status)
            status_values = fractions.loc[status_mask].dropna()
            status_weights = weights.loc[status_values.index]
            if status is TerminalStatus.DNS_WITHDRAWAL:
                self._retirement_beta[status] = (1.0, 1000.0)
            elif status_values.empty:
                self._retirement_beta[status] = (
                    (100.0, 1.0)
                    if status is TerminalStatus.CLASSIFIED_FINISH
                    else (2.0, 2.0)
                )
            else:
                successes = float((status_values * status_weights).sum())
                failures = float(((1.0 - status_values) * status_weights).sum())
                prior = (
                    (10.0, 1.0)
                    if status is TerminalStatus.CLASSIFIED_FINISH
                    else (2.0, 2.0)
                )
                self._retirement_beta[status] = (
                    prior[0] + successes,
                    prior[1] + failures,
                )

        self.training_rows = len(rows)
        self._fitted = True
        return self

    @staticmethod
    def _group_value(row: pd.Series, candidates: Iterable[str]) -> str | None:
        for column in candidates:
            if column not in row.index or pd.isna(row[column]):
                continue
            value = str(row[column]).strip()
            if value and value.lower() not in {"nan", "none", "null"}:
                return value
        return None

    def _blend_group(
        self,
        probabilities: np.ndarray,
        family: str,
        value: str | None,
        weight: float,
    ) -> np.ndarray:
        if value is None:
            return probabilities
        posterior = self._group_posteriors.get(family, {}).get(value)
        if posterior is None:
            return probabilities
        reliability = posterior.support / (posterior.support + self.config.prior_strength)
        log_p = np.log(np.clip(probabilities, self.config.minimum_probability, 1.0))
        log_group = np.log(
            np.clip(posterior.probabilities, self.config.minimum_probability, 1.0)
        )
        blended = np.exp(log_p + (weight * reliability * (log_group - log_p)))
        return blended / blended.sum()

    @staticmethod
    def _numeric(row: pd.Series, column: str) -> float | None:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        return None if pd.isna(value) else float(value)

    def _predict_row(self, row: pd.Series) -> tuple[np.ndarray, dict[str, float]]:
        p = self._global.copy()
        p = self._blend_group(
            p,
            "team",
            self._group_value(row, ("team_name", "constructor_name", "team_id")),
            self.config.team_weight,
        )
        p = self._blend_group(
            p,
            "driver",
            self._group_value(row, ("driver_id",)),
            self.config.driver_weight,
        )
        p = self._blend_group(
            p,
            "power_unit",
            self._group_value(row, ("power_unit", "power_unit_manufacturer", "engine_manufacturer")),
            self.config.power_unit_weight,
        )
        p = self._blend_group(
            p,
            "circuit",
            self._group_value(row, ("circuit_id", "circuit_name", "track_id")),
            self.config.circuit_weight,
        )

        index = {status: offset for offset, status in enumerate(TERMINAL_STATUSES)}
        classified_idx = index[TerminalStatus.CLASSIFIED_FINISH]
        mechanical_idx = index[TerminalStatus.MECHANICAL_POWER_UNIT]
        incident_idx = index[TerminalStatus.COLLISION_INCIDENT]
        nonclassified_idx = index[TerminalStatus.NON_CLASSIFIED]
        dns_idx = index[TerminalStatus.DNS_WITHDRAWAL]

        eligible = self._numeric(row, "race_starter_eligible")
        if eligible is not None and eligible <= 0.0:
            forced = np.zeros_like(p)
            forced[dns_idx] = 1.0
            return forced, {status.value: self._retirement_mean(status) for status in TERMINAL_STATUSES}

        terminal = 1.0 - p[classified_idx]
        risk_delta = 0.0
        team_mechanical = self._numeric(row, "race_team_mechanical_rate")
        pu_mechanical = self._numeric(row, "race_power_unit_mechanical_rate")
        driver_incident = self._numeric(row, "race_driver_incident_rate")
        circuit_dnf = self._numeric(row, "race_circuit_dnf_rate")
        stoppages = self._numeric(row, "race_weekend_stoppage_count")
        missed = self._numeric(row, "race_missed_practice_share")
        wet = self._numeric(row, "race_wet_probability")
        pit_lane = self._numeric(row, "race_pit_lane_start")

        global_mechanical = float(self._global[mechanical_idx])
        global_incident = float(self._global[incident_idx])
        global_terminal = float(1.0 - self._global[classified_idx])
        if team_mechanical is not None:
            risk_delta += 0.90 * (np.clip(team_mechanical, 0.0, 1.0) - global_mechanical)
            p[mechanical_idx] *= np.exp(
                1.50 * (np.clip(team_mechanical, 0.0, 1.0) - global_mechanical)
            )
        if pu_mechanical is not None:
            risk_delta += 0.65 * (np.clip(pu_mechanical, 0.0, 1.0) - global_mechanical)
            p[mechanical_idx] *= np.exp(
                1.25 * (np.clip(pu_mechanical, 0.0, 1.0) - global_mechanical)
            )
        if driver_incident is not None:
            risk_delta += 0.70 * (np.clip(driver_incident, 0.0, 1.0) - global_incident)
            p[incident_idx] *= np.exp(
                1.30 * (np.clip(driver_incident, 0.0, 1.0) - global_incident)
            )
        if circuit_dnf is not None:
            risk_delta += 1.10 * (np.clip(circuit_dnf, 0.0, 1.0) - global_terminal)
            p[nonclassified_idx] *= np.exp(
                0.80 * (np.clip(circuit_dnf, 0.0, 1.0) - global_terminal)
            )
        risk_delta += 0.12 * max(0.0, stoppages or 0.0)
        risk_delta += 0.35 * np.clip(missed or 0.0, 0.0, 1.0)
        risk_delta += 0.18 * np.clip(wet or 0.0, 0.0, 1.0)
        risk_delta += 0.03 * np.clip(pit_lane or 0.0, 0.0, 1.0)

        p = np.clip(p, self.config.minimum_probability, None)
        p /= p.sum()
        adjusted_terminal = _sigmoid(_logit(terminal) + risk_delta)
        terminal_mix = p.copy()
        terminal_mix[classified_idx] = 0.0
        terminal_mix /= terminal_mix.sum()
        p = terminal_mix * adjusted_terminal
        p[classified_idx] = 1.0 - adjusted_terminal
        p /= p.sum()
        means = {status.value: self._retirement_mean(status) for status in TERMINAL_STATUSES}
        return p, means

    def _retirement_mean(self, status: TerminalStatus) -> float:
        alpha, beta = self._retirement_beta.get(status, (2.0, 2.0))
        return float(alpha / (alpha + beta))

    def retirement_beta(self, status: TerminalStatus) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("terminal hazard must be fitted before inference")
        return self._retirement_beta[status]

    def predict_proba(
        self,
        frame: pd.DataFrame,
        *,
        prediction_as_of: object | None = None,
        feature_as_of_col: str = "feature_as_of",
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("terminal hazard must be fitted before inference")
        if frame.empty:
            return pd.DataFrame(index=frame.index)
        if prediction_as_of is not None:
            cutoff = _utc_timestamp(prediction_as_of, "prediction_as_of")
            if self.training_max_as_of is not None:
                train_max = _utc_timestamp(self.training_max_as_of, "training_max_as_of")
                if train_max >= cutoff:
                    raise ValueError("terminal model training evidence is not strictly pre-cutoff")
            if feature_as_of_col in frame.columns:
                feature_times = pd.to_datetime(frame[feature_as_of_col], errors="coerce", utc=True)
                if feature_times.isna().any() or (feature_times > cutoff).any():
                    raise ValueError("terminal features contain invalid or post-cutoff evidence")

        features = engineer_survival_aware_race_features(frame)
        records: list[dict[str, object]] = []
        for index, row in features.iterrows():
            probabilities, retirement_means = self._predict_row(row)
            record: dict[str, object] = {
                "driver_id": str(row.get("driver_id", index)),
                "p_terminal": float(
                    1.0 - probabilities[TERMINAL_STATUSES.index(TerminalStatus.CLASSIFIED_FINISH)]
                ),
                "expected_retirement_fraction": float(
                    sum(
                        probabilities[offset] * retirement_means[status.value]
                        for offset, status in enumerate(TERMINAL_STATUSES)
                    )
                ),
                "terminal_hazard_backend": self.backend,
                "terminal_training_rows": self.training_rows,
                "terminal_training_max_as_of": self.training_max_as_of,
            }
            for offset, status in enumerate(TERMINAL_STATUSES):
                record[f"p_{status.value}"] = float(probabilities[offset])
                record[f"expected_retirement_fraction_{status.value}"] = retirement_means[
                    status.value
                ]
            records.append(record)
        return pd.DataFrame(records, index=frame.index)


__all__ = ["PartialPooledTerminalHazard", "TerminalHazardConfig"]
