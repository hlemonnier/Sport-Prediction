"""Driver-conditioned discrete-time competing-risk Race hazard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from packages.f1.features.race import engineer_survival_aware_race_features
from packages.f1.models.pre_race.status import (
    TERMINAL_STATUSES,
    TerminalLabelGranularity,
    TerminalStatus,
    reason_code_terminal_status,
    terminal_label_granularity,
)


_TIMED_CAUSES: tuple[TerminalStatus, ...] = (
    TerminalStatus.MECHANICAL_POWER_UNIT,
    TerminalStatus.COLLISION_INCIDENT,
    TerminalStatus.NON_CLASSIFIED,
)

_HAZARD_COVARIATE_COLUMNS: tuple[str, ...] = (
    "race_team_mechanical_rate",
    "race_power_unit_mechanical_rate",
    "race_driver_incident_rate",
    "race_weekend_stoppage_count",
    "race_missed_practice_share",
    "race_circuit_dnf_rate",
    "race_safety_car_probability",
    "race_wet_probability",
    "race_weather_uncertainty",
    "race_current_weekend_mechanical_stop_share",
    "race_power_unit_grid_penalty",
)


def _utc_timestamp(value: object, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return pd.Timestamp(parsed)


def _clip_probability(value: float, epsilon: float = 1e-8) -> float:
    return float(np.clip(float(value), epsilon, 1.0 - epsilon))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0))))


def _observed_hazard_terms(
    hazards: np.ndarray,
    cause_codes: np.ndarray,
    fractions: np.ndarray,
    censor_bins: np.ndarray,
    *,
    log_no_event: np.ndarray | None = None,
    log_cause_hazard: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed log likelihood and conditional sufficient statistics.

    A known failure without a distance is interval censored over the entire
    race. Its posterior interval weights are proportional to S(b-1) h_k(b).
    They are an integration device, never newly asserted timing observations.
    Non-failures contribute only the intervals through which survival is known.
    """
    from scipy.special import logsumexp

    n, bins, causes = hazards.shape
    log_survival = (
        np.log1p(-hazards.sum(axis=2)) if log_no_event is None else log_no_event
    )
    prefix = np.concatenate([np.zeros((n, 1)), np.cumsum(log_survival, axis=1)], axis=1)
    likelihood = prefix[np.arange(n), censor_bins].copy()
    exposure = (np.arange(bins)[None, :] < censor_bins[:, None]).astype(float)
    events = np.zeros((n, bins, causes), dtype=float)
    failures = np.flatnonzero(cause_codes >= 0)
    if failures.size:
        cause_log_probability = (
            np.log(hazards[failures, :, cause_codes[failures]])
            if log_cause_hazard is None
            else log_cause_hazard[failures, :, cause_codes[failures]]
        )
        event_log_mass = prefix[failures, :-1] + cause_log_probability
        normalizer = logsumexp(event_log_mass, axis=1)
        posterior = np.exp(event_log_mass - normalizer[:, None])
        observed = np.isfinite(fractions[failures])
        if observed.any():
            local = np.flatnonzero(observed)
            event_bin = np.minimum(
                bins - 1, np.floor(fractions[failures[local]] * bins).astype(int)
            )
            posterior[local] = 0.0
            posterior[local, event_bin] = 1.0
            normalizer[local] = event_log_mass[local, event_bin]
        likelihood[failures] = normalizer
        exposure[failures] = np.cumsum(posterior[:, ::-1], axis=1)[:, ::-1]
        events[failures, :, cause_codes[failures]] = posterior
    return likelihood, exposure, events


def _observation_encoding(
    statuses: pd.Series, fractions: pd.Series, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = np.asarray([
        _TIMED_CAUSES.index(status) if status in _TIMED_CAUSES else -1
        for status in statuses
    ], dtype=int)
    distances = fractions.to_numpy(dtype=float)
    censor = np.full(len(statuses), bins, dtype=int)
    for row, status in enumerate(statuses):
        if status is TerminalStatus.DNS_WITHDRAWAL:
            censor[row] = 0
        elif status is TerminalStatus.DISQUALIFIED:
            # A steward's exclusion is not evidence of a running failure.
            # Use only positively observed distance as right-censored survival.
            censor[row] = int(np.floor(distances[row] * bins)) if np.isfinite(distances[row]) else 0
    return codes, distances, censor


@dataclass(frozen=True)
class TerminalHazardConfig:
    """Regularization, interval, and shared-shock contract."""

    prior_strength: float = 12.0
    recency_half_life_days: float = 365.0
    minimum_probability: float = 1e-5
    team_weight: float = 0.35
    driver_weight: float = 0.20
    power_unit_weight: float = 0.20
    circuit_weight: float = 0.15
    time_bins: int = 12
    maximum_interval_hazard: float = 0.75
    shared_event_shock_std: float = 0.30
    shared_team_mechanical_std: float = 0.35
    shared_power_unit_mechanical_std: float = 0.45
    shared_team_pace_std: float = 0.18
    shared_weather_incident_std: float = 0.35
    shared_safety_car_incident_std: float = 0.30
    covariate_l2_c: float = 0.25
    covariate_z_clip: float = 3.0
    minimum_cause_events_for_covariate_fit: int = 3

    def __post_init__(self) -> None:
        if self.prior_strength <= 0.0:
            raise ValueError("prior_strength must be positive")
        if self.recency_half_life_days <= 0.0:
            raise ValueError("recency_half_life_days must be positive")
        if not 0.0 < self.minimum_probability < 0.1:
            raise ValueError("minimum_probability must be in (0, 0.1)")
        if int(self.time_bins) < 4:
            raise ValueError("time_bins must be at least 4")
        if not 0.0 < float(self.maximum_interval_hazard) < 1.0:
            raise ValueError("maximum_interval_hazard must be in (0, 1)")
        shock_values = (
            self.shared_event_shock_std,
            self.shared_team_mechanical_std,
            self.shared_power_unit_mechanical_std,
            self.shared_team_pace_std,
            self.shared_weather_incident_std,
            self.shared_safety_car_incident_std,
        )
        if any(float(value) < 0.0 for value in shock_values):
            raise ValueError("shared-shock scales cannot be negative")
        if float(self.covariate_l2_c) <= 0.0:
            raise ValueError("covariate_l2_c must be positive")
        if not np.isfinite(float(self.covariate_z_clip)) or float(
            self.covariate_z_clip
        ) <= 0.0:
            raise ValueError("covariate_z_clip must be finite and positive")
        if int(self.minimum_cause_events_for_covariate_fit) < 1:
            raise ValueError("minimum_cause_events_for_covariate_fit must be positive")


@dataclass(frozen=True)
class _Posterior:
    probabilities: np.ndarray
    support: float


@dataclass(frozen=True)
class SharedRaceShocks:
    """One causal event draw shared across entrants in a joint simulation."""

    event_chaos: float
    weather: float
    safety_car: float
    team_mechanical: Mapping[str, float]
    power_unit_mechanical: Mapping[str, float]
    team_pace: Mapping[str, float]


@dataclass(frozen=True)
class PreparedTerminalHazards:
    """Deterministic per-entrant hazards reused by every joint draw.

    Preparing a forecast is target-free and consumes no random numbers.  The
    expensive feature standardization and row-hazard construction therefore
    happen once per entrant, rather than once per entrant per simulation.
    """

    features: pd.DataFrame
    driver_ids: tuple[str, ...]
    dns_probabilities: np.ndarray
    exclusion_given_start_probabilities: np.ndarray
    interval_hazards: np.ndarray
    event_masses: np.ndarray
    survival_traces: np.ndarray
    retirement_means: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class BinaryTerminalCalibrator:
    """Monotone Platt mapping fitted on a declared calibration event block."""

    intercept: float
    slope: float
    calibration_rows: int
    calibration_event_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.intercept)):
            raise ValueError("terminal calibration intercept must be finite")
        if not np.isfinite(float(self.slope)) or float(self.slope) < 0.0:
            raise ValueError("terminal calibration slope must be finite and non-negative")
        if int(self.calibration_rows) <= 0:
            raise ValueError("terminal calibration requires positive row support")
        if not self.calibration_event_keys:
            raise ValueError("terminal calibration event keys cannot be empty")

    def transform(self, probability: float) -> float:
        value = _clip_probability(probability)
        logit = np.log(value / (1.0 - value))
        return _clip_probability(_sigmoid(float(self.intercept) + float(self.slope) * logit))


class PartialPooledTerminalHazard:
    """Partial-pooled cause hazard expanded over race-distance intervals.

    Historical rows are expanded into at-risk intervals up to an observed
    retirement bin; classified finishers are right-censored after the final
    interval. Untimed failures contribute their event-by-race-end likelihood,
    integrating over every possible failure interval. Exclusions are modeled
    separately from running failures. Team, power-unit, driver and
    circuit evidence shrink to a recency-weighted global prior.
    """

    backend = "partial_pooled_interval_censored_competing_risk_v4"

    def __init__(self, config: TerminalHazardConfig | None = None) -> None:
        self.config = config or TerminalHazardConfig()
        self._fitted = False
        self._global = np.full(len(TERMINAL_STATUSES), 1.0 / len(TERMINAL_STATUSES))
        self._group_posteriors: dict[str, dict[str, _Posterior]] = {}
        self._baseline_hazard = np.zeros(
            (int(self.config.time_bins), len(_TIMED_CAUSES)), dtype=float
        )
        self._covariate_medians = np.zeros(len(_HAZARD_COVARIATE_COLUMNS), dtype=float)
        self._covariate_scales = np.ones(len(_HAZARD_COVARIATE_COLUMNS), dtype=float)
        self._covariate_coefficients = np.zeros(
            (
                len(_TIMED_CAUSES),
                int(self.config.time_bins) + 2 * len(_HAZARD_COVARIATE_COLUMNS),
            ),
            dtype=float,
        )
        self._covariate_intercepts = np.zeros(len(_TIMED_CAUSES), dtype=float)
        self._covariate_fit_causes: set[TerminalStatus] = set()
        self._terminal_calibrator: BinaryTerminalCalibrator | None = None
        self._retirement_beta: dict[TerminalStatus, tuple[float, float]] = {}
        self.training_max_as_of: str | None = None
        self.training_rows = 0
        self.timing_evidence_rows = 0
        self.coarse_terminal_rows = 0
        self.interval_censored_failure_rows = 0
        self._baseline_em_iterations = 0
        self._covariate_optimization: dict[str, object] = {}
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
            "timing_evidence_rows": self.timing_evidence_rows,
            "coarse_terminal_rows": self.coarse_terminal_rows,
            "interval_censored_failure_rows": self.interval_censored_failure_rows,
            "untimed_failure_likelihood": "sum_over_all_failure_intervals",
            "baseline_em_iterations": self._baseline_em_iterations,
            "exclusion_model": "pooled_exclusion_given_start_applied_after_running_order",
            "time_bins": int(self.config.time_bins),
            "timed_causes": [status.value for status in _TIMED_CAUSES],
            "global_status_probabilities": {
                status.value: float(self._global[index])
                for index, status in enumerate(TERMINAL_STATUSES)
            },
            "baseline_interval_hazards": {
                status.value: self._baseline_hazard[:, index].tolist()
                for index, status in enumerate(_TIMED_CAUSES)
            },
            "covariate_model": {
                "backend": "l2_multinomial_discrete_hazard_observed_data_likelihood",
                "optimization": self._covariate_optimization,
                "regularization_c": float(self.config.covariate_l2_c),
                "columns": list(_HAZARD_COVARIATE_COLUMNS),
                "missingness_indicators": True,
                "standardization": {
                    "fit_scope": "strictly_pre_cutoff_training_rows",
                    "center": "training_median",
                    "scale": "max_training_mad_or_standard_deviation",
                    "z_clip": float(self.config.covariate_z_clip),
                    "missingness_captured_before_imputation": True,
                    "training_medians": {
                        column: float(self._covariate_medians[index])
                        for index, column in enumerate(_HAZARD_COVARIATE_COLUMNS)
                    },
                    "training_scales": {
                        column: float(self._covariate_scales[index])
                        for index, column in enumerate(_HAZARD_COVARIATE_COLUMNS)
                    },
                },
                "fitted_causes": [
                    cause.value
                    for cause in _TIMED_CAUSES
                    if cause in self._covariate_fit_causes
                ],
                "coefficients": {
                    cause.value: {
                        name: float(
                            self._covariate_coefficients[cause_index][
                                int(self.config.time_bins) + feature_index
                            ]
                        )
                        for feature_index, name in enumerate(_HAZARD_COVARIATE_COLUMNS)
                    }
                    for cause_index, cause in enumerate(_TIMED_CAUSES)
                    if cause in self._covariate_fit_causes
                },
            },
            "partial_pool_group_counts": {
                family: len(posteriors)
                for family, posteriors in self._group_posteriors.items()
            },
            "prior_strength": self.config.prior_strength,
            "recency_half_life_days": self.config.recency_half_life_days,
            "coarse_label_policy": (
                "non_classified remains an explicit coarse cause and is never "
                "redistributed to mechanical or collision"
            ),
            "remaining_limitations": [
                (
                    "coarse non_classified is still fitted as an explicit competing "
                    "cause; a binary terminal model followed by observed-cause "
                    "factorization remains future work"
                )
            ],
            "shared_shocks": {
                "event_chaos_std": self.config.shared_event_shock_std,
                "team_mechanical_std": self.config.shared_team_mechanical_std,
                "power_unit_mechanical_std": self.config.shared_power_unit_mechanical_std,
                "team_pace_std": self.config.shared_team_pace_std,
                "weather_incident_std": self.config.shared_weather_incident_std,
                "safety_car_incident_std": self.config.shared_safety_car_incident_std,
                "multiplicative_normalization": (
                    "exp(log_shock - 0.5 * marginal_log_variance)"
                ),
            },
            "binary_terminal_calibration": (
                None
                if self._terminal_calibrator is None
                else {
                    "backend": "monotone_platt_logit",
                    "intercept": float(self._terminal_calibrator.intercept),
                    "slope": float(self._terminal_calibrator.slope),
                    "calibration_rows": int(
                        self._terminal_calibrator.calibration_rows
                    ),
                    "calibration_event_keys": list(
                        self._terminal_calibrator.calibration_event_keys
                    ),
                }
            ),
        }

    def set_terminal_calibrator(
        self,
        calibrator: BinaryTerminalCalibrator | None,
    ) -> "PartialPooledTerminalHazard":
        if calibrator is not None and not isinstance(
            calibrator, BinaryTerminalCalibrator
        ):
            raise TypeError("calibrator must be a BinaryTerminalCalibrator")
        self._terminal_calibrator = calibrator
        return self

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
        rows = frame.reset_index(drop=True).copy()
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
        if encoded.isna().any():
            examples = sorted(
                rows.loc[encoded.isna(), status_col].astype(str).unique().tolist()
            )[:5]
            raise ValueError(
                "terminal hazard target contains unrecognized reason codes; failed closed: "
                f"{examples}"
            )
        if event_times is None:
            weights = pd.Series(1.0, index=rows.index, dtype=float)
            self.training_max_as_of = None
        else:
            reference = event_times.max()
            age_days = (reference - event_times).dt.total_seconds() / 86400.0
            weights = np.power(0.5, age_days / self.config.recency_half_life_days)
            self.training_max_as_of = reference.isoformat().replace("+00:00", "Z")

        status_index = {status: index for index, status in enumerate(TERMINAL_STATUSES)}
        counts = np.ones(len(TERMINAL_STATUSES), dtype=float)
        for index, status in encoded.items():
            counts[status_index[status]] += float(weights.loc[index])
        self._global = counts / counts.sum()

        group_columns = {
            "team": ("team_name", "constructor_name", "team_id"),
            "driver": ("driver_id",),
            "power_unit": (
                "power_unit",
                "power_unit_manufacturer",
                "engine_manufacturer",
            ),
            "circuit": ("circuit_id", "circuit_name", "track_id"),
        }
        self._group_posteriors = {}
        for family, candidates in group_columns.items():
            column = next((candidate for candidate in candidates if candidate in rows.columns), None)
            if column is None:
                self._group_posteriors[family] = {}
                continue
            normalized = rows[column].astype("string").str.strip()
            family_posteriors: dict[str, _Posterior] = {}
            for value in sorted(normalized.dropna().unique().tolist()):
                mask = normalized.eq(value).fillna(False)
                group_counts = np.zeros(len(TERMINAL_STATUSES), dtype=float)
                for index, status in encoded.loc[mask].items():
                    group_counts[status_index[status]] += float(weights.loc[index])
                support = float(group_counts.sum())
                posterior = (
                    group_counts + self.config.prior_strength * self._global
                ) / (support + self.config.prior_strength)
                family_posteriors[str(value)] = _Posterior(posterior, support)
            self._group_posteriors[family] = family_posteriors

        raw_fractions = rows.get(
            retirement_fraction_col,
            pd.Series(np.nan, index=rows.index, dtype=float),
        )
        fractions = pd.to_numeric(raw_fractions, errors="coerce").clip(0.0, 1.0)
        n_bins = int(self.config.time_bins)
        codes, distances, censor_bins = _observation_encoding(encoded, fractions, n_bins)
        starter_terminal = float(
            self._global[[status_index[cause] for cause in _TIMED_CAUSES]].sum()
        )
        outcome_mass = max(
            self.config.minimum_probability,
            starter_terminal + float(self._global[status_index[TerminalStatus.CLASSIFIED_FINISH]]),
        )
        conditional_terminal = np.clip(starter_terminal / outcome_mass, 0.0, 0.95)
        prior_total_hazard = 1.0 - (1.0 - conditional_terminal) ** (1.0 / n_bins)
        global_cause = np.asarray(
            [self._global[status_index[cause]] for cause in _TIMED_CAUSES], dtype=float
        )
        prior_hazard = prior_total_hazard * global_cause / global_cause.sum()
        baseline = np.tile(prior_hazard, (n_bins, 1))
        row_weight = weights.to_numpy(dtype=float)
        # EM maximizes the penalized observed likelihood. Fully observed rows
        # retain exact timing; unknown failure intervals are integrated out.
        for iteration in range(500):
            _, exposure, events = _observed_hazard_terms(
                np.broadcast_to(baseline, (len(rows), *baseline.shape)),
                codes, distances, censor_bins,
            )
            at_risk = np.einsum("n,nb->b", row_weight, exposure)
            event_counts = np.einsum("n,nbk->bk", row_weight, events)
            updated = (event_counts + self.config.prior_strength * prior_hazard) / (
                at_risk[:, None] + self.config.prior_strength
            )
            updated = np.vstack([self._cap_hazard_row(values) for values in updated])
            change = float(np.max(np.abs(updated - baseline)))
            baseline = updated
            self._baseline_em_iterations = iteration + 1
            if change < 1e-10:
                break
        self._baseline_hazard = baseline
        self.interval_censored_failure_rows = int(((codes >= 0) & ~np.isfinite(distances)).sum())
        self.timing_evidence_rows = int(
            (((codes >= 0) & np.isfinite(distances)) | encoded.eq(TerminalStatus.CLASSIFIED_FINISH).to_numpy()).sum()
        )
        self._fit_covariate_hazards(
            rows,
            encoded=encoded,
            fractions=fractions,
            row_weights=weights,
        )

        if "terminal_label_granularity" in rows.columns:
            granularity = rows["terminal_label_granularity"].astype(str)
        else:
            granularity = rows[status_col].map(terminal_label_granularity).map(
                lambda value: value.value if value is not None else "unknown"
            )
        self.coarse_terminal_rows = int(
            granularity.eq(TerminalLabelGranularity.COARSE_TERMINAL.value).sum()
        )

        self._retirement_beta = {}
        for status in TERMINAL_STATUSES:
            status_values = fractions.loc[encoded.eq(status)].dropna()
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

    def _cap_hazard_row(self, values: np.ndarray) -> np.ndarray:
        hazard = np.clip(np.asarray(values, dtype=float), self.config.minimum_probability, None)
        total = float(hazard.sum())
        if total > self.config.maximum_interval_hazard:
            hazard *= self.config.maximum_interval_hazard / total
        return hazard

    def _standardized_covariates(
        self,
        features: pd.DataFrame,
        *,
        fit: bool,
    ) -> np.ndarray:
        numeric = np.column_stack(
            [
                pd.to_numeric(
                    features.get(column, pd.Series(np.nan, index=features.index)),
                    errors="coerce",
                ).to_numpy(dtype=float)
                for column in _HAZARD_COVARIATE_COLUMNS
            ]
        )
        missing = ~np.isfinite(numeric)
        if fit:
            medians = np.zeros(numeric.shape[1], dtype=float)
            scales = np.ones(numeric.shape[1], dtype=float)
            for column_index in range(numeric.shape[1]):
                observed = numeric[~missing[:, column_index], column_index]
                if observed.size:
                    medians[column_index] = float(np.median(observed))
                    robust_scale = float(
                        np.median(np.abs(observed - medians[column_index])) * 1.4826
                    )
                    standard_scale = float(np.std(observed))
                    scales[column_index] = max(robust_scale, standard_scale, 1e-6)
            self._covariate_medians = medians
            self._covariate_scales = scales
        filled = np.where(missing, self._covariate_medians[None, :], numeric)
        standardized = (filled - self._covariate_medians[None, :]) / self._covariate_scales[
            None, :
        ]
        standardized = np.clip(
            standardized,
            -float(self.config.covariate_z_clip),
            float(self.config.covariate_z_clip),
        )
        return np.column_stack([standardized, missing.astype(float)])

    def _fit_covariate_hazards(
        self,
        rows: pd.DataFrame,
        *,
        encoded: pd.Series,
        fractions: pd.Series,
        row_weights: pd.Series,
    ) -> None:
        """Fit a joint competing-risk likelihood, including untimed failures.

        Multinomial logits include the no-event outcome, so cause probabilities
        share one denominator. This also avoids separately fitted binary heads
        producing incompatible hazards that need an arbitrary rescaling.
        """
        from scipy.optimize import minimize
        from scipy.special import log_softmax

        features = engineer_survival_aware_race_features(rows)
        standardized = self._standardized_covariates(features, fit=True)
        bins = int(self.config.time_bins)
        codes, distances, censor_bins = _observation_encoding(encoded, fractions, bins)
        width = bins + standardized.shape[1]
        self._covariate_coefficients = np.zeros((len(_TIMED_CAUSES), width))
        self._covariate_intercepts = np.zeros(len(_TIMED_CAUSES))
        minimum = int(self.config.minimum_cause_events_for_covariate_fit)
        fitted_indices = np.asarray([
            index for index in range(len(_TIMED_CAUSES)) if int((codes == index).sum()) >= minimum
        ], dtype=int)
        self._covariate_fit_causes = {_TIMED_CAUSES[index] for index in fitted_indices}
        self._covariate_optimization = {"success": True, "iterations": 0}
        if not len(fitted_indices):
            return
        design = np.concatenate([
            np.broadcast_to(np.eye(bins), (len(rows), bins, bins)),
            np.broadcast_to(standardized[:, None, :], (len(rows), bins, standardized.shape[1])),
        ], axis=2)
        base = self._baseline_hazard
        offset = np.log(base) - np.log1p(-base.sum(axis=1))[:, None]
        weight = row_weights.to_numpy(dtype=float)

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            coefficients = np.zeros_like(self._covariate_coefficients)
            coefficients[fitted_indices] = flat.reshape(len(fitted_indices), width)
            logits = offset[None, :, :] + np.einsum("nbf,kf->nbk", design, coefficients)
            log_probabilities = log_softmax(np.concatenate([np.zeros((*logits.shape[:2], 1)), logits], axis=2), axis=2)
            probabilities = np.exp(log_probabilities)
            hazard = probabilities[:, :, 1:]
            log_likelihood, exposure, expected_event = _observed_hazard_terms(
                hazard, codes, distances, censor_bins,
                log_no_event=log_probabilities[:, :, 0],
                log_cause_hazard=log_probabilities[:, :, 1:],
            )
            score = exposure[:, :, None] * hazard - expected_event
            gradient = np.einsum("n,nbk,nbf->kf", weight, score, design)
            penalty = 1.0 / float(self.config.covariate_l2_c)
            value = -float(np.dot(weight, log_likelihood)) + 0.5 * penalty * float(np.dot(flat, flat))
            gradient += penalty * coefficients
            return value, gradient[fitted_indices].ravel()

        result = minimize(
            objective, np.zeros(len(fitted_indices) * width), jac=True,
            method="L-BFGS-B", bounds=[(-8.0, 8.0)] * (len(fitted_indices) * width),
            options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-6},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"interval-censored hazard fit did not converge: {result.message}")
        self._covariate_coefficients[fitted_indices] = result.x.reshape(len(fitted_indices), width)
        self._covariate_optimization = {
            "success": bool(result.success), "iterations": int(result.nit),
            "penalized_negative_log_likelihood": float(result.fun),
            "untimed_failure_rows": int(((codes >= 0) & ~np.isfinite(distances)).sum()),
        }

    def _learned_interval_hazard(self, row: pd.Series) -> np.ndarray:
        from scipy.special import softmax

        base = self._baseline_hazard
        if not self._covariate_fit_causes:
            return base.copy()
        standardized = self._standardized_covariates(pd.DataFrame([row]), fit=False)[0]
        bins = int(self.config.time_bins)
        design = np.concatenate([np.eye(bins), np.tile(standardized, (bins, 1))], axis=1)
        logits = np.log(base) - np.log1p(-base.sum(axis=1))[:, None]
        logits += design @ self._covariate_coefficients.T
        return softmax(np.column_stack([np.zeros(bins), logits]), axis=1)[:, 1:]

    def _apply_binary_calibration(
        self,
        dns: float,
        hazard: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        calibrator = self._terminal_calibrator
        if calibrator is None or dns >= 1.0 - self.config.minimum_probability:
            return dns, hazard
        raw_survival = (1.0 - dns) * float(
            np.prod(1.0 - np.clip(hazard.sum(axis=1), 0.0, 1.0))
        )
        raw_terminal = float(np.clip(1.0 - raw_survival, 1e-8, 1.0 - 1e-8))
        target_terminal = calibrator.transform(raw_terminal)
        dns_target = float(
            np.clip(
                target_terminal * dns / raw_terminal,
                self.config.minimum_probability,
                target_terminal,
            )
        )
        timed_target = max(0.0, target_terminal - dns_target)
        conditional_target = float(
            np.clip(
                timed_target / max(1.0 - dns_target, self.config.minimum_probability),
                0.0,
                1.0 - self.config.minimum_probability,
            )
        )

        def conditional_terminal(scale: float) -> float:
            scaled = np.vstack(
                [self._cap_hazard_row(values * scale) for values in hazard]
            )
            return float(1.0 - np.prod(1.0 - scaled.sum(axis=1)))

        lower, upper = 0.0, 1.0
        while conditional_terminal(upper) < conditional_target and upper < 128.0:
            upper *= 2.0
        for _ in range(50):
            midpoint = (lower + upper) / 2.0
            if conditional_terminal(midpoint) < conditional_target:
                lower = midpoint
            else:
                upper = midpoint
        calibrated = np.vstack(
            [self._cap_hazard_row(values * ((lower + upper) / 2.0)) for values in hazard]
        )
        return dns_target, calibrated

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
        log_base = np.log(np.clip(probabilities, self.config.minimum_probability, 1.0))
        log_group = np.log(
            np.clip(posterior.probabilities, self.config.minimum_probability, 1.0)
        )
        blended = np.exp(log_base + weight * reliability * (log_group - log_base))
        return blended / blended.sum()

    @staticmethod
    def _numeric(row: pd.Series, column: str) -> float | None:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        return None if pd.isna(value) else float(value)

    def _pooled_outcome_prior(self, row: pd.Series) -> np.ndarray:
        probabilities = self._global.copy()
        probabilities = self._blend_group(
            probabilities,
            "team",
            self._group_value(row, ("team_name", "constructor_name", "team_id")),
            self.config.team_weight,
        )
        probabilities = self._blend_group(
            probabilities,
            "driver",
            self._group_value(row, ("driver_id",)),
            self.config.driver_weight,
        )
        probabilities = self._blend_group(
            probabilities,
            "power_unit",
            self._group_value(
                row,
                ("power_unit", "power_unit_manufacturer", "engine_manufacturer"),
            ),
            self.config.power_unit_weight,
        )
        return self._blend_group(
            probabilities,
            "circuit",
            self._group_value(row, ("circuit_id", "circuit_name", "track_id")),
            self.config.circuit_weight,
        )

    def _row_hazard(
        self,
        row: pd.Series,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        pooled = self._pooled_outcome_prior(row)
        status_index = {status: index for index, status in enumerate(TERMINAL_STATUSES)}
        eligible = self._numeric(row, "race_starter_eligible")
        if eligible is not None and eligible <= 0.0:
            means = {status.value: self._retirement_mean(status) for status in TERMINAL_STATUSES}
            means[TerminalStatus.DNS_WITHDRAWAL.value] = 0.0
            return 1.0, 0.0, np.zeros_like(self._baseline_hazard), np.zeros_like(
                self._baseline_hazard
            ), np.zeros(int(self.config.time_bins)), means

        dns = float(
            np.clip(
                pooled[status_index[TerminalStatus.DNS_WITHDRAWAL]],
                self.config.minimum_probability,
                0.45,
            )
        )
        exclusion = float(np.clip(
            pooled[status_index[TerminalStatus.DISQUALIFIED]] / (1.0 - dns),
            0.0, 1.0 - self.config.minimum_probability,
        ))
        global_cause = np.asarray(
            [self._global[status_index[cause]] for cause in _TIMED_CAUSES], dtype=float
        )
        pooled_cause = np.asarray(
            [pooled[status_index[cause]] for cause in _TIMED_CAUSES], dtype=float
        )
        cause_ratio = np.clip(
            pooled_cause / np.clip(global_cause, self.config.minimum_probability, None),
            0.25,
            4.0,
        )
        hazard = self._learned_interval_hazard(row) * np.sqrt(cause_ratio)[None, :]
        hazard = np.vstack([self._cap_hazard_row(values) for values in hazard])
        pre_running_terminal = dns + (1.0 - dns) * exclusion
        calibrated_pre, hazard = self._apply_binary_calibration(pre_running_terminal, hazard)
        if pre_running_terminal > 0.0:
            dns = calibrated_pre * dns / pre_running_terminal
            exclusion = (calibrated_pre - dns) / max(1.0 - dns, self.config.minimum_probability)

        event_mass = np.zeros_like(hazard)
        survival_trace = np.zeros(int(self.config.time_bins), dtype=float)
        survival = (1.0 - dns) * (1.0 - exclusion)
        for bin_index in range(int(self.config.time_bins)):
            event_mass[bin_index] = survival * hazard[bin_index]
            survival *= max(0.0, 1.0 - float(hazard[bin_index].sum()))
            survival_trace[bin_index] = survival
        bin_centres = (np.arange(int(self.config.time_bins), dtype=float) + 0.5) / float(
            self.config.time_bins
        )
        means: dict[str, float] = {}
        for cause_index, cause in enumerate(_TIMED_CAUSES):
            mass = event_mass[:, cause_index]
            means[cause.value] = (
                float(np.dot(mass, bin_centres) / mass.sum())
                if mass.sum() > self.config.minimum_probability
                else self._retirement_mean(cause)
            )
        means[TerminalStatus.DNS_WITHDRAWAL.value] = 0.0
        means[TerminalStatus.DISQUALIFIED.value] = float("nan")
        means[TerminalStatus.CLASSIFIED_FINISH.value] = 1.0
        return dns, exclusion, hazard, event_mass, survival_trace, means

    def _predict_row(self, row: pd.Series) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
        dns, exclusion, hazard, event_mass, survival_trace, means = self._row_hazard(row)
        probabilities = self._probabilities_from_prepared_row(
            dns=dns,
            exclusion=exclusion,
            event_mass=event_mass,
            survival_trace=survival_trace,
        )
        return probabilities, means, hazard, survival_trace

    def _probabilities_from_prepared_row(
        self,
        *,
        dns: float,
        exclusion: float,
        event_mass: np.ndarray,
        survival_trace: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct the exact row probabilities from prepared components."""

        status_index = {status: index for index, status in enumerate(TERMINAL_STATUSES)}
        probabilities = np.zeros(len(TERMINAL_STATUSES), dtype=float)
        if dns >= 1.0 - self.config.minimum_probability:
            probabilities[status_index[TerminalStatus.DNS_WITHDRAWAL]] = 1.0
            return probabilities
        probabilities[status_index[TerminalStatus.DNS_WITHDRAWAL]] = dns
        probabilities[status_index[TerminalStatus.DISQUALIFIED]] = (1.0 - dns) * exclusion
        for cause_index, cause in enumerate(_TIMED_CAUSES):
            probabilities[status_index[cause]] = float(event_mass[:, cause_index].sum())
        probabilities[status_index[TerminalStatus.CLASSIFIED_FINISH]] = float(
            survival_trace[-1]
        )
        # Components already form one normalized law. Flooring each final
        # outcome separately would change both the sampler law and distances.
        probabilities /= probabilities.sum()
        return probabilities

    def _validate_prediction_cutoff(
        self,
        frame: pd.DataFrame,
        *,
        prediction_as_of: object | None,
        feature_as_of_col: str,
    ) -> None:
        if prediction_as_of is None:
            return
        cutoff = _utc_timestamp(prediction_as_of, "prediction_as_of")
        if self.training_max_as_of is not None:
            train_max = _utc_timestamp(self.training_max_as_of, "training_max_as_of")
            if train_max >= cutoff:
                raise ValueError("terminal model training evidence is not strictly pre-cutoff")
        if feature_as_of_col in frame.columns:
            feature_times = pd.to_datetime(
                frame[feature_as_of_col], errors="coerce", utc=True
            )
            if feature_times.isna().any() or (feature_times > cutoff).any():
                raise ValueError("terminal features contain invalid or post-cutoff evidence")

    @staticmethod
    def _driver_ids(frame: pd.DataFrame) -> tuple[str, ...]:
        if "driver_id" in frame.columns:
            return tuple(frame["driver_id"].astype(str).tolist())
        return tuple(frame.index.astype(str).tolist())

    def _validate_prepared(
        self,
        frame: pd.DataFrame,
        prepared: PreparedTerminalHazards,
    ) -> None:
        if not isinstance(prepared, PreparedTerminalHazards):
            raise TypeError("prepared must be PreparedTerminalHazards")
        count = len(frame)
        if prepared.driver_ids != self._driver_ids(frame):
            raise ValueError("prepared terminal hazards do not match the entrant order")
        if prepared.dns_probabilities.shape != (count,):
            raise ValueError("prepared DNS probabilities have an invalid shape")
        if prepared.exclusion_given_start_probabilities.shape != (count,):
            raise ValueError("prepared exclusion probabilities have an invalid shape")
        expected_hazard_shape = (
            count,
            int(self.config.time_bins),
            len(_TIMED_CAUSES),
        )
        if prepared.interval_hazards.shape != expected_hazard_shape:
            raise ValueError("prepared interval hazards have an invalid shape")
        if prepared.event_masses.shape != expected_hazard_shape:
            raise ValueError("prepared event masses have an invalid shape")
        if prepared.survival_traces.shape != (
            count,
            int(self.config.time_bins),
        ):
            raise ValueError("prepared survival traces have an invalid shape")
        if len(prepared.features) != count or len(prepared.retirement_means) != count:
            raise ValueError("prepared terminal hazard rows are incomplete")

    def prepare_joint_outcomes(
        self,
        frame: pd.DataFrame,
        *,
        prediction_as_of: object | None = None,
        feature_as_of_col: str = "feature_as_of",
    ) -> PreparedTerminalHazards:
        """Prepare all deterministic terminal components once for a forecast."""

        if not self._fitted:
            raise RuntimeError("terminal hazard must be fitted before inference")
        self._validate_prediction_cutoff(
            frame,
            prediction_as_of=prediction_as_of,
            feature_as_of_col=feature_as_of_col,
        )
        features = engineer_survival_aware_race_features(frame)
        dns_values: list[float] = []
        exclusion_values: list[float] = []
        hazards: list[np.ndarray] = []
        event_masses: list[np.ndarray] = []
        survival_traces: list[np.ndarray] = []
        retirement_means: list[Mapping[str, float]] = []
        for _, row in features.iterrows():
            dns, exclusion, hazard, event_mass, survival_trace, means = self._row_hazard(row)
            dns_values.append(float(dns))
            exclusion_values.append(float(exclusion))
            hazards.append(np.asarray(hazard, dtype=float))
            event_masses.append(np.asarray(event_mass, dtype=float))
            survival_traces.append(np.asarray(survival_trace, dtype=float))
            retirement_means.append(dict(means))
        count = len(features)
        hazard_shape = (count, int(self.config.time_bins), len(_TIMED_CAUSES))
        trace_shape = (count, int(self.config.time_bins))
        prepared = PreparedTerminalHazards(
            features=features,
            driver_ids=self._driver_ids(frame),
            dns_probabilities=np.asarray(dns_values, dtype=float),
            exclusion_given_start_probabilities=np.asarray(exclusion_values, dtype=float),
            interval_hazards=(
                np.stack(hazards, axis=0)
                if hazards
                else np.empty(hazard_shape, dtype=float)
            ),
            event_masses=(
                np.stack(event_masses, axis=0)
                if event_masses
                else np.empty(hazard_shape, dtype=float)
            ),
            survival_traces=(
                np.stack(survival_traces, axis=0)
                if survival_traces
                else np.empty(trace_shape, dtype=float)
            ),
            retirement_means=tuple(retirement_means),
        )
        self._validate_prepared(frame, prepared)
        return prepared

    def _retirement_mean(self, status: TerminalStatus) -> float:
        alpha, beta = self._retirement_beta.get(status, (2.0, 2.0))
        return float(alpha / (alpha + beta))

    def retirement_beta(self, status: TerminalStatus) -> tuple[float, float]:
        """Compatibility diagnostic; joint sampling uses discrete bins."""

        if not self._fitted:
            raise RuntimeError("terminal hazard must be fitted before inference")
        return self._retirement_beta[status]

    def draw_shared_shocks(
        self,
        frame: pd.DataFrame,
        rng: np.random.Generator,
        *,
        prepared: PreparedTerminalHazards | None = None,
    ) -> SharedRaceShocks:
        if prepared is None:
            features = engineer_survival_aware_race_features(frame)
        else:
            self._validate_prepared(frame, prepared)
            features = prepared.features
        team_values = sorted(
            {
                value
                for value in (
                    self._group_value(row, ("team_name", "constructor_name", "team_id"))
                    for _, row in features.iterrows()
                )
                if value is not None
            }
        )
        pu_values = sorted(
            {
                value
                for value in (
                    self._group_value(
                        row,
                        ("power_unit", "power_unit_manufacturer", "engine_manufacturer"),
                    )
                    for _, row in features.iterrows()
                )
                if value is not None
            }
        )
        return SharedRaceShocks(
            event_chaos=float(rng.normal(0.0, self.config.shared_event_shock_std)),
            weather=float(rng.normal(0.0, self.config.shared_weather_incident_std)),
            safety_car=float(
                rng.normal(0.0, self.config.shared_safety_car_incident_std)
            ),
            team_mechanical={
                value: float(rng.normal(0.0, self.config.shared_team_mechanical_std))
                for value in team_values
            },
            power_unit_mechanical={
                value: float(
                    rng.normal(0.0, self.config.shared_power_unit_mechanical_std)
                )
                for value in pu_values
            },
            team_pace={
                value: float(rng.normal(0.0, self.config.shared_team_pace_std))
                for value in team_values
            },
        )

    def _shared_hazard_multiplier(
        self,
        row: pd.Series,
        shared: SharedRaceShocks,
    ) -> np.ndarray:
        """Return marginal-mean-one cause multipliers for one entrant.

        Each shared draw is Gaussian on the log scale.  Subtracting half of
        the cause-specific log variance prevents shared uncertainty from
        mechanically increasing expected hazard before any evidence is seen.
        Cross-driver and cross-cause dependence is preserved because the same
        underlying draws still enter every affected multiplier.
        """

        mech_index = _TIMED_CAUSES.index(TerminalStatus.MECHANICAL_POWER_UNIT)
        incident_index = _TIMED_CAUSES.index(TerminalStatus.COLLISION_INCIDENT)
        coarse_index = _TIMED_CAUSES.index(TerminalStatus.NON_CLASSIFIED)
        team = self._group_value(row, ("team_name", "constructor_name", "team_id"))
        power_unit = self._group_value(
            row,
            ("power_unit", "power_unit_manufacturer", "engine_manufacturer"),
        )
        team_key = team or ""
        power_unit_key = power_unit or ""
        wet = np.clip(self._numeric(row, "race_wet_probability") or 0.0, 0.0, 1.0)
        weather_uncertainty = np.clip(
            self._numeric(row, "race_weather_uncertainty") or 0.0,
            0.0,
            1.0,
        )
        safety_car = np.clip(
            self._numeric(row, "race_safety_car_probability") or 0.0,
            0.0,
            1.0,
        )

        log_multiplier = np.zeros(len(_TIMED_CAUSES), dtype=float)
        log_variance = np.zeros(len(_TIMED_CAUSES), dtype=float)
        event_std = float(self.config.shared_event_shock_std)

        mechanical_event_loading = 0.20
        log_multiplier[mech_index] += mechanical_event_loading * shared.event_chaos
        log_variance[mech_index] += (mechanical_event_loading * event_std) ** 2
        if team_key in shared.team_mechanical:
            log_multiplier[mech_index] += shared.team_mechanical[team_key]
            log_variance[mech_index] += float(
                self.config.shared_team_mechanical_std
            ) ** 2
        if power_unit_key in shared.power_unit_mechanical:
            log_multiplier[mech_index] += shared.power_unit_mechanical[power_unit_key]
            log_variance[mech_index] += float(
                self.config.shared_power_unit_mechanical_std
            ) ** 2

        incident_event_loading = 0.25 + 0.50 * wet + 0.25 * weather_uncertainty
        log_multiplier[incident_index] += (
            incident_event_loading * shared.event_chaos
        )
        log_variance[incident_index] += (incident_event_loading * event_std) ** 2
        log_multiplier[incident_index] += wet * shared.weather
        log_variance[incident_index] += (
            wet * float(self.config.shared_weather_incident_std)
        ) ** 2
        log_multiplier[incident_index] += safety_car * shared.safety_car
        log_variance[incident_index] += (
            safety_car * float(self.config.shared_safety_car_incident_std)
        ) ** 2

        coarse_event_loading = 0.30
        log_multiplier[coarse_index] += coarse_event_loading * shared.event_chaos
        log_variance[coarse_index] += (coarse_event_loading * event_std) ** 2
        return np.exp(log_multiplier - 0.5 * log_variance)

    def sample_joint_outcomes(
        self,
        frame: pd.DataFrame,
        rng: np.random.Generator,
        *,
        shocks: SharedRaceShocks | None = None,
        prepared: PreparedTerminalHazards | None = None,
    ) -> tuple[list[TerminalStatus], np.ndarray, SharedRaceShocks]:
        """Sample competing risks with event/team/PU shocks shared by drivers."""

        if not self._fitted:
            raise RuntimeError("terminal hazard must be fitted before inference")
        prepared_hazards = prepared or self.prepare_joint_outcomes(frame)
        self._validate_prepared(frame, prepared_hazards)
        features = prepared_hazards.features
        shared = shocks or self.draw_shared_shocks(
            frame,
            rng,
            prepared=prepared_hazards,
        )
        statuses: list[TerminalStatus] = []
        fractions = np.ones(len(features), dtype=float)
        for output_index, (_, row) in enumerate(features.iterrows()):
            dns = float(prepared_hazards.dns_probabilities[output_index])
            base_hazard = prepared_hazards.interval_hazards[output_index]
            if dns >= 1.0 - self.config.minimum_probability or rng.random() < dns:
                statuses.append(TerminalStatus.DNS_WITHDRAWAL)
                fractions[output_index] = 0.0
                continue
            hazard = base_hazard.copy()
            hazard *= self._shared_hazard_multiplier(row, shared)[None, :]
            hazard = np.vstack([self._cap_hazard_row(values) for values in hazard])
            sampled_status = TerminalStatus.CLASSIFIED_FINISH
            sampled_fraction = 1.0
            for bin_index, interval in enumerate(hazard):
                total = float(interval.sum())
                draw = float(rng.random())
                if draw >= total:
                    continue
                cause_draw = draw / max(total, self.config.minimum_probability)
                cumulative = np.cumsum(interval / total)
                cause_index = int(np.searchsorted(cumulative, cause_draw, side="right"))
                cause_index = min(cause_index, len(_TIMED_CAUSES) - 1)
                sampled_status = _TIMED_CAUSES[cause_index]
                sampled_fraction = float(
                    np.clip(
                        (bin_index + rng.random()) / float(self.config.time_bins),
                        0.0,
                        1.0,
                    )
                )
                break
            statuses.append(sampled_status)
            fractions[output_index] = sampled_fraction
            if rng.random() < float(prepared_hazards.exclusion_given_start_probabilities[output_index]):
                statuses[-1] = TerminalStatus.DISQUALIFIED
        return statuses, fractions, shared

    def predict_proba(
        self,
        frame: pd.DataFrame,
        *,
        prediction_as_of: object | None = None,
        feature_as_of_col: str = "feature_as_of",
        prepared: PreparedTerminalHazards | None = None,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("terminal hazard must be fitted before inference")
        if frame.empty:
            return pd.DataFrame(index=frame.index)
        self._validate_prediction_cutoff(
            frame,
            prediction_as_of=prediction_as_of,
            feature_as_of_col=feature_as_of_col,
        )
        prepared_hazards = prepared or self.prepare_joint_outcomes(
            frame,
            prediction_as_of=prediction_as_of,
            feature_as_of_col=feature_as_of_col,
        )
        self._validate_prepared(frame, prepared_hazards)
        features = prepared_hazards.features
        records: list[dict[str, object]] = []
        classified_index = TERMINAL_STATUSES.index(TerminalStatus.CLASSIFIED_FINISH)
        for output_index, (index, row) in enumerate(features.iterrows()):
            dns = float(prepared_hazards.dns_probabilities[output_index])
            event_mass = prepared_hazards.event_masses[output_index]
            hazard = prepared_hazards.interval_hazards[output_index]
            survival_trace = prepared_hazards.survival_traces[output_index]
            retirement_means = prepared_hazards.retirement_means[output_index]
            probabilities = self._probabilities_from_prepared_row(
                dns=dns,
                exclusion=float(prepared_hazards.exclusion_given_start_probabilities[output_index]),
                event_mass=event_mass,
                survival_trace=survival_trace,
            )
            record: dict[str, object] = {
                "driver_id": str(row.get("driver_id", index)),
                "p_terminal": float(1.0 - probabilities[classified_index]),
                "expected_retirement_fraction": float(
                    sum(
                        probabilities[offset] * retirement_means[status.value]
                        for offset, status in enumerate(TERMINAL_STATUSES)
                        if status is not TerminalStatus.DISQUALIFIED
                    )
                    / max(1.0 - float(prepared_hazards.exclusion_given_start_probabilities[output_index]), self.config.minimum_probability)
                ),
                "distance_semantics": "expected_running_distance_including_excluded_cars",
                "terminal_hazard_backend": self.backend,
                "terminal_hazard_time_bins": int(self.config.time_bins),
                "terminal_training_rows": self.training_rows,
                "terminal_timing_evidence_rows": self.timing_evidence_rows,
                "terminal_coarse_label_rows": self.coarse_terminal_rows,
                "terminal_training_max_as_of": self.training_max_as_of,
            }
            for offset, status in enumerate(TERMINAL_STATUSES):
                record[f"p_{status.value}"] = float(probabilities[offset])
                record[f"expected_retirement_fraction_{status.value}"] = retirement_means[
                    status.value
                ]
            for bin_index in range(int(self.config.time_bins)):
                record[f"terminal_interval_hazard_{bin_index + 1}"] = float(
                    hazard[bin_index].sum()
                )
                record[f"survival_through_interval_{bin_index + 1}"] = float(
                    survival_trace[bin_index]
                )
            records.append(record)
        return pd.DataFrame(records, index=frame.index)


__all__ = [
    "BinaryTerminalCalibrator",
    "PartialPooledTerminalHazard",
    "PreparedTerminalHazards",
    "SharedRaceShocks",
    "TerminalHazardConfig",
]
