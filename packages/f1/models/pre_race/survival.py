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
    interval.  Missing retirement distance is excluded from the timing fit but
    still informs the coarse outcome model.  Team, power-unit, driver and
    circuit evidence shrink to a recency-weighted global prior.
    """

    backend = "partial_pooled_discrete_competing_risk_v3"

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
                "backend": "sklearn_l2_logistic_discrete_hazard",
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
        at_risk = np.zeros(n_bins, dtype=float)
        event_counts = np.zeros((n_bins, len(_TIMED_CAUSES)), dtype=float)
        timing_rows = 0
        for index, status in encoded.items():
            if status is TerminalStatus.DNS_WITHDRAWAL:
                continue
            weight = float(weights.loc[index])
            fraction = fractions.loc[index]
            if status is TerminalStatus.CLASSIFIED_FINISH:
                at_risk += weight
                timing_rows += 1
                continue
            if pd.isna(fraction) or status not in _TIMED_CAUSES:
                continue
            event_bin = min(n_bins - 1, max(0, int(np.floor(float(fraction) * n_bins))))
            at_risk[: event_bin + 1] += weight
            event_counts[event_bin, _TIMED_CAUSES.index(status)] += weight
            timing_rows += 1

        starter_terminal = float(
            self._global[[status_index[cause] for cause in _TIMED_CAUSES]].sum()
        )
        starter_mass = max(
            self.config.minimum_probability,
            1.0 - float(self._global[status_index[TerminalStatus.DNS_WITHDRAWAL]]),
        )
        conditional_terminal = np.clip(starter_terminal / starter_mass, 0.0, 0.95)
        prior_total_hazard = 1.0 - (1.0 - conditional_terminal) ** (1.0 / n_bins)
        global_cause = np.asarray(
            [self._global[status_index[cause]] for cause in _TIMED_CAUSES],
            dtype=float,
        )
        cause_share = global_cause / max(global_cause.sum(), self.config.minimum_probability)
        prior_hazard = prior_total_hazard * cause_share
        baseline = np.empty_like(event_counts)
        for bin_index in range(n_bins):
            baseline[bin_index] = (
                event_counts[bin_index] + self.config.prior_strength * prior_hazard
            ) / (at_risk[bin_index] + self.config.prior_strength)
            baseline[bin_index] = self._cap_hazard_row(baseline[bin_index])
        self._baseline_hazard = baseline
        self.timing_evidence_rows = int(timing_rows)
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
        """Learn cause-specific interval hazards from strictly causal features."""

        features = engineer_survival_aware_race_features(rows)
        standardized = self._standardized_covariates(features, fit=True)
        n_bins = int(self.config.time_bins)
        expanded_design: list[np.ndarray] = []
        expanded_labels: list[np.ndarray] = []
        expanded_weights: list[float] = []
        encoded_values = encoded.to_numpy(dtype=object)
        fraction_values = fractions.to_numpy(dtype=float)
        weight_values = row_weights.to_numpy(dtype=float)
        for row_offset in range(len(rows)):
            status = encoded_values[row_offset]
            if status is TerminalStatus.DNS_WITHDRAWAL:
                continue
            fraction = fraction_values[row_offset]
            if status is TerminalStatus.CLASSIFIED_FINISH:
                final_bin = n_bins - 1
                event_bin: int | None = None
            elif pd.notna(fraction) and status in _TIMED_CAUSES:
                final_bin = min(
                    n_bins - 1,
                    max(0, int(np.floor(float(fraction) * n_bins))),
                )
                event_bin = final_bin
            else:
                # Unknown retirement timing is useful to the outcome prior but
                # cannot be invented for the interval likelihood.
                continue
            for bin_index in range(final_bin + 1):
                interval = np.zeros(n_bins, dtype=float)
                interval[bin_index] = 1.0
                expanded_design.append(
                    np.concatenate([interval, standardized[row_offset]])
                )
                label = np.zeros(len(_TIMED_CAUSES), dtype=int)
                if event_bin == bin_index:
                    label[_TIMED_CAUSES.index(status)] = 1
                expanded_labels.append(label)
                expanded_weights.append(float(weight_values[row_offset]))

        width = n_bins + 2 * len(_HAZARD_COVARIATE_COLUMNS)
        self._covariate_coefficients = np.zeros(
            (len(_TIMED_CAUSES), width), dtype=float
        )
        self._covariate_intercepts = np.zeros(len(_TIMED_CAUSES), dtype=float)
        self._covariate_fit_causes = set()
        if not expanded_design:
            return
        x = np.asarray(expanded_design, dtype=float)
        y = np.asarray(expanded_labels, dtype=int)
        sample_weight = np.asarray(expanded_weights, dtype=float)
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:
            raise RuntimeError(
                "regularized person-period terminal hazard requires scikit-learn"
            ) from exc
        for cause_index, cause in enumerate(_TIMED_CAUSES):
            target = y[:, cause_index]
            positives = int(target.sum())
            negatives = int((1 - target).sum())
            minimum = int(self.config.minimum_cause_events_for_covariate_fit)
            if positives < minimum or negatives < minimum:
                continue
            estimator = LogisticRegression(
                C=float(self.config.covariate_l2_c),
                solver="lbfgs",
                max_iter=1000,
                random_state=0,
            )
            estimator.fit(x, target, sample_weight=sample_weight)
            self._covariate_coefficients[cause_index] = estimator.coef_[0]
            self._covariate_intercepts[cause_index] = float(estimator.intercept_[0])
            self._covariate_fit_causes.add(cause)

    def _learned_interval_hazard(self, row: pd.Series) -> np.ndarray:
        hazard = self._baseline_hazard.copy()
        if not self._covariate_fit_causes:
            return hazard
        standardized = self._standardized_covariates(pd.DataFrame([row]), fit=False)[0]
        n_bins = int(self.config.time_bins)
        for bin_index in range(n_bins):
            interval = np.zeros(n_bins, dtype=float)
            interval[bin_index] = 1.0
            design = np.concatenate([interval, standardized])
            for cause_index, cause in enumerate(_TIMED_CAUSES):
                if cause not in self._covariate_fit_causes:
                    continue
                logit = self._covariate_intercepts[cause_index] + float(
                    np.dot(self._covariate_coefficients[cause_index], design)
                )
                hazard[bin_index, cause_index] = _sigmoid(logit)
        return hazard

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
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        pooled = self._pooled_outcome_prior(row)
        status_index = {status: index for index, status in enumerate(TERMINAL_STATUSES)}
        eligible = self._numeric(row, "race_starter_eligible")
        if eligible is not None and eligible <= 0.0:
            means = {status.value: self._retirement_mean(status) for status in TERMINAL_STATUSES}
            return 1.0, np.zeros_like(self._baseline_hazard), np.zeros_like(
                self._baseline_hazard
            ), np.zeros(int(self.config.time_bins)), means

        dns = float(
            np.clip(
                pooled[status_index[TerminalStatus.DNS_WITHDRAWAL]],
                self.config.minimum_probability,
                0.45,
            )
        )
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
        dns, hazard = self._apply_binary_calibration(dns, hazard)

        event_mass = np.zeros_like(hazard)
        survival_trace = np.zeros(int(self.config.time_bins), dtype=float)
        survival = 1.0 - dns
        for bin_index in range(int(self.config.time_bins)):
            event_mass[bin_index] = survival * hazard[bin_index]
            survival *= max(0.0, 1.0 - float(hazard[bin_index].sum()))
            survival_trace[bin_index] = survival
        probabilities = np.zeros(len(TERMINAL_STATUSES), dtype=float)
        probabilities[status_index[TerminalStatus.DNS_WITHDRAWAL]] = dns
        for cause_index, cause in enumerate(_TIMED_CAUSES):
            probabilities[status_index[cause]] = float(event_mass[:, cause_index].sum())
        probabilities[status_index[TerminalStatus.CLASSIFIED_FINISH]] = survival
        probabilities = np.clip(probabilities, self.config.minimum_probability, None)
        probabilities /= probabilities.sum()

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
        means[TerminalStatus.CLASSIFIED_FINISH.value] = 1.0
        return dns, hazard, event_mass, survival_trace, means

    def _predict_row(self, row: pd.Series) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
        dns, hazard, event_mass, survival_trace, means = self._row_hazard(row)
        probabilities = self._probabilities_from_prepared_row(
            dns=dns,
            event_mass=event_mass,
            survival_trace=survival_trace,
        )
        return probabilities, means, hazard, survival_trace

    def _probabilities_from_prepared_row(
        self,
        *,
        dns: float,
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
        for cause_index, cause in enumerate(_TIMED_CAUSES):
            probabilities[status_index[cause]] = float(event_mass[:, cause_index].sum())
        probabilities[status_index[TerminalStatus.CLASSIFIED_FINISH]] = float(
            survival_trace[-1]
        )
        probabilities = np.clip(probabilities, self.config.minimum_probability, None)
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
        hazards: list[np.ndarray] = []
        event_masses: list[np.ndarray] = []
        survival_traces: list[np.ndarray] = []
        retirement_means: list[Mapping[str, float]] = []
        for _, row in features.iterrows():
            dns, hazard, event_mass, survival_trace, means = self._row_hazard(row)
            dns_values.append(float(dns))
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
                    )
                ),
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
