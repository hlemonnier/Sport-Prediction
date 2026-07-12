"""Joint terminal-status and Plackett-Luce race classification simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from packages.f1.domain.starting_grid import RacePredictionHorizon
from packages.f1.features.race import engineer_survival_aware_race_features
from packages.f1.models.pre_race.ranking import BradleyTerryOrderRanker
from packages.f1.models.pre_race.status import TERMINAL_STATUSES, TerminalStatus
from packages.f1.models.pre_race.status import reason_code_terminal_status
from packages.f1.models.pre_race.survival import PartialPooledTerminalHazard


_STATUS_COLUMNS = tuple(f"p_{status.value}" for status in TERMINAL_STATUSES)


def _hungarian_minimize(cost: np.ndarray) -> np.ndarray:
    """Exact deterministic square linear assignment without a SciPy dependency."""

    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("assignment cost must be a square matrix")
    n = matrix.shape[0]
    if n == 0:
        return np.asarray([], dtype=int)
    if not np.isfinite(matrix).all():
        raise ValueError("assignment cost must contain only finite values")
    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(n + 1, dtype=float)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    for row in range(1, n + 1):
        p[0] = row
        min_value = np.full(n + 1, np.inf, dtype=float)
        used = np.zeros(n + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, n + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(n + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = np.empty(n, dtype=int)
    for column in range(1, n + 1):
        assignment[p[column] - 1] = column - 1
    return assignment


def minimum_expected_absolute_assignment(
    position_samples: np.ndarray,
    *,
    status_groups: np.ndarray | None = None,
) -> np.ndarray:
    """Minimize posterior expected absolute-position loss.

    ``status_groups`` optionally imposes contiguous FIA-style eligibility
    blocks.  Race-distance ordering may mix classified finishers and late
    retirees; only DNS/withdrawals are forced behind cars that took the start.
    The Hungarian solution remains exact inside each constrained block.
    """

    samples = np.asarray(position_samples, dtype=int)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("position_samples must have shape (drivers, simulations)")
    n_drivers = samples.shape[0]
    expected_positions = np.arange(1, n_drivers + 1, dtype=float)
    cost = np.mean(
        np.abs(samples[:, :, None] - expected_positions[None, None, :]),
        axis=1,
    )
    # Resolve exact ties deterministically without materially changing loss.
    cost += np.arange(n_drivers, dtype=float)[:, None] * 1e-12
    assignment = np.empty(n_drivers, dtype=int)
    if status_groups is None:
        assignment[:] = _hungarian_minimize(cost) + 1
        return assignment
    groups = np.asarray(status_groups, dtype=int)
    if groups.shape != (n_drivers,):
        raise ValueError("status_groups must have one entry per driver")
    cursor = 0
    for group in sorted(np.unique(groups).tolist()):
        row_indices = np.flatnonzero(groups == group)
        columns = np.arange(cursor, cursor + len(row_indices), dtype=int)
        local = cost[np.ix_(row_indices, columns)]
        local_assignment = _hungarian_minimize(local)
        assignment[row_indices] = columns[local_assignment] + 1
        cursor += len(row_indices)
    return assignment


@dataclass(frozen=True)
class JointRaceForecast:
    """Inspectable joint posterior and reconciled legal point classification."""

    point_classification: pd.DataFrame
    status_probabilities: pd.DataFrame
    position_probabilities: pd.DataFrame
    position_samples: np.ndarray
    status_samples: np.ndarray
    horizon: RacePredictionHorizon
    prediction_as_of: str | None
    simulations: int
    seed: int

    def __post_init__(self) -> None:
        self.position_samples.setflags(write=False)
        self.status_samples.setflags(write=False)


class SurvivalAwareRaceModel:
    """Walk-forward Race Final Position model over complete event rosters."""

    def __init__(
        self,
        terminal_model: PartialPooledTerminalHazard | None = None,
        order_model: BradleyTerryOrderRanker | None = None,
    ) -> None:
        self.terminal_model = terminal_model or PartialPooledTerminalHazard()
        self.order_model = order_model or BradleyTerryOrderRanker()
        self._fitted = False

    def fit(
        self,
        history: pd.DataFrame,
        *,
        event_col: str = "event_key",
        finish_position_col: str = "finish_position",
        terminal_status_col: str = "terminal_status",
        retirement_fraction_col: str = "retirement_fraction",
        event_as_of_col: str = "event_as_of",
        cutoff: object | None = None,
    ) -> "SurvivalAwareRaceModel":
        """Fit both factors on past complete events using one causal cutoff."""

        required = {event_col, "driver_id", finish_position_col, terminal_status_col}
        missing = sorted(required - set(history.columns))
        if missing:
            raise ValueError(f"joint race history missing required columns: {missing}")
        rows = history.copy()
        if cutoff is not None:
            if event_as_of_col not in rows.columns:
                raise ValueError("joint causal cutoff requires event_as_of evidence")
            event_times = pd.to_datetime(rows[event_as_of_col], errors="coerce", utc=True)
            cutoff_time = pd.to_datetime(cutoff, errors="coerce", utc=True)
            if pd.isna(cutoff_time) or event_times.isna().any():
                raise ValueError("joint race history has invalid causal timestamps")
            rows = rows.loc[event_times < cutoff_time].copy()
            if rows.empty:
                raise ValueError("joint race history has no complete event before cutoff")
        if rows.duplicated([event_col, "driver_id"]).any():
            raise ValueError("joint race history has duplicate driver rows inside an event")
        encoded_status = rows[terminal_status_col].map(reason_code_terminal_status)
        if encoded_status.isna().any():
            examples = sorted(
                rows.loc[encoded_status.isna(), terminal_status_col]
                .astype(str)
                .unique()
                .tolist()
            )[:5]
            raise ValueError(f"joint race history has unknown terminal statuses: {examples}")
        numeric_finish = pd.to_numeric(rows[finish_position_col], errors="coerce")
        if numeric_finish.isna().any():
            raise ValueError("joint race history has missing finish positions")
        for event, group in rows.assign(_finish=numeric_finish).groupby(
            event_col, sort=True, dropna=False
        ):
            positions = sorted(group["_finish"].astype(int).tolist())
            if positions != list(range(1, len(group) + 1)):
                raise ValueError(
                    f"joint race history event {event!r} is not a complete classification permutation"
                )
        self.terminal_model.fit(
            rows,
            status_col=terminal_status_col,
            event_as_of_col=event_as_of_col,
            cutoff=cutoff,
            retirement_fraction_col=retirement_fraction_col,
        )
        self.order_model.fit(
            rows,
            event_col=event_col,
            target_col=finish_position_col,
            terminal_status_col=terminal_status_col,
            event_as_of_col=event_as_of_col,
            cutoff=cutoff,
        )
        self._fitted = True
        return self

    @property
    def model_card(self) -> dict[str, object]:
        if not self._fitted:
            raise RuntimeError("joint race model must be fitted before inspection")
        return {
            "factorization": "terminal_status_time_x_conditional_running_order",
            "terminal": self.terminal_model.model_card,
            "conditional_order": {
                "backend": self.order_model.backend,
                "training_events": self.order_model.training_events,
                "training_pairs": self.order_model.training_pairs,
                "training_max_as_of": self.order_model.training_max_as_of,
                "grid_prior_weight": self.order_model.config.grid_prior_weight,
                "residual_weight": self.order_model.config.residual_weight,
                "coefficients": self.order_model.coefficients,
            },
        }

    @staticmethod
    def _validate_roster(
        features: pd.DataFrame,
        *,
        horizon: RacePredictionHorizon,
    ) -> pd.DataFrame:
        if features.empty:
            raise ValueError("joint race prediction requires a complete event roster")
        if "driver_id" not in features.columns:
            raise ValueError("joint race roster requires driver_id")
        rows = engineer_survival_aware_race_features(features.reset_index(drop=True))
        driver_ids = rows["driver_id"].astype(str).str.strip()
        if driver_ids.eq("").any() or driver_ids.duplicated().any():
            raise ValueError("joint race roster requires unique non-empty driver_id values")
        if horizon is RacePredictionHorizon.POST_GRID_PRE_RACE:
            required_columns = {
                "grid_snapshot_available",
                "grid_evidence_complete",
                "grid_resolution_status",
                "race_information_horizon",
            }
            missing = sorted(required_columns - set(rows.columns))
            if missing:
                raise ValueError(
                    "post_grid_pre_race requires immutable snapshot columns: "
                    f"{missing}"
                )
            if not rows["grid_snapshot_available"].fillna(False).astype(bool).all():
                raise ValueError("post_grid_pre_race snapshot is unavailable")
            if not rows["grid_evidence_complete"].fillna(False).astype(bool).all():
                raise ValueError("post_grid_pre_race grid evidence is incomplete")
            if not rows["grid_resolution_status"].astype(str).eq("resolved").all():
                raise ValueError("post_grid_pre_race grid is not resolver-approved")
            if not rows["race_information_horizon"].astype(str).eq(horizon.value).all():
                raise ValueError("race roster mixes prediction horizons")
        eligibility = pd.to_numeric(rows["race_starter_eligible"], errors="coerce")
        if (
            horizon is RacePredictionHorizon.POST_QUALIFYING_PRE_GRID
            and eligibility.isna().any()
        ):
            # This is an explicit provisional entrant assumption, not final
            # starter evidence.  The hazard may still sample DNS/withdrawal.
            eligibility = eligibility.fillna(1.0)
            rows["race_starter_eligible"] = eligibility
            rows["race_proxy_starter_assumption"] = True
        else:
            rows["race_proxy_starter_assumption"] = False
        if eligibility.isna().any():
            raise ValueError("starter eligibility is unresolved for at least one entrant")
        starter = eligibility.gt(0.0)
        pit_lane = pd.to_numeric(rows["race_pit_lane_start"], errors="coerce").fillna(0.0).gt(0.0)
        numeric_grid = pd.to_numeric(rows.get("grid_position"), errors="coerce")
        grid_values = numeric_grid.dropna()
        if grid_values.duplicated().any() or grid_values.le(0.0).any():
            raise ValueError("race roster physical grid positions must be unique and positive")
        if (starter & ~pit_lane & numeric_grid.isna()).any():
            raise ValueError("eligible non-pit-lane starters require a physical grid position")
        if (pit_lane & numeric_grid.notna()).any():
            raise ValueError("pit-lane starters must not carry a synthetic physical grid position")
        if (~starter & numeric_grid.notna()).any():
            raise ValueError("declared nonstarters must not carry a physical grid position")
        return rows

    def predict_joint(
        self,
        roster_features: pd.DataFrame,
        *,
        horizon: RacePredictionHorizon = RacePredictionHorizon.POST_GRID_PRE_RACE,
        prediction_as_of: object | None = None,
        simulations: int = 4000,
        seed: int = 17,
        plackett_luce_temperature: float = 1.0,
    ) -> JointRaceForecast:
        if not self._fitted:
            raise RuntimeError("joint race model must be fitted before prediction")
        if simulations < 100:
            raise ValueError("simulations must be at least 100")
        if plackett_luce_temperature <= 0.0:
            raise ValueError("plackett_luce_temperature must be positive")
        rows = self._validate_roster(roster_features, horizon=horizon)
        terminal = self.terminal_model.predict_proba(
            rows,
            prediction_as_of=prediction_as_of,
        ).reset_index(drop=True)
        order = self.order_model.score(
            rows,
            prediction_as_of=prediction_as_of,
        ).reset_index(drop=True)
        drivers = rows["driver_id"].astype(str).to_numpy()
        n_drivers = len(drivers)
        probability_matrix = terminal.loc[:, _STATUS_COLUMNS].to_numpy(dtype=float)
        probability_matrix /= probability_matrix.sum(axis=1, keepdims=True)
        conditional_scores = order["conditional_order_score"].to_numpy(dtype=float)
        grid = pd.to_numeric(rows.get("grid_position"), errors="coerce").fillna(
            n_drivers + 1
        ).to_numpy(dtype=float)
        scheduled_laps = pd.to_numeric(
            rows.get("race_scheduled_laps", pd.Series(60.0, index=rows.index)),
            errors="coerce",
        ).fillna(60.0).clip(lower=1.0).to_numpy(dtype=float)

        rng = np.random.default_rng(seed)
        position_samples = np.empty((n_drivers, simulations), dtype=np.int16)
        status_samples = np.empty((n_drivers, simulations), dtype="U24")
        cumulative = probability_matrix.cumsum(axis=1)
        for simulation in range(simulations):
            draws = rng.random(n_drivers)
            status_indices = np.asarray(
                [np.searchsorted(cumulative[row], draws[row], side="right") for row in range(n_drivers)],
                dtype=int,
            )
            status_indices = np.minimum(status_indices, len(TERMINAL_STATUSES) - 1)
            statuses = [TERMINAL_STATUSES[index] for index in status_indices]
            status_samples[:, simulation] = [status.value for status in statuses]

            retirement_fraction = np.empty(n_drivers, dtype=float)
            for row, status in enumerate(statuses):
                alpha, beta = self.terminal_model.retirement_beta(status)
                retirement_fraction[row] = rng.beta(alpha, beta)
            gumbel = rng.gumbel(0.0, plackett_luce_temperature, size=n_drivers)
            sampled_utility = conditional_scores + gumbel

            starter_rows = [
                row
                for row, status in enumerate(statuses)
                if status is not TerminalStatus.DNS_WITHDRAWAL
            ]
            dns_rows = [
                row
                for row, status in enumerate(statuses)
                if status is TerminalStatus.DNS_WITHDRAWAL
            ]
            completed_laps = np.floor(retirement_fraction * scheduled_laps).astype(int)
            starter_rows.sort(
                key=lambda row: (
                    -completed_laps[row],
                    -retirement_fraction[row],
                    -sampled_utility[row],
                    drivers[row],
                )
            )
            dns_rows.sort(key=lambda row: (grid[row], -sampled_utility[row], drivers[row]))
            classification = [*starter_rows, *dns_rows]
            for position, row in enumerate(classification, start=1):
                position_samples[row, simulation] = position

        modal_indices = np.argmax(probability_matrix, axis=1)
        modal_statuses = [TERMINAL_STATUSES[index] for index in modal_indices]
        status_groups = np.asarray(
            [
                1 if status is TerminalStatus.DNS_WITHDRAWAL else 0
                for status in modal_statuses
            ],
            dtype=int,
        )
        point_positions = minimum_expected_absolute_assignment(
            position_samples,
            status_groups=status_groups,
        )

        position_probability = pd.DataFrame(
            {
                f"p_position_{position}": np.mean(position_samples == position, axis=1)
                for position in range(1, n_drivers + 1)
            }
        )
        position_probability.insert(0, "driver_id", drivers)
        status_probability = terminal.loc[
            :,
            ["driver_id", "p_terminal", "expected_retirement_fraction", *_STATUS_COLUMNS],
        ].copy()
        point = pd.DataFrame(
            {
                "driver_id": drivers,
                "predicted_position": point_positions,
                "predicted_terminal_status": [status.value for status in modal_statuses],
                "predicted_status_probability": probability_matrix[
                    np.arange(n_drivers), modal_indices
                ],
                "expected_position": position_samples.mean(axis=1),
                "median_position": np.median(position_samples, axis=1),
                "conditional_order_score": conditional_scores,
                "grid_position": grid,
                "starter_eligible": pd.to_numeric(
                    rows["race_starter_eligible"], errors="coerce"
                ).astype(bool),
            }
        ).sort_values("predicted_position", kind="mergesort").reset_index(drop=True)
        if point["predicted_position"].tolist() != list(range(1, n_drivers + 1)):
            raise RuntimeError("joint race reconciliation emitted an illegal permutation")

        prediction_text = None
        if prediction_as_of is not None:
            parsed = pd.to_datetime(prediction_as_of, errors="coerce", utc=True)
            if pd.isna(parsed):
                raise ValueError("prediction_as_of must be a valid timestamp")
            prediction_text = parsed.isoformat().replace("+00:00", "Z")
        return JointRaceForecast(
            point_classification=point,
            status_probabilities=status_probability,
            position_probabilities=position_probability,
            position_samples=position_samples,
            status_samples=status_samples,
            horizon=horizon,
            prediction_as_of=prediction_text,
            simulations=simulations,
            seed=seed,
        )


__all__ = [
    "JointRaceForecast",
    "SurvivalAwareRaceModel",
    "minimum_expected_absolute_assignment",
]
