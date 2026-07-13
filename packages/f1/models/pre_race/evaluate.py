"""Pre-race evaluation entrypoints."""

from __future__ import annotations

from packages.f1.orchestration.backtest import evaluate_prediction_rows
from packages.f1.models.pre_race.status import (
    TERMINAL_STATUSES,
    TerminalLabelGranularity,
    TerminalStatus,
    reason_code_terminal_status,
    terminal_label_granularity,
)

import numpy as np
import pandas as pd


def evaluate_pre_race_predictions(*args: object, **kwargs: object) -> object:
    """Evaluate predicted race order against classified race results."""

    return evaluate_prediction_rows(*args, **kwargs)


def evaluate_terminal_status_probabilities(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    *,
    driver_col: str = "driver_id",
    status_col: str = "terminal_status",
    calibration_bins: int = 5,
) -> dict[str, object]:
    """Score terminal probability calibration without reducing it to AUC."""

    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2")
    required_actual = {driver_col, status_col}
    probability_columns = [f"p_{status.value}" for status in TERMINAL_STATUSES]
    required_predicted = {driver_col, *probability_columns}
    if not required_actual.issubset(actual.columns):
        raise ValueError("actual terminal-status rows are incomplete")
    if not required_predicted.issubset(predicted.columns):
        raise ValueError("predicted terminal-status probabilities are incomplete")
    actual_columns = [driver_col, status_col]
    for column in ("retirement_fraction", "terminal_label_granularity", "race_status_raw"):
        if column in actual.columns:
            actual_columns.append(column)
    predicted_columns = [driver_col, *probability_columns]
    for column in predicted.columns:
        if column == "expected_retirement_fraction" or column.startswith(
            "expected_retirement_fraction_"
        ):
            predicted_columns.append(column)
    merged = actual[actual_columns].merge(
        predicted[predicted_columns],
        on=driver_col,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(actual) or len(merged) != len(predicted):
        raise ValueError("terminal-status evaluation requires identical complete rosters")
    encoded = merged[status_col].map(reason_code_terminal_status)
    if encoded.isna().any():
        raise ValueError("terminal-status evaluation found unknown actual status")
    probabilities = merged[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("terminal probabilities must be finite and non-negative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("terminal probabilities must sum to one per entrant")
    actual_index = np.asarray(
        [TERMINAL_STATUSES.index(status) for status in encoded], dtype=int
    )
    one_hot = np.eye(len(TERMINAL_STATUSES), dtype=float)[actual_index]
    multiclass_brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    log_loss = float(
        -np.mean(np.log(np.clip(probabilities[np.arange(len(merged)), actual_index], 1e-12, 1.0)))
    )
    classified_index = TERMINAL_STATUSES.index(TerminalStatus.CLASSIFIED_FINISH)
    predicted_terminal = 1.0 - probabilities[:, classified_index]
    actual_terminal = (actual_index != classified_index).astype(float)
    terminal_brier = float(np.mean((predicted_terminal - actual_terminal) ** 2))
    terminal_log_loss = float(
        -np.mean(
            actual_terminal * np.log(np.clip(predicted_terminal, 1e-12, 1.0))
            + (1.0 - actual_terminal)
            * np.log(np.clip(1.0 - predicted_terminal, 1e-12, 1.0))
        )
    )
    classwise_brier = {
        status.value: float(np.mean((probabilities[:, index] - one_hot[:, index]) ** 2))
        for index, status in enumerate(TERMINAL_STATUSES)
    }

    point_index = probabilities.argmax(axis=1)
    reason_recall: dict[str, float | None] = {}
    for status_index, status in enumerate(TERMINAL_STATUSES):
        mask = actual_index == status_index
        reason_recall[status.value] = (
            float(np.mean(point_index[mask] == status_index)) if mask.any() else None
        )

    bin_edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    calibration: list[dict[str, float | int]] = []
    for bin_index in range(calibration_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        mask = (
            (predicted_terminal >= lower)
            & (
                predicted_terminal <= upper
                if bin_index == calibration_bins - 1
                else predicted_terminal < upper
            )
        )
        if not mask.any():
            continue
        calibration.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_predicted_terminal": float(predicted_terminal[mask].mean()),
                "observed_terminal_rate": float(actual_terminal[mask].mean()),
            }
        )
    terminal_ece = float(
        sum(
            int(item["count"])
            * abs(
                float(item["mean_predicted_terminal"])
                - float(item["observed_terminal_rate"])
            )
            for item in calibration
        )
        / max(1, len(merged))
    )

    if "terminal_label_granularity" in merged.columns:
        granularity = merged["terminal_label_granularity"].astype(str)
    elif "race_status_raw" in merged.columns:
        granularity = merged["race_status_raw"].map(terminal_label_granularity).map(
            lambda value: value.value if value is not None else "unknown"
        )
    else:
        granularity = encoded.map(terminal_label_granularity).map(
            lambda value: value.value if value is not None else "unknown"
        )
    exact_reason_mask = granularity.eq(TerminalLabelGranularity.EXACT_CAUSE.value).to_numpy()
    coarse_mask = granularity.eq(TerminalLabelGranularity.COARSE_TERMINAL.value).to_numpy()
    exact_reason_log_loss = (
        float(
            -np.mean(
                np.log(
                    np.clip(
                        probabilities[
                            np.flatnonzero(exact_reason_mask),
                            actual_index[exact_reason_mask],
                        ],
                        1e-12,
                        1.0,
                    )
                )
            )
        )
        if exact_reason_mask.any()
        else None
    )

    timing_rows = 0
    retirement_fraction_mae: float | None = None
    retirement_fraction_mae_by_cause: dict[str, float] = {}
    if "retirement_fraction" in merged.columns:
        observed_fraction = pd.to_numeric(
            merged["retirement_fraction"], errors="coerce"
        ).to_numpy(dtype=float)
        predicted_fraction = np.full(len(merged), np.nan, dtype=float)
        for row_index, status in enumerate(encoded):
            cause_column = f"expected_retirement_fraction_{status.value}"
            source_column = (
                cause_column
                if cause_column in merged.columns
                else "expected_retirement_fraction"
                if "expected_retirement_fraction" in merged.columns
                else None
            )
            if source_column is not None:
                predicted_fraction[row_index] = pd.to_numeric(
                    pd.Series([merged.iloc[row_index][source_column]]), errors="coerce"
                ).iloc[0]
        timed_status = np.asarray(
            [status in {
                TerminalStatus.MECHANICAL_POWER_UNIT,
                TerminalStatus.COLLISION_INCIDENT,
                TerminalStatus.NON_CLASSIFIED,
            } for status in encoded],
            dtype=bool,
        )
        timing_mask = timed_status & np.isfinite(observed_fraction) & np.isfinite(
            predicted_fraction
        )
        timing_rows = int(timing_mask.sum())
        if timing_rows:
            retirement_fraction_mae = float(
                np.mean(np.abs(observed_fraction[timing_mask] - predicted_fraction[timing_mask]))
            )
            for status in (
                TerminalStatus.MECHANICAL_POWER_UNIT,
                TerminalStatus.COLLISION_INCIDENT,
                TerminalStatus.NON_CLASSIFIED,
            ):
                mask = timing_mask & np.asarray([value is status for value in encoded])
                if mask.any():
                    retirement_fraction_mae_by_cause[status.value] = float(
                        np.mean(np.abs(observed_fraction[mask] - predicted_fraction[mask]))
                    )
    return {
        "rows": int(len(merged)),
        "multiclass_brier": multiclass_brier,
        "terminal_brier": terminal_brier,
        "terminal_log_loss": terminal_log_loss,
        "multiclass_log_loss": log_loss,
        "classwise_brier": classwise_brier,
        "reason_recall": reason_recall,
        "exact_reason_rows": int(exact_reason_mask.sum()),
        "coarse_terminal_rows": int(coarse_mask.sum()),
        "exact_reason_log_loss": exact_reason_log_loss,
        "terminal_calibration": calibration,
        "terminal_expected_calibration_error": terminal_ece,
        "retirement_timing_rows": timing_rows,
        "retirement_fraction_mae": retirement_fraction_mae,
        "retirement_fraction_mae_by_cause": retirement_fraction_mae_by_cause,
    }


__all__ = [
    "evaluate_pre_race_predictions",
    "evaluate_prediction_rows",
    "evaluate_terminal_status_probabilities",
]
