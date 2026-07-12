"""Pre-race evaluation entrypoints."""

from __future__ import annotations

from packages.f1.orchestration.backtest import evaluate_prediction_rows
from packages.f1.models.pre_race.status import TERMINAL_STATUSES, TerminalStatus, reason_code_terminal_status

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
    merged = actual[[driver_col, status_col]].merge(
        predicted[[driver_col, *probability_columns]],
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
    return {
        "rows": int(len(merged)),
        "multiclass_brier": multiclass_brier,
        "terminal_brier": terminal_brier,
        "multiclass_log_loss": log_loss,
        "reason_recall": reason_recall,
        "terminal_calibration": calibration,
    }


__all__ = [
    "evaluate_pre_race_predictions",
    "evaluate_prediction_rows",
    "evaluate_terminal_status_probabilities",
]
