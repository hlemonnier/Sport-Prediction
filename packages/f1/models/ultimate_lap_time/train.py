"""Ultimate lap-time training surface."""

from __future__ import annotations

import pandas as pd

from packages.f1.models.ultimate_lap_time.model import (
    UltimateLapTimeConfig,
    UltimateLapTimeModel,
    fit_ultimate_lap_time_model,
)


def train_ultimate_lap_time(
    laps: pd.DataFrame,
    *,
    config: UltimateLapTimeConfig | None = None,
) -> UltimateLapTimeModel:
    """Train a deterministic theoretical best-lap pace baseline."""

    return fit_ultimate_lap_time_model(laps, config=config)


__all__ = ["train_ultimate_lap_time", "fit_ultimate_lap_time_model"]
