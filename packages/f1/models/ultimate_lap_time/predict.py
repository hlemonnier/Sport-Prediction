"""Ultimate lap-time prediction entrypoints."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from packages.f1.models.ultimate_lap_time.model import UltimateLapTimeModel


def predict_ultimate_lap_time(
    model: UltimateLapTimeModel,
    context: Mapping[str, Any] | pd.Series | pd.DataFrame,
    *,
    return_details: bool = False,
) -> float | pd.Series | pd.DataFrame:
    """Predict theoretical best-lap pace for one or more context rows."""

    if not isinstance(model, UltimateLapTimeModel):
        raise TypeError("model must be an UltimateLapTimeModel returned by train_ultimate_lap_time")
    if return_details:
        return model.predict_details(context)
    return model.predict(context)


__all__ = ["predict_ultimate_lap_time"]
