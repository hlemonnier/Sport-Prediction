"""F1 circuit-card feature helpers."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from packages.f1.data.schemas.circuit import (
    CIRCUIT_INTERACTION_FEATURES,
    CIRCUIT_NUMERIC_FEATURES,
    attach_circuit_card,
    circuit_card_from_event,
)


def circuit_feature_payload(event_name: object, track_stats: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return model-ready circuit-card metadata and numeric priors."""

    return circuit_card_from_event(event_name, dict(track_stats or {})).to_payload()


def attach_circuit_features(
    frame: pd.DataFrame,
    *,
    event_name: object,
    track_stats: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Attach a circuit-card feature row to every driver row in a frame."""

    if frame.empty:
        return frame
    return attach_circuit_card(frame, event_name=event_name, track_stats=dict(track_stats or {}))


__all__ = [
    "CIRCUIT_INTERACTION_FEATURES",
    "CIRCUIT_NUMERIC_FEATURES",
    "attach_circuit_features",
    "circuit_feature_payload",
]
