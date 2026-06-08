"""Generic form/rolling-window helpers."""

from __future__ import annotations

from collections.abc import Iterable


def rolling_mean(values: Iterable[float | int | None], window: int) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    tail = clean[-max(1, int(window)) :]
    return sum(tail) / float(len(tail))
