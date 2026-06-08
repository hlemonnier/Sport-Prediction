"""Ultimate lap-time model contract.

This is a reserved model surface for theoretical best-lap pace. It is not wired
into production predictions until we have reliable stint, tyre, fuel, traffic,
and weather training targets.
"""

from __future__ import annotations


def predict_ultimate_lap_time(*_args: object, **_kwargs: object) -> None:
    raise NotImplementedError("Ultimate lap-time model is an explicit future model surface.")
