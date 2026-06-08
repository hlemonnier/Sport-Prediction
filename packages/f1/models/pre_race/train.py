"""Pre-race training entrypoints."""

from __future__ import annotations

from packages.f1.models.training import train_model


def train_pre_race_model(*args: object, **kwargs: object) -> object:
    """Train the shared rank model for race-order targets."""

    return train_model(*args, **kwargs)


__all__ = ["train_model", "train_pre_race_model"]
