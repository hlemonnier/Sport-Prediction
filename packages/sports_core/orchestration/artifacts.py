"""Artifact path helpers shared by backtests and predictions."""

from __future__ import annotations

from pathlib import Path


def artifact_path(root: str | Path, *parts: str) -> Path:
    target = Path(root).expanduser().joinpath(*parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
