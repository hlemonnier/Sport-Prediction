"""Path resolution helpers shared by sport packages."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root without depending on fixed parent depth."""

    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() and (candidate / "research").exists():
            return candidate
        if (candidate / "packages").exists() and (candidate / "research").exists():
            return candidate
    raise RuntimeError(f"Could not resolve repository root from {current}")
