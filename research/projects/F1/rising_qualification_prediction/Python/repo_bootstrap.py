"""Import bootstrap for legacy F1 scripts and tests."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_repo_root() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "packages").is_dir():
            parent_text = str(parent)
            if parent_text not in sys.path:
                sys.path.insert(0, parent_text)
            return
    raise RuntimeError("Could not locate sport-prediction repo root.")


ensure_repo_root()
