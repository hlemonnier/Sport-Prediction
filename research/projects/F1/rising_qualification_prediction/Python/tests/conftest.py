"""Pytest bootstrap for the relocated F1 package."""

from __future__ import annotations

from pathlib import Path
import sys


for parent in Path(__file__).resolve().parents:
    if (parent / "packages").is_dir():
        parent_text = str(parent)
        if parent_text not in sys.path:
            sys.path.insert(0, parent_text)
        break
