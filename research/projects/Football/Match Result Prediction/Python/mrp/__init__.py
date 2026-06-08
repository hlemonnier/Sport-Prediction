"""Compatibility shim for the relocated football prediction package.

Authoritative code now lives in ``packages/football/mrp``. This shim keeps the
legacy experiment runner path importable.
"""

from __future__ import annotations

from pathlib import Path
import sys

_LEGACY_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[6]
_PACKAGE_PARENT = _REPO_ROOT / "packages" / "football"
_PACKAGE_DIR = _PACKAGE_PARENT / "mrp"
_SPORTS_CORE_PARENT = _REPO_ROOT / "packages" / "sports_core"

for _path in (_SPORTS_CORE_PARENT, _PACKAGE_PARENT):
    _path_text = str(_path)
    if _path.exists() and _path_text not in sys.path:
        sys.path.insert(0, _path_text)

__path__ = [str(_LEGACY_DIR), str(_PACKAGE_DIR)]

from .config import PredictionConfig  # noqa: E402
from .prediction import PredictionResult, run_prediction  # noqa: E402

__all__ = ["PredictionConfig", "PredictionResult", "run_prediction"]
