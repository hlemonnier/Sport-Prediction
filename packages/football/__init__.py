"""Football domain package root."""

from __future__ import annotations

from pathlib import Path
import sys

_PACKAGES_ROOT = Path(__file__).resolve().parents[1]
_SPORTS_CORE = _PACKAGES_ROOT / "sports_core"
for _path in (_SPORTS_CORE, _PACKAGES_ROOT / "football"):
    _path_text = str(_path)
    if _path.exists() and _path_text not in sys.path:
        sys.path.insert(0, _path_text)
