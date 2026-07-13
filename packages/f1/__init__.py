"""F1 domain package root with side-effect-free lazy public exports.

Importing a narrow module such as ``packages.f1.models.grouped_ranking`` must
not initialize every prediction backend.  In particular, the optional native
XGBoost and LightGBM runtimes are loaded only when their explicit fit or runtime
diagnostic paths are called.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BettingConfig": ("packages.f1.betting", "BettingConfig"),
    "CircuitCard": ("packages.f1.data.schemas", "CircuitCard"),
    "PredictionConfig": ("packages.f1.data.schemas", "PredictionConfig"),
    "PredictionResult": ("packages.f1.data.schemas", "PredictionResult"),
    "build_betting_recommendations": (
        "packages.f1.betting",
        "build_betting_recommendations",
    ),
    "circuit_card_from_event": (
        "packages.f1.data.schemas.circuit",
        "circuit_card_from_event",
    ),
    "run_prediction": ("packages.f1.orchestration.prediction", "run_prediction"),
}

__all__ = [
    "BettingConfig",
    "CircuitCard",
    "PredictionConfig",
    "PredictionResult",
    "build_betting_recommendations",
    "circuit_card_from_event",
    "run_prediction",
]


def __getattr__(name: str) -> Any:
    """Resolve one documented public symbol without eager backend imports."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
