"""Runtime doctor for optional F1 learning-to-rank implementations."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import json
from typing import Optional


@dataclass(frozen=True)
class OptionalModelRuntime:
    package: str
    installed: bool
    importable: bool
    version: Optional[str]
    issue: Optional[str]
    detail: Optional[str]

    def to_payload(self) -> dict[str, object]:
        return {
            "package": self.package,
            "installed": bool(self.installed),
            "importable": bool(self.importable),
            "version": self.version,
            "issue": self.issue,
            "detail": self.detail,
        }


def inspect_optional_model_runtime(package: str) -> OptionalModelRuntime:
    """Inspect an optional model package without making it a hard dependency."""

    normalized = str(package).strip()
    if normalized not in {"xgboost", "lightgbm"}:
        raise ValueError("supported optional model packages are xgboost and lightgbm")
    if importlib.util.find_spec(normalized) is None:
        return OptionalModelRuntime(normalized, False, False, None, "package_not_installed", None)
    try:
        module = importlib.import_module(normalized)
        return OptionalModelRuntime(
            package=normalized,
            installed=True,
            importable=True,
            version=str(getattr(module, "__version__", "unknown")),
            issue=None,
            detail=None,
        )
    except Exception as exc:  # optional native extensions fail in several platform-specific ways
        detail = f"{type(exc).__name__}: {exc}"
        lowered = detail.lower()
        if "libomp" in lowered or "openmp" in lowered or "omp.dylib" in lowered:
            issue = "openmp_runtime_missing"
        else:
            issue = "package_import_failed"
        return OptionalModelRuntime(normalized, True, False, None, issue, detail[:1000])


def f1_model_runtime_doctor() -> dict[str, object]:
    """Report the deterministic sklearn fallback and optional LTR runtimes."""

    try:
        from sklearn.linear_model import HuberRegressor, LogisticRegression  # noqa: F401

        fallback = {"available": True, "issue": None}
    except Exception as exc:  # pragma: no cover - only exercised in broken runtimes
        fallback = {"available": False, "issue": f"{type(exc).__name__}: {exc}"[:1000]}
    optional = [inspect_optional_model_runtime(name).to_payload() for name in ("xgboost", "lightgbm")]
    return {
        "sklearn_fallback": fallback,
        "optional_learning_to_rank": optional,
        "ready": bool(fallback["available"]),
        "policy": "sklearn fallback remains authoritative unless an optional LTR model passes event-block gates",
    }


def main() -> int:
    payload = f1_model_runtime_doctor()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if bool(payload["ready"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "OptionalModelRuntime",
    "f1_model_runtime_doctor",
    "inspect_optional_model_runtime",
]
