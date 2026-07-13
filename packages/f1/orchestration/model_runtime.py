"""Runtime discovery for optional F1 tree-model challengers.

XGBoost and LightGBM are intentionally optional.  On macOS their wheels load
``libomp.dylib`` at import time, which means checking ``find_spec`` alone can
report a package as present even though it is unusable.  This module discovers
an environment-local OpenMP runtime, preloads it with global symbol visibility,
and reports the exact runtime used before importing either package.

Nothing here installs or mutates system software.  Missing native runtimes are
reported as an explicit unavailable state and callers remain free to use the
deterministic sklearn challengers.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import glob
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Optional, Sequence


SUPPORTED_OPTIONAL_MODEL_PACKAGES: tuple[str, ...] = ("xgboost", "lightgbm")
OPENMP_PATH_ENV = "F1_LIBOMP_PATH"


@dataclass(frozen=True)
class OpenMPRuntimeCandidate:
    """One side-effect-free OpenMP runtime discovery result."""

    path: str
    source: str
    exists: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "source": self.source,
            "exists": bool(self.exists),
        }


@dataclass(frozen=True)
class OpenMPRuntimeStatus:
    """Result of attempting to make one OpenMP runtime process-visible."""

    status: str
    selected_path: Optional[str]
    selected_source: Optional[str]
    preloaded: bool
    candidates: tuple[OpenMPRuntimeCandidate, ...]
    issue: Optional[str] = None
    detail: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.status == "available" and self.preloaded

    def to_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "available": self.available,
            "selected_path": self.selected_path,
            "selected_source": self.selected_source,
            "preloaded": bool(self.preloaded),
            "issue": self.issue,
            "detail": self.detail,
            "candidates": [candidate.to_payload() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class OptionalModelRuntime:
    package: str
    installed: bool
    importable: bool
    version: Optional[str]
    issue: Optional[str]
    detail: Optional[str]
    status: str = "unavailable"
    openmp_runtime_path: Optional[str] = None
    openmp_runtime_preloaded: bool = False

    @property
    def available(self) -> bool:
        return self.status == "available" and self.importable

    def to_payload(self) -> dict[str, object]:
        return {
            "package": self.package,
            "status": self.status,
            "available": self.available,
            "installed": bool(self.installed),
            "importable": bool(self.importable),
            "version": self.version,
            "issue": self.issue,
            "detail": self.detail,
            "openmp_runtime_path": self.openmp_runtime_path,
            "openmp_runtime_preloaded": bool(self.openmp_runtime_preloaded),
        }


def _library_names() -> tuple[str, ...]:
    system = platform.system().lower()
    if system == "darwin":
        return ("libomp.dylib", "libgomp.dylib")
    if system == "windows":
        return ("libomp.dll", "libgomp-1.dll", "vcomp140.dll")
    return ("libomp.so", "libomp.so.5", "libgomp.so.1", "libgomp.so")


def _candidate_from_path(value: object, source: str) -> list[OpenMPRuntimeCandidate]:
    text = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if not text:
        return []
    path = Path(text)
    if path.is_dir():
        return [
            OpenMPRuntimeCandidate(str(path / name), source, (path / name).is_file())
            for name in _library_names()
        ]
    return [OpenMPRuntimeCandidate(str(path), source, path.is_file())]


def discover_openmp_runtimes(
    *,
    extra_paths: Sequence[str | os.PathLike[str]] = (),
) -> tuple[OpenMPRuntimeCandidate, ...]:
    """Discover likely OpenMP libraries without importing optional models.

    Priority is explicit override, the active Python environment, package-local
    runtimes, and finally common system package-manager locations.  Existing
    candidates are returned first.  A non-existent explicit override is kept in
    the report so a bad deployment setting is diagnosable.
    """

    candidates: list[OpenMPRuntimeCandidate] = []
    explicit = os.environ.get(OPENMP_PATH_ENV)
    if explicit:
        candidates.extend(_candidate_from_path(explicit, f"env:{OPENMP_PATH_ENV}"))
    for path in extra_paths:
        candidates.extend(_candidate_from_path(path, "explicit_argument"))

    prefixes: list[tuple[str, str]] = [(sys.prefix, "sys_prefix")]
    for variable in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        value = os.environ.get(variable)
        if value:
            prefixes.append((value, f"env:{variable}"))
    for prefix, source in prefixes:
        candidates.extend(_candidate_from_path(Path(prefix) / "lib", source))
        candidates.extend(_candidate_from_path(Path(prefix) / "Library" / "bin", source))

    # scikit-learn wheels often bundle a compatible runtime.  ``find_spec`` is
    # deliberately used instead of importing sklearn, which can itself load OMP.
    try:
        sklearn_spec = importlib.util.find_spec("sklearn")
    except (ImportError, ValueError):  # a partially imported package can raise
        sklearn_spec = None
    if sklearn_spec is not None and sklearn_spec.submodule_search_locations:
        for root in sklearn_spec.submodule_search_locations:
            candidates.extend(_candidate_from_path(Path(root) / ".dylibs", "sklearn_wheel"))
            candidates.extend(_candidate_from_path(Path(root) / ".libs", "sklearn_wheel"))

    common_directories = (
        "/opt/homebrew/opt/libomp/lib",
        "/usr/local/opt/libomp/lib",
        "/opt/local/lib/libomp",
        "/usr/local/lib",
        "/usr/lib",
    )
    for directory in common_directories:
        candidates.extend(_candidate_from_path(directory, "system_search"))
    if platform.system().lower() == "linux":
        for match in sorted(glob.glob("/usr/lib/*/libgomp.so.1")):
            candidates.extend(_candidate_from_path(match, "system_search"))

    # Preserve priority and collapse aliases resolving to the same file.
    seen: set[str] = set()
    unique: list[OpenMPRuntimeCandidate] = []
    for candidate in candidates:
        key = os.path.realpath(candidate.path) if candidate.exists else os.path.abspath(candidate.path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    unique.sort(key=lambda item: (not item.exists, candidates.index(item)))
    return tuple(unique)


# Keeping the CDLL object alive prevents an implementation from unloading the
# runtime while XGBoost or LightGBM still references its global symbols.
_OPENMP_HANDLES: dict[str, Any] = {}


def _load_openmp_library(path: str) -> Any:
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    return ctypes.CDLL(path, mode=mode)


def preload_openmp_runtime(
    path: str | os.PathLike[str] | None = None,
) -> OpenMPRuntimeStatus:
    """Preload the first usable OpenMP runtime into the current process."""

    extra = () if path is None else (path,)
    candidates = discover_openmp_runtimes(extra_paths=extra)
    errors: list[str] = []
    for candidate in candidates:
        if not candidate.exists:
            continue
        resolved = os.path.realpath(candidate.path)
        if resolved in _OPENMP_HANDLES:
            return OpenMPRuntimeStatus(
                status="available",
                selected_path=candidate.path,
                selected_source=candidate.source,
                preloaded=True,
                candidates=candidates,
                detail="OpenMP runtime was already preloaded in this process",
            )
        try:
            _OPENMP_HANDLES[resolved] = _load_openmp_library(candidate.path)
            return OpenMPRuntimeStatus(
                status="available",
                selected_path=candidate.path,
                selected_source=candidate.source,
                preloaded=True,
                candidates=candidates,
            )
        except (OSError, RuntimeError) as exc:
            errors.append(f"{candidate.path}: {type(exc).__name__}: {exc}")

    existing = [candidate for candidate in candidates if candidate.exists]
    if existing:
        issue = "openmp_runtime_load_failed"
        detail = "; ".join(errors)[:2000] or "discovered runtime could not be loaded"
    else:
        issue = "openmp_runtime_not_found"
        explicit_candidates = [
            candidate.path
            for candidate in candidates
            if candidate.source in {f"env:{OPENMP_PATH_ENV}", "explicit_argument"}
        ]
        detail = (
            f"configured candidate(s) do not exist: {explicit_candidates}"
            if explicit_candidates
            else "no environment-local or system OpenMP runtime was discovered"
        )
    return OpenMPRuntimeStatus(
        status="unavailable",
        selected_path=None,
        selected_source=None,
        preloaded=False,
        candidates=candidates,
        issue=issue,
        detail=detail,
    )


def _installed_version(package: str) -> Optional[str]:
    try:
        return str(importlib.metadata.version(package))
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:  # metadata is diagnostic only
        return None


def inspect_optional_model_runtime(package: str) -> OptionalModelRuntime:
    """Inspect and, when possible, activate an optional model package."""

    normalized = str(package).strip().lower()
    if normalized not in SUPPORTED_OPTIONAL_MODEL_PACKAGES:
        raise ValueError("supported optional model packages are xgboost and lightgbm")
    try:
        spec = importlib.util.find_spec(normalized)
    except (ImportError, ValueError):
        spec = None
    version = _installed_version(normalized)
    if spec is None:
        return OptionalModelRuntime(
            normalized,
            False,
            False,
            version,
            "package_not_installed",
            None,
            status="unavailable",
        )

    openmp = preload_openmp_runtime()
    try:
        module = importlib.import_module(normalized)
        return OptionalModelRuntime(
            package=normalized,
            installed=True,
            importable=True,
            version=str(getattr(module, "__version__", version or "unknown")),
            issue=None,
            detail=None,
            status="available",
            openmp_runtime_path=openmp.selected_path,
            openmp_runtime_preloaded=openmp.preloaded,
        )
    except Exception as exc:  # optional native extensions fail in platform-specific ways
        detail = f"{type(exc).__name__}: {exc}"
        lowered = detail.lower()
        if "libomp" in lowered or "openmp" in lowered or "omp.dylib" in lowered or "libgomp" in lowered:
            issue = (
                openmp.issue
                if openmp.issue in {"openmp_runtime_not_found", "openmp_runtime_load_failed"}
                else "openmp_runtime_missing"
            )
        else:
            issue = "package_import_failed"
        return OptionalModelRuntime(
            normalized,
            True,
            False,
            version,
            issue,
            detail[:2000],
            status="unavailable",
            openmp_runtime_path=openmp.selected_path,
            openmp_runtime_preloaded=openmp.preloaded,
        )


def f1_model_runtime_doctor() -> dict[str, object]:
    """Report deterministic fallback and optional native-model availability."""

    openmp = preload_openmp_runtime()
    try:
        from sklearn.linear_model import HuberRegressor, LogisticRegression  # noqa: F401

        fallback: dict[str, object] = {
            "status": "available",
            "available": True,
            "version": _installed_version("scikit-learn"),
            "issue": None,
        }
    except Exception as exc:  # pragma: no cover - only exercised in broken runtimes
        fallback = {
            "status": "unavailable",
            "available": False,
            "version": _installed_version("scikit-learn"),
            "issue": f"{type(exc).__name__}: {exc}"[:1000],
        }
    optional = [
        inspect_optional_model_runtime(name).to_payload()
        for name in SUPPORTED_OPTIONAL_MODEL_PACKAGES
    ]
    return {
        "openmp_runtime": openmp.to_payload(),
        "sklearn_fallback": fallback,
        "optional_learning_to_rank": optional,
        "optional_available": any(bool(item["available"]) for item in optional),
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
    "OPENMP_PATH_ENV",
    "SUPPORTED_OPTIONAL_MODEL_PACKAGES",
    "OpenMPRuntimeCandidate",
    "OpenMPRuntimeStatus",
    "OptionalModelRuntime",
    "discover_openmp_runtimes",
    "f1_model_runtime_doctor",
    "inspect_optional_model_runtime",
    "preload_openmp_runtime",
]
