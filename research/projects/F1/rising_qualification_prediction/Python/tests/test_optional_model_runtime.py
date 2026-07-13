from __future__ import annotations

from pathlib import Path

from packages.f1.orchestration import model_runtime


def test_openmp_discovery_prioritizes_active_python_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "lib" / "libomp.dylib"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"test-runtime-placeholder")
    monkeypatch.setattr(model_runtime.sys, "prefix", str(tmp_path))
    monkeypatch.delenv(model_runtime.OPENMP_PATH_ENV, raising=False)

    candidates = model_runtime.discover_openmp_runtimes()

    matched = [candidate for candidate in candidates if candidate.path == str(runtime)]
    assert len(matched) == 1
    assert matched[0].source == "sys_prefix"
    assert matched[0].exists is True


def test_openmp_preload_reports_exact_selected_path(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "libomp.dylib"
    runtime.write_bytes(b"test-runtime-placeholder")
    loaded: list[str] = []
    monkeypatch.setattr(model_runtime, "_OPENMP_HANDLES", {})
    monkeypatch.setattr(
        model_runtime,
        "_load_openmp_library",
        lambda path: loaded.append(path) or object(),
    )

    status = model_runtime.preload_openmp_runtime(runtime)

    assert status.status == "available"
    assert status.available is True
    assert status.preloaded is True
    assert status.selected_path == str(runtime)
    assert status.selected_source == "explicit_argument"
    assert loaded == [str(runtime)]
    assert status.to_payload()["selected_path"] == str(runtime)


def test_optional_runtime_payload_has_explicit_availability_state() -> None:
    status = model_runtime.inspect_optional_model_runtime("xgboost")
    payload = status.to_payload()

    assert payload["status"] in {"available", "unavailable"}
    assert payload["available"] is (payload["status"] == "available")
    if payload["available"]:
        assert payload["version"]
        assert payload["importable"] is True
    else:
        assert payload["issue"] in {
            "package_not_installed",
            "openmp_runtime_not_found",
            "openmp_runtime_load_failed",
            "openmp_runtime_missing",
            "package_import_failed",
        }


def test_runtime_doctor_reports_openmp_and_each_optional_backend() -> None:
    payload = model_runtime.f1_model_runtime_doctor()

    assert payload["openmp_runtime"]["status"] in {"available", "unavailable"}
    by_package = {
        item["package"]: item for item in payload["optional_learning_to_rank"]
    }
    assert set(by_package) == {"xgboost", "lightgbm"}
    assert all(item["status"] in {"available", "unavailable"} for item in by_package.values())
