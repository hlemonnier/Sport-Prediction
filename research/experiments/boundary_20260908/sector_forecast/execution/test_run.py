import numpy as np
import pytest

from research.experiments.boundary_20260908.sector_forecast.execution import run


def record(event=202301, identity="202301:1:20:S1"):
    return {"event_key": event, "driver": "1", "ledger_id": 0, "issuance_id": identity,
            "status": "issued", "sector": 1, "checkpoint_ms": 1000,
            "features": {name: None for name in run.features.FEATURES}, "points": {"reference": 90.}}


def test_flatten_uses_global_issuance_identity_and_fixed_string_event_key():
    frame = run.frame([record(), record(202302, "202302:1:20:S1")])
    assert frame["event_key"].tolist() == ["202301", "202302"]
    assert frame["ledger_id"].is_unique
    assert "y_true" not in frame and "target_id" not in frame
    assert frame[list(run.features.FEATURES)].isna().all().all()


def test_unsupported_rows_stay_in_ledger_but_do_not_create_predictions():
    a, b = record(), record(); b["status"] = "insufficient_history"
    assert len(run.frame([a, b])) == 1
    assert len([a, b]) == 2 and b["status"] == "insufficient_history"


def test_artifact_json_missingness_is_explicit_and_infinity_is_rejected(tmp_path):
    path = tmp_path / "rows.jsonl"
    run.save_rows(path, [{"missing": np.nan, "actual": 2.}])
    assert run.load_rows(path) == [{"missing": None, "actual": 2.}]
    with pytest.raises(ValueError, match="Infinity"):
        run.safe_json({"bad": np.inf})
    with pytest.raises(FileExistsError): run.save_rows(path, [])


def test_completed_artifacts_are_not_overwritten(tmp_path):
    path = tmp_path / "selection.json"; run.save(path, {"passed": False})
    before = path.read_bytes()
    with pytest.raises(FileExistsError): run.save(path, {"passed": True})
    assert path.read_bytes() == before


@pytest.mark.parametrize("status,terminal", [("Finished", True), ("Finalised", True), ("Ends", True), ("Started", False)])
def test_label_pass_uses_validated_terminal_classification(monkeypatch, tmp_path, status, terminal):
    metadata = {"event_key": 202301, "diagnostics": {"terminal": terminal, "terminal_status": status}}
    monkeypatch.setattr(run, "manifest", lambda: {"sessions": [{"event_key": 202301, "original_laps_path": "unused.csv"}]})
    monkeypatch.setattr(run, "locked_records", lambda out, years: [(metadata, [record()])])
    seen = []
    def attach(records, path, terminal_flag):
        seen.append(terminal_flag)
        return records
    monkeypatch.setattr(run.data, "attach_targets", attach)
    assert len(run.labels(tmp_path, [2023])) == 1
    assert seen == [terminal]
