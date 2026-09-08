"""Independent synthetic checks of runner closure and advancement boundaries.

Suggested commit: test(football): enforce immutable past-market stage boundaries.
"""
from copy import deepcopy

import numpy as np
import pytest

from . import run


def closed_predecessor(tmp_path, monkeypatch, decision=True):
    """A small closed predecessor, with every path/hash checked by the runner."""
    monkeypatch.setattr(run, "ROOT", tmp_path)
    stage = "selection_delay1"
    run.save(tmp_path / "design_lock.json", {})
    run.save(tmp_path / "data_lock.json", {})
    run.save(tmp_path / "fit.json", {"coefficient": 1.0})
    run.save(tmp_path / "issued.json", [])
    run.save(tmp_path / "fit_index.json", [run.file_record(tmp_path / "fit.json")])
    run.save(tmp_path / "readout.json", {})
    run.save(tmp_path / f"{stage}_issuance_lock.json", {
        "labels_attached": False, "rows": 0,
        "data_lock_sha256": run.sha(tmp_path / "data_lock.json"),
        "forecasts": run.file_record(tmp_path / "issued.json"),
        "fit_index": run.file_record(tmp_path / "fit_index.json"),
        "elo_readout": run.file_record(tmp_path / "readout.json"),
    })
    run.save(tmp_path / f"{stage}.json", {
        "summary": {"passes_all_gates": True, "selected_candidate": "past_market"},
        "design_lock_sha256": run.sha(tmp_path / "design_lock.json"),
        "data_lock_sha256": run.sha(tmp_path / "data_lock.json"),
        "issuance_lock_sha256": run.sha(tmp_path / f"{stage}_issuance_lock.json"),
    })
    run.save(tmp_path / f"{stage}_decision_lock.json", {
        "passes_all_gates": decision, "selected_candidate": "past_market",
        "design_lock_sha256": run.sha(tmp_path / "design_lock.json"),
        "result_sha256": run.sha(tmp_path / f"{stage}.json"),
    })
    run.save(tmp_path / f"{stage}_verification.json", {
        "status": "passed", "result_sha256": run.sha(tmp_path / f"{stage}.json"),
        "design_lock_sha256": run.sha(tmp_path / "design_lock.json"),
    })
    return stage


def test_valid_fully_bound_predecessor_is_accepted(tmp_path, monkeypatch):
    stage = closed_predecessor(tmp_path, monkeypatch)
    assert run.predecessor(tmp_path, stage) == "past_market"


def test_failed_predecessor_cannot_advance_even_with_surviving_pass_lock(tmp_path, monkeypatch):
    stage = closed_predecessor(tmp_path, monkeypatch)
    run.save(tmp_path / f"{stage}_failure.json", {"advancement_allowed": False})
    with pytest.raises(ValueError, match="[Ff]ail"):
        run.predecessor(tmp_path, stage)


@pytest.mark.parametrize("truthy_nonboolean", [1, "false", [True]])
def test_predecessor_decision_requires_literal_true(tmp_path, monkeypatch, truthy_nonboolean):
    stage = closed_predecessor(tmp_path, monkeypatch, truthy_nonboolean)
    with pytest.raises(ValueError, match="[Pp]ass|[Dd]ecision|[Bb]ool"):
        run.predecessor(tmp_path, stage)


@pytest.mark.parametrize("changed", ["issued.json", "fit_index.json", "readout.json", "fit.json"])
def test_predecessor_rechecks_every_forecast_and_coefficient_binding(tmp_path, monkeypatch, changed):
    stage = closed_predecessor(tmp_path, monkeypatch)
    (tmp_path / changed).write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="changed"):
        run.predecessor(tmp_path, stage)


def test_complete_forecasts_and_coefficients_close_before_external_labels(tmp_path, monkeypatch):
    spec = run.read(run.SPEC)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    references = [{
        "match_id": match_id, "league": "E0", "home": "A", "away": "B",
        "forecast_cutoff_utc": cutoff,
        "probabilities": {name: [0.375, 0.25, 0.375] for name in spec["selection"]["references"]
                          if name not in run.CONTROLS},
    } for match_id, cutoff in [("later", "2022-08-03T00:00:00+00:00"),
                              ("earlier", "2022-08-02T00:00:00+00:00")]]
    original = deepcopy(references)
    run.save(tmp_path / "references_selection.json", references)
    run.save(tmp_path / "data_lock.json", {})
    archive = object()
    trace = []
    monkeypatch.setattr(run.data, "load_archive", lambda *args, **kwargs: archive)
    monkeypatch.setattr(run.data, "calibration_rows", lambda *args: ([0.0], [0], {}))
    def readout_fit(x, y):
        trace.append("readout_fit")
        return {"kind": "synthetic_readout"}
    monkeypatch.setattr(run.model, "fit_ordered_logit", readout_fit)
    monkeypatch.setattr(run.data, "training_rows", lambda *args: ([{"match_id": "past", "weight": 1.0}], {}))
    monkeypatch.setattr(run.data, "support_for", lambda batch, training: [row["match_id"] == "earlier" for row in batch])
    def strength_fit(training, kind):
        trace.append(kind)
        assert training == [{"match_id": "past", "weight": 1.0}]
        return {"kind": kind}
    monkeypatch.setattr(run.model, "fit_strength", strength_fit)
    monkeypatch.setattr(run.model, "predict_strength", lambda model, rows: np.tile([0.6, 0.25, 0.15], (len(rows), 1)))
    class Cursor:
        def __init__(self, received_archive, delay):
            assert received_archive is archive and delay == 1
        def query(self, batch, cutoff):
            return {"x": [0.0] * len(batch)}
    monkeypatch.setattr(run.data, "EloCursor", Cursor)
    monkeypatch.setattr(run.model, "predict_ordered_logit", lambda model, x: np.tile([0.5, 0.25, 0.25], (len(x), 1)))
    def attach(issued, received_archive, *, forecast_closure_path):
        assert received_archive is archive
        assert forecast_closure_path.exists()
        closure = run.verify_phase_files(tmp_path, "selection_delay1")
        assert closure["optimizer_fits"] == 5
        assert closure["rows"] == 2 and closure["labels_attached"] is False
        assert run.read(tmp_path / closure["forecasts"]["path"]) == issued
        assert [row["match_id"] for row in issued] == ["later", "earlier"]
        assert all("label" not in row and "outcome" not in row for row in issued)
        assert trace == ["readout_fit", "past_market", "outcome_control", "past_market", "outcome_control"]
        for name in [*spec["candidates"], *run.CONTROLS]:
            assert issued[0]["probabilities"][name] == original[0]["probabilities"][run.INCUMBENT]
        for saved, old in zip(issued, original):
            for name, vector in old["probabilities"].items():
                assert saved["probabilities"][name] == vector
        trace.append("external_labels")
        return [{**deepcopy(row), "label": 0} for row in issued]
    monkeypatch.setattr(run.data, "attach_labels", attach)
    labeled = run.fit_and_issue(tmp_path, "selection", 1)
    assert all(row["label"] == 0 for row in labeled)
    assert trace[-1] == "external_labels"
    assert run.read(tmp_path / "references_selection.json") == original
