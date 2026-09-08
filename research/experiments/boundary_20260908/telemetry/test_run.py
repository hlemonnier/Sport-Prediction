"""Synthetic staged execution: immutable closures, ordering and target isolation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from . import data, models, run
from .test_data import laps


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Two tiny synthetic races; original parsers/encoders, no real model fits."""
    monkeypatch.setattr(run, "ROOT", tmp_path)
    out = tmp_path/"execution";out.mkdir()
    inventory, streams, matched = [], [], []
    for key in (202201, 202301):
        path = tmp_path/f"{key}_laps.csv";laps(last=6).to_csv(path, index=False)
        raw, _ = data.read_laps(path)
        issued, paired = data._original(raw, key);paired["year"] = key//100
        matched.append(paired)
        inventory.append({"event_key": key, "path": str(path), "sha256": data.sha(path),
            "raw_rows": len(raw), "issuances": len(issued), "matched_rows": len(paired)})
        streams.append({"event_key": key, "status": "unavailable"})
    cache = tmp_path/"discovery_data.pkl";pd.concat(matched, ignore_index=True).to_pickle(cache)
    acq = tmp_path/"acquisition.json";run.save(acq, {"streams": streams})
    spec = {"original_frontier_bindings": [{"path": str(cache), "sha256": data.sha(cache)}],
        "discovery": {"event_keys": [202201, 202301], "expected_original_issuances": sum(r["issuances"] for r in inventory),
            "matched_rows": sum(r["matched_rows"] for r in inventory), "train_matched_rows": len(matched[0]),
            "selection_matched_rows": len(matched[1])}}
    lock = {"acquisition_manifest": run.record(acq), "original_input_manifest": inventory}
    run.save(out/"design_lock.json", lock)
    monkeypatch.setattr(run, "verify_design", lambda output: (lock, spec))
    return SimpleNamespace(out=out, spec=spec, lock=lock, cache=cache, matched=matched)


def fake_models(monkeypatch, state, *, mutate=False):
    def fit(frame):
        assert (state.out/"labels_2022/label_closure.json").exists()
        assert not (state.out/"labels_2023").exists()
        assert frame.year.eq(2022).all() and frame.outcome_status.eq("matched").all()
        run.assert_matched_parity(frame, state.matched[0])
        return {"fit_summary": {"fits": 3, "rows_per_fit": len(frame)}, "synthetic": True}
    def predict(bundle, frame, *, latency_seconds):
        assert bundle["synthetic"] is True
        assert not (state.out/"labels_2023").exists()
        assert not set(run.TARGET_COLUMNS)&set(frame)
        if mutate: frame.loc[frame.index[0], "forecast_naive_seconds"] += 1
        return {name: frame.forecast_naive_seconds.to_numpy().copy() for name in models.MODEL_NAMES}
    def evaluate(primary, pp, sensitivity, sp, *, expected_matched, expected_issuances):
        assert (state.out/"selection_issuance_lock.json").exists()
        run.assert_original_parity(primary, expected_issuances)
        run.assert_original_parity(sensitivity, expected_issuances)
        run.assert_matched_parity(primary.loc[primary.outcome_status.eq("matched")], expected_matched)
        assert primary.outcome_status.eq("unmatched").sum() == 4
        for name in models.MODEL_NAMES:
            np.testing.assert_array_equal(pp[name], sp[name])
            np.testing.assert_array_equal(pp[name], primary.forecast_naive_seconds)
        return {"advances_to_later_evaluation": False, "synthetic": True}
    monkeypatch.setattr(run.models, "fit_models", fit)
    monkeypatch.setattr(run.models, "predict_models", predict)
    monkeypatch.setattr(run.evaluate, "evaluate_selection", evaluate)


def test_prepare_closes_both_years_before_validation_without_labels(synthetic, monkeypatch, capsys):
    original = run.validate_year
    def validate(path, output):
        for year in (2022, 2023): assert (synthetic.out/f"features_{year}/feature_closure.json").exists()
        assert not list(synthetic.out.glob("labels_*"))
        return original(path, output)
    monkeypatch.setattr(run, "validate_year", validate)
    monkeypatch.setattr(run.data, "attach_labels", lambda *a, **kw: pytest.fail("Prepare attached external labels"))
    run.prepare(synthetic.out)
    locked = run.read(synthetic.out/"data_lock.json")
    assert locked["external_labels_attached"] is False and locked["model_fits"] == 0
    assert locked["original_issuances_per_lag"] == synthetic.spec["discovery"]["expected_original_issuances"]
    for year in (2022, 2023):
        primary = run.load_year(locked, year, 2);zero = run.load_year(locked, year, 0)
        assert primary.issuance_id.tolist() == zero.issuance_id.tolist()
        assert not primary.telemetry_supported.any()
        assert not set(run.TARGET_COLUMNS)&set(primary)
        run.assert_original_parity(primary, run.reference_year(locked, year))
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [v["stage"] for v in progress] == ["features_event", "features_event", "validated_event", "validated_event", "prepared"]
    with pytest.raises(FileExistsError): run.prepare(synthetic.out)


def test_select_closes_all_forecasts_before_external_selection_labels_or_cache_read(synthetic, monkeypatch):
    run.prepare(synthetic.out);fake_models(monkeypatch, synthetic)
    original_labels, original_pickle = run.label_year, pd.read_pickle
    reads = []
    def check_closed():
        closure = run.read(synthetic.out/"selection_issuance_lock.json")
        assert closure["selection_labels_attached"] is False
        assert set(closure["forecasts"]) == {"0", "2"}
        for value in closure["forecasts"].values(): run.check_record(value)
    def labels(output, locked, year):
        if year == 2023: check_closed()
        return original_labels(output, locked, year)
    def read_pickle(path, *args, **kwargs):
        check_closed();reads.append(str(path));return original_pickle(path, *args, **kwargs)
    monkeypatch.setattr(run, "label_year", labels);monkeypatch.setattr(pd, "read_pickle", read_pickle)
    run.select(synthetic.out)
    assert reads and set(reads) == {str(synthetic.cache)}
    assert run.read(synthetic.out/"original_population_parity.json")["ordered_values_exact"] is True
    assert run.read(synthetic.out/"selection_lock.json")["promotion"] is False
    assert run.read(synthetic.out/"selection.json")["summary"]["synthetic"] is True
    before = data.sha(synthetic.out/"selection.json")
    with pytest.raises(FileExistsError): run.select(synthetic.out)
    assert data.sha(synthetic.out/"selection.json") == before


def test_prediction_mutation_stops_before_selection_labels_and_preserves_fit(synthetic, monkeypatch):
    run.prepare(synthetic.out);fake_models(monkeypatch, synthetic, mutate=True)
    with pytest.raises(ValueError, match="mutated target-free"):
        run.select(synthetic.out)
    assert (synthetic.out/"fit_lock.json").exists()
    assert not (synthetic.out/"selection_issuance_lock.json").exists()
    assert not (synthetic.out/"labels_2023").exists()


def test_saved_feature_corruption_prevents_model_fitting(synthetic, monkeypatch):
    run.prepare(synthetic.out)
    path = synthetic.out/"features_2022/202201_lag2_issued.jsonl"
    with path.open("a") as stream: stream.write("{}\n")
    monkeypatch.setattr(run.models, "fit_models", lambda *a: pytest.fail("Fit occurred with corrupt features"))
    with pytest.raises(ValueError, match="hash|Hash|changed"):
        run.select(synthetic.out)
    assert not (synthetic.out/"models.pkl").exists()


def test_download_integrity_failure_retains_forecast_feature_history(synthetic, monkeypatch):
    run.prepare(synthetic.out)
    closure = synthetic.out/"features_2022/feature_closure.json"
    original = data.verify_feature_closure(closure)
    changed = deepcopy(original);changed["events"][0]["source"].update(telemetry_status="downloaded", telemetry_path="synthetic-corrupt")
    monkeypatch.setattr(data, "verify_feature_closure", lambda _: changed)
    def invalid(path, *, stats):
        stats.update(complete=False, stream_input_valid=False)
        yield {"synthetic": True}
    monkeypatch.setattr(data.packets, "iter_packets", invalid)
    target = synthetic.out/"invalid_input.json"
    before = data.sha(closure)
    with pytest.raises(ValueError, match="validation incomplete"): run.validate_year(closure, target)
    assert run.read(target)["status"] == "invalid_downloaded_input"
    assert data.sha(closure) == before


@pytest.mark.parametrize("approval", [False, "true", "false", 1, None])
def test_freeze_requires_literal_true_exact_source_review(tmp_path, monkeypatch, approval):
    spec = tmp_path/"specification.json";run.save(spec, {})
    monkeypatch.setattr(run, "HERE", tmp_path)
    monkeypatch.setattr(run, "validate_spec", lambda _: None)
    monkeypatch.setattr(run, "sources", lambda: {"source.py": "abc"})
    review = tmp_path/"review.json";run.save(review, {"approved_for_execution_lock": approval, "source_files": {"source.py": "abc"}})
    monkeypatch.setattr(run, "input_bindings", lambda *args: pytest.fail("Unapproved design reached inputs"))
    with pytest.raises(ValueError, match="exactly these source bytes"): run.freeze(tmp_path/"out", review, tmp_path/"acquisition.json")


def test_freeze_rejects_changed_reviewed_source(tmp_path, monkeypatch):
    run.save(tmp_path/"specification.json", {})
    monkeypatch.setattr(run, "HERE", tmp_path);monkeypatch.setattr(run, "validate_spec", lambda _: None)
    monkeypatch.setattr(run, "sources", lambda: {"source.py": "new"})
    review = tmp_path/"review.json";run.save(review, {"approved_for_execution_lock": True, "source_files": {"source.py": "old"}})
    with pytest.raises(ValueError, match="exactly these source bytes"): run.freeze(tmp_path/"out", review, tmp_path/"acquisition.json")


def test_no_historical_prepare_without_design_lock(tmp_path):
    with pytest.raises(FileNotFoundError): run.prepare(tmp_path)
    assert not (tmp_path/"prepare_attempt.json").exists()


def test_all_issuance_parity_rejects_unmatched_drop_and_row_reordering(synthetic):
    run.prepare(synthetic.out);lock = run.read(synthetic.out/"data_lock.json")
    frame = run.load_year(lock, 2023, 2);expected = run.reference_year(lock, 2023)
    with pytest.raises(ValueError, match="population"): run.assert_original_parity(frame.iloc[:-1], expected)
    with pytest.raises(ValueError, match="field/order"): run.assert_original_parity(frame.iloc[::-1], expected)
    frame.loc[0, data.BASE_FEATURES[0]] += .001
    with pytest.raises(ValueError, match="field/order"): run.assert_original_parity(frame, expected)


def test_prediction_row_hash_detects_unchanged_id_feature_poison(synthetic):
    run.prepare(synthetic.out);lock = run.read(synthetic.out/"data_lock.json")
    frame = run.load_year(lock, 2023, 2)
    points = {name: np.full(len(frame), 90.) for name in models.MODEL_NAMES}
    path = synthetic.out/"predictions.jsonl";data._write_rows(path, run.forecast_rows(frame, points, 2))
    binding = run.record(path);run.load_forecasts(frame, binding, 2)
    frame.loc[0, "forecast_naive_seconds"] += .01
    with pytest.raises(ValueError, match="feature-row binding"): run.load_forecasts(frame, binding, 2)


def test_model_progress_wraps_three_calls_without_changing_fit_and_restores_on_failure(monkeypatch, capsys):
    def original(frame, config, features): return (len(frame), config, features)
    monkeypatch.setattr(models.original, "fit_model", original)
    with run.fit_progress():
        for name in models.MODEL_NAMES:
            assert models.original.fit_model([1, 2], {"name": name}, ["x"]) == (2, {"name": name}, ["x"])
    assert models.original.fit_model is original
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["model"] for r in reports[::2]] == list(models.MODEL_NAMES)
    assert [r["stage"] for r in reports] == ["model_fit_started", "model_fit_completed"]*3
    with pytest.raises(RuntimeError), run.fit_progress(): raise RuntimeError("synthetic")
    assert models.original.fit_model is original


def test_cli_has_no_transfer_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(run.sys, "argv", ["run", "transfer", "--out", str(tmp_path)])
    with pytest.raises(SystemExit) as failure: run.main()
    assert failure.value.code == 2


def test_cli_preserves_first_failure_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(run.sys, "argv", ["run", "prepare", "--out", str(tmp_path)])
    with pytest.raises(FileNotFoundError): run.main()
    failures = list(tmp_path.glob("prepare_failure_*.json"))
    assert len(failures) == 1 and run.read(failures[0])["existing_outputs_preserved"] is True


@pytest.fixture
def input_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    def file(name, body):
        path = tmp_path/name
        if isinstance(body, dict): run.save(path, body)
        else: path.parent.mkdir(parents=True, exist_ok=True);path.write_text(body)
        return {"path": name, "sha256": data.sha(path)}
    def sized(binding): return {"sha256": binding["sha256"], "bytes": (tmp_path/binding["path"]).stat().st_size}
    pilot_receipt = file("pilot/old_receipt.json", {"historical": True})
    pilot = file("pilot/independent_verification.json", {"bindings": {pilot_receipt["path"]: sized(pilot_receipt)}})
    parent = file("parent.json", {"synthetic": True});measurement = file("measurement.md", "fixed")
    laps_file = file("laps.csv", "metadata only")
    session = {"event_key": 202201, "original_laps_path": laps_file["path"], "original_laps_sha256": laps_file["sha256"], "session_path": "2022/Race/"}
    contract = file("contract.json", {"sessions": [session], "parent_manifest": parent, "bindings": [pilot]})
    original = file("old/selection.json", {"features": list(data.BASE_FEATURES), "input_manifest": [
        {"event_key": 202201, **laps_file, "issuances": 4, "matched_rows": 3}]})
    cache = file("old/discovery_data.pkl", "never unpickle")
    source = file("source.py", "pass\n")
    review = file("acquisition_review.json", {"approved": True})
    lock = file("acquisition_lock.json", {"acquisition_review_sha256": review["sha256"]})
    receipt = file("new_receipt.json", {"status": 200})
    acquisition = file("acquisition.json", {"status": "all_discovery_attempts_recorded", "contract_sha256": contract["sha256"],
        "sessions": [session], "acquisition_lock_sha256": lock["sha256"], "source_files": {source["path"]: source["sha256"]},
        "output_bindings": {receipt["path"]: sized(receipt)},
        "streams": [{"event_key": 202201, "session_path": session["session_path"], "status": "unavailable"}]})
    spec = {"original_frontier_bindings": [original, cache], "acquisition_contract": contract, "measurement_contract": measurement,
        "discovery": {"event_keys": [202201], "expected_original_issuances": 4, "matched_rows": 3}}
    return SimpleNamespace(root=tmp_path, spec=spec, acquisition=tmp_path/acquisition["path"], receipt=receipt, pilot_receipt=pilot_receipt, review=review)


def test_acquisition_binding_includes_receipts_review_and_reused_pilot_dependencies(input_graph):
    bindings, inventory = run.input_bindings(input_graph.spec, input_graph.acquisition)
    assert len(inventory) == 1
    for value in (input_graph.receipt, input_graph.pilot_receipt): assert bindings[value["path"]] == value["sha256"]
    assert bindings[str((input_graph.root/input_graph.review["path"]).resolve())] == input_graph.review["sha256"]


@pytest.mark.parametrize("attribute", ["receipt", "pilot_receipt", "review"])
def test_acquisition_receipt_drift_prevents_freeze(input_graph, attribute):
    item = getattr(input_graph, attribute)
    with (input_graph.root/item["path"]).open("a") as stream: stream.write(" ")
    with pytest.raises(ValueError, match="byte count|hash mismatch"):
        run.input_bindings(input_graph.spec, input_graph.acquisition)
