"""Synthetic temporal, lineage and closure regressions; no historical assets."""
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
import json
from pathlib import Path

import pytest

from . import data as d


class Poison(dict):
    def __getitem__(self, key):
        raise AssertionError("Future goal payload inspected")


def raw(mid, day, league="E0", season=2023, home="A", away="B"):
    return d.RawFixture(mid, league, season, date.fromisoformat(day), home, away,
                        "synthetic.csv", "a" * 64, 2, "b" * 64)


def make_inputs(tmp_path, league="E0"):
    first = date(2023, 1, 1)
    archive = tuple(raw(f"{league}:old:{i}", (first + timedelta(days=i)).isoformat(), league)
                    for i in range(300))
    cutoff_day = first + timedelta(days=301)
    current = raw(f"{league}:new:0", cutoff_day.isoformat(), league)
    future = raw(f"{league}:new:1", (cutoff_day + timedelta(days=1)).isoformat(), league)
    records = [d.match_record(r, [1, 0]) for r in archive]
    fixture = {**{k: v for k, v in current.metadata().items() if k in d.METADATA},
               "fit_cutoff_utc": d.legacy_clock(d.midnight(cutoff_day, league)), "fit_id": "synthetic-fit",
               "probabilities": {d.REFERENCE: [.5, .3, .2], d.CANDIDATE: [.51, .29, .2]}}
    compact = [[r.match_id, r.day.isoformat(), 1, 0] for r in archive]
    ids = [r.match_id for r in archive]
    descriptor = {"block_id": league + ":synthetic-fit", "league": league, "fit_id": "synthetic-fit",
        "fit_cutoff_utc": fixture["fit_cutoff_utc"], "fixture_metadata": [fixture], "saved_full_model": {"rho": 0.},
        "expected_partitions": d.chronological_populations(records).metadata(), "saved_full_lineage": {
            "fit_match_ids": ids, "fit_ids_sha256": d.digest(ids), "fit_content_sha256": d.digest(compact),
            "cutoff_utc": fixture["fit_cutoff_utc"], "latest_result_available_at": records[-1].result_available_at.isoformat(),
            "goal_state_sha256": d.digest({"rho": 0.})}}
    payloads = {r.match_id: {"FTHG": "1", "FTAG": "0", "FTR": "H"} for r in archive}
    payloads.update({current.match_id: Poison(), future.match_id: Poison()})
    inputs = d.Inputs({"timing": {"history_days": 1095}}, tmp_path, archive + (current, future),
                      [fixture], [descriptor], [], {}, {}, payloads)
    return inputs, descriptor


@pytest.mark.parametrize("league", d.LEAGUES)
def test_admitted_matches_reproduce_exact_legacy_records_and_partitions(tmp_path, league):
    inputs, descriptor = make_inputs(tmp_path, league)
    output = d.prepare_block(inputs, descriptor)
    assert len(output["history"]) == 300
    assert output["expected_partitions"] == descriptor["expected_partitions"]
    assert output["saved_full_lineage"]["match_records_sha256"] == d.records_sha256(output["history"])
    assert output["history"][0].league == ("epl" if league == "E0" else league)
    assert output["fixtures"][0].date == d.utc(descriptor["fit_cutoff_utc"]).replace(tzinfo=None)
    assert output["expected_reference_probabilities"] == [[.5, .3, .2]]
    assert output["expected_candidate_probabilities"] == [[.51, .29, .2]]
    assert "match_records_sha256" not in descriptor["saved_full_lineage"]
    assert len(output["expected_partitions"]["fit"]["match_ids"]) < 300


def test_future_and_current_payload_poison_leave_previous_materialization_unchanged(tmp_path):
    inputs, descriptor = make_inputs(tmp_path)
    before = d.prepare_block(inputs, descriptor)
    extra = raw("E0:2030:future", "2030-01-01")
    inputs.archive += (extra,)
    inputs._payloads[extra.match_id] = Poison()
    assert d.prepare_block(inputs, descriptor) == before


def test_historical_numeric_reads_are_deferred_until_lazy_prepare(tmp_path):
    inputs, descriptor = make_inputs(tmp_path)
    inputs._payloads[inputs.archive[0].match_id] = Poison()
    assert len(d.metadata_rows(inputs)) == 302
    assert len(d.admitted_metadata(inputs, descriptor)) == 300
    with pytest.raises(AssertionError, match="Future goal"):
        d.prepare_block(inputs, descriptor)


@pytest.mark.parametrize("league", d.LEAGUES)
@pytest.mark.parametrize("day,hours", [("2024-03-31", 23), ("2024-10-27", 25)])
def test_availability_uses_calendar_midnight_across_dst(league, day, hours):
    start = date.fromisoformat(day)
    assert (d.midnight(start + timedelta(days=1), league) - d.midnight(start, league)).total_seconds() == 3600 * hours


@pytest.mark.parametrize("league", d.LEAGUES)
def test_inclusive_local_window_equal_availability_and_exclusive_current_day(tmp_path, league):
    day = date(2024, 4, 1)
    days = [day - timedelta(days=1096), day - timedelta(days=1095), day - timedelta(days=1), day]
    archive = tuple(raw(str(i), when.isoformat(), league) for i, when in enumerate(days))
    inputs = d.Inputs({"timing": {"history_days": 1095}}, tmp_path, archive, [], [], [], {}, {}, {})
    desc = {"league": league, "fit_cutoff_utc": d.legacy_clock(d.midnight(day, league))}
    selected = d.admitted_metadata(inputs, desc)
    assert [r.match_id for r in selected] == ["1", "2"]
    assert d.utc(selected[-1].metadata()["result_available_at"]) == d.utc(desc["fit_cutoff_utc"])


def test_non_midnight_clock_fails_without_goal_read(tmp_path):
    inputs, descriptor = make_inputs(tmp_path)
    descriptor["fit_cutoff_utc"] = "2023-10-29T12:00:00"
    with pytest.raises(ValueError, match="local midnight"):
        d.prepare_block(inputs, descriptor)


@pytest.mark.parametrize("tamper", ["order", "missing_id", "id_hash", "goal_content", "partition", "latest_clock", "goal_state", "lineage_clock"])
def test_old_lineage_tampering_fails(tmp_path, tamper):
    inputs, descriptor = make_inputs(tmp_path)
    line = descriptor["saved_full_lineage"]
    if tamper == "order": line["fit_match_ids"].reverse()
    if tamper == "missing_id": line["fit_match_ids"].pop()
    if tamper == "id_hash": line["fit_ids_sha256"] = "0" * 64
    if tamper == "goal_content": inputs._payloads[inputs.archive[0].match_id]["FTHG"] = "2"
    if tamper == "partition": descriptor["expected_partitions"]["fit"]["data_sha256"] = "0" * 64
    if tamper == "latest_clock": line["latest_result_available_at"] = "2022-01-01T00:00:00"
    if tamper == "goal_state": descriptor["saved_full_model"]["rho"] = .1
    if tamper == "lineage_clock": line["cutoff_utc"] = "2022-01-01T00:00:00"
    with pytest.raises(ValueError): d.prepare_block(inputs, descriptor)


def test_wrong_membership_rejected_before_any_numeric_read(tmp_path):
    inputs, descriptor = make_inputs(tmp_path)
    inputs._payloads[inputs.archive[0].match_id] = Poison()
    descriptor["saved_full_lineage"]["fit_match_ids"].reverse()
    with pytest.raises(ValueError, match="history identities"):
        d.prepare_block(inputs, descriptor)


@pytest.mark.parametrize("value", [True, 1.2, "1.5", "nan", "-1", None])
def test_goal_values_do_not_silently_truncate_or_coerce(value):
    with pytest.raises(ValueError): d._goals({"FTHG": value, "FTAG": "0", "FTR": "H"})


def test_result_label_consistency():
    with pytest.raises(ValueError): d._goals({"FTHG": "1", "FTAG": "0", "FTR": "D"})


def test_metadata_csv_reader_preserves_poison_goal_strings_and_source_hash(tmp_path):
    p = tmp_path / "E0_2023_2024.csv"
    p.write_text("\ufeffDiv,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,01/08/2023, A ,B,poison,poison,poison\nE0,02/08/2023,B,A,poison,poison,poison\n")
    binding = {"path": p.name, "sha256": d.sha(p)}
    rows, payloads = d.read_csv_metadata(p, binding, expected_rows=2, expected_teams=2)
    assert [r.match_id for r in rows] == ["E0:2023:A:B", "E0:2023:B:A"]
    assert rows[0].source_row_hash == d.digest(payloads[rows[0].match_id])
    assert payloads[rows[0].match_id]["FTHG"] == "poison"
    assert not hasattr(rows[0], "payload")
    assert "FTHG" not in rows[0].metadata()


def test_resumed_fixture_keeps_its_provider_completion_day(tmp_path):
    p = tmp_path / "transfer_I1_2024_2025.csv"
    p.write_text("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nI1,06/02/2025,Fiorentina,Inter,x,x,x\nI1,09/02/2025,Inter,Fiorentina,x,x,x\n")
    rows, _ = d.read_csv_metadata(p, {"path": p.name, "sha256": d.sha(p)}, expected_rows=2, expected_teams=2)
    assert rows[0].match_id == "I1:2024:Fiorentina:Inter"
    assert rows[0].day.isoformat() == "2025-02-06"
    assert rows[0].metadata()["forecast_cutoff_utc"] == "2025-02-05T23:00:00"


def test_fixture_whitelist_never_reads_legacy_targets():
    class Legacy(dict):
        def __getitem__(self, key):
            if key in ("goals", "label", "log_shot_means"): raise AssertionError("target accessed")
            return super().__getitem__(key)
    row = Legacy({k: k for k in d.METADATA})
    row["probabilities"] = {d.REFERENCE: [.5, .3, .2], d.CANDIDATE: [.51, .29, .2]}
    assert set(d.fixture_row(row)) == {*d.METADATA, "probabilities"}


def closed_fixture(tmp_path):
    inputs, _ = make_inputs(tmp_path)
    issued = deepcopy(inputs.fixture_rows)
    path = tmp_path / "issued.json"
    path.write_text(json.dumps(issued))
    closure = {"forecasts": {"path": path.name, "sha256": d.sha(path), "bytes": path.stat().st_size},
               "rows": len(issued), "labels_attached": False}
    cp = tmp_path / "closure.json"
    cp.write_text(json.dumps(closure))
    return inputs, issued, path, closure, cp


def test_labels_require_closed_exact_population_and_leave_original_rows_unchanged(tmp_path):
    inputs, issued, _, _, cp = closed_fixture(tmp_path)
    inputs._payloads[issued[0]["match_id"]] = {"FTHG": "0", "FTAG": "1", "FTR": "A"}
    labeled = d.attach_labels(issued, inputs, forecast_closure_path=cp)
    assert labeled[0]["goals"] == [0, 1] and labeled[0]["label"] == 2
    assert "goals" not in issued[0] and "label" not in issued[0]


@pytest.mark.parametrize("tamper", ["flag_int", "flag_true", "count_bool", "count", "sha", "bytes", "id", "metadata", "label", "missing"])
def test_bad_closure_or_population_fails_before_any_target_read(tmp_path, tamper):
    inputs, issued, path, closure, cp = closed_fixture(tmp_path)
    if tamper == "flag_int": closure["labels_attached"] = 0
    if tamper == "flag_true": closure["labels_attached"] = True
    if tamper == "count_bool": closure["rows"] = True
    if tamper == "count": closure["rows"] = 2
    if tamper == "sha": closure["forecasts"]["sha256"] = "0" * 64
    if tamper == "bytes": closure["forecasts"]["bytes"] += 1
    if tamper == "id": issued[0]["match_id"] = "different"
    if tamper == "metadata": issued[0]["day"] = "2025-01-01"
    if tamper == "label": issued[0]["label"] = 0
    if tamper == "missing": issued.clear()
    if tamper in ("id", "metadata", "label", "missing"):
        path.write_text(json.dumps(issued))
        closure["forecasts"].update(sha256=d.sha(path), bytes=path.stat().st_size)
    cp.write_text(json.dumps(closure))
    with pytest.raises(ValueError): d.attach_labels(issued, inputs, forecast_closure_path=cp)


def test_source_binding_checks_both_bytes_and_hash(tmp_path):
    p = tmp_path / "raw.csv"; p.write_bytes(b"hello")
    binding = {"path": p.name, "sha256": d.sha(p), "bytes": 5}
    assert d._bound(binding, tmp_path, {}) == p
    p.write_bytes(b"HELLO")
    with pytest.raises(ValueError): d._bound(binding, tmp_path, {})


@pytest.fixture(scope="module")
def closed_input_graph(tmp_path_factory):
    """27 synthetic full provider seasons; six retained target fixture rows.

    Every goal payload is invalid. The metadata-only loader must still succeed.
    Synthetic saved partitions come from separate known constant-goal records.
    """
    root = tmp_path_factory.mktemp("deployment-data-graph")
    def save(name, value):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
        return {"path": name, "sha256": d.sha(p), "bytes": p.stat().st_size}
    source = save("parent.py", {"synthetic": True})
    sources = {source["path"]: source["sha256"]}
    spec = {"proposal": save("README.md", {}), "inputs": {
        "canonical_global_caller_sources": [source], "runtime_gap": source,
        "runtime_scoreline_diagnostic": source, "cached_csv_total": 27,
        "inherited_source_map": {"entries": 1, "canonical_sha256": d.digest(sources)}},
        "population": {"rows": 6, "countries": list(d.LEAGUES), "season_start_years": [2024, 2025],
                       "rows_each_country_season": 1, "fit_blocks": 6, "fit_blocks_each_country": 2},
        "timing": {"history_days": 1095}}
    teams = [f"T{i:02}" for i in range(20)]
    archive, files, file_maps = [], [], {league: {} for league in d.LEAGUES}
    for league in d.LEAGUES:
        for year in range(2017, 2026):
            name = ("" if league == "E0" else "transfer_") + f"{league}_{year}_{year+1}.csv"
            path = root / name
            rows = [(h, a) for h in teams for a in teams if h != a]
            path.write_text("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n" + "".join(
                f"{league},01/08/{year},{h},{a},POISON,POISON,POISON\n" for h, a in rows))
            entry = {"name": name, "sha256": d.sha(path), "bytes": path.stat().st_size}
            files.append(entry); file_maps[league][name] = entry["sha256"]
            rr, _ = d.read_csv_metadata(path, {"path": name, **entry})
            archive.extend(rr)
    spec["inputs"]["acquisition_manifests"] = [save("acquisition.json", {"files": files})]
    all_fixtures, features, inventory = [], {}, []
    for league in d.LEAGUES:
        batches, fits = [], []
        feature_path = f"features_{league}.json"
        for index, year in enumerate([2024, 2025]):
            current = next(r for r in archive if r.league == league and r.season == year)
            cutoff = d.midnight(current.day, league)
            prior = sorted([r for r in archive if r.league == league and current.day-timedelta(days=1095) <= r.day < current.day],
                           key=lambda r: (r.day, r.match_id))
            records = [d.match_record(r, [1, 0]) for r in prior]
            ids = [r.match_id for r in prior]
            fit_id = f"{current.day}:{d.digest(ids)[:12]}"
            row = {k: v for k, v in current.metadata().items() if k in d.METADATA}
            row.update(fit_id=fit_id, fit_cutoff_utc=d.legacy_clock(cutoff), label="POISON", goals="POISON",
                       probabilities={d.REFERENCE: [.5, .3, .2], d.CANDIDATE: [.51, .29, .2]})
            batches.append(row); all_fixtures.append(row)
            base = {"fit_id": fit_id, "cutoff_utc": d.legacy_clock(cutoff), "fit_match_ids": ids,
                "fit_ids_sha256": d.digest(ids), "fit_content_sha256": d.digest([[r.match_id, r.day.isoformat(), 1, 0] for r in prior]),
                "latest_result_available_at": records[-1].result_available_at.isoformat(),
                "dixon_coles": {d.CANDIDATE: {"rho": 0.}}}
            fit = {"base": base, "production_default": {"populations": d.chronological_populations(records).metadata()}}
            fits.append(fit)
            inventory.append(d.inventory_record(league, feature_path, index, fit, [current.match_id]))
        value = {"rows": batches, "fits": fits, "source_files": sources, "source_sha256": d.digest(sources),
                 "spec_sha256": "a" * 64, "input_hashes": file_maps[league]}
        features[league] = save(feature_path, value)
    all_fixtures.sort(key=lambda r: (r["forecast_cutoff_utc"], r["match_id"]))
    inventory.sort(key=lambda x: (x["fit_cutoff_utc"], x["league"], x["fit_id"]))
    spec["population"]["ordered_original_match_ids_sha256"] = d.digest([r["match_id"] for r in all_fixtures])
    spec["population"]["block_inventory_binding"] = {"complete_derived_inventory_sha256": d.digest(inventory)}
    spec["inputs"]["feature_artifacts"] = features
    spec["inputs"]["cached_csv_maps"] = [{"artifact": features[l]["path"], "entries": 9,
        "canonical_sha256": d.digest(file_maps[l])} for l in d.LEAGUES]
    evaluation = {"predictions": all_fixtures, "source_files": sources, "source_sha256": d.digest(sources),
                  "spec_sha256": "a" * 64, "feature_artifacts_sha256": {l: item["sha256"] for l, item in features.items()}}
    spec["inputs"]["old_evaluation"] = save("evaluation.json", evaluation)
    spec["inputs"]["old_design_lock"] = save("design.json", {"spec_sha256": "a" * 64})
    spec["inputs"]["old_verification"] = save("verification.json", {"status": "passed", "source_sha256": d.digest(sources),
        "evaluation_sha256": spec["inputs"]["old_evaluation"]["sha256"]})
    return root, spec


def test_complete_graph_load_is_metadata_only_and_preserves_references(closed_input_graph, monkeypatch):
    root, spec = closed_input_graph
    def forbid(*args): raise AssertionError("Numeric goals cannot be opened during preparation")
    monkeypatch.setattr(d, "_goals", forbid)
    inputs = d.load_inputs(spec, root)
    assert len(inputs.archive) == 10260 and len(inputs.fixture_rows) == 6 and len(inputs.blocks) == 6
    assert d.digest(inputs.block_inventory) == spec["population"]["block_inventory_binding"]["complete_derived_inventory_sha256"]
    assert all("goals" not in r and "label" not in r for r in inputs.fixture_rows)
    assert all(row["probabilities"][d.CANDIDATE] == [.51, .29, .2] for row in inputs.fixture_rows)
    assert all(b["saved_full_lineage"]["goal_state_sha256"] == d.digest(b["saved_full_model"]) for b in inputs.blocks)


@pytest.mark.parametrize("tamper", ["file_hash", "population_order", "block_inventory", "source_map", "missing_feature"])
def test_complete_graph_rejects_frozen_contract_mismatch(closed_input_graph, tamper):
    root, original = closed_input_graph
    spec = deepcopy(original)
    if tamper == "file_hash": spec["inputs"]["feature_artifacts"]["E0"]["sha256"] = "0" * 64
    if tamper == "population_order": spec["population"]["ordered_original_match_ids_sha256"] = "0" * 64
    if tamper == "block_inventory": spec["population"]["block_inventory_binding"]["complete_derived_inventory_sha256"] = "0" * 64
    if tamper == "source_map": spec["inputs"]["inherited_source_map"]["canonical_sha256"] = "0" * 64
    if tamper == "missing_feature": spec["inputs"]["feature_artifacts"].pop("I1")
    with pytest.raises(ValueError): d.load_inputs(spec, root)
