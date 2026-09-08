"""Frozen race-ranking experiment with forecasts persisted before current labels."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from research.experiments.boundary_20260908.race_practice.features import NAMES, session_measurements, aggregate
from research.experiments.boundary_20260908.race_practice.models import fit_predict

PRIOR_PATH = ROOT / "research/experiments/performance_20260907/race/run.py"
_spec = importlib.util.spec_from_file_location("race_practice_prior", PRIOR_PATH)
prior = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = prior
_spec.loader.exec_module(prior)
PRIOR_RESULT = ROOT / "artifacts/research/performance_20260907/race/v3/results.json"
OUT = ROOT / "artifacts/research/boundary_20260908/race_practice"
REFERENCES = ["baseline", "huber_movement_1", "boosted_absolute_movement_1"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_manifest():
    paths = sorted(HERE.glob("*.py")) + [HERE / "specification.json", PRIOR_PATH,
                                          PRIOR_PATH.with_name("specification.json")]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def input_events(reference, max_year):
    excluded = {d["event"] for d in reference["inventory"]["excluded"]}
    inventory = reference["input_manifest"]
    paths = sorted(ROOT / p for p in inventory if p.endswith("/weekend_metadata.json"))
    for path in paths:
        year = int(path.parent.parent.name)
        if year > max_year:
            continue
        number = int(path.parent.name.split("_")[1]); key = f"{year}:{number:02d}"
        if key in excluded:
            continue
        if sha(path) != inventory[str(path.relative_to(ROOT))]:
            raise ValueError(f"inherited metadata drift: {path}")
        meta = json.loads(path.read_text())
        qualifying = [s for s in meta["sessions"] if s["session_type"] == "qualifying"]
        race = [s for s in meta["sessions"] if s["session_type"] == "race"]
        if len(qualifying) != 1 or len(race) != 1:
            raise ValueError("ambiguous Grand Prix sessions")
        qp = path.parent / Path(qualifying[0]["results_path"]).name
        rp = path.parent / Path(race[0]["results_path"]).name
        for p in (qp, rp):
            if sha(p) != inventory[str(p.relative_to(ROOT))]:
                raise ValueError(f"inherited classification drift: {p}")
        q = pd.read_csv(qp).sort_values("Abbreviation").reset_index(drop=True)
        ids = q.Abbreviation.astype(str).tolist()
        values = pd.to_numeric(q.Position, errors="raise").to_numpy(float)
        if len(set(ids)) != len(ids) or not np.isfinite(values).all():
            raise ValueError("invalid inherited qualifying roster")
        event = prior.Event(key, year, number, path.parent.name.split("_", 2)[2],
                            "sprint" if any(s["session_type"] in {"sprint", "sprint_qualifying", "sprint_shootout"}
                                            for s in meta["sessions"]) else "standard",
                            ids, q.TeamId.astype(str).tolist(), prior.ranks(values, ids),
                            np.zeros(len(ids)), np.zeros(len(ids), dtype=bool))
        yield event, meta, path, qualifying[0], rp


def practice(event, metadata, folder, qualifying, inputs):
    sessions, audits = [], []
    for s in sorted(metadata["sessions"], key=lambda s: int(s["session_order"])):
        if s["session_type"] != "free_practice" or int(s["session_order"]) >= int(qualifying["session_order"]):
            continue
        if s.get("completed", True) is not True or not s.get("results_rows", 0):
            continue
        p = folder / Path(s["laps_path"]).name
        if not p.exists():
            audits.append({"session_order": int(s["session_order"]), "status": "unavailable"})
            continue
        inputs[str(p.relative_to(ROOT))] = sha(p)
        measured, audit = session_measurements(pd.read_csv(p))
        sessions.append(measured)
        audits.append({"session_order": int(s["session_order"]), "path": str(p.relative_to(ROOT)), **audit})
    x, detail = aggregate(event.ids, event.teams, event.qualifying, sessions)
    return x, {**detail, "sessions": audits}


def read_target(event, path):
    r = pd.read_csv(path)
    if r.Abbreviation.isna().any() or r.Abbreviation.duplicated().any() or set(r.Abbreviation) != set(event.ids):
        raise ValueError("inherited target roster no longer matches")
    r = r.set_index("Abbreviation").loc[event.ids]
    y = pd.to_numeric(r.Position, errors="raise").to_numpy(float)
    if sorted(y.tolist()) != list(range(1, len(event.ids) + 1)):
        raise ValueError("inherited target is not a permutation")
    status = r.Status.fillna("").astype(str).str.lower().str.strip()
    event.target = y
    event.classified_finish = (status.eq("finished") | status.str.match(r"^\+\d+\s+laps?$")).to_numpy()


def history_update(event, index, history):
    n = len(event.ids)
    for i in range(n):
        history.append({"index": index, "year": event.year, "driver": event.ids[i], "team": event.teams[i],
                        "circuit": event.circuit, "q": (event.qualifying[i] - 1) / (n - 1),
                        "target": (event.target[i] - 1) / (n - 1), "finished": bool(event.classified_finish[i])})


def paired(records, reference_records):
    assert [r["event"] for r in records] == [r["event"] for r in reference_records]
    delta = np.array([a["mae"] - b["mae"] for a, b in zip(records, reference_records)])
    year = np.array([r["year"] for r in records]); n = len(delta)
    rng = np.random.default_rng(20260908)
    event_boot = np.zeros(20000); block_boot = np.zeros(20000)
    for y in sorted(set(year)):
        d = delta[year == y]; k = len(d)
        event_boot += d[rng.integers(k, size=(20000, k))].sum(axis=1) / n
        starts = rng.integers(k, size=(20000, int(np.ceil(k / 3))))
        index = ((starts[..., None] + np.arange(3)) % k).reshape(20000, -1)[:, :k]
        block_boot += d[index].sum(axis=1) / n
    a = float(np.mean([r["mae"] for r in records])); b = float(np.mean([r["mae"] for r in reference_records]))
    loo = (delta.sum() - delta) / (n - 1)
    return {"events": n, "candidate_mae": a, "reference_mae": b, "relative_gain": 1 - a / b,
            "delta": float(delta.mean()), "event_ci95": np.quantile(event_boot, [.025, .975]).tolist(),
            "block3_ci95": np.quantile(block_boot, [.025, .975]).tolist(),
            "loo_min": float(loo.min()), "loo_max": float(loo.max()), "event_wins": int((delta < 0).sum())}


def execute(max_year, configs, reference, output, stage):
    history, events, panels, raw_panels = [], [], [], []
    scores = {k: [] for k in [*configs, *[k + "_without_practice" for k in configs], *REFERENCES]}
    input_hashes, inventory, fit_details, points, feature_ledger = {}, [], [], [], []
    prior_predictions = ROOT / reference["predictions_path"]
    assert sha(prior_predictions) == reference["predictions_sha256"]
    old = pd.read_csv(prior_predictions, usecols=["event", "driver_id", "candidate", "prediction"])
    for index, (event, meta, metadata_path, qualifying, target_path) in enumerate(input_events(reference, max_year)):
        x0 = prior.features_for(event, history, index)
        extra, support = practice(event, meta, metadata_path.parent, qualifying, input_hashes)
        x = np.column_stack([x0, extra])
        feature_ledger.append({"event": event.key, "ids": event.ids, "features": x.tolist()})
        inventory.append({"event": event.key, "year": event.year, **support})
        current = {"baseline": event.qualifying.astype(int)}
        scoring = event.year == 2023 if stage == "selection" else event.year >= 2024
        if scoring:
            for name, config in configs.items():
                for features, train, suffix in [(x, panels, ""), (x0, raw_panels, "_without_practice")]:
                    key = name + suffix
                    current[key], detail = fit_predict(train, events, features, event.ids, config)
                    fit_details.append({"event": event.key, "candidate": key, "train_events": [e.key for e in events], **detail})
            for name in REFERENCES[1:]:
                take = old.loc[old.event.eq(event.key) & old.candidate.eq(name)].set_index("driver_id")
                if set(take.index) != set(event.ids):
                    raise ValueError(f"inherited comparator roster mismatch: {event.key} {name}")
                current[name] = take.loc[event.ids].prediction.to_numpy(int)
            record = {"event": event.key, "ids": event.ids, "feature_sha256": hashlib.sha256(canonical(x.tolist())).hexdigest(),
                      "predictions": {k: v.tolist() for k, v in current.items()}}
            record["forecast_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
            with (output / f"{stage}_forecasts_before_targets.jsonl").open("ab") as f:
                f.write(canonical(record) + b"\n")
        # This is the first read of current race labels. Features and forecasts already exist.
        read_target(event, target_path)
        input_hashes[str(target_path.relative_to(ROOT))] = sha(target_path)
        input_hashes[str(metadata_path.relative_to(ROOT))] = sha(metadata_path)
        qp = metadata_path.parent / Path(qualifying["results_path"]).name
        input_hashes[str(qp.relative_to(ROOT))] = sha(qp)
        if scoring:
            for name, value in current.items():
                if sorted(value.tolist()) != list(range(1, len(event.ids) + 1)):
                    raise ValueError("prediction is not a permutation")
                scores[name].append(prior.event_score(event, value))
                points.extend({"event": event.key, "year": event.year, "driver": d, "candidate": name,
                               "prediction": int(value[i]), "target": int(event.target[i])} for i, d in enumerate(event.ids))
            print(stage, event.key, "supported", support["supported_drivers"], "of", len(event.ids), flush=True)
        panels.append(x); raw_panels.append(x0); events.append(event)
        history_update(event, index, history)
    save(output / f"{stage}_feature_ledger.json", feature_ledger)
    save(output / f"{stage}_forecasts.json", points)
    save(output / f"{stage}_fits.json", fit_details)
    return {"scores": scores, "input_manifest": input_hashes, "support": inventory,
            "included_by_year": {str(y): sum(e.year == y for e in events) for y in sorted({e.year for e in events})},
            "output_manifest": {str(p.relative_to(ROOT)): sha(p) for p in output.glob(stage + "_*")}}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["selection", "transfer"], default="selection")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((HERE / "specification.json").read_text())
    reference = json.loads(PRIOR_RESULT.read_text())
    lock = OUT / (args.stage + "_design_lock.json")
    if lock.exists():
        raise FileExistsError("Frozen attempt already exists; preserve it")
    configs = spec["candidates"]
    if args.stage == "transfer":
        frozen = json.loads((OUT / "selection_frozen.json").read_text())
        if sha(OUT / "selection.json") != frozen["selection_sha256"]:
            raise ValueError("Frozen selection result changed")
        if sha(OUT / "selection_design_lock.json") != frozen["selection_design_lock_sha256"]:
            raise ValueError("Frozen selection design changed")
        if sha(PRIOR_RESULT) != frozen["prior_result_sha256"]:
            raise ValueError("Frozen prior result changed")
        if source_manifest() != frozen["source_manifest"]:
            raise ValueError("Tested selection source changed before transfer")
        selection = json.loads((OUT / "selection.json").read_text())
        for p, expected in {**selection["input_manifest"], **selection["output_manifest"]}.items():
            if sha(ROOT / p) != expected:
                raise ValueError(f"Frozen selection input/output changed: {p}")
        if not selection["advancement_passed"]:
            raise ValueError("Selection failed; transfer is not authorized by this protocol")
        configs = {selection["selected"]: configs[selection["selected"]]}
    source = source_manifest()
    save(lock, {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "source_manifest": source,
                "prior_result_sha256": sha(PRIOR_RESULT), "stage": args.stage, "specification": spec})
    result = execute(2023 if args.stage == "selection" else 2026, configs, reference, OUT, args.stage)
    assert result["included_by_year"] == ({"2022": 21, "2023": 20} if args.stage == "selection" else reference["inventory"]["included_by_year"])
    if args.stage == "selection":
        chosen = min(configs, key=lambda k: (np.mean([r["mae"] for r in result["scores"][k]]), k))
        comparisons = {r: paired(result["scores"][chosen], result["scores"][r])
                       for r in [*REFERENCES, chosen + "_without_practice"]}
        result.update({"selected": chosen, "comparisons": comparisons,
                       "advancement_passed": all(v["relative_gain"] >= .01 and v["loo_max"] < 0 for v in comparisons.values()),
                       "all_candidate_comparisons": {k: {r: paired(result["scores"][k], result["scores"][r])
                        for r in [*REFERENCES, k + "_without_practice"]} for k in configs}})
    else:
        chosen = next(iter(configs)); comparisons = {}
        for label, years in [("historical", [2024, 2025]), ("2024", [2024]), ("2025", [2025]), ("2026", [2026])]:
            take = lambda name: [r for r in result["scores"][name] if r["year"] in years]
            comparisons[label] = {r: paired(take(chosen), take(r)) for r in [*REFERENCES, chosen + "_without_practice"]}
        result.update({"selected": chosen, "comparisons": comparisons,
                       "substantial_gate_passed": all(v["relative_gain"] >= (.02 if r.endswith("without_practice") else .1)
                             and v["block3_ci95"][1] < 0 for r, v in comparisons["historical"].items())
                             and all(v["relative_gain"] > 0 for y in ["2024", "2025", "2026"] for v in comparisons[y].values())})
    assert all(sha(ROOT / p) == h for p, h in source.items())
    result.update({"source_manifest": source, "specification_sha256": sha(HERE / "specification.json"),
                   "prior_result_sha256": sha(PRIOR_RESULT), "production_changed": False, "promotion": False})
    save(OUT / (args.stage + ".json"), result)
    if args.stage == "selection":
        save(OUT / "selection_frozen.json", {"selection_sha256": sha(OUT / "selection.json"),
            "selection_design_lock_sha256": sha(lock), "source_manifest": source,
            "prior_result_sha256": sha(PRIOR_RESULT), "selected": result["selected"],
            "advancement_passed": result["advancement_passed"]})
    print(json.dumps({k: v for k, v in result.items() if k in {"selected", "comparisons", "advancement_passed", "substantial_gate_passed"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
