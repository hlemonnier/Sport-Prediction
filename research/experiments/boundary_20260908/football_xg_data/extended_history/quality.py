"""Same immutable strict-join policy for older warmup matches; no model evaluation."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from research.experiments.boundary_20260908.football_xg_data import acquire as base, join_quality as q
from research.experiments.boundary_20260908.football_xg_data import verify as first_verify
from research.experiments.boundary_20260908.football_xg_data.extended_history import acquire as a

ALIASES_PATH = a.HERE / "team_aliases.json"
ALIASES = json.loads(ALIASES_PATH.read_text())


def parse(payload, league, year):
    records, excluded = q.parse_matches(payload, league, year)
    for row in records:
        row["home"] = ALIASES[league].get(row["provider_home"], row["home"])
        row["away"] = ALIASES[league].get(row["provider_away"], row["away"])
    return records, excluded


def load():
    lock = json.loads((a.OUT / "acquisition_lock.json").read_text())
    assert lock["canonical_inputs"] == a.canonical_inputs()
    assert base.sha(a.__file__) == lock["source_sha256"]
    for file, expected in lock["first_twelve_immutable"].items():
        assert base.sha(base.ROOT / file) == expected
    manifest = json.loads((a.OUT / "acquisition_manifest.json").read_text())
    assert len(manifest["pages"]) == 15 and not manifest["failures"]
    seasons = []
    for item in manifest["pages"]:
        assert item["season_start_year"] in a.YEARS
        for p, h in (("path", "sha256"), ("receipt_path", "receipt_sha256")):
            assert base.sha(base.ROOT / item[p]) == item[h]
        payload = json.loads((base.ROOT / item["path"]).read_text())
        code, year = item["canonical_league"], item["season_start_year"]
        records, rejected = parse(payload, code, year)
        cp = base.ROOT / "data/football/performance_20260907" / (("" if code == "E0" else "transfer_")+f"{code}_{year}_{year+1}.csv")
        canonical = q.canonical_rows(cp, code, year)
        joined, unmatched = q.join_rows(records, canonical)
        teams = {row[side] for row in canonical for side in ("home", "away")}
        assert len(teams) == 20
        assert all(sum(row[side] == team for row in canonical for side in ("home", "away")) == 38 for team in teams)
        seasons.append((item, joined, rejected, unmatched, payload))
    return lock, manifest, seasons


def dependencies():
    paths = [Path(__file__), ALIASES_PATH, Path(a.__file__), Path(q.__file__), q.ALIASES_PATH,
             Path(first_verify.__file__), a.OUT / "acquisition_manifest.json", a.OUT / "acquisition_lock.json"]
    return {str(p.relative_to(base.ROOT)): base.sha(p) for p in paths}


def create_quality():
    report_path = a.OUT / "join_quality.json"
    if report_path.exists():
        raise FileExistsError("immutable extended quality report already exists")
    lock, manifest, inputs = load()
    source = dependencies()
    all_rows, rejected, unmatched, summaries, outputs = [], [], [], [], {}
    for item, rows, excluded, absent, _ in inputs:
        path = a.DATA / f'{item["league"]}_{item["season_start_year"]}_joined_matches.csv'
        if path.exists():
            raise FileExistsError(path)
        q.write_csv(path, rows); outputs[str(path.relative_to(base.ROOT))] = base.sha(path)
        counts = dict(Counter(r["join_status"] for r in rows))
        summaries.append({"league": item["canonical_league"], "season_start_year": item["season_start_year"],
            "listed": item["matches_listed"], "parsed_completed": len(rows), "canonical_rows": 380,
            "exact_accepted": sum(r["accepted_exact_join"] for r in rows), "status_counts": counts,
            "parse_excluded": len(excluded), "canonical_unmatched": len(absent)})
        all_rows.extend(rows); rejected.extend(excluded); unmatched.extend(absent)
        print(item["league"], item["season_start_year"], counts, "parse_excluded", len(excluded), flush=True)
    path = a.DATA / "joined_matches.csv"
    if path.exists():
        raise FileExistsError(path)
    q.write_csv(path, all_rows); outputs[str(path.relative_to(base.ROOT))] = base.sha(path)
    total = {"seasons": 15, "canonical_matches": 5700, "provider_listed": sum(s["listed"] for s in summaries),
        "parsed_completed": len(all_rows), "exact_accepted": sum(r["accepted_exact_join"] for r in all_rows),
        "status_counts": dict(Counter(r["join_status"] for r in all_rows)),
        "parse_excluded": len(rejected), "canonical_unmatched": len(unmatched)}
    assert source == dependencies()
    base.write(report_path, {"status": "quality_audited_no_model_fitted", "source_manifest": source,
        "canonical_input_manifest": lock["canonical_inputs"], "first_twelve_immutable": lock["first_twelve_immutable"],
        "parsed_output_manifest": outputs, "seasons": summaries, "totals": total,
        "join_discrepancies": [r for r in all_rows if not r["accepted_exact_join"]],
        "parse_exclusions": rejected, "canonical_unmatched": unmatched,
        "date_policy": "Unchanged first12 policy: exact mapped home/away, calendar date and goals required. No implicit date repair.",
        "availability": "Unverified original publication/revisions. Later modelling must use past matches only, seven full days after next local midnight following validated completion, fourteen-day sensitivity; never original interrupted kickoff.",
        "same_schema_as_first_twelve": True, "fitted_models": 0, "performance_scores_computed": False})
    print(json.dumps(total, indent=2))


def verify():
    path = a.OUT / "join_quality.json"
    quality = json.loads(path.read_text())
    for name in ("source_manifest", "canonical_input_manifest", "first_twelve_immutable", "parsed_output_manifest"):
        for file, expected in quality[name].items():
            assert base.sha(base.ROOT / file) == expected, file
    _, _, inputs = load()
    all_rows, mirrors = [], 0
    for item, rows, rejected, unmatched, payload in inputs:
        assert not rejected and not unmatched
        first_verify.compare_csv(a.DATA / f'{item["league"]}_{item["season_start_year"]}_joined_matches.csv', rows)
        for row in rows:
            for side, other in (("home", "away"), ("away", "home")):
                history = payload["teams"][row[f"provider_{side}_id"]]["history"]
                matches = [h for h in history if h["date"] == row["provider_datetime_naive"] and h["h_a"] == side[0]]
                assert len(matches) == 1
                h = matches[0]
                assert abs(float(h["xG"])-row[f"{side}_xg"]) < 1e-10
                assert abs(float(h["xGA"])-row[f"{other}_xg"]) < 1e-10
                assert int(h["scored"]) == row[f"{side}_goals"] and int(h["missed"]) == row[f"{other}_goals"]
                mirrors += 1
        all_rows.extend(rows)
    first_verify.compare_csv(a.DATA / "joined_matches.csv", all_rows)
    assert len(all_rows) == 5700 and len({r["understat_match_id"] for r in all_rows}) == 5700
    assert [r for r in all_rows if not r["accepted_exact_join"]] == quality["join_discrepancies"]
    assert sum(r["accepted_exact_join"] for r in all_rows) == quality["totals"]["exact_accepted"]
    with (base.DATA / "joined_matches.csv").open(newline="") as handle:
        first = list(csv.DictReader(handle))
    assert list(first[0]) == list(all_rows[0])
    assert not {r["understat_match_id"] for r in first} & {r["understat_match_id"] for r in all_rows}
    result = {"status": "passed", "source_sha256": base.sha(__file__), "quality_sha256": base.sha(path),
        "raw_and_receipt_files_verified": 30, "canonical_files_verified": 17, "derived_csv_files_verified": 16,
        "reconstructed_join_rows": len(all_rows), "provider_team_history_xg_goal_mirrors": mirrors,
        "exact_accepted": quality["totals"]["exact_accepted"], "excluded": len(quality["join_discrepancies"]),
        "schema_matches_frozen_first12": True, "first12_frozen_artifacts_unchanged": True,
        "no_provider_id_overlap_with_first12": True, "fitted_models": 0, "performance_scores_computed": False}
    base.write(a.OUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("quality", "verify"))
    args = parser.parse_args()
    {"quality": create_quality, "verify": verify}[args.phase]()
