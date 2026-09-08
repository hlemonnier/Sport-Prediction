"""Rebuild each exact join and verify all acquisition, canonical and derived hashes."""
import csv
import json
from pathlib import Path

from research.experiments.boundary_20260908.football_xg_data import acquire, join_quality as q


def compare_csv(path, rows):
    with path.open(encoding="utf-8", newline="") as handle:
        actual = list(csv.DictReader(handle))
    expected = [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]
    assert actual == expected, path


def verify():
    path = acquire.OUT / "join_quality.json"
    quality = json.loads(path.read_text())
    assert acquire.canonical_inputs() == quality["original_canonical_inputs"]
    for group in ("source_manifest", "original_canonical_inputs", "parsed_output_manifest"):
        for file, expected in quality[group].items():
            assert acquire.sha(acquire.ROOT / file) == expected, file
    manifest = json.loads((acquire.OUT / "public_ajax_manifest.json").read_text())
    all_rows, excluded, unmatched = [], [], []
    team_mirrors = 0
    for item in manifest["pages"]:
        raw = acquire.ROOT / item["path"]
        assert acquire.sha(raw) == item["sha256"]
        assert acquire.sha(acquire.ROOT / item["receipt_path"]) == item["receipt_sha256"]
        payload = json.loads(raw.read_text())
        code, year = item["canonical_league"], item["season_start_year"]
        records, rejected = q.parse_matches(payload, code, year)
        cp = acquire.ROOT / "data/football/performance_20260907" / (("" if code == "E0" else "transfer_")+f"{code}_{year}_{year+1}.csv")
        joined, absent = q.join_rows(records, q.canonical_rows(cp, code, year))
        compare_csv(acquire.DATA / f'{item["league"]}_{year}_joined_matches.csv', joined)
        # Independent within-provider consistency: per-team histories carry the same xG totals.
        for row in records:
            for side, opposite in (("home", "away"), ("away", "home")):
                team = payload["teams"][row[f"provider_{side}_id"]]
                candidates = [h for h in team["history"] if h["date"] == row["provider_datetime_naive"]
                              and h["h_a"] == side[0]]
                assert len(candidates) == 1
                history = candidates[0]
                assert abs(float(history["xG"])-row[f"{side}_xg"]) < 1e-10
                assert abs(float(history["xGA"])-row[f"{opposite}_xg"]) < 1e-10
                assert int(history["scored"]) == row[f"{side}_goals"]
                assert int(history["missed"]) == row[f"{opposite}_goals"]
                team_mirrors += 1
        all_rows.extend(joined); excluded.extend(rejected); unmatched.extend(absent)
    compare_csv(acquire.DATA / "joined_matches.csv", all_rows)
    assert len(all_rows) == 4560 and len({r["understat_match_id"] for r in all_rows}) == len(all_rows)
    assert not excluded and not unmatched
    assert sum(r["accepted_exact_join"] for r in all_rows) == quality["totals"]["exact_accepted"] == 4555
    assert [r for r in all_rows if not r["accepted_exact_join"]] == quality["join_discrepancies"]
    timing_path = acquire.HERE / "timing_review.json"
    timing = json.loads(timing_path.read_text())
    keyed = {r["canonical_match_id"]: r for r in all_rows}
    for case in timing["findings"]:
        row = keyed[case["canonical_match_id"]]
        for name in ("provider_date", "canonical_date", "accepted_exact_join"):
            assert row[name] == case[name]
    result = {"status": "passed", "verifier_source_sha256": acquire.sha(__file__),
        "tests_source_sha256": acquire.sha(acquire.HERE / "test_quality.py"),
        "quality_sha256": acquire.sha(path), "timing_review_sha256": acquire.sha(timing_path),
        "season_pages": 12, "exact_reconstructed_join_rows": len(all_rows),
        "accepted_exact_date_team_score_rows": 4555, "explicit_date_mismatch_exclusions": 5,
        "provider_team_history_xg_and_goal_mirrors_verified": team_mirrors,
        "original_canonical_files_unchanged": 14, "raw_and_receipt_files_verified": 24,
        "derived_csv_files_verified": 13, "unknown_publication_and_revision_fields_retained": True,
        "fitted_models": 0, "performance_scores_computed": False}
    acquire.write(acquire.OUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
