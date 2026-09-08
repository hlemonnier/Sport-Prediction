"""Bind both immutable xG acquisitions and the optional7/14day proxy semantics."""
import csv
import json
from pathlib import Path

from research.experiments.boundary_20260908.football_xg_data import acquire as a, availability


def main():
    paths = [a.DATA / "joined_matches.csv", a.DATA / "extended_history/joined_matches.csv"]
    frames = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            frames.extend(csv.DictReader(handle))
    assert len(frames) == 10260 and len({r["canonical_match_id"] for r in frames}) == len(frames)
    accepted = [r for r in frames if r["accepted_exact_join"] == "True"]
    rejected = [r for r in frames if r["accepted_exact_join"] == "False"]
    assert len(accepted) == 10249 and len(rejected) == 11
    for row in accepted:
        assert availability.proxy_available_at(row, 14) > availability.proxy_available_at(row, 7)
    assert all(availability.proxy_available_at(row) is None for row in rejected)
    evidence_paths = [a.OUT / "join_quality.json", a.OUT / "verification.json",
        a.OUT / "extended_history/join_quality.json", a.OUT / "extended_history/verification.json",
        a.HERE / "timing_review.json", Path(availability.__file__), Path(__file__),
        a.HERE / "test_quality.py", a.HERE / "extended_history/test_extended.py"]
    files = {str(p.relative_to(a.ROOT)): a.sha(p) for p in [*paths, *evidence_paths]}
    fiorentina = next(r for r in frames if r["canonical_match_id"] == "I1:2024:Fiorentina:Inter")
    output = a.OUT / "availability_contract.json"
    if output.exists():
        raise FileExistsError("immutable combined data contract already exists")
    a.write(output, {"schema_version": "understat_exact_join_with_unverified_availability_v1",
        "artifact_manifest": files, "joined_files": [str(p.relative_to(a.ROOT)) for p in paths],
        "columns": list(frames[0]), "rows": len(frames), "accepted_exact_rows": len(accepted),
        "excluded_match_ids": sorted(r["canonical_match_id"] for r in rejected),
        "canonical_id": "{league}:{season_start_year}:{home}:{away}",
        "exact_join_key": ["league", "season_start_year", "canonical_date", "home", "away"],
        "xg_fields": {"home_xg": "nonnegative expected goals float", "away_xg": "nonnegative expected goals float"},
        "accepted_boolean": "True boolean or exact CSV text True only; numeric1 and text False must not be truthy acceptance",
        "availability_is_assumption": True, "original_publication_times_known": False, "revision_history_known": False,
        "proxy_formula": "validated completion day -> next local midnight -> convert to UTC -> add delay_days*24hours",
        "league_timezones": availability.ZONES, "primary_delay_days": 7, "sensitivity_delay_days": 14,
        "minimum_information_rule": "Only accepted previously completed matches whose proxy availability is at or before forecast cutoff. Never same-match xG. Keep the existing target population and use an explicitly frozen fallback for unavailable xG.",
        "resumption_marker": {"canonical_match_id": fiorentina["canonical_match_id"],
            "completion_day": fiorentina["canonical_date"], "original_interruption_day": "2024-12-01",
            "seven_day_proxy_utc": availability.proxy_available_at(fiorentina, 7).isoformat(),
            "fourteen_day_proxy_utc": availability.proxy_available_at(fiorentina, 14).isoformat(),
            "source": "research/experiments/boundary_20260908/football_xg_data/timing_review.json"},
        "tests_passed": 25, "same_raw_match_xg_and_team_history_mirrors_verified": 20520,
        "fitted_models": 0, "predictive_performance_scores_computed": False})
    print(a.sha(output), len(frames), len(accepted))


if __name__ == "__main__":
    main()
