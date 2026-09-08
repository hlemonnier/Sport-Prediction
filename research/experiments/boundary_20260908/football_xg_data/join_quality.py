"""Exact bounded xG data-quality join; never fits or scores a prediction model."""
from collections import Counter
import csv
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path

from research.experiments.boundary_20260908.football_xg_data import acquire

ALIASES_PATH = acquire.HERE / "team_aliases.json"
ALIASES = json.loads(ALIASES_PATH.read_text())


def number(value, integer=False):
    if isinstance(value, bool) or value is None:
        raise ValueError("missing or boolean numeric field")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (integer and not result.is_integer()):
        raise ValueError("invalid nonnegative numeric field")
    return int(result) if integer else result


def parse_matches(payload, league, year):
    parsed, excluded = [], []
    if not isinstance(payload, dict) or not isinstance(payload.get("dates"), list):
        raise ValueError("missing public dates list")
    ids = [str(row.get("id", "")) for row in payload["dates"]]
    if "" in ids or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate provider match identities")
    for row in payload["dates"]:
        key = str(row["id"])
        try:
            if row.get("isResult") is not True:
                raise ValueError("isResult is not the boolean true")
            when = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
            if not date(year, 7, 1) <= when.date() <= date(year+1, 8, 31):
                raise ValueError("provider date outside season-start label")
            home, away = row["h"]["title"], row["a"]["title"]
            if not isinstance(home, str) or not isinstance(away, str) or not home or not away or home == away:
                raise ValueError("invalid provider teams")
            parsed.append({"league": league, "season_start_year": year, "understat_match_id": key,
                "provider_home_id": str(row["h"]["id"]), "provider_away_id": str(row["a"]["id"]),
                "provider_home": home, "provider_away": away,
                "home": ALIASES[league].get(home, home), "away": ALIASES[league].get(away, away),
                "provider_datetime_naive": row["datetime"], "provider_date": when.date().isoformat(),
                "home_goals": number(row["goals"]["h"], True), "away_goals": number(row["goals"]["a"], True),
                "home_xg": number(row["xG"]["h"]), "away_xg": number(row["xG"]["a"]),
                "provider_original_published_at": None, "provider_last_revised_at": None})
        except (KeyError, TypeError, ValueError) as exc:
            excluded.append({"understat_match_id": key, "league": league, "season_start_year": year,
                             "reason": str(exc), "raw_datetime": row.get("datetime"),
                             "raw_isResult": row.get("isResult")})
    return parsed, excluded


def canonical_rows(path, league, year):
    result = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not any(row.values()):
                continue
            assert row["Div"] == league
            day = datetime.strptime(row["Date"], "%d/%m/%Y" if len(row["Date"].split("/")[-1]) == 4 else "%d/%m/%y").date()
            result.append({"league": league, "season_start_year": year, "home": row["HomeTeam"].strip(),
                "away": row["AwayTeam"].strip(), "canonical_date": day.isoformat(),
                "canonical_home_goals": number(row["FTHG"], True), "canonical_away_goals": number(row["FTAG"], True),
                "match_id": f'{league}:{year}:{row["HomeTeam"].strip()}:{row["AwayTeam"].strip()}'})
    assert len(result) == 380 and len({r["match_id"] for r in result}) == 380
    return result


def join_rows(source, canonical):
    lookup = {(r["league"], r["season_start_year"], r["home"], r["away"]): r for r in canonical}
    if len(lookup) != len(canonical):
        raise ValueError("duplicate canonical season home-away fixture")
    source_keys = [(r["league"], r["season_start_year"], r["home"], r["away"]) for r in source]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("duplicate provider season home-away fixture after explicit alias map")
    result, matched = [], set()
    for row, key in zip(source, source_keys):
        base = lookup.get(key)
        out = {**row, "canonical_match_id": None, "canonical_date": None,
            "canonical_home_goals": None, "canonical_away_goals": None,
            "canonical_minus_provider_days": None, "accepted_exact_join": False}
        if base is None:
            out["join_status"] = "unmatched_team_or_fixture"
        else:
            matched.add(base["match_id"])
            out.update({"canonical_match_id": base["match_id"], **{k: base[k] for k in (
                "canonical_date", "canonical_home_goals", "canonical_away_goals")}})
            delta = (date.fromisoformat(base["canonical_date"])-date.fromisoformat(row["provider_date"])).days
            out["canonical_minus_provider_days"] = delta
            same_score = (row["home_goals"], row["away_goals"]) == (base["canonical_home_goals"], base["canonical_away_goals"])
            if not same_score:
                out["join_status"] = "score_disagreement_excluded"
            elif delta:
                out["join_status"] = "fixture_score_match_date_mismatch_excluded"
            else:
                out["join_status"] = "exact_date_teams_score"
                out["accepted_exact_join"] = True
        result.append(out)
    unmatched = [r for r in canonical if r["match_id"] not in matched]
    return result, unmatched


def write_csv(path, rows):
    if not rows:
        raise ValueError("empty parsed output")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    output = acquire.OUT / "join_quality.json"
    if output.exists():
        raise FileExistsError("immutable quality analysis already exists")
    manifest_path = acquire.OUT / "public_ajax_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    lock = json.loads((acquire.OUT / "acquisition_lock.json").read_text())
    assert acquire.canonical_inputs() == lock["canonical_inputs"]
    assert len(manifest["pages"]) == 12 and not manifest["failures"]
    source_files = {str(p.relative_to(acquire.ROOT)): acquire.sha(p)
                    for p in (Path(__file__), ALIASES_PATH, Path(acquire.__file__), manifest_path,
                              acquire.OUT / "acquisition_lock.json", acquire.OUT / "public_ajax_lock.json")}
    all_joined, all_excluded, all_unmatched, seasons, parsed_hashes = [], [], [], [], {}
    for item in manifest["pages"]:
        raw_path = acquire.ROOT / item["path"]
        assert acquire.sha(raw_path) == item["sha256"]
        assert acquire.sha(acquire.ROOT / item["receipt_path"]) == item["receipt_sha256"]
        league, year = item["canonical_league"], item["season_start_year"]
        parsed, excluded = parse_matches(json.loads(raw_path.read_text()), league, year)
        cp = acquire.ROOT / "data/football/performance_20260907" / (("" if league == "E0" else "transfer_")+f"{league}_{year}_{year+1}.csv")
        canonical = canonical_rows(cp, league, year)
        joined, unmatched = join_rows(parsed, canonical)
        path = acquire.DATA / f'{item["league"]}_{year}_joined_matches.csv'
        if path.exists():
            raise FileExistsError(path)
        write_csv(path, joined); parsed_hashes[str(path.relative_to(acquire.ROOT))] = acquire.sha(path)
        counts = dict(Counter(r["join_status"] for r in joined))
        seasons.append({"league": league, "understat_league": item["league"], "season_start_year": year,
            "listed_matches": item["matches_listed"], "parsed_completed_matches": len(parsed),
            "canonical_matches": len(canonical), "exact_accepted": sum(r["accepted_exact_join"] for r in joined),
            "join_status_counts": counts, "parse_excluded": len(excluded), "canonical_unmatched": len(unmatched),
            "provider_min_date": min(r["provider_date"] for r in parsed), "provider_max_date": max(r["provider_date"] for r in parsed)})
        all_joined.extend(joined); all_excluded.extend(excluded); all_unmatched.extend(unmatched)
        print(item["league"], year, counts, "parse_excluded", len(excluded), flush=True)
    rows_path = acquire.DATA / "joined_matches.csv"
    if rows_path.exists():
        raise FileExistsError(rows_path)
    write_csv(rows_path, all_joined); parsed_hashes[str(rows_path.relative_to(acquire.ROOT))] = acquire.sha(rows_path)
    discrepancies = [r for r in all_joined if not r["accepted_exact_join"]]
    accepted = sum(r["accepted_exact_join"] for r in all_joined)
    report = {"status": "quality_audited_no_model_fitted", "source_manifest": source_files,
        "original_canonical_inputs": lock["canonical_inputs"], "raw_manifest_sha256": acquire.sha(manifest_path),
        "parsed_output_manifest": parsed_hashes, "seasons": seasons,
        "totals": {"seasons": len(seasons), "canonical_matches": sum(s["canonical_matches"] for s in seasons),
            "provider_listed_matches": sum(s["listed_matches"] for s in seasons), "parsed_completed_matches": len(all_joined),
            "exact_accepted": accepted, "exact_fraction_of_canonical": accepted/sum(s["canonical_matches"] for s in seasons),
            "excluded_join_rows": len(discrepancies), "parse_excluded_rows": len(all_excluded),
            "canonical_unmatched_rows": len(all_unmatched), "status_counts": dict(Counter(r["join_status"] for r in all_joined)),
            "home_xg_min": min(r["home_xg"] for r in all_joined), "home_xg_max": max(r["home_xg"] for r in all_joined),
            "away_xg_min": min(r["away_xg"] for r in all_joined), "away_xg_max": max(r["away_xg"] for r in all_joined)},
        "join_discrepancies": discrepancies, "parse_exclusions": all_excluded, "canonical_unmatched": all_unmatched,
        "canonical_key": ["league", "season_start_year", "canonical_date", "home", "away"],
        "stable_fixture_id": "{league}:{season_start_year}:{home}:{away}; unique only within the tested double-round-robin season",
        "date_policy": "Primary acceptance requires exact source date, mapped home/away teams and final goals. Unique fixture+score date mismatches are explicit exclusions, not silently retimed.",
        "clock_limit": "Provider datetime is stored as its raw naive string and lexical calendar day. No timezone assertion is needed for accepted exact-day joins; original xG publication and revision times remain unknown.",
        "future_research_proxy": "If separately authorized and frozen before fitting, use only past completed-match xG after the later canonical/provider completion date, next local midnight, plus seven full days. Compare longer-delay sensitivity. This is an availability assumption, not point-in-time evidence; exclude unresolved date or score discrepancies.",
        "fitted_models": 0, "performance_scores_computed": False, "same_match_xg_is_not_a_prematch_feature": True}
    assert acquire.canonical_inputs() == lock["canonical_inputs"]
    assert all(acquire.sha(acquire.ROOT / p) == h for p, h in source_files.items())
    acquire.write(output, report)
    print(json.dumps(report["totals"], indent=2), flush=True)


if __name__ == "__main__":
    main()
