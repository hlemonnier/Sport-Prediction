"""Fixed out-of-league transfer; explicit loader and context for frozen EPL math."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo
from threadpoolctl import threadpool_limits
import numpy as np
from research.experiments.performance_20260907.football import benchmark as b

SPEC_PATH = b.LANE / "transfer_spec.json"
MODELS = ["dc_equal", "dc365_elo50"]


@contextmanager
def league_context(timezone_name: str):
    """Parameterize only timezone and required DC fits; leave module files intact."""
    previous_timezone, previous_dc = b.LONDON, b.DC
    b.LONDON = ZoneInfo(timezone_name)
    b.DC = {"dc_equal": None, "dc_365": 365.0}
    try:
        yield
    finally:
        b.LONDON, b.DC = previous_timezone, previous_dc


def parse_season(path: Path, league: str, year: int, expected_matches: int = 380) -> list[b.Row]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            if not any(raw.values()):
                continue
            if raw.get("Div") != league:
                raise ValueError("Unexpected transfer league")
            d = raw["Date"]
            day = datetime.strptime(d, "%d/%m/%Y" if len(d.split("/")[-1]) == 4 else "%d/%m/%y").date()
            if not date(year, 7, 1) <= day <= date(year+1, 8, 31):
                raise ValueError("Transfer date outside season")
            hg, ag = int(raw["FTHG"]), int(raw["FTAG"])
            label = 0 if hg > ag else 1 if hg == ag else 2
            if min(hg, ag) < 0 or raw["FTR"] != "HDA"[label]:
                raise ValueError("Invalid or inconsistent transfer score")
            h, a = raw["HomeTeam"].strip(), raw["AwayTeam"].strip()
            if not h or not a or h == a:
                raise ValueError("Invalid transfer teams")
            match = b.MatchRecord(f"{league}:{year}:{h}:{a}", b.utc_midnight(day), year, league, None,
                                 h, a, hg, ag, None, None,
                                 result_available_at=b.utc_midnight(day+timedelta(days=1)))
            # Closing odds are not parsed: this extension only tests fixed transport.
            rows.append(b.Row(day, match, label))
    if len(rows) != expected_matches:
        raise ValueError(f"Incomplete transfer season: {league}/{year} has{len(rows)} rows")
    ids = [r.match.match_id for r in rows]
    teams = {r.match.home_team_id for r in rows} | {r.match.away_team_id for r in rows}
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate transfer fixture")
    if len(teams) != 20 or any(sum(t in (r.match.home_team_id, r.match.away_team_id) for r in rows) != 38 for t in teams):
        raise ValueError("Transfer schedule is not balanced20team38match")
    return rows


def load_league(league: str, spec: dict) -> tuple[list[b.Row], dict]:
    manifest = json.loads((b.DATA / "transfer_source_manifest.json").read_text())
    entries = {f["name"]: f for f in manifest["files"]}
    rows, populations = [], {}
    for year in spec["input_season_start_years"]:
        name = f"transfer_{league}_{year}_{year+1}.csv"
        path = b.DATA / name
        if b.file_sha(path) != entries[name]["sha256"]:
            raise ValueError("Transfer input changed")
        season = parse_season(path, league, year, spec["expected_matches_per_season"])
        populations[str(year)] = {"n": len(season), "ids_sha256": b.digest(sorted(r.match.match_id for r in season)),
            "first_source_day": min(r.day for r in season).isoformat(), "last_source_day": max(r.day for r in season).isoformat()}
        rows.extend(season)
    return sorted(rows, key=lambda r: (r.day, r.match.match_id)), populations


def summary(records: list[dict], spec: dict) -> dict:
    metrics = {m: b.metrics(records, m) for m in MODELS}
    paired = b.paired_uncertainty(records, "dc365_elo50", {"uncertainty": spec["inherited_uncertainty"]}, 28)
    return {"n": len(records), "population_sha256": b.digest([r["match_id"] for r in records]),
            "models": metrics, "paired_28_day_interval": paired}


def verify_predictions(records: list[dict], fits: list[dict], rows: list[b.Row], elo_spec: dict) -> dict:
    by_id = {r.match.match_id: r for r in rows}
    fit_by_id = {f["fit_id"]: f for f in fits}
    features = b.causal_elo_features(rows, elo_spec)
    maximum = 0.0
    for r in records:
        f = fit_by_id[r["fit_id"]]
        cutoff = datetime.fromisoformat(f["cutoff_utc"])
        assert r["match_id"] not in f["fit_match_ids"]
        assert all(b.match_available_at(by_id[mid].match) <= cutoff for mid in f["fit_match_ids"])
        assert r["forecast_cutoff_utc"] == b.utc_midnight(date.fromisoformat(r["day"])).isoformat()
        assert r["elo_features"] == features[r["match_id"]]
        params = f["elo_mapping"]
        logits = np.asarray(params["coefficients"]) @ np.asarray(r["elo_features"]) + params["intercept"]
        elo = np.exp(logits-logits.max()); elo /= elo.sum()
        for name in ("dc_equal", "dc_365"):
            p = f["dixon_coles"][name]
            lh = np.exp(p["home_intercept"]+p["attack"].get(r["home"], 0)+p["defense"].get(r["away"], 0))
            la = np.exp(p["away_intercept"]+p["attack"].get(r["away"], 0)+p["defense"].get(r["home"], 0))
            expected = np.asarray(b.build_score_distribution(lh, la, p["rho"]).outcome_probabilities)
            maximum = max(maximum, float(np.max(np.abs(expected-r["probabilities"][name]))))
            if name == "dc_365":
                maximum = max(maximum, float(np.max(np.abs((expected+elo)/2-r["probabilities"]["dc365_elo50"]))))
    assert maximum < 1e-12
    return {"passed": True, "prediction_rows": len(records), "maximum_probability_reconstruction_error": maximum,
            "checks": ["same fit population", "prior-result availability", "target absent from fit", "league local day-start cutoff",
                       "causal Elo features", "DC probabilities and fixed mixture reconstructed from stored parameters"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-name")
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    parent_spec = json.loads((b.LANE / "spec.json").read_text())
    parent_evidence = json.loads((b.OUT / "evidence.json").read_text())
    assert b.file_sha(b.LANE / "spec.json") == spec["parent_epl_spec_sha256"]
    assert b.file_sha(b.OUT / "evidence.json") == spec["parent_epl_evidence_sha256"]
    assert b.digest(b.sources()) == parent_evidence["provenance"]["source_code_sha256"]
    assert parent_spec["elo"] == spec["inherited_elo"]
    assert parent_evidence["selected_before_test"] == spec["frozen_selected_model"]
    lock = json.loads((b.OUT / "transfer_design_lock.json").read_text())
    assert lock["spec_sha256"] == b.file_sha(SPEC_PATH)
    out = b.OUT
    if args.replay_name:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.replay_name):
            raise ValueError("Invalid replay name")
        out = out / f"transfer_replay_{args.replay_name}"
        out.mkdir(exist_ok=False)
    if (out / "transfer_evidence.json").exists() or any((out / f"transfer_{league}.json").exists() for league in spec["leagues"]):
        raise ValueError("Transfer results already exist; use a fresh replay name")
    source_hashes = {str(p.relative_to(b.ROOT)): b.file_sha(p) for p in (SPEC_PATH, Path(__file__), b.LANE / "transfer_timing_notes.json")}
    provenance = {"source_files": source_hashes, "original_epl_source_code_sha256": b.digest(b.sources()),
                  "transfer_input_manifest_sha256": b.file_sha(b.DATA / "transfer_source_manifest.json"),
                  "input_manifest": json.loads((b.DATA / "transfer_source_manifest.json").read_text()),
                  "python": b.platform.python_version(), "numpy": np.__version__, "scipy": b.scipy.__version__,
                  "sklearn": b.sklearn.__version__, "blas_threads": 1}
    b.write_json(out / "transfer_run_lock.json", {"created_before_metrics_at_utc": b.now(),
        "spec_sha256": b.file_sha(SPEC_PATH), "provenance": provenance,
        "selected_model_already_frozen_on_epl": spec["frozen_selected_model"], "transfer_selection": False})
    evidence, pooled = {}, []
    for league, info in spec["leagues"].items():
        with league_context(info["timezone"]), threadpool_limits(limits=1):
            rows, populations = load_league(league, spec)
            predictions, fits = b.predict_phase(rows, spec["test_seasons"], parent_spec)
            verification = verify_predictions(predictions, fits, rows, parent_spec["elo"])
        overall = summary(predictions, spec)
        yearly = {str(y): summary([r for r in predictions if r["season"] == y], spec) for y in spec["test_seasons"]}
        entry = {"league": info, "overall": overall, "by_season": yearly, "verification": verification,
                 "convergence": {"joint_fits": 2*len(fits), "refit_blocks": len(fits),
                     "maximum_kkt_residual": max(m["diagnostics"]["kkt_stationarity_inf_norm"] for f in fits for m in f["dixon_coles"].values())}}
        if league == "I1":
            sensitivity = [r for r in predictions if r["match_id"] != "I1:2024:Fiorentina:Inter"]
            assert len(sensitivity) == len(predictions)-1
            entry["pre_metric_known_resumption_exclusion_sensitivity"] = summary(sensitivity, spec)
        b.write_json(out / f"transfer_{league}.json", {"spec_sha256": b.file_sha(SPEC_PATH), "provenance": provenance,
            "populations": populations, "summary": entry, "predictions": predictions, "fits": fits})
        evidence[league] = entry
        # Unique league-season strata retain the identical bootstrap implementation.
        pooled.extend([{**r, "season": f"{league}:{r['season']}"} for r in predictions])
        print(json.dumps({"league": league, "fixed_transfer": overall}, indent=2), flush=True)
    result = {"experiment_id": spec["experiment_id"], "completed_at_utc": b.now(),
        "status": "historical_fixed_out_of_league_transfer", "promotion": False, "selection_on_transfer": False,
        "specification": spec, "spec_sha256": b.file_sha(SPEC_PATH), "provenance": provenance,
        "timing_notes": json.loads((b.LANE / "transfer_timing_notes.json").read_text()),
        "leagues": evidence, "pooled_1520_match_transfer": summary(pooled, spec),
        "artifacts": {name: b.file_sha(out / name) for name in ["transfer_run_lock.json", "transfer_SP1.json", "transfer_I1.json"]}}
    b.write_json(out / "transfer_evidence.json", result)
    print(json.dumps({"pooled": result["pooled_1520_match_transfer"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
