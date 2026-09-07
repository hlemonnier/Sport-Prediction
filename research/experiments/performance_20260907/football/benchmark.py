"""Frozen EPL prequential benchmark; run validation before unlocking test.

Research runner only. It imports production probability math without changing it.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time as clock
import warnings
from zoneinfo import ZoneInfo

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from packages.football.mrp.data import MatchRecord, match_available_at
from packages.football.mrp.joint import fit_dixon_coles
from packages.football.mrp.score_distribution import build_score_distribution

ROOT = Path(__file__).resolve().parents[4]
LANE = Path(__file__).resolve().parent
DATA = ROOT / "data/football/performance_20260907"
OUT = ROOT / "artifacts/research/performance_20260907/football"
LONDON = ZoneInfo("Europe/London")
DC = {"dc_equal": None, "dc_365": 365.0, "dc_180": 180.0}
CAUSAL_MODELS = [*DC, "elo_component", "dc365_elo50", "league_frequency"]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sources() -> dict:
    files = sorted((ROOT / "packages/football/mrp").glob("*.py")) + [LANE / "benchmark.py"]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in files}


def utc_midnight(day: date) -> datetime:
    return datetime.combine(day, time(), LONDON).astimezone(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class Row:
    day: date
    match: MatchRecord
    label: int
    closing: tuple[float, float, float] | None = None


def normalized_odds(values: list[str]) -> tuple[float, float, float] | None:
    try:
        odds = np.array([float(v) for v in values])
    except (ValueError, TypeError):
        return None
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1):
        return None
    p = 1 / odds
    return tuple(map(float, p / p.sum()))


def load_rows(years: list[int], spec: dict) -> tuple[list[Row], dict]:
    manifest = json.loads((DATA / "source_manifest.json").read_text())
    file_info = {item["name"]: item for item in manifest["files"]}
    rows = []
    populations = {}
    for year in years:
        name = f"E0_{year}_{year+1}.csv"
        path = DATA / name
        if file_sha(path) != file_info[name]["sha256"]:
            raise ValueError(f"Input changed: {name}")
        season_rows = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            for raw in csv.DictReader(f):
                if not any(v for v in raw.values()):
                    continue
                if raw.get("Div") != "E0":
                    raise ValueError("Unexpected league")
                d = raw["Date"]
                day = datetime.strptime(d, "%d/%m/%Y" if len(d.split("/")[-1]) == 4 else "%d/%m/%y").date()
                if not date(year, 7, 1) <= day <= date(year+1, 8, 31):
                    raise ValueError("Date outside declared season (including COVID extension)")
                hg, ag = int(raw["FTHG"]), int(raw["FTAG"])
                if min(hg, ag) < 0:
                    raise ValueError("Negative score")
                label = 0 if hg > ag else 1 if hg == ag else 2
                if raw["FTR"] != "HDA"[label]:
                    raise ValueError("Score/result mismatch")
                h, a = raw["HomeTeam"].strip(), raw["AwayTeam"].strip()
                if not h or not a or h == a:
                    raise ValueError("Invalid teams")
                match = MatchRecord(f"E0:{year}:{h}:{a}", utc_midnight(day), year, "epl", None,
                                    h, a, hg, ag, None, None,
                                    result_available_at=utc_midnight(day + timedelta(days=1)))
                season_rows.append(Row(day, match, label,
                    normalized_odds([raw.get(c, "") for c in ("AvgCH", "AvgCD", "AvgCA")])))
        if len(season_rows) != spec["expected_matches_per_season"]:
            raise ValueError(f"{year}: expected 380 complete scored rows, found {len(season_rows)}")
        ids = [r.match.match_id for r in season_rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate home-away fixture in a season")
        teams = {r.match.home_team_id for r in season_rows} | {r.match.away_team_id for r in season_rows}
        if len(teams) != 20 or any(sum(t in (r.match.home_team_id, r.match.away_team_id) for r in season_rows) != 38 for t in teams):
            raise ValueError("Unbalanced EPL schedule")
        populations[str(year)] = {"matches": len(ids), "teams": len(teams),
            "first_day": min(r.day for r in season_rows).isoformat(),
            "last_day": max(r.day for r in season_rows).isoformat(),
            "ids_sha256": digest(sorted(ids)), "closing_available": sum(r.closing is not None for r in season_rows)}
        rows.extend(season_rows)
    return sorted(rows, key=lambda r: (r.day, r.match.match_id)), populations


def causal_elo_features(rows: list[Row], elo_spec: dict) -> dict[str, list[float]]:
    ratings = defaultdict(lambda: elo_spec["initial_rating"])
    features = {}
    by_day = defaultdict(list)
    for row in rows:
        by_day[row.day].append(row)
    scale, k, advantage = (elo_spec[x] for x in ("logistic_scale", "k_factor", "home_advantage_rating_points"))
    for day in sorted(by_day):
        changes = defaultdict(float)
        for row in by_day[day]:
            h, a = row.match.home_team_id, row.match.away_team_id
            difference = ratings[h] - ratings[a]
            features[row.match.match_id] = [difference / scale, abs(difference) / scale]
            expected = 1 / (1 + 10 ** (-(difference + advantage) / scale))
            change = k * (elo_spec["win_draw_loss_score"][row.label] - expected)
            changes[h] += change
            changes[a] -= change
        # Atomic day updates: no fixture uses another same-date match result.
        for team, change in changes.items():
            ratings[team] += change
    return features


def validate_probabilities(p: object) -> np.ndarray:
    a = np.asarray(p, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or np.any(~np.isfinite(a)) or np.any(a <= 0) or not np.allclose(a.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError("Invalid strictly positive normalized 1X2 probabilities")
    return a


def metrics(records: list[dict], model: str) -> dict:
    p = validate_probabilities([r["probabilities"][model] for r in records])
    y = np.array([r["label"] for r in records])
    true = np.eye(3)[y]
    confidence = p.max(1)
    correct = p.argmax(1) == y
    bins = []
    ece = 0.0
    for k in range(10):
        mask = (confidence >= k/10) & ((confidence < (k+1)/10) if k < 9 else (confidence <= 1))
        n = int(mask.sum())
        if n:
            gap = float(confidence[mask].mean() - correct[mask].mean())
            ece += n / len(y) * abs(gap)
            bins.append({"lower": k/10, "upper": (k+1)/10, "n": n,
                         "mean_confidence": float(confidence[mask].mean()), "accuracy": float(correct[mask].mean())})
    return {"n": len(y), "log_loss": float(-np.log(p[np.arange(len(y)), y]).mean()),
            "brier_sum_classes": float(((p-true)**2).sum(1).mean()), "accuracy": float(correct.mean()),
            "top_label_ece_10_bins": ece, "calibration_bins": bins,
            "class_mean_probability_minus_frequency": (p.mean(0)-true.mean(0)).tolist()}


def summarize(records: list[dict]) -> dict:
    result = {"common_population_sha256": digest([r["match_id"] for r in records]),
              "all_matches": {m: metrics(records, m) for m in CAUSAL_MODELS}, "by_season": {}}
    for season in sorted({r["season"] for r in records}):
        subset = [r for r in records if r["season"] == season]
        result["by_season"][str(season)] = {m: metrics(subset, m) for m in CAUSAL_MODELS}
    market = [r for r in records if "closing_market" in r["probabilities"]]
    result["later_information_closing_market_subset"] = {
        "n": len(market), "excluded_match_ids": [r["match_id"] for r in records if "closing_market" not in r["probabilities"]],
        "timing": "closing odds unavailable at the model's day-start cutoff; descriptive only",
        "models": {m: metrics(market, m) for m in [*CAUSAL_MODELS, "closing_market"]} if market else {}}
    return result


def predict_phase(rows: list[Row], scored_seasons: list[int], spec: dict) -> tuple[list[dict], list[dict]]:
    features = causal_elo_features(rows, spec["elo"])
    predictions, fits = [], []
    bucket = None
    for row in rows:
        if row.match.season not in scored_seasons:
            continue
        key = (row.day.year, (row.day.month-1)//2)
        cutoff = utc_midnight(row.day)
        if key != bucket:
            training = [r for r in rows if row.day-timedelta(days=spec["rolling_fit_calendar_days"]) <= r.day < row.day
                        and match_available_at(r.match) <= cutoff]
            ids = [r.match.match_id for r in training]
            started = clock.monotonic()
            models = {name: fit_dixon_coles([r.match for r in training], [], half_life_days=half_life,
                                          reference_time=cutoff) for name, half_life in DC.items()}
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                elo = LogisticRegression(C=1, solver="lbfgs", max_iter=1000, tol=1e-8).fit(
                    [features[r.match.match_id] for r in training], [r.label for r in training])
            assert list(elo.classes_) == [0, 1, 2]
            counts = np.bincount([r.label for r in training], minlength=3) + 1
            league = (counts / counts.sum()).tolist()
            fit_id = f"{row.day.isoformat()}:{digest(ids)[:12]}"
            fits.append({"fit_id": fit_id, "cutoff_utc": cutoff.isoformat(), "fit_match_ids": ids,
                         "fit_ids_sha256": digest(ids), "fit_content_sha256": digest([
                             [r.match.match_id, r.day.isoformat(), r.match.home_goals, r.match.away_goals] for r in training]),
                         "latest_result_available_at": max(match_available_at(r.match) for r in training).isoformat(),
                         "seconds": clock.monotonic()-started,
                         "dixon_coles": {m: {"attack": obj.attack, "defense": obj.defense,
                             "home_intercept": obj.home_intercept, "away_intercept": obj.away_intercept,
                             "rho": obj.rho, "diagnostics": {k: v for k, v in obj.diagnostics.items() if k != "fit_match_ids"}}
                             for m, obj in models.items()},
                         "elo_mapping": {"coefficients": elo.coef_.tolist(), "intercept": elo.intercept_.tolist(),
                                         "iterations": elo.n_iter_.tolist()}})
            bucket = key
            print(json.dumps({"fitted": fit_id, "n": len(ids), "seconds": round(fits[-1]["seconds"], 2)}), flush=True)
        probabilities = {}
        for name, model in models.items():
            rates = model.expected_goals(row.match.home_team_id, row.match.away_team_id)
            probabilities[name] = list(build_score_distribution(*rates, model.rho).outcome_probabilities)
        probabilities["elo_component"] = elo.predict_proba([features[row.match.match_id]])[0].tolist()
        probabilities["dc365_elo50"] = ((np.array(probabilities["dc_365"]) + probabilities["elo_component"]) / 2).tolist()
        probabilities["league_frequency"] = league
        if row.closing is not None:
            probabilities["closing_market"] = list(row.closing)
        validate_probabilities(list(probabilities.values()))
        predictions.append({"match_id": row.match.match_id, "day": row.day.isoformat(), "season": row.match.season,
                            "forecast_cutoff_utc": cutoff.isoformat(), "fit_id": fit_id,
                            "home": row.match.home_team_id, "away": row.match.away_team_id,
                            "label": row.label, "goals": [row.match.home_goals, row.match.away_goals],
                            "elo_features": features[row.match.match_id], "probabilities": probabilities})
    return predictions, fits


def paired_uncertainty(records: list[dict], model: str, spec: dict, block_days: int) -> dict:
    n_resamples = spec["uncertainty"]["resamples"]
    rng = np.random.default_rng(spec["uncertainty"]["seed"])
    boot = np.zeros(n_resamples)
    deltas = []
    blocks_by_season = {}
    for season in sorted({r["season"] for r in records}):
        selected = [r for r in records if r["season"] == season]
        anchor = min(date.fromisoformat(r["day"]) for r in selected)
        blocks = defaultdict(list)
        for r in selected:
            p, baseline = r["probabilities"][model][r["label"]], r["probabilities"]["dc_equal"][r["label"]]
            delta = float(-np.log(p)+np.log(baseline))
            deltas.append(delta)
            block = (date.fromisoformat(r["day"])-anchor).days // block_days
            blocks[block].append(delta)
        totals = np.array([sum(v) for v in blocks.values()])
        counts = np.array([len(v) for v in blocks.values()])
        samples = rng.integers(0, len(totals), size=(n_resamples, len(totals)))
        boot += len(selected)/len(records) * (totals[samples].sum(1)/counts[samples].sum(1))
        blocks_by_season[str(season)] = len(blocks)
    return {"difference_candidate_minus_dc_equal": float(np.mean(deltas)),
            "percentile_95_interval": np.quantile(boot, [.025, .975]).tolist(),
            "bootstrap_fraction_difference_below_zero": float(np.mean(boot < 0)),
            "block_calendar_days": block_days, "blocks_by_season": blocks_by_season,
            "resamples": n_resamples, "note": "Paired, season-stratified fixed-block resampling; conditional historical uncertainty, not a posterior probability or season-transfer guarantee"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["validation", "test", "report"], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((LANE / "spec.json").read_text())
    design_lock = json.loads((OUT / "design_lock.json").read_text())
    assert design_lock["spec_sha256"] == file_sha(LANE / "spec.json")
    source_hashes = sources()
    source_manifest = json.loads((DATA / "source_manifest.json").read_text())
    provenance = {"spec_sha256": file_sha(LANE / "spec.json"), "source_code_sha256": digest(source_hashes),
                  "source_code_files": source_hashes, "input_manifest_sha256": file_sha(DATA / "source_manifest.json"),
                  "input_files": source_manifest["files"], "python": platform.python_version(),
                  "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                  "platform": platform.platform(), "blas_threads": 1}
    if args.phase == "report":
        val = json.loads((OUT / "validation.json").read_text())
        test = json.loads((OUT / "test.json").read_text())
        selection = json.loads((OUT / "selection_lock.json").read_text())
        for prior in (val, test):
            assert prior["provenance"]["source_code_sha256"] == provenance["source_code_sha256"]
        comparisons = {m: {"day_blocks": paired_uncertainty(test["predictions"], m, spec, 1),
                           "blocks_28_days": paired_uncertainty(test["predictions"], m, spec, 28)}
                       for m in CAUSAL_MODELS if m != "dc_equal"}
        write_json(OUT / "evidence.json", {"experiment_id": spec["experiment_id"], "created_at_utc": now(),
            "status": "historical_retrospective_research_only", "promotion": False,
            "test_status": "held aside during this research cycle; not claimed universally unseen or prospective",
            "selected_before_test": selection["selected_model"], "selection": selection,
            "provenance": provenance, "validation": val["summary"], "test": test["summary"],
            "paired_test_uncertainty": comparisons, "spec": spec,
            "artifacts": {name: file_sha(OUT / name) for name in ("design_lock.json", "selection_lock.json", "validation.json", "test.json")},
            "convergence": {phase: {"refit_blocks": len(report["fits"]),
                "joint_fits": sum(len(f["dixon_coles"]) for f in report["fits"]),
                "maximum_kkt_residual": max(m["diagnostics"]["kkt_stationarity_inf_norm"] for f in report["fits"] for m in f["dixon_coles"].values())}
                for phase, report in (("validation", val), ("test", test))}})
        print(json.dumps({"selected_model": selection["selected_model"], "test": test["summary"]["all_matches"],
                          "paired": comparisons}, indent=2))
        return
    if (OUT / f"{args.phase}.json").exists():
        raise ValueError("Phase artifact already exists; do not silently overwrite frozen results")
    if args.phase == "test":
        selection = json.loads((OUT / "selection_lock.json").read_text())
        if selection["source_code_sha256"] != provenance["source_code_sha256"] or selection["spec_sha256"] != provenance["spec_sha256"]:
            raise ValueError("Code/spec changed after validation selection")
        if selection["input_manifest_sha256"] != provenance["input_manifest_sha256"]:
            raise ValueError("Inputs changed after validation selection")
        if selection["validation_artifact_sha256"] != file_sha(OUT / "validation.json"):
            raise ValueError("Validation artifact changed after selection")
    years = spec["warmup_seasons"] + spec["validation_seasons"] + (spec["test_seasons"] if args.phase == "test" else [])
    rows, populations = load_rows(years, spec)
    with threadpool_limits(limits=1):
        predictions, fits = predict_phase(rows, spec[f"{args.phase}_seasons"], spec)
    summary = summarize(predictions)
    write_json(OUT / f"{args.phase}.json", {"phase": args.phase, "completed_at_utc": now(),
               "provenance": provenance, "populations": populations, "summary": summary,
               "predictions": predictions, "fits": fits})
    if args.phase == "validation":
        selected = min(spec["selectable_models_in_tie_order"], key=lambda m: summary["all_matches"][m]["log_loss"])
        write_json(OUT / "selection_lock.json", {"selected_model": selected, "selected_at_utc": now(),
            "selection_metric": "pooled_validation_log_loss", "test_labels_loaded": False,
            "all_tried_models": CAUSAL_MODELS + ["closing_market"],
            "selectable_models": spec["selectable_models_in_tie_order"],
            "validation_scores": {m: summary["all_matches"][m]["log_loss"] for m in CAUSAL_MODELS},
            "spec_sha256": provenance["spec_sha256"], "source_code_sha256": provenance["source_code_sha256"],
            "input_manifest_sha256": provenance["input_manifest_sha256"],
            "validation_artifact_sha256": file_sha(OUT / "validation.json")})
    print(json.dumps({"phase": args.phase, "n": len(predictions), "scores": {m: v["log_loss"] for m, v in summary["all_matches"].items()}}), flush=True)


if __name__ == "__main__":
    main()
