"""Current-peer checkpoint refinement with exact no-support cycle1 fallback."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from research.experiments.boundary_20260908.checkpoint import run_experiment as c1

HERE = Path(__file__).resolve().parent
OUT = c1.OUT/"cycle2"
SPEC_PATH = HERE/"spec.json"
SPEC = json.loads(SPEC_PATH.read_text())
C1_SELECTION = json.loads((c1.OUT/"selection.json").read_text())
C1_VARIANT = C1_SELECTION["selected"]


def peer_features(raw, issued):
    """All issuance keys are target-independent; all peer records are in the past."""
    key = int(issued.event_key.iloc[0])
    lookup = {(r.driver_id, r.checkpoint_lap, r.checkpoint_time):r for r in issued.itertuples()}
    states, histories, history_times = {}, defaultdict(list), defaultdict(list)
    emitted = []
    ordered = raw.assign(_driver=raw.DriverNumber.map(c1.encoder.driver_key)).sort_values(
        ["Time", "_driver", "LapNumber"], kind="mergesort")
    for timestamp, batch in ordered.groupby("Time", sort=False):
        before = {d:s["peer"] for d,s in states.items() if s["peer"] is not None}
        for row in batch.to_dict("records"):
            d, lap = row["_driver"], int(row["LapNumber"])
            s = states.setdefault(d, dict(stint=np.nan, compound="UNKNOWN", generation=0,
                                         peer=None, offsets=deque(maxlen=6), stint_offsets=deque(maxlen=6)))
            compound, stint = c1.compound_name(row.get("Compound")), c1.encoder.numeric(row.get("Stint"))
            reset = (s["generation"] == 0 or c1.available_pit(row, "PitOutTime")
                     or (compound != "UNKNOWN" and s["compound"] != "UNKNOWN" and compound != s["compound"])
                     or (np.isfinite(stint) and np.isfinite(s["stint"]) and stint != s["stint"]))
            if reset:
                s["generation"] += 1; s["peer"] = None; s["stint_offsets"].clear()
            if np.isfinite(stint): s["stint"] = stint
            if compound != "UNKNOWN": s["compound"] = compound
            current = [p for other,p in before.items() if other != d and 0 < timestamp-p["time"] <= 180
                       and abs(lap-p["lap"]) <= 1]
            is_clean = c1.clean(row)
            if is_clean and len(current) >= 3:
                offset = float(np.clip(float(row["LapTime"])-np.median([p["y"] for p in current]), -20, 20))
                s["offsets"].append(offset); s["stint_offsets"].append(offset)
            checkpoint = lookup.get((d, lap, float(timestamp)))
            if checkpoint is not None:
                new = [p for p in current if p["time"] > checkpoint.prior_time]
                paired, same_stint = [], []
                for p in new:
                    i = bisect_left(history_times[p["driver"]], checkpoint.prior_time)-1
                    if i < 0: continue
                    old = histories[p["driver"]][i]
                    if checkpoint.prior_time-old["time"] > 180 or abs(checkpoint.prior_lap-old["lap"]) > 1: continue
                    change = float(np.clip(p["y"]-old["y"], -20, 20))
                    paired.append(change)
                    if p["generation"] == old["generation"] and p["compound"] == old["compound"]:
                        same_stint.append(change)
                same_compound = [p for p in new if compound != "UNKNOWN" and p["compound"] == compound]
                feat = {}
                def summary(name, values):
                    values = np.asarray(values, dtype=float)
                    supported = len(values) >= 3
                    med = float(np.median(values)) if supported else 0.
                    feat[name+"_count"] = float(len(values))
                    feat[name+"_median"] = med
                    feat[name+"_mad"] = float(np.median(abs(values-med))) if supported else 0.
                    feat[name+"_missing"] = float(not supported)
                summary("current_field", [p["y"] for p in current])
                summary("new_field", [p["y"] for p in new])
                summary("same_compound_field", [p["y"] for p in same_compound])
                summary("paired_change", paired)
                summary("same_stint_paired_change", same_stint)
                summary("driver_offset", list(s["offsets"]))
                summary("driver_stint_offset", list(s["stint_offsets"]))
                feat["new_peer_count"] = float(len(new))
                feat["new_peer_age_median"] = float(np.median([timestamp-p["time"] for p in new])) if new else 0.
                feat["new_peer_age_max"] = float(max(timestamp-p["time"] for p in new)) if new else 0.
                old_supported = checkpoint.p_x_peer_count >= 3
                old_field = checkpoint.prior_naive_seconds-checkpoint.p_x_relative_field_pace
                feat["new_minus_old_field"] = float(np.clip(feat["new_field_median"]-old_field, -30,30)) if old_supported and len(new)>=3 else 0.
                feat["new_minus_old_field_missing"] = float(not old_supported or len(new)<3)
                emitted.append(dict(event_key=key, driver_id=d, checkpoint_lap=lap, checkpoint_time=float(timestamp),
                                    peer_evidence_max_timestamp=max((p["time"] for p in new), default=np.nan),
                                    **{"a_"+k:float(v) for k,v in feat.items()}))
            if is_clean:
                record = dict(driver=d, time=float(timestamp), lap=lap, y=float(row["LapTime"]),
                              compound=compound, generation=s["generation"])
                s["peer"] = record
                histories[d].append(record); history_times[d].append(float(timestamp))
    result = pd.DataFrame(emitted)
    assert len(result) == len(issued) and not result.duplicated(c1.KEYS).any()
    assert (result.peer_evidence_max_timestamp.isna() | (result.peer_evidence_max_timestamp < result.checkpoint_time)).all()
    return result


def enrich(frame, issued, manifest):
    pieces = []
    for item in manifest:
        raw = pd.read_csv(c1.ROOT/item["path"])
        assert c1.sha(c1.ROOT/item["path"]) == item["sha256"]
        points = issued.loc[issued.event_key.eq(item["event_key"])]
        pieces.append(peer_features(raw, points))
    extras = pd.concat(pieces, ignore_index=True)
    merged = frame.merge(extras, on=c1.KEYS, validate="one_to_one", sort=False)
    assert len(merged) == len(frame)
    return merged, extras


def active(frame):
    return (~frame.eligible & frame.a_new_peer_count.ge(3)).to_numpy()


def features(frame, base, cycle1):
    result = c1.expert_features(frame, base)
    for column in sorted(c for c in frame if c.startswith("a_")): result[column] = frame[column].to_numpy()
    result["cycle1_point"] = cycle1
    result["cycle1_correction"] = cycle1-base
    for name in ["driver_offset", "driver_stint_offset"]:
        supported = frame["a_"+name+"_missing"].eq(0) & frame.a_new_field_missing.eq(0)
        result[name+"_field_estimate_gap"] = np.where(supported,
            np.clip(frame.a_new_field_median+frame["a_"+name+"_median"]-cycle1,-30,30),0.)
        result[name+"_field_estimate_missing"] = (~supported).astype(float)
    assert np.isfinite(result.to_numpy()).all()
    return result


def fit(frame, variant):
    gate = active(frame); f = frame.loc[gate]
    model = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=.06, max_iter=150,
          max_leaf_nodes=variant["max_leaf_nodes"], min_samples_leaf=40, l2_regularization=10,
          early_stopping=False, random_state=SPEC["seed"])
    model.fit(features(f, f.reference.to_numpy(), f.cycle1_prediction.to_numpy()),
              np.clip(f.target_seconds-f.cycle1_prediction,-6,6), sample_weight=c1.event_weights(f))
    return model


def predict(frame, base, cycle1, model, variant):
    output = np.array(cycle1, copy=True); gate = active(frame)
    if gate.any():
        change = model.predict(features(frame.loc[gate], base[gate], cycle1[gate]))
        output[gate] += np.clip(change, -variant["prediction_clip_seconds"], variant["prediction_clip_seconds"])
    np.testing.assert_array_equal(output[~gate], cycle1[~gate])
    np.testing.assert_array_equal(output[frame.eligible], base[frame.eligible])
    assert np.isfinite(output).all() and (output>0).all()
    return output


def report(frame, base, cycle1, candidate):
    return dict(vs_production=c1.reports(frame, base, candidate), vs_cycle1=c1.reports(frame, cycle1, candidate),
                support_rows=int(active(frame).sum()), unsupported_ineligible_rows=int((~frame.eligible & ~active(frame)).sum()))


def dependencies():
    return {str(p.relative_to(c1.ROOT)):c1.sha(p) for p in [Path(__file__),SPEC_PATH,
            c1.HERE/"run_experiment.py",c1.OUT/"fit_lock.json",c1.OUT/"selection.json",c1.OUT/"models.pkl",
            c1.OUT/"results.json",c1.OUT/"verification.json",c1.OUT/"amended_input_lock.json",
            c1.OUT/"crossfit_residual_training.pkl",c1.OUT/"selection_forecasts.pkl"]}


def discover():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"design_lock.json").exists(): raise FileExistsError("existing cycle2 design")
    c1.verify_lock()
    c1.write(OUT/"design_lock.json",dict(frozen_at=datetime.now(timezone.utc).isoformat(), specification=SPEC, dependencies=dependencies()))
    frame = pd.read_pickle(c1.OUT/"crossfit_residual_training.pkl")
    issued = pd.read_pickle(c1.OUT/"discovery_issuances.pkl")
    full = pd.read_pickle(c1.OUT/"discovery_checkpoints.pkl")
    full, extras = enrich(full, issued, C1_SELECTION["input_manifest"])
    joined = frame.merge(extras,on=c1.KEYS,validate="one_to_one",sort=False)
    assert len(joined)==len(frame)
    with (c1.OUT/"models.pkl").open("rb") as f: old_models=pickle.load(f)
    training_parts, stage1_models, folds = [], {}, []
    for i,fold in enumerate(SPEC["training"]["2022_cycle1_expert_crossfit"]):
        fitting=joined.loc[joined.event_key.isin(fold["train_events"])]
        scoring=joined.loc[joined.event_key.isin(fold["score_events"])].copy()
        assert max(fold["train_events"])<min(fold["score_events"])
        expert=c1.fit_expert(fitting,fitting.reference.to_numpy(),C1_VARIANT)
        scoring["cycle1_prediction"]=c1.predict_expert(scoring,scoring.reference.to_numpy(),expert,C1_VARIANT)
        training_parts.append(scoring);stage1_models[str(i)]=expert;folds.append(fold)
    training=pd.concat(training_parts,ignore_index=True)
    validation=joined.loc[joined.year.eq(2023)].copy()
    validation["cycle1_prediction"]=c1.predict_expert(validation,validation.reference.to_numpy(),
        old_models["discovery_experts"][C1_VARIANT["name"]],C1_VARIANT)
    results, predicted, discovery_models = {}, {}, {}
    for variant in SPEC["training"]["grid"]:
        model=fit(training,variant)
        values=predict(validation,validation.reference.to_numpy(),validation.cycle1_prediction.to_numpy(),model,variant)
        results[variant["name"]]=report(validation,validation.reference.to_numpy(),validation.cycle1_prediction.to_numpy(),values)
        predicted[variant["name"]]=values;discovery_models[variant["name"]]=model
        print("peer-selection",variant["name"],results[variant["name"]]["vs_cycle1"]["full_checkpoint"]["relative_reduction"],flush=True)
    selected=min(SPEC["training"]["grid"],key=lambda v:results[v["name"]]["vs_production"]["full_checkpoint"]["candidate_mae"])
    final_training=pd.concat([training,validation],ignore_index=True)
    final_models={v["name"]:fit(final_training,v) for v in SPEC["training"]["grid"]}
    with (OUT/"models.pkl").open("wb") as f:pickle.dump(dict(stage1_crossfit=stage1_models,discovery=discovery_models,final=final_models),f)
    final_training.to_pickle(OUT/"crossfit_training.pkl")
    for name,values in predicted.items():validation["prediction_"+name]=values
    validation.to_pickle(OUT/"selection_forecasts.pkl")
    c1.write(OUT/"selection.json",dict(selected=selected, all_selection_results=results, folds=folds,
        selected_beats_cycle1_2023=results[selected["name"]]["vs_cycle1"]["full_checkpoint"]["delta"]<0,
        training_rows=len(training),training_active_rows=int(active(training).sum()),
        final_training_rows=len(final_training),final_active_rows=int(active(final_training).sum()),
        final_events=sorted(int(k) for k in final_training.event_key.unique()),
        input_manifest=C1_SELECTION["input_manifest"],promotion=False))
    c1.write(OUT/"fit_lock.json",dict(frozen_before_transfer=datetime.now(timezone.utc).isoformat(), dependencies=dependencies(),
        selection_sha256=c1.sha(OUT/"selection.json"),models_sha256=c1.sha(OUT/"models.pkl"),
        training_sha256=c1.sha(OUT/"crossfit_training.pkl"),selection_forecasts_sha256=c1.sha(OUT/"selection_forecasts.pkl")))


def verify_lock():
    lock=json.loads((OUT/"fit_lock.json").read_text())
    assert lock["dependencies"]==dependencies()
    for name,field in [("selection.json","selection_sha256"),("models.pkl","models_sha256"),
                       ("crossfit_training.pkl","training_sha256"),("selection_forecasts.pkl","selection_forecasts_sha256")]:
        assert c1.sha(OUT/name)==lock[field]
    c1.verify_lock()
    return lock


def transfer():
    verify_lock()
    if (OUT/"results.json").exists():raise FileExistsError("existing cycle2 transfer")
    selection=json.loads((OUT/"selection.json").read_text())
    original=json.loads((c1.OUT/"results.json").read_text())
    frame=pd.read_pickle(c1.OUT/"transfer_forecasts.pkl")
    issued=pd.read_pickle(c1.OUT/"transfer_issuances.pkl")
    frame,extras=enrich(frame,issued,original["input_manifest"])
    frame["cycle1_prediction"]=frame["prediction_"+C1_VARIANT["name"]]
    with (OUT/"models.pkl").open("rb") as f:models=pickle.load(f)
    base,cycle1=frame.reference.to_numpy(),frame.cycle1_prediction.to_numpy()
    predictions={v["name"]:predict(frame,base,cycle1,models["final"][v["name"]],v) for v in SPEC["training"]["grid"]}
    masks={"2024":frame.year.eq(2024),"2025":frame.year.eq(2025),"2024_2025":frame.year.isin([2024,2025]),
           "2026_exposed":frame.year.eq(2026),"2026_recent_exposed":frame.event_key.ge(202610)}
    results={period:{name:report(frame.loc[mask],base[mask],cycle1[mask],values[mask]) for name,values in predictions.items()}
             for period,mask in masks.items()}
    name=selection["selected"]["name"];hist=results["2024_2025"][name];current=results["2026_exposed"][name]
    criteria=dict(historical_full_gain_10pct=hist["vs_production"]["full_checkpoint"]["relative_reduction"]>=.1,
        historical_target_balanced_gain_5pct=hist["vs_production"]["target_balanced"]["relative_reduction"]>=.05,
        both_historical_years_improve=all(results[y][name]["vs_production"]["full_checkpoint"]["delta"]<0 for y in ["2024","2025"]),
        historical_block_ci_negative=hist["vs_production"]["full_checkpoint"]["block3_ci95"][1]<0,
        historical_all_loo_improve=hist["vs_production"]["full_checkpoint"]["loo_max_delta"]<0,
        exposed2026_improves=current["vs_production"]["full_checkpoint"]["delta"]<0,
        historical_both_weights_improve_vs_cycle1=all(hist["vs_cycle1"][w]["delta"]<0 for w in ["full_checkpoint","target_balanced"]),
        exposed2026_improves_vs_cycle1=current["vs_cycle1"]["full_checkpoint"]["delta"]<0,
        exact_no_support_fallback=all(np.array_equal(p[~active(frame)],cycle1[~active(frame)]) for p in predictions.values()),
        exact_eligible_points=all(np.array_equal(p[frame.eligible],base[frame.eligible]) for p in predictions.values()))
    for candidate,values in predictions.items():frame["prediction_"+candidate]=values
    frame.to_pickle(OUT/"transfer_forecasts.pkl");extras.to_pickle(OUT/"transfer_issuance_peer_features.pkl")
    columns=[*c1.KEYS,"target_lap","target_time","target_seconds","eligible","category","reference","cycle1_prediction", "a_new_peer_count",
             *["prediction_"+v["name"] for v in SPEC["training"]["grid"]]]
    frame[columns].to_csv(OUT/"transfer_forecasts.csv.gz",index=False)
    verify_lock()
    c1.write(OUT/"results.json",dict(experiment=SPEC["experiment"],selected=name,results=results,gain_criteria=criteria,
        substantial_research_gain_passed=all(criteria.values()),promotion=False,fit_lock_sha256=c1.sha(OUT/"fit_lock.json"),
        forecasts_sha256=c1.sha(OUT/"transfer_forecasts.pkl"),csv_sha256=c1.sha(OUT/"transfer_forecasts.csv.gz"),
        input_manifest=original["input_manifest"],exact_cycle1_population=True,retrospective_only=True))
    print(json.dumps(dict(selected=name,criteria=criteria,summary={period:{ref:{k:v for k,v in scores[name][ref]["full_checkpoint"].items() if k!="per_event"}
              for ref in ["vs_production","vs_cycle1"]} for period,scores in results.items()}),indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["discover","transfer"])
    args=parser.parse_args();discover() if args.phase=="discover" else transfer()
