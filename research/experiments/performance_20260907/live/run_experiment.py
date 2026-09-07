"""Frozen four-mechanism causal live next-lap experiment; no production imports."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SPEC_PATH = HERE / "spec.json"
SPEC = json.loads(SPEC_PATH.read_text())
FEATURES = SPEC["mechanisms"]["ridge_correction"]["features"]
LEVELS = [(a, c) for a in (.4, .7) for c in (1., 2.)]
SHRINK = (.25, .5, 1.)
KEYS = ["event_key", "driver_id", "issued_after_lap_number", "issued_at_timestamp"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value):
    try:
        out = float(value)
        return out if np.isfinite(out) else np.nan
    except (ValueError, TypeError):
        return np.nan


def truth(value):
    return str(value).lower().strip() in {"true", "1", "1.0"}


def driver_key(value):
    number = numeric(value)
    return str(int(number)) if np.isfinite(number) and number.is_integer() else str(value).strip()


def level_name(a, c):
    return f"robust_level_a{a:g}_c{c:g}"


@dataclass
class DriverState:
    provider_stint: float = np.nan
    compound: str = ""
    last_lap: int = 0
    last_timestamp: float = -np.inf
    last_clean: float = np.nan
    last_clean_lap: int = 0
    total_clean: int = 0
    stint_clean: int = 0
    stint_generation: int = 0
    last_delta: float = 0.
    deltas: deque = field(default_factory=lambda: deque(maxlen=4))
    levels: dict = field(default_factory=dict)
    common_record: tuple | None = None


def stream_event(frame, event_key, *, end_at=None):
    """Emit before touching any future target; match pending forecasts on arrival.

    Other-car summaries are copied before each timestamp batch. Own completed
    lap is available at issuance; same-timestamp other cars are deliberately not.
    """
    work = frame.copy()
    work["Time"] = pd.to_numeric(work["Time"], errors="coerce")
    if end_at is not None:
        work = work.loc[work.Time <= end_at]
    if not np.isfinite(work.Time).all():
        raise ValueError("every observation requires a finite global timestamp")
    work["_driver"] = work.DriverNumber.map(driver_key)
    work = work.sort_values(["Time", "_driver", "LapNumber"], kind="mergesort")
    states, pending, emitted, matched = {}, {}, [], []
    for timestamp, batch in work.groupby("Time", sort=False):
        common_before = {key: state.common_record for key, state in states.items() if state.common_record is not None}
        for row in batch.to_dict("records"):
            driver = row["_driver"]
            state = states.setdefault(driver, DriverState())
            lap_float = numeric(row["LapNumber"])
            if not np.isfinite(lap_float) or lap_float != int(lap_float) or lap_float <= state.last_lap or timestamp <= state.last_timestamp:
                raise ValueError(f"invalid driver/lap chronology: {event_key}/{driver}/{lap_float}")
            lap = int(lap_float)
            compound = str(row.get("Compound", "UNKNOWN")).upper()
            provider_stint = numeric(row.get("Stint"))
            pit_out = np.isfinite(numeric(row.get("PitOutTime")))
            pit_in = np.isfinite(numeric(row.get("PitInTime")))
            reset = state.last_lap == 0 or pit_out or (compound != state.compound and state.compound != "") or (
                np.isfinite(provider_stint) and np.isfinite(state.provider_stint) and provider_stint != state.provider_stint)
            if reset:
                state.stint_generation += 1
                state.stint_clean = 0
                state.levels.clear()
                state.deltas.clear()
                state.last_delta = 0.
                state.common_record = None
                if not np.isfinite(provider_stint):
                    state.provider_stint = np.nan
            if np.isfinite(provider_stint):
                state.provider_stint = provider_stint
            state.compound = compound
            state.last_lap, state.last_timestamp = lap, float(timestamp)
            y = numeric(row.get("LapTime"))
            status = str(row.get("TrackStatus", ""))
            eligible = np.isfinite(y) and y > 0 and truth(row.get("IsAccurate", False)) and not (pit_in or pit_out) and not any(code in status for code in "4567")
            if not eligible:
                continue
            if driver in pending:
                issuance = pending.pop(driver)
                matched.append({**issuance, "target_lap_number": lap, "target_timestamp": float(timestamp),
                                "lap_time_seconds": y, "target_same_stint": issuance["stint_generation"] == state.stint_generation,
                                "skipped_nonrepresentative_laps": lap - issuance["issued_after_lap_number"] - 1})
            gap = lap - state.last_clean_lap if state.stint_clean else 1
            if state.stint_clean:
                delta = (y - state.last_clean) / gap
                state.last_delta = float(np.clip(delta, -1., 1.))
                state.deltas.append(state.last_delta)
                # Only successive clean laps provide a local common increment.
                state.common_record = (float(timestamp), lap, state.last_delta) if gap == 1 else None
            else:
                state.common_record = None
            for a, c in LEVELS:
                old = state.levels.get((a, c), y)
                state.levels[(a, c)] = old + a * float(np.clip(y - old, -c, c))
            state.last_clean, state.last_clean_lap = y, lap
            state.total_clean += 1
            state.stint_clean += 1
            own_trend = float(np.median(state.deltas)) if len(state.deltas) >= 2 else 0.
            others = [record for key, record in common_before.items() if key != driver
                      and 0 < timestamp - record[0] <= 180. and abs(lap - record[1]) <= 1]
            common = float(np.median([record[2] for record in others])) if len(others) >= 3 else 0.
            if state.total_clean < 3:
                continue
            tyre_age = numeric(row.get("TyreLife"))
            features = [state.levels[(.7, 2.)] - y, state.last_delta, own_trend, common,
                        min(len(others), 10) / 10., np.log1p(state.stint_clean),
                        min(max(tyre_age, 0.), 100.) if np.isfinite(tyre_age) else float(state.stint_clean),
                        float(gap), float(compound in {"INTERMEDIATE", "WET"})]
            issuance = dict(event_key=int(event_key), driver_id=driver, issued_after_lap_number=lap,
                issued_at_timestamp=float(timestamp), forecast_naive_seconds=y, stint_generation=state.stint_generation,
                compound=compound, stint_clean_count=state.stint_clean, common_other_drivers=len(others),
                common_evidence_max_timestamp=max((record[0] for record in others), default=np.nan))
            issuance.update(zip(FEATURES, features))
            issuance.update({level_name(a,c): value for (a,c),value in state.levels.items()})
            issuance.update({f"robust_trend_w{w:g}": y + w*own_trend for w in SHRINK})
            issuance.update({f"common_increment_w{w:g}": y + w*common for w in SHRINK})
            emitted.append(issuance)
            pending[driver] = issuance
    return pd.DataFrame(emitted), pd.DataFrame(matched)


def event_errors(frame, prediction):
    return (frame[prediction] - frame.lap_time_seconds).abs().groupby(frame.event_key).mean()


def ridge_fit(frame, alpha):
    x = frame[FEATURES].to_numpy(float)
    y = np.clip(frame.lap_time_seconds.to_numpy() - frame.forecast_naive_seconds.to_numpy(), -3., 3.)
    counts = frame.event_key.value_counts()
    weights = len(frame) / (len(counts) * frame.event_key.map(counts).to_numpy())
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale[scale < 1e-9] = 1.
    model = Ridge(alpha=alpha).fit((x-mean)/scale, y, sample_weight=weights)
    return {"alpha": alpha, "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": model.coef_.tolist(), "intercept": float(model.intercept_)}


def ridge_predict(frame, model):
    x = (frame[FEATURES].to_numpy(float)-np.asarray(model["mean"]))/np.asarray(model["scale"])
    correction = np.clip(x @ np.asarray(model["coef"]) + model["intercept"], -2., 2.)
    return frame.forecast_naive_seconds.to_numpy() + correction


def attach_selected_forecasts(frame, selected, model):
    """Forecast columns cannot overwrite same-named causal input features."""
    before=frame[FEATURES].copy()
    out=frame.copy()
    for family,column in selected.items():
        out[f"prediction_{family}"]=ridge_predict(frame,model) if family=="ridge_correction" else frame[column]
    pd.testing.assert_frame_equal(before,out[FEATURES],check_exact=True)
    return out


def diagnostics(frame, prediction):
    baseline = event_errors(frame, "forecast_naive_seconds")
    candidate = event_errors(frame, prediction).reindex(baseline.index)
    delta = (candidate-baseline).to_numpy()
    rng = np.random.default_rng(20260907)
    draws = delta[rng.integers(len(delta), size=(20000, len(delta)))].mean(axis=1)
    loo = [(delta.sum()-value)/(len(delta)-1) for value in delta] if len(delta)>1 else []
    return dict(events=len(delta), rows=len(frame), baseline_event_mae_seconds=float(baseline.mean()),
        candidate_event_mae_seconds=float(candidate.mean()), delta_seconds=float(delta.mean()),
        relative_improvement=float(1-candidate.mean()/baseline.mean()), ci95_delta_seconds=np.quantile(draws,[.025,.975]).tolist(),
        bootstrap_fraction_improving=float(np.mean(draws<0)), events_improved=int(np.sum(delta<0)),
        loo_mean_deltas=loo, loo_all_improve=bool(loo and max(loo)<0),
        per_event=[dict(event_key=int(key), baseline_mae=float(baseline.loc[key]), candidate_mae=float(candidate.loc[key]), delta=float(candidate.loc[key]-baseline.loc[key])) for key in baseline.index])


def prefix_regression(files, selected, model):
    checked=[]
    for event_key in [202201,202506,202601]:
        path=files[event_key]
        raw=pd.read_csv(path)
        full,_=stream_event(raw,event_key)
        for fraction in [.25,.6]:
            cutoff=float(np.quantile(raw.Time,fraction))
            prefix,_=stream_event(raw,event_key,end_at=cutoff)
            poisoned=raw.copy()
            future=poisoned.Time>cutoff
            poisoned.loc[future,"LapTime"]=98765.
            poisoned.loc[future,"IsAccurate"]=False
            poisoned.loc[future,"Stint"]=999.
            poisoned.loc[future,"Compound"]="WET"
            poison,_=stream_event(poisoned,event_key)
            expected=full.loc[full.issued_at_timestamp<=cutoff].reset_index(drop=True)
            for variant in [prefix,poison.loc[poison.issued_at_timestamp<=cutoff]]:
                actual=variant.reset_index(drop=True)
                pd.testing.assert_frame_equal(expected,actual,check_exact=True)
                np.testing.assert_array_equal(ridge_predict(expected,model),ridge_predict(actual,model))
            evidence=expected.common_evidence_max_timestamp.dropna()
            assert (evidence < expected.loc[evidence.index,"issued_at_timestamp"]).all()
            checked.append(dict(event_key=event_key,cutoff=cutoff,issuances=len(expected),all_mechanisms_unchanged=True))
    return checked


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/research/performance_20260907/live")
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    if (out/"results.json").exists():
        raise FileExistsError("refusing to replace a completed performance experiment")
    frozen_spec_hash,frozen_code_hash=sha(SPEC_PATH),sha(__file__)
    paths=sorted((ROOT/"data/f1/raw/weekends").glob("*/*/*_race_laps.csv"))
    inventory=[];files={};matches=[];issuance_count=0
    for path in paths:
        year=int(path.parts[-3]);round_number=int(path.parts[-2].split("_")[1]);event_key=year*100+round_number
        if year not in [2022,2023,2024,2025,2026]:continue
        raw=pd.read_csv(path);files[event_key]=path
        emitted,matched=stream_event(raw,event_key)
        issuance_count+=len(emitted)
        matches.append(matched)
        inventory.append(dict(event_key=event_key,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size,
                              raw_rows=len(raw),issuances=len(emitted),matched_rows=len(matched),missing_stint_rows=int(raw.Stint.isna().sum())))
    frame=pd.concat(matches,ignore_index=True)
    frame["year"]=frame.event_key//100
    fit=frame.loc[frame.year==2022]
    selection=frame.loc[frame.year==2023].copy()
    selection_variants={}
    models={}
    for alpha in [1.,10.,100.]:
        name=f"ridge_correction_a{alpha:g}"
        models[name]=ridge_fit(fit,alpha)
        selection[name]=ridge_predict(selection,models[name])
    families={
        "robust_level":[level_name(a,c) for a,c in LEVELS],
        "robust_trend":[f"robust_trend_w{w:g}" for w in SHRINK],
        "common_increment":[f"common_increment_w{w:g}" for w in SHRINK],
        "ridge_correction":list(models),
    }
    selected={}
    for family,variants in families.items():
        scores={variant:float(event_errors(selection,variant).mean()) for variant in variants}
        chosen=min(variants,key=lambda name:scores[name])
        selected[family]=chosen
        selection_variants[family]={"all_variant_event_mae_seconds":scores,"selected_variant":chosen}
    preferred=min(selected,key=lambda family:selection_variants[family]["all_variant_event_mae_seconds"][selected[family]])
    chosen_model=ridge_fit(frame.loc[frame.year.isin([2022,2023])],models[selected["ridge_correction"]]["alpha"])
    frame=attach_selected_forecasts(frame,selected,chosen_model)
    selected_models=dict(selected_variants=selection_variants,preferred_mechanism_by_2023_only=preferred,
                         baseline_2023_event_mae_seconds=float(event_errors(selection,"forecast_naive_seconds").mean()),
                         final_ridge_model=chosen_model)
    # Selection is persisted before any transfer scores are calculated.
    (out/"selected_models.json").write_text(json.dumps(selected_models,indent=2)+"\n")
    regressions=prefix_regression(files,selected,chosen_model)
    results={}
    for block,years in [("transfer_2024",[2024]),("transfer_2025",[2025]),("transfer_2024_2025",[2024,2025]),("exposed_2026",[2026])]:
        subset=frame.loc[frame.year.isin(years)]
        results[block]={family:diagnostics(subset,f"prediction_{family}") for family in families}
    reference_path=ROOT/"artifacts/backtests/f1/live_next_lap/2026_math_remediation_v10_20260907.json"
    reference=json.loads(reference_path.read_text())
    old=pd.concat([pd.DataFrame(item["matched_scoring_rows"]).assign(event_key=item["event_key"]) for item in reference["events"]],ignore_index=True)
    old.driver_id=old.driver_id.astype(str)
    target_keys=KEYS+["target_lap_number","target_timestamp"]
    current=frame.loc[frame.year==2026]
    joined=current.merge(old[target_keys+["forecast_ssm_seconds","forecast_naive_seconds","lap_time_seconds"]],on=target_keys,validate="one_to_one",suffixes=("","_reference"))
    if not len(joined)==len(current)==len(old):raise AssertionError("2026 population differs from retained next-lap contract")
    np.testing.assert_allclose(joined.forecast_naive_seconds,joined.forecast_naive_seconds_reference,rtol=0,atol=1e-10)
    np.testing.assert_allclose(joined.lap_time_seconds,joined.lap_time_seconds_reference,rtol=0,atol=1e-10)
    error_structure={}
    for label,subset in [("all",joined),("same_stint",joined.loc[joined.target_same_stint]),("new_stint",joined.loc[~joined.target_same_stint]),("next_numbered_lap",joined.loc[joined.skipped_nonrepresentative_laps==0]),("skipped_laps",joined.loc[joined.skipped_nonrepresentative_laps>0])]:
        errors=subset.lap_time_seconds-subset.forecast_naive_seconds
        error_structure[label]=dict(rows=len(subset),naive_row_mae_seconds=float(errors.abs().mean()),ssm_row_mae_seconds=float((subset.lap_time_seconds-subset.forecast_ssm_seconds).abs().mean()),
                                   naive_error_quantiles=errors.quantile([.01,.1,.5,.9,.99]).to_dict())
    for item in inventory:
        assert sha(ROOT/item["path"])==item["sha256"],"input mutated during experiment"
    assert sha(SPEC_PATH)==frozen_spec_hash and sha(__file__)==frozen_code_hash,"specification or code changed during scoring"
    frame.to_csv(out/"matched_forecasts.csv.gz",index=False)
    payload=dict(experiment_id=SPEC["experiment_id"],generated_at_utc=datetime.now(timezone.utc).isoformat(),
                 evidence_role="historical_predictive_performance_research_not_prospective_promotion",specification=SPEC,
                 specification_sha256=frozen_spec_hash,runner_sha256=frozen_code_hash,inventory=inventory,
                 stream_count=len(inventory),raw_rows=sum(item["raw_rows"] for item in inventory),issuances=issuance_count,matched_rows=len(frame),
                 selected_models=selected_models,selection_artifact_sha256=sha(out/"selected_models.json"),prefix_regressions=regressions,
                 results=results,naive_ssm_2026_error_structure=error_structure,ssm_reference_path=reference_path.relative_to(ROOT).as_posix(),
                 ssm_reference_sha256=sha(reference_path),ssm_reference_population_exactly_matched=True,
                 matched_forecasts_sha256=sha(out/"matched_forecasts.csv.gz"),runtime={"numpy":np.__version__,"pandas":pd.__version__,"threads":1})
    (out/"results.json").write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(out/"results.json"),"preferred_2023":preferred,"selected":selected,"transfer":{key:{k:v for k,v in value.items() if k not in {"per_event","loo_mean_deltas"}} for key,value in results["transfer_2024_2025"].items()},"exposed_2026":{key:{k:v for k,v in value.items() if k not in {"per_event","loo_mean_deltas"}} for key,value in results["exposed_2026"].items()}},indent=2))


if __name__=="__main__":
    main()
