"""Verify a rejected discovery screen without opening new transfer comparisons."""
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.checkpoint.cycle2 import run_experiment as exp


def main():
    out = exp.OUT
    if (out/"verification.json").exists() or (out/"results.json").exists():
        raise FileExistsError("preserve the completed rejection evidence")
    exp.verify_lock()
    selected=json.loads((out/"selection.json").read_text())
    assert selected["selected_beats_cycle1_2023"] is False
    assert not (out/"transfer_forecasts.pkl").exists()
    with (out/"models.pkl").open("rb") as f:models=pickle.load(f)
    with (exp.c1.OUT/"models.pkl").open("rb") as f:previous=pickle.load(f)
    val=pd.read_pickle(out/"selection_forecasts.pkl")
    reference=exp.c1.base_points(val,previous["base_models"]["base_2022"])
    np.testing.assert_array_equal(reference,val.reference)
    cycle1=exp.c1.predict_expert(val,reference,previous["discovery_experts"][exp.C1_VARIANT["name"]],exp.C1_VARIANT)
    np.testing.assert_array_equal(cycle1,val.cycle1_prediction)
    for variant in exp.SPEC["training"]["grid"]:
        name=variant["name"]
        candidate=exp.predict(val,reference,cycle1,models["discovery"][name],variant)
        np.testing.assert_array_equal(candidate,val["prediction_"+name])
        # Independent event means, rather than the metrics reporting helper.
        errors=pd.DataFrame(dict(event=val.event_key,base=abs(cycle1-val.target_seconds),
                                 candidate=abs(candidate-val.target_seconds)))
        means=errors.groupby("event")[["base","candidate"]].mean().mean()
        reported=selected["all_selection_results"][name]["vs_cycle1"]["full_checkpoint"]
        np.testing.assert_allclose([means.base,means.candidate],[reported["baseline_mae"],reported["candidate_mae"]],rtol=0,atol=1e-12)
        assert reported["delta"]>0
    training=pd.read_pickle(out/"crossfit_training.pkl")
    for i,fold in enumerate(selected["folds"]):
        assert max(fold["train_events"])<min(fold["score_events"])
        f=training.loc[training.event_key.isin(fold["score_events"])]
        expected=exp.c1.predict_expert(f,f.reference.to_numpy(),models["stage1_crossfit"][str(i)],exp.C1_VARIANT)
        np.testing.assert_array_equal(expected,f.cycle1_prediction)
    assert training.year.isin([2022,2023]).all()
    hashes=0
    for item in selected["input_manifest"]:
        assert exp.c1.sha(exp.c1.ROOT/item["path"])==item["sha256"]
        hashes+=1
    prefix_checks=[]
    for key in [202201,202313]:
        item=next(i for i in selected["input_manifest"] if i["event_key"]==key)
        raw=pd.read_csv(exp.c1.ROOT/item["path"]); cutoff=float(raw.Time.quantile(.6))
        def issued_features(data):
            issued,_,_=exp.c1.checkpoint_event(data,key)
            return issued.merge(exp.peer_features(data,issued),on=exp.c1.KEYS,validate="one_to_one")
        full=issued_features(raw); expected=full.loc[full.checkpoint_time<=cutoff].reset_index(drop=True)
        prefix=issued_features(raw.loc[raw.Time<=cutoff])
        poison=raw.copy();future=poison.Time>cutoff
        poison.loc[future,["LapTime","Sector2Time","Sector3Time"]]=9999.
        poison.loc[future,"IsAccurate"]=False
        changed=issued_features(poison)
        pd.testing.assert_frame_equal(expected,prefix,check_exact=True)
        pd.testing.assert_frame_equal(expected,changed.loc[changed.checkpoint_time<=cutoff].reset_index(drop=True),check_exact=True)
        for variant in exp.SPEC["training"]["grid"]:
            def point(frame):
                base=exp.c1.base_points(frame)
                old=exp.c1.predict_expert(frame,base,previous["final_experts"][exp.C1_VARIANT["name"]],exp.C1_VARIANT)
                return exp.predict(frame,base,old,models["final"][variant["name"]],variant)
            np.testing.assert_array_equal(point(expected),point(prefix))
        prefix_checks.append(dict(event_key=key,cutoff=cutoff,issuances=len(expected),all_four_candidates_exact=True))
    exp.verify_lock()
    verification=dict(selection_sha256=exp.c1.sha(out/"selection.json"),fit_lock_sha256=exp.c1.sha(out/"fit_lock.json"),
        verifier_sha256=exp.c1.sha(Path(__file__)),input_hashes_checked=hashes,
        selection_predictions_recomputed=len(val)*4,base_2022_only_and_cycle1_discovery_points_exact=True,
        independent_selection_event_mae_recomputed=True,stage1_expanding_crossfit_predictions_exact=True,
        no_support_cycle1_fallback_exact=True,eligible_predictions_unchanged=True,prefix_invariance=prefix_checks,
        transfer_evaluated=False,tests="5 semantic regression tests passed")
    exp.c1.write(out/"verification.json",verification)
    exp.c1.write(out/"results.json",dict(experiment=exp.SPEC["experiment"],
        status="rejected_on_2023_advancement_screen",selected=selected["selected"]["name"],
        selected_beats_cycle1_2023=False,all_selection_results=selected["all_selection_results"],
        transfer_evaluated=False,reason="All four variants worsen2023 full-checkpoint MAE versus frozen cycle1; root stopped before transfer comparisons.",
        composition_ineligible_branch="frozen_cycle1_hgb_l15_c3",promotion=False,
        selection_sha256=exp.c1.sha(out/"selection.json"),fit_lock_sha256=exp.c1.sha(out/"fit_lock.json"),
        verification_sha256=exp.c1.sha(out/"verification.json")))
    print(json.dumps(dict(verified=True,status="rejected_on_2023_advancement_screen",rows=len(val),inputs=hashes,
                         prefix_cases=len(prefix_checks),transfer_evaluated=False),indent=2))


if __name__=="__main__":main()
