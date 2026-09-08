"""Replay exact tree routing, leaf gradients, probabilities and temporal membership."""
from datetime import datetime
import json
import numpy as np
from scipy.special import logsumexp
from research.experiments.boundary_20260908.football.cycle2 import model, run
from research.experiments.boundary_20260908.football import run as parent
from research.experiments.performance_20260907.football import benchmark as b


def main():
    spec=run.specification();lock=json.loads((run.OUT/"selection_lock.json").read_text())
    assert b.file_sha(run.OUT/"selection.json")==lock["selection_artifact_sha256"]
    checked_models=checked_trees=checked_leaves=0
    max_error=max_gradient=0.
    artifacts={}
    for phase in ("selection","evaluation"):
        artifact=json.loads((run.OUT/f"{phase}.json").read_text());artifacts[phase]=artifact
        assert artifact["source_sha256"]==b.digest(run.sources())
        assert artifact["spec_sha256"]==b.file_sha(run.SPEC)
        features,_,hashes=run.inputs(phase,spec)
        assert hashes==artifact["input_feature_artifacts_sha256"]
        for block in artifact["fits"]:
            training=parent.prior_training_rows(features,datetime.fromisoformat(block["cutoff"]))
            ids=[r["match_id"] for r in training]
            assert ids==block["training_ids"] and b.digest(ids)==block["training_ids_sha256"]
            batch=[r for r in artifact["predictions"] if r["league"]==block["league"] and r["fit_id"]==block["fit_id"]]
            assert not set(ids).intersection(r["match_id"] for r in batch)
            x=model.features(training);y=np.asarray([r["label"] for r in training]);ridge=spec["tree"]["leaf_l2_sum_objective"]
            for name,fitted in block["models"].items():
                p=model.predict(batch,fitted);stored=np.asarray([r["probabilities"][name] for r in batch])
                max_error=max(max_error,float(np.max(np.abs(p-stored))))
                np.testing.assert_allclose(p,stored,atol=1e-12,rtol=0)
                logits=np.log(np.asarray([r["probabilities"]["dc365_elo50"] for r in training]));penalty=0.
                for k,tree in enumerate(fitted["trees"]):
                    stack=[(0,np.ones(len(x),dtype=bool))]
                    while stack:
                        node,mask=stack.pop()
                        left=tree["children_left"][node]
                        if left>=0:
                            # Compare float32-rounded inputs in float64 against
                            # sklearn's double threshold, avoiding weak-scalar
                            # promotion that would round the threshold itself.
                            goes_left=x[:,tree["feature"][node]].astype(np.float64) <= tree["threshold"][node]
                            stack.extend([(left,mask&goes_left),(tree["children_right"][node],mask&~goes_left)])
                        else:
                            assert int(mask.sum())==tree["leaves"][str(node)]["n"]
                            unshrunk=np.asarray(tree["leaf_vectors"][node])/spec["tree"]["learning_rate"]
                            _,g,_=model.leaf_objective(unshrunk,logits[mask],y[mask],ridge)
                            norm=float(np.max(np.abs(g)));max_gradient=max(max_gradient,norm)
                            assert norm <= spec["numerics"]["leaf_gradient_inf_tolerance"]+1e-10
                            checked_leaves+=1
                    logits+=model.tree_predict(tree,x)
                    vectors=np.asarray(tree["leaf_vectors"]);penalty+=ridge/2*float(np.sum(vectors*vectors))
                    objective=float(np.mean(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])+penalty/len(y))
                    np.testing.assert_allclose(objective,fitted["objective_history"][k+1],atol=1e-12,rtol=0)
                    checked_trees+=1
                assert np.all(np.diff(fitted["objective_history"])<=1e-10)
                checked_models+=1
    evaluation=artifacts["evaluation"];candidate=evaluation["selected_model"]
    assert candidate==lock["selected_model"]
    assert len(evaluation["predictions"])==2280 and len(artifacts["selection"]["predictions"])==760
    assert parent.summarize(evaluation["predictions"],candidate,spec)==evaluation["summary"]["pooled"]
    filtered=[r for r in evaluation["predictions"] if r["match_id"]!="I1:2024:Fiorentina:Inter"]
    assert len(filtered)==2279
    verification={"verified_at_utc":b.now(),"status":"passed","models_replayed":checked_models,"trees_replayed":checked_trees,
        "regularized_leaf_gradients_recomputed":checked_leaves,"max_leaf_gradient_inf":max_gradient,"max_probability_replay_error":max_error,
        "training_membership_availability_checked":True,"objective_monotonicity_checked":True,"source_sha256":b.digest(run.sources())}
    b.write_json(run.OUT/"verification.json",verification)
    b.write_json(run.OUT/"evidence.json",{"experiment_id":spec["id"],"created_at_utc":b.now(),"status":spec["status"],"promotion":False,
        "selected_model":candidate,"known_exposure":spec["known_exposure"],"spec_sha256":b.file_sha(run.SPEC),"source_files":run.sources(),"source_sha256":b.digest(run.sources()),
        "artifacts":{name:b.file_sha(run.OUT/name) for name in ["design_lock.json","selection.json","selection_lock.json","evaluation.json","verification.json"]},
        "all_candidates_selection_metrics":artifacts["selection"]["summary"]["selection_metrics"],"evaluation_summary":evaluation["summary"],
        "resumed_fixture_exclusion":parent.summarize(filtered,candidate,spec),"verification":verification,"limitations":spec["limitations"],
        "parent_evidence_sha256":b.file_sha(parent.OUT/"evidence.json")})
    print(json.dumps(verification,indent=2))


if __name__=="__main__":
    main()
