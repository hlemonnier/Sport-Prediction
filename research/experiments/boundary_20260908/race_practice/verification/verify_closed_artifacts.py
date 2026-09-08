"""Reconstruct fits, features and references without changing frozen artifacts."""
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.race_practice import run


def main():
    output = run.OUT
    selection = json.loads((output / "selection.json").read_text())
    frozen = json.loads((output / "selection_frozen.json").read_text())
    assert run.sha(output / "selection.json") == frozen["selection_sha256"]
    assert run.sha(output / "selection_design_lock.json") == frozen["selection_design_lock_sha256"]
    assert run.source_manifest() == frozen["source_manifest"]
    append_only_log = None
    for p, expected in {**selection["input_manifest"], **selection["output_manifest"]}.items():
        if p.endswith("/selection_execution.log"):
            # The frozen runner accidentally hashed its still-open stdout log.
            # Prove the originally bound bytes remain an exact line prefix;
            # permit only appended output, without weakening any model/data hash.
            raw = (run.ROOT / p).read_bytes()
            prefix = b""
            matches = []
            for line in raw.splitlines(keepends=True):
                prefix += line
                if hashlib.sha256(prefix).hexdigest() == expected:
                    matches.append(len(prefix))
            assert len(matches) == 1, "original execution-log prefix changed"
            append_only_log = {"path": p, "bound_prefix_bytes": matches[0],
                               "bound_prefix_sha256": expected, "closed_bytes": len(raw),
                               "closed_sha256": hashlib.sha256(raw).hexdigest()}
        else:
            assert run.sha(run.ROOT / p) == expected, p
    reference = json.loads(run.PRIOR_RESULT.read_text())
    spec = json.loads((run.HERE / "specification.json").read_text())
    # Keep temporary output under ROOT because paths in the replay manifest are relative to ROOT.
    with tempfile.TemporaryDirectory(prefix="verification_", dir=output) as folder:
        replay = run.execute(2023, spec["candidates"], reference, Path(folder), "selection")
        assert replay["scores"] == selection["scores"]
        assert replay["input_manifest"] == selection["input_manifest"]
        for name in ["selection_feature_ledger.json", "selection_forecasts.json", "selection_fits.json",
                     "selection_forecasts_before_targets.jsonl"]:
            assert (Path(folder) / name).read_bytes() == (output / name).read_bytes(), name
    points = pd.DataFrame(json.loads((output / "selection_forecasts.json").read_text()))
    old = pd.read_csv(run.ROOT / reference["predictions_path"])
    source_binding = 0
    for candidate in run.REFERENCES:
        a = points.loc[points.candidate.eq(candidate)].sort_values(["event", "driver"])
        b = old.loc[old.candidate.eq(candidate) & old.year.eq(2023)].sort_values(["event", "driver_id"])
        assert a.event.tolist() == b.event.tolist() and a.driver.tolist() == b.driver_id.tolist()
        np.testing.assert_array_equal(a.prediction.to_numpy(), b.prediction.to_numpy())
        np.testing.assert_array_equal(a.target.to_numpy(), b.actual.to_numpy())
        source_binding += len(a)
    chosen = min(spec["candidates"], key=lambda k: (np.mean([r["mae"] for r in selection["scores"][k]]), k))
    assert chosen == selection["selected"]
    comparisons = {r: run.paired(selection["scores"][chosen], selection["scores"][r])
                   for r in [*run.REFERENCES, chosen + "_without_practice"]}
    assert comparisons == selection["comparisons"]
    advance = all(v["relative_gain"] >= .01 and v["loo_max"] < 0 for v in comparisons.values())
    assert advance == selection["advancement_passed"]
    # Recompute event MAE directly from saved driver predictions, independently of event_score.
    points["absolute_error"] = (points.prediction - points.target).abs()
    direct = points.groupby(["candidate", "event"]).absolute_error.mean()
    for candidate, records in selection["scores"].items():
        for r in records:
            assert direct.loc[candidate, r["event"]] == r["mae"]
    report = {"status": "passed_with_append_only_log_binding_amendment", "selected": chosen, "advancement_passed": advance,
              "exact_replayed_predictions": len(points), "exact_reference_target_bindings": source_binding,
              "input_hashes": len(selection["input_manifest"]), "source_hashes": len(frozen["source_manifest"]),
              "original_selection_sha256": run.sha(output / "selection.json"),
              "append_only_log_binding_amendment": append_only_log,
              "verification_source_sha256": run.sha(Path(__file__)),
              "all_model_fits_reconstructed": True, "all_features_reconstructed": True,
              "all_selection_metrics_reconstructed": True, "production_changed": False}
    run.save(output / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
