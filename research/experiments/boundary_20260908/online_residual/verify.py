"""Check frozen selection and real-data delayed-label invariance independently."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.online_residual import run as m


def main():
    design = json.loads((m.OUT / "design_lock.json").read_text())
    selection = json.loads((m.OUT / "selection.json").read_text())
    assert design["source_sha256"] == m.sha(m.HERE / "run.py")
    assert design["spec_sha256"] == m.sha(m.HERE / "specification.json")
    assert design["original_source_sha256"] == m.sha(m.original.__file__)
    assert design["discovery_data_sha256"] == m.sha(m.OLD / "discovery_data.pkl")
    assert selection["design_lock_sha256"] == m.sha(m.OUT / "design_lock.json")
    original_selection = json.loads((m.OLD / "selection.json").read_text())
    for source in original_selection["input_manifest"]:
        assert m.sha(m.ROOT / source["path"]) == source["sha256"]
    data = pd.read_pickle(m.OUT / "selection_predictions.pkl")
    discovery = pd.read_pickle(m.OLD / "discovery_data.pkl")
    expected = discovery.loc[discovery.year.eq(2023)].reset_index(drop=True)
    pd.testing.assert_frame_equal(data[expected.columns], expected, check_exact=True)
    assert set(data.year) == {2023}
    seen_validation = set()
    for fold in selection["crossfit_blocks"]:
        fit, val = fold["fit_events"], fold["prediction_events"]
        assert max(fit) < min(val)
        assert all(x // 100 == 2022 for x in fit+val)
        assert not (set(val) & seen_validation)
        seen_validation.update(val)
    with (m.OUT / "selection_models.pkl").open("rb") as f:
        models = pickle.load(f)
    assert set(models) == set(selection["metrics"])
    xx = m.assimilate(data, data.strong_baseline.to_numpy())
    for name, model in models.items():
        prediction = data.strong_baseline.to_numpy()+m.correction(model, xx)
        np.testing.assert_array_equal(prediction, data["prediction_"+name].to_numpy())
        errors = pd.Series(np.abs(prediction-data.lap_time_seconds.to_numpy()), index=data.index)
        mae = float(errors.groupby(data.event_key).mean().mean())
        assert abs(mae-selection["metrics"][name]["candidate_mae"]) < 1e-12
        recalculated = m.metric(data, prediction, data.strong_baseline.to_numpy())
        for k in ["block3_ci95", "event_ci95", "loo_max_delta"]:
            np.testing.assert_array_equal(recalculated[k], selection["metrics"][name][k])
    preferred = min(selection["metrics"], key=lambda n: (selection["metrics"][n]["candidate_mae"], n))
    assert preferred == selection["preferred"]
    assert selection["validation_gain_screen_passed"] == (selection["metrics"][preferred]["relative_gain"] >= .005)
    probes = []
    for event_key in [202301, 202313, 202322]:
        event = data.loc[data.event_key.eq(event_key)].reset_index(drop=True)
        if event.empty:
            continue
        baseline = event.strong_baseline.to_numpy()
        original = m.assimilate(event, baseline)
        for quantile in [.3, .7]:
            cutoff = float(event.issued_at_timestamp.quantile(quantile))
            poisoned = event.copy()
            future = poisoned.target_timestamp > cutoff
            poisoned.loc[future, "lap_time_seconds"] = -1e9
            poisoned.loc[future, "target_same_stint"] = False
            actual = m.assimilate(poisoned, baseline)
            keep = event.issued_at_timestamp <= cutoff
            pd.testing.assert_frame_equal(original.loc[keep], actual.loc[keep], check_exact=True)
            probes.append({"event_key": event_key, "cutoff": cutoff, "issuances": int(keep.sum())})
    if not selection["validation_gain_screen_passed"]:
        assert not (m.OUT / "fit_lock.json").exists()
        assert not (m.OUT / "transfer_predictions.pkl").exists()
        assert not (m.OUT / "results.json").exists()
    result = {"status": "passed", "selection_sha256": m.sha(m.OUT / "selection.json"),
        "verifier_sha256": m.sha(__file__), "raw_sources_verified": len(original_selection["input_manifest"]),
        "selection_rows": len(data), "candidate_predictions_reproduced": len(models),
        "crossfit_validation_events": len(seen_validation), "future_outcome_poisoning": probes,
        "rejected_before_transfer": not selection["validation_gain_screen_passed"]}
    m.write(m.OUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
