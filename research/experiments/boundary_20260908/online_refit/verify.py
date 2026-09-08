"""Re-fit the selected frozen policy and independently check selection evidence."""
import json

import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.online_refit import run as m


def main():
    selected = json.loads((m.OUT / "selection.json").read_text())
    design = json.loads((m.OUT / "design_lock.json").read_text())
    assert selected["source_sha256"] == m.sha(m.HERE / "run.py")
    assert selected["design_lock_sha256"] == m.sha(m.OUT / "design_lock.json")
    assert selected["predictions_sha256"] == m.sha(m.OUT / "selection_predictions.pkl")
    assert design["source_sha256"] == selected["source_sha256"]
    assert design["spec_sha256"] == m.sha(m.HERE / "specification.json")
    assert design["discovery_sha256"] == m.sha(m.old.OUT / "discovery_data.pkl")
    prior = json.loads((m.old.OUT / "selection.json").read_text())
    for item in prior["input_manifest"]:
        assert m.sha(m.old.ROOT / item["path"]) == item["sha256"]
    data = pd.read_pickle(m.old.OUT / "discovery_data.pkl")
    val = pd.read_pickle(m.OUT / "selection_predictions.pkl")
    pd.testing.assert_frame_equal(data.loc[data.year.eq(2023)], val[data.columns], check_exact=True)
    assert len(selected["metrics"]) == len(design["spec"]["candidate_grid"]) == 5
    checks = 0
    for name, ledger in selected["fit_ledgers"].items():
        observed = []
        for block in ledger:
            fitted, predicted = block["fit_events"], block["prediction_events"]
            assert max(fitted) < min(predicted)
            assert fitted == sorted(data.loc[data.event_key < min(predicted), "event_key"].unique())
            assert block["fit_rows"] == data.event_key.isin(fitted).sum()
            assert block["prediction_rows"] == data.event_key.isin(predicted).sum()
            observed.extend(predicted)
            checks += 1
        assert observed == sorted(val.event_key.unique())
        error = (val[name]-val.lap_time_seconds).abs()
        assert abs(float(error.groupby(val.event_key).mean().mean())-selected["metrics"][name]["candidate_mae"]) < 1e-12
        recalculated = m.metric(val, val[name].to_numpy(), val.strong_baseline.to_numpy())
        for key in ["event_ci95", "block3_ci95", "loo_max_delta"]:
            np.testing.assert_array_equal(recalculated[key], selected["metrics"][name][key])
    preferred = min(selected["metrics"], key=lambda n: (selected["metrics"][n]["candidate_mae"], n))
    assert preferred == selected["preferred"]
    predictions, ledger = m.fit_predict(data, selected["features"], [2023], selected["config"])
    assert ledger == selected["fit_ledgers"][preferred]
    np.testing.assert_array_equal(predictions.loc[val.index].to_numpy(), val[preferred].to_numpy())
    assert selected["validation_screen_passed"] == (selected["metrics"][preferred]["relative_gain"] >= .005)
    if not selected["validation_screen_passed"]:
        assert not (m.OUT / "transfer_predictions.pkl").exists()
        assert not (m.OUT / "results.json").exists()
    m.write(m.OUT / "verification.json", {"status":"passed",
        "source_sha256":m.sha(m.HERE / "run.py"), "verifier_sha256":m.sha(__file__),
        "selection_sha256":m.sha(m.OUT / "selection.json"), "raw_sources_verified":len(prior["input_manifest"]),
        "fit_prediction_blocks_checked":checks, "selected_policy_exactly_refitted":True,
        "selection_rows":len(val), "rejected_before_transfer":not selected["validation_screen_passed"]})
    print("verified selected policy, all scores, chronology and source bindings", flush=True)


if __name__ == "__main__":
    main()
