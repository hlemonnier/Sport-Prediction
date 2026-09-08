"""Freeze the failed-screen branch choices and verify exact cycle-one identity."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.composition.compose import compose
from research.experiments.boundary_20260908.checkpoint import run_experiment as c1
from research.experiments.boundary_20260908.online_residual.run import sha, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ART = ROOT / "artifacts/research/boundary_20260908"
OUT = ART / "composition"


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("Frozen composition already verified")
    design = json.loads((OUT / "design_lock.json").read_text())
    assert design["spec_sha256"] == sha(HERE / "specification.json")
    anchor_path = ART / "anchor/selection.json"
    peer_path = ART / "checkpoint/cycle2/selection.json"
    anchor = json.loads(anchor_path.read_text())
    peer = json.loads(peer_path.read_text())
    assert not anchor["validation_advancement_screen_passed"]
    assert not peer["selected_beats_cycle1_2023"]
    selected = anchor["candidate_metrics"][anchor["selected"]]
    # The original component verifier owns the exact anchor screen arithmetic;
    # bind its entire selection rather than silently inventing another threshold.
    assert isinstance(selected, dict)
    peer_name = peer["selected"]["name"]
    peer_comparison = peer["all_selection_results"][peer_name]["vs_cycle1"]["full_checkpoint"]
    assert peer_comparison["candidate_mae"] >= peer_comparison["baseline_mae"]
    source = ART / "checkpoint/transfer_forecasts.pkl"
    lock = {"frozen_at": datetime.now(timezone.utc).isoformat(),
        "design_lock_sha256": sha(OUT / "design_lock.json"),
        "anchor_selection_sha256": sha(anchor_path), "peer_selection_sha256": sha(peer_path),
        "eligible_branch": "current_hgb", "ineligible_branch": "checkpoint_cycle1_hgb_l15_c3",
        "reason": "Both newly selected components failed their frozen2023advancement rules.",
        "source_forecasts_sha256": sha(source), "compositor_sha256": sha(HERE / "compose.py"),
        "runner_sha256": sha(__file__), "new_component_transfer_read": False}
    write(OUT / "component_lock.json", lock)
    # Only now open the already published cycle-one transfer predictions.
    frame = pd.read_pickle(source)
    point = compose(frame, "prediction_hgb_l15_c3")
    np.testing.assert_array_equal(point, frame.prediction_hgb_l15_c3.to_numpy())
    np.testing.assert_array_equal(point[frame.eligible], frame.loc[frame.eligible, "reference"].to_numpy())
    results = {}
    for name, mask in [("2024",frame.year.eq(2024)),("2025",frame.year.eq(2025)),
                       ("2024_2025",frame.year.isin([2024,2025])),("2026",frame.year.eq(2026))]:
        results[name] = c1.reports(frame.loc[mask], frame.loc[mask,"reference"].to_numpy(), point[mask])
    h = results["2024_2025"]
    criteria = {"historical_full_gain_at_least_10pct":h["full_checkpoint"]["relative_reduction"] >= .10,
        "historical_target_balanced_gain_at_least_5pct":h["target_balanced"]["relative_reduction"] >= .05,
        "each_year_improves":all(results[y]["full_checkpoint"]["delta"] < 0 for y in ["2024","2025","2026"]),
        "historical_block3_upper_negative":h["full_checkpoint"]["block3_ci95"][1] < 0,
        "historical_all_loo_improve":h["full_checkpoint"]["loo_max_delta"] < 0,
        "original_eligible_never_changed":True,
        "improves_over_selected_cycle1":False}
    write(OUT / "results.json", {"status":"verified_identity_fallback_not_new_gain",
        "component_lock_sha256":sha(OUT / "component_lock.json"), "rows":len(frame),
        "original_eligible_rows":int(frame.eligible.sum()), "maximum_difference_from_cycle1":0.0,
        "results":results, "criteria":criteria, "substantial_gate_passed":all(criteria.values()),
        "production_change":False, "new_component_transfer_read":False})
    print(json.dumps({"status":"verified identity fallback", "rows":len(frame),
        "historical_gain":h["full_checkpoint"]["relative_reduction"], "new_gain_over_cycle1":0.,
        "substantial_gate_passed":False}))


if __name__ == "__main__":
    main()
