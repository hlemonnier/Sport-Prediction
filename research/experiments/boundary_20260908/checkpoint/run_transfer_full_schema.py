"""Use the canonical full-schema input manifest without refitting or reselection.

The original runner's recent CSV directory predates the frontier's recovery of
sectors/speeds. Keep that runner and every fitting hash frozen; explicitly bind
this narrow input-path amendment before computing any transfer scores.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from research.experiments.boundary_20260908.checkpoint import run_experiment as exp

CANONICAL = exp.ROOT/"artifacts/research/frontier_20260907/live/corrected_input_contract/results.json"
RECOVERY = exp.ROOT/"artifacts/research/frontier_20260907/input_recovery/manifest.json"


def canonical_build(years):
    assert years == [2024, 2025, 2026], "this adapter cannot change fitting inputs"
    manifest = json.loads(CANONICAL.read_text())["input_manifest"]
    assert len(manifest) == 61 and len({i["event_key"] for i in manifest}) == 61
    frames, issued_frames, inventory = [], [], []
    for item in sorted(manifest, key=lambda i:i["event_key"]):
        key = item["event_key"]; path = exp.ROOT/item["path"]
        assert key//100 in years and exp.sha(path) == item["sha256"]
        raw = pd.read_csv(path)
        issued, matched, _ = exp.checkpoint_event(raw, key)
        matched["year"] = key//100; issued["year"] = key//100
        frames.append(matched); issued_frames.append(issued)
        inventory.append(dict(event_key=key, path=item["path"], sha256=item["sha256"],
                              raw_rows=len(raw), issued=len(issued), matched=len(matched),
                              eligible_matched=int(matched.eligible.sum()),
                              future_pit_values_masked=sum(int((pd.to_numeric(raw[c], errors="coerce") > raw.Time).sum())
                                                           for c in ["PitInTime", "PitOutTime"])))
        print("canonical-checkpoints", key, len(matched), flush=True)
    return pd.concat(frames, ignore_index=True), pd.concat(issued_frames, ignore_index=True), inventory


def main():
    exp.verify_lock()
    if (exp.OUT/"results.json").exists() or (exp.OUT/"amended_input_lock.json").exists():
        raise FileExistsError("completed transfer or input amendment exists")
    items = json.loads(CANONICAL.read_text())["input_manifest"]
    exp.write(exp.OUT/"failed_transfer_attempt.json", dict(
        status="source_schema_failure_before_scoring",
        original_runner_sha256=exp.sha(exp.HERE/"run_experiment.py"),
        fit_lock_sha256=exp.sha(exp.OUT/"fit_lock.json"),
        last_completed_event_feature_build=202609, failed_event=202610,
        error="ValueError: Observed-input schema is incomplete: FreshTyre, Position, Sector1Time, Sector2Time, Sector3Time, SpeedFL, SpeedI1, SpeedI2, SpeedST",
        original_recent_directory="data/f1/performance_20260907/live_recent_final",
        transfer_scores_computed=False, source_contract_relaxed=False))
    exp.write(exp.OUT/"amended_input_lock.json", dict(
        frozen_before_transfer_scores=datetime.now(timezone.utc).isoformat(),
        adapter_sha256=exp.sha(Path(__file__)),
        original_fit_lock_sha256=exp.sha(exp.OUT/"fit_lock.json"),
        canonical_input_manifest_container_sha256=exp.sha(CANONICAL),
        prior_input_recovery_manifest_sha256=exp.sha(RECOVERY),
        inputs=[dict(event_key=i["event_key"], path=i["path"], sha256=i["sha256"]) for i in items],
        change="Use the existing canonical full observed-input manifest for all61 transfer events, including restored nine observed fields in R10–13. No feature, target, fitted parameter, selection or criterion changes.",
        refit_permitted=False))
    adapter_hash = exp.sha(Path(__file__))
    exp.build = canonical_build
    exp.transfer()
    assert exp.sha(Path(__file__)) == adapter_hash
    exp.verify_lock()
    # Complete the result's provenance within this same execution, before
    # exposing a finalized artifact to downstream verification.
    result_path = exp.OUT/"results.json"
    result = json.loads(result_path.read_text())
    result["amended_input_lock_sha256"] = exp.sha(exp.OUT/"amended_input_lock.json")
    result["failed_pre_scoring_attempt_sha256"] = exp.sha(exp.OUT/"failed_transfer_attempt.json")
    result["input_contract_recovery_without_refit"] = True
    exp.write(result_path, result)


if __name__ == "__main__": main()
