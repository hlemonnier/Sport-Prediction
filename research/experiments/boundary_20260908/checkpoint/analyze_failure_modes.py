"""Posthoc support/error decomposition only; never trains or selects a model."""
from collections import Counter
import json

import numpy as np
import pandas as pd

from research.experiments.boundary_20260908.checkpoint import run_experiment as exp


def main():
    out = exp.OUT/"failure_analysis.json"
    if out.exists(): raise FileExistsError("preserve completed failure analysis")
    result = json.loads((exp.OUT/"results.json").read_text())
    frame = pd.read_pickle(exp.OUT/"transfer_forecasts.pkl")
    selected = result["selected"]
    observations = {}
    # Count already observed peer updates without making a new prediction.
    # Peers sharing the checkpoint timestamp are excluded as an entire batch.
    for item in result["input_manifest"]:
        raw = pd.read_csv(exp.ROOT/item["path"])
        by_key = {(r.driver_id, r.checkpoint_lap, r.checkpoint_time):r
                  for r in frame.loc[frame.event_key.eq(item["event_key"])].itertuples()}
        raw["_driver"] = raw.DriverNumber.map(exp.encoder.driver_key)
        raw = raw.sort_values(["Time", "_driver", "LapNumber"], kind="mergesort")
        peers = {}
        for timestamp, batch in raw.groupby("Time", sort=False):
            before = dict(peers)
            for row in batch.to_dict("records"):
                driver, lap = row["_driver"], int(row["LapNumber"])
                checkpoint = by_key.get((driver, lap, float(timestamp)))
                if checkpoint is not None:
                    recent = [(d, t, l) for d,(t,l) in before.items()
                              if d != driver and checkpoint.prior_time < t < timestamp
                              and timestamp-t <= 180. and abs(lap-l) <= 1]
                    observations[(item["event_key"], driver, lap, float(timestamp))] = dict(
                        new_eligible_peer_count=len(recent),
                        peer_max_timestamp=max((t for _,t,_ in recent), default=np.nan),
                        future_pit_marker_masked=any(
                            exp.encoder.numeric(row.get(c)) > timestamp for c in ["PitInTime", "PitOutTime"]))
                if exp.clean(row): peers[driver] = (float(timestamp), lap)
    f = frame.copy()
    support = pd.DataFrame([observations[(r.event_key, r.driver_id, r.checkpoint_lap, r.checkpoint_time)]
                            for r in f.itertuples()], index=f.index)
    assert (support.peer_max_timestamp.isna() | (support.peer_max_timestamp < f.checkpoint_time)).all()
    assert (support.peer_max_timestamp.isna() | (support.peer_max_timestamp > f.prior_time)).all()
    summaries = {}
    for label, mask in [("2024_2025",f.year.isin([2024,2025])), ("2026_exposed",f.year.eq(2026))]:
        g = f.loc[mask]; s = support.loc[mask]
        gate = ~g.eligible
        cases = dict(all_ineligible=gate,
                     ineligible_with_observed_stint_transition=gate & g.c_observed_stint_transition.eq(1),
                     ineligible_current_compound_known=gate & g.c_compound_missing.eq(0),
                     ineligible_sectors_2_and_3_available=gate & g.c_sector23_missing.eq(0),
                     ineligible_future_pit_marker_was_masked=gate & s.future_pit_marker_masked,
                     ineligible_at_least_3_new_strictly_prior_eligible_peers=gate & s.new_eligible_peer_count.ge(3),
                     ineligible_fewer_than_3_new_eligible_peers=gate & s.new_eligible_peer_count.lt(3))
        summaries[label] = {}
        for name, subset in cases.items():
            h = g.loc[subset]
            metric = exp.metrics(h, h.reference.to_numpy(), h["prediction_"+selected].to_numpy())
            metric.pop("per_event", None)
            summaries[label][name] = metric
    absolute_error = np.abs(f["prediction_"+selected]-f.target_seconds)
    baseline_error = np.abs(f.reference-f.target_seconds)
    changed = ~f.eligible
    exp.write(out, dict(
        evidence_role="Posthoc diagnostic support and error decomposition; no new forecasts, fitting, selection or gain criteria.",
        result_sha256=exp.sha(exp.OUT/"results.json"),
        selected=selected, summaries=summaries,
        eligible_peer_rule="Different driver, clean observation timestamp strictly after original HGB issuance and strictly before checkpoint, age<=180seconds and lap difference<=1. Timestamp batches excluded. This is available-observation support, not a fitted field-pace prediction.",
        all_peer_timestamps_strictly_prior_verified=True,
        gated_correction_cap_hits=int((changed & np.isclose(abs(f["prediction_"+selected]-f.reference),3.)).sum()),
        gated_unchanged_corrections=int((changed & np.isclose(f["prediction_"+selected],f.reference)).sum()),
        largest_checkpoint_regressions=f.loc[changed].assign(error_increase=absolute_error-baseline_error).nlargest(12,"error_increase")
             [[*exp.KEYS,"category","prior_lap","target_lap","target_seconds","reference","prediction_"+selected,"error_increase"]].to_dict("records")))
    print(json.dumps({k:{name:dict(rows=m["rows"], events=m["events"], relative_reduction=m.get("relative_reduction"))
                         for name,m in v.items()} for k,v in summaries.items()},indent=2))


if __name__ == "__main__": main()
