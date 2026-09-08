The existing frozen HGB is connected to the current default live prediction
path. I found no material integration disconnect. This review made no changes
to production, fitted models or earlier research artifacts, read no 2024–26
scores, and fitted no models.

`PredictionConfig.f1_live_next_lap_point_model` defaults to `frontier_hgb`.
`sources.py` preserves the original required observations and converts duration
fields to seconds before `predict.py` calls the shared `forecast_laps` function.
The platform's local and remote services both attach that same hash-pinned
model's `lap_time_forecast`. Order distributions retain their separate heuristic
semantics; the HGB certifies neither order probabilities nor forecast intervals.
The portable reference, global dispatch, baseline switch, reducer-to-service
parity and unavailable-input tests passed: 18 tests in the two focused suites.

The proposed checkpoint expert addresses a real behavior. In
`packages/f1/models/live_race/next_lap_features.py:312`, ineligible observations
are skipped before issuance. In `predict.py:1300`, a new eligible issuance
replaces the pending point; otherwise its value and issuance metadata persist.
The actual snapshot adapter retained driver 44's lap-10 payload unchanged after
his ineligible lap 11 in 2022 Bahrain. Both models must be compared at the new
checkpoint, with the older point's true issuance time still recorded. The
original eligible-issuance benchmark must remain a separate result.

An independent replay of only 2022–2023 found 44 streams and 47,997 raw records,
44,557 checkpoints with a pending model point, and 43,373 matched checkpoints.
Of the matched checkpoints, 5,003 (11.5%) were ineligible. There were only
38,370 distinct driver-target laps; 1,840 targets had multiple checkpoints,
with maximum multiplicity nine.

| Matched checkpoint category | Rows | Carried HGB MAE |
|---|---:|---:|
| Eligible | 38,370 | 0.486194s |
| Pit entry | 1,448 | 2.768238s |
| Pit exit | 1,479 | 2.871225s |
| Neutralized | 2,071 | 2.516611s |
| Other ineligible | 5 | 0.516516s |

These are descriptive errors on the bundled HGB's training seasons, not
independent performance evidence. They establish support and error headroom,
not an attainable gain. Event-balanced checkpoint MAE was 0.718863s; a perfect
expert on ineligible checkpoints alone could remove at most 0.297989s of it.
That arithmetic upper bound makes the hypothesis worth testing, but it cannot
predict transfer performance.

The following corrections are required before claiming a valid checkpoint
experiment:

1. The bundled HGB was refitted on 2022–2023. It cannot serve as a target-unseen
   stage-one predictor for residual selection in 2023. Use the already fixed HGB
   hyperparameters fitted on 2022 only for that selection period. Prefer blocked
   out-of-fold stage-one points for residual training; disclose any in-sample
   residual training. Freeze both final stages before transfer evaluation.
2. When an eligible target arrives, resolve every earlier checkpoint for that
   driver first, then refresh the HGB point and issue the current checkpoint
   for a strictly later eligible target. Eligibility of an issuance must not
   require the existence, duration or future gap of its target. Keep unmatched
   terminal checkpoints in the issuance ledger.
3. Preserve repeated target identities and bootstrap whole events, not rows.
   Report checkpoint-weighted performance plus a target-balanced sensitivity;
   a long neutralized period creates repeated predictions of the same outcome.
   Predeclare which weighting selects the candidate. Report the gated subgroup
   and full checkpoint population, with eligible forecasts exactly unchanged.
4. A completed-lap `Time` is not proof that every field was available then.
   There are 91 pit timestamps later than their row's `Time` in these files,
   including 82 in 2023 Netherlands. Driver 1, lap 49 has `Time=7851.742` and
   `PitInTime=7852.324`; lap 2's pit entry is 0.875s after `Time`. A new expert
   cannot consume this future pit evidence at the earlier timestamp. Either
   delay the checkpoint and order all observations by their justified available
   time, or mask unavailable fields and state the remaining availability
   assumptions. `IsAccurate` is calculated from the reconstructed row, including
   pit flags. Ordinary future-row poisoning does not detect this same-row issue.
5. Missing inputs need explicit indicators and causal state. In issued pit-entry
   checkpoints, `SpeedFL` is missing in 1,548 of 1,549 records; pit exits have
   211 missing lap durations and 47 missing first-sector times. Unknown compound,
   tyre age or position cannot mean zero or an observed tyre change. Preserve
   the frozen HGB features and point separately from checkpoint measurements;
   never overwrite historical feature columns with refreshed values.

The timestamp issue in item 4 is an **existing retrospective data-availability
limitation**, not a newly introduced production integration defect. The current
production implementation reproduces its frozen research encoder's handling of
the recorded rows. These CSVs also omit `FastF1Generated`, so raw-record counts
alone cannot prove that every record represents a genuinely completed lap.
Installed FastF1 code can generate a terminal retirement row with an assumed
end time; the matched-target evaluation naturally cannot score a terminal row
without a later eligible target. A future data contract should distinguish
record receipt time, lap endpoint, reconstructed flags and generated records.

`inventory.json` contains input hashes, counts, missingness, per-event descriptive
errors, script/model hashes and the actual adapter carry probe. Its runner reads
only the 2022 and 2023 directories and refuses to overwrite completed evidence:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/boundary_20260908/live_review/review_inventory.py
```

No new checkpoint model was fitted or activated. Suggested commit:
`docs(f1-live): record checkpoint hypothesis and runtime parity review`.
