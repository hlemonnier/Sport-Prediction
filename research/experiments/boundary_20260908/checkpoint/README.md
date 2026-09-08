The frozen checkpoint expert improves the 2024–2025 checkpoint benchmark by
6.879%, but misses the predeclared substantial-gain thresholds. It is research
only. Every original eligible-issuance prediction and target remains unchanged.

| Period | HGB carried-point MAE | Checkpoint expert MAE | Reduction | Target-balanced reduction |
|---|---:|---:|---:|---:|
| 2024 | 0.637568s | 0.583174s | 8.531% | 3.759% |
| 2025 | 0.590224s | 0.560153s | 5.095% | 3.041% |
| 2024–2025 | 0.613896s | 0.571664s | 6.879% | 3.419% |
| Exposed 2026 R1–R13 | 0.643080s | 0.612973s | 4.682% | 1.202% |
| Exposed 2026 R10–R13 | 0.499643s | 0.484528s | 3.025% | 0.741% |

The 48-event historical delta is −0.042232s, with paired event CI
[−0.059958, −0.022652]s and within-year three-event block CI
[−0.061368, −0.021851]s. Forty-three events improve and every leave-one-event-out
mean remains improving. The 2025 checkpoint CI crosses zero on its own; its
target-balanced CI does not. The exposed 2026 checkpoint CI
[−0.096238, +0.021360]s crosses zero and only seven of 13 events improve.
The substantial-gain criteria fail because historical checkpoint improvement
is below 10% and target-balanced improvement is below 5%.

The new population issues after every observed lap record once the current HGB
has a point for that driver, then targets the strictly later next eligible lap.
Both methods are compared at the same checkpoint. The expert has later own-car
information than the earlier HGB issuance; this is explicitly a different
information set. Eligible observations receive zero correction. Ineligible
records may contain useful pit, tyre and sector information despite being
unsuitable clean-lap targets. Multiple checkpoints can share one target, so
target-balanced results give equal weight to each driver's distinct target lap.

One family and four variants were fixed before scoring. HGB residual models use
absolute-error loss, 150 iterations, learning rate 0.06, minimum leaf size 40,
L2 10, and a residual target clipped to ±10s. The grid crosses 7/15 leaves with
±3/±6s output caps. Their historical full-checkpoint reductions are respectively
7.080%, 6.874%, 6.879% and 6.631%. The 2023-selected **15-leaf, ±3s** candidate
remains selected; transfer results do not replace it with the best later score.

Selection uses a base HGB whose coefficients were fitted only on 2022. The base
architecture is inherited from previous research, not newly selected here.
Residual training uses three expanding 2022 folds, with each score event later
than all base-fit events. Final residual fitting uses 4,219 gated rows from
38 events: 2022 events 7–22 plus 2023 predictions from the 2022-fitted base.
No 2024–26 labels enter either fitted stage. Transfer uses the exact bundled
production HGB, with no weakened or missing-column baseline.

The gated historical subgroup improves from 1.932216s to 1.307895s (32.311%).
Pit exits improve 37.031% and pit entries 29.653%, while neutralized observations
deteriorate 0.447%. In 2026, pit exits still improve 28.176%, but neutralized
observations deteriorate 4.437%; this subgroup improves only two of 13 races.
These patterns explain why large pit-transition gains do not produce a large,
stable aggregate gain. The tiny other-ineligible subgroup also deteriorates;
its few-event intervals are descriptive and cannot establish stable behavior.

A posthoc support diagnostic finds at least three newly completed clean peer
observations strictly between original issuance and checkpoint for 2,822 of
4,477 historical gated checkpoints and 949 of 1,714 exposed 2026 gated points.
Peers must be different drivers, at most 180s old and within one numbered lap;
same-timestamp observations are excluded. This cycle carries the prior HGB's
peer features unchanged. Those counts identify available information for a
separate future experiment, not a tested peer-based improvement.

Future-valued PitInTime/PitOutTime fields are masked before expert features;
there is no feature indicating that a masked future value exists. Recorded
IsAccurate/eligibility still derives from retrospective FastF1 records, and
field publication latency is unknown. All years are previously exposed
retrospective data, not prospective validation. Terminal/generated observations
with no later eligible target remain unmatched issuances. The raw CSVs do not
prove that every record represents an actual completed lap.

The first transfer attempt stopped before computing scores because the older
four-race recent exports omitted required sectors/speeds and other fields.
`run_transfer_full_schema.py` uses the existing canonical 61-event full-input
manifest. `failed_transfer_attempt.json` and `amended_input_lock.json` preserve
the failure and bind the input-only amendment before scoring. The original
runner, spec, selected models and fit lock remain unchanged.

Canonical results and evidence live in
`artifacts/research/boundary_20260908/checkpoint/`: `results.json`,
`transfer_forecasts.csv.gz`, `verification.json`, `original_eligible_parity.json`
and `failure_analysis.json`. All 61,948 transfer predictions were recomputed;
105 input hashes checked; four real-stream truncation/poisoning tests passed for
all four candidates. All 55,757 original eligible predictions and targets match
the prior frozen frontier artifact exactly, with maximum point difference 0s.
Seven semantic regression tests pass. No production changes or promotion.

Run from the repository root with a fresh output directory if repeating a
completed experiment; runners refuse to overwrite their locks/results:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q -p no:cacheprovider research/experiments/boundary_20260908/checkpoint/test_experiment.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/boundary_20260908/checkpoint/run_experiment.py discover
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/boundary_20260908/checkpoint/run_transfer_full_schema.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv-f1/bin/python research/experiments/boundary_20260908/checkpoint/verify_experiment.py
```

Suggested commit: `research(f1-live): evaluate a frozen checkpoint residual expert`.
