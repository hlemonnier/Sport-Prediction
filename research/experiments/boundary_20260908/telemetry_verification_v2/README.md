# Independent telemetry selection verification

The [closed publication](../../../../docs/research/boundary_telemetry_selection_20260908.md)
records a passed evidence replay and a failed model advancement decision. No
model is fitted by this verifier.

The original sibling `telemetry_verification/verify.py` is preserved with its
failed receipt. This version changes exactly one metadata key:
`selection_labels_read` becomes `external_selection_labels_attached`, matching
the frozen fitting runner. A regression checks the exact source difference and
requires that the flag is false. Statistical procedures and sample selection
are unchanged.

The replay requires the closed local selection and its hash-matched raw inputs,
feature/forecast ledgers and saved models. The compact committed evidence does
not include that complete dataset. Existing evidence is never overwritten; an
independent replay must specify a new output path:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.telemetry_verification_v2.verify --output /tmp/telemetry-independent-replay.json
```

The 28 synthetic tests require no historical inputs and are included in the
research CI workflow. The original verifier source must remain available for
the exact-diff regression. The expanded workflow passed **876 tests**, with two
optional raw-pilot tests skipped, from a source-only copy without data or fitted
artifacts. The [local receipt](../../../../docs/research/evidence/boundary_telemetry_verifier_ci_20260908.json)
records that check; hosted CI is verified separately after pushing.

Suggested commit: `research(f1-live): verify frozen telemetry outcomes`.
