# Compact rival-gap evidence publication

This helper publishes an already closed, independently verified experiment. It
does not fit models, rerun predictions, parse provider streams or inspect
feature/label/forecast row payloads. It hashes omitted local files to validate
the recorded evidence graph.

The destination must not exist. The helper requires exact design, selection and
verification SHA256 values, a successful verifier, identical reported and
recomputed decisions, complete declared forecast/prefix verification, ordered
timezone-aware stage timestamps, and no failed-execution marker. It checks all
declared source bindings before and after copying. Each included JSON receipt
is copied byte for byte; 4 MiB per-file and 16 MiB total limits reject oversized
packages before output begins. Raw streams, JSONL tables, arrays and models are
excluded. Partial output remains in place with a failure receipt if an input
changes during publication.

The package includes specification/protocol, independent review and pre-fit
tests, all three stage attempts, design/data/two yearly feature closures,
fit/forecast/selection closures and the independent verification result. An
optional local source-only CI receipt must have a successful exit, explicit
no-historical-input/no-download/no-fit and source-stability confirmations, plus a
`source_files` (or `gap_and_verifier_sources`) hash map. Only those declared CI
paths are checked; newly added publication sources are bound separately. Local
CI evidence is not a claim that a remote workflow ran.

No actual publication occurs as part of the synthetic tests. Root review and a
successful independent verifier are required before the real command is run.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m pytest -q --import-mode=importlib -p no:cacheprovider research/experiments/boundary_20260908/rival_gap_publication/test_publish.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv-f1/bin/python -m research.experiments.boundary_20260908.rival_gap_publication.publish \
  --verification artifacts/research/boundary_20260908/rival_gap_forecast/verification/result.json \
  --selection-sha256 SELECTION_SHA256 \
  --verification-sha256 VERIFICATION_SHA256 \
  --ci-receipt artifacts/research/boundary_20260908/rival_gap_publication/ci_source_only_publication.json \
  --ci-sha256 CI_RECEIPT_SHA256
```

The default execution is `artifacts/research/boundary_20260908/rival_gap_forecast`;
the default design SHA is
`cf656765fad43c8c95e831f2abb3c6b4edf621b8a92b3605ed4fdd3f26d4d49b`.
The default destination is
`docs/research/evidence/boundary_rival_gap_forecast_20260908`. Explicit
`--execution`, `--design-sha256` and `--out` overrides support an intentionally
chosen sibling execution without modifying the original publication.

Suggested commit: `research(f1-live): publish verified rival-gap prediction evidence`.
