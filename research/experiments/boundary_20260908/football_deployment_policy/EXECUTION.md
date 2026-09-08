# Fixed deployment-policy execution

Suggested commit: `research(football): execute full-history deployment assessment`.

The committed specification is the authority. This implementation compares its
two policies on the complete original population. It is an operational policy
assessment on exposed historical dates, not a fresh holdout or a new model search.

Use the repository Python environment with `PYTHONPATH=.` and
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Every stage is exclusive;
an execution failure leaves its attempt and failure record and stops this run.

1. Independently review the exact `run.sources()` map and save a review with
   `approved_for_execution_lock: true` and that `source_files` map.
2. Run `python -m research.experiments.boundary_20260908.football_deployment_policy.run freeze --review PATH`.
   This checks metadata and serialized full-history states, runs the synthetic
   contracts, and saves immutable copies of every reference source. It fixes
   reuse of all 36 saved full-history states before any new score attachment.
3. Run the same module with `prepare`. This closes metadata, population, and
   block inventories; it does not materialize historical numeric training rows.
4. Run with `forecast`. Each block admits rows by metadata before parsing its
   training goals. The unchanged canonical caller fits one prefix DC model and
   calls its original automatic calibrator. The complete restored full-history
   candidate is also replayed through the caller's output path with injected
   state, without a duplicate fit. All 2,280 paired outputs close before scoring.
5. Run with `score`. External goal labels attach only to the closed full ledger.
   Both policies' actual finite matrices are scored; there is no probability
   floor, observed-score support expansion, removed row, or tuned alternative.
6. Run `python -m research.experiments.boundary_20260908.football_deployment_policy.verify` in a fresh interpreter.
   The verifier reconstructs memberships, parameters, calibrator transforms,
   complete outputs, caller parity, metrics and paired uncertainty without fits.

The old reference source snapshot is retained even if a later successful result
authorizes canonical default integration. Any such integration needs separate
API/CLI/default tests and must retain the prefix model's historical assessment
lineage. Explicit nondefault settings are outside the observed improvement.

Large raw forecast ledgers and input data stay local and hash-bound. Publication
must include the exact compact results, verification, source/input locks and
artifact manifest; an observed failed gate is published as a failed candidate.
