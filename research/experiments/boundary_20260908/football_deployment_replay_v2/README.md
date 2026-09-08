# Corrective replay of an unchanged deployment-policy assessment

Suggested commit: `fix(research): reconstruct complete deployment provenance`.

The first independent verifier stopped before numeric replay because its expected
block metadata omitted three fields present in the actual closed schema:
`goal_state_sha256`, `source_artifact`, and `source_artifact_sha256`. The raw
archive metadata, fixture metadata and block inventory matched exactly.

The original run and its terminal verification failure remain immutable. This
separate corrective replay permits only that exact recorded failure and adds the
three fields to the independent expectation using the already pinned original
feature file and complete model-state digest. It retains every other original
verification check. It does not edit forecasts, raw inputs, labels, learned
parameters, metrics, gates, population, uncertainty, or the original verifier.

There are zero new model fits, calibration calls or forecasts. A new fixed source
and synthetic-test closure precedes the one historical replay. Any additional
failure aborts this corrective attempt and remains recorded. Activation requires
the unchanged original numerical gates plus this independently reconstructed
complete-output verification and subsequent canonical caller integration tests.

The improvement remains a retrospective operational-policy comparison on exposed
dates. A corrected provenance verifier does not create a fresh holdout or new
predictive evidence.
