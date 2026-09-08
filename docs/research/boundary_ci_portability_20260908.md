# Research CI portability correction

The first publication at `cc995ca3110cf4a6b3846a6572febaa58b14f4d8` passed all eight existing production CI jobs. Its new research workflow completed with 198 tests passed, two optional raw-archive tests skipped and three failures in a batch-size invariance assertion. [Original CI](https://github.com/hlemonnier/Sport-Prediction/actions/runs/34181664195), [preserved failing research run](https://github.com/hlemonnier/Sport-Prediction/actions/runs/34181664260).

The failing assertion required bit-identical probabilities for a single-row prediction and that same row in a two-row prediction batch. Linux reported differences no larger than 1.1102230246251565e-16. Different BLAS accumulation paths need not produce identical final bits across matrix shapes. The probability normalization, optimization, same-shape serialized replay, unreadable-target checks and fitted scaling invariance assertions did not fail.

The portable test suite changes only this comparison to `rtol=0, atol=8 * np.finfo(np.float64).eps`, approximately 1.78e-15 in probability units. All four configurations retain the batch-independence test. The original 100-test source is preserved byte-for-byte because its checksum belongs to the pre-fit experiment lock. A new source-integrity test verifies the archived checksum and the exact single-assertion difference in the portable copy. [Portable suite and rationale](../../research/experiments/boundary_20260908/football_xg/ci_contracts/README.md).

The research workflow now selects that copy, the original five runner tests and the sector tests. It does not select the archived platform-dependent assertion. This adds one integrity check to the prior 203-test local suite. No production source, model coefficient, recorded prediction, performance metric, selection criterion or frozen research artifact was edited.

The [earlier checkpoint manifest](evidence/boundary_integration_checkpoint_20260908.json) remains an exact record of the initial publication. Its workflow checksum is intentionally superseded by this correction; its other 63 file bindings and all 351 earlier publication bindings remain unchanged. The [correction manifest](evidence/boundary_ci_portability_20260908.json) records both workflow hashes and the added files.

Suggested commit: `fix(ci): allow floating-point roundoff across prediction batch sizes`.
